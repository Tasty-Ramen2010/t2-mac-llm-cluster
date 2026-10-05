// bwtest.cu: how much of the Orin's memory bandwidth can a Q8_0 matrix-vector product reach?
//   1. rd16      : pure streaming read ceiling (16-byte loads)
//   2. mv_base   : llama.cpp-style mmvq for Q8_0 (one CTA per row, 4 warps, small unaligned loads straight from global memory)
//   3. mv_tile   : new kernel: CTA streams R consecutive rows into shared memory with 16-byte cp.async (double buffered),
//                  then runs the SAME dp4a dot product from shared memory.
// Build: nvcc -O3 -arch=sm_87 -o bwtest bwtest.cu     Run: ./bwtest
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

struct block_q8_0 { half d; int8_t qs[32]; };           // 34 bytes
struct block_q8_1 { half2 ds; int8_t qs[32]; };         // 36 bytes (d, sum)

__device__ __forceinline__ int get_int_b2(const void * x, int i32) {          // 2-byte aligned 32-bit read (what ggml does for q8_0)
    const uint16_t * x16 = (const uint16_t *) x;
    return x16[2 * i32] | (x16[2 * i32 + 1] << 16);
}
__device__ __forceinline__ int get_int_b4(const void * x, int i32) { return ((const int *) x)[i32]; }

// dot of one q8_0 weight block (8 ints) with one q8_1 activation block, restricted to ints [iqs, iqs+vdr)
__device__ __forceinline__ float dot_q8_0_q8_1(const block_q8_0 * w, const block_q8_1 * a, int iqs, int vdr) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {                      // vdr == 2
        const int v = get_int_b2(w->qs, iqs + i);
        const int u = get_int_b4(a->qs, iqs + i);
        sumi = __dp4a(v, u, sumi);
    }
    return __half2float(w->d) * __low2float(a->ds) * sumi;
}

// ---------------- 2. baseline: llama.cpp-like ----------------
template <int NW>
__global__ void mv_base(const block_q8_0 * __restrict__ W, const block_q8_1 * __restrict__ A, float * __restrict__ out, int nblk) {
    const int row = blockIdx.x, tid = threadIdx.y * 32 + threadIdx.x;
    const block_q8_0 * wr = W + (size_t) row * nblk;
    float acc = 0.f;
    constexpr int qi = 8, vdr = 2, per_iter = vdr * NW * 32 / qi;       // 32 blocks per iteration for NW = 4
    for (int kb = tid / (qi / vdr); kb < nblk; kb += per_iter)
        acc += dot_q8_0_q8_1(wr + kb, A + kb, vdr * (tid % (qi / vdr)), vdr);
    __shared__ float red[NW][32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (threadIdx.x == 0) red[threadIdx.y][0] = acc;
    __syncthreads();
    if (tid == 0) { float s = 0; for (int w = 0; w < NW; ++w) s += red[w][0]; out[row] = s; }
}

// ---------------- 3. tiled: cp.async 16-byte streaming into shared memory ----------------
__device__ __forceinline__ void cp_async16(void * smem, const void * gmem) {
    unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// one CTA = R rows x all K; grid-stride over row tiles; two shared-memory buffers (prefetch the next tile while computing this one)
template <int R, int NT>
__global__ void __launch_bounds__(NT) mv_tile(const block_q8_0 * __restrict__ W, const block_q8_1 * __restrict__ A, float * __restrict__ out, int nblk, int nrows) {
    extern __shared__ __align__(16) uint8_t smem[];
    const size_t rowbytes = (size_t) nblk * sizeof(block_q8_0);          // multiple of 16 when nblk % 8 == 0
    const size_t tilebytes = R * rowbytes;
    uint8_t * buf[2] = { smem, smem + tilebytes };
    block_q8_1 * sa = (block_q8_1 *) (smem + 2 * tilebytes);             // activation vector staged once
    const int tid = threadIdx.x;
    for (int i = tid; i < nblk * (int) sizeof(block_q8_1) / 16; i += NT) ((uint4 *) sa)[i] = ((const uint4 *) A)[i];

    auto issue = [&](int tile, int b) {
        const uint8_t * src = (const uint8_t *) W + (size_t) tile * tilebytes;
        const int n16 = (int) (tilebytes / 16);
        const int rows_left = nrows - tile * R;
        const int lim = rows_left >= R ? n16 : (int) ((size_t) rows_left * rowbytes / 16);
        for (int i = tid; i < lim; i += NT) cp_async16(buf[b] + (size_t) i * 16, src + (size_t) i * 16);
        cp_commit();
    };
    const int ntiles = (nrows + R - 1) / R;
    int t = blockIdx.x;
    int b = 0;
    if (t < ntiles) issue(t, 0);
    __syncthreads();
    constexpr int NWARP = NT / 32;
    for (; t < ntiles; t += gridDim.x) {
        const int tn = t + gridDim.x;
        if (tn < ntiles) issue(tn, b ^ 1);
        if (tn < ntiles) cp_wait<1>(); else cp_wait<0>();
        __syncthreads();
        // each warp takes rows round-robin; lanes stride over the blocks of the row (same pattern as ggml: 4 lanes per block, 2 ints each)
        for (int r = tid / 32; r < R; r += NWARP) {
            const int row = t * R + r;
            const block_q8_0 * wr = (const block_q8_0 *) (buf[b] + (size_t) r * rowbytes);
            float acc = 0.f;
            const int lane = tid % 32;
            for (int kb = lane / 4; kb < nblk; kb += 8)
                acc += dot_q8_0_q8_1(wr + kb, sa + kb, 2 * (lane % 4), 2);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
            if (lane == 0 && row < nrows) out[row] = acc;
        }
        __syncthreads();
        b ^= 1;
    }
}

// ---------------- 1. pure read ceiling ----------------
__global__ void rd16(const uint4 * __restrict__ p, size_t n, unsigned long long * sink) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x, st = (size_t) gridDim.x * blockDim.x;
    unsigned x = 0;
#pragma unroll 4
    for (; i < n; i += st) { uint4 v = __ldg(p + i); x ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (x == 0x9e3779b9u) sink[0] = x;
}

static float timeit(void (*f)(void *), void * ctx, int iters) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) f(ctx);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) f(ctx);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms / iters;
}

int main(int argc, char ** argv) {
    int dev; cudaGetDevice(&dev); cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("%s: %d SMs, %.0f MHz, L2 %d KB, smem/SM %zu KB (opt-in max %zu KB)\n", p.name, p.multiProcessorCount, p.clockRate / 1e3, p.l2CacheSize / 1024, p.sharedMemPerMultiprocessor / 1024, p.sharedMemPerBlockOptin / 1024);

    // 1. ceiling
    const size_t big = 512ull << 20;
    uint4 * d; CK(cudaMalloc(&d, big)); CK(cudaMemset(d, 1, big));
    unsigned long long * sink; CK(cudaMalloc(&sink, 8));
    printf("\n[1] pure read bandwidth (512 MB, 16-byte loads):\n");
    double best = 0;
    for (int bps : {1, 2, 4, 8}) for (int thr : {256, 512, 1024}) {
        int grid = p.multiProcessorCount * bps;
        struct C { uint4 * d; size_t n; unsigned long long * s; int g, t; } c{d, big / 16, sink, grid, thr};
        auto f = [](void * v) { C * c = (C *) v; rd16<<<c->g, c->t>>>(c->d, c->n, c->s); };
        float ms = timeit(f, &c, 20);
        double gbs = big / 1e9 / (ms / 1e3); if (gbs > best) best = gbs;
        printf("   grid=%3d x %4d threads: %6.1f GB/s\n", grid, thr, gbs);
    }
    printf("   => ceiling ~ %.0f GB/s\n", best);

    // 2/3. q8_0 matvec like the model's projections: 8192 rows x 2048 (17.8 MB), 4096 x 2048, and a MoE-ish 2048 x 512
    struct Shape { int rows, K; } shapes[] = {{8192, 2048}, {4096, 2048}, {2048, 4096}, {2048, 512}};
    for (auto s : shapes) {
        const int nblk = s.K / 32;
        std::vector<block_q8_0> hw((size_t) s.rows * nblk);
        std::vector<block_q8_1> ha(nblk);
        srand(1);
        for (auto & b : hw) { b.d = __float2half((rand() % 100) / 5000.f + 0.001f); for (int i = 0; i < 32; ++i) b.qs[i] = (int8_t) (rand() % 255 - 127); }
        for (auto & b : ha) { b.ds = __floats2half2_rn((rand() % 100) / 5000.f + 0.001f, 0.f); for (int i = 0; i < 32; ++i) b.qs[i] = (int8_t) (rand() % 255 - 127); }
        block_q8_0 * dw; block_q8_1 * da; float * o0, * o1;
        // many independent copies so the working set exceeds L2 like in the real model (layers differ)
        const int copies = 16;
        CK(cudaMalloc(&dw, hw.size() * sizeof(block_q8_0) * copies));
        for (int c = 0; c < copies; ++c) CK(cudaMemcpy((char *) dw + c * hw.size() * sizeof(block_q8_0), hw.data(), hw.size() * sizeof(block_q8_0), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&da, ha.size() * sizeof(block_q8_1))); CK(cudaMemcpy(da, ha.data(), ha.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&o0, s.rows * 4)); CK(cudaMalloc(&o1, s.rows * 4));
        const double bytes = (double) hw.size() * sizeof(block_q8_0);
        struct Ctx { block_q8_0 * w; block_q8_1 * a; float * o; int nblk, rows, it; size_t stride; int mode; int grid; };
        Ctx cx{dw, da, o0, nblk, s.rows, 0, hw.size(), 0, 0};
        auto run = [](void * v) {
            Ctx * c = (Ctx *) v; const block_q8_0 * w = c->w + (c->it++ % 16) * c->stride;
            if (c->mode == 0) mv_base<4><<<c->rows, dim3(32, 4)>>>(w, c->a, c->o, c->nblk);
            else { constexpr int R = 8, NT = 256; size_t sm = 2 * R * (size_t) c->nblk * 34 + (size_t) c->nblk * 36;
                   mv_tile<R, NT><<<c->grid, NT, sm>>>(w, c->a, c->o, c->nblk, c->rows); }
        };
        cx.mode = 0; float t0 = timeit(run, &cx, 200);
        std::vector<float> r0(s.rows), r1(s.rows);
        cx.it = 0; run(&cx); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(r0.data(), o0, s.rows * 4, cudaMemcpyDeviceToHost));
        printf("\n[2/3] q8_0 matvec %d x %d (%.1f MB):\n", s.rows, s.K, bytes / 1e6);
        printf("   baseline (ggml-style)      : %7.1f us  -> %6.1f GB/s\n", t0 * 1e3, bytes / 1e9 / (t0 / 1e3));
        constexpr int R = 8, NT = 256;
        size_t sm = 2 * R * (size_t) nblk * 34 + (size_t) nblk * 36;
        CK(cudaFuncSetAttribute(mv_tile<R, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm));
        for (int per : {1, 2, 3, 4}) {
            cx.o = o1; cx.mode = 1; cx.grid = p.multiProcessorCount * per;
            float t1 = timeit(run, &cx, 200);
            cx.it = 0; run(&cx); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(r1.data(), o1, s.rows * 4, cudaMemcpyDeviceToHost));
            double maxerr = 0; for (int i = 0; i < s.rows; ++i) maxerr = fmax(maxerr, fabs(r0[i] - r1[i]) / (fabs(r0[i]) + 1e-3));
            printf("   tiled cp.async (CTAs/SM=%d) : %7.1f us  -> %6.1f GB/s   (max rel diff vs baseline %.2g, smem %zu KB)\n", per, t1 * 1e3, bytes / 1e9 / (t1 / 1e3), maxerr, sm / 1024);
        }
        cudaFree(dw); cudaFree(da); cudaFree(o0); cudaFree(o1);
    }
    return 0;
}
