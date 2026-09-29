// attnbench: GPU multi-query attention (decode) on DeepSeek MLA shapes vs exact math; KV cache stays in host RAM
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CK(x) do { cl_int e_ = (x); if (e_) { fprintf(stderr, "%s: %d\n", #x, e_); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static unsigned short f2h(float f) { _Float16 h = (_Float16) f; unsigned short u; memcpy(&u, &h, 2); return u; }
static float h2f(unsigned short u) { _Float16 h; memcpy(&h, &u, 2); return (float) h; }
int main(int argc, char ** argv) {
    const int H = 8, DK = 576, DV = 512, CH = 256;
    int n_kv = argc > 1 ? atoi(argv[1]) : 8192;
    size_t kv_bytes = ((size_t) n_kv * DK * 2 + 4095) & ~4095UL;
    unsigned short * K = aligned_alloc(4096, kv_bytes);
    srand(1);
    for (size_t i = 0; i < (size_t) n_kv * DK; i++) K[i] = f2h(((rand() % 2001) - 1000) / 1000.0f);
    float * Q = aligned_alloc(4096, 4096 * 5);
    for (int i = 0; i < H * DK; i++) Q[i] = ((rand() % 2001) - 1000) / 400.0f;
    const float scale = 1.0f / sqrtf(192.0f);
    int ch = getenv("CHK") ? atoi(getenv("CHK")) : 64; if (!getenv("CHK")) while (ch < CH && (n_kv + ch - 1) / ch > 72) ch *= 2;
    int n_chunks = (n_kv + ch - 1) / ch;
    size_t part_bytes = ((size_t) n_chunks * H * (2 + DV) * 4 + 4095) & ~4095UL;
    float * part = aligned_alloc(4096, part_bytes);

    cl_platform_id pl; cl_device_id dev; CK(clGetPlatformIDs(1, &pl, NULL)); CK(clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    cl_int e; cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e); CK(e);
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 }; cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &e); CK(e);
    const char * kfile = getenv("KFILE") ? getenv("KFILE") : "attn.cl"; FILE * f = fopen(kfile, "rb"); static char src[65536]; size_t n = fread(src, 1, sizeof src - 1, f); src[n] = 0; fclose(f);
    const char * s = src; cl_program prog = clCreateProgramWithSource(ctx, 1, &s, NULL, &e); CK(e);
    if (clBuildProgram(prog, 1, &dev, getenv("KOPTS") ? getenv("KOPTS") : "-cl-std=CL1.2 -cl-mad-enable -cl-fast-relaxed-math", NULL, NULL)) { static char log[65536]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL); puts(log); return 1; }
    cl_kernel k = clCreateKernel(prog, "attn_chunk", &e); CK(e);
    cl_mem mK;
    if (getenv("KDEV")) {    // driver-owned GPU memory (copied once) instead of zero-copy host memory
        mK = clCreateBuffer(ctx, CL_MEM_READ_ONLY, kv_bytes, NULL, &e); CK(e);
        CK(clEnqueueWriteBuffer(q, mK, CL_TRUE, 0, kv_bytes, K, 0, NULL, NULL));
    } else { mK = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, kv_bytes, K, &e); CK(e); }
    cl_mem mQ = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 4096 * 5, Q, &e); CK(e);
    cl_mem mP = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, part_bytes, part, &e); CK(e);
    cl_ulong ks = DK, vs = DK, qs = DK; cl_mem nomask = NULL;
    CK(clSetKernelArg(k, 0, sizeof mK, &mK)); CK(clSetKernelArg(k, 1, 8, &ks)); CK(clSetKernelArg(k, 2, sizeof mK, &mK)); CK(clSetKernelArg(k, 3, 8, &vs));
    CK(clSetKernelArg(k, 4, sizeof mQ, &mQ)); CK(clSetKernelArg(k, 5, 8, &qs)); CK(clSetKernelArg(k, 6, sizeof(cl_mem), &nomask));
    CK(clSetKernelArg(k, 7, 4, &scale)); CK(clSetKernelArg(k, 8, 4, &n_kv)); CK(clSetKernelArg(k, 9, sizeof mP, &mP)); CK(clSetKernelArg(k, 10, 4, &ch));
    size_t lws = getenv("WGS") ? atoi(getenv("WGS")) : 64, gws = (size_t) n_chunks * lws;
    for (int w = 0; w < 5; w++) CK(clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, &lws, 0, NULL, NULL)); clFinish(q);
    int reps = 54; double t0 = now(), gpu_ns = 0;
    for (int r = 0; r < reps; r++) {
        cl_event ev; CK(clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, &lws, 0, NULL, &ev)); clFinish(q);
        cl_ulong a, b; clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, 8, &a, NULL); clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, 8, &b, NULL);
        gpu_ns += (double) (b - a); clReleaseEvent(ev);
    }
    double per = (now() - t0) / reps;
    printf("[kernel executes %.3f ms of %.3f ms] ", gpu_ns / reps / 1e6, per * 1e3);
    // merge partials (host) and compare with exact math
    void * mp = clEnqueueMapBuffer(q, mP, CL_TRUE, CL_MAP_READ, 0, part_bytes, 0, NULL, NULL, &e); CK(e);
    double t1 = now(); double worst = 0, mag = 0;
    static float out[8][512];
    for (int h = 0; h < H; h++) {
        float M = -INFINITY, S = 0; static float acc[512]; memset(acc, 0, sizeof acc);
        for (int c = 0; c < n_chunks; c++) {
            const float * pc = (const float *) mp + ((size_t) c * H + h) * (2 + DV);
            if (pc[1] == 0) continue;
            float Mn = fmaxf(M, pc[0]), a = M == -INFINITY ? 0 : expf(M - Mn), b = expf(pc[0] - Mn);
            for (int d = 0; d < DV; d++) acc[d] = acc[d] * a + pc[2 + d] * b;
            S = S * a + pc[1] * b; M = Mn;
        }
        for (int d = 0; d < DV; d++) out[h][d] = acc[d] / S;
    }
    double merge = now() - t1;
    for (int h = 0; h < H; h++) {
        static double w[65536]; double m = -INFINITY;
        for (int r = 0; r < n_kv; r++) { double dot = 0; for (int d = 0; d < DK; d++) dot += (double) Q[h * DK + d] * h2f(K[(size_t) r * DK + d]); w[r] = dot * scale; if (w[r] > m) m = w[r]; }
        double sum = 0; for (int r = 0; r < n_kv; r++) { w[r] = exp(w[r] - m); sum += w[r]; }
        for (int d = 0; d < DV; d += 7) { double ex = 0; for (int r = 0; r < n_kv; r++) ex += w[r] * h2f(K[(size_t) r * DK + d]); ex /= sum;
            worst = fmax(worst, fabs(ex - out[h][d])); mag = fmax(mag, fabs(ex)); }
    }
    printf("chunk %3d ", ch); printf("n_kv %5d: GPU %.3f ms per layer (x27 = %.1f ms per word) + host merge %.3f ms | max error %.2g (largest value %.2g)\n",
           n_kv, per * 1e3, per * 27e3, merge * 1e3, worst, mag);
    return 0;
}
