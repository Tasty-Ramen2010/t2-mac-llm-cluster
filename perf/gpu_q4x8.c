// gpu_q4x8: OpenCL GEMV on llama.cpp's repacked Q4_0 layout (block_q4_0x8: 8 fp16 scales + 128 interleaved bytes),
// straight from host memory (zero copy) -- can the UHD 630 add useful bandwidth next to the CPU?
// Streams many matrices (like one token of Gemma 12B) so nothing sits in cache. Checks results against a CPU reference.
// build: gcc -O2 gpu_q4x8.c -o gpu_q4x8 -lOpenCL -lm      run: ./gpu_q4x8 ROWS K [WG_SUBGROUPS]
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static const char *src =
"#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n"
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"// one 16-lane subgroup per group of 8 rows; lanes stride over 32-wide blocks; x staged in local memory\n"
"__attribute__((intel_reqd_sub_group_size(16)))\n"
"__kernel void gemv_q4x8(__global const uchar *W, __global const float *x, __global float *y, int nb, __local float *xs) {\n"
"  const int K = nb * 32;\n"
"  for (int i = get_local_id(0); i < K; i += get_local_size(0)) xs[i] = x[i];\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  const int lane = get_sub_group_local_id();\n"
"  const int g = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();\n"
"  float acc[8] = {0,0,0,0,0,0,0,0};\n"
"  for (int b = lane; b < nb; b += 16) {\n"
"    __global const uchar *blk = W + ((size_t)g * nb + b) * 144;\n"
"    const float8 d = vload_half8(0, (__global const half *)blk);\n"
"    XB_DECL\n"
"    const float8 x0 = vload8(0, xb), x1 = vload8(1, xb), x2 = vload8(2, xb), x3 = vload8(3, xb);\n"
"    const uchar16 qa = vload16(0, blk + 16), qb = vload16(1, blk + 16), qc = vload16(2, blk + 16), qd = vload16(3, blk + 16);\n"
"    const uchar16 qe = vload16(4, blk + 16), qf = vload16(5, blk + 16), qg = vload16(6, blk + 16), qh = vload16(7, blk + 16);\n"
"    // row r: bytes r*8..r*8+7 (elements 0-7 low, 16-23 high) and 64+r*8.. (elements 8-15 low, 24-31 high)\n"
"#define ROW(r, A, B) { \\\n"
"      const uchar8 u = (r & 1) ? A.hi : A.lo, v = (r & 1) ? B.hi : B.lo; \\\n"
"      const float8 l0 = convert_float8(as_char8(u << (uchar8)4) >> (char8)4), h0 = convert_float8(as_char8(u) >> (char8)4); \\\n"
"      const float8 l1 = convert_float8(as_char8(v << (uchar8)4) >> (char8)4), h1 = convert_float8(as_char8(v) >> (char8)4); \\\n"
"      const float8 p = l0 * x0 + l1 * x1 + h0 * x2 + h1 * x3; \\\n"
"      acc[r] += d[r] * (p.s0 + p.s1 + p.s2 + p.s3 + p.s4 + p.s5 + p.s6 + p.s7); }\n"
"    ROW(0, qa, qe) ROW(1, qa, qe) ROW(2, qb, qf) ROW(3, qb, qf) ROW(4, qc, qg) ROW(5, qc, qg) ROW(6, qd, qh) ROW(7, qd, qh)\n"
"  }\n"
"  for (int r = 0; r < 8; r++) { const float t = sub_group_reduce_add(acc[r]); if (lane == r) y[g * 8 + r] = t; }\n"
"}\n";

static float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float f = e == 0 ? ldexpf((float)m, -24) : ldexpf((float)(m | 1024), (int)e - 25);
    return s ? -f : f;
}

int main(int argc, char **argv) {
    const int rows = argc > 1 ? atoi(argv[1]) : 7680, K = argc > 2 ? atoi(argv[2]) : 3840, sgs = argc > 3 ? atoi(argv[3]) : 8;
    const int nb = K / 32, groups = rows / 8;
    const size_t mat = (size_t)groups * nb * 144;
    const int nmat = (int)((size_t)(640u << 20) / mat);   // ~640 MB of weights, far bigger than any cache
    cl_platform_id p; cl_device_id d; cl_int e;
    clGetPlatformIDs(1, &p, NULL); clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, NULL);
    cl_context c = clCreateContext(NULL, 1, &d, NULL, NULL, &e);
    cl_command_queue q = clCreateCommandQueueWithProperties(c, d, NULL, &e);
    cl_program pr = clCreateProgramWithSource(c, 1, &src, NULL, &e);
    if (clBuildProgram(pr, 1, &d, getenv("GLOBALX") ? "-cl-std=CL2.0 -cl-mad-enable -DXB_DECL=\"__global const float *xb = x + b * 32;\"" : "-cl-std=CL2.0 -cl-mad-enable -DXB_DECL=\"__local const float *xb = xs + b * 32;\"", NULL, NULL) != CL_SUCCESS) {
        static char log[65536]; clGetProgramBuildInfo(pr, d, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL); printf("build failed:\n%s\n", log); return 1; }
    cl_kernel k = clCreateKernel(pr, "gemv_q4x8", &e);

    const size_t wtot = (mat * nmat + 4095) / 4096 * 4096;
    uint8_t *W = aligned_alloc(4096, wtot);
    float *x = aligned_alloc(4096, 65536), *y = aligned_alloc(4096, ((size_t)rows * 4 + 4095) / 4096 * 4096);
    srand(1);
    for (size_t i = 0; i < wtot; i++) W[i] = (uint8_t)rand();
    for (size_t gb = 0; gb < (size_t)groups * nb * nmat; gb++) for (int r = 0; r < 8; r++) {   // sane fp16 scales ~0.01
        uint16_t h = 0x2000 + (rand() & 0x3ff); memcpy(W + gb * 144 + r * 2, &h, 2); }
    if (getenv("ONES")) { memset(W, 0x11, wtot); for (size_t gb = 0; gb < (size_t)groups * nb * nmat; gb++) for (int r = 0; r < 8; r++) { uint16_t h = 0x3C00; memcpy(W + gb * 144 + r * 2, &h, 2); } }
    for (int i = 0; i < K; i++) x[i] = (rand() % 2000 - 1000) / 1000.0f;

    cl_mem bW = clCreateBuffer(c, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, wtot, W, &e);
    cl_mem bx = clCreateBuffer(c, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 65536, x, &e);
    cl_mem by = clCreateBuffer(c, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, ((size_t)rows * 4 + 4095) / 4096 * 4096, y, &e);
    const size_t ls = (size_t)16 * sgs, gs = (size_t)groups * 16;
    if (groups % sgs) { printf("row groups %d not divisible by %d subgroups\n", groups, sgs); return 1; }

    // correctness on matrix 0
    cl_buffer_region reg = {0, mat};
    cl_mem sub = clCreateSubBuffer(bW, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg, &e);
    clSetKernelArg(k, 0, sizeof(sub), &sub); clSetKernelArg(k, 1, sizeof(bx), &bx); clSetKernelArg(k, 2, sizeof(by), &by);
    clSetKernelArg(k, 3, sizeof(int), &nb); clSetKernelArg(k, 4, (size_t)K * 4, NULL);
    e = clEnqueueNDRangeKernel(q, k, 1, NULL, &gs, &ls, 0, NULL, NULL); clFinish(q);
    if (e) { printf("enqueue error %d\n", e); return 1; }
    double maxerr = 0, maxv = 0;
    for (int row = 0; row < rows; row += (getenv("DBG") ? 1 : 97)) {
        const int g = row / 8, r = row % 8; double s = 0;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = W + ((size_t)g * nb + b) * 144; uint16_t h; memcpy(&h, blk + r * 2, 2);
            const float dd = h2f(h); double t = 0;
            for (int j = 0; j < 16; j++) {
                const uint8_t byte = blk[16 + (j < 8 ? r * 8 + j : 64 + r * 8 + j - 8)];
                const int lo = (int8_t)(byte << 4) >> 4, hi = (int8_t)byte >> 4;
                t += lo * x[b * 32 + j] + hi * x[b * 32 + j + 16];
            }
            s += dd * t;
        }
        maxerr = fmax(maxerr, fabs(s - y[row])); maxv = fmax(maxv, fabs(s));
        if (getenv("DBG") && row < 800) printf("row %d ref %.4f gpu %.4f\n", row, s, y[row]);
    }

    // bandwidth: stream all matrices, one launch each, wait after each (what a synchronous offload would do)
    cl_mem *subs = malloc(sizeof(cl_mem) * nmat);
    for (int m = 0; m < nmat; m++) { cl_buffer_region rg = {(size_t)m * mat, mat}; subs[m] = clCreateSubBuffer(bW, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &rg, &e);
        if (e) { printf("sub-buffer %d error %d (offset alignment?)\n", m, e); return 1; } }
    double best_sync = 1e9, best_async = 1e9;
    for (int rep = 0; rep < 4; rep++) {
        double t0 = now();
        for (int m = 0; m < nmat; m++) { clSetKernelArg(k, 0, sizeof(cl_mem), &subs[m]); clEnqueueNDRangeKernel(q, k, 1, NULL, &gs, &ls, 0, NULL, NULL); clFinish(q); }
        best_sync = fmin(best_sync, now() - t0);
        t0 = now();
        for (int m = 0; m < nmat; m++) { clSetKernelArg(k, 0, sizeof(cl_mem), &subs[m]); clEnqueueNDRangeKernel(q, k, 1, NULL, &gs, &ls, 0, NULL, NULL); }
        clFinish(q);
        best_async = fmin(best_async, now() - t0);
    }
    if (getenv("LOOP")) {   // keep streaming for LOOP seconds, report sustained GB/s
        const double T = atof(getenv("LOOP")); double t0 = now(); long n = 0;
        while (now() - t0 < T) { for (int m = 0; m < nmat; m++) { clSetKernelArg(k, 0, sizeof(cl_mem), &subs[m]); clEnqueueNDRangeKernel(q, k, 1, NULL, &gs, &ls, 0, NULL, NULL); } clFinish(q); n += nmat; }
        printf("GPU sustained over %.1fs: %.1f GB/s\n", now() - t0, n * mat / (now() - t0) / 1e9);
        return 0;
    }
    printf("GPU q4_0x8 GEMV %d x %d, %d subgroups/WG: %d matrices of %.1f MB | max err %.2g (|y| up to %.3g)\n", rows, K, sgs, nmat, mat / 1e6, maxerr, maxv);
    printf("  launch+wait each: %.1f GB/s, %.0f us per matrix | queued back-to-back: %.1f GB/s, %.0f us per matrix\n",
           nmat * mat / best_sync / 1e9, best_sync / nmat * 1e6, nmat * mat / best_async / 1e9, best_async / nmat * 1e6);
    return 0;
}
