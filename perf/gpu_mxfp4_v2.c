#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *src =
"#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n"
"#ifndef ROWS\n#define ROWS 4\n#endif\n"
"inline float16 fp4dec(uchar16 q) {\n"
"  int16 v = convert_int16(q);\n"
"  int16 s = (v >> 3) & 1, ee = (v >> 1) & 3, m = v & 1;\n"
"  int16 bits = select(select((int16)0, (int16)0x3F000000, m != 0), ((ee + 126) << 23) | (m << 22), ee != 0);\n"
"  return as_float16(bits | (s << 31));\n"
"}\n"
"__attribute__((intel_reqd_sub_group_size(16)))\n"
"__kernel void gemv(__global const uchar *W, __global const float *x, __global float *y, int nblk) {\n"
"  __local float xl[4096];\n"
"  const int lane = get_sub_group_local_id();\n"
"  for (int i = lane; i < nblk * 32; i += 16) xl[i] = x[i];\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  const int row0 = get_group_id(0) * ROWS;\n"
"  float acc[ROWS];\n"
"  for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;\n"
"  for (int b = lane; b < nblk; b += 16) {\n"
"    const float16 xa = vload16(0, xl + b * 32), xb = vload16(0, xl + b * 32 + 16);\n"
"    for (int r = 0; r < ROWS; r++) {\n"
"      __global const uchar *blk = W + ((size_t)(row0 + r) * nblk + b) * 17;\n"
"      const uchar16 qs = vload16(0, blk + 1);\n"
"      const float d = as_float(((uint)blk[0]) << 23);\n"
"      const float16 lo = fp4dec(qs & (uchar)15), hi = fp4dec(qs >> (uchar)4);\n"
"      const float16 p = lo * xa + hi * xb;\n"
"      acc[r] += d * (p.s0+p.s1+p.s2+p.s3+p.s4+p.s5+p.s6+p.s7+p.s8+p.s9+p.sa+p.sb+p.sc+p.sd+p.se+p.sf);\n"
"    }\n"
"  }\n"
"  for (int r = 0; r < ROWS; r++) { const float t = sub_group_reduce_add(acc[r]); if (lane == 0) y[row0 + r] = t; }\n"
"}\n";
int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 3840, K = 2880, nblk = K / 32;
    cl_platform_id p; cl_device_id d; cl_int e;
    clGetPlatformIDs(1, &p, NULL); clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, NULL);
    cl_context c = clCreateContext(NULL, 1, &d, NULL, NULL, &e);
    cl_command_queue q = clCreateCommandQueueWithProperties(c, d, NULL, &e);
    cl_program pr = clCreateProgramWithSource(c, 1, &src, NULL, &e);
    char opts[128]; snprintf(opts, sizeof(opts), "-cl-std=CL2.0 -cl-mad-enable -DROWS=%d", getenv("ROWS") ? atoi(getenv("ROWS")) : 4);
    if (clBuildProgram(pr, 1, &d, opts, NULL, NULL) != CL_SUCCESS) {
        char log[8192]; clGetProgramBuildInfo(pr, d, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL); printf("build failed:\n%s\n", log); return 1; }
    cl_kernel k = clCreateKernel(pr, "gemv", &e);
    size_t wbytes = (size_t)rows * nblk * 17;
    unsigned char *W = aligned_alloc(4096, (wbytes + 4095) / 4096 * 4096); float *x = aligned_alloc(4096, 16384), *y = aligned_alloc(4096, (rows * 4 + 4095) / 4096 * 4096);
    srand(1); for (size_t i = 0; i < wbytes; i++) W[i] = rand();
    for (size_t b = 0; b < (size_t)rows * nblk; b++) W[b * 17] = 120 + rand() % 8;   /* sane exponents */
    for (int i = 0; i < K; i++) x[i] = (rand() % 2000 - 1000) / 1000.0f;
    cl_mem bW = clCreateBuffer(c, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, wbytes, W, &e);
    cl_mem bx = clCreateBuffer(c, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, K * 4, x, &e);
    cl_mem by = clCreateBuffer(c, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, rows * 4, y, &e);
    clSetKernelArg(k, 0, sizeof(bW), &bW); clSetKernelArg(k, 1, sizeof(bx), &bx); clSetKernelArg(k, 2, sizeof(by), &by); clSetKernelArg(k, 3, sizeof(int), &nblk);
    int R = getenv("ROWS") ? atoi(getenv("ROWS")) : 4;
    size_t gs = (size_t)rows / R * 16, ls = 16;
    for (int w = 0; w < 20; w++) { clEnqueueNDRangeKernel(q, k, 1, NULL, &gs, &ls, 0, NULL, NULL); clFinish(q); }
    int it = 200; double t0 = now();
    for (int i = 0; i < it; i++) {
        clEnqueueNDRangeKernel(q, k, 1, NULL, &gs, &ls, 0, NULL, NULL);
        void *m = clEnqueueMapBuffer(q, by, CL_TRUE, CL_MAP_READ, 0, rows * 4, 0, NULL, NULL, &e); clEnqueueUnmapMemObject(q, by, m, 0, NULL, NULL);
    }
    clFinish(q);
    double per = (now() - t0) / it;
    /* verify a few rows on the CPU */
    static const int KV[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}; double maxerr = 0;
    for (int r = 0; r < 8; r++) { double s = 0; for (int b = 0; b < nblk; b++) { unsigned char *blk = W + ((size_t)r * nblk + b) * 17; float dd = ldexpf(1.0f, blk[0] - 128);
        for (int j = 0; j < 16; j++) s += dd * (KV[blk[1+j] & 15] * x[b*32+j] + KV[blk[1+j] >> 4] * x[b*32+j+16]); }
        double err = fabs(s - y[r]) / (fabs(s) + 1e-6); if (err > maxerr) maxerr = err; }
    printf("GPU MXFP4 GEMV %d x %d: %.3f ms per call incl. launch+readback, %.1f GB/s effective, max rel err %.2e\n", rows, K, per * 1e3, wbytes / per / 1e9, maxerr);
    return 0;
}
