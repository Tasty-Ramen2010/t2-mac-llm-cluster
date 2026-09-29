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
"__constant float KV[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};\n"
"__attribute__((intel_reqd_sub_group_size(16)))\n"
"__kernel void gemv(__global const uchar *W, __global const float *x, __global float *y, int nblk) {\n"
"  const int row = get_group_id(0);\n"
"  const int lane = get_sub_group_local_id();\n"
"  __global const uchar *r = W + (size_t)row * nblk * 17;\n"
"  float acc = 0.0f;\n"
"  for (int b = lane; b < nblk; b += 16) {\n"
"    __global const uchar *blk = r + b * 17;\n"
"    const uint e = blk[0];\n"
"    const float d = as_float((e < 2 ? (0x00200000u << e) : ((e - 1u) << 23)));\n"
"    __global const float *xb = x + b * 32;\n"
"    float s = 0.0f;\n"
"    for (int j = 0; j < 16; j++) { const uchar q = blk[1 + j]; s += KV[q & 15] * xb[j] + KV[q >> 4] * xb[j + 16]; }\n"
"    acc += d * s;\n"
"  }\n"
"  acc = sub_group_reduce_add(acc);\n"
"  if (lane == 0) y[row] = acc;\n"
"}\n";
int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 3840, K = 2880, nblk = K / 32;
    cl_platform_id p; cl_device_id d; cl_int e;
    clGetPlatformIDs(1, &p, NULL); clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, NULL);
    cl_context c = clCreateContext(NULL, 1, &d, NULL, NULL, &e);
    cl_command_queue q = clCreateCommandQueueWithProperties(c, d, NULL, &e);
    cl_program pr = clCreateProgramWithSource(c, 1, &src, NULL, &e);
    if (clBuildProgram(pr, 1, &d, "-cl-std=CL2.0 -cl-mad-enable", NULL, NULL) != CL_SUCCESS) {
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
    size_t gs = (size_t)rows * 16, ls = 16;
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
