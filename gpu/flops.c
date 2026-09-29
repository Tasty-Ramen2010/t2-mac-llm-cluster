// flops: the UHD 630's real multiply-add rate in a perfect register loop (fp32 and fp16)
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define CK(x) do { cl_int e_ = (x); if (e_) { fprintf(stderr, "%s: %d\n", #x, e_); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char * SRC =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void f32(__global float * o, float a) { float8 x = (float8)(get_global_id(0)), y = x + 1, z = x + 2, w = x + 3;\n"
"  float8 b = (float8)(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);\n"
"  for (int i = 0; i < 1024; i++) { x = fma(x, b, y); y = fma(y, b, z); z = fma(z, b, w); w = fma(w, b, x); }\n"
"  o[get_global_id(0)] = x.s0 + y.s1 + z.s2 + w.s3; }\n"
"__kernel void f16(__global float * o, float af) { half a = (half) af; half8 x = (half8)(get_global_id(0)), y = x + (half)1, z = x + (half)2, w = x + (half)3;\n"
"  half8 x2 = x + (half)4, y2 = x + (half)5, z2 = x + (half)6, w2 = x + (half)7;\n"
"  half8 b = convert_half8((float8)(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]));\n"
"  for (int i = 0; i < 1024; i++) { x = fma(x, b, y); y = fma(y, b, z); z = fma(z, b, w); w = fma(w, b, x2);\n"
"    x2 = fma(x2, b, y2); y2 = fma(y2, b, z2); z2 = fma(z2, b, w2); w2 = fma(w2, b, x); }\n"
"  o[get_global_id(0)] = x.s0 + y.s1 + z.s2 + w.s3 + x2.s0 + y2.s1 + z2.s2 + w2.s3; }\n";
int main(void) {
    cl_platform_id pl; cl_device_id dev; CK(clGetPlatformIDs(1, &pl, NULL)); CK(clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    cl_int e; cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e); CK(e);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &e); CK(e);
    cl_program p = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &e); CK(e); CK(clBuildProgram(p, 1, &dev, "", NULL, NULL));
    size_t G = 24 * 7 * 16 * 16; cl_mem o = clCreateBuffer(ctx, CL_MEM_READ_WRITE, G * 4, NULL, &e); CK(e); { float z[8] = {0.5f,0.4f,0.3f,0.2f,0.1f,0.6f,0.7f,0.8f}; CK(clEnqueueWriteBuffer(q, o, CL_TRUE, 0, 32, z, 0, NULL, NULL)); } float a = 0.999f;
    const char * names[2] = { "f32", "f16" }; double fmas[2] = { 32.0 * 1024, 64.0 * 1024 };
    for (int k = 0; k < 2; k++) {
        cl_kernel kk = clCreateKernel(p, names[k], &e); CK(e); CK(clSetKernelArg(kk, 0, sizeof o, &o)); CK(clSetKernelArg(kk, 1, 4, &a));
        CK(clEnqueueNDRangeKernel(q, kk, 1, NULL, &G, NULL, 0, NULL, NULL)); clFinish(q);
        double t0 = now(); for (int r = 0; r < 10; r++) CK(clEnqueueNDRangeKernel(q, kk, 1, NULL, &G, NULL, 0, NULL, NULL)); clFinish(q);
        printf("%s: %.0f G multiply-adds/s\n", names[k], G * fmas[k] * 10 / (now() - t0) / 1e9);
    }
    return 0;
}
