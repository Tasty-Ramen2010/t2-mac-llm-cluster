#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *src = "__kernel void add(__global float *x) { size_t i = get_global_id(0); x[i] = x[i] + 1.0f; }";
int main(void) {
    cl_platform_id p; cl_device_id d; cl_int e;
    clGetPlatformIDs(1, &p, NULL); clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, NULL);
    char name[256]; clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name), name, NULL);
    cl_context c = clCreateContext(NULL, 1, &d, NULL, NULL, &e);
    cl_command_queue q = clCreateCommandQueueWithProperties(c, d, NULL, &e);
    cl_program pr = clCreateProgramWithSource(c, 1, &src, NULL, &e); clBuildProgram(pr, 1, &d, NULL, NULL, NULL);
    cl_kernel k = clCreateKernel(pr, "add", &e);
    size_t n = 2880; static float host[2880];
    cl_mem buf = clCreateBuffer(c, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, n * sizeof(float), host, &e);
    clSetKernelArg(k, 0, sizeof(buf), &buf);
    for (int w = 0; w < 50; w++) { clEnqueueNDRangeKernel(q, k, 1, NULL, &n, NULL, 0, NULL, NULL); clFinish(q); }
    int iters = 2000; double t0 = now();
    for (int i = 0; i < iters; i++) {
        clEnqueueNDRangeKernel(q, k, 1, NULL, &n, NULL, 0, NULL, NULL);
        void *m = clEnqueueMapBuffer(q, buf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, n * sizeof(float), 0, NULL, NULL, &e);
        clEnqueueUnmapMemObject(q, buf, m, 0, NULL, NULL);
    }
    clFinish(q);
    double per = (now() - t0) / iters;
    printf("%s: launch tiny kernel + read result back = %.1f microseconds per round trip\n", name, per * 1e6);
    printf("-> 24 round trips per word (one per layer) = %.2f ms per word of pure overhead\n", per * 24e3);
    return 0;
}
