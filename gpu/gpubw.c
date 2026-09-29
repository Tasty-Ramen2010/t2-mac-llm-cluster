// gpubw: can the UHD 630 add memory bandwidth on top of the CPU? zero-copy OpenCL buffer over host RAM.
// 1) GPU alone  2) CPU (N threads) alone  3) both at once  4) GPU dispatch round-trip latency
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define CK(x) do { cl_int e_ = (x); if (e_) { fprintf(stderr, "%s: %d\n", #x, e_); exit(1); } } while (0)
static const char * SRC =
"__kernel void rd(__global const uint4 * p, ulong n, __global uint * out) {\n"
"  size_t g = get_global_id(0), G = get_global_size(0); uint4 a = 0;\n"
"  for (ulong i = g; i < n; i += G) a ^= p[i];\n"
"  out[g] = a.x ^ a.y ^ a.z ^ a.w; }\n"
"__kernel void tiny(__global uint * out) { out[get_global_id(0)] += 1; }\n";
static char * cbuf; static size_t cbytes; static atomic_int stop; static double cpu_bytes[8];
static void * cpu_reader(void * arg) {
    long id = (long) arg; int nt = (int) cpu_bytes[7]; size_t chunk = cbytes / nt, lo = id * chunk;
    __m256i acc = _mm256_setzero_si256(); double done = 0;
    while (!atomic_load(&stop)) {
        for (size_t i = lo; i < lo + chunk; i += 128) {
            acc = _mm256_xor_si256(acc, _mm256_load_si256((__m256i *) (cbuf + i)));
            acc = _mm256_xor_si256(acc, _mm256_load_si256((__m256i *) (cbuf + i + 32)));
            acc = _mm256_xor_si256(acc, _mm256_load_si256((__m256i *) (cbuf + i + 64)));
            acc = _mm256_xor_si256(acc, _mm256_load_si256((__m256i *) (cbuf + i + 96)));
        }
        done += chunk;
    }
    cpu_bytes[id] = done + _mm256_extract_epi32(acc, 0) * 0.0;
    return NULL;
}
int main(int argc, char ** argv) {
    int nthr = argc > 1 ? atoi(argv[1]) : 3;
    size_t gbytes = (size_t) 768 << 20; cbytes = (size_t) 768 << 20;
    char * gbuf = aligned_alloc(4096, gbytes); memset(gbuf, 3, gbytes);
    cbuf = aligned_alloc(1 << 21, cbytes); madvise(cbuf, cbytes, MADV_HUGEPAGE); memset(cbuf, 5, cbytes);
    cl_platform_id pl; cl_device_id dev; CK(clGetPlatformIDs(1, &pl, NULL)); CK(clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    cl_int e; cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e); CK(e);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &e); CK(e);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &e); CK(e);
    if (clBuildProgram(prog, 1, &dev, "-cl-std=CL1.2", NULL, NULL)) { char log[8192]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL); puts(log); return 1; }
    cl_kernel krd = clCreateKernel(prog, "rd", &e); CK(e); cl_kernel ktiny = clCreateKernel(prog, "tiny", &e); CK(e);
    const char * mode = argc > 2 ? argv[2] : "hostptr";
    cl_mem mw;
    if (!strcmp(mode, "hostptr")) { mw = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, gbytes, gbuf, &e); CK(e); }   // zero-copy, coherent
    else {                                                     // driver-owned GPU memory (filled once by copy)
        mw = clCreateBuffer(ctx, CL_MEM_READ_ONLY, gbytes, NULL, &e); CK(e);
        CK(clEnqueueWriteBuffer(q, mw, CL_TRUE, 0, gbytes, gbuf, 0, NULL, NULL));
    }
    printf("-- GPU buffer: %s, CPU threads %d\n", mode, nthr);
    size_t G = (size_t) (24 * 7 * 16 * 8 * (getenv("GPUSCALE") ? atof(getenv("GPUSCALE")) : 1.0)) / 256 * 256; if (G < 256) G = 256; cl_mem mo = clCreateBuffer(ctx, CL_MEM_READ_WRITE, G * 4, NULL, &e); CK(e);
    cl_ulong n = gbytes / 16;
    CK(clSetKernelArg(krd, 0, sizeof mw, &mw)); CK(clSetKernelArg(krd, 1, sizeof n, &n)); CK(clSetKernelArg(krd, 2, sizeof mo, &mo));
    CK(clSetKernelArg(ktiny, 0, sizeof mo, &mo));
    size_t lws = 256;
    for (int w = 0; w < 3; w++) { CK(clEnqueueNDRangeKernel(q, krd, 1, NULL, &G, &lws, 0, NULL, NULL)); } clFinish(q);
    // 1) GPU alone
    int reps = 20; double t0 = now();
    for (int r = 0; r < reps; r++) CK(clEnqueueNDRangeKernel(q, krd, 1, NULL, &G, &lws, 0, NULL, NULL));
    clFinish(q); double gpu_alone = gbytes * reps / (now() - t0) / 1e9;
    printf("GPU alone:            %5.1f GB/s\n", gpu_alone);
    // 2) CPU alone
    cpu_bytes[7] = nthr; pthread_t th[8]; atomic_store(&stop, 0);
    for (long i = 0; i < nthr; i++) pthread_create(&th[i], NULL, cpu_reader, (void *) i);
    t0 = now(); struct timespec s2 = {1, 500000000}; nanosleep(&s2, NULL); atomic_store(&stop, 1);
    for (int i = 0; i < nthr; i++) pthread_join(th[i], NULL);
    double el = now() - t0, cb = 0; for (int i = 0; i < nthr; i++) cb += cpu_bytes[i];
    printf("CPU alone (%d thr):    %5.1f GB/s\n", nthr, cb / el / 1e9);
    // 3) both
    atomic_store(&stop, 0);
    for (long i = 0; i < nthr; i++) pthread_create(&th[i], NULL, cpu_reader, (void *) i);
    t0 = now(); int gr = 0;
    while (now() - t0 < 1.5) { CK(clEnqueueNDRangeKernel(q, krd, 1, NULL, &G, &lws, 0, NULL, NULL)); clFinish(q); gr++; }
    double tg = now() - t0; atomic_store(&stop, 1);
    for (int i = 0; i < nthr; i++) pthread_join(th[i], NULL);
    el = now() - t0; cb = 0; for (int i = 0; i < nthr; i++) cb += cpu_bytes[i];
    double g_both = (double) gbytes * gr / tg / 1e9, c_both = cb / el / 1e9;
    printf("Both at once:         CPU %5.1f + GPU %5.1f = %5.1f GB/s\n", c_both, g_both, c_both + g_both);
    // 4) dispatch latency
    size_t one = 1; for (int w = 0; w < 50; w++) { clEnqueueNDRangeKernel(q, ktiny, 1, NULL, &one, NULL, 0, NULL, NULL); clFinish(q); }
    t0 = now(); for (int r = 0; r < 2000; r++) { clEnqueueNDRangeKernel(q, ktiny, 1, NULL, &one, NULL, 0, NULL, NULL); clFinish(q); }
    printf("Dispatch + wait:      %5.1f us per kernel\n", (now() - t0) / 2000 * 1e6);
    return 0;
}
