// gemmbench: GPU GEMM for block_q4_0x8 weights vs CPU, prompt-style (many tokens), checked against a reference.
// W: [rows, K] as block_q4_0x8 (8 rows interleaved). X: [toks, K] fp32. Y: [toks, rows] fp32.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
void ggml_gemm_q4_0_8x8_q8_0(int, float *, size_t, const void *, const void *, int, int);
void ggml_quantize_mat_q8_0_4x8(const float *, void *, int64_t);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e_=(x);if(e_){fprintf(stderr,"%s: %d\n",#x,e_);exit(1);} }while(0)
static unsigned short f2h(float f){_Float16 h=(_Float16)f;unsigned short u;memcpy(&u,&h,2);return u;}

int main(int argc, char** argv){
    int rows = argc>1?atoi(argv[1]):1024, K = argc>2?atoi(argv[2]):2048, toks = argc>3?atoi(argv[3]):256;
    int nb = K/32;
    // build random block_q4_0x8 weights (144 bytes per 8 rows x 32 cols)
    size_t wb = (size_t)(rows/8)*nb*144;
    unsigned char* W = aligned_alloc(4096, wb);
    srand(1);
    for(size_t i=0;i<wb;i++) W[i]=rand();
    for(size_t g=0; g<(size_t)(rows/8)*nb; g++){ unsigned short* d=(unsigned short*)(W+g*144); for(int i=0;i<8;i++) d[i]=f2h(0.03f); }
    float* X = aligned_alloc(4096, (size_t)toks*K*4);
    for(size_t i=0;i<(size_t)toks*K;i++) X[i]=((rand()%2001)-1000)/1000.0f;
    float* Y = aligned_alloc(4096, (size_t)toks*rows*4);

    cl_platform_id pl; cl_device_id dev; CK(clGetPlatformIDs(1,&pl,NULL)); CK(clGetDeviceIDs(pl,CL_DEVICE_TYPE_GPU,1,&dev,NULL));
    cl_int e; cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e); CK(e);
    cl_queue_properties qp[]={CL_QUEUE_PROPERTIES,CL_QUEUE_PROFILING_ENABLE,0};
    cl_command_queue q=clCreateCommandQueueWithProperties(ctx,dev,qp,&e); CK(e);
    FILE* f=fopen(getenv("KFILE")?getenv("KFILE"):"gemm.cl","rb"); static char src[65536]; size_t n=fread(src,1,sizeof src-1,f); src[n]=0; fclose(f);
    const char* s=src; cl_program prog=clCreateProgramWithSource(ctx,1,&s,NULL,&e); CK(e);
    char opts[256]; snprintf(opts,sizeof opts,"-cl-std=CL1.2 -cl-mad-enable -cl-fast-relaxed-math -DK=%d",K);
    if(clBuildProgram(prog,1,&dev,opts,NULL,NULL)){ static char log[65536]; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,NULL); puts(log); return 1; }
    cl_kernel k=clCreateKernel(prog,"gemm_q4x8",&e); CK(e);
    cl_mem mW=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,wb,W,&e); CK(e);
    cl_mem mX=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,(size_t)toks*K*4,X,&e); CK(e);
    cl_mem mY=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY|CL_MEM_USE_HOST_PTR,(size_t)toks*rows*4,Y,&e); CK(e);
    int nbv=nb;
    CK(clSetKernelArg(k,0,sizeof mW,&mW)); CK(clSetKernelArg(k,1,sizeof mX,&mX)); CK(clSetKernelArg(k,2,sizeof mY,&mY));
    CK(clSetKernelArg(k,3,4,&nbv)); CK(clSetKernelArg(k,4,4,&rows)); CK(clSetKernelArg(k,5,4,&toks));
    // work: one group per (8-row block, token-tile). global = (rows/8 * SG_ROWS_LANES, toks/TT)
    int TT = getenv("TT")?atoi(getenv("TT")):8;
    int LSZ=getenv("LSZ")?atoi(getenv("LSZ")):16; size_t lws[2]={(size_t)LSZ,1}, gws[2]={(size_t)(rows/8)*LSZ, (size_t)(toks+TT-1)/TT};
    CK(clSetKernelArg(k,6,4,&TT));
    for(int w=0;w<5;w++) CK(clEnqueueNDRangeKernel(q,k,2,NULL,gws,lws,0,NULL,NULL)); clFinish(q);
    int reps=20; double t0=now(); double gpu_ns=0;
    for(int r=0;r<reps;r++){ cl_event ev; CK(clEnqueueNDRangeKernel(q,k,2,NULL,gws,lws,0,NULL,&ev)); clFinish(q);
        cl_ulong a,b; clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,8,&a,NULL); clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END,8,&b,NULL); gpu_ns+=(b-a); clReleaseEvent(ev); }
    double gpu_ms=(now()-t0)/reps*1e3;
    clEnqueueMapBuffer(q,mY,CL_TRUE,CL_MAP_READ,0,(size_t)toks*rows*4,0,NULL,NULL,&e); CK(e);
    double gmac = (double)rows*K*toks/1e9;
    printf("GPU  gemm: %.2f ms  (%.0f GMAC/s, kernel %.2f ms)  ", gpu_ms, gmac/(gpu_ms/1e3), gpu_ns/reps/1e6);
    // CPU reference (ggml gemm) for both timing and correctness
    size_t ab=(size_t)toks/4*nb*(4*34); void* A=aligned_alloc(64,ab+64);
    for(int t=0;t<toks;t+=4) ggml_quantize_mat_q8_0_4x8(X+(size_t)t*K, (char*)A+(size_t)(t/4)*nb*4*34, K); // 4-row tiles
    float* Yc=aligned_alloc(64,(size_t)toks*rows*4);
    t0=now(); for(int r=0;r<reps;r++) ggml_gemm_q4_0_8x8_q8_0(K,Yc,rows,W,A,toks,rows); double cpu_ms=(now()-t0)/reps*1e3;
    printf("CPU  gemm: %.2f ms  (%.0f GMAC/s)\n", cpu_ms, gmac/(cpu_ms/1e3));
    double worst=0,mag=0; for(int t=0;t<toks;t++) for(int rr=0;rr<rows;rr+=53){ double a=Y[(size_t)t*rows+rr],b=Yc[(size_t)t*rows+rr]; worst=fmax(worst,fabs(a-b)); mag=fmax(mag,fabs(b)); }
    printf("  GPU/CPU speed %.2fx | max |gpu-cpu| %.3g (largest %.3g)\n", cpu_ms/gpu_ms, worst, mag);
    return 0;
}
