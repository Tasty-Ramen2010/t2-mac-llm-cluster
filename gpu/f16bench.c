#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e_=(x);if(e_){fprintf(stderr,"%s:%d\n",#x,e_);exit(1);}}while(0)
int main(int argc,char**argv){
    int rows=argc>1?atoi(argv[1]):1024,K=argc>2?atoi(argv[2]):2048,toks=argc>3?atoi(argv[3]):256,TT=getenv("TT")?atoi(getenv("TT")):8;
    unsigned short *W=aligned_alloc(4096,(size_t)rows*K*2),*X=aligned_alloc(4096,(size_t)toks*K*2); memset(W,0,(size_t)rows*K*2); memset(X,0,(size_t)toks*K*2);
    float* Y=aligned_alloc(4096,(size_t)toks*rows*4);
    cl_platform_id pl;cl_device_id dev;CK(clGetPlatformIDs(1,&pl,NULL));CK(clGetDeviceIDs(pl,CL_DEVICE_TYPE_GPU,1,&dev,NULL));
    cl_int e;cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e);CK(e);
    cl_queue_properties qp[]={CL_QUEUE_PROPERTIES,CL_QUEUE_PROFILING_ENABLE,0};cl_command_queue q=clCreateCommandQueueWithProperties(ctx,dev,qp,&e);CK(e);
    FILE*f=fopen("f16gemm.cl","rb");static char s[16384];size_t n=fread(s,1,sizeof s-1,f);s[n]=0;fclose(f);const char*sp=s;
    cl_program pr=clCreateProgramWithSource(ctx,1,&sp,NULL,&e);CK(e);char o[128];snprintf(o,sizeof o,"-cl-std=CL1.2 -cl-fast-relaxed-math -DK=%d",K);
    if(clBuildProgram(pr,1,&dev,o,NULL,NULL)){static char l[8192];clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,sizeof l,l,NULL);puts(l);return 1;}
    cl_kernel k=clCreateKernel(pr,"f16gemm",&e);CK(e);
    cl_mem mW=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,(size_t)rows*K*2,W,&e);
    cl_mem mX=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,(size_t)toks*K*2,X,&e);
    cl_mem mY=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY|CL_MEM_USE_HOST_PTR,(size_t)toks*rows*4,Y,&e);
    CK(clSetKernelArg(k,0,sizeof mW,&mW));CK(clSetKernelArg(k,1,sizeof mX,&mX));CK(clSetKernelArg(k,2,sizeof mY,&mY));
    CK(clSetKernelArg(k,3,4,&rows));CK(clSetKernelArg(k,4,4,&toks));CK(clSetKernelArg(k,5,4,&TT));
    size_t lws[2]={16,1},gws[2]={(rows+15)/16*16,(toks+TT-1)/TT};
    for(int w=0;w<5;w++)CK(clEnqueueNDRangeKernel(q,k,2,NULL,gws,lws,0,NULL,NULL));clFinish(q);
    int reps=30;double t0=now();for(int r=0;r<reps;r++)CK(clEnqueueNDRangeKernel(q,k,2,NULL,gws,lws,0,NULL,NULL));clFinish(q);
    double ms=(now()-t0)/reps*1e3;printf("fp16 dense GEMM ceiling: %.2f ms  %.0f GMAC/s (rows %d K %d toks %d TT %d)\n",ms,(double)rows*K*toks/1e9/(ms/1e3),rows,K,toks,TT);
    return 0;
}
