#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e_=(x);if(e_){fprintf(stderr,"%s:%d\n",#x,e_);exit(1);}}while(0)
int main(int argc,char**argv){
    int rows=argc>1?atoi(argv[1]):1024,K=argc>2?atoi(argv[2]):2048,toks=argc>3?atoi(argv[3]):256;
    int BM=atoi(getenv("BM")),BN=atoi(getenv("BN")),BK=atoi(getenv("BK")),MR=atoi(getenv("MR")),NR=atoi(getenv("NR")),PAD=getenv("PAD")?atoi(getenv("PAD")):1;
    unsigned short*W=aligned_alloc(4096,(size_t)rows*K*2),*X=aligned_alloc(4096,(size_t)toks*K*2);memset(W,0,(size_t)rows*K*2);memset(X,0,(size_t)toks*K*2);
    float*Y=aligned_alloc(4096,(size_t)toks*rows*4);
    cl_platform_id pl;cl_device_id dev;CK(clGetPlatformIDs(1,&pl,NULL));CK(clGetDeviceIDs(pl,CL_DEVICE_TYPE_GPU,1,&dev,NULL));
    cl_int e;cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e);CK(e);
    cl_queue_properties qp[]={CL_QUEUE_PROPERTIES,CL_QUEUE_PROFILING_ENABLE,0};cl_command_queue q=clCreateCommandQueueWithProperties(ctx,dev,qp,&e);CK(e);
    FILE*f=fopen(getenv("KFILE")?getenv("KFILE"):"sgemm.cl","rb");static char s[16384];size_t n=fread(s,1,sizeof s-1,f);s[n]=0;fclose(f);const char*sp=s;
    cl_program pr=clCreateProgramWithSource(ctx,1,&sp,NULL,&e);CK(e);char o[256];snprintf(o,sizeof o,"-cl-std=CL1.2 -cl-fast-relaxed-math %s ",getenv("XOPT")?getenv("XOPT"):""); char o2[256]; snprintf(o2,sizeof o2," -DK=%d -DBM=%d -DBN=%d -DBK=%d -DMR=%d -DNR=%d -DPAD=%d",K,BM,BN,BK,MR,NR,PAD); strcat(o,o2);
    if(clBuildProgram(pr,1,&dev,o,NULL,NULL)){static char l[8192];clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,sizeof l,l,NULL);puts(l);return 1;}
    cl_kernel k=clCreateKernel(pr,"sgemm",&e);CK(e);
    cl_mem mW=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,(size_t)rows*K*2,W,&e);
    cl_mem mX=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,(size_t)toks*K*2,X,&e);
    cl_mem mY=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY|CL_MEM_USE_HOST_PTR,(size_t)toks*rows*4,Y,&e);
    CK(clSetKernelArg(k,0,sizeof mW,&mW));CK(clSetKernelArg(k,1,sizeof mX,&mX));CK(clSetKernelArg(k,2,sizeof mY,&mY));CK(clSetKernelArg(k,3,4,&rows));CK(clSetKernelArg(k,4,4,&toks));
    size_t lws[2]={(size_t)BN/NR,(size_t)BM/MR}, gws[2]={(size_t)(toks+BN-1)/BN*(BN/NR),(size_t)(rows+BM-1)/BM*(BM/MR)};
    for(int w=0;w<5;w++)CK(clEnqueueNDRangeKernel(q,k,2,NULL,gws,lws,0,NULL,NULL));clFinish(q);
    int reps=30;double t0=now(),gns=0;
    for(int r=0;r<reps;r++){cl_event ev;CK(clEnqueueNDRangeKernel(q,k,2,NULL,gws,lws,0,NULL,&ev));clFinish(q);cl_ulong a,b;clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,8,&a,NULL);clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END,8,&b,NULL);gns+=b-a;clReleaseEvent(ev);}
    double ms=(now()-t0)/reps*1e3;
    printf("sgemm BM%d BN%d BK%d MR%d NR%d: %.2f ms  %.0f GMAC/s (kernel %.0f)\n",BM,BN,BK,MR,NR,ms,(double)rows*K*toks/1e9/(ms/1e3),(double)rows*K*toks/1e9/(gns/reps/1e9));
    return 0;
}
