// gemvbench: GPU Q4_0x8 GEMV (generation shape, 1 token) throughput in GB/s of weights, vs the CPU's ~25 GB/s.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e_=(x);if(e_){fprintf(stderr,"%s:%d\n",#x,e_);exit(1);}}while(0)
static unsigned short f2h(float f){_Float16 h=(_Float16)f;unsigned short u;memcpy(&u,&h,2);return u;}
int main(int argc,char**argv){
    int rows=argc>1?atoi(argv[1]):8192, K=argc>2?atoi(argv[2]):2048; int nb=K/32;
    size_t wb=(size_t)(rows/8)*nb*144; unsigned char*W=aligned_alloc(4096,wb);
    for(size_t i=0;i<wb;i++)W[i]=i; for(size_t g=0;g<(size_t)(rows/8)*nb;g++){unsigned short*d=(unsigned short*)(W+g*144);for(int i=0;i<8;i++)d[i]=f2h(0.03f);}
    float*X=aligned_alloc(4096,K*4); for(int i=0;i<K;i++)X[i]=0.01f*i; float*Y=aligned_alloc(4096,rows*4);
    cl_platform_id pl;cl_device_id dev;CK(clGetPlatformIDs(1,&pl,NULL));CK(clGetDeviceIDs(pl,CL_DEVICE_TYPE_GPU,1,&dev,NULL));
    cl_int e;cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e);CK(e);
    cl_queue_properties qp[]={CL_QUEUE_PROPERTIES,CL_QUEUE_PROFILING_ENABLE,0};cl_command_queue q=clCreateCommandQueueWithProperties(ctx,dev,qp,&e);CK(e);
    FILE*f=fopen("gemv.cl","rb");static char s[16384];size_t n=fread(s,1,sizeof s-1,f);s[n]=0;fclose(f);const char*sp=s;
    cl_program pr=clCreateProgramWithSource(ctx,1,&sp,NULL,&e);CK(e);
    if(clBuildProgram(pr,1,&dev,"-cl-std=CL2.0 -cl-mad-enable",NULL,NULL)){static char l[8192];clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,sizeof l,l,NULL);puts(l);return 1;}
    cl_kernel k=clCreateKernel(pr,"gemv_q4x8",&e);CK(e);
    cl_mem mW=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,wb,W,&e);
    cl_mem mX=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,K*4,X,&e);
    cl_mem mY=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY|CL_MEM_USE_HOST_PTR,rows*4,Y,&e);
    cl_ulong z=0; int SG=getenv("SG")?atoi(getenv("SG")):8;
    CK(clSetKernelArg(k,0,sizeof mW,&mW));CK(clSetKernelArg(k,1,8,&z));CK(clSetKernelArg(k,2,sizeof mX,&mX));CK(clSetKernelArg(k,3,8,&z));
    CK(clSetKernelArg(k,4,sizeof mY,&mY));CK(clSetKernelArg(k,5,8,&z));CK(clSetKernelArg(k,6,4,&nb));CK(clSetKernelArg(k,7,(size_t)nb*32*4,NULL));
    size_t lws=16*SG, gws=(size_t)rows/8*16;
    for(int w=0;w<5;w++)CK(clEnqueueNDRangeKernel(q,k,1,NULL,&gws,&lws,0,NULL,NULL));clFinish(q);
    int reps=200;double t0=now(),gns=0;
    for(int r=0;r<reps;r++){cl_event ev;CK(clEnqueueNDRangeKernel(q,k,1,NULL,&gws,&lws,0,NULL,&ev));clFinish(q);cl_ulong a,b;clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,8,&a,NULL);clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END,8,&b,NULL);gns+=b-a;clReleaseEvent(ev);}
    double ms=(now()-t0)/reps*1e3; double bytes=(double)(rows/8)*nb*144;
    printf("GPU GEMV %d rows K%d SG%d: %.3f ms/call, kernel %.3f ms -> %.1f GB/s (with launch), %.1f GB/s (kernel only)\n",
        rows,K,SG,ms,gns/reps/1e6,bytes/(ms/1e3)/1e9,bytes/(gns/reps/1e9)/1e9);
    return 0;
}
