#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do{cl_int e_=(x);if(e_){fprintf(stderr,"%s:%d\n",#x,e_);exit(1);}}while(0)
static unsigned short f2h(float f){_Float16 h=(_Float16)f;unsigned short u;memcpy(&u,&h,2);return u;}
static float h2f(unsigned short u){_Float16 h;memcpy(&h,&u,2);return h;}
int main(){
    int rows=256,K=512,toks=64,BM=128,BN=128,BK=32;
    unsigned short*W=aligned_alloc(4096,(size_t)rows*K*2),*X=aligned_alloc(4096,(size_t)toks*K*2);
    srand(3); for(size_t i=0;i<(size_t)rows*K;i++)W[i]=f2h(((rand()%200)-100)/100.0f); for(size_t i=0;i<(size_t)toks*K;i++)X[i]=f2h(((rand()%200)-100)/100.0f);
    float*Y=aligned_alloc(4096,(size_t)toks*rows*4);
    cl_platform_id pl;cl_device_id dev;CK(clGetPlatformIDs(1,&pl,NULL));CK(clGetDeviceIDs(pl,CL_DEVICE_TYPE_GPU,1,&dev,NULL));
    cl_int e;cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e);cl_command_queue q=clCreateCommandQueueWithProperties(ctx,dev,NULL,&e);
    FILE*f=fopen("sgemm7.cl","rb");static char s[16384];size_t n=fread(s,1,sizeof s-1,f);s[n]=0;fclose(f);const char*sp=s;
    cl_program pr=clCreateProgramWithSource(ctx,1,&sp,NULL,&e);char o[256];snprintf(o,sizeof o,"-cl-std=CL1.2 -cl-fast-relaxed-math -DK=%d -DBM=%d -DBN=%d -DBK=%d",K,BM,BN,BK);
    if(clBuildProgram(pr,1,&dev,o,NULL,NULL)){static char l[8192];clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,sizeof l,l,NULL);puts(l);return 1;}
    cl_kernel k=clCreateKernel(pr,"sgemm",&e);
    cl_mem mW=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,(size_t)rows*K*2,W,&e),mX=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,(size_t)toks*K*2,X,&e),mY=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY|CL_MEM_USE_HOST_PTR,(size_t)toks*rows*4,Y,&e);
    CK(clSetKernelArg(k,0,sizeof mW,&mW));CK(clSetKernelArg(k,1,sizeof mX,&mX));CK(clSetKernelArg(k,2,sizeof mY,&mY));CK(clSetKernelArg(k,3,4,&rows));CK(clSetKernelArg(k,4,4,&toks));
    size_t lws[2]={BN/8,BM/8},gws[2]={(size_t)(toks+BN-1)/BN*(BN/8),(size_t)(rows+BM-1)/BM*(BM/8)};
    CK(clEnqueueNDRangeKernel(q,k,2,NULL,gws,lws,0,NULL,NULL));clFinish(q);
    clEnqueueMapBuffer(q,mY,CL_TRUE,CL_MAP_READ,0,(size_t)toks*rows*4,0,NULL,NULL,&e);
    double worst=0,mag=0;
    for(int t=0;t<toks;t+=7)for(int r=0;r<rows;r+=11){double ref=0;for(int kk=0;kk<K;kk++)ref+=(double)h2f(X[(size_t)t*K+kk])*h2f(W[(size_t)r*K+kk]);
        worst=fmax(worst,fabs(ref-Y[(size_t)t*rows+r]));mag=fmax(mag,fabs(ref));}
    printf("correctness: max |gpu-exact| = %.4g (largest value %.4g, rel %.2g)\n",worst,mag,worst/mag);
    return 0;
}
