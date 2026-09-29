// int8/int32 arithmetic peak: does Gen9 do integer MACs faster than fp32? Tests mad_sat/mul24 and a manual int8 dot.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define CK(x) do{cl_int e_=(x);if(e_){fprintf(stderr,"%s:%d\n",#x,e_);exit(1);}}while(0)
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static const char* SRC=
"#define NC 16\n"
"__kernel void i32(__global int* o,int s){ int8 a[NC],b=(int8)(s|1);\n"
"  for(int c=0;c<NC;c++)a[c]=(int8)(s+c);\n"
"  for(int i=0;i<2048;i++){ for(int c=0;c<NC;c++) a[c]=a[c]*b+a[c^1]; }\n"
"  int8 r=0; for(int c=0;c<NC;c++)r+=a[c]; o[get_global_id(0)]=r.s0+r.s7; }\n"
"__kernel void mul24k(__global int* o,int s){ int8 a[NC],b=(int8)(s|1);\n"
"  for(int c=0;c<NC;c++)a[c]=(int8)((s+c)&0x7fff);\n"
"  for(int i=0;i<2048;i++){ for(int c=0;c<NC;c++) a[c]=mad24(a[c],b,a[c^1]); }\n"
"  int8 r=0; for(int c=0;c<NC;c++)r+=a[c]; o[get_global_id(0)]=r.s0+r.s7; }\n";
int main(){
    cl_platform_id pl;cl_device_id dev;CK(clGetPlatformIDs(1,&pl,NULL));CK(clGetDeviceIDs(pl,CL_DEVICE_TYPE_GPU,1,&dev,NULL));
    cl_int e;cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e);cl_command_queue q=clCreateCommandQueueWithProperties(ctx,dev,NULL,&e);
    cl_program p=clCreateProgramWithSource(ctx,1,&SRC,NULL,&e);if(clBuildProgram(p,1,&dev,"",NULL,NULL)){char l[8192];clGetProgramBuildInfo(p,dev,CL_PROGRAM_BUILD_LOG,sizeof l,l,NULL);puts(l);return 1;}
    size_t G=24*7*8*64;cl_mem o=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,G*4,NULL,&e);int s=12345;
    const char* nm[2]={"i32","mul24k"};
    for(int m=0;m<2;m++){cl_kernel k=clCreateKernel(p,nm[m],&e);CK(clSetKernelArg(k,0,sizeof o,&o));CK(clSetKernelArg(k,1,4,&s));
        CK(clEnqueueNDRangeKernel(q,k,1,NULL,&G,NULL,0,NULL,NULL));clFinish(q);
        double t0=now();for(int r=0;r<20;r++)CK(clEnqueueNDRangeKernel(q,k,1,NULL,&G,NULL,0,NULL,NULL));clFinish(q);
        printf("%s: %.0f G int-MAC/s\n",nm[m],G*16.0*8*2048*20/(now()-t0)/1e9);}
    return 0;
}
