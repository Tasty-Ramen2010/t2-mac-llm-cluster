// Rigorous fp16/fp32 FMA peak: many independent accumulator chains per lane (hide FPU latency), full occupancy,
// results consumed so nothing is optimized away. Reports GMAC/s (1 FMA = 1 multiply-add).
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define CK(x) do{cl_int e_=(x);if(e_){fprintf(stderr,"%s:%d\n",#x,e_);exit(1);}}while(0)
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static const char* SRC=
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#define NC 16\n"                          // 16 independent chains > FPU latency*throughput
"__kernel void pf32(__global float* o,float s){ float8 a[NC],b=(float8)(s);\n"
"  for(int c=0;c<NC;c++)a[c]=(float8)(s+c);\n"
"  for(int i=0;i<2048;i++){ for(int c=0;c<NC;c++) a[c]=fma(a[c],b,a[c^1]); }\n"
"  float8 r=0; for(int c=0;c<NC;c++)r+=a[c]; o[get_global_id(0)]=r.s0+r.s1+r.s2+r.s3+r.s4+r.s5+r.s6+r.s7; }\n"
"__kernel void pf16(__global float* o,float s){ half8 a[NC],b=(half8)((half)s);\n"
"  for(int c=0;c<NC;c++)a[c]=(half8)((half)(s+c));\n"
"  for(int i=0;i<2048;i++){ for(int c=0;c<NC;c++) a[c]=fma(a[c],b,a[c^1]); }\n"
"  half8 r=0; for(int c=0;c<NC;c++)r+=a[c]; o[get_global_id(0)]=r.s0+r.s1+r.s2+r.s3+r.s4+r.s5+r.s6+r.s7; }\n";
int main(){
    cl_platform_id pl;cl_device_id dev;CK(clGetPlatformIDs(1,&pl,NULL));CK(clGetDeviceIDs(pl,CL_DEVICE_TYPE_GPU,1,&dev,NULL));
    cl_uint eu=0,freq=0;clGetDeviceInfo(dev,CL_DEVICE_MAX_COMPUTE_UNITS,4,&eu,NULL);clGetDeviceInfo(dev,CL_DEVICE_MAX_CLOCK_FREQUENCY,4,&freq,NULL);
    printf("device: %u EUs @ %u MHz\n",eu,freq);
    printf("theoretical fp32 = EU*2FPU*4SIMD*2(FMA=mul+add, count as 1 MAC each ->*1)*clk = %.0f GMAC/s; fp16 2x = %.0f\n",
        eu*2.0*4*1*freq/1e3, eu*2.0*4*1*freq/1e3*2);
    cl_int e;cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e);cl_command_queue q=clCreateCommandQueueWithProperties(ctx,dev,NULL,&e);
    cl_program p=clCreateProgramWithSource(ctx,1,&SRC,NULL,&e);CK(clBuildProgram(p,1,&dev,"-cl-fast-relaxed-math",NULL,NULL));
    size_t G=eu*7*8*64; cl_mem o=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,G*4,NULL,&e); float s=0.999f;
    const char* nm[2]={"pf32","pf16"}; double per[2]={16.0*8*2048,16.0*8*2048};
    for(int m=0;m<2;m++){ cl_kernel k=clCreateKernel(p,nm[m],&e);CK(clSetKernelArg(k,0,sizeof o,&o));CK(clSetKernelArg(k,1,4,&s));
        CK(clEnqueueNDRangeKernel(q,k,1,NULL,&G,NULL,0,NULL,NULL));clFinish(q);
        double t0=now();for(int r=0;r<20;r++)CK(clEnqueueNDRangeKernel(q,k,1,NULL,&G,NULL,0,NULL,NULL));clFinish(q);
        printf("%s: %.0f GMAC/s\n",nm[m],G*per[m]*20/(now()-t0)/1e9); }
    return 0;
}
