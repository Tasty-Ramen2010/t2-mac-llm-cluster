#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// v8: same as v7 but ACCUMULATE IN FP16 (half). Gen9 does fp16 FMA at 2x fp32 rate, so this shows the true fp16
// ceiling for this GEMM shape. Precision is lower (fine for a demonstration / short-K); real use keeps fp32 (v7).
#ifndef BM
#define BM 128
#define BN 128
#define BK 32
#endif
#define MR 8
#define NR 8
__attribute__((reqd_work_group_size(BN/NR, BM/MR, 1))) __attribute__((intel_reqd_sub_group_size(8)))
__kernel void sgemm(__global const half* W, __global const half* X, __global float* Y, int rows, int toks){
    __local half As[2][BK][BM];
    __local half Bs[2][BK][BN];
    const int lx=get_local_id(0), ly=get_local_id(1);
    const int wgm=get_group_id(1)*BM, wgn=get_group_id(0)*BN;
    const int tid=ly*(BN/NR)+lx, nth=(BM/MR)*(BN/NR);
    half8 acc[MR]; for(int i=0;i<MR;i++)acc[i]=(half8)0;
    int buf=0;
    for(int i=tid;i<BM*BK;i+=nth){int c=i/BM,r=i%BM; As[0][c][r]=vload_half(0,W+(size_t)(wgm+r)*K+c);}
    for(int i=tid;i<BN*BK;i+=nth){int c=i/BN,r=i%BN; Bs[0][c][r]=vload_half(0,X+(size_t)(wgn+r)*K+c);}
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int k0=0;k0<K;k0+=BK){
        int nb=buf^1;
        if(k0+BK<K){
            for(int i=tid;i<BM*BK;i+=nth){int c=i/BM,r=i%BM; As[nb][c][r]=vload_half(0,W+(size_t)(wgm+r)*K+k0+BK+c);}
            for(int i=tid;i<BN*BK;i+=nth){int c=i/BN,r=i%BN; Bs[nb][c][r]=vload_half(0,X+(size_t)(wgn+r)*K+k0+BK+c);}
        }
        for(int k=0;k<BK;k++){
            const uint4 ai=vload4(0,(__local const uint*)&As[buf][k][ly*MR]);
            const half8 bv=as_half8(vload4(0,(__local const uint*)&Bs[buf][k][lx*NR]));
            half a[MR]; const half2 a0=as_half2(ai.s0),a1=as_half2(ai.s1),a2=as_half2(ai.s2),a3=as_half2(ai.s3);
            a[0]=a0.s0;a[1]=a0.s1;a[2]=a1.s0;a[3]=a1.s1;a[4]=a2.s0;a[5]=a2.s1;a[6]=a3.s0;a[7]=a3.s1;
            for(int i=0;i<MR;i++)acc[i]=fma((half8)a[i],bv,acc[i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        buf=nb;
    }
    for(int i=0;i<MR;i++){int r=wgm+ly*MR+i; if(r<rows){half tmp[8];vstore8(acc[i],0,tmp);
        for(int j=0;j<NR;j++){int t=wgn+lx*NR+j; if(t<toks)Y[(size_t)t*rows+r]=tmp[j];}}}
}
