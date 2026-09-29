#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// v5: SLM laid out so each work-item's MR-row / NR-col fragment is CONTIGUOUS -> one half8 vector load each per k,
// then the 8x8 outer product as 8 vector FMAs (av[i] broadcast * bv). Double-buffered. MR=NR=8 fixed.
#ifndef BM
#define BM 128
#define BN 128
#define BK 16
#endif
#define MR 8
#define NR 8
#define NTM (BM/MR)
#define NTN (BN/NR)
__attribute__((reqd_work_group_size(NTN, NTM, 1)))
__kernel void sgemm(__global const half* W, __global const half* X, __global float* Y, int rows, int toks){
    __local half As[2][BK][NTM][MR];
    __local half Bs[2][BK][NTN][NR];
    const int lx=get_local_id(0), ly=get_local_id(1);
    const int wgm=get_group_id(1)*BM, wgn=get_group_id(0)*BN;
    const int tid=ly*NTN+lx, nth=NTM*NTN;
    float8 acc[MR]; for(int i=0;i<MR;i++)acc[i]=(float8)0.0f;
    int buf=0;
    for(int i=tid;i<BM*BK;i+=nth){int c=i/BM,r=i%BM; As[0][c][r/MR][r%MR]=vload_half(0,W+(size_t)(wgm+r)*K+c);}
    for(int i=tid;i<BN*BK;i+=nth){int c=i/BN,r=i%BN; Bs[0][c][r/NR][r%NR]=vload_half(0,X+(size_t)(wgn+r)*K+c);}
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int k0=0;k0<K;k0+=BK){
        int nb=buf^1;
        if(k0+BK<K){
            for(int i=tid;i<BM*BK;i+=nth){int c=i/BM,r=i%BM; As[nb][c][r/MR][r%MR]=vload_half(0,W+(size_t)(wgm+r)*K+k0+BK+c);}
            for(int i=tid;i<BN*BK;i+=nth){int c=i/BN,r=i%BN; Bs[nb][c][r/NR][r%NR]=vload_half(0,X+(size_t)(wgn+r)*K+k0+BK+c);}
        }
        for(int k=0;k<BK;k++){
            const float8 av=convert_float8(vload8(0,&As[buf][k][ly][0]));
            const float8 bv=convert_float8(vload8(0,&Bs[buf][k][lx][0]));
            acc[0]=fma((float8)av.s0,bv,acc[0]); acc[1]=fma((float8)av.s1,bv,acc[1]);
            acc[2]=fma((float8)av.s2,bv,acc[2]); acc[3]=fma((float8)av.s3,bv,acc[3]);
            acc[4]=fma((float8)av.s4,bv,acc[4]); acc[5]=fma((float8)av.s5,bv,acc[5]);
            acc[6]=fma((float8)av.s6,bv,acc[6]); acc[7]=fma((float8)av.s7,bv,acc[7]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        buf=nb;
    }
    for(int i=0;i<MR;i++){int r=wgm+ly*MR+i; if(r<rows){ float tmp[8]; vstore8(acc[i],0,tmp);
        for(int j=0;j<NR;j++){int t=wgn+lx*NR+j; if(t<toks)Y[(size_t)t*rows+r]=tmp[j];}}}
}
