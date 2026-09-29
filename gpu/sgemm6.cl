#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// v6: double-buffered SLM (global latency hidden) + register prefetch of the next k-slice's fragments (SLM read
// latency hidden behind the FMAs) + unrolled k. SIMD8, 8x8 micro-tile. Fragments stored contiguous for wide reads.
#ifndef BM
#define BM 128
#define BN 128
#define BK 16
#endif
#define MR 8
#define NR 8
#define NTM (BM/MR)
#define NTN (BN/NR)
__attribute__((reqd_work_group_size(NTN, NTM, 1))) __attribute__((intel_reqd_sub_group_size(8)))
__kernel void sgemm(__global const half* W, __global const half* X, __global float* Y, int rows, int toks){
    __local half As[2][BK][BM];
    __local half Bs[2][BK][BN];
    const int lx=get_local_id(0), ly=get_local_id(1);
    const int wgm=get_group_id(1)*BM, wgn=get_group_id(0)*BN;
    const int tid=ly*NTN+lx, nth=NTM*NTN;
    float acc[MR][NR]; for(int i=0;i<MR;i++)for(int j=0;j<NR;j++)acc[i][j]=0.0f;
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
        // register prefetch: load k=0 fragments, then loop loads k+1 while computing k
        float a[MR],b[NR],an[MR],bn[NR];
        for(int i=0;i<MR;i++)a[i]=As[buf][0][ly*MR+i];
        for(int j=0;j<NR;j++)b[j]=Bs[buf][0][lx*NR+j];
        #pragma unroll 4
        for(int k=0;k<BK;k++){
            if(k+1<BK){ for(int i=0;i<MR;i++)an[i]=As[buf][k+1][ly*MR+i]; for(int j=0;j<NR;j++)bn[j]=Bs[buf][k+1][lx*NR+j]; }
            for(int i=0;i<MR;i++)for(int j=0;j<NR;j++)acc[i][j]=fma(a[i],b[j],acc[i][j]);
            for(int i=0;i<MR;i++)a[i]=an[i]; for(int j=0;j<NR;j++)b[j]=bn[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        buf=nb;
    }
    for(int i=0;i<MR;i++)for(int j=0;j<NR;j++){int r=wgm+ly*MR+i,t=wgn+lx*NR+j; if(r<rows&&t<toks)Y[(size_t)t*rows+r]=acc[i][j];}
}
