#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// Tiled fp16 GEMM v2: local tiles stored K-major (As[BK][BM]) so inner reads are contiguous; vectorized MRxNR
// micro-tile with float accumulation; each work-item computes MR(=8) x NR(=8) using half8 broadcasts.
#ifndef BM
#define BM 64
#define BN 64
#define BK 16
#define MR 8
#define NR 8
#endif
__attribute__((reqd_work_group_size(BN/NR, BM/MR, 1)))
__kernel void sgemm(__global const half* W, __global const half* X, __global float* Y, int rows, int toks){
    __local half As[BK][BM];
    __local half Bs[BK][BN];
    const int lx=get_local_id(0), ly=get_local_id(1);
    const int wgm=get_group_id(1)*BM, wgn=get_group_id(0)*BN;
    const int tid=ly*(BN/NR)+lx, nth=(BM/MR)*(BN/NR);
    float acc[MR][NR]; for(int i=0;i<MR;i++)for(int j=0;j<NR;j++)acc[i][j]=0.0f;
    for(int k0=0;k0<K;k0+=BK){
        for(int i=tid;i<BM*BK;i+=nth){int c=i/BM,r=i%BM; As[c][r]=W[(size_t)(wgm+r)*K+k0+c];}
        for(int i=tid;i<BN*BK;i+=nth){int c=i/BN,r=i%BN; Bs[c][r]=X[(size_t)(wgn+r)*K+k0+c];}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int k=0;k<BK;k++){
            float a[MR],b[NR];
            for(int i=0;i<MR;i++)a[i]=As[k][ly*MR+i];
            for(int j=0;j<NR;j++)b[j]=Bs[k][lx*NR+j];
            for(int i=0;i<MR;i++)for(int j=0;j<NR;j++)acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for(int i=0;i<MR;i++)for(int j=0;j<NR;j++){int r=wgm+ly*MR+i,t=wgn+lx*NR+j; if(r<rows&&t<toks)Y[(size_t)t*rows+r]=acc[i][j];}
}
