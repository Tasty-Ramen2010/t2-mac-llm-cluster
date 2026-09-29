#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// v3: double-buffered local tiles (load next slab while computing current) to hide local/global latency.
#ifndef BM
#define BM 128
#define BN 128
#define BK 16
#define MR 8
#define NR 8
#endif
__attribute__((reqd_work_group_size(BN/NR, BM/MR, 1)))
__kernel void sgemm(__global const half* W, __global const half* X, __global float* Y, int rows, int toks){
    __local half As[2][BK][BM];
    __local half Bs[2][BK][BN];
    const int lx=get_local_id(0), ly=get_local_id(1);
    const int wgm=get_group_id(1)*BM, wgn=get_group_id(0)*BN;
    const int tid=ly*(BN/NR)+lx, nth=(BM/MR)*(BN/NR);
    float acc[MR][NR]; for(int i=0;i<MR;i++)for(int j=0;j<NR;j++)acc[i][j]=0.0f;
    int buf=0;
    for(int i=tid;i<BM*BK;i+=nth){int c=i/BM,r=i%BM; As[0][c][r]=W[(size_t)(wgm+r)*K+c];}
    for(int i=tid;i<BN*BK;i+=nth){int c=i/BN,r=i%BN; Bs[0][c][r]=X[(size_t)(wgn+r)*K+c];}
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int k0=0;k0<K;k0+=BK){
        int nb=buf^1;
        if(k0+BK<K){
            for(int i=tid;i<BM*BK;i+=nth){int c=i/BM,r=i%BM; As[nb][c][r]=W[(size_t)(wgm+r)*K+k0+BK+c];}
            for(int i=tid;i<BN*BK;i+=nth){int c=i/BN,r=i%BN; Bs[nb][c][r]=X[(size_t)(wgn+r)*K+k0+BK+c];}
        }
        for(int k=0;k<BK;k++){
            float a[MR],b[NR];
            for(int i=0;i<MR;i++)a[i]=As[buf][k][ly*MR+i];
            for(int j=0;j<NR;j++)b[j]=Bs[buf][k][lx*NR+j];
            for(int i=0;i<MR;i++)for(int j=0;j<NR;j++)acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        buf=nb;
    }
    for(int i=0;i<MR;i++)for(int j=0;j<NR;j++){int r=wgm+ly*MR+i,t=wgn+lx*NR+j; if(r<rows&&t<toks)Y[(size_t)t*rows+r]=acc[i][j];}
}
