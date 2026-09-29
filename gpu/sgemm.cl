#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// Shared-memory tiled fp16 GEMM. Coalesced loads of a BMxBK slab of W and BKxBN slab of X into local memory, then
// each work-item computes an MRxNR micro-tile from local. WG = (BN/NR, BM/MR). C[t,row]=sum_k W[row,k]*X[t,k].
// BM,BN output tile; BK depth. W[rows,K] row-major, X[toks,K] row-major, Y[toks,rows].
#ifndef BM
#define BM 64
#define BN 64
#define BK 16
#define MR 4
#define NR 4
#endif
__attribute__((reqd_work_group_size(BN/NR, BM/MR, 1)))
__kernel void sgemm(__global const half* W, __global const half* X, __global float* Y, int rows, int toks){
    __local half As[BM][BK];   // W rows
    __local half Bs[BN][BK];   // X rows (tokens)
    const int lx=get_local_id(0), ly=get_local_id(1);      // lx over BN/NR, ly over BM/MR
    const int wgm=get_group_id(1)*BM, wgn=get_group_id(0)*BN;
    const int tid=ly*(BN/NR)+lx, nth=(BM/MR)*(BN/NR);
    float acc[MR][NR]; for(int i=0;i<MR;i++)for(int j=0;j<NR;j++)acc[i][j]=0.0f;
    for(int k0=0;k0<K;k0+=BK){
        for(int i=tid;i<BM*BK;i+=nth){int r=i/BK,c=i%BK; As[r][c]=W[(size_t)(wgm+r)*K+k0+c];}
        for(int i=tid;i<BN*BK;i+=nth){int r=i/BK,c=i%BK; Bs[r][c]=X[(size_t)(wgn+r)*K+k0+c];}
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int k=0;k<BK;k++){
            half a[MR],b[NR];
            for(int i=0;i<MR;i++)a[i]=As[ly*MR+i][k];
            for(int j=0;j<NR;j++)b[j]=Bs[lx*NR+j][k];
            for(int i=0;i<MR;i++)for(int j=0;j<NR;j++)acc[i][j]+=(float)(a[i]*b[j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for(int i=0;i<MR;i++)for(int j=0;j<NR;j++){int r=wgm+ly*MR+i,t=wgn+lx*NR+j; if(r<rows&&t<toks)Y[(size_t)t*rows+r]=acc[i][j];}
}
