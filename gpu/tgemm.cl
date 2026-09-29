#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// Register-blocked dense fp16 GEMM: each work-item computes an MR x NR output tile, rank-1 updates in registers.
// Y[t,row] = sum_k X[t,k]*W[row,k]. W row-major [rows,K], X row-major [toks,K].
// Loads MR W-values and NR X-values per k, does MR*NR FMAs -> compute:load ratio MR*NR/(MR+NR).
#ifndef MR
#define MR 8
#endif
#ifndef NR
#define NR 8
#endif
__kernel void tgemm(__global const half* W, __global const half* X, __global float* Y, int rows, int toks){
    const int row0 = get_global_id(0)*MR;   // MR rows
    const int t0   = get_global_id(1)*NR;   // NR tokens
    float acc[MR][NR];
    for(int i=0;i<MR;i++)for(int j=0;j<NR;j++)acc[i][j]=0.0f;
    for(int k=0;k<K;k++){
        half w[MR], x[NR];
        for(int i=0;i<MR;i++) w[i]=W[(size_t)(row0+i)*K+k];
        for(int j=0;j<NR;j++) x[j]=X[(size_t)(t0+j)*K+k];
        for(int i=0;i<MR;i++)for(int j=0;j<NR;j++) acc[i][j]+=(float)(w[i]*x[j]);
    }
    for(int i=0;i<MR;i++)for(int j=0;j<NR;j++) if(row0+i<rows&&t0+j<toks) Y[(size_t)(t0+j)*rows+row0+i]=acc[i][j];
}
