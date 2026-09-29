#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// Best-case dense fp16 GEMM ceiling: Y[tok,row] = sum_k W[row,k]*X[tok,k]. Each lane does 1 row x TT tokens,
// 8-wide fp16 dot per step. This is the friendliest possible shape (no int4 decode) -> upper bound for the iGPU.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void f16gemm(__global const half* W, __global const half* X, __global float* Y, int rows, int toks, int TT){
    const int row = get_global_id(0);
    const int t0 = get_group_id(1)*TT;
    if (row>=rows) return;
    float acc[8]; for(int t=0;t<TT;t++) acc[t]=0.0f;
    for (int k=0;k<K;k+=8){
        const half8 w = vload8(0, W + (size_t)row*K + k);
        for (int t=0;t<TT;t++){ const half8 x=vload8(0, X+(size_t)(t0+t)*K+k); const half8 p=w*x; acc[t]+=p.s0+p.s1+p.s2+p.s3+p.s4+p.s5+p.s6+p.s7; }
    }
    for (int t=0;t<TT;t++) if(t0+t<toks) Y[(size_t)(t0+t)*rows+row]=acc[t];
}
