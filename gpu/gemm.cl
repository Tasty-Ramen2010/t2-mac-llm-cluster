#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// GEMM for block_q4_0x8 weights (prompt processing): each 16-lane subgroup computes 8 rows x TT tokens.
// Weights are decoded once per (block,row) and reused across all TT tokens in the tile -> arithmetic-bound, which is
// where the GPU beats the CPU. 144 bytes per 8 rows x 32 cols: 8 fp16 scales, then row r byte j at 16 + (j<8? r*8+j : 64+r*8+(j-8)),
// low nibble = element j, high nibble = element j+16 (nibbles are two's-complement, offset by 8 -> subtract 8).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemm_q4x8(__global const uchar* W, __global const float* X, __global float* Y,
                        int nb, int rows, int toks, int TT) {
    const int lane = get_sub_group_local_id();
    const int g = get_group_id(0);              // which group of 8 rows
    const int t0 = get_group_id(1) * TT;        // first token of this tile
    float acc[8][8];                            // [row][token], TT<=8
    for (int r=0;r<8;r++) for (int t=0;t<TT;t++) acc[r][t]=0.0f;
    for (int b = lane; b < nb; b += 16) {
        __global const uchar* blk = W + ((size_t)g*nb + b)*144;
        const float8 d = vload_half8(0, (__global const half*)blk);
        const uchar16 qa=vload16(0,blk+16),qb=vload16(1,blk+16),qc=vload16(2,blk+16),qd=vload16(3,blk+16);
        const uchar16 qe=vload16(4,blk+16),qf=vload16(5,blk+16),qg=vload16(6,blk+16),qh=vload16(7,blk+16);
        // decode all 8 rows: v[r] = 32 signed weights (lo j, hi j+16)
        float wlo[8][8], whi[8][8];             // [row][8 lanes-of-vector] low(0..7)/high(16..23)? -> use vectors
        #define DEC(r, A) { const uchar8 u = (r&1)?A.hi:A.lo; \
            const float8 lo = convert_float8(as_char8(u<<(uchar8)4)>>(char8)4); \
            const float8 hi = convert_float8(as_char8(u)>>(char8)4); \
            vstore8(lo,0,wlo[r]); vstore8(hi,0,whi[r]); }
        DEC(0,qa) DEC(1,qa) DEC(2,qb) DEC(3,qb) DEC(4,qc) DEC(5,qc) DEC(6,qd) DEC(7,qd)
        float8 wl2[8], wh2[8];
        #define DEC2(r, A) { const uchar8 u=(r&1)?A.hi:A.lo; \
            wl2[r]=convert_float8(as_char8(u<<(uchar8)4)>>(char8)4); wh2[r]=convert_float8(as_char8(u)>>(char8)4); }
        DEC2(0,qe) DEC2(1,qe) DEC2(2,qf) DEC2(3,qf) DEC2(4,qg) DEC2(5,qg) DEC2(6,qh) DEC2(7,qh)
        for (int t=0;t<TT;t++){
            __global const float* xb = X + (size_t)(t0+t)*K + b*32;
            const float8 x0=vload8(0,xb),x1=vload8(1,xb),x2=vload8(2,xb),x3=vload8(3,xb);
            for (int r=0;r<8;r++){
                const float8 p = vload8(0,wlo[r])*x0 + wl2[r]*x1 + vload8(0,whi[r])*x2 + wh2[r]*x3;
                acc[r][t] += d[r]*(p.s0+p.s1+p.s2+p.s3+p.s4+p.s5+p.s6+p.s7);
            }
        }
    }
    for (int r=0;r<8;r++) for (int t=0;t<TT;t++){
        const float v = sub_group_reduce_add(acc[r][t]);
        if (lane==0 && t0+t<toks) Y[(size_t)(t0+t)*rows + g*8 + r] = v;
    }
}
