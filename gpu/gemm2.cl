#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// v2: 4 rows per subgroup (2 subgroups per 8-row block), TT tokens. Fewer accumulators -> less register spilling.
// Activations for the tile staged in local memory once and reused by both subgroups.
#ifndef TTOK
#define TTOK 8
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemm_q4x8(__global const uchar* W, __global const float* X, __global float* Y,
                        int nb, int rows, int toks, int TT) {
    const int lane = get_sub_group_local_id();
    const int sg = get_sub_group_id();          // 0 or 1 -> rows 0..3 or 4..7
    const int g = get_group_id(0);
    const int t0 = get_group_id(1) * TTOK;
    __local float xs[TTOK][K];
    for (int i = get_local_id(0); i < TTOK*K; i += get_local_size(0)) { int t=i/K, c=i%K; xs[t][c] = (t0+t<toks)? X[(size_t)(t0+t)*K + c] : 0.0f; }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int rbase = sg*4;
    float acc[4][TTOK];
    for (int r=0;r<4;r++) for (int t=0;t<TTOK;t++) acc[r][t]=0.0f;
    for (int b = lane; b < nb; b += 16) {
        __global const uchar* blk = W + ((size_t)g*nb + b)*144;
        const float8 d = vload_half8(0, (__global const half*)blk);
        uchar16 Q[4];
        Q[0]=vload16(rbase+0<4?0:4,blk+16); // placeholder; fill below
        // load the two source bytes-vectors this subgroup needs
        const uchar16 v0=vload16(rbase/2+0,blk+16), v1=vload16(rbase/2+1,blk+16);
        const uchar16 v2=vload16(4+rbase/2+0,blk+16), v3=vload16(4+rbase/2+1,blk+16);
        float8 wl[4],wh[4],wl2[4],wh2[4];
        #define D(r,A,B){const uchar8 u=(r&1)?A.hi:A.lo,w=(r&1)?B.hi:B.lo; \
          wl[r]=convert_float8(as_char8(u<<(uchar8)4)>>(char8)4); wh[r]=convert_float8(as_char8(u)>>(char8)4); \
          wl2[r]=convert_float8(as_char8(w<<(uchar8)4)>>(char8)4); wh2[r]=convert_float8(as_char8(w)>>(char8)4);}
        D(0,v0,v2) D(1,v0,v2) D(2,v1,v3) D(3,v1,v3)
        for (int t=0;t<TTOK;t++){
            __local const float* xb = &xs[t][b*32];
            const float8 x0=vload8(0,xb),x1=vload8(1,xb),x2=vload8(2,xb),x3=vload8(3,xb);
            for (int r=0;r<4;r++){
                const float8 p = wl[r]*x0 + wl2[r]*x1 + wh[r]*x2 + wh2[r]*x3;
                acc[r][t] += d[rbase+r]*(p.s0+p.s1+p.s2+p.s3+p.s4+p.s5+p.s6+p.s7);
            }
        }
    }
    for (int r=0;r<4;r++) for (int t=0;t<TTOK;t++){
        const float v = sub_group_reduce_add(acc[r][t]);
        if (lane==0 && t0+t<toks) Y[(size_t)(t0+t)*rows + g*8 + rbase + r] = v;
    }
}
