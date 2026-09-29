
#ifndef WG
#define WG 64
#endif
#define H 8
#define DK 576
#define DV 512
#define CHMAX 256
__attribute__((reqd_work_group_size(WG, 1, 1))) __attribute__((intel_reqd_sub_group_size(8)))
__kernel void attn_chunk(__global const half * K, ulong k_stride, __global const half * V, ulong v_stride,
                         __global const float * Q, ulong q_stride, __global const half * mask, float scale, int n_kv,
                         __global float * part, int ch) {
    __local float P[H][CHMAX];
    __local float Ml[H], Sl[H];
    const int c = get_group_id(0), lid = get_local_id(0), r0 = c * ch;
    for (int j = 0; j < ch / WG; j++) {
        const int rl = j * WG + lid, r = r0 + rl;
        float acc[H];
        for (int h = 0; h < H; h++) acc[h] = 0.0f;
        if (r < n_kv) {
            __global const half * kr = K + (size_t) r * k_stride;
            for (int d = 0; d < DK; d += 8) {
                const float8 kv = vload_half8(0, kr + d);
                for (int h = 0; h < H; h++) {
                    const float8 qv = vload8(0, Q + h * q_stride + d);
                    acc[h] += dot(kv.lo, qv.lo) + dot(kv.hi, qv.hi);
                }
            }
            const float mv = mask ? vload_half(r, mask) : 0.0f;
            for (int h = 0; h < H; h++) P[h][rl] = acc[h] * scale + mv;
        } else {
            for (int h = 0; h < H; h++) P[h][rl] = -INFINITY;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < H) {
        float m = -INFINITY;
        for (int i = 0; i < ch; i++) m = fmax(m, P[lid][i]);
        float s = 0.0f;
        for (int i = 0; i < ch; i++) { const float e = m == -INFINITY ? 0.0f : exp(P[lid][i] - m); P[lid][i] = e; s += e; }
        Ml[lid] = m; Sl[lid] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int nr = min(ch, n_kv - r0);
    for (int dd = lid * 8; dd < DV; dd += WG * 8) {
        float8 o[H];
        for (int h = 0; h < H; h++) o[h] = (float8) 0.0f;
        for (int rr = 0; rr < nr; rr++) {
            const float8 vv = vload_half8(0, V + (size_t) (r0 + rr) * v_stride + dd);
            for (int h = 0; h < H; h++) o[h] = fma((float8) P[h][rr], vv, o[h]);
        }
        __global float * pc = part + (size_t) c * H * (2 + DV);
        for (int h = 0; h < H; h++) vstore8(o[h], 0, pc + h * (2 + DV) + 2 + dd);
    }
    if (lid < H) { __global float * pc = part + (size_t) c * H * (2 + DV); pc[lid * (2 + DV)] = Ml[lid]; pc[lid * (2 + DV) + 1] = Sl[lid]; }
}
