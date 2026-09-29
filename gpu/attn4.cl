// Multi-query attention for one query token (decoding) -- v4: block reads, 4 rows in flight per head.
#define H 8
#define DK 576
#define DV 512
#define WG 64
#define CHMAX 256
#define NT (DK / 64)
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
inline float8 kpiece(__global const half * row, int t) {
    return convert_float8(as_half8(intel_sub_group_block_read_us8((__global const ushort *) (row + t * 64))));
}
__attribute__((reqd_work_group_size(WG, 1, 1))) __attribute__((intel_reqd_sub_group_size(8)))
__kernel void attn_chunk(__global const half * K, ulong k_stride, __global const half * V, ulong v_stride,
                         __global const float * Q, ulong q_stride, __global const half * mask, float scale, int n_kv,
                         __global float * part, int ch) {
    __local float P[H][CHMAX];
    __local float Ml[H], Sl[H];
    const int c = get_group_id(0), lid = get_local_id(0), r0 = c * ch;
    const int h = get_sub_group_id(), lane = get_sub_group_local_id();
    // query slice in block-read order: lane l, element i of piece t  <->  dim t*64 + i*8 + l
    float8 qv[NT];
    for (int t = 0; t < NT; t++) {
        float8 x;
        x.s0 = Q[h * q_stride + t * 64 + 0 * 8 + lane]; x.s1 = Q[h * q_stride + t * 64 + 1 * 8 + lane];
        x.s2 = Q[h * q_stride + t * 64 + 2 * 8 + lane]; x.s3 = Q[h * q_stride + t * 64 + 3 * 8 + lane];
        x.s4 = Q[h * q_stride + t * 64 + 4 * 8 + lane]; x.s5 = Q[h * q_stride + t * 64 + 5 * 8 + lane];
        x.s6 = Q[h * q_stride + t * 64 + 6 * 8 + lane]; x.s7 = Q[h * q_stride + t * 64 + 7 * 8 + lane];
        qv[t] = x;
    }
    const int nr = min(ch, n_kv - r0);
    int rr = 0;
    for (; rr + 4 <= nr; rr += 4) {
        __global const half * k0 = K + (size_t) (r0 + rr) * k_stride;
        float8 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int t = 0; t < NT; t++) {
            a0 = fma(kpiece(k0, t), qv[t], a0);
            a1 = fma(kpiece(k0 + k_stride, t), qv[t], a1);
            a2 = fma(kpiece(k0 + 2 * k_stride, t), qv[t], a2);
            a3 = fma(kpiece(k0 + 3 * k_stride, t), qv[t], a3);
        }
        float s0 = sub_group_reduce_add(a0.s0 + a0.s1 + a0.s2 + a0.s3 + a0.s4 + a0.s5 + a0.s6 + a0.s7);
        float s1 = sub_group_reduce_add(a1.s0 + a1.s1 + a1.s2 + a1.s3 + a1.s4 + a1.s5 + a1.s6 + a1.s7);
        float s2 = sub_group_reduce_add(a2.s0 + a2.s1 + a2.s2 + a2.s3 + a2.s4 + a2.s5 + a2.s6 + a2.s7);
        float s3 = sub_group_reduce_add(a3.s0 + a3.s1 + a3.s2 + a3.s3 + a3.s4 + a3.s5 + a3.s6 + a3.s7);
        if (lane == 0) {
            const int g = r0 + rr;
            P[h][rr]     = s0 * scale + (mask ? vload_half(g, mask) : 0.0f);
            P[h][rr + 1] = s1 * scale + (mask ? vload_half(g + 1, mask) : 0.0f);
            P[h][rr + 2] = s2 * scale + (mask ? vload_half(g + 2, mask) : 0.0f);
            P[h][rr + 3] = s3 * scale + (mask ? vload_half(g + 3, mask) : 0.0f);
        }
    }
    for (; rr < nr; rr++) {
        __global const half * k0 = K + (size_t) (r0 + rr) * k_stride;
        float8 a0 = 0.0f;
        for (int t = 0; t < NT; t++) a0 = fma(kpiece(k0, t), qv[t], a0);
        const float s0 = sub_group_reduce_add(a0.s0 + a0.s1 + a0.s2 + a0.s3 + a0.s4 + a0.s5 + a0.s6 + a0.s7);
        if (lane == 0) P[h][rr] = s0 * scale + (mask ? vload_half(r0 + rr, mask) : 0.0f);
    }
    float m = -INFINITY;
    for (int i = lane; i < nr; i += 8) m = fmax(m, P[h][i]);
    m = sub_group_reduce_max(m);
    float s = 0.0f;
    for (int i = lane; i < nr; i += 8) { const float e = m == -INFINITY ? 0.0f : exp(P[h][i] - m); P[h][i] = e; s += e; }
    s = sub_group_reduce_add(s);
    if (lane == 0) { Ml[h] = m; Sl[h] = s; }
    barrier(CLK_LOCAL_MEM_FENCE);
    float8 o[H];
    for (int hh = 0; hh < H; hh++) o[hh] = (float8) 0.0f;
    for (int r = 0; r < nr; r++) {
        const float8 vv = vload_half8(0, V + (size_t) (r0 + r) * v_stride + lid * 8);
        for (int hh = 0; hh < H; hh++) o[hh] = fma((float8) P[hh][r], vv, o[hh]);
    }
    __global float * pc = part + (size_t) c * H * (2 + DV);
    for (int hh = 0; hh < H; hh++) {
        if (lid == 0) { pc[hh * (2 + DV)] = Ml[hh]; pc[hh * (2 + DV) + 1] = Sl[hh]; }
        vstore8(o[hh], 0, pc + hh * (2 + DV) + 2 + lid * 8);
    }
}
