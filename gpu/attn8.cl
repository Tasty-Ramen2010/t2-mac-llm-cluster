// Multi-query attention for one query token (decoding) -- v6: subgroup broadcasts instead of repeated loads.
#define H 8
#define DK 576
#define DV 512
#define WG 64
#define CHMAX 256
__attribute__((reqd_work_group_size(WG, 1, 1))) __attribute__((intel_reqd_sub_group_size(8)))
__kernel void attn_chunk(__global const half * K, ulong k_stride, __global const half * V, ulong v_stride,
                         __global const float * Q, ulong q_stride, __global const half * mask, float scale, int n_kv,
                         __global float * part, int ch) {
    __local float P[H][CHMAX];
    __local float Ml[H], Sl[H];
    const int c = get_group_id(0), lid = get_local_id(0), r0 = c * ch;
    const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
    // 1) scores as a small matrix multiply: a subgroup takes 4 rows at a time; lane l reads dims [t*64 + 8l, +8) of each
    //    row (adjacent lanes -> adjacent 16 bytes, coalesced) and keeps 4 rows x 8 heads of partial sums in registers;
    //    each query load is reused for the 4 rows; the 8 lanes' partials are added once per row at the end.
    for (int g = sg * 4; g < ch; g += WG / 2) {
        float acc[4][H];
        for (int rr = 0; rr < 4; rr++) for (int h = 0; h < H; h++) acc[rr][h] = 0.0f;
        for (int t = 0; t < DK / 64; t++) {
            float8 q[H];
            for (int h = 0; h < H; h++) q[h] = vload8(0, Q + h * q_stride + t * 64 + lane * 8);
            for (int rr = 0; rr < 4; rr++) {
                const int r = min(r0 + g + rr, n_kv - 1);
                const float8 kv = convert_float8(as_half8(vload4(0, (__global const uint *) (K + (size_t) r * k_stride + t * 64 + lane * 8))));
                for (int h = 0; h < H; h++) acc[rr][h] += dot(kv.lo, q[h].lo) + dot(kv.hi, q[h].hi);
            }
        }
        for (int rr = 0; rr < 4; rr++) {
            const int r = r0 + g + rr;
            for (int h = 0; h < H; h++) {
                const float sc = sub_group_reduce_add(acc[rr][h]);
                if (lane == 0) P[h][g + rr] = (r < n_kv) ? sc * scale + (mask ? vload_half(r, mask) : 0.0f) : -INFINITY;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // 2) softmax statistics: subgroup sg handles head sg
    {
        const int h = sg;
        float m = -INFINITY;
        for (int i = lane; i < ch; i += 8) m = fmax(m, P[h][i]);
        m = sub_group_reduce_max(m);
        float s = 0.0f;
        for (int i = lane; i < ch; i += 8) { const float e = m == -INFINITY ? 0.0f : exp(P[h][i] - m); P[h][i] = e; s += e; }
        s = sub_group_reduce_add(s);
        if (lane == 0) { Ml[h] = m; Sl[h] = s; }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // 3) V: lane owns 8 output dims; weights for 8 rows loaded at once, handed out by broadcast
    const int nr = min(ch, n_kv - r0);
    float8 o[H];
    for (int h = 0; h < H; h++) o[h] = (float8) 0.0f;
    for (int rr = 0; rr < nr; rr += 8) {
        float p[H];
        for (int h = 0; h < H; h++) p[h] = (rr + lane < nr) ? P[h][rr + lane] : 0.0f;
        const int nj = min(8, nr - rr);
        __global const half * vb = V + (size_t) (r0 + rr) * v_stride + lid * 8;
        if (0 < nj) { const float8 vv = convert_float8(as_half8(vload4(0, (__global const uint *) (vb + 0 * v_stride)))); o[0] = fma((float8) sub_group_broadcast(p[0], 0), vv, o[0]); o[1] = fma((float8) sub_group_broadcast(p[1], 0), vv, o[1]); o[2] = fma((float8) sub_group_broadcast(p[2], 0), vv, o[2]); o[3] = fma((float8) sub_group_broadcast(p[3], 0), vv, o[3]); o[4] = fma((float8) sub_group_broadcast(p[4], 0), vv, o[4]); o[5] = fma((float8) sub_group_broadcast(p[5], 0), vv, o[5]); o[6] = fma((float8) sub_group_broadcast(p[6], 0), vv, o[6]); o[7] = fma((float8) sub_group_broadcast(p[7], 0), vv, o[7]); }
        if (1 < nj) { const float8 vv = convert_float8(as_half8(vload4(0, (__global const uint *) (vb + 1 * v_stride)))); o[0] = fma((float8) sub_group_broadcast(p[0], 1), vv, o[0]); o[1] = fma((float8) sub_group_broadcast(p[1], 1), vv, o[1]); o[2] = fma((float8) sub_group_broadcast(p[2], 1), vv, o[2]); o[3] = fma((float8) sub_group_broadcast(p[3], 1), vv, o[3]); o[4] = fma((float8) sub_group_broadcast(p[4], 1), vv, o[4]); o[5] = fma((float8) sub_group_broadcast(p[5], 1), vv, o[5]); o[6] = fma((float8) sub_group_broadcast(p[6], 1), vv, o[6]); o[7] = fma((float8) sub_group_broadcast(p[7], 1), vv, o[7]); }
        if (2 < nj) { const float8 vv = convert_float8(as_half8(vload4(0, (__global const uint *) (vb + 2 * v_stride)))); o[0] = fma((float8) sub_group_broadcast(p[0], 2), vv, o[0]); o[1] = fma((float8) sub_group_broadcast(p[1], 2), vv, o[1]); o[2] = fma((float8) sub_group_broadcast(p[2], 2), vv, o[2]); o[3] = fma((float8) sub_group_broadcast(p[3], 2), vv, o[3]); o[4] = fma((float8) sub_group_broadcast(p[4], 2), vv, o[4]); o[5] = fma((float8) sub_group_broadcast(p[5], 2), vv, o[5]); o[6] = fma((float8) sub_group_broadcast(p[6], 2), vv, o[6]); o[7] = fma((float8) sub_group_broadcast(p[7], 2), vv, o[7]); }
        if (3 < nj) { const float8 vv = convert_float8(as_half8(vload4(0, (__global const uint *) (vb + 3 * v_stride)))); o[0] = fma((float8) sub_group_broadcast(p[0], 3), vv, o[0]); o[1] = fma((float8) sub_group_broadcast(p[1], 3), vv, o[1]); o[2] = fma((float8) sub_group_broadcast(p[2], 3), vv, o[2]); o[3] = fma((float8) sub_group_broadcast(p[3], 3), vv, o[3]); o[4] = fma((float8) sub_group_broadcast(p[4], 3), vv, o[4]); o[5] = fma((float8) sub_group_broadcast(p[5], 3), vv, o[5]); o[6] = fma((float8) sub_group_broadcast(p[6], 3), vv, o[6]); o[7] = fma((float8) sub_group_broadcast(p[7], 3), vv, o[7]); }
        if (4 < nj) { const float8 vv = convert_float8(as_half8(vload4(0, (__global const uint *) (vb + 4 * v_stride)))); o[0] = fma((float8) sub_group_broadcast(p[0], 4), vv, o[0]); o[1] = fma((float8) sub_group_broadcast(p[1], 4), vv, o[1]); o[2] = fma((float8) sub_group_broadcast(p[2], 4), vv, o[2]); o[3] = fma((float8) sub_group_broadcast(p[3], 4), vv, o[3]); o[4] = fma((float8) sub_group_broadcast(p[4], 4), vv, o[4]); o[5] = fma((float8) sub_group_broadcast(p[5], 4), vv, o[5]); o[6] = fma((float8) sub_group_broadcast(p[6], 4), vv, o[6]); o[7] = fma((float8) sub_group_broadcast(p[7], 4), vv, o[7]); }
        if (5 < nj) { const float8 vv = convert_float8(as_half8(vload4(0, (__global const uint *) (vb + 5 * v_stride)))); o[0] = fma((float8) sub_group_broadcast(p[0], 5), vv, o[0]); o[1] = fma((float8) sub_group_broadcast(p[1], 5), vv, o[1]); o[2] = fma((float8) sub_group_broadcast(p[2], 5), vv, o[2]); o[3] = fma((float8) sub_group_broadcast(p[3], 5), vv, o[3]); o[4] = fma((float8) sub_group_broadcast(p[4], 5), vv, o[4]); o[5] = fma((float8) sub_group_broadcast(p[5], 5), vv, o[5]); o[6] = fma((float8) sub_group_broadcast(p[6], 5), vv, o[6]); o[7] = fma((float8) sub_group_broadcast(p[7], 5), vv, o[7]); }
        if (6 < nj) { const float8 vv = convert_float8(as_half8(vload4(0, (__global const uint *) (vb + 6 * v_stride)))); o[0] = fma((float8) sub_group_broadcast(p[0], 6), vv, o[0]); o[1] = fma((float8) sub_group_broadcast(p[1], 6), vv, o[1]); o[2] = fma((float8) sub_group_broadcast(p[2], 6), vv, o[2]); o[3] = fma((float8) sub_group_broadcast(p[3], 6), vv, o[3]); o[4] = fma((float8) sub_group_broadcast(p[4], 6), vv, o[4]); o[5] = fma((float8) sub_group_broadcast(p[5], 6), vv, o[5]); o[6] = fma((float8) sub_group_broadcast(p[6], 6), vv, o[6]); o[7] = fma((float8) sub_group_broadcast(p[7], 6), vv, o[7]); }
        if (7 < nj) { const float8 vv = convert_float8(as_half8(vload4(0, (__global const uint *) (vb + 7 * v_stride)))); o[0] = fma((float8) sub_group_broadcast(p[0], 7), vv, o[0]); o[1] = fma((float8) sub_group_broadcast(p[1], 7), vv, o[1]); o[2] = fma((float8) sub_group_broadcast(p[2], 7), vv, o[2]); o[3] = fma((float8) sub_group_broadcast(p[3], 7), vv, o[3]); o[4] = fma((float8) sub_group_broadcast(p[4], 7), vv, o[4]); o[5] = fma((float8) sub_group_broadcast(p[5], 7), vv, o[5]); o[6] = fma((float8) sub_group_broadcast(p[6], 7), vv, o[6]); o[7] = fma((float8) sub_group_broadcast(p[7], 7), vv, o[7]); }
    }
    __global float * pc = part + (size_t) c * H * (2 + DV);
    for (int h = 0; h < H; h++) {
        if (lid == 0) { pc[h * (2 + DV)] = Ml[h]; pc[h * (2 + DV) + 1] = Sl[h]; }
        vstore8(o[h], 0, pc + h * (2 + DV) + 2 + lid * 8);
    }
}
