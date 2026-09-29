// Multi-query attention for one query token (decoding) -- v3.
// Work-group = one chunk of cached rows; 8 subgroups of 8 lanes; subgroup h = attention head h.
// Scores: the 8 lanes of a head read each K row together (coalesced 16-byte pieces), each lane keeps its slice of
// that head's query in registers, a subgroup reduction adds the slices. V: all 64 lanes split the DV dims, every
// row is read once (coalesced) and applied to all heads. Output: per-chunk partials (M, S, sum p*V) per head.
#define H 8
#define DK 576
#define DV 512
#define WG 64
#define CHMAX 256
#define NT (DK / 64)
__attribute__((reqd_work_group_size(WG, 1, 1))) __attribute__((intel_reqd_sub_group_size(8)))
__kernel void attn_chunk(__global const half * K, ulong k_stride, __global const half * V, ulong v_stride,
                         __global const float * Q, ulong q_stride, __global const half * mask, float scale, int n_kv,
                         __global float * part, int ch) {
    __local float P[H][CHMAX];
    __local float Ml[H], Sl[H];
    const int c = get_group_id(0), lid = get_local_id(0), r0 = c * ch;
    const int h = get_sub_group_id(), lane = get_sub_group_local_id();
    float8 qv[NT];
    for (int t = 0; t < NT; t++) qv[t] = vload8(0, Q + h * q_stride + t * 64 + lane * 8);
    const int nr = min(ch, n_kv - r0);
    for (int rr = 0; rr < nr; rr++) {
        __global const half * kr = K + (size_t) (r0 + rr) * k_stride + lane * 8;
        float acc = 0.0f;
        for (int t = 0; t < NT; t++) {
            const float8 kv = vload_half8(0, kr + t * 64);
            acc += dot(kv.lo, qv[t].lo) + dot(kv.hi, qv[t].hi);
        }
        acc = sub_group_reduce_add(acc);
        if (lane == 0) P[h][rr] = acc * scale + (mask ? vload_half(r0 + rr, mask) : 0.0f);
    }
    for (int rr = nr + lane; rr < ch; rr += 8) P[h][rr] = -INFINITY;
    // softmax statistics for this head (its own subgroup)
    float m = -INFINITY;
    for (int i = lane; i < nr; i += 8) m = fmax(m, P[h][i]);
    m = sub_group_reduce_max(m);
    float s = 0.0f;
    for (int i = lane; i < nr; i += 8) { const float e = m == -INFINITY ? 0.0f : exp(P[h][i] - m); P[h][i] = e; s += e; }
    s = sub_group_reduce_add(s);
    if (lane == 0) { Ml[h] = m; Sl[h] = s; }
    barrier(CLK_LOCAL_MEM_FENCE);
    // V for all heads: lane owns DV/WG dims
    float8 o[H];
    for (int hh = 0; hh < H; hh++) o[hh] = (float8) 0.0f;
    for (int rr = 0; rr < nr; rr++) {
        const float8 vv = vload_half8(0, V + (size_t) (r0 + rr) * v_stride + lid * 8);
        for (int hh = 0; hh < H; hh++) o[hh] += P[hh][rr] * vv;
    }
    __global float * pc = part + (size_t) c * H * (2 + DV);
    for (int hh = 0; hh < H; hh++) {
        if (lid == 0) { pc[hh * (2 + DV)] = Ml[hh]; pc[hh * (2 + DV) + 1] = Sl[hh]; }
        vstore8(o[hh], 0, pc + hh * (2 + DV) + 2 + lid * 8);
    }
}
