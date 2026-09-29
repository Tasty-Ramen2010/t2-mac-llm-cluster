// Multi-query attention on the GPU for one query token (decoding), flash-decoding style.
// One work-group = one chunk of CH cached rows, all H heads that share them. Writes per-chunk partials
// (max M, sum S, unnormalized V sum) for every head; the host merges the chunks.
// K and V rows are f16 in the model's KV cache (read in place, zero-copy); V may be the first DV values of K (MLA).
#ifndef H
#define H 8
#endif
#ifndef DK
#define DK 576
#endif
#ifndef DV
#define DV 512
#endif
#define CH 256
#define WG 64
__attribute__((reqd_work_group_size(WG, 1, 1))) __attribute__((intel_reqd_sub_group_size(8)))
__kernel void attn_chunk(__global const half * K, ulong k_stride, __global const half * V, ulong v_stride,
                         __global const float * Q, ulong q_stride, __global const half * mask, float scale, int n_kv,
                         __global float * part, int ch) {
    __local float P[H][CH];
    __local float Ml[H], Sl[H];
    __local float Qs[H * DK];                      // the query vectors, loaded once, broadcast to all lanes
    const int c = get_group_id(0), lid = get_local_id(0), r0 = c * ch;
    for (int i = lid; i < H * DK; i += WG) Qs[i] = Q[(i / DK) * q_stride + (i % DK)];
    barrier(CLK_LOCAL_MEM_FENCE);
    // 1) scores: each lane takes CH/WG rows, all heads per row (the row is read and converted once)
    for (int j = 0; j < ch / WG; j++) {
        const int rl = j * WG + lid, r = r0 + rl;
        float acc[H];
        for (int h = 0; h < H; h++) acc[h] = 0.0f;
        if (r < n_kv) {
            __global const half * kr = K + (size_t) r * k_stride;
            for (int d = 0; d < DK; d += 8) {
                const float8 kv = vload_half8(0, kr + d);
                for (int h = 0; h < H; h++) {
                    const float8 qv = vload8(0, Qs + h * DK + d);
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
    // 2) softmax statistics per head over this chunk
    if (lid < H) {
        float m = -INFINITY;
        for (int i = 0; i < ch; i++) m = fmax(m, P[lid][i]);
        Ml[lid] = m;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int j = 0; j < ch / WG; j++) {
        const int rl = j * WG + lid;
        for (int h = 0; h < H; h++) { const float s = P[h][rl]; P[h][rl] = (s == -INFINITY || Ml[h] == -INFINITY) ? 0.0f : exp(s - Ml[h]); }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < H) { float s = 0.0f; for (int i = 0; i < ch; i++) s += P[lid][i]; Sl[lid] = s; }
    // 3) V: each lane owns DV/WG output dims for all heads; rows are read once (coalesced across lanes)
    float8 o[H];
    for (int h = 0; h < H; h++) o[h] = (float8) 0.0f;
    const int nr = min(ch, n_kv - r0);
    for (int rr = 0; rr < nr; rr++) {
        const float8 vv = vload_half8(0, V + (size_t) (r0 + rr) * v_stride + lid * 8);
        for (int h = 0; h < H; h++) o[h] += P[h][rr] * vv;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __global float * pc = part + (size_t) c * H * (2 + DV);
    for (int h = 0; h < H; h++) {
        if (lid == 0) { pc[h * (2 + DV)] = Ml[h]; pc[h * (2 + DV) + 1] = Sl[h]; }
        vstore8(o[h], 0, pc + h * (2 + DV) + 2 + lid * 8);
    }
}
