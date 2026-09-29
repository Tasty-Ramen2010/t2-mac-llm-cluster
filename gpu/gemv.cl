
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// block_q4_0x8: 8 fp16 scales, then 128 bytes: row r's bytes 0-7 at r*8, bytes 8-15 at 64+r*8 (nibbles xor 8, i.e.
// two's complement 4-bit). Byte j of a row holds element j (low nibble) and j+16 (high nibble).
// One 16-lane subgroup per group of 8 rows, lanes stride over blocks, x staged in local memory.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_q4x8(__global const uchar * Wb, ulong woff, __global const uchar * Xb, ulong xoff,
                        __global uchar * Yb, ulong yoff, int nb, __local float * xs) {
    __global const uchar * W = Wb + woff;
    __global const float * x = (__global const float *)(Xb + xoff);
    __global float * y = (__global float *)(Yb + yoff);
    const int K = nb * 32;
    for (int i = get_local_id(0); i < K; i += get_local_size(0)) xs[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    const int lane = get_sub_group_local_id();
    const int g = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int b = lane; b < nb; b += 16) {
        __global const uchar * blk = W + ((size_t)g * nb + b) * 144;
        const float8 d = vload_half8(0, (__global const half *)blk);
        __local const float * xb = xs + b * 32;
        const float8 x0 = vload8(0, xb), x1 = vload8(1, xb), x2 = vload8(2, xb), x3 = vload8(3, xb);
        const uchar16 qa = vload16(0, blk + 16), qb = vload16(1, blk + 16), qc = vload16(2, blk + 16), qd = vload16(3, blk + 16);
        const uchar16 qe = vload16(4, blk + 16), qf = vload16(5, blk + 16), qg = vload16(6, blk + 16), qh = vload16(7, blk + 16);
#define ROW(r, A, B) { \
            const uchar8 u = (r & 1) ? A.hi : A.lo, v = (r & 1) ? B.hi : B.lo; \
            const float8 l0 = convert_float8(as_char8(u << (uchar8)4) >> (char8)4), h0 = convert_float8(as_char8(u) >> (char8)4); \
            const float8 l1 = convert_float8(as_char8(v << (uchar8)4) >> (char8)4), h1 = convert_float8(as_char8(v) >> (char8)4); \
            const float8 p = l0 * x0 + l1 * x1 + h0 * x2 + h1 * x3; \
            acc[r] += d[r] * (p.s0 + p.s1 + p.s2 + p.s3 + p.s4 + p.s5 + p.s6 + p.s7); }
        ROW(0, qa, qe) ROW(1, qa, qe) ROW(2, qb, qf) ROW(3, qb, qf) ROW(4, qc, qg) ROW(5, qc, qg) ROW(6, qd, qh) ROW(7, qd, qh)
    }
    for (int r = 0; r < 8; r++) { const float t = sub_group_reduce_add(acc[r]); if (lane == r) y[g * 8 + r] = t; }
}
