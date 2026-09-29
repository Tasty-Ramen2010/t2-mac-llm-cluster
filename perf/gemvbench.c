// gemvbench: why do llama.cpp's gemv kernels only reach ~21 GB/s when plain streaming reads reach ~31?
// Simulates one token of gpt-oss TP expert work: many 1440x2880 MXFP4 matrices (one per expert per layer),
// 4 threads, barrier after every matrix. Variants: pure read, MXFP4 dot, MXFP4 dot + software prefetch.
// build: gcc -O3 -mavx2 -mfma -fopenmp gemvbench.c -o gemvbench
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>

#define QK 32
#define BLK 17                       // e8m0 scale + 16 bytes of 4-bit codes
static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

static int ROWS = 1440, COLS = 2880, PF = 0;

static inline float e8m0_half(uint8_t e) { uint32_t b = e < 2 ? (0x00200000u << e) : ((uint32_t)(e - 1) << 23); float f; memcpy(&f, &b, 4); return f; }

// y[r] = W[r] . x   (x pre-quantized to int8 blocks xq with scales xd)
static void gemv_rows(const uint8_t *W, const int8_t *xq, const float *xd, float *y, int r0, int r1) {
    const int nb = COLS / QK;
    const __m128i lut = _mm_loadu_si128((const __m128i *)kv), m4 = _mm_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; r++) {
        const uint8_t *row = W + (size_t)r * nb * BLK;
        if (PF) _mm_prefetch((const char *)row + PF, _MM_HINT_T0), _mm_prefetch((const char *)row + PF + 64, _MM_HINT_T0);
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        for (int b = 0; b < nb; b += 2) {
            if (PF && (b & 7) == 0) _mm_prefetch((const char *)row + b * BLK + PF + 128, _MM_HINT_T0);
            for (int k = 0; k < 2; k++) {
                const uint8_t *blk = row + (b + k) * BLK;
                const __m128i q = _mm_loadu_si128((const __m128i *)(blk + 1));
                const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, m4));
                const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
                const __m256i w = _mm256_set_m128i(hi, lo);
                const __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + (b + k) * QK));
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(xv, w)), ones);
                const __m256 d = _mm256_set1_ps(e8m0_half(blk[0]) * xd[b + k]);
                if (k == 0) acc0 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p), acc0);
                else        acc1 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p), acc1);
            }
        }
        __m256 a = _mm256_add_ps(acc0, acc1);
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        y[r] = _mm_cvtss_f32(s);
    }
}


// ---- lean kernel: 256-bit broadcast + one pshufb, unsigned LUT (w+12) so no sign ops, table scales ----
static const int8_t kvu[32] = {12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0, 12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0};
static float e8tab[256];
// per-token activation prep: dx[b] = x scale, cb[b] = 12*dx*sum(xq in block)/8 (offset correction, per lane)
static float *dx, *cb;
static void prep_x(const int8_t *xq, const float *xd) {
    for (int b = 0; b < COLS / QK; b++) { int s = 0; for (int i = 0; i < QK; i++) s += xq[b * QK + i]; dx[b] = xd[b]; cb[b] = 12.0f * xd[b] * s / 8.0f; }
}
static inline __m256i nib(const uint8_t *qs, __m256i lut, __m256i sh, __m256i m4) {
    const __m256i q = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)qs));
    return _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srlv_epi64(q, sh), m4));
}
static void gemv2_rows(const uint8_t *W, const int8_t *xq, float *y, int r0, int r1, int two) {
    const int nb = COLS / QK;
    const __m256i lut = _mm256_loadu_si256((const __m256i *)kvu), m4 = _mm256_set1_epi8(0x0F), sh = _mm256_set_epi64x(4, 4, 0, 0);
    const __m256i ones = _mm256_set1_epi16(1);
    int r = r0;
    if (two) for (; r + 1 < r1; r += 2) {
        const uint8_t *ra = W + (size_t)r * nb * BLK, *rb2 = ra + nb * BLK;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, b0 = a0, b1 = a0;
        for (int b = 0; b < nb; b += 2) {
            for (int k = 0; k < 2; k++) {
                const __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + (b + k) * QK));
                const __m256 vdx = _mm256_broadcast_ss(dx + b + k), vcb = _mm256_broadcast_ss(cb + b + k);
                const uint8_t *ba = ra + (b + k) * BLK, *bb = rb2 + (b + k) * BLK;
                const __m256i pa = _mm256_madd_epi16(_mm256_maddubs_epi16(nib(ba + 1, lut, sh, m4), xv), ones);
                const __m256i pb = _mm256_madd_epi16(_mm256_maddubs_epi16(nib(bb + 1, lut, sh, m4), xv), ones);
                const __m256 fa = _mm256_fmsub_ps(_mm256_cvtepi32_ps(pa), vdx, vcb), fb = _mm256_fmsub_ps(_mm256_cvtepi32_ps(pb), vdx, vcb);
                if (k == 0) { a0 = _mm256_fmadd_ps(_mm256_broadcast_ss(e8tab + ba[0]), fa, a0); b0 = _mm256_fmadd_ps(_mm256_broadcast_ss(e8tab + bb[0]), fb, b0); }
                else        { a1 = _mm256_fmadd_ps(_mm256_broadcast_ss(e8tab + ba[0]), fa, a1); b1 = _mm256_fmadd_ps(_mm256_broadcast_ss(e8tab + bb[0]), fb, b1); }
            }
        }
        __m256 a = _mm256_add_ps(a0, a1), c = _mm256_add_ps(b0, b1);
        __m256 h = _mm256_hadd_ps(a, c); h = _mm256_hadd_ps(h, h);
        __m128 t = _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps(h, 1));
        y[r] = _mm_cvtss_f32(t); y[r + 1] = _mm_cvtss_f32(_mm_shuffle_ps(t, t, 1));
    }
    for (; r < r1; r++) {
        const uint8_t *row = W + (size_t)r * nb * BLK;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0;
        for (int b = 0; b < nb; b += 2) {
            for (int k = 0; k < 2; k++) {
                const uint8_t *blk = row + (b + k) * BLK;
                const __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + (b + k) * QK));
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(nib(blk + 1, lut, sh, m4), xv), ones);
                const __m256 f = _mm256_fmsub_ps(_mm256_cvtepi32_ps(p), _mm256_broadcast_ss(dx + b + k), _mm256_broadcast_ss(cb + b + k));
                if (k == 0) a0 = _mm256_fmadd_ps(_mm256_broadcast_ss(e8tab + blk[0]), f, a0);
                else        a1 = _mm256_fmadd_ps(_mm256_broadcast_ss(e8tab + blk[0]), f, a1);
            }
        }
        __m256 a = _mm256_add_ps(a0, a1);
        __m128 t = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t);
        y[r] = _mm_cvtss_f32(t);
    }
}

static __m256i read_rows(const uint8_t *W, int r0, int r1) {
    const size_t rb = (size_t)COLS / QK * BLK;
    const __m256i *p = (const __m256i *)(W + r0 * rb), *e = (const __m256i *)(W + r1 * rb);
    __m256i s = _mm256_setzero_si256();
    for (; p + 1 < e; p += 2) s = _mm256_xor_si256(s, _mm256_xor_si256(_mm256_loadu_si256(p), _mm256_loadu_si256(p + 1)));
    return s;
}

int main(int argc, char **argv) {
    int mode = argc > 1 ? atoi(argv[1]) : 1;     // 0 read, 1 mxfp4 gemv
    ROWS = argc > 2 ? atoi(argv[2]) : 1440;      // rows per matrix
    PF = argc > 3 ? atoi(argv[3]) : 0;           // prefetch distance in bytes (0 = off)
    int nthr = argc > 4 ? atoi(argv[4]) : 4;
    const size_t rb = (size_t)COLS / QK * BLK, mat = (size_t)ROWS * rb;
    const size_t total = (size_t)(getenv("TOTALMB") ? atoi(getenv("TOTALMB")) : 768) << 20;
    const int nmat = total / mat;
    uint8_t *W = aligned_alloc(2 << 20, (size_t)nmat * mat + 4096);
    madvise(W, (size_t)nmat * mat, MADV_HUGEPAGE);
    for (size_t i = 0; i < (size_t)nmat * mat; i++) W[i] = (uint8_t)(i * 2654435761u >> 13);
    for (size_t i = 0; i < (size_t)nmat * mat; i += BLK) W[i] = 120 + (i & 7);   // sane exponents
    int8_t *xq = aligned_alloc(64, COLS); float *xd = malloc(COLS / QK * 4), *y = malloc(ROWS * 4);
    for (int i = 0; i < COLS; i++) xq[i] = (int8_t)(i * 7 % 255 - 127);
    for (int i = 0; i < COLS / QK; i++) xd[i] = 0.01f;
    for (int e = 0; e < 256; e++) e8tab[e] = e8m0_half(e);
    dx = malloc(COLS / QK * 4); cb = malloc(COLS / QK * 4); prep_x(xq, xd);
    {   // correctness: v1 vs v2 vs v2x2 on the first matrix
        float *y1 = malloc(ROWS * 4), *y2 = malloc(ROWS * 4), *y3 = malloc(ROWS * 4);
        gemv_rows(W, xq, xd, y1, 0, ROWS); gemv2_rows(W, xq, y2, 0, ROWS, 0); gemv2_rows(W, xq, y3, 0, ROWS, 1);
        double e2 = 0, e3 = 0, m = 0;
        for (int r = 0; r < ROWS; r++) { e2 = fmax(e2, fabs(y2[r] - y1[r])); e3 = fmax(e3, fabs(y3[r] - y1[r])); m = fmax(m, fabs(y1[r])); }
        if (mode >= 2) printf("check: max|y| %.3g, max err v2 %.3g, v2x2 %.3g\n", m, e2, e3);
    }
    omp_set_num_threads(nthr);
    double best = 1e9; volatile int sink = 0;
    const int REPS = getenv("REPS") ? atoi(getenv("REPS")) : 5; double tall = omp_get_wtime();
    for (int rep = 0; rep < REPS; rep++) {
        double t0 = omp_get_wtime();
        #pragma omp parallel
        {
            const int t = omp_get_thread_num(), n = omp_get_num_threads();
            const int r0 = ROWS * t / n, r1 = ROWS * (t + 1) / n;
            __m256i s = _mm256_setzero_si256();
            for (int m = 0; m < nmat; m++) {
                const uint8_t *Wm = W + (size_t)m * mat;
                if (mode == 0) s = _mm256_xor_si256(s, read_rows(Wm, r0, r1));
                else if (mode == 1) gemv_rows(Wm, xq, xd, y, r0, r1);
                else gemv2_rows(Wm, xq, y, r0, r1, mode == 3);
                #pragma omp barrier
            }
            if (t == 0) sink += _mm256_extract_epi32(s, 0);
        }
        double dt = omp_get_wtime() - t0;
        if (dt < best) best = dt;
    }
    if (getenv("REPS")) printf("CPU sustained over %.1fs: %.1f GB/s\n", omp_get_wtime() - tall, (double)REPS * nmat * mat / (omp_get_wtime() - tall) / 1e9);
    printf("mode %s rows %d pf %d thr %d: %d matrices of %.2f MB, %.1f GB/s, %.1f us per matrix\n", mode == 0 ? "read " : mode == 1 ? "mxfp4" : mode == 2 ? "lean1" : "lean2",
           ROWS, PF, nthr, nmat, mat / 1e6, (double)nmat * mat / best / 1e9, best / nmat * 1e6);
    return 0;
}
