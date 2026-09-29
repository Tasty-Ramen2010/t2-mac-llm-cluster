// kbench: speed of ggml's repacked GEMV kernels in isolation (GB/s of weights), from cache and from RAM, 1 and N threads
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
typedef void (*gemv_t)(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_q4_K_8x8_q8_K(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q4_0_8x8_q8_0(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q5_0_8x8_q8_0(int, float *, size_t, const void *, const void *, int, int);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
struct job { gemv_t f; int n, nc; const char * w; const void * a; float * out; size_t row_bytes; int reps; };
static void * run(void * p) { struct job * j = p; for (int r = 0; r < j->reps; r++) j->f(j->n, j->out, 0, j->w, j->a, 1, j->nc); return NULL; }
static void bench(const char * name, gemv_t f, double bytes_per_w, size_t act_bytes, int n, int nthr, int block_bytes, int hdr_bytes, int act_blk, int act_is_k) {
    const size_t rows_cache = 64, rows_ram = 1 << 15;          // 64 rows: ~70 KB (L2); 32768 rows: ~40 MB (RAM)
    for (int mode = 0; mode < 2; mode++) {
        size_t rows = mode ? rows_ram : rows_cache;
        size_t wbytes = (size_t) (rows * n * bytes_per_w);
        char * w = aligned_alloc(1 << 21, wbytes + (1 << 21)); madvise(w, wbytes, MADV_HUGEPAGE);
        for (size_t i = 0; i < wbytes; i++) w[i] = (char) (rand() & 0xff);
        // make every fp16 scale sane: set all 16-bit words that look like fp16 scales -> harmless for timing; use 0x3c00 pattern in first bytes
        // realistic fp16 scales (0.01) in every block header: random ones can be denormal and make the maths slow
        size_t blk = (size_t) (block_bytes), hdr = (size_t) hdr_bytes;
        for (size_t b = 0; b + blk <= wbytes; b += blk) for (size_t i = 0; i < hdr; i += 2) { w[b + i] = 0x1f; w[b + i + 1] = 0x21; }
        char * a = aligned_alloc(64, act_bytes * 4); memset(a, 1, act_bytes * 4);
        for (size_t b = 0; b + act_blk <= act_bytes; b += act_blk) {           // activation scales: 1.0
            if (act_is_k) { float one = 1.0f; memcpy(a + b, &one, 4); } else { a[b] = 0x00; a[b + 1] = 0x3c; }
        }
        float * out = aligned_alloc(64, rows * sizeof(float) * 2);
        int threads = mode ? nthr : 1;
        int reps = mode ? 6 : 3000;
        pthread_t th[8]; struct job jb[8];
        size_t per = rows / threads; per -= per % 8;
        size_t row_bytes = (size_t) (n * bytes_per_w);
        double t0 = now();
        for (int t = 0; t < threads; t++) {
            jb[t] = (struct job){ f, n, (int) per, w + t * per * row_bytes, a, out + t * per, row_bytes, reps };
            pthread_create(&th[t], NULL, run, &jb[t]);
        }
        for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
        double s = now() - t0;
        printf("%-8s %-5s %d thr: %6.1f GB/s  (%.1f us per 1 MB)\n", name, mode ? "RAM" : "cache", threads,
               (double) per * threads * row_bytes * reps / s / 1e9, s / (per * threads * row_bytes * reps / 1e6) * 1e6);
        free(w); free(a); free(out);
    }
}
int main(int argc, char ** argv) {
    int nthr = argc > 1 ? atoi(argv[1]) : 3, n = 2048;
    bench("q4_K", ggml_gemv_q4_K_8x8_q8_K, 144.0 / 256, 292 * (n / 256), n, nthr, 1152, 32, 292, 1);
    bench("q4_0", ggml_gemv_q4_0_8x8_q8_0, 18.0 / 32, 34 * (n / 32), n, nthr, 144, 16, 34, 0);
    bench("q5_0", ggml_gemv_q5_0_8x8_q8_0, 22.0 / 32, 34 * (n / 32), n, nthr, 176, 16, 34, 0);
    return 0;
}
