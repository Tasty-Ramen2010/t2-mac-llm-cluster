// cpugemm: ggml's repacked CPU GEMM (prompt processing) throughput, in multiply-adds per second
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
void ggml_gemm_q4_0_8x8_q8_0(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q4_K_8x8_q8_K(int, float *, size_t, const void *, const void *, int, int);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
typedef void (*gemm_t)(int, float *, size_t, const void *, const void *, int, int);
struct job { gemm_t f; int K, rows, toks; char * w; char * a; float * out; int reps; };
static void * run(void * p) { struct job * j = p; for (int r = 0; r < j->reps; r++) j->f(j->K, j->out, j->rows, j->w, j->a, j->toks, j->rows); return NULL; }
static void bench(const char * name, gemm_t f, double wbpw, int blk, int hdr, int K, int rows, int toks, int nthr, int act_blk4) {
    size_t wb = (size_t) (rows * (double) K * wbpw);
    char * w = aligned_alloc(4096, wb * nthr); memset(w, 0x33, wb * nthr);
    for (size_t b = 0; b + blk <= wb * nthr; b += blk) for (int i = 0; i < hdr; i += 2) { w[b + i] = 0x1f; w[b + i + 1] = 0x21; }
    size_t ablk = act_blk4 == 1 ? 4 * 34 : 16 + 1024 + 128;   // block_q8_0x4 / block_q8_Kx4
    size_t ab = (size_t) toks / 4 * (K / (act_blk4 == 1 ? 32 : 256)) * ablk;
    char * a = aligned_alloc(64, ab + 4096); memset(a, 1, ab + 4096);
    for (size_t b = 0; b + ablk <= ab; b += ablk) {            // sane scales (junk bytes are denormal floats: very slow)
        if (act_blk4 == 1) for (int i = 0; i < 4; i++) { a[b + 2 * i] = 0x00; a[b + 2 * i + 1] = 0x3c; }
        else { float one = 1.0f; for (int i = 0; i < 4; i++) memcpy(a + b + 4 * i, &one, 4); }
    }
    float * out = aligned_alloc(64, (size_t) rows * toks * sizeof(float) * nthr);
    pthread_t th[8]; struct job jb[8]; int reps = 4;
    double t0 = now();
    for (int t = 0; t < nthr; t++) { jb[t] = (struct job){ f, K, rows, toks, w + t * wb, a, out + (size_t) t * rows * toks, reps }; pthread_create(&th[t], NULL, run, &jb[t]); }
    for (int t = 0; t < nthr; t++) pthread_join(th[t], NULL);
    double s = now() - t0;
    printf("%s  %d thr: %6.1f GMAC/s  (rows %d, K %d, tokens %d)\n", name, nthr, (double) rows * K * toks * reps * nthr / s / 1e9, rows, K, toks);
}
int main(void) {
    for (int t = 1; t <= 4; t += 3) {
        bench("q4_0x8 gemm", ggml_gemm_q4_0_8x8_q8_0, 18.0 / 32, 144, 16, 2048, 1024, 256, t, 1);
        bench("q4_Kx8 gemm", ggml_gemm_q4_K_8x8_q8_K, 144.0 / 256, 1152, 32, 2048, 1024, 256, t, 2);
    }
    return 0;
}
