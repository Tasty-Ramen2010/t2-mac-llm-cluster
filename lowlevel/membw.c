// membw: read bandwidth with N threads (AVX2 loads, like a gemv streaming weights). usage: membw MB THREADS
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
static size_t n; static char *buf; static int nt; static volatile double sink;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *work(void *a) {
    long id = (long) a; size_t chunk = n / nt, lo = id * chunk, hi = lo + chunk;
    __m256i acc = _mm256_setzero_si256();
    for (int rep = 0; rep < 5; rep++)
        for (size_t i = lo; i < hi; i += 128) {
            acc = _mm256_xor_si256(acc, _mm256_load_si256((__m256i *)(buf + i)));
            acc = _mm256_xor_si256(acc, _mm256_load_si256((__m256i *)(buf + i + 32)));
            acc = _mm256_xor_si256(acc, _mm256_load_si256((__m256i *)(buf + i + 64)));
            acc = _mm256_xor_si256(acc, _mm256_load_si256((__m256i *)(buf + i + 96)));
        }
    sink += _mm256_extract_epi32(acc, 0);
    return NULL;
}
int main(int argc, char **argv) {
    n = (size_t) atoi(argv[1]) << 20; nt = atoi(argv[2]);
    buf = aligned_alloc(2 << 20, n); madvise(buf, n, MADV_HUGEPAGE); memset(buf, 1, n);
    pthread_t th[16]; double t0 = now();
    for (long i = 0; i < nt; i++) pthread_create(&th[i], NULL, work, (void *) i);
    for (int i = 0; i < nt; i++) pthread_join(th[i], NULL);
    double s = now() - t0;
    printf("%d threads: %.1f GB/s\n", nt, 5.0 * n / s / 1e9);
    return 0;
}
