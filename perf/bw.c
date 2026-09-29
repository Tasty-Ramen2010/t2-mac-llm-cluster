#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <sys/mman.h>
int main(int argc, char **argv) {
    size_t n = (size_t)1 << 28;  /* 2 GiB of floats */
    int huge = argc > 1;
    float *a = aligned_alloc(2 << 20, n * sizeof(float));
    if (huge) madvise(a, n * sizeof(float), MADV_HUGEPAGE);
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) a[i] = 1.0f;
    for (int t = 1; t <= 4; t++) {
        omp_set_num_threads(t);
        double best = 1e9;
        for (int r = 0; r < 5; r++) {
            double t0 = omp_get_wtime();
            __m256 total = _mm256_setzero_ps();
            #pragma omp parallel
            {
                __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
                #pragma omp for nowait
                for (size_t i = 0; i < n; i += 32) {
                    s0 = _mm256_add_ps(s0, _mm256_load_ps(a + i));
                    s1 = _mm256_add_ps(s1, _mm256_load_ps(a + i + 8));
                    s2 = _mm256_add_ps(s2, _mm256_load_ps(a + i + 16));
                    s3 = _mm256_add_ps(s3, _mm256_load_ps(a + i + 24));
                }
                #pragma omp critical
                total = _mm256_add_ps(total, _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
            }
            double dt = omp_get_wtime() - t0;
            if (dt < best) best = dt;
            volatile float sink = total[0]; (void)sink;
        }
        printf("%s %d thread(s): %.1f GB/s\n", huge ? "hugepages" : "normal   ", t, n * sizeof(float) / best / 1e9);
    }
    return 0;
}
