/*
 * Assignment 3 - shared-memory parallel dense matrix multiplication (OpenMP), in C.
 * Storage: one flat double array per matrix, row-major (element (i, j) at i*n + j), as in Assignments 1-2.
 *
 * Methods (same algorithm, same arithmetic, same order of the k terms for every C[i][j]):
 *   seq    i-k-j kernel of Assignment 2 (mm_ikj in matmul2.c): sequential baseline, no OpenMP inside.
 *          Only change: C is zeroed row by row (memset of row i before computing it) instead of with one
 *          memset of the whole matrix, so that seq and par execute exactly the same row body.
 *   par    the same row body; the outer loop over i is shared among p OpenMP threads with
 *          schedule(static): thread t receives one contiguous band of about n/p rows of C.
 *          Every thread writes only its own rows of C and only reads A and B -> no data race,
 *          no locks, no atomics. The implicit barrier at the end of the loop joins the threads.
 *   naive  i-j-k loop of Assignment 1, used only as correctness reference in the tests.
 *
 * Build: gcc -O2 -fno-tree-vectorize -fopenmp -std=c11 -Wall -o build/matmul3 src/matmul3.c -lm
 * Usage:
 *   matmul3 test                                   correctness tests (CSV on stdout)
 *   matmul3 bench METHOD n p warmup reps datadir   METHOD = seq | par; p = threads (ignored by seq)
 * Output of bench, one row per measured repetition:
 *   method,n,p,team,rep,batch,time_s,cpu_s,checksum,peak_rss_kb
 *   team  = number of threads that actually ran the parallel region (1 for seq)
 *   cpu_s = process CPU time (all threads, user+system) per multiplication in the timed region
 * Environment: BUDGET_S = time budget in seconds for ONE timed run (exit code 3 on timeout).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <omp.h>

/* ---------------- kernels ---------------- */

/* Row i of C = sum over k of A[i][k] * (row k of B); inner loop with stride 1 (Assignment 2, i-k-j). */
static inline void row_ikj(const double *a, const double *b, double *c, int n, int i) {
    double *ci = c + (size_t) i * n;
    const double *ai = a + (size_t) i * n;
    memset(ci, 0, (size_t) n * sizeof(double));
    for (int k = 0; k < n; k++) {
        const double aik = ai[k];
        const double *bk = b + (size_t) k * n;
        for (int j = 0; j < n; j++) ci[j] += aik * bk[j];
    }
}

static void mm_seq(const double *a, const double *b, double *c, int n) {
    for (int i = 0; i < n; i++) row_ikj(a, b, c, n, i);
}

static void mm_par(const double *a, const double *b, double *c, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) row_ikj(a, b, c, n, i);
}

/* Assignment 1 baseline (i-j-k), reference for the tests only. */
static void mm_naive(const double *a, const double *b, double *c, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += a[(size_t) i * n + k] * b[(size_t) k * n + j];
            c[(size_t) i * n + j] = s;
        }
}

/* ---------------- helpers ---------------- */

static uint64_t sm_state;                  /* splitmix64, test inputs only (same generator as Assignment 2) */
static uint64_t sm_next(void) {
    uint64_t z = (sm_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void gen(double *d, long len, uint64_t seed) {
    sm_state = seed;
    for (long x = 0; x < len; x++) d[x] = 2.0 * ((sm_next() >> 11) * 0x1.0p-53) - 1.0;
}

static double *load(const char *path, int n) {
    double *m = malloc((size_t) n * n * sizeof(double));
    FILE *f = fopen(path, "rb");
    if (!m || !f || fread(m, sizeof(double), (size_t) n * n, f) != (size_t) n * n) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    fclose(f);
    return m;
}

static long peak_rss_kb(void) {            /* VmHWM = peak resident set size of the process (all threads) */
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    long kb = -1;
    while (f && fgets(line, sizeof line, f))
        if (strncmp(line, "VmHWM:", 6) == 0) { kb = strtol(line + 6, NULL, 10); break; }
    if (f) fclose(f);
    return kb;
}

static double checksum(const double *c, long len) {
    double s = 0.0;
    for (long i = 0; i < len; i++) s += c[i];
    return s;
}

static double clk(clockid_t id) {
    struct timespec t;
    clock_gettime(id, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static unsigned budget_s = 0;
static void on_alarm(int sig) { (void) sig; static const char m[] = "TIMEOUT\n"; if (write(2, m, sizeof m - 1) < 0) _exit(4); _exit(3); }

static int batch_size(int n) {             /* small sizes timed in batches, as in Assignments 1-2 */
    long b = (1L << 24) / ((long) n * n * n);
    return b < 1 ? 1 : (int) b;
}

/* ---------------- correctness tests ---------------- */

static double maxdiff(const double *x, const double *ref, long len, double atol, double rtol, int *ok) {
    double worst = 0.0;
    *ok = 1;
    for (long i = 0; i < len; i++) {
        double e = fabs(x[i] - ref[i]);
        if (e > atol + rtol * fabs(ref[i])) *ok = 0;
        if (e > worst) worst = e;
    }
    return worst;
}

static int tests(void) {
    const double ATOL = 1e-12, RTOL = 1e-12;
    int all = 1, ok;
    omp_set_dynamic(0);
    printf("test,n,method,p,max_abs_diff,pass\n");
    double a2[] = {1, 2, 3, 4}, b2[] = {5, 6, 7, 8}, e2[] = {19, 22, 43, 50}, c2[4];
    mm_seq(a2, b2, c2, 2);
    double d = maxdiff(c2, e2, 4, 0, 0, &ok);
    printf("known_2x2,2,seq,1,%g,%d\n", d, ok); all &= ok;
    int ps[] = {1, 2, 3, 4, 8};
    for (int t = 0; t < 5; t++) {
        omp_set_num_threads(ps[t]);
        mm_par(a2, b2, c2, 2);
        d = maxdiff(c2, e2, 4, 0, 0, &ok);
        printf("known_2x2,2,par,%d,%g,%d\n", ps[t], d, ok); all &= ok;
    }
    int sizes[] = {1, 7, 67, 130, 257};      /* n = 1 < p and sizes not divisible by p */
    for (int s = 0; s < 5; s++) {
        int n = sizes[s];
        long len = (long) n * n;
        double *a = malloc(len * 8), *b = malloc(len * 8), *id = calloc(len, 8), *z = calloc(len, 8),
               *ref = malloc(len * 8), *cs = malloc(len * 8), *c = malloc(len * 8);
        gen(a, len, 11 + n);
        gen(b, len, 99 + n);
        for (int i = 0; i < n; i++) id[(size_t) i * n + i] = 1.0;
        mm_naive(a, b, ref, n);
        mm_seq(a, b, cs, n);
        d = maxdiff(cs, ref, len, ATOL, RTOL, &ok);
        printf("seq_vs_naive,%d,seq,1,%g,%d\n", n, d, ok); all &= ok;
        for (int t = 0; t < 5; t++) {
            omp_set_num_threads(ps[t]);
            mm_par(a, id, c, n);
            d = maxdiff(c, a, len, 0, 0, &ok);
            printf("A*I=A,%d,par,%d,%g,%d\n", n, ps[t], d, ok); all &= ok;
            for (long x = 0; x < len; x++) c[x] = 7.0;     /* garbage in C: the kernel must overwrite it */
            mm_par(a, z, c, n);
            d = maxdiff(c, z, len, 0, 0, &ok);
            printf("A*0=0,%d,par,%d,%g,%d\n", n, ps[t], d, ok); all &= ok;
            mm_par(a, b, c, n);
            d = maxdiff(c, cs, len, 0, 0, &ok);        /* tolerance 0: must be bitwise equal to seq */
            printf("par_equals_seq_bitwise,%d,par,%d,%g,%d\n", n, ps[t], d, ok); all &= ok;
        }
        free(a); free(b); free(id); free(z); free(ref); free(cs); free(c);
    }
    fprintf(stderr, all ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    return all ? 0 : 1;
}

/* ---------------- main ---------------- */

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    if (!strcmp(argv[1], "test")) return tests();
    if (strcmp(argv[1], "bench") || argc < 8) { fprintf(stderr, "usage: matmul3 bench seq|par n p warmup reps datadir\n"); return 1; }
    if (getenv("BUDGET_S")) { budget_s = (unsigned) atoi(getenv("BUDGET_S")); signal(SIGALRM, on_alarm); }

    const char *method = argv[2];
    int n = atoi(argv[3]), p = atoi(argv[4]), warmup = atoi(argv[5]), reps = atoi(argv[6]);
    int par = !strcmp(method, "par");
    if (!par && strcmp(method, "seq")) { fprintf(stderr, "unknown method %s\n", method); return 1; }
    char pa[1024], pb[1024];
    snprintf(pa, sizeof pa, "%s/A_%d.bin", argv[7], n);
    snprintf(pb, sizeof pb, "%s/B_%d.bin", argv[7], n);
    double *a = load(pa, n), *b = load(pb, n);              /* outside the timed region */
    double *c = calloc((size_t) n * n, sizeof(double));     /* output allocated outside */

    int team = 1;
    if (par) {                                              /* fix and verify the thread count */
        omp_set_dynamic(0);
        omp_set_num_threads(p);
        #pragma omp parallel
        {
            #pragma omp single
            team = omp_get_num_threads();
        }
    }
    int batch = batch_size(n);
    for (int r = -warmup; r < reps; r++) {
        if (budget_s) alarm(budget_s);
        double c0 = clk(CLOCK_PROCESS_CPUTIME_ID), t0 = clk(CLOCK_MONOTONIC);
        for (int i = 0; i < batch; i++) { if (par) mm_par(a, b, c, n); else mm_seq(a, b, c, n); }
        double t1 = clk(CLOCK_MONOTONIC), c1 = clk(CLOCK_PROCESS_CPUTIME_ID);
        if (budget_s) alarm(0);
        double chk = checksum(c, (long) n * n);             /* consumes the result */
        if (r >= 0) printf("%s,%d,%d,%d,%d,%d,%.9f,%.9f,%.12e,%ld\n", method, n, par ? p : 1, team, r, batch,
                           (t1 - t0) / batch, (c1 - c0) / batch, chk, peak_rss_kb());
        fflush(stdout);
    }
    free(a); free(b); free(c);
    return 0;
}
