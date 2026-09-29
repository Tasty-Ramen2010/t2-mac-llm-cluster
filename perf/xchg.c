// xchg: latency of a symmetric exchange (both sides send N bytes then receive N bytes), like llama-ep's allreduce.
// node1: ./xchg server 10.10.10.1 N [spin]    node2: ./xchg client 10.10.10.1 N [spin]
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static int spin = 0;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void recv_all(int fd, char *p, size_t n) {
    while (n) {
        ssize_t k = recv(fd, p, n, spin ? MSG_DONTWAIT : 0);
        if (k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (k <= 0) { perror("recv"); exit(1); }
        p += k; n -= k;
    }
}
static void send_all(int fd, const char *p, size_t n) {
    while (n) { ssize_t k = send(fd, p, n, 0); if (k <= 0) { perror("send"); exit(1); } p += k; n -= k; }
}
int main(int argc, char **argv) {
    const int server = strcmp(argv[1], "server") == 0;
    const size_t n = atoi(argv[3]);
    spin = argc > 4 && atoi(argv[4]);
    int one = 1, fd;
    struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(50060); inet_pton(AF_INET, argv[2], &a.sin_addr);
    if (server) {
        int l = socket(AF_INET, SOCK_STREAM, 0); setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &one, 4);
        if (bind(l, (void *)&a, sizeof a)) { perror("bind"); return 1; }
        listen(l, 1); fd = accept(l, 0, 0); close(l);
    } else {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        while (connect(fd, (void *)&a, sizeof a)) { usleep(100000); close(fd); fd = socket(AF_INET, SOCK_STREAM, 0); }
    }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, 4);
    char *sb = calloc(1, n), *rb = calloc(1, n);
    const int iters = 3000;
    double t0 = 0, best = 1e9, worst = 0;
    for (int i = 0; i < iters + 200; i++) {
        if (i == 200) t0 = now();
        double s = now();
        send_all(fd, sb, n); recv_all(fd, rb, n);
        double d = now() - s;
        if (i >= 200) { if (d < best) best = d; if (d > worst) worst = d; }
    }
    double avg = (now() - t0) / iters;
    if (server) printf("%zu bytes, %s: avg %.1f us, best %.1f us, worst %.1f us per exchange\n", n, spin ? "busy-poll" : "blocking ", avg * 1e6, best * 1e6, worst * 1e6);
    close(fd);
    return 0;
}
