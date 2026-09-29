// pingpong: measure one tensor-parallel sync (both sides send N bytes, then receive N bytes) over TCP.
// usage: pingpong server PORT BYTES MODE | pingpong client HOST PORT BYTES MODE
// MODE: block (plain recv), spin (non-blocking recv loop), busypoll (SO_BUSY_POLL 50us + block), spinbp (both)
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int spin;
static void sendall(int fd, char *p, size_t n) { while (n) { ssize_t k = send(fd, p, n, 0); if (k <= 0) { if (errno == EAGAIN) continue; perror("send"); exit(1); } p += k; n -= k; } }
static void recvall(int fd, char *p, size_t n) {
    while (n) {
        ssize_t k = recv(fd, p, n, spin ? MSG_DONTWAIT : 0);
        if (k < 0 && errno == EAGAIN) continue;
        if (k <= 0) { perror("recv"); exit(1); }
        p += k; n -= k;
    }
}
int main(int argc, char **argv) {
    int server = !strcmp(argv[1], "server");
    const char *host = server ? NULL : argv[2];
    int port = atoi(argv[server ? 2 : 3]); size_t n = atoi(argv[server ? 3 : 4]); const char *mode = argv[server ? 4 : 5];
    spin = !strncmp(mode, "spin", 4);
    int fd, one = 1;
    struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(port) };
    if (server) {
        int l = socket(AF_INET, SOCK_STREAM, 0); setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &one, 4);
        a.sin_addr.s_addr = INADDR_ANY; if (bind(l, (void *) &a, sizeof a) || listen(l, 1)) { perror("bind"); return 1; }
        fd = accept(l, NULL, NULL); close(l);
    } else {
        inet_pton(AF_INET, host, &a.sin_addr); fd = socket(AF_INET, SOCK_STREAM, 0);
        for (int i = 0; connect(fd, (void *) &a, sizeof a); i++) { if (i > 100) { perror("connect"); return 1; } usleep(100000); close(fd); fd = socket(AF_INET, SOCK_STREAM, 0); }
    }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, 4);
    if (strstr(mode, "bp")|| !strcmp(mode, "busypoll")) { int us = 50; if (setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &us, 4)) perror("SO_BUSY_POLL"); }
    char *sb = malloc(n), *rb = malloc(n); memset(sb, 7, n);
    int iters = 5000; double t0 = 0;
    for (int i = -200; i < iters; i++) {
        if (i == 0) t0 = now();
        sendall(fd, sb, n); recvall(fd, rb, n);
    }
    double us = (now() - t0) / iters * 1e6;
    if (server) printf("%-9s %6zu bytes: %6.1f us per sync\n", mode, n, us);
    close(fd); return 0;
}
