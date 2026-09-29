// udppong: one tensor-parallel sync over UDP (both send N bytes as <=1472-byte datagrams, then receive N), spin-polling
// usage: udppong server PORT BYTES | udppong client HOST PORT BYTES
#include <arpa/inet.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int server = !strcmp(argv[1], "server");
    int port = atoi(argv[server ? 2 : 3]); size_t n = atoi(argv[server ? 3 : 4]);
    int fd = socket(AF_INET, SOCK_DGRAM, 0), big = 4 << 20;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &big, 4); setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &big, 4);
    struct sockaddr_in me = { .sin_family = AF_INET, .sin_port = htons(port) }, peer = me;
    if (server) { me.sin_addr.s_addr = INADDR_ANY; bind(fd, (void *) &me, sizeof me); }
    else inet_pton(AF_INET, argv[2], &peer.sin_addr);
    char sb[65536], rb[65536]; memset(sb, 7, sizeof sb);
    socklen_t pl = sizeof peer;
    if (server) recvfrom(fd, rb, sizeof rb, 0, (void *) &peer, &pl);           // learn the client's address
    else { sendto(fd, "hi", 2, 0, (void *) &peer, sizeof peer); }
    connect(fd, (void *) &peer, sizeof peer);
    const size_t MTU = 1472; int iters = 5000; double t0 = 0;
    for (int i = -300; i < iters; i++) {
        if (i == 0) t0 = now();
        for (size_t off = 0; off < n; off += MTU) send(fd, sb + off, n - off < MTU ? n - off : MTU, 0);
        size_t got = 0;
        while (got < n) { ssize_t k = recv(fd, rb, sizeof rb, MSG_DONTWAIT); if (k > 0) got += k; else if (k < 0 && errno != EAGAIN) { perror("recv"); return 1; } }
    }
    if (server) printf("udp spin %6zu bytes: %6.1f us per sync\n", n, (now() - t0) / iters * 1e6);
    return 0;
}
