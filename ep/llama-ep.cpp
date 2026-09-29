#include "llama-ep.h"

#include "ggml.h"
#include "llama.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct ep_state {
    bool active = false;
    int  rank   = 0;
    int  lo     = 0;
    int  hi     = -1;
    int  fd     = -1;
    bool sliced = false;  // model file holds only owned experts, indexed from 0
    bool tp     = false;
    bool vsplit = false;
    bool solo   = false;  // no peer: allreduce is a copy (for single-node compute benchmarks)
    std::string addr;
    std::vector<char> rbuf, sbuf;
    // profiling (LLAMA_EP_PROFILE=1): time computing between syncs vs time inside syncs
    bool   prof = false;
    double t_compute = 0, t_sync = 0, t_last_exit = 0;
    long   n_sync = 0;
};

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

ep_state & st() {
    static ep_state s;
    static std::once_flag once;
    std::call_once(once, [] {
        const char * r = getenv("LLAMA_EP_RANK");
        const char * a = getenv("LLAMA_EP_ADDR");
        const char * g = getenv("LLAMA_EP_RANGE");
        const char * m = getenv("LLAMA_EP_MODE");
        const bool tp_mode = m && std::string(m) == "tp";
        if (!r || !a || (!g && !tp_mode)) {
            return;
        }
        if (!g) {
            g = "0-0";
        }
        s.rank = atoi(r);
        s.addr = a;
        if (sscanf(g, "%d-%d", &s.lo, &s.hi) != 2) {
            fprintf(stderr, "llama-ep: bad LLAMA_EP_RANGE '%s'\n", g);
            abort();
        }
        s.active = true;
        s.sliced = getenv("LLAMA_EP_SLICED") != nullptr;
        s.solo   = s.addr == "solo";
        s.vsplit = getenv("LLAMA_EP_VOCAB") && std::string(getenv("LLAMA_EP_VOCAB")) == "split";
        s.tp     = getenv("LLAMA_EP_MODE") && std::string(getenv("LLAMA_EP_MODE")) == "tp";
        s.prof = getenv("LLAMA_EP_PROFILE") != nullptr;
        if (s.prof) {
            atexit([] {
                auto & p = st();
                if (p.n_sync) fprintf(stderr, "llama-ep profile: %ld syncs, compute between syncs %.1f ms total, inside syncs %.1f ms total, per sync %.3f ms\n",
                        p.n_sync, p.t_compute * 1e3, p.t_sync * 1e3, p.t_sync * 1e3 / p.n_sync);
            });
        }
        fprintf(stderr, "llama-ep: rank %d owns experts %d-%d, peer %s\n", s.rank, s.lo, s.hi, a);
    });
    return s;
}

bool owned(int e) {
    const auto & s = st();
    return e >= s.lo && e <= s.hi;
}

void send_all(int fd, const char * p, size_t n) {
    while (n) {
        ssize_t k = ::send(fd, p, n, 0);
        if (k <= 0) { perror("llama-ep send"); abort(); }
        p += k; n -= (size_t) k;
    }
}

void recv_all(int fd, char * p, size_t n) {
    while (n) {
        ssize_t k = ::recv(fd, p, n, 0);
        if (k <= 0) { perror("llama-ep recv"); abort(); }
        p += k; n -= (size_t) k;
    }
}

int connect_peer() {
    auto & s = st();
    if (s.fd >= 0) {
        return s.fd;
    }
    std::string host = s.addr.substr(0, s.addr.rfind(':'));
    std::string port = s.addr.substr(s.addr.rfind(':') + 1);
    addrinfo hints = {}, * res = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) {
        fprintf(stderr, "llama-ep: cannot resolve %s\n", s.addr.c_str());
        abort();
    }
    int one = 1;
    if (s.rank == 0) {
        int l = socket(res->ai_family, SOCK_STREAM, 0);
        setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(l, res->ai_addr, res->ai_addrlen) != 0 || listen(l, 1) != 0) {
            perror("llama-ep bind/listen");
            abort();
        }
        fprintf(stderr, "llama-ep: rank 0 waiting for peer on %s\n", s.addr.c_str());
        s.fd = accept(l, nullptr, nullptr);
        close(l);
    } else {
        for (int attempt = 0; attempt < 600; attempt++) {
            s.fd = socket(res->ai_family, SOCK_STREAM, 0);
            if (connect(s.fd, res->ai_addr, res->ai_addrlen) == 0) {
                break;
            }
            close(s.fd);
            s.fd = -1;
            usleep(200000);
        }
        if (s.fd < 0) {
            fprintf(stderr, "llama-ep: could not connect to %s\n", s.addr.c_str());
            abort();
        }
    }
    freeaddrinfo(res);
    setsockopt(s.fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    fprintf(stderr, "llama-ep: connected to peer\n");
    return s.fd;
}

// selected_experts: I32 [n_used, n_tokens] (may be a strided view); dst is contiguous
void op_remap_ids(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void *) {
    if (ith != 0) {
        return;
    }
    (void) nth;
    const int64_t n_used = a->ne[0], n_tok = a->ne[1];
    const int lo = st().lo;
    for (int64_t t = 0; t < n_tok; t++) {
        const char * src = (const char *) a->data + t * a->nb[1];
        int32_t * out = (int32_t *) ((char *) dst->data + t * dst->nb[1]);
        const int32_t off = st().sliced ? lo : 0;
        for (int64_t i = 0; i < n_used; i++) {
            const int32_t e = *(const int32_t *) (src + i * a->nb[0]);
            out[i] = owned(e) ? e - off : -1;  // -1: skipped by mul_mat_id / add_id
        }
    }
}

// weights: F32 [1, n_used, n_tokens]; b = selected_experts I32 [n_used, n_tokens]
void op_mask_weights(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void *) {
    if (ith != 0) {
        return;
    }
    (void) nth;
    const int64_t n_used = a->ne[1], n_tok = a->ne[2];
    for (int64_t t = 0; t < n_tok; t++) {
        for (int64_t i = 0; i < n_used; i++) {
            const float w  = *(const float *) ((const char *) a->data + i * a->nb[1] + t * a->nb[2]);
            const int32_t e = *(const int32_t *) ((const char *) b->data + i * b->nb[0] + t * b->nb[1]);
            *(float *) ((char *) dst->data + i * dst->nb[1] + t * dst->nb[2]) = owned(e) ? w : 0.0f;
        }
    }
}

// dst = a + peer's a. a must be contiguous F32.
void op_allreduce(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void *) {
    if (ith != 0) {
        return;
    }
    (void) nth;
    GGML_ASSERT(a->type == GGML_TYPE_F32 && ggml_is_contiguous(a));
    const size_t nbytes = ggml_nbytes(a);
    if (st().solo) {
        memcpy(dst->data, a->data, nbytes);
        return;
    }
    const int fd = connect_peer();
    auto & s = st();
    const double t_enter = s.prof ? now_s() : 0;
    if (s.prof && s.t_last_exit > 0 && t_enter - s.t_last_exit < 1.0) {
        s.t_compute += t_enter - s.t_last_exit;
    }
    // wire format: 8-byte header (element count | F16 flag) + payload, sent in one piece. Payload is fp16 when
    // every value fits (halves the bytes on the 1 GbE cable); each side picks independently, receiver follows the flag.
    const size_t n = nbytes / sizeof(float);
    const float * x = (const float *) a->data;
    static const bool f16_ok = [] { const char * e = getenv("LLAMA_EP_F16"); return !(e && e[0] == '0'); }();
    bool f16 = f16_ok;
    if (f16) {
        float m = 0.0f;
        for (size_t i = 0; i < n; i++) m = std::max(m, std::fabs(x[i]));
        f16 = m < 60000.0f;
    }
    constexpr uint64_t F16_FLAG = 1ull << 63;
    const size_t payload = f16 ? n * sizeof(ggml_fp16_t) : nbytes;
    if (s.sbuf.size() < 8 + nbytes) s.sbuf.resize(8 + nbytes);
    if (s.rbuf.size() < nbytes)     s.rbuf.resize(nbytes);
    const uint64_t hdr = n | (f16 ? F16_FLAG : 0);
    memcpy(s.sbuf.data(), &hdr, 8);
    if (f16) ggml_fp32_to_fp16_row(x, (ggml_fp16_t *) (s.sbuf.data() + 8), (int64_t) n);
    else     memcpy(s.sbuf.data() + 8, x, nbytes);

    uint64_t peer_hdr = 0;
    auto recv_side = [&] {
        recv_all(fd, (char *) &peer_hdr, sizeof(peer_hdr));
        GGML_ASSERT((peer_hdr & ~F16_FLAG) == n && "llama-ep: ranks out of sync");
        recv_all(fd, s.rbuf.data(), (peer_hdr & F16_FLAG) ? n * sizeof(ggml_fp16_t) : nbytes);
    };
    // full duplex: send on a helper thread for large tensors, inline for small ones
    if (payload <= 64 * 1024) {
        send_all(fd, s.sbuf.data(), 8 + payload);
        recv_side();
    } else {
        std::thread sender([&] { send_all(fd, s.sbuf.data(), 8 + payload); });
        recv_side();
        sender.join();
    }
    float * z = (float *) dst->data;
    if (peer_hdr & F16_FLAG) {
        const ggml_fp16_t * y = (const ggml_fp16_t *) s.rbuf.data();
        for (size_t i = 0; i < n; i++) z[i] = x[i] + ggml_fp16_to_fp32(y[i]);
    } else {
        const float * y = (const float *) s.rbuf.data();
        for (size_t i = 0; i < n; i++) z[i] = x[i] + y[i];
    }
    if (s.prof) {
        s.t_last_exit = now_s();
        s.t_sync += s.t_last_exit - t_enter;
        s.n_sync++;
    }
}

constexpr int EP_TOPK = 64;

// exchange a small buffer with the peer (both send, then both receive)
void exchange_small(const void * out, size_t n, void * in) {
    const int fd = connect_peer();
    const uint64_t hdr = n;
    send_all(fd, (const char *) &hdr, sizeof(hdr));
    send_all(fd, (const char *) out, n);
    uint64_t peer_hdr = 0;
    recv_all(fd, (char *) &peer_hdr, sizeof(peer_hdr));
    GGML_ASSERT(peer_hdr == hdr && "llama-ep: ranks out of sync");
    recv_all(fd, (char *) in, n);
}

struct idx_val { int32_t i; float v; };

// a: [n_vocab, n_out] with this rank's n_local logits in rows [0, n_local); dst: full logits
void op_logits_gather(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * ud) {
    if (ith != 0) {
        return;
    }
    (void) nth;
    const int64_t n_vocab = a->ne[0], n_out = a->ne[1];
    const int64_t n_local = (int64_t) (intptr_t) ud;
    const int64_t off     = st().rank * n_local;
    const int k = (int) std::min<int64_t>(EP_TOPK, n_local);
    std::vector<idx_val> mine(n_out * k), peer(n_out * k);
    std::vector<int32_t> order(n_local);
    for (int64_t c = 0; c < n_out; c++) {
        const float * own = (const float *) ((const char *) a->data + c * a->nb[1]);
        float * out = (float *) ((char *) dst->data + c * dst->nb[1]);
        std::fill(out, out + n_vocab, -INFINITY);
        std::copy(own, own + n_local, out + off);
        for (int64_t i = 0; i < n_local; i++) order[i] = (int32_t) i;
        std::nth_element(order.begin(), order.begin() + k, order.end(), [&](int32_t x, int32_t y) { return own[x] > own[y]; });
        for (int j = 0; j < k; j++) mine[c * k + j] = { (int32_t) (order[j] + off), own[order[j]] };
    }
    if (!st().solo) {
        exchange_small(mine.data(), mine.size() * sizeof(idx_val), peer.data());
        for (int64_t c = 0; c < n_out; c++) {
            float * out = (float *) ((char *) dst->data + c * dst->nb[1]);
            for (int j = 0; j < k; j++) out[peer[c * k + j].i] = peer[c * k + j].v;
        }
    }
}

int g_ctrl_fd = -1;

int ctrl_fd() {
    if (g_ctrl_fd >= 0) {
        return g_ctrl_fd;
    }
    const std::string addr = getenv("LLAMA_EP_CTRL");
    std::string host = addr.substr(0, addr.rfind(':')), port = addr.substr(addr.rfind(':') + 1);
    addrinfo hints = {}, * res = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) { fprintf(stderr, "llama-ep: bad LLAMA_EP_CTRL\n"); abort(); }
    int one = 1;
    int l = socket(res->ai_family, SOCK_STREAM, 0);
    setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (bind(l, res->ai_addr, res->ai_addrlen) != 0 || listen(l, 1) != 0) { perror("llama-ep ctrl bind"); abort(); }
    fprintf(stderr, "llama-ep: waiting for follower on control %s\n", addr.c_str());
    g_ctrl_fd = accept(l, nullptr, nullptr);
    close(l);
    freeaddrinfo(res);
    setsockopt(g_ctrl_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    fprintf(stderr, "llama-ep: follower attached\n");
    return g_ctrl_fd;
}

} // namespace

bool llama_ep_mirroring() {
    static const bool on = st().active && st().rank == 0 && getenv("LLAMA_EP_CTRL") != nullptr;
    return on;
}

void llama_ep_mirror_op(llama_ep_op op, int32_t a0, int32_t a1, int32_t a2, int32_t a3) {
    const int32_t msg[5] = { (int32_t) op, a0, a1, a2, a3 };
    send_all(ctrl_fd(), (const char *) msg, sizeof(msg));
}

void llama_ep_mirror_decode(const llama_batch & b, int n_embd) {
    const int32_t n = b.n_tokens;
    const int32_t flags = (b.token ? 1 : 0) | (b.embd ? 2 : 0) | (b.pos ? 4 : 0) | (b.seq_id ? 8 : 0) | (b.logits ? 16 : 0);
    std::vector<char> buf;
    auto put = [&](const void * p, size_t sz) { buf.insert(buf.end(), (const char *) p, (const char *) p + sz); };
    const int32_t hdr[5] = { (int32_t) LLAMA_EP_OP_DECODE, n, flags, n_embd, 0 };
    put(hdr, sizeof(hdr));
    if (b.token)  put(b.token, n * sizeof(llama_token));
    if (b.embd)   put(b.embd, (size_t) n * n_embd * sizeof(float));
    if (b.pos)    put(b.pos, n * sizeof(llama_pos));
    if (b.seq_id) {
        put(b.n_seq_id, n * sizeof(int32_t));
        for (int32_t i = 0; i < n; i++) put(b.seq_id[i], b.n_seq_id[i] * sizeof(llama_seq_id));
    }
    if (b.logits) put(b.logits, n * sizeof(int8_t));
    send_all(ctrl_fd(), buf.data(), buf.size());
}

bool llama_ep_active() { return st().active; }
int  llama_ep_n_vocab_local(int n_vocab) { return st().active && st().vsplit ? n_vocab / 2 : n_vocab; }

ggml_tensor * llama_ep_logits_gather(ggml_context * ctx, ggml_tensor * logits_local, int n_vocab) {
    const int64_t n_local = logits_local->ne[0];
    ggml_tensor * padded = ggml_pad(ctx, logits_local, n_vocab - n_local, 0, 0, 0);
    return ggml_map_custom1(ctx, padded, op_logits_gather, 1, (void *) (intptr_t) n_local);
}

bool llama_ep_tp()     { return st().active && st().tp; }
bool llama_ep_attn_split() {
    static const bool on = [] { const char * e = getenv("LLAMA_EP_ATTN"); return llama_ep_active() && e && std::string(e) == "split"; }();
    return on;
}
int  llama_ep_n_local(int n_expert) { return st().active && st().sliced ? st().hi - st().lo + 1 : n_expert; }
int  llama_ep_rank()   { return st().rank; }

ggml_tensor * llama_ep_mask_weights(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * selected_experts) {
    return ggml_map_custom2(ctx, weights, selected_experts, op_mask_weights, 1, nullptr);
}

ggml_tensor * llama_ep_remap_ids(ggml_context * ctx, ggml_tensor * selected_experts) {
    return ggml_map_custom1(ctx, selected_experts, op_remap_ids, 1, nullptr);
}

ggml_tensor * llama_ep_allreduce(ggml_context * ctx, ggml_tensor * t) {
    return ggml_map_custom1(ctx, ggml_cont(ctx, t), op_allreduce, 1, nullptr);
}
