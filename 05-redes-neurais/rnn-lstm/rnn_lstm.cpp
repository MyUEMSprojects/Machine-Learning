// RNN simples e LSTM do zero com BPTT (backpropagation through time) + Adam.
// Tarefa de memória: sequência de T números; o rótulo é o SINAL do PRIMEIRO elemento
// (a rede precisa "lembrar" x_0 por T passos).
// Build: g++ -std=c++17 -O2 rnn_lstm.cpp -o rnn_lstm
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;

double sigm(double z) { return 1 / (1 + exp(-z)); }

struct Model {
    Vec P, G, m, v; int t = 0;  // parâmetros, gradientes e estado do Adam
    virtual ~Model() = default;
    virtual double loss_grad(const Vec& seq, int y) = 0;    // acumula em G, retorna perda
    virtual double logit(const Vec& seq) = 0;
    void adam(double lr, int batch) {
        if (m.empty()) { m.assign(P.size(), 0); v.assign(P.size(), 0); }
        ++t;
        double norm = 0;
        for (double g : G) norm += g * g;
        norm = sqrt(norm) / batch;
        double clip = norm > 1 ? 1 / norm : 1;  // gradient clipping (norma <= 1)
        for (size_t i = 0; i < P.size(); ++i) {
            double g = G[i] / batch * clip;
            m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g;
            P[i] -= lr * (m[i] / (1 - pow(0.9, t))) / (sqrt(v[i] / (1 - pow(0.999, t))) + 1e-8);
            G[i] = 0;
        }
    }
};

// ---------------- RNN simples: h_t = tanh(wx x_t + Wh h_{t-1} + b) ----------------
struct RNN : Model {
    int H; // layout: Wx[H], Wh[H*H], b[H], wo[H], bo
    RNN(int H_, mt19937& rng) : H(H_) {
        P.assign(H * H + 3 * H + 1, 0); G = P;
        normal_distribution<double> N(0, 1);
        for (int i = 0; i < H; ++i) P[i] = N(rng);
        for (int i = 0; i < H * H; ++i) P[H + i] = N(rng) * 0.5 / sqrt(H);
        for (int i = 0; i < H; ++i) P[H + H * H + H + i] = N(rng) * 0.5;
    }
    double* Wx() { return &P[0]; } double* Wh() { return &P[H]; } double* b() { return &P[H + H * H]; }
    double* wo() { return &P[H + H * H + H]; } double* bo() { return &P[H + H * H + 2 * H]; }
    double* gWx() { return &G[0]; } double* gWh() { return &G[H]; } double* gb() { return &G[H + H * H]; }
    double* gwo() { return &G[H + H * H + H]; } double* gbo() { return &G[H + H * H + 2 * H]; }

    vector<Vec> run(const Vec& seq) {  // h[0]=0, h[t+1] após consumir seq[t]
        vector<Vec> h(seq.size() + 1, Vec(H, 0));
        for (size_t t = 0; t < seq.size(); ++t)
            for (int i = 0; i < H; ++i) {
                double s = b()[i] + Wx()[i] * seq[t];
                for (int j = 0; j < H; ++j) s += Wh()[i * H + j] * h[t][j];
                h[t + 1][i] = tanh(s);
            }
        return h;
    }
    double logit(const Vec& seq) override {
        auto h = run(seq);
        double z = *bo();
        for (int i = 0; i < H; ++i) z += wo()[i] * h.back()[i];
        return z;
    }
    double loss_grad(const Vec& seq, int y) override {
        auto h = run(seq);
        double z = *bo();
        for (int i = 0; i < H; ++i) z += wo()[i] * h.back()[i];
        double p = sigm(z), dz = p - y;
        *gbo() += dz;
        Vec dh(H);
        for (int i = 0; i < H; ++i) { gwo()[i] += dz * h.back()[i]; dh[i] = dz * wo()[i]; }
        for (int t = seq.size() - 1; t >= 0; --t) {  // BPTT: do último passo até o primeiro
            Vec da(H), dprev(H, 0);
            for (int i = 0; i < H; ++i) da[i] = dh[i] * (1 - h[t + 1][i] * h[t + 1][i]);
            for (int i = 0; i < H; ++i) {
                gWx()[i] += da[i] * seq[t]; gb()[i] += da[i];
                for (int j = 0; j < H; ++j) { gWh()[i * H + j] += da[i] * h[t][j]; dprev[j] += Wh()[i * H + j] * da[i]; }
            }
            dh = dprev;  // aqui o gradiente é multiplicado por Wh e tanh' a cada passo => some/explode
        }
        return -(y ? log(p + 1e-12) : log(1 - p + 1e-12));
    }
};

// ---------------- LSTM ----------------
// gates: i, f, g, o (ordem das linhas). W: 4H x (1+H) sobre [x_t, h_{t-1}], b: 4H, wo: H, bo
struct LSTM : Model {
    int H, C;  // C = 1 + H (colunas)
    LSTM(int H_, mt19937& rng) : H(H_), C(1 + H_) {
        P.assign(4 * H * C + 4 * H + H + 1, 0); G = P;
        normal_distribution<double> N(0, 1);
        for (int i = 0; i < 4 * H * C; ++i) P[i] = N(rng) * 0.5 / sqrt(C);
        for (int i = 0; i < H; ++i) P[4 * H * C + H + i] = 1.0;  // viés do forget gate = 1 (lembrar por padrão)
        for (int i = 0; i < H; ++i) P[4 * H * C + 4 * H + i] = N(rng) * 0.5;
    }
    double* W() { return &P[0]; } double* b() { return &P[4 * H * C]; }
    double* wo() { return &P[4 * H * C + 4 * H]; } double* bo() { return &P[4 * H * C + 4 * H + H]; }
    double* gW() { return &G[0]; } double* gb() { return &G[4 * H * C]; }
    double* gwo() { return &G[4 * H * C + 4 * H]; } double* gbo() { return &G[4 * H * C + 4 * H + H]; }

    struct Step { Vec xh, i, f, g, o, c, cprev, h; };
    vector<Step> run(const Vec& seq) {
        vector<Step> st(seq.size());
        Vec h(H, 0), c(H, 0);
        for (size_t t = 0; t < seq.size(); ++t) {
            Step& s = st[t];
            s.xh.assign(C, 0); s.xh[0] = seq[t];
            for (int j = 0; j < H; ++j) s.xh[1 + j] = h[j];
            s.i.resize(H); s.f.resize(H); s.g.resize(H); s.o.resize(H); s.c.resize(H); s.h.resize(H); s.cprev = c;
            for (int k = 0; k < H; ++k) {
                double a[4];
                for (int gate = 0; gate < 4; ++gate) {
                    int r = gate * H + k;
                    a[gate] = b()[r];
                    for (int j = 0; j < C; ++j) a[gate] += W()[r * C + j] * s.xh[j];
                }
                s.i[k] = sigm(a[0]); s.f[k] = sigm(a[1]); s.g[k] = tanh(a[2]); s.o[k] = sigm(a[3]);
                s.c[k] = s.f[k] * c[k] + s.i[k] * s.g[k];  // estado da célula: "esteira" com soma (gradiente flui bem)
                s.h[k] = s.o[k] * tanh(s.c[k]);
            }
            c = s.c; h = s.h;
        }
        return st;
    }
    double logit(const Vec& seq) override {
        auto st = run(seq);
        double z = *bo();
        for (int i = 0; i < H; ++i) z += wo()[i] * st.back().h[i];
        return z;
    }
    double loss_grad(const Vec& seq, int y) override {
        auto st = run(seq);
        double z = *bo();
        for (int i = 0; i < H; ++i) z += wo()[i] * st.back().h[i];
        double p = sigm(z), dz = p - y;
        *gbo() += dz;
        Vec dh(H), dc(H, 0);
        for (int i = 0; i < H; ++i) { gwo()[i] += dz * st.back().h[i]; dh[i] = dz * wo()[i]; }
        for (int t = seq.size() - 1; t >= 0; --t) {
            Step& s = st[t];
            Vec dxh(C, 0), dcprev(H);
            for (int k = 0; k < H; ++k) {
                double tc = tanh(s.c[k]);
                double d_o = dh[k] * tc;
                double dck = dc[k] + dh[k] * s.o[k] * (1 - tc * tc);
                double da[4] = {dck * s.g[k] * s.i[k] * (1 - s.i[k]),            // i
                                dck * s.cprev[k] * s.f[k] * (1 - s.f[k]),        // f
                                dck * s.i[k] * (1 - s.g[k] * s.g[k]),            // g
                                d_o * s.o[k] * (1 - s.o[k])};                    // o
                dcprev[k] = dck * s.f[k];
                for (int gate = 0; gate < 4; ++gate) {
                    int r = gate * H + k;
                    gb()[r] += da[gate];
                    for (int j = 0; j < C; ++j) { gW()[r * C + j] += da[gate] * s.xh[j]; dxh[j] += W()[r * C + j] * da[gate]; }
                }
            }
            for (int j = 0; j < H; ++j) dh[j] = dxh[1 + j];
            dc = dcprev;
        }
        return -(y ? log(p + 1e-12) : log(1 - p + 1e-12));
    }
};

void experimento(const char* nome, function<unique_ptr<Model>(mt19937&)> make, int T, mt19937& rng) {
    normal_distribution<double> N(0, 1);
    // x_0 = +-1 é o sinal a lembrar; os demais passos são ruído distrator pequeno
    auto sample = [&](Vec& seq, int& y) { seq.resize(T); for (double& v : seq) v = 0.2 * N(rng); y = N(rng) > 0; seq[0] = y ? 1.0 : -1.0; };
    auto model = make(rng);
    const int BATCH = 16, STEPS = 3000;
    double ema = 0.7;
    for (int s = 1; s <= STEPS; ++s) {
        for (int b = 0; b < BATCH; ++b) { Vec q; int y; sample(q, y); ema = 0.99 * ema + 0.01 * model->loss_grad(q, y); }
        model->adam(0.01, BATCH);
    }
    int ok = 0, n = 1000;
    for (int i = 0; i < n; ++i) { Vec q; int y; sample(q, y); ok += (model->logit(q) > 0) == y; }
    printf("  %-5s T=%2d  loss(media movel)=%.3f  acc teste=%.3f\n", nome, T, ema, (double)ok / n);
}

int main() {
    mt19937 rng(0);
    cout << "Tarefa: lembrar o sinal de x_0 (acaso = 0.50)\n";
    for (int T : {5, 20, 40, 60}) {
        experimento("RNN", [](mt19937& r) { return unique_ptr<Model>(new RNN(16, r)); }, T, rng);
        experimento("LSTM", [](mt19937& r) { return unique_ptr<Model>(new LSTM(16, r)); }, T, rng);
    }
}
