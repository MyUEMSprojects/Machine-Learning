// CNN do zero: convolução 2D, ReLU, max-pooling e camada densa com softmax, treinada com SGD.
// Tarefa: imagens 8x8 com uma linha horizontal / vertical / diagonal em posição aleatória + ruído.
// Também mostra um filtro de Sobel (detecção de bordas) aplicado à mão.
// Build: g++ -std=c++17 -O2 cnn.cpp -o cnn
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

using Img = vector<vector<double>>;
using Vec = vector<double>;

const int S = 8, KS = 3, F = 4, C = 3;       // imagem SxS, filtro KSxKS, F filtros, C classes
const int CO = S - KS + 1, PO = CO / 2;      // saída da conv (6) e do pooling (3)
const int FLAT = F * PO * PO;

struct CNN {
    double W[F][KS][KS], b[F], D[C][FLAT], bd[C];
    // caches do forward
    double conv[F][CO][CO], act[F][CO][CO], pool[F][PO][PO]; int arg[F][PO][PO][2]; Vec flat, prob;

    CNN(mt19937& rng) {
        normal_distribution<double> N(0, 1);
        for (int f = 0; f < F; ++f) { b[f] = 0; for (int u = 0; u < KS; ++u) for (int v = 0; v < KS; ++v) W[f][u][v] = N(rng) * sqrt(2.0 / (KS * KS)); }
        for (int c = 0; c < C; ++c) { bd[c] = 0; for (int j = 0; j < FLAT; ++j) D[c][j] = N(rng) * sqrt(2.0 / FLAT); }
    }
    Vec forward(const Img& x) {
        for (int f = 0; f < F; ++f)
            for (int i = 0; i < CO; ++i) for (int j = 0; j < CO; ++j) {
                double s = b[f];  // correlação cruzada: mesmo filtro deslizando (compartilhamento de pesos)
                for (int u = 0; u < KS; ++u) for (int v = 0; v < KS; ++v) s += W[f][u][v] * x[i + u][j + v];
                conv[f][i][j] = s; act[f][i][j] = max(0.0, s);  // ReLU
            }
        flat.assign(FLAT, 0);
        for (int f = 0; f < F; ++f)
            for (int i = 0; i < PO; ++i) for (int j = 0; j < PO; ++j) {  // max-pool 2x2
                double best = -1e18;
                for (int u = 0; u < 2; ++u) for (int v = 0; v < 2; ++v)
                    if (act[f][2 * i + u][2 * j + v] > best) { best = act[f][2 * i + u][2 * j + v]; arg[f][i][j][0] = 2 * i + u; arg[f][i][j][1] = 2 * j + v; }
                pool[f][i][j] = best; flat[(f * PO + i) * PO + j] = best;
            }
        Vec z(C); double mx = -1e18;
        for (int c = 0; c < C; ++c) { z[c] = bd[c]; for (int j = 0; j < FLAT; ++j) z[c] += D[c][j] * flat[j]; mx = max(mx, z[c]); }
        double s = 0;
        for (double& v : z) { v = exp(v - mx); s += v; }
        for (double& v : z) v /= s;
        return prob = z;
    }
    // Um passo de SGD numa amostra (retorna a perda)
    double train_step(const Img& x, int y, double lr) {
        forward(x);
        double loss = -log(prob[y] + 1e-12);
        Vec dz = prob; dz[y] -= 1;                 // softmax + entropia cruzada
        Vec dflat(FLAT, 0);
        for (int c = 0; c < C; ++c) { for (int j = 0; j < FLAT; ++j) { dflat[j] += D[c][j] * dz[c]; D[c][j] -= lr * dz[c] * flat[j]; } bd[c] -= lr * dz[c]; }
        double dact[F][CO][CO] = {};               // pooling: gradiente vai só para o máximo
        for (int f = 0; f < F; ++f) for (int i = 0; i < PO; ++i) for (int j = 0; j < PO; ++j)
            dact[f][arg[f][i][j][0]][arg[f][i][j][1]] += dflat[(f * PO + i) * PO + j];
        double dW[F][KS][KS] = {}, db[F] = {};
        for (int f = 0; f < F; ++f) for (int i = 0; i < CO; ++i) for (int j = 0; j < CO; ++j) {
            double d = conv[f][i][j] > 0 ? dact[f][i][j] : 0;  // ReLU
            db[f] += d;
            for (int u = 0; u < KS; ++u) for (int v = 0; v < KS; ++v) dW[f][u][v] += d * x[i + u][j + v];
        }
        for (int f = 0; f < F; ++f) { b[f] -= lr * db[f]; for (int u = 0; u < KS; ++u) for (int v = 0; v < KS; ++v) W[f][u][v] -= lr * dW[f][u][v]; }
        return loss;
    }
    int predict(const Img& x) { Vec p = forward(x); return max_element(p.begin(), p.end()) - p.begin(); }
};

Img make_image(int cls, mt19937& rng) {
    Img im(S, Vec(S, 0));
    normal_distribution<double> N(0, 0.2);
    int pos = uniform_int_distribution<int>(0, S - 1)(rng);
    for (int k = 0; k < S; ++k) {
        if (cls == 0) im[pos][k] = 1;                  // horizontal
        else if (cls == 1) im[k][pos] = 1;             // vertical
        else im[k][(k + pos) % S] = 1;                 // diagonal (com wrap)
    }
    for (auto& r : im) for (double& v : r) v += N(rng);
    return im;
}

int main() {
    // Filtro de Sobel vertical numa imagem com uma borda: resposta forte só na borda
    Img im(6, Vec(6, 0));
    for (int i = 0; i < 6; ++i) for (int j = 3; j < 6; ++j) im[i][j] = 1;
    double sobel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    cout << "Sobel vertical (saida 4x4) - a borda vertical em j=3 se destaca:\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) { double s = 0; for (int u = 0; u < 3; ++u) for (int v = 0; v < 3; ++v) s += sobel[u][v] * im[i + u][j + v]; printf("%5.1f ", s); }
        printf("\n");
    }

    mt19937 rng(0);
    CNN net(rng);
    vector<Img> Xtr, Xte; vector<int> ytr, yte;
    for (int i = 0; i < 600; ++i) { Xtr.push_back(make_image(i % 3, rng)); ytr.push_back(i % 3); }
    for (int i = 0; i < 300; ++i) { Xte.push_back(make_image(i % 3, rng)); yte.push_back(i % 3); }
    vector<int> idx(Xtr.size());
    iota(idx.begin(), idx.end(), 0);
    cout << "\nTreinando CNN (" << F << " filtros " << KS << "x" << KS << ", maxpool 2x2, densa->softmax):\n";
    for (int ep = 1; ep <= 30; ++ep) {
        shuffle(idx.begin(), idx.end(), rng);
        double L = 0;
        for (int i : idx) L += net.train_step(Xtr[i], ytr[i], 0.02);
        if (ep % 5 == 0 || ep == 1) {
            int ok = 0;
            for (size_t i = 0; i < Xte.size(); ++i) ok += net.predict(Xte[i]) == yte[i];
            printf("  epoca %2d  loss=%.4f  acc teste=%.3f\n", ep, L / Xtr.size(), (double)ok / Xte.size());
        }
    }
    printf("parametros: conv=%d + densa=%d = %d  (uma MLP 64->%d->3 teria %d)\n", F * KS * KS + F, C * FLAT + C, F * KS * KS + F + C * FLAT + C, 36, 64 * 36 + 36 + 36 * 3 + 3);
}
