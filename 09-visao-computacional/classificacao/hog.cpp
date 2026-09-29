// Classificação de imagens "clássica": extração de características HOG (Histogram of Oriented Gradients)
// do zero + k-NN, comparada a pixels brutos. Formas sintéticas (círculo, quadrado, triângulo) com deslocamentos.
// Build: g++ -std=c++17 -O2 hog.cpp -o hog
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
using namespace std;

using Img = vector<vector<double>>;
using Vec = vector<double>;

const int S = 24;

Img desenha(int cls, int jitter, mt19937& rng) {
    uniform_int_distribution<int> J(-jitter, jitter);
    uniform_real_distribution<double> R(4.5, 7.0), Br(0.4, 1.0);
    normal_distribution<double> N(0, 0.08);
    int cy = S / 2 + J(rng), cx = S / 2 + J(rng);
    double r = R(rng), br = Br(rng);
    Img im(S, Vec(S, 0));
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) {
            bool in = false;
            if (cls == 0) in = (i - cy) * (i - cy) + (j - cx) * (j - cx) <= r * r;                       // círculo
            else if (cls == 1) in = abs(i - cy) <= r * 0.85 && abs(j - cx) <= r * 0.85;                    // quadrado
            else in = i >= cy - r && i <= cy + r && abs(j - cx) <= (i - (cy - r)) * 0.5;                   // triângulo
            im[i][j] = (in ? br : 0.0) + N(rng);
        }
    return im;
}

// HOG: gradiente por diferenças centrais -> histograma de orientações (9 bins, 0-180°) por célula 6x6, ponderado pela magnitude
Vec hog(const Img& im) {
    const int CELL = 6, BINS = 9, NC = S / CELL;
    vector<Vec> hist(NC * NC, Vec(BINS, 0.0));
    for (int i = 1; i < S - 1; ++i)
        for (int j = 1; j < S - 1; ++j) {
            double gx = im[i][j + 1] - im[i][j - 1], gy = im[i + 1][j] - im[i - 1][j];
            double mag = sqrt(gx * gx + gy * gy), ang = atan2(gy, gx) * 180 / M_PI;
            if (ang < 0) ang += 180;
            if (ang >= 180) ang -= 180;
            hist[(i / CELL) * NC + j / CELL][min(BINS - 1, (int)(ang / (180.0 / BINS)))] += mag;
        }
    Vec f;
    double norm = 1e-6;
    for (auto& h : hist) for (double v : h) norm += v * v;
    norm = sqrt(norm);  // normalização global (versão simplificada da normalização por blocos): robusta a mudanças de brilho
    for (auto& h : hist) for (double v : h) f.push_back(v / norm);
    return f;
}

Vec pixels(const Img& im) { Vec f; for (auto& r : im) for (double v : r) f.push_back(v); return f; }

int knn(const vector<Vec>& X, const vector<int>& y, const Vec& q, int k = 5) {
    vector<pair<double, int>> d;
    for (size_t i = 0; i < X.size(); ++i) {
        double s = 0;
        for (size_t j = 0; j < q.size(); ++j) s += (X[i][j] - q[j]) * (X[i][j] - q[j]);
        d.push_back({s, y[i]});
    }
    partial_sort(d.begin(), d.begin() + k, d.end());
    int votos[3] = {0, 0, 0};
    for (int i = 0; i < k; ++i) ++votos[d[i].second];
    return max_element(votos, votos + 3) - votos;
}

int main() {
    mt19937 rng(0);
    vector<Vec> Xp, Xh; vector<int> y;
    for (int i = 0; i < 300; ++i) { int c = i % 3; Img im = desenha(c, 1, rng); Xp.push_back(pixels(im)); Xh.push_back(hog(im)); y.push_back(c); }
    printf("treino: 300 imagens (deslocamento de ate 1 px). Teste com deslocamentos maiores:\n");
    printf("  deslocamento | acc pixels brutos | acc HOG\n");
    for (int jit : {1, 3, 5}) {
        int okp = 0, okh = 0, n = 300;
        for (int i = 0; i < n; ++i) {
            int c = i % 3; Img im = desenha(c, jit, rng);
            okp += knn(Xp, y, pixels(im)) == c; okh += knn(Xh, y, hog(im)) == c;
        }
        printf("    +-%d px     |      %.3f        | %.3f\n", jit, (double)okp / n, (double)okh / n);
    }
    printf("HOG resume a DIRECAO das bordas por celula (local), ficando mais estavel a pequenas translacoes e a brilho do que pixels brutos.\n"
           "CNNs aprendem esses filtros automaticamente (e muito mais) — ver ../../05-redes-neurais/cnn/.\n");
}
