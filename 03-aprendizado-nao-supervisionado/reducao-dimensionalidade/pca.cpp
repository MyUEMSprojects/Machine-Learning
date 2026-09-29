// PCA do zero: centralizar -> matriz de covariância -> autovetores (Jacobi) -> projetar.
// Dados: 6 features geradas a partir de apenas 2 fatores latentes + ruído.
// Build: g++ -std=c++17 -O2 pca.cpp -o pca
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

// Autovalores e autovetores (colunas de V) de matriz simétrica por rotações de Jacobi.
void jacobi(Mat A, Vec& eval, Mat& V, int sweeps = 100) {
    int n = A.size();
    V.assign(n, Vec(n, 0.0));
    for (int i = 0; i < n; ++i) V[i][i] = 1;
    for (int s = 0; s < sweeps; ++s) {
        double off = 0;
        for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) off += A[i][j] * A[i][j];
        if (off < 1e-22) break;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) {
                if (fabs(A[p][q]) < 1e-16) continue;
                double th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
                double t = (th >= 0 ? 1.0 : -1.0) / (fabs(th) + sqrt(th * th + 1));
                double c = 1 / sqrt(t * t + 1), sn = t * c;
                for (int k = 0; k < n; ++k) {
                    double akp = A[k][p], akq = A[k][q];
                    A[k][p] = c * akp - sn * akq; A[k][q] = sn * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    double apk = A[p][k], aqk = A[q][k];
                    A[p][k] = c * apk - sn * aqk; A[q][k] = sn * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - sn * vkq; V[k][q] = sn * vkp + c * vkq;
                }
            }
    }
    eval.resize(n);
    for (int i = 0; i < n; ++i) eval[i] = A[i][i];
}

struct PCA {
    Vec mean, eval; Mat comps;  // comps[k] = k-ésimo componente principal (ordenado por variância)
    void fit(const Mat& X) {
        int n = X.size(), d = X[0].size();
        mean.assign(d, 0);
        for (auto& r : X) for (int j = 0; j < d; ++j) mean[j] += r[j] / n;
        Mat C(d, Vec(d, 0));
        for (auto& r : X) for (int p = 0; p < d; ++p) for (int q = 0; q < d; ++q) C[p][q] += (r[p] - mean[p]) * (r[q] - mean[q]) / (n - 1);
        Vec ev; Mat V; jacobi(C, ev, V);
        vector<int> ord(d);
        iota(ord.begin(), ord.end(), 0);
        sort(ord.begin(), ord.end(), [&](int a, int b) { return ev[a] > ev[b]; });
        eval.clear(); comps.clear();
        for (int k : ord) { eval.push_back(ev[k]); Vec c(d); for (int j = 0; j < d; ++j) c[j] = V[j][k]; comps.push_back(c); }
    }
    Vec transform(const Vec& x, int k) const {
        Vec z(k, 0);
        for (int i = 0; i < k; ++i) for (size_t j = 0; j < x.size(); ++j) z[i] += comps[i][j] * (x[j] - mean[j]);
        return z;
    }
    Vec inverse(const Vec& z) const {
        Vec x = mean;
        for (size_t i = 0; i < z.size(); ++i) for (size_t j = 0; j < x.size(); ++j) x[j] += z[i] * comps[i][j];
        return x;
    }
};

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 1);
    int n = 500, d = 6;
    Mat W = {{2, 0.5, 1, -1, 0.3, 0.8}, {0.2, 1.5, -1, 0.7, 1, -0.5}};  // 2 fatores latentes -> 6 features
    Mat X;
    for (int i = 0; i < n; ++i) {
        double z1 = N(rng) * 2, z2 = N(rng);
        Vec x(d);
        for (int j = 0; j < d; ++j) x[j] = z1 * W[0][j] + z2 * W[1][j] + 0.1 * N(rng);
        X.push_back(x);
    }
    PCA pca; pca.fit(X);
    double tot = accumulate(pca.eval.begin(), pca.eval.end(), 0.0), acum = 0;
    cout << "componente | autovalor | variancia explicada | acumulada\n";
    for (int k = 0; k < d; ++k) { acum += pca.eval[k]; printf("    %d      |  %7.3f  |       %5.1f%%        |  %5.1f%%\n", k + 1, pca.eval[k], 100 * pca.eval[k] / tot, 100 * acum / tot); }
    for (int k : {1, 2, 3}) {
        double err = 0;
        for (auto& x : X) { Vec r = pca.inverse(pca.transform(x, k)); for (int j = 0; j < d; ++j) err += pow(x[j] - r[j], 2); }
        printf("erro de reconstrucao com %d componente(s): %.4f (por amostra)\n", k, err / n);
    }
}
