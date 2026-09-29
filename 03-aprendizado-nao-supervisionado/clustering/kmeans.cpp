// k-means (algoritmo de Lloyd) com inicialização k-means++ e múltiplas reinicializações.
// Também varre k e imprime a inércia (método do cotovelo) e a silhueta média.
// Build: g++ -std=c++17 -O2 kmeans.cpp -o kmeans
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

double dist2(const Vec& a, const Vec& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return s;
}

struct Result { Mat centers; vector<int> labels; double inertia; };

// k-means++: primeiro centro aleatório; próximos sorteados com prob. ∝ distância² ao centro mais próximo.
Mat init_pp(const Mat& X, int k, mt19937& rng) {
    Mat C;
    C.push_back(X[uniform_int_distribution<int>(0, X.size() - 1)(rng)]);
    while ((int)C.size() < k) {
        Vec d(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            d[i] = numeric_limits<double>::max();
            for (auto& c : C) d[i] = min(d[i], dist2(X[i], c));
        }
        discrete_distribution<int> pick(d.begin(), d.end());
        C.push_back(X[pick(rng)]);
    }
    return C;
}

Result kmeans_once(const Mat& X, int k, mt19937& rng, int max_iter = 300) {
    int n = X.size(), d = X[0].size();
    Result r; r.centers = init_pp(X, k, rng); r.labels.assign(n, -1);
    for (int it = 0; it < max_iter; ++it) {
        bool changed = false;
        for (int i = 0; i < n; ++i) {  // passo E: atribui ao centro mais próximo
            int best = 0; double bd = dist2(X[i], r.centers[0]);
            for (int c = 1; c < k; ++c) { double dd = dist2(X[i], r.centers[c]); if (dd < bd) { bd = dd; best = c; } }
            if (best != r.labels[i]) { r.labels[i] = best; changed = true; }
        }
        if (!changed) break;
        Mat S(k, Vec(d, 0.0)); vector<int> cnt(k, 0);  // passo M: centro = média do cluster
        for (int i = 0; i < n; ++i) { ++cnt[r.labels[i]]; for (int j = 0; j < d; ++j) S[r.labels[i]][j] += X[i][j]; }
        for (int c = 0; c < k; ++c) if (cnt[c]) for (int j = 0; j < d; ++j) r.centers[c][j] = S[c][j] / cnt[c];
    }
    r.inertia = 0;
    for (int i = 0; i < n; ++i) r.inertia += dist2(X[i], r.centers[r.labels[i]]);
    return r;
}

Result kmeans(const Mat& X, int k, mt19937& rng, int n_init = 10) {
    Result best = kmeans_once(X, k, rng);
    for (int i = 1; i < n_init; ++i) { Result r = kmeans_once(X, k, rng); if (r.inertia < best.inertia) best = r; }
    return best;
}

// Silhueta média: s = (b - a) / max(a, b); a = dist média intra-cluster, b = dist média ao cluster vizinho mais próximo
double silhouette(const Mat& X, const vector<int>& lab, int k) {
    int n = X.size(); double tot = 0;
    for (int i = 0; i < n; ++i) {
        Vec sum(k, 0.0); vector<int> cnt(k, 0);
        for (int j = 0; j < n; ++j) if (j != i) { sum[lab[j]] += sqrt(dist2(X[i], X[j])); ++cnt[lab[j]]; }
        double a = cnt[lab[i]] ? sum[lab[i]] / cnt[lab[i]] : 0, b = 1e18;
        for (int c = 0; c < k; ++c) if (c != lab[i] && cnt[c]) b = min(b, sum[c] / cnt[c]);
        tot += (b - a) / max(a, b);
    }
    return tot / n;
}

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 0.7);
    vector<Vec> centros = {{0, 0}, {5, 0}, {2.5, 4.5}, {-3, 4}};  // 4 grupos verdadeiros
    Mat X;
    for (int i = 0; i < 400; ++i) { auto& c = centros[i % 4]; X.push_back({c[0] + N(rng), c[1] + N(rng)}); }

    cout << " k | inercia | silhueta\n";
    for (int k = 2; k <= 8; ++k) {
        Result r = kmeans(X, k, rng);
        printf("%2d | %7.1f |  %.3f\n", k, r.inertia, silhouette(X, r.labels, k));
    }
    Result r = kmeans(X, 4, rng);
    cout << "\nCentros encontrados com k=4 (verdadeiros: (0,0) (5,0) (2.5,4.5) (-3,4)):\n";
    for (auto& c : r.centers) printf("  (%.2f, %.2f)\n", c[0], c[1]);
}
