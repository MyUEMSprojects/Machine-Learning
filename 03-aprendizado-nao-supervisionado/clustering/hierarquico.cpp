// Clustering hierárquico aglomerativo (bottom-up) com ligações single, complete e average.
// Imprime a sequência de fusões (dendrograma em texto) e corta em k clusters.
// Build: g++ -std=c++17 -O2 hierarquico.cpp -o hierarquico
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

double dist(const Vec& a, const Vec& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(s);
}

struct Merge { int a, b; double h; int size; };

// linkage: "single" (min), "complete" (max), "average" (média)
vector<Merge> agglomerative(const Mat& X, const string& linkage) {
    int n = X.size();
    Mat D(n, Vec(n));
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) D[i][j] = dist(X[i], X[j]);
    vector<bool> alive(n, true);
    vector<int> sz(n, 1), id(n);
    for (int i = 0; i < n; ++i) id[i] = i;  // id do cluster que ocupa a linha i
    vector<Merge> merges;
    for (int step = 0; step < n - 1; ++step) {
        int bi = -1, bj = -1; double bd = 1e18;
        for (int i = 0; i < n; ++i) if (alive[i])
            for (int j = i + 1; j < n; ++j) if (alive[j] && D[i][j] < bd) { bd = D[i][j]; bi = i; bj = j; }
        merges.push_back({id[bi], id[bj], bd, sz[bi] + sz[bj]});
        for (int k = 0; k < n; ++k) if (alive[k] && k != bi && k != bj) {  // fórmula de Lance–Williams
            double d;
            if (linkage == "single") d = min(D[bi][k], D[bj][k]);
            else if (linkage == "complete") d = max(D[bi][k], D[bj][k]);
            else d = (sz[bi] * D[bi][k] + sz[bj] * D[bj][k]) / (sz[bi] + sz[bj]);
            D[bi][k] = D[k][bi] = d;
        }
        sz[bi] += sz[bj]; alive[bj] = false; id[bi] = n + step;  // novo cluster
    }
    return merges;
}

// Corta o dendrograma deixando k clusters: aplica as primeiras n-k fusões (union-find).
vector<int> cut(int n, const vector<Merge>& m, int k) {
    vector<int> parent(2 * n);
    for (int i = 0; i < 2 * n; ++i) parent[i] = i;
    auto find = [&](int x) { while (parent[x] != x) x = parent[x] = parent[parent[x]]; return x; };
    for (int s = 0; s < n - k; ++s) { parent[find(m[s].a)] = n + s; parent[find(m[s].b)] = n + s; }
    vector<int> lab(n), map_(2 * n, -1); int nxt = 0;
    for (int i = 0; i < n; ++i) { int r = find(i); if (map_[r] < 0) map_[r] = nxt++; lab[i] = map_[r]; }
    return lab;
}

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 0.5);
    vector<Vec> c = {{0, 0}, {4, 0}, {2, 3.5}};
    Mat X;
    for (int i = 0; i < 60; ++i) X.push_back({c[i % 3][0] + N(rng), c[i % 3][1] + N(rng)});

    for (string link : {"single", "complete", "average"}) {
        auto m = agglomerative(X, link);
        auto lab = cut(X.size(), m, 3);
        vector<int> cnt(3, 0);
        for (int l : lab) ++cnt[l];
        printf("%-8s: tamanhos dos 3 clusters = %d %d %d | altura das 3 ultimas fusoes = %.2f %.2f %.2f\n", link.c_str(),
               cnt[0], cnt[1], cnt[2], m[m.size() - 3].h, m[m.size() - 2].h, m[m.size() - 1].h);
    }
    cout << "(alturas altas nas ultimas fusoes indicam onde cortar o dendrograma)\n";
}
