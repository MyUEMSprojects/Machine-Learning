// DBSCAN: clustering por densidade. Acha formas arbitrárias (duas luas) e marca ruído (-1),
// onde o k-means falha.
// Build: g++ -std=c++17 -O2 dbscan.cpp -o dbscan
#include <cmath>
#include <iostream>
#include <queue>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

double dist(const Vec& a, const Vec& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(s);
}

vector<int> neighbors(const Mat& X, int i, double eps) {
    vector<int> r;
    for (size_t j = 0; j < X.size(); ++j) if (dist(X[i], X[j]) <= eps) r.push_back(j);
    return r;
}

// Retorna rótulos: -1 = ruído, 0.. = cluster
vector<int> dbscan(const Mat& X, double eps, int min_pts) {
    int n = X.size(), cluster = 0;
    vector<int> label(n, -2);  // -2 = não visitado
    for (int i = 0; i < n; ++i) {
        if (label[i] != -2) continue;
        auto nb = neighbors(X, i, eps);
        if ((int)nb.size() < min_pts) { label[i] = -1; continue; }  // não é ponto núcleo (por ora, ruído)
        label[i] = cluster;
        queue<int> q;
        for (int j : nb) q.push(j);
        while (!q.empty()) {  // expande o cluster por alcançabilidade de densidade
            int j = q.front(); q.pop();
            if (label[j] == -1) label[j] = cluster;  // ruído vira ponto de borda
            if (label[j] != -2) continue;
            label[j] = cluster;
            auto nb2 = neighbors(X, j, eps);
            if ((int)nb2.size() >= min_pts) for (int m : nb2) q.push(m);  // j também é núcleo
        }
        ++cluster;
    }
    return label;
}

int main() {
    mt19937 rng(0);
    uniform_real_distribution<double> U(0, M_PI), UB(-1.5, 2.5), UY(-1.5, 2);
    normal_distribution<double> N(0, 0.08);
    Mat X; vector<int> verdade;
    for (int i = 0; i < 300; ++i) {  // duas luas
        double t = U(rng);
        if (i % 2 == 0) { X.push_back({cos(t) + N(rng), sin(t) + N(rng)}); verdade.push_back(0); }
        else { X.push_back({1 - cos(t) + N(rng), 0.5 - sin(t) + N(rng)}); verdade.push_back(1); }
    }
    for (int i = 0; i < 15; ++i) { X.push_back({UB(rng), UY(rng)}); verdade.push_back(-1); }  // ruído uniforme

    auto lab = dbscan(X, 0.2, 5);
    int nc = 0, noise = 0;
    for (int l : lab) { nc = max(nc, l + 1); noise += l == -1; }
    printf("DBSCAN(eps=0.2, min_pts=5): %d clusters, %d pontos de ruido\n", nc, noise);
    // pureza: fração de pontos das luas cujo cluster é majoritariamente da mesma lua
    int cont[2][8] = {};
    for (size_t i = 0; i < X.size(); ++i) if (verdade[i] >= 0 && lab[i] >= 0 && lab[i] < 8) ++cont[verdade[i]][lab[i]];
    for (int m = 0; m < 2; ++m) { printf("lua %d ->", m); for (int c = 0; c < nc && c < 8; ++c) printf(" cluster%d:%d", c, cont[m][c]); printf("\n"); }
}
