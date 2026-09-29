// Teoria da informação: entropia, entropia cruzada, KL, informação mútua.
// Build: g++ -std=c++17 -O2 teoria_informacao.cpp -o teoria_informacao
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

double entropia(const Vec& p) {  // bits
    double h = 0;
    for (double pi : p) if (pi > 0) h -= pi * log2(pi);
    return h;
}

double entropia_cruzada(const Vec& p, const Vec& q) {
    double h = 0;
    for (size_t i = 0; i < p.size(); ++i) if (p[i] > 0) h -= p[i] * log2(q[i]);
    return h;
}

double kl(const Vec& p, const Vec& q) { return entropia_cruzada(p, q) - entropia(p); }

// I(X;Y) = KL(p(x,y) || p(x)p(y))
double informacao_mutua(const Mat& pxy) {
    size_t r = pxy.size(), c = pxy[0].size();
    Vec px(r, 0), py(c, 0);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) { px[i] += pxy[i][j]; py[j] += pxy[i][j]; }
    double mi = 0;
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            if (pxy[i][j] > 0) mi += pxy[i][j] * log2(pxy[i][j] / (px[i] * py[j]));
    return mi;
}

int main() {
    cout << "H(moeda justa)     = " << entropia({0.5, 0.5}) << " bits\n";
    cout << "H(moeda viciada .9)= " << entropia({0.9, 0.1}) << " bits\n";
    cout << "H(dado justo)      = " << entropia(Vec(6, 1.0 / 6)) << " bits (log2 6 = " << log2(6.0) << ")\n";

    Vec p = {0.7, 0.2, 0.1}, q = {0.5, 0.3, 0.2};
    cout << "H(p,q) = " << entropia_cruzada(p, q) << "  H(p) = " << entropia(p) << "\n";
    cout << "KL(p||q) = " << kl(p, q) << "  KL(q||p) = " << kl(q, p) << " (assimetrica)\n";
    cout << "KL(p||p) = " << kl(p, p) << "\n";

    Mat indep = {{0.25, 0.25}, {0.25, 0.25}};  // X e Y independentes
    Mat dep = {{0.5, 0.0}, {0.0, 0.5}};         // Y = X
    cout << "I(X;Y) independentes = " << informacao_mutua(indep) << "\n";
    cout << "I(X;Y) Y=X           = " << informacao_mutua(dep) << " bit\n";
}
