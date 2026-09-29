// Validação cruzada k-fold para escolher o grau de um polinômio.
// Erro de treino sempre cai com o grau; o erro de CV revela o overfitting.
// Build: g++ -std=c++17 -O2 validacao_cruzada.cpp -o validacao_cruzada
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

Vec solve(Mat A, Vec b) {
    int n = A.size();
    for (int c = 0; c < n; ++c) {
        int p = c;
        for (int r = c + 1; r < n; ++r) if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
        swap(A[p], A[c]); swap(b[p], b[c]);
        for (int r = c + 1; r < n; ++r) {
            double f = A[r][c] / A[c][c];
            for (int k = c; k < n; ++k) A[r][k] -= f * A[c][k];
            b[r] -= f * b[c];
        }
    }
    Vec x(n);
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
    return x;
}

Vec polyfit(const Vec& x, const Vec& y, int d) {
    int m = d + 1;
    Mat A(m, Vec(m, 0)); Vec b(m, 0);
    for (size_t i = 0; i < x.size(); ++i)
        for (int p = 0; p < m; ++p) {
            b[p] += pow(x[i], p) * y[i];
            for (int q = 0; q < m; ++q) A[p][q] += pow(x[i], p + q);
        }
    for (int p = 0; p < m; ++p) A[p][p] += 1e-8;
    return solve(A, b);
}

double polyval(const Vec& w, double x) {
    double s = 0;
    for (size_t p = 0; p < w.size(); ++p) s += w[p] * pow(x, p);
    return s;
}

double mse_poly(const Vec& w, const Vec& x, const Vec& y) {
    double s = 0;
    for (size_t i = 0; i < x.size(); ++i) s += pow(polyval(w, x[i]) - y[i], 2);
    return s / x.size();
}

// Retorna o erro médio de validação em k folds.
double kfold_cv(const Vec& x, const Vec& y, int d, int k, mt19937& rng) {
    int n = x.size();
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    shuffle(idx.begin(), idx.end(), rng);
    double total = 0;
    for (int f = 0; f < k; ++f) {
        Vec xtr, ytr, xva, yva;
        for (int i = 0; i < n; ++i) {
            int j = idx[i];
            if (i % k == f) { xva.push_back(x[j]); yva.push_back(y[j]); }
            else { xtr.push_back(x[j]); ytr.push_back(y[j]); }
        }
        total += mse_poly(polyfit(xtr, ytr, d), xva, yva);
    }
    return total / k;
}

int main() {
    mt19937 rng(7);
    uniform_real_distribution<double> U(0, 1);
    normal_distribution<double> ruido(0, 0.25);
    int n = 40;
    Vec x(n), y(n);
    for (int i = 0; i < n; ++i) { x[i] = U(rng); y[i] = sin(2 * M_PI * x[i]) + ruido(rng); }

    cout << "grau | erro treino | erro CV (5-fold)\n";
    int melhor = 0; double melhor_cv = 1e18;
    for (int d = 0; d <= 10; ++d) {
        double tr = mse_poly(polyfit(x, y, d), x, y);
        double cv = kfold_cv(x, y, d, 5, rng);
        printf(" %2d  |   %.4f    |   %.4f\n", d, tr, cv);
        if (cv < melhor_cv) { melhor_cv = cv; melhor = d; }
    }
    cout << "Melhor grau segundo a CV: " << melhor << "\n";
}
