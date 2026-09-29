// Otimização: gradiente descendente, momentum, Adam e método de Newton
// na função de Rosenbrock f(x,y) = (1-x)^2 + 100 (y-x^2)^2 (mínimo em (1,1)).
// Build: g++ -std=c++17 -O2 otimizacao.cpp -o otimizacao
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

using Vec = vector<double>;

double f(const Vec& p) {
    double x = p[0], y = p[1];
    return (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
}
Vec grad(const Vec& p) {
    double x = p[0], y = p[1];
    return {-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)};
}

Vec gd(Vec p, double lr, int iters) {
    for (int i = 0; i < iters; ++i) {
        Vec g = grad(p);
        for (int k = 0; k < 2; ++k) p[k] -= lr * g[k];
    }
    return p;
}

Vec momentum(Vec p, double lr, double beta, int iters) {
    Vec v(2, 0.0);
    for (int i = 0; i < iters; ++i) {
        Vec g = grad(p);
        for (int k = 0; k < 2; ++k) {
            v[k] = beta * v[k] + g[k];
            p[k] -= lr * v[k];
        }
    }
    return p;
}

Vec adam(Vec p, double lr, int iters, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8) {
    Vec m(2, 0.0), v(2, 0.0);
    for (int t = 1; t <= iters; ++t) {
        Vec g = grad(p);
        for (int k = 0; k < 2; ++k) {
            m[k] = b1 * m[k] + (1 - b1) * g[k];
            v[k] = b2 * v[k] + (1 - b2) * g[k] * g[k];
            double mh = m[k] / (1 - pow(b1, t)), vh = v[k] / (1 - pow(b2, t));
            p[k] -= lr * mh / (sqrt(vh) + eps);
        }
    }
    return p;
}

// Newton: p <- p - H^{-1} g (usa Hessiana analítica 2x2)
Vec newton(Vec p, int iters) {
    for (int i = 0; i < iters; ++i) {
        double x = p[0], y = p[1];
        Vec g = grad(p);
        double h11 = 2 - 400 * (y - x * x) + 800 * x * x, h12 = -400 * x, h22 = 200;
        double det = h11 * h22 - h12 * h12;
        p[0] -= (h22 * g[0] - h12 * g[1]) / det;
        p[1] -= (-h12 * g[0] + h11 * g[1]) / det;
    }
    return p;
}

void mostra(const char* nome, const Vec& p) {
    cout << nome << ": (" << p[0] << ", " << p[1] << ")  f=" << f(p) << "\n";
}

int main() {
    Vec start = {-1.2, 1.0};
    mostra("GD (50k it)      ", gd(start, 1e-3, 50000));
    mostra("Momentum (50k it)", momentum(start, 1e-4, 0.9, 50000));
    mostra("Adam (20k it)    ", adam(start, 0.01, 20000));
    mostra("Newton (20 it)   ", newton(start, 20));
}
