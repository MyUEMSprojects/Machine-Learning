// Cálculo para ML: derivada numérica, gradiente, diferenciação automática (modo forward
// com números duais) e gradient checking.
// Build: g++ -std=c++17 -O2 calculo.cpp -o calculo
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>
using namespace std;

using Vec = vector<double>;

// Derivada numérica por diferença central: erro O(h^2).
double derivada(const function<double(double)>& f, double x, double h = 1e-5) {
    return (f(x + h) - f(x - h)) / (2 * h);
}

Vec gradiente_numerico(const function<double(const Vec&)>& f, Vec x, double h = 1e-5) {
    Vec g(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        double xi = x[i];
        x[i] = xi + h; double fp = f(x);
        x[i] = xi - h; double fm = f(x);
        x[i] = xi;
        g[i] = (fp - fm) / (2 * h);
    }
    return g;
}

// Números duais: a + b*eps, com eps^2 = 0. Propaga valor e derivada juntos.
struct Dual {
    double v, d;
    Dual(double v = 0, double d = 0) : v(v), d(d) {}
};
Dual operator+(Dual a, Dual b) { return {a.v + b.v, a.d + b.d}; }
Dual operator-(Dual a, Dual b) { return {a.v - b.v, a.d - b.d}; }
Dual operator*(Dual a, Dual b) { return {a.v * b.v, a.d * b.v + a.v * b.d}; }
Dual operator/(Dual a, Dual b) { return {a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)}; }
Dual sin(Dual a) { return {sin(a.v), cos(a.v) * a.d}; }
Dual exp(Dual a) { return {exp(a.v), exp(a.v) * a.d}; }
Dual log(Dual a) { return {log(a.v), a.d / a.v}; }

// f(x, y) = x^2 * y + sin(x*y)
template <class T>
T f(T x, T y) { return x * x * y + sin(x * y); }

int main() {
    // 1) derivada numérica vs analítica de sin(x) em x = 1
    cout << "d/dx sin(1): numerica=" << derivada([](double x) { return sin(x); }, 1.0)
         << " analitica=" << cos(1.0) << "\n";

    // 2) gradiente de f(x,y) por três vias
    double x = 1.5, y = -0.7;
    Vec gn = gradiente_numerico([](const Vec& p) { return f<double>(p[0], p[1]); }, {x, y});
    double dfdx = f(Dual(x, 1), Dual(y, 0)).d;  // semente d/dx
    double dfdy = f(Dual(x, 0), Dual(y, 1)).d;  // semente d/dy
    double ax = 2 * x * y + y * cos(x * y);      // analítico
    double ay = x * x + x * cos(x * y);
    cout << "df/dx: numerico=" << gn[0] << " autodiff=" << dfdx << " analitico=" << ax << "\n";
    cout << "df/dy: numerico=" << gn[1] << " autodiff=" << dfdy << " analitico=" << ay << "\n";

    // 3) regra da cadeia via autodiff: sigmoide(w*x) derivada em w
    Dual w(0.8, 1), xin(2.0, 0);
    Dual z = w * xin;
    Dual s = Dual(1) / (Dual(1) + exp(Dual(0) - z));
    double sv = 1 / (1 + exp(-1.6));
    cout << "d sigmoid(w*x)/dw: autodiff=" << s.d << " analitico=" << sv * (1 - sv) * 2.0 << "\n";
}
