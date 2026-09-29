// Álgebra linear do zero: matmul, transposta, solução de Ax=b (LU com pivoteamento),
// autovalor dominante (power iteration) e autovalores de matriz simétrica (Jacobi).
// Build: g++ -std=c++17 -O2 algebra_linear.cpp -o algebra_linear
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

Mat transpose(const Mat& A) {
    Mat T(A[0].size(), Vec(A.size()));
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j) T[j][i] = A[i][j];
    return T;
}

Mat matmul(const Mat& A, const Mat& B) {
    Mat C(A.size(), Vec(B[0].size(), 0.0));
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t k = 0; k < B.size(); ++k)
            for (size_t j = 0; j < B[0].size(); ++j) C[i][j] += A[i][k] * B[k][j];
    return C;
}

Vec matvec(const Mat& A, const Vec& x) {
    Vec y(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
    return y;
}

double dot(const Vec& a, const Vec& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

double norm(const Vec& a) { return sqrt(dot(a, a)); }

// Resolve Ax = b por eliminação de Gauss com pivoteamento parcial (equivale a LU).
Vec solve(Mat A, Vec b) {
    size_t n = A.size();
    for (size_t c = 0; c < n; ++c) {
        size_t p = c;
        for (size_t r = c + 1; r < n; ++r)
            if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
        if (fabs(A[p][c]) < 1e-12) throw runtime_error("matriz singular");
        swap(A[p], A[c]);
        swap(b[p], b[c]);
        for (size_t r = c + 1; r < n; ++r) {
            double f = A[r][c] / A[c][c];
            for (size_t k = c; k < n; ++k) A[r][k] -= f * A[c][k];
            b[r] -= f * b[c];
        }
    }
    Vec x(n);
    for (int i = (int)n - 1; i >= 0; --i) {
        double s = b[i];
        for (size_t j = i + 1; j < n; ++j) s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
    return x;
}

// Autovalor de maior módulo e seu autovetor.
pair<double, Vec> power_iteration(const Mat& A, int iters = 1000) {
    Vec v(A.size(), 1.0);
    double lambda = 0;
    for (int it = 0; it < iters; ++it) {
        Vec w = matvec(A, v);
        double n = norm(w);
        for (auto& x : w) x /= n;
        v = w;
        lambda = dot(v, matvec(A, v));
    }
    return {lambda, v};
}

// Autovalores de matriz simétrica pelo método de Jacobi (rotações).
Vec jacobi_eigenvalues(Mat A, int sweeps = 100) {
    size_t n = A.size();
    for (int s = 0; s < sweeps; ++s) {
        double off = 0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j) off += A[i][j] * A[i][j];
        if (off < 1e-20) break;
        for (size_t p = 0; p < n; ++p)
            for (size_t q = p + 1; q < n; ++q) {
                if (fabs(A[p][q]) < 1e-15) continue;
                double theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
                double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1));
                double c = 1 / sqrt(t * t + 1), sn = t * c;
                for (size_t k = 0; k < n; ++k) {  // A <- A * J
                    double akp = A[k][p], akq = A[k][q];
                    A[k][p] = c * akp - sn * akq;
                    A[k][q] = sn * akp + c * akq;
                }
                for (size_t k = 0; k < n; ++k) {  // A <- J^T * A
                    double apk = A[p][k], aqk = A[q][k];
                    A[p][k] = c * apk - sn * aqk;
                    A[q][k] = sn * apk + c * aqk;
                }
            }
    }
    Vec ev(n);
    for (size_t i = 0; i < n; ++i) ev[i] = A[i][i];
    return ev;
}

int main() {
    Mat A = {{4, 1, 2}, {1, 3, 0}, {2, 0, 5}};
    Vec b = {1, 2, 3};

    Vec x = solve(A, b);
    Vec r = matvec(A, x);
    cout << "x = [" << x[0] << ", " << x[1] << ", " << x[2] << "]\n";
    cout << "A*x = [" << r[0] << ", " << r[1] << ", " << r[2] << "] (deve ser [1,2,3])\n";

    auto [lam, v] = power_iteration(A);
    cout << "autovalor dominante (power iteration) = " << lam << "\n";

    Vec ev = jacobi_eigenvalues(A);
    cout << "autovalores (Jacobi) = ";
    for (double e : ev) cout << e << " ";
    cout << "\n";
    // traço = soma dos autovalores
    cout << "traco = " << A[0][0] + A[1][1] + A[2][2] << ", soma autovalores = " << ev[0] + ev[1] + ev[2] << "\n";
}
