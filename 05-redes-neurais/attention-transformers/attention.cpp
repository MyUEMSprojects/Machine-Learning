// Atenção e bloco Transformer (só forward) do zero, com verificações numéricas das propriedades:
//  1) atenção = busca "suave" em dicionário; 2) máscara causal; 3) self-attention sem posição é
//  equivariante a permutações; 4) codificação posicional sinusoidal quebra essa simetria.
// Build: g++ -std=c++17 -O2 attention.cpp -o attention
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

Mat matmul(const Mat& A, const Mat& B) {
    Mat C(A.size(), Vec(B[0].size(), 0));
    for (size_t i = 0; i < A.size(); ++i) for (size_t k = 0; k < B.size(); ++k) for (size_t j = 0; j < B[0].size(); ++j) C[i][j] += A[i][k] * B[k][j];
    return C;
}
Mat transpose(const Mat& A) {
    Mat T(A[0].size(), Vec(A.size()));
    for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) T[j][i] = A[i][j];
    return T;
}
void softmax_rows(Mat& S) {
    for (auto& r : S) {
        double mx = *max_element(r.begin(), r.end()), s = 0;
        for (double& v : r) { v = exp(v - mx); s += v; }
        for (double& v : r) v /= s;
    }
}

// Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k) [+ máscara]) V ; devolve também os pesos
Mat attention(const Mat& Q, const Mat& K, const Mat& V, bool causal, Mat* weights = nullptr) {
    Mat S = matmul(Q, transpose(K));
    double scale = 1.0 / sqrt((double)K[0].size());
    for (size_t i = 0; i < S.size(); ++i)
        for (size_t j = 0; j < S[0].size(); ++j) { S[i][j] *= scale; if (causal && j > i) S[i][j] = -1e30; }
    softmax_rows(S);
    if (weights) *weights = S;
    return matmul(S, V);
}

// Multi-head: projeta em h subespaços, atende em cada um, concatena e projeta de volta
struct MHA {
    int d, h;
    Mat Wq, Wk, Wv, Wo;
    MHA(int d_, int h_, mt19937& rng) : d(d_), h(h_) {
        normal_distribution<double> N(0, 1.0 / sqrt(d_));
        auto init = [&]() { Mat M(d, Vec(d)); for (auto& r : M) for (double& v : r) v = N(rng); return M; };
        Wq = init(); Wk = init(); Wv = init(); Wo = init();
    }
    Mat operator()(const Mat& X, bool causal) const {
        Mat Q = matmul(X, Wq), K = matmul(X, Wk), V = matmul(X, Wv);
        int dk = d / h, n = X.size();
        Mat concat(n, Vec(d));
        for (int hd = 0; hd < h; ++hd) {
            auto slice = [&](const Mat& M) { Mat S(n, Vec(dk)); for (int i = 0; i < n; ++i) for (int j = 0; j < dk; ++j) S[i][j] = M[i][hd * dk + j]; return S; };
            Mat O = attention(slice(Q), slice(K), slice(V), causal);
            for (int i = 0; i < n; ++i) for (int j = 0; j < dk; ++j) concat[i][hd * dk + j] = O[i][j];
        }
        return matmul(concat, Wo);
    }
};

Mat layer_norm(Mat X) {
    for (auto& r : X) {
        double m = accumulate(r.begin(), r.end(), 0.0) / r.size(), v = 0;
        for (double x : r) v += (x - m) * (x - m);
        v /= r.size();
        for (double& x : r) x = (x - m) / sqrt(v + 1e-5);
    }
    return X;
}

Mat add(const Mat& A, const Mat& B) { Mat C = A; for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) C[i][j] += B[i][j]; return C; }

struct Block {  // pré-LN: x + MHA(LN(x)); x + FFN(LN(x))
    MHA att; Mat W1, W2;
    Block(int d, int h, mt19937& rng) : att(d, h, rng) {
        normal_distribution<double> N(0, 1.0 / sqrt(d));
        W1.assign(d, Vec(4 * d)); W2.assign(4 * d, Vec(d));
        for (auto& r : W1) for (double& v : r) v = N(rng);
        for (auto& r : W2) for (double& v : r) v = N(rng) * 0.5;
    }
    Mat operator()(const Mat& X, bool causal) const {
        Mat a = add(X, att(layer_norm(X), causal));
        Mat f = matmul(layer_norm(a), W1);
        for (auto& r : f) for (double& v : r) v = 0.5 * v * (1 + erf(v / sqrt(2.0)));  // GELU
        return add(a, matmul(f, W2));
    }
};

Mat positional_encoding(int n, int d) {
    Mat P(n, Vec(d));
    for (int p = 0; p < n; ++p) for (int i = 0; i < d; i += 2) {
        double w = p / pow(10000.0, (double)i / d);
        P[p][i] = sin(w); if (i + 1 < d) P[p][i + 1] = cos(w);
    }
    return P;
}

double max_abs_diff(const Mat& A, const Mat& B) {
    double m = 0;
    for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) m = max(m, fabs(A[i][j] - B[i][j]));
    return m;
}

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 1);

    // 1) atenção como busca em dicionário: chaves quase ortogonais, query alinhada com a 2ª chave
    Mat K = {{5, 0, 0}, {0, 5, 0}, {0, 0, 5}}, V = {{1, 0}, {0, 1}, {1, 1}}, Q = {{0, 3, 0}};
    Mat W;
    Mat out = attention(Q, K, V, false, &W);
    printf("1) query alinhada a chave 2 -> pesos = [%.3f %.3f %.3f], saida = [%.3f %.3f] (~ valor 2 = [0 1])\n", W[0][0], W[0][1], W[0][2], out[0][0], out[0][1]);

    // 2) máscara causal: posição i só enxerga j <= i
    int n = 5, d = 8;
    Mat X(n, Vec(d));
    for (auto& r : X) for (double& v : r) v = N(rng);
    attention(X, X, X, true, &W);
    printf("2) mascara causal — pesos (linhas somam 1, triangulo superior = 0):\n");
    for (auto& r : W) { printf("   "); for (double v : r) printf("%.2f ", v); printf("\n"); }

    // 3) equivariância a permutação: permutar a entrada permuta a saída igualmente
    MHA mha(d, 2, rng);
    vector<int> perm = {3, 0, 4, 1, 2};
    Mat Xp(n);
    for (int i = 0; i < n; ++i) Xp[i] = X[perm[i]];
    Mat Y = mha(X, false), Yp = mha(Xp, false), Yperm(n);
    for (int i = 0; i < n; ++i) Yperm[i] = Y[perm[i]];
    printf("3) sem posicao: max|MHA(perm(X)) - perm(MHA(X))| = %.2e  (self-attention nao sabe a ordem!)\n", max_abs_diff(Yp, Yperm));

    // 4) com codificação posicional, a simetria é quebrada
    Mat P = positional_encoding(n, d);
    Mat XP = add(X, P), XpP = add(Xp, P);
    Mat Z = mha(XP, false), Zp = mha(XpP, false), Zperm(n);
    for (int i = 0; i < n; ++i) Zperm[i] = Z[perm[i]];
    printf("4) com posicao: max|MHA(perm(X)+P) - perm(MHA(X+P))| = %.3f  (ordem agora importa)\n", max_abs_diff(Zp, Zperm));

    // 5) bloco Transformer completo: forma preservada, custo O(n^2 d) da atenção
    Block blk(d, 2, rng);
    Mat B = blk(XP, true);
    printf("5) bloco Transformer: entrada %zux%zu -> saida %zux%zu; matriz de atencao tem %d x %d entradas (O(n^2))\n", XP.size(), XP[0].size(), B.size(), B[0].size(), n, n);
}
