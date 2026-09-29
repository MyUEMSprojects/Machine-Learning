// Naive Bayes: Gaussiano (features contínuas) e Multinomial (contagem de palavras, com suavização de Laplace).
// Build: g++ -std=c++17 -O2 naive_bayes.cpp -o naive_bayes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

// ---------- Gaussiano ----------
struct GaussianNB {
    int K, d;
    Vec prior; Mat mu, var;
    void fit(const Mat& X, const vector<int>& y, int K_) {
        K = K_; d = X[0].size();
        prior.assign(K, 0); mu.assign(K, Vec(d, 0)); var.assign(K, Vec(d, 0));
        vector<int> cnt(K, 0);
        for (size_t i = 0; i < X.size(); ++i) {
            ++cnt[y[i]];
            for (int j = 0; j < d; ++j) mu[y[i]][j] += X[i][j];
        }
        for (int k = 0; k < K; ++k) for (int j = 0; j < d; ++j) mu[k][j] /= cnt[k];
        for (size_t i = 0; i < X.size(); ++i)
            for (int j = 0; j < d; ++j) var[y[i]][j] += pow(X[i][j] - mu[y[i]][j], 2);
        for (int k = 0; k < K; ++k) {
            prior[k] = (double)cnt[k] / X.size();
            for (int j = 0; j < d; ++j) var[k][j] = var[k][j] / cnt[k] + 1e-9;
        }
    }
    int predict(const Vec& x) const {
        int best = 0; double bs = -1e300;
        for (int k = 0; k < K; ++k) {
            double s = log(prior[k]);  // log P(c) + sum_j log P(x_j | c)  (independência "ingênua")
            for (int j = 0; j < d; ++j)
                s += -0.5 * log(2 * M_PI * var[k][j]) - pow(x[j] - mu[k][j], 2) / (2 * var[k][j]);
            if (s > bs) { bs = s; best = k; }
        }
        return best;
    }
};

// ---------- Multinomial (texto) ----------
struct MultinomialNB {
    map<string, int> vocab;
    vector<double> logprior;
    vector<vector<double>> loglik;
    static vector<string> tok(const string& s) {
        istringstream is(s); vector<string> t; string w;
        while (is >> w) t.push_back(w);
        return t;
    }
    void fit(const vector<string>& docs, const vector<int>& y, int K, double alpha = 1.0) {
        for (auto& d : docs) for (auto& w : tok(d)) if (!vocab.count(w)) { int id = vocab.size(); vocab[w] = id; }
        int V = vocab.size();
        vector<vector<double>> cnt(K, vector<double>(V, alpha));  // suavização de Laplace
        vector<int> nd(K, 0);
        for (size_t i = 0; i < docs.size(); ++i) {
            ++nd[y[i]];
            for (auto& w : tok(docs[i])) cnt[y[i]][vocab[w]] += 1;
        }
        logprior.resize(K); loglik.assign(K, vector<double>(V));
        for (int k = 0; k < K; ++k) {
            logprior[k] = log((double)nd[k] / docs.size());
            double tot = 0;
            for (double c : cnt[k]) tot += c;
            for (int v = 0; v < V; ++v) loglik[k][v] = log(cnt[k][v] / tot);
        }
    }
    int predict(const string& doc) const {
        int best = 0; double bs = -1e300;
        for (size_t k = 0; k < logprior.size(); ++k) {
            double s = logprior[k];
            for (auto& w : tok(doc)) { auto it = vocab.find(w); if (it != vocab.end()) s += loglik[k][it->second]; }
            if (s > bs) { bs = s; best = k; }
        }
        return best;
    }
};

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 1.2);
    vector<Vec> c = {{-3, 0}, {3, 0}, {0, 4}};
    auto gera = [&](int n, Mat& X, vector<int>& y) {
        for (int i = 0; i < n; ++i) { int k = i % 3; X.push_back({c[k][0] + N(rng), c[k][1] + N(rng)}); y.push_back(k); }
    };
    Mat Xtr, Xte; vector<int> ytr, yte;
    gera(300, Xtr, ytr); gera(300, Xte, yte);
    GaussianNB g; g.fit(Xtr, ytr, 3);
    int ok = 0;
    for (size_t i = 0; i < Xte.size(); ++i) ok += g.predict(Xte[i]) == yte[i];
    printf("GaussianNB: acuracia teste = %.3f\n", (double)ok / Xte.size());

    // classificador de spam minúsculo
    vector<string> docs = {"ganhe dinheiro rapido agora", "oferta gratis dinheiro clique", "promocao gratis ganhe premio",
                           "reuniao amanha projeto equipe", "relatorio projeto entrega prazo", "almoco equipe amanha reuniao"};
    vector<int> y = {1, 1, 1, 0, 0, 0};  // 1 = spam
    MultinomialNB m; m.fit(docs, y, 2);
    for (string t : {"ganhe premio gratis", "reuniao do projeto amanha", "clique agora para ganhar dinheiro"})
        printf("\"%s\" -> %s\n", t.c_str(), m.predict(t) ? "SPAM" : "ok");
}
