// Aprendizado Federado: FedAvg do zero (regressão logística) com clientes IID e NÃO-IID (viés de rótulos),
// comparado com treino centralizado e com modelos treinados só localmente.
// Os dados nunca saem dos clientes: só os PESOS do modelo são enviados ao servidor e agregados.
// Build: g++ -std=c++17 -O2 fedavg.cpp -o fedavg
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;
const int D = 6;

struct Data { Mat X; vector<int> y; };

double sig(double z) { return 1 / (1 + exp(-z)); }
double logit(const Vec& w, const Vec& x) { double z = w[D]; for (int j = 0; j < D; ++j) z += w[j] * x[j]; return z; }

void local_train(Vec& w, const Data& d, int epochs, double lr, int batch, mt19937& rng) {  // SGD em mini-lotes
    int n = d.X.size();
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    for (int e = 0; e < epochs; ++e) {
        shuffle(idx.begin(), idx.end(), rng);
        for (int s = 0; s < n; s += batch) {
            int m = min(batch, n - s); Vec g(D + 1, 0.0);
            for (int t = 0; t < m; ++t) {
                int i = idx[s + t]; double err = sig(logit(w, d.X[i])) - d.y[i];
                for (int j = 0; j < D; ++j) g[j] += err * d.X[i][j] / m;
                g[D] += err / m;
            }
            for (int j = 0; j <= D; ++j) w[j] -= lr * g[j];
        }
    }
}

double acc(const Vec& w, const Data& d) {
    int ok = 0;
    for (size_t i = 0; i < d.X.size(); ++i) ok += (logit(w, d.X[i]) > 0) == d.y[i];
    return (double)ok / d.X.size();
}

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 1);
    Vec mu0(D, -0.5), mu1(D, 0.5);  // duas classes com sobreposição
    auto gera = [&](int cls) { Vec x(D); for (int j = 0; j < D; ++j) x[j] = (cls ? mu1[j] : mu0[j]) + N(rng); return x; };

    int K = 10, n_por_cliente = 100;
    Data teste;
    for (int i = 0; i < 2000; ++i) { int c = i % 2; teste.X.push_back(gera(c)); teste.y.push_back(c); }

    for (bool iid : {true, false}) {
        // partição: IID = 50/50 em cada cliente; NÃO-IID = metade dos clientes só tem ~95% classe 1, a outra metade ~95% classe 0
        vector<Data> clientes(K);
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < n_por_cliente; ++i) {
                double p1 = iid ? 0.5 : (k % 2 ? 0.95 : 0.05);
                int c = uniform_real_distribution<double>(0, 1)(rng) < p1;
                clientes[k].X.push_back(gera(c)); clientes[k].y.push_back(c);
            }
        Data central;  // união dos dados (baseline "ideal" que viola a privacidade)
        for (auto& c : clientes) { central.X.insert(central.X.end(), c.X.begin(), c.X.end()); central.y.insert(central.y.end(), c.y.begin(), c.y.end()); }

        printf("\n=== Clientes %s ===\n", iid ? "IID (50/50 de cada classe)" : "NAO-IID (cada cliente ~95% de uma so classe)");
        Vec wc(D + 1, 0.0); local_train(wc, central, 30, 0.1, 32, rng);
        printf("centralizado (todos os dados juntos): acc teste = %.3f\n", acc(wc, teste));
        double media_local = 0;
        for (auto& c : clientes) { Vec w(D + 1, 0.0); local_train(w, c, 30, 0.1, 32, rng); media_local += acc(w, teste) / K; }
        printf("so local (cada cliente sozinho)      : acc teste media = %.3f\n", media_local);

        for (int E : {1, 5, 20}) {
            Vec g(D + 1, 0.0);  // modelo global
            printf("FedAvg (E=%2d epocas locais/rodada)   : acc por rodada ->", E);
            for (int r = 1; r <= 30; ++r) {
                Vec novo(D + 1, 0.0);
                double total = 0;
                for (int k = 0; k < K; ++k) {  // cada cliente parte do modelo global e treina LOCALMENTE
                    Vec w = g;
                    local_train(w, clientes[k], E, 0.1, 32, rng);
                    double peso = clientes[k].X.size(); total += peso;  // média ponderada pelo nº de exemplos (FedAvg)
                    for (int j = 0; j <= D; ++j) novo[j] += peso * w[j];
                }
                for (double& v : novo) v /= total;
                g = novo;
                if (r == 1 || r == 5 || r == 10 || r == 30) printf(" r%d:%.3f", r, acc(g, teste));
            }
            printf("\n");
        }
    }
    printf("\nCom dados NAO-IID, o modelo treinado so localmente falha (cada cliente so viu ~1 classe), mas a media federada recupera o\n"
           "desempenho do centralizado sem mover os dados. Neste problema linear simples o \"client drift\" quase nao aparece; ele fica\n"
           "evidente em redes neurais com clientes que veem poucas classes (ver fedavg_pytorch.py).\n");
}
