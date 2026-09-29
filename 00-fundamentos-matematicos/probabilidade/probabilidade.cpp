// Probabilidade: amostragem de distribuições, Monte Carlo, teorema de Bayes e TCL.
// Build: g++ -std=c++17 -O2 probabilidade.cpp -o probabilidade
#include <cmath>
#include <iostream>
#include <random>
using namespace std;

int main() {
    mt19937 rng(42);
    uniform_real_distribution<double> U(0, 1);
    normal_distribution<double> N(0, 1);

    // 1) Monte Carlo: estimar pi sorteando pontos no quadrado unitário
    long dentro = 0, n = 2000000;
    for (long i = 0; i < n; ++i) {
        double x = U(rng), y = U(rng);
        if (x * x + y * y <= 1) ++dentro;
    }
    cout << "pi ~ " << 4.0 * dentro / n << "\n";

    // 2) Teorema de Bayes: teste diagnóstico
    // P(doente)=1%, sensibilidade P(+|doente)=99%, falso positivo P(+|saudável)=5%
    double pD = 0.01, sens = 0.99, fp = 0.05;
    double pPos = sens * pD + fp * (1 - pD);
    cout << "P(doente | +) = " << sens * pD / pPos << " (analitico)\n";
    long pos = 0, pos_doente = 0;  // verificação por simulação
    for (long i = 0; i < n; ++i) {
        bool doente = U(rng) < pD;
        bool teste = U(rng) < (doente ? sens : fp);
        if (teste) { ++pos; pos_doente += doente; }
    }
    cout << "P(doente | +) = " << (double)pos_doente / pos << " (simulado)\n";

    // 3) Lei dos grandes números + TCL: média de 30 uniformes ~ Normal(0.5, 1/(12*30))
    int m = 30, amostras = 100000;
    double soma = 0, soma2 = 0;
    for (int i = 0; i < amostras; ++i) {
        double s = 0;
        for (int j = 0; j < m; ++j) s += U(rng);
        s /= m;
        soma += s; soma2 += s * s;
    }
    double media = soma / amostras, var = soma2 / amostras - media * media;
    cout << "media das medias = " << media << " (esperado 0.5)\n";
    cout << "variancia das medias = " << var << " (esperado " << 1.0 / (12 * m) << ")\n";

    // 4) Esperança e variância de N(0,1) por amostragem
    double e = 0, e2 = 0;
    for (long i = 0; i < n; ++i) { double x = N(rng); e += x; e2 += x * x; }
    cout << "E[X]=" << e / n << " Var[X]=" << e2 / n - (e / n) * (e / n) << "\n";
}
