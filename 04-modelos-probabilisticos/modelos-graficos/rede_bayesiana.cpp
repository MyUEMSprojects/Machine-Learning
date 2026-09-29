// Rede Bayesiana clássica (Alarme): inferência exata por enumeração da distribuição conjunta.
//   Burglary(B) -> Alarm(A) <- Earthquake(E);  A -> JohnCalls(J);  A -> MaryCalls(M)
// Mostra o efeito "explaining away". Build: g++ -std=c++17 -O2 rede_bayesiana.cpp -o rede_bayesiana
#include <iostream>
#include <map>
#include <string>
#include <vector>
using namespace std;

// Variáveis: 0=B 1=E 2=A 3=J 4=M  (valores 0/1). Fatoração: P(B)P(E)P(A|B,E)P(J|A)P(M|A)
const char* nome[] = {"Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"};

double joint(const int v[5]) {
    double pB = 0.001, pE = 0.002;
    double pA[2][2] = {{0.001, 0.29}, {0.94, 0.95}};  // P(A=1 | B, E)
    double pJ[2] = {0.05, 0.90}, pM[2] = {0.01, 0.70};  // P(J=1|A), P(M=1|A)
    double p = (v[0] ? pB : 1 - pB) * (v[1] ? pE : 1 - pE);
    double a = pA[v[0]][v[1]];
    p *= v[2] ? a : 1 - a;
    p *= v[3] ? pJ[v[2]] : 1 - pJ[v[2]];
    p *= v[4] ? pM[v[2]] : 1 - pM[v[2]];
    return p;
}

// P(query = 1 | evidência): soma a conjunta sobre as variáveis ocultas
double query(int q, const map<int, int>& ev) {
    double num = 0, den = 0;
    for (int mask = 0; mask < 32; ++mask) {
        int v[5];
        for (int i = 0; i < 5; ++i) v[i] = mask >> i & 1;
        bool ok = true;
        for (auto& [var, val] : ev) if (v[var] != val) ok = false;
        if (!ok) continue;
        double p = joint(v);
        den += p;
        if (v[q]) num += p;
    }
    return num / den;
}

int main() {
    printf("P(Burglary)                                  = %.4f\n", query(0, {}));
    printf("P(Burglary | JohnCalls=1)                    = %.4f\n", query(0, {{3, 1}}));
    printf("P(Burglary | JohnCalls=1, MaryCalls=1)       = %.4f\n", query(0, {{3, 1}, {4, 1}}));
    printf("\nExplaining away:\n");
    printf("P(Burglary | Alarm=1)                        = %.4f\n", query(0, {{2, 1}}));
    printf("P(Burglary | Alarm=1, Earthquake=1)          = %.4f  <- terremoto explica o alarme, assalto cai\n", query(0, {{2, 1}, {1, 1}}));
    printf("\nIndependencia condicional: P(J | A=1, M=1) = %.4f = P(J | A=1) = %.4f (J _|_ M | A)\n", query(3, {{2, 1}, {4, 1}}), query(3, {{2, 1}}));
}
