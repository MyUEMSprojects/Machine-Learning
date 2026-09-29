// Bandits multi-armados: greedy, epsilon-greedy, UCB1 e Thompson Sampling (Bernoulli) no "10-armed testbed".
// Build: g++ -std=c++17 -O2 bandits.cpp -o bandits
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

const int K = 10, STEPS = 2000, RUNS = 500;

struct Result { double reward = 0, optimal = 0, regret_end = 0; };

// Cada política recebe (contagens, médias, sucessos, falhas, t) e retorna o braço
using Policy = function<int(const vector<int>&, const vector<double>&, const vector<int>&, const vector<int>&, int, mt19937&)>;

Result run(const Policy& pol, mt19937& rng) {
    Result res;
    for (int r = 0; r < RUNS; ++r) {
        vector<double> p(K);  // probabilidades de sucesso (Bernoulli) sorteadas para cada braço
        for (double& x : p) x = uniform_real_distribution<double>(0.05, 0.95)(rng);
        int best = max_element(p.begin(), p.end()) - p.begin();
        vector<int> n(K, 0), succ(K, 0), fail(K, 0); vector<double> mean(K, 0);
        double regret = 0;
        for (int t = 1; t <= STEPS; ++t) {
            int a = pol(n, mean, succ, fail, t, rng);
            int rew = bernoulli_distribution(p[a])(rng);
            ++n[a]; (rew ? succ[a] : fail[a])++;
            mean[a] += (rew - mean[a]) / n[a];  // média incremental
            res.reward += rew; res.optimal += (a == best);
            regret += p[best] - p[a];
        }
        res.regret_end += regret;
    }
    res.reward /= (double)RUNS * STEPS; res.optimal = 100 * res.optimal / ((double)RUNS * STEPS); res.regret_end /= RUNS;
    return res;
}

int argmax_random_tie(const vector<double>& v, mt19937& rng) {
    double m = *max_element(v.begin(), v.end());
    vector<int> c;
    for (int i = 0; i < K; ++i) if (v[i] == m) c.push_back(i);
    return c[uniform_int_distribution<int>(0, c.size() - 1)(rng)];
}

int main() {
    mt19937 rng(0);
    vector<pair<string, Policy>> pols = {
        {"aleatorio", [](auto&, auto&, auto&, auto&, int, mt19937& g) { return (int)uniform_int_distribution<int>(0, K - 1)(g); }},
        {"greedy", [](auto&, auto& m, auto&, auto&, int, mt19937& g) { return argmax_random_tie(m, g); }},
        {"eps-greedy 0.01", [](auto&, auto& m, auto&, auto&, int, mt19937& g) { return uniform_real_distribution<double>(0, 1)(g) < 0.01 ? (int)uniform_int_distribution<int>(0, K - 1)(g) : argmax_random_tie(m, g); }},
        {"eps-greedy 0.1", [](auto&, auto& m, auto&, auto&, int, mt19937& g) { return uniform_real_distribution<double>(0, 1)(g) < 0.1 ? (int)uniform_int_distribution<int>(0, K - 1)(g) : argmax_random_tie(m, g); }},
        {"UCB1 (c=1)", [](auto& n, auto& m, auto&, auto&, int t, mt19937& g) {
             vector<double> u(K);
             for (int i = 0; i < K; ++i) u[i] = n[i] == 0 ? 1e9 : m[i] + sqrt(2 * log((double)t) / n[i]);  // média + bônus de incerteza
             return argmax_random_tie(u, g); }},
        {"Thompson (Beta)", [](auto&, auto&, auto& s, auto& f, int, mt19937& g) {
             vector<double> smp(K);
             for (int i = 0; i < K; ++i) {  // amostra de Beta(s+1, f+1) via duas Gammas
                 gamma_distribution<double> ga(s[i] + 1, 1), gb(f[i] + 1, 1);
                 double x = ga(g), y = gb(g); smp[i] = x / (x + y);
             }
             return argmax_random_tie(smp, g); }},
    };
    printf("%-16s | recompensa media | %% acao otima | arrependimento acumulado (%d passos)\n", "politica", STEPS);
    for (auto& [nome, pol] : pols) {
        Result r = run(pol, rng);
        printf("%-16s |      %.3f       |    %5.1f     |   %.1f\n", nome.c_str(), r.reward, r.optimal, r.regret_end);
    }
    cout << "(arrependimento = recompensa que o braco otimo teria dado a mais. UCB1 e Thompson tem arrependimento logaritmico em teoria;\n em horizontes curtos o bonus do UCB1 e conservador e eps-greedy pode ganhar dele; Thompson costuma ser o melhor na pratica)\n";
}
