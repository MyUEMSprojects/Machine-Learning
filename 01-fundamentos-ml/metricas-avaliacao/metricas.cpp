// Métricas de avaliação: regressão (MSE, RMSE, MAE, R^2) e classificação
// (matriz de confusão, acurácia, precisão, recall, F1, ROC AUC, log loss).
// Build: g++ -std=c++17 -O2 metricas.cpp -o metricas
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>
using namespace std;

using Vec = vector<double>;

double mse(const Vec& y, const Vec& p) {
    double s = 0;
    for (size_t i = 0; i < y.size(); ++i) s += (y[i] - p[i]) * (y[i] - p[i]);
    return s / y.size();
}
double mae(const Vec& y, const Vec& p) {
    double s = 0;
    for (size_t i = 0; i < y.size(); ++i) s += fabs(y[i] - p[i]);
    return s / y.size();
}
double r2(const Vec& y, const Vec& p) {
    double m = accumulate(y.begin(), y.end(), 0.0) / y.size(), ss_res = 0, ss_tot = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        ss_res += (y[i] - p[i]) * (y[i] - p[i]);
        ss_tot += (y[i] - m) * (y[i] - m);
    }
    return 1 - ss_res / ss_tot;
}

struct Confusao { int tp = 0, fp = 0, tn = 0, fn = 0; };

Confusao confusao(const vector<int>& y, const vector<int>& pred) {
    Confusao c;
    for (size_t i = 0; i < y.size(); ++i) {
        if (y[i] && pred[i]) ++c.tp;
        else if (!y[i] && pred[i]) ++c.fp;
        else if (!y[i] && !pred[i]) ++c.tn;
        else ++c.fn;
    }
    return c;
}

// AUC = P(score de um positivo aleatório > score de um negativo aleatório) (Mann–Whitney)
double roc_auc(const vector<int>& y, const Vec& score) {
    double ganhos = 0; long pares = 0;
    for (size_t i = 0; i < y.size(); ++i)
        for (size_t j = 0; j < y.size(); ++j)
            if (y[i] == 1 && y[j] == 0) {
                ++pares;
                ganhos += score[i] > score[j] ? 1.0 : (score[i] == score[j] ? 0.5 : 0.0);
            }
    return ganhos / pares;
}

double log_loss(const vector<int>& y, const Vec& p) {
    double s = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        double q = min(max(p[i], 1e-15), 1 - 1e-15);
        s -= y[i] * log(q) + (1 - y[i]) * log(1 - q);
    }
    return s / y.size();
}

int main() {
    // regressão
    Vec y = {3.0, -0.5, 2.0, 7.0}, p = {2.5, 0.0, 2.0, 8.0};
    cout << "MSE=" << mse(y, p) << " RMSE=" << sqrt(mse(y, p)) << " MAE=" << mae(y, p) << " R2=" << r2(y, p) << "\n";

    // classificação desbalanceada: 95% negativos. Um classificador "sempre 0" tem 95% de acurácia!
    vector<int> yc(100, 0);
    for (int i = 0; i < 5; ++i) yc[i] = 1;
    vector<int> sempre0(100, 0);
    Confusao c0 = confusao(yc, sempre0);
    cout << "\n'sempre 0': acuracia=" << (double)(c0.tp + c0.tn) / 100 << " recall=" << 0.0 << "  <- acuracia engana\n";

    // um classificador real com scores
    Vec score(100);
    for (int i = 0; i < 100; ++i) score[i] = yc[i] ? 0.4 + 0.6 * ((i * 37) % 10) / 10.0 : 0.5 * ((i * 53) % 10) / 10.0;
    vector<int> pred(100);
    for (int i = 0; i < 100; ++i) pred[i] = score[i] >= 0.5;
    Confusao c = confusao(yc, pred);
    double prec = (double)c.tp / (c.tp + c.fp), rec = (double)c.tp / (c.tp + c.fn);
    cout << "TP=" << c.tp << " FP=" << c.fp << " TN=" << c.tn << " FN=" << c.fn << "\n";
    cout << "acuracia=" << (double)(c.tp + c.tn) / 100 << " precisao=" << prec << " recall=" << rec
         << " F1=" << 2 * prec * rec / (prec + rec) << "\n";
    cout << "ROC AUC=" << roc_auc(yc, score) << "  log loss=" << log_loss(yc, score) << "\n";
}
