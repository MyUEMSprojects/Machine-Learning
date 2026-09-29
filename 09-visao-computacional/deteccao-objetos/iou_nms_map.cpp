// Componentes centrais da detecção de objetos, do zero: IoU, Non-Maximum Suppression (NMS) e Average Precision / mAP.
// Cenário sintético: caixas verdadeiras + detecções ruidosas (duplicadas, deslocadas e falsos positivos).
// Build: g++ -std=c++17 -O2 iou_nms_map.cpp -o iou_nms_map
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

struct Box { double x1, y1, x2, y2; };
struct Det { Box b; double score; int cls; int img; };
struct GT { Box b; int cls; int img; };

double area(const Box& b) { return max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1); }

// IoU = área da interseção / área da união
double iou(const Box& a, const Box& b) {
    Box i{max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2)};
    double inter = area(i);
    return inter / (area(a) + area(b) - inter + 1e-12);
}

// NMS guloso: pega a de maior score, remove as que sobrepõem (IoU > limiar) com ela, repete.
vector<Det> nms(vector<Det> d, double thr) {
    sort(d.begin(), d.end(), [](auto& a, auto& b) { return a.score > b.score; });
    vector<Det> keep;
    vector<bool> dead(d.size(), false);
    for (size_t i = 0; i < d.size(); ++i) {
        if (dead[i]) continue;
        keep.push_back(d[i]);
        for (size_t j = i + 1; j < d.size(); ++j)
            if (!dead[j] && d[j].cls == d[i].cls && d[j].img == d[i].img && iou(d[i].b, d[j].b) > thr) dead[j] = true;
    }
    return keep;
}

// AP de uma classe: ordena detecções por score; TP se IoU>=0.5 com um GT ainda não "usado"; área sob a curva precisão-recall
double average_precision(vector<Det> dets, const vector<GT>& gts, int cls, double iou_thr = 0.5) {
    vector<Det> D;
    for (auto& d : dets) if (d.cls == cls) D.push_back(d);
    sort(D.begin(), D.end(), [](auto& a, auto& b) { return a.score > b.score; });
    int npos = 0;
    for (auto& g : gts) npos += g.cls == cls;
    vector<bool> used(gts.size(), false);
    vector<double> prec, rec; int tp = 0, fp = 0;
    for (auto& d : D) {
        int best = -1; double bi = iou_thr;
        for (size_t g = 0; g < gts.size(); ++g)
            if (gts[g].cls == cls && gts[g].img == d.img && !used[g]) { double v = iou(d.b, gts[g].b); if (v >= bi) { bi = v; best = g; } }
        if (best >= 0) { used[best] = true; ++tp; } else ++fp;  // detecção duplicada de um GT já casado conta como FP
        prec.push_back((double)tp / (tp + fp)); rec.push_back((double)tp / npos);
    }
    // interpolação: precisão máxima à direita de cada nível de recall (estilo Pascal VOC, todos os pontos)
    for (int i = (int)prec.size() - 2; i >= 0; --i) prec[i] = max(prec[i], prec[i + 1]);
    double ap = 0, prev = 0;
    for (size_t i = 0; i < prec.size(); ++i) { ap += (rec[i] - prev) * prec[i]; prev = rec[i]; }
    return ap;
}

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 1);
    uniform_real_distribution<double> U(0, 1);
    printf("IoU: caixas identicas=%.2f | metade sobreposta=%.2f | disjuntas=%.2f\n",
           iou({0, 0, 10, 10}, {0, 0, 10, 10}), iou({0, 0, 10, 10}, {5, 0, 15, 10}), iou({0, 0, 10, 10}, {20, 20, 30, 30}));

    // 20 imagens, 2 classes; 1-3 objetos por imagem
    vector<GT> gts; vector<Det> dets;
    for (int img = 0; img < 20; ++img) {
        int n = 1 + rng() % 3;
        for (int k = 0; k < n; ++k) {
            double x = 20 + 100 * U(rng), y = 20 + 100 * U(rng), w = 20 + 30 * U(rng), h = 20 + 30 * U(rng);
            int cls = rng() % 2;
            Box g{x, y, x + w, y + h};
            gts.push_back({g, cls, img});
            // 1 detecção boa + 1-2 duplicatas deslocadas (típico de detectores sem NMS) com scores altos, e um erro de classe
            for (int d = 0; d < 3; ++d) {
                if (d == 0 && U(rng) < 0.15) continue;  // 15% dos objetos não têm a detecção "boa" (só duplicatas fracas)
                double jit = d == 0 ? 2 : 6;
                Box b{g.x1 + jit * N(rng), g.y1 + jit * N(rng), g.x2 + jit * N(rng), g.y2 + jit * N(rng)};
                double sc = d == 0 ? 0.6 + 0.35 * U(rng) : 0.5 + 0.4 * U(rng);  // duplicatas podem ter score alto, "furando a fila"
                dets.push_back({b, sc, cls, img});
            }
        }
        for (int fp = 0; fp < 3; ++fp) {  // falsos positivos aleatórios, scores mais baixos
            double x = 120 * U(rng), y = 120 * U(rng);
            dets.push_back({{x, y, x + 25, y + 25}, 0.2 + 0.5 * U(rng), (int)(rng() % 2), img});
        }
    }
    auto mAP = [&](const vector<Det>& d) { return (average_precision(d, gts, 0) + average_precision(d, gts, 1)) / 2; };
    printf("\n%zu caixas verdadeiras, %zu deteccoes brutas\n", gts.size(), dets.size());
    printf("mAP@0.5 sem NMS               : %.3f  (duplicatas viram falsos positivos)\n", mAP(dets));
    for (double thr : {0.3, 0.5, 0.7}) {
        auto k = nms(dets, thr);
        printf("mAP@0.5 com NMS (IoU>%.1f)     : %.3f  (%zu deteccoes restantes)\n", thr, mAP(k), k.size());
    }
}
