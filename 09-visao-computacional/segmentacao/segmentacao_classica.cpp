// Segmentação clássica do zero: limiar de Otsu, operações morfológicas (erosão/dilatação/abertura),
// rotulagem de componentes conexas (segmentação de instâncias) e métricas IoU/Dice contra a máscara verdadeira.
// Build: g++ -std=c++17 -O2 segmentacao_classica.cpp -o segmentacao_classica
#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <random>
#include <vector>
using namespace std;

using Img = vector<vector<double>>;
using Mask = vector<vector<int>>;

const int H = 48, W = 48;

// Otsu: escolhe o limiar que maximiza a variância ENTRE classes (fundo × objeto) no histograma
double otsu(const Img& im) {
    const int B = 64;
    vector<double> h(B, 0);
    for (auto& r : im) for (double v : r) h[min(B - 1, max(0, (int)(v * B)))] += 1;
    double total = H * W, sum = 0;
    for (int i = 0; i < B; ++i) sum += i * h[i];
    double wb = 0, sb = 0, best = -1; int bt = 0;
    for (int t = 0; t < B; ++t) {
        wb += h[t]; if (wb == 0) continue;
        double wf = total - wb; if (wf == 0) break;
        sb += t * h[t];
        double mb = sb / wb, mf = (sum - sb) / wf, var = wb * wf * (mb - mf) * (mb - mf);
        if (var > best) { best = var; bt = t; }
    }
    return (bt + 1.0) / B;
}

Mask morf(const Mask& m, bool erode) {  // elemento estruturante 3x3
    Mask o(H, vector<int>(W, 0));
    for (int i = 0; i < H; ++i) for (int j = 0; j < W; ++j) {
        int all = 1, any = 0;
        for (int di = -1; di <= 1; ++di) for (int dj = -1; dj <= 1; ++dj) {
            int a = i + di, b = j + dj, v = (a >= 0 && a < H && b >= 0 && b < W) ? m[a][b] : 0;
            all &= v; any |= v;
        }
        o[i][j] = erode ? all : any;
    }
    return o;
}

// Componentes conexas (4-vizinhança) por BFS; retorna nº de componentes e rótulos 1..n
int componentes(const Mask& m, Mask& lab) {
    lab.assign(H, vector<int>(W, 0));
    int n = 0;
    for (int i = 0; i < H; ++i) for (int j = 0; j < W; ++j) {
        if (!m[i][j] || lab[i][j]) continue;
        ++n; queue<pair<int, int>> q; q.push({i, j}); lab[i][j] = n;
        while (!q.empty()) {
            auto [a, b] = q.front(); q.pop();
            const int d[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (auto& dd : d) {
                int x = a + dd[0], y = b + dd[1];
                if (x >= 0 && x < H && y >= 0 && y < W && m[x][y] && !lab[x][y]) { lab[x][y] = n; q.push({x, y}); }
            }
        }
    }
    return n;
}

void metricas(const Mask& p, const Mask& g, double& iou, double& dice) {
    double inter = 0, sp = 0, sg = 0;
    for (int i = 0; i < H; ++i) for (int j = 0; j < W; ++j) { inter += p[i][j] && g[i][j]; sp += p[i][j]; sg += g[i][j]; }
    iou = inter / (sp + sg - inter + 1e-12); dice = 2 * inter / (sp + sg + 1e-12);
}

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 0.12);
    Img im(H, vector<double>(W)); Mask gt(H, vector<int>(W, 0));
    // 3 discos brilhantes sobre fundo escuro com gradiente de iluminação e ruído
    struct C { int y, x, r; } cs[] = {{12, 12, 7}, {14, 34, 6}, {35, 24, 8}};
    for (int i = 0; i < H; ++i) for (int j = 0; j < W; ++j) {
        for (auto& c : cs) if ((i - c.y) * (i - c.y) + (j - c.x) * (j - c.x) <= c.r * c.r) gt[i][j] = 1;
        im[i][j] = (gt[i][j] ? 0.75 : 0.25) + 0.1 * j / W + N(rng);
    }
    double t = otsu(im);
    Mask bin(H, vector<int>(W));
    for (int i = 0; i < H; ++i) for (int j = 0; j < W; ++j) bin[i][j] = im[i][j] > t;
    Mask aberta = morf(morf(bin, true), false);  // abertura = erosão seguida de dilatação: remove ruído "sal"
    Mask lab;
    int n = componentes(aberta, lab);

    double iou, dice;
    printf("limiar de Otsu = %.3f (fundo ~0.25-0.35, objetos ~0.75-0.85)\n", t);
    metricas(bin, gt, iou, dice);
    printf("apos limiar          : IoU=%.3f Dice=%.3f\n", iou, dice);
    metricas(aberta, gt, iou, dice);
    printf("apos abertura (3x3)  : IoU=%.3f Dice=%.3f\n", iou, dice);
    printf("componentes conexas (instancias) encontradas: %d (verdadeiro: 3)\n", n);
    for (int k = 1; k <= n; ++k) { int a = 0; for (auto& r : lab) for (int v : r) a += v == k; printf("  instancia %d: %d pixels\n", k, a); }
    printf("\nMascara final (# = objeto):\n");
    for (int i = 0; i < H; i += 2) { for (int j = 0; j < W; ++j) putchar(aberta[i][j] ? '#' : '.'); putchar('\n'); }
}
