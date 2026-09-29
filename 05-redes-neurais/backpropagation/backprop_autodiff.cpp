// Mini-framework de autodiferenciação em modo reverso (estilo micrograd) — o que PyTorch faz por baixo.
// Constrói o grafo computacional durante o forward e aplica a regra da cadeia no backward.
// Treina um pequeno MLP com ele e valida os gradientes por diferença finita (gradient check).
// Build: g++ -std=c++17 -O2 backprop_autodiff.cpp -o backprop_autodiff
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>
using namespace std;

struct Value;
using V = shared_ptr<Value>;

struct Value {
    double data, grad = 0;
    vector<V> prev;
    function<void()> backward_fn = [] {};
    Value(double d, vector<V> p = {}) : data(d), prev(move(p)) {}
};

V val(double d) { return make_shared<Value>(d); }

V operator+(const V& a, const V& b) {
    auto o = make_shared<Value>(a->data + b->data, vector<V>{a, b});
    Value* op = o.get();
    o->backward_fn = [a, b, op] { a->grad += op->grad; b->grad += op->grad; };
    return o;
}
V operator*(const V& a, const V& b) {
    auto o = make_shared<Value>(a->data * b->data, vector<V>{a, b});
    Value* op = o.get();
    o->backward_fn = [a, b, op] { a->grad += b->data * op->grad; b->grad += a->data * op->grad; };
    return o;
}
V operator-(const V& a) { return a * val(-1); }
V operator-(const V& a, const V& b) { return a + (-b); }
V vtanh(const V& a) {
    double t = tanh(a->data);
    auto o = make_shared<Value>(t, vector<V>{a});
    Value* op = o.get();
    o->backward_fn = [a, t, op] { a->grad += (1 - t * t) * op->grad; };
    return o;
}
V vexp(const V& a) {
    double e = exp(a->data);
    auto o = make_shared<Value>(e, vector<V>{a});
    Value* op = o.get();
    o->backward_fn = [a, e, op] { a->grad += e * op->grad; };
    return o;
}
V vlog(const V& a) {
    auto o = make_shared<Value>(log(a->data), vector<V>{a});
    Value* op = o.get();
    o->backward_fn = [a, op] { a->grad += op->grad / a->data; };
    return o;
}
V vinv(const V& a) {
    auto o = make_shared<Value>(1.0 / a->data, vector<V>{a});
    Value* op = o.get();
    o->backward_fn = [a, op] { a->grad += -1.0 / (a->data * a->data) * op->grad; };
    return o;
}
V sigmoid(const V& a) { return vinv(val(1) + vexp(-a)); }

// Ordena topologicamente o grafo e propaga gradientes da saída para as folhas.
void backward(const V& root) {
    vector<V> topo; unordered_set<Value*> seen;
    function<void(const V&)> build = [&](const V& v) {
        if (seen.count(v.get())) return;
        seen.insert(v.get());
        for (auto& p : v->prev) build(p);
        topo.push_back(v);
    };
    build(root);
    root->grad = 1;
    for (int i = topo.size() - 1; i >= 0; --i) topo[i]->backward_fn();
}

struct Neuron {
    vector<V> w; V b; bool linear;
    Neuron(int n, mt19937& rng, bool lin = false) : b(val(0)), linear(lin) {
        normal_distribution<double> N(0, 1.0 / sqrt(n));
        for (int i = 0; i < n; ++i) w.push_back(val(N(rng)));
    }
    V operator()(const vector<V>& x) const {
        V s = b;
        for (size_t i = 0; i < w.size(); ++i) s = s + w[i] * x[i];
        return linear ? s : vtanh(s);
    }
};

struct MLP {
    vector<vector<Neuron>> layers;
    MLP(vector<int> sizes, mt19937& rng) {
        for (size_t l = 1; l < sizes.size(); ++l) {
            vector<Neuron> layer;
            for (int j = 0; j < sizes[l]; ++j) layer.emplace_back(sizes[l - 1], rng, l == sizes.size() - 1);
            layers.push_back(layer);
        }
    }
    V operator()(vector<V> x) const {
        for (auto& layer : layers) {
            vector<V> out;
            for (auto& n : layer) out.push_back(n(x));
            x = out;
        }
        return x[0];  // logit
    }
    vector<V> params() const {
        vector<V> p;
        for (auto& l : layers) for (auto& n : l) { for (auto& w : n.w) p.push_back(w); p.push_back(n.b); }
        return p;
    }
};

// perda logística estável para logit z e rótulo y in {0,1}:  -y log s(z) - (1-y) log(1-s(z))
V bce(const V& z, int y) { return y ? vlog(val(1) + vexp(-z)) : vlog(val(1) + vexp(z)); }

int main() {
    // 1) exemplo pequeno da regra da cadeia: f = tanh(w*x + b)
    {
        V x = val(2), w = val(-0.5), b = val(0.3);
        V f = vtanh(w * x + b);
        backward(f);
        double t = tanh(-0.5 * 2 + 0.3);
        printf("df/dw: autodiff=%.6f  analitico=%.6f   df/db: autodiff=%.6f  analitico=%.6f\n", w->grad, (1 - t * t) * 2, b->grad, 1 - t * t);
    }
    mt19937 rng(0);
    // dados: duas luas
    vector<vector<double>> X; vector<int> y;
    uniform_real_distribution<double> U(0, M_PI);
    normal_distribution<double> N(0, 0.12);
    for (int i = 0; i < 100; ++i) {
        double t = U(rng);
        if (i % 2 == 0) { X.push_back({cos(t) + N(rng), sin(t) + N(rng)}); y.push_back(0); }
        else { X.push_back({1 - cos(t) + N(rng), 0.5 - sin(t) + N(rng)}); y.push_back(1); }
    }
    MLP net({2, 8, 8, 1}, rng);
    auto total_loss = [&]() {
        V loss = val(0);
        for (size_t i = 0; i < X.size(); ++i) loss = loss + bce(net({val(X[i][0]), val(X[i][1])}), y[i]);
        return loss * val(1.0 / X.size());
    };
    auto params = net.params();

    // 2) gradient check: compara backprop com diferença central em alguns parâmetros
    {
        for (auto& p : params) p->grad = 0;
        V loss = total_loss(); backward(loss);
        double maxdiff = 0;
        for (int k : {0, 5, 20, 40, (int)params.size() - 1}) {
            double g = params[k]->grad, orig = params[k]->data, h = 1e-5;
            params[k]->data = orig + h; double lp = total_loss()->data;
            params[k]->data = orig - h; double lm = total_loss()->data;
            params[k]->data = orig;
            maxdiff = max(maxdiff, fabs(g - (lp - lm) / (2 * h)));
        }
        printf("gradient check: maior diferenca backprop vs numerico = %.2e\n", maxdiff);
    }
    // 3) treino por gradiente descendente
    for (int epoch = 0; epoch <= 300; ++epoch) {
        for (auto& p : params) p->grad = 0;
        V loss = total_loss(); backward(loss);
        for (auto& p : params) p->data -= 0.5 * p->grad;
        if (epoch % 100 == 0) {
            int ok = 0;
            for (size_t i = 0; i < X.size(); ++i) ok += (net({val(X[i][0]), val(X[i][1])})->data > 0) == y[i];
            printf("epoca %3d  loss=%.4f  acc=%.2f\n", epoch, loss->data, (double)ok / X.size());
        }
    }
}
