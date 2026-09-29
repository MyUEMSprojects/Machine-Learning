// Representação de texto: tokenização, bag-of-words, TF-IDF, similaridade do cosseno e BM25 (busca).
// Build: g++ -std=c++17 -O2 tfidf.cpp -o tfidf
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

using Vec = vector<double>;

vector<string> tokenize(const string& s) {  // minúsculas, só letras/dígitos
    vector<string> t; string w;
    for (unsigned char c : s) {
        if (isalnum(c)) w += tolower(c);
        else if (!w.empty()) { t.push_back(w); w.clear(); }
    }
    if (!w.empty()) t.push_back(w);
    return t;
}

double cosine(const Vec& a, const Vec& b) {
    double d = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / (sqrt(na) * sqrt(nb) + 1e-12);
}

int main() {
    vector<string> docs = {
        "o gato sobe no telhado e o gato mia",
        "o cachorro late para o gato no quintal",
        "a rede neural aprende com o gradiente descendente",
        "o gradiente descendente atualiza os pesos da rede neural",
        "o gato dorme no sofa o dia inteiro",
        "modelos de linguagem preveem a proxima palavra"};
    vector<string> stop = {"o", "a", "e", "os", "da", "de", "no", "para", "com"};

    // vocabulário (sem stopwords)
    map<string, int> vocab;
    vector<vector<string>> toks;
    for (auto& d : docs) {
        vector<string> t;
        for (auto& w : tokenize(d)) if (find(stop.begin(), stop.end(), w) == stop.end()) t.push_back(w);
        toks.push_back(t);
        for (auto& w : t) if (!vocab.count(w)) { int id = vocab.size(); vocab[w] = id; }
    }
    int V = vocab.size(), N = docs.size();
    printf("vocabulario: %d termos, %d documentos\n", V, N);

    // bag-of-words (contagens) e df (nº de documentos em que o termo aparece)
    vector<Vec> tf(N, Vec(V, 0)); Vec df(V, 0);
    for (int i = 0; i < N; ++i) {
        for (auto& w : toks[i]) tf[i][vocab[w]] += 1;
        for (int j = 0; j < V; ++j) if (tf[i][j] > 0) df[j] += 1;
    }
    // TF-IDF (mesma fórmula suavizada do scikit-learn): idf = ln((1+N)/(1+df)) + 1, e normalização L2
    vector<Vec> tfidf(N, Vec(V));
    Vec idf(V);
    for (int j = 0; j < V; ++j) idf[j] = log((1.0 + N) / (1.0 + df[j])) + 1;
    for (int i = 0; i < N; ++i) {
        double norm = 0;
        for (int j = 0; j < V; ++j) { tfidf[i][j] = tf[i][j] * idf[j]; norm += tfidf[i][j] * tfidf[i][j]; }
        norm = sqrt(norm);
        for (int j = 0; j < V; ++j) tfidf[i][j] /= norm;
    }
    // termos mais "característicos" do doc 0
    vector<pair<double, string>> top;
    for (auto& [w, j] : vocab) top.push_back({tfidf[0][j], w});
    sort(top.rbegin(), top.rend());
    printf("termos com maior TF-IDF no doc 0 ('%s'): ", docs[0].c_str());
    for (int k = 0; k < 3; ++k) printf("%s(%.2f) ", top[k].second.c_str(), top[k].first);
    printf("\n\n");

    // busca: consulta -> ranking por cosseno (TF-IDF) e por BM25
    string consulta = "gradiente da rede neural";
    Vec q(V, 0);
    for (auto& w : tokenize(consulta)) if (vocab.count(w)) q[vocab[w]] += idf[vocab[w]];
    printf("consulta: \"%s\"\n", consulta.c_str());
    double avgdl = 0;
    for (auto& t : toks) avgdl += t.size() / (double)N;
    const double k1 = 1.5, b = 0.75;
    vector<pair<double, int>> rc, rb;
    for (int i = 0; i < N; ++i) {
        rc.push_back({cosine(q, tfidf[i]), i});
        double bm = 0;
        for (auto& w : tokenize(consulta)) {
            if (!vocab.count(w)) continue;
            int j = vocab[w];
            double idfb = log(1 + (N - df[j] + 0.5) / (df[j] + 0.5));
            bm += idfb * tf[i][j] * (k1 + 1) / (tf[i][j] + k1 * (1 - b + b * toks[i].size() / avgdl));
        }
        rb.push_back({bm, i});
    }
    sort(rc.rbegin(), rc.rend()); sort(rb.rbegin(), rb.rend());
    printf("  ranking cosseno TF-IDF:\n");
    for (int k = 0; k < 3; ++k) printf("    %.3f  %s\n", rc[k].first, docs[rc[k].second].c_str());
    printf("  ranking BM25:\n");
    for (int k = 0; k < 3; ++k) printf("    %.3f  %s\n", rb[k].first, docs[rb[k].second].c_str());
    printf("\nLimite do BoW/TF-IDF: 'gato' e 'cachorro' sao tao distintos quanto 'gato' e 'rede' (sem semantica) -> embeddings.\n");
    printf("cos(doc0, doc1) = %.3f (compartilham 'gato')  cos(doc0, doc2) = %.3f\n", cosine(tfidf[0], tfidf[1]), cosine(tfidf[0], tfidf[2]));
}
