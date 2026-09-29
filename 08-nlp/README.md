# Processamento de Linguagem Natural

De contagem de palavras aos LLMs: representação, embeddings, modelos de linguagem, decodificação e fine-tuning.

**Pré-requisitos:** módulos 02 (naive Bayes) e 05 (RNN, Transformer).

## Tópicos

| Tópico | O que cobre | C++ | Python |
|---|---|:-:|:-:|
| [`representacao-texto/`](representacao-texto/) | BoW, TF-IDF, BM25, tokenização BPE | ✅ | ✅ |
| [`embeddings/`](embeddings/) | word2vec (SGNS do zero), PPMI+SVD, sentence embeddings | ✅ | ✅ |
| [`modelos-de-linguagem/`](modelos-de-linguagem/) | n-gramas, perplexidade, LSTM de caracteres | ✅ | ✅ |
| [`llms/`](llms/) | mini-GPT, amostragem (temperatura/top-k/top-p), Hugging Face | ✅ | ✅ |
| [`fine-tuning/`](fine-tuning/) | SFT, LoRA do zero, PEFT | — | ✅ |

Cada tópico tem um `README.md` com a explicação (intuição, matemática, prós e contras, referências) e o código ao lado. Os exemplos em C++ compilam com `g++ -std=c++17 -O2 arquivo.cpp -o arquivo`; os de Python estão descritos em [`../requirements.txt`](../requirements.txt).
