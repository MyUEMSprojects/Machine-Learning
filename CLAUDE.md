# Repositório de estudo de Machine Learning

Objetivo: aprender ML, dos fundamentos a tópicos avançados. Sem app/produto; só material de estudo (texto em **português**).

## Estrutura

- Módulos numerados na raiz (`00-...` a `10-...`), na ordem sugerida de estudo. Cada módulo tem um `README.md` (índice, gerado com tabela de tópicos) e um subdiretório por tópico (`kebab-case`); tópicos podem ter subtópicos (ex.: `02-aprendizado-supervisionado/ensembles/adaboost/`).
- Cada tópico = `README.md` (intuição, matemática, prós/contras, implementação, referências) + código ao lado. Novos tópicos seguem `_template/`.

## Convenções de código

- **C++17**, arquivo único e autocontido (só biblioteca padrão), compila com `g++ -std=c++17 -O2 -Wall x.cpp`. Usado para algoritmos "puros" implementados do zero. Dados sintéticos gerados no código; a saída deve **verificar** o algoritmo (comparar com solução analítica, acurácia, etc.).
- **Python** para demonstrar bibliotecas prontas (scikit-learn, PyTorch, statsmodels, Hugging Face…) e tópicos difíceis de fazer em C++ (deep learning avançado, difusão, GNN, LLMs). Dependências em `requirements.txt`; cada script indica no docstring o que instala.
- Comentários explicam o *porquê* da matemática/algoritmo; sem código morto. Scripts devem rodar em CPU em poucos minutos.
- Ao criar/alterar código, **compile e rode** antes de dizer que funciona; resultados citados nos READMEs devem vir de execuções reais (ou ser qualitativos).
- Ao adicionar um tópico, atualize a tabela do README do módulo (`README.md` do módulo lista C++/Python por tópico) e o índice do README da raiz.
