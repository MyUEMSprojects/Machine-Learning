# Uso:
#   make <nome>            compila (C++) e/ou executa qualquer exemplo pelo nome, sem extensão.  Ex.: make svm
#   make <nome> ARGS="..." repassa argumentos ao programa.                                     Ex.: make hf_generate ARGS=gpt2
#   make list              lista todos os exemplos disponíveis
#   make clean             remove binários C++ compilados (build/) e caches Python
#   make help              mostra esta ajuda
#
# O programa roda dentro da pasta do próprio arquivo. Binários C++ ficam em build/ (ignorado pelo git).
# Python: usa .venv/bin/python se existir; senão python3 (ou defina PYTHON=/caminho/python).

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall
BUILD    := build
ROOT     := $(CURDIR)
PYTHON   ?= $(if $(wildcard $(ROOT)/.venv/bin/python),$(ROOT)/.venv/bin/python,python3)

SOURCES := $(shell find . -type f \( -name '*.cpp' -o -name '*.py' \) -not -path './.git/*' -not -path './.venv/*' -not -path './build/*' | sort)

.DEFAULT_GOAL := help
.PHONY: help list clean

help:
	@echo "Uso:  make <nome_do_arquivo>   (sem extensão; ex.: make svm, make mini_gpt)"
	@echo "      make <nome> ARGS=\"...\"   repassa argumentos ao programa"
	@echo "      make list                lista os exemplos"
	@echo "      make clean               remove build/ e caches Python"

list:
	@for f in $(SOURCES); do echo "$$f"; done | sed 's|^\./||' | awk -F/ '{n=$$NF; sub(/\.[a-z]+$$/,"",n); printf "%-28s %s\n", n, $$0}'

clean:
	rm -rf $(BUILD)
	find . -type d -name __pycache__ -not -path './.git/*' -not -path './.venv/*' -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
	rm -rf 10-topicos-avancados/mlops/artefatos

# Qualquer outro alvo é tratado como nome de exemplo.
%:
	@name="$(basename $@)"; name="$${name%.cpp}"; name="$${name%.py}"; \
	matches=$$(find . -type f \( -name "$$name.cpp" -o -name "$$name.py" \) -not -path './.git/*' -not -path './.venv/*' -not -path './build/*'); \
	count=$$(printf '%s\n' "$$matches" | grep -c .); \
	if [ "$$count" -eq 0 ]; then echo "Nenhum arquivo '$$name.cpp' ou '$$name.py' encontrado. Veja: make list" >&2; exit 1; fi; \
	if [ "$$count" -gt 1 ]; then echo "Nome ambíguo '$$name' (existe em mais de um lugar); renomeie um dos arquivos. Encontrados:" >&2; echo "$$matches" >&2; exit 1; fi; \
	file=$$(echo "$$matches" | sed 's|^\./||'); dir=$$(dirname "$$file"); base=$$(basename "$$file"); \
	case "$$base" in \
	  *.cpp) mkdir -p $(BUILD); \
	         echo ">> compilando $$file"; \
	         $(CXX) $(CXXFLAGS) "$$file" -o "$(BUILD)/$$name" || exit 1; \
	         echo ">> executando $(BUILD)/$$name"; \
	         (cd "$$dir" && "$(ROOT)/$(BUILD)/$$name" $(ARGS));; \
	  *.py)  echo ">> executando $$file com $(PYTHON)"; \
	         (cd "$$dir" && "$(PYTHON)" "$$base" $(ARGS));; \
	esac
