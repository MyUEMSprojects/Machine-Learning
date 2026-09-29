"""Usando um LLM pré-treinado com Hugging Face Transformers: geração, estratégias de decodificação e log-probabilidades.

pip install torch transformers
Uso: python hf_generate.py [modelo]      (padrão: distilgpt2, ~350 MB; use "sshleifer/tiny-gpt2" para testar rápido)
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

nome = sys.argv[1] if len(sys.argv) > 1 else "distilgpt2"
tok = AutoTokenizer.from_pretrained(nome)
modelo = AutoModelForCausalLM.from_pretrained(nome).eval()
print(f"modelo: {nome}  parâmetros: {sum(p.numel() for p in modelo.parameters()):,}")

prompt = "Machine learning is"
ids = tok(prompt, return_tensors="pt")
print("tokens do prompt:", tok.convert_ids_to_tokens(ids["input_ids"][0]))

# 1) distribuição do próximo token
with torch.no_grad():
    logits = modelo(**ids).logits[0, -1]
p = logits.softmax(-1)
top = p.topk(5)
print("top-5 próximos tokens:", [(tok.decode(i), round(v.item(), 3)) for v, i in zip(top.values, top.indices)])
print("entropia do próximo token (bits):", round(-(p * p.clamp_min(1e-12).log2()).sum().item(), 2))

# 2) estratégias de decodificação
kw = dict(max_new_tokens=25, pad_token_id=tok.eos_token_id)
torch.manual_seed(0)
for rotulo, args in {
    "guloso": dict(do_sample=False),
    "beam search (4)": dict(do_sample=False, num_beams=4, no_repeat_ngram_size=3),
    "temperatura 0.7 + top-p 0.9": dict(do_sample=True, temperature=0.7, top_p=0.9),
    "temperatura 1.5 (caótico)": dict(do_sample=True, temperature=1.5),
}.items():
    out = modelo.generate(**ids, **kw, **args)
    print(f"\n[{rotulo}]\n{tok.decode(out[0], skip_special_tokens=True)}")

# 3) perplexidade de um texto (quão "surpreso" o modelo fica)
for texto in ["The cat sat on the mat.", "Mat the on sat cat the."]:
    e = tok(texto, return_tensors="pt")
    with torch.no_grad():
        loss = modelo(**e, labels=e["input_ids"]).loss
    print(f"perplexidade de '{texto}': {loss.exp().item():.1f}")
