#!/usr/bin/env bash
# Seis corridas de LLM da decisão de 2026-11-03: zero-shot e few-shot,
# sementes 0, 1 e 2, subamostra estratificada de 1000 do teste oficial.
#
# Não guarda chave. O processo precisa herdar:
#   OPENAI_API_KEY
#   OPENAI_BASE_URL=https://api.groq.com/openai/v1
#   LLM_MODEL=openai/gpt-oss-20b
#
# O cliente manda User-Agent estável. Sem isso a Cloudflare da Groq
# responde 403 com "error code: 1010" para o urllib. 429, 5xx e falha
# transitória de rede repetem com espera limitada; a primeira 429 não
# derruba a semente.
#
# Uso, na raiz do repositório ou de qualquer diretório:
#   scripts/run-llm-gptoss20b-sub1k.sh
#   scripts/run-llm-gptoss20b-sub1k.sh data/sat_dsat_annotation.jsonl
#
# Sem argumento, a carga é --hf (microsoft/WildFeedback).
# Os JSON vão para runs/llm-gptoss20b-sub1k-{zero|few}-seed{N}.json
# runs/ fica fora do git.

set -euo pipefail

cd "$(dirname "$0")/.."

: "${OPENAI_API_KEY:?exporte OPENAI_API_KEY; este script não guarda chave}"
: "${OPENAI_BASE_URL:?exporte OPENAI_BASE_URL=https://api.groq.com/openai/v1}"
: "${LLM_MODEL:?exporte LLM_MODEL=openai/gpt-oss-20b}"

base="${OPENAI_BASE_URL%/}"
if [[ "$base" != "https://api.groq.com/openai/v1" ]]; then
  echo "OPENAI_BASE_URL precisa ser https://api.groq.com/openai/v1 para estas corridas." >&2
  exit 1
fi
if [[ "$LLM_MODEL" != "openai/gpt-oss-20b" ]]; then
  echo "LLM_MODEL=${LLM_MODEL} não é o id fechado openai/gpt-oss-20b." >&2
  exit 1
fi

source_args=(--hf)
if [[ $# -ge 1 ]]; then
  source_args=(--data "$1")
fi

mkdir -p runs
echo "LLM_MODEL=${LLM_MODEL}"
echo "OPENAI_BASE_URL=${base}"
echo "seis corridas: zero e few, sementes 0 1 2, subamostra 1000."

for shot in zero few; do
  for seed in 0 1 2; do
    k_args=()
    if [[ "$shot" == "few" ]]; then
      k_args=(--k 3)
    fi
    python -m user_feedback_analysis.llm.run \
      "${source_args[@]}" \
      --shot "$shot" \
      "${k_args[@]}" \
      --test-fraction 0.12 \
      --test-subsample 1000 \
      --per-class 5000 \
      --strict-per-class \
      --inferences 3 \
      --seed "$seed" \
      --live \
      --output "runs/llm-gptoss20b-sub1k-${shot}-seed${seed}.json"
  done
done
