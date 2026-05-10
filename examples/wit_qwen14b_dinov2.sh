#!/usr/bin/env bash
set -euo pipefail

LIMIT=1024
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 [--limit N]" >&2
      exit 1
      ;;
  esac
done

python run_embed.py \
  --model facebook/dinov2-base \
  --model-kind vision \
  --dataset wit_1024 \
  --out-root runs_seed0 \
  --limit "${LIMIT}"

python run_embed.py \
  --model Qwen/Qwen3-14B \
  --dataset wit_1024 \
  --prompt imagine_see \
  --max-new-tokens 128 \
  --out-root runs_seed0 \
  --seed 0 \
  --do-sample \
  --limit "${LIMIT}"

python run_tokenwise_alignment.py \
  --source-dir runs_seed0/wit_1024_imagine/Qwen3-14B_tokens128_see \
  --target-dir runs_seed0/wit_1024_sensory_encoders/dinov2-base \
  --out-dir runs_seed0/wit_1024_imagine/alignment/see_Qwen3-14B
