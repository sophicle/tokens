#!/usr/bin/env bash
set -euo pipefail

LIMIT=1024
OUT_ROOT="${OUT_ROOT:-runs_seed0}"
DATA_FILE="${DATA_FILE:-data/uniprot/parsed_uniprot_db.csv}"
ESM_TARGET_DIR="${ESM_TARGET_DIR:-${OUT_ROOT}/uniprot_sensory_encoders/ESM3_sm_open_v0_structure}"

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

if [[ ! -d "${ESM_TARGET_DIR}" ]]; then
  echo "missing ESM_TARGET_DIR=${ESM_TARGET_DIR}" >&2
  echo "set ESM_TARGET_DIR to a directory of pre-pooled ESM .pt files for the same PDB/chains" >&2
  exit 1
fi

python run_embed.py \
  --model Qwen/Qwen3-14B \
  --dataset csv \
  --data-file "${DATA_FILE}" \
  --id-field sample_id \
  --text-field text \
  --prompt protein \
  --run-name uniprot_protein \
  --max-new-tokens 128 \
  --out-root "${OUT_ROOT}" \
  --seed 0 \
  --do-sample \
  --limit "${LIMIT}"

python run_tokenwise_alignment.py \
  --source-dir "${OUT_ROOT}/uniprot_protein/Qwen3-14B_tokens128_protein" \
  --target-dir "${ESM_TARGET_DIR}" \
  --out-dir "${OUT_ROOT}/uniprot_alignment/protein_vs_esm3_structure" \
  --prefix protein_vs_esm3_structure \
  --match-key pdb_chain
