<h1 align="center">The Truth Lies Somewhere in the Middle<br>(of the Generated Tokens)</h1>

<h3 align="center">
<a href="">Paper</a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://www.sophielwang.com/tokens">Project Page</a>
</h3>

<h5 align="center">
<a href="https://www.sophielwang.com/">Sophie L. Wang</a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://web.mit.edu/phillipi/">Phillip Isola</a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://briancheung.github.io/">Brian Cheung</a>
</h5>

## Installation

```bash
git clone https://github.com/sophicle/tokens.git
cd tokens
conda env create -f environment.yml
conda activate tokens
pip install -e .
```

If a model or dataset requires Hugging Face authentication:

```bash
export HF_TOKEN="your_token_here"
```

## What This Code Does

This repository saves language-model hidden states from generated tokens, saves reference embeddings from another representation space, and computes alignment between them.

The main scripts are:

- `run_embed.py`: save language-model token features or reference embeddings.
- `run_tokenwise_alignment.py`: compute tokenwise and cumulative alignment.
- `scripts/prepare_uniprot.py`: prepare UniProt rows from a UniProt TSV.
- `scripts/embed_uniprot_esm3_structure.py`: create ESM3 structure targets for UniProt rows.

Alignment defaults to linear CKA using the debiased HSIC estimator
(`--metric debiased_cka`). Use `--metric cka` for the standard biased HSIC
estimator, or `--metric mknn` for mutual nearest-neighbor overlap.

## Quick Start

Use `--limit` to cap the number of examples. The example shell scripts default
to `--limit 1024`; override that with, for example,
`bash examples/wit_qwen14b_dinov2.sh --limit 256`.

Generate Qwen token features:

```bash
python run_embed.py \
  --model Qwen/Qwen3-14B \
  --dataset wit_1024 \
  --prompt imagine_see \
  --max-new-tokens 128 \
  --out-root runs_seed0 \
  --seed 0 \
  --do-sample \
  --limit 1024
```

Save DINOv2 image embeddings:

```bash
python run_embed.py \
  --model facebook/dinov2-base \
  --model-kind vision \
  --dataset wit_1024 \
  --out-root runs_seed0 \
  --limit 1024
```

Compute alignment:

```bash
python run_tokenwise_alignment.py \
  --source-dir runs_seed0/wit_1024_imagine/Qwen3-14B_tokens128_see \
  --target-dir runs_seed0/wit_1024_sensory_encoders/dinov2-base \
  --out-dir runs_seed0/wit_1024_imagine/alignment/see_Qwen3-14B
```

Run mKNN instead:

```bash
python run_tokenwise_alignment.py \
  --source-dir runs_seed0/wit_1024_imagine/Qwen3-14B_tokens128_see \
  --target-dir runs_seed0/wit_1024_sensory_encoders/dinov2-base \
  --out-dir runs_seed0/wit_1024_imagine/alignment/see_Qwen3-14B_mknn \
  --metric mknn \
  --ks 10
```

Run biased CKA instead:

```bash
python run_tokenwise_alignment.py \
  --source-dir runs_seed0/wit_1024_imagine/Qwen3-14B_tokens128_see \
  --target-dir runs_seed0/wit_1024_sensory_encoders/dinov2-base \
  --out-dir runs_seed0/wit_1024_imagine/alignment/see_Qwen3-14B_biased \
  --metric cka
```

## Datasets

Built-in dataset names:

- `wit_1024`
- `math_500_problem`
- `math_500_solution`
- `math_500_answer`
- `gpqa_diamond_question`
- `gpqa_diamond_explanation`
- `gpqa_diamond_correct_answer`
- `gpqa_diamond_incorrect_answer_1`
- `gpqa_diamond_incorrect_answer_2`
- `gpqa_diamond_incorrect_answer_3`

Custom data can be loaded with `--dataset csv` or `--dataset jsonl`:

```bash
python run_embed.py \
  --model Qwen/Qwen3-14B \
  --dataset csv \
  --data-file data/examples.csv \
  --id-field sample_id \
  --text-field text \
  --prompt caption \
  --max-new-tokens 0 \
  --out-root runs_seed0
```

CSV and JSONL rows are de-duplicated by `sample_id` before `--limit` is applied.

## Prompts

Built-in prompt templates:

- `caption`: `{text}`
- `imagine_see`: `Imagine what it would look like to see: {text}`
- `math_solve`: `Solve the following problem and give the correct answer: {text}`
- `gpqa_solve`: `Solve the following problem and output answer as \boxed{A/B/C/D}. {text}`
- `protein`: `Provide a thorough summary of {text}. Include its gene name, protein family, molecular weight, known structural domains, function in the cell, binding sites, any known interactions or pathways it participates in.`

Use `--prompt-template` to provide a custom template. Templates can reference `{text}`.

## UniProt

Download a UniProt TSV:

```bash
mkdir -p data/uniprot
curl -L --get "https://rest.uniprot.org/uniprotkb/search" \
  --data-urlencode "query=(structure_3d:true) AND (reviewed:true)" \
  --data-urlencode "format=tsv" \
  --data-urlencode "fields=accession,id,protein_name,gene_names,organism_name,cc_function,xref_pdb" \
  --data-urlencode "size=1024" \
  -o data/uniprot/unparsed_uniprot_db.tsv
```

Parse one row per UniProt/PDB/chain:

```bash
python scripts/prepare_uniprot.py \
  --input-file data/uniprot/unparsed_uniprot_db.tsv \
  --output-file data/uniprot/parsed_uniprot_db.csv \
  --fetch-json-chains
```

Generate protein-summary token features:

```bash
python run_embed.py \
  --model Qwen/Qwen3-14B \
  --dataset csv \
  --data-file data/uniprot/parsed_uniprot_db.csv \
  --id-field sample_id \
  --text-field text \
  --prompt protein \
  --run-name uniprot_protein \
  --max-new-tokens 128 \
  --out-root runs_seed0 \
  --seed 0 \
  --do-sample \
  --limit 1024
```

Generate ESM3 structure targets:

```bash
python scripts/embed_uniprot_esm3_structure.py \
  --data-file data/uniprot/parsed_uniprot_db.csv \
  --output-dir runs_seed0/uniprot_sensory_encoders/ESM3_sm_open_v0_structure \
  --structure-cache-dir data/uniprot/pdb_cache \
  --limit 1024 \
  --device cuda
```

Align generated protein tokens to ESM3 structure targets:

```bash
python run_tokenwise_alignment.py \
  --source-dir runs_seed0/uniprot_protein/Qwen3-14B_tokens128_protein \
  --target-dir runs_seed0/uniprot_sensory_encoders/ESM3_sm_open_v0_structure \
  --out-dir runs_seed0/uniprot_alignment/protein_vs_esm3_structure \
  --prefix protein_vs_esm3_structure \
  --match-key pdb_chain
```

Use `--match-key pdb_chain` when source files are named `ACCESSION_PDB_CHAIN.pt` and target files are named `PDB_CHAIN.pt`.

## Outputs

Generation runs save:

```text
runs_seed0/<dataset>_<prompt>/<model_tokensX_promptKey>/
```

Language-model generation tensors have shape `(tokens, hidden)`. Reference tensors should have shape `(hidden,)` or `(1, hidden)`.

By default, alignment writes:

```text
<prefix>_mean_summary.csv
<prefix>_tokenwise_debiased_cka.csv
<prefix>_tokenwise_cumulative_debiased_cka.csv
<prefix>_paired_samples.csv
```

With `--metric cka`, alignment writes:

```text
<prefix>_tokenwise_cka.csv
<prefix>_tokenwise_cumulative_cka.csv
```

With `--metric mknn`, alignment writes:

```text
<prefix>_tokenwise_mknn_k<k>.csv
<prefix>_tokenwise_cumulative_mknn_k<k>.csv
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@inproceedings{wang2026truth,
  title={The Truth Lies Somewhere in the Middle (of the Generated Tokens)},
  author={Wang, Sophie L. and Isola, Phillip and Cheung, Brian},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026},
  note={To appear}
}
```
