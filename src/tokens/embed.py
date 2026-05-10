from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from tokens.data import load_rows, prompt_key_and_template, sanitize_stem
from tokens.models import CausalLMExtractor, load_vision_model


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed text generations or sensory inputs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-kind", choices=("auto", "lm", "vision"), default="auto")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--image-field")
    parser.add_argument("--id-field")
    parser.add_argument("--prompt", default="caption")
    parser.add_argument("--prompt-template")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_model_kind(model_name: str, model_kind: str) -> str:
    if model_kind != "auto":
        return model_kind
    if model_name.startswith("facebook/dinov2") or "vit" in model_name.lower() or "clip" in model_name.lower():
        return "vision"
    return "lm"


def output_dirs(args: argparse.Namespace, prompt_key: str, kind: str) -> tuple[Path, Path]:
    model_name = args.model.rstrip("/").split("/")[-1]
    if args.run_name:
        run_root = args.out_root / args.run_name
    elif kind == "lm":
        prompt_family = args.prompt.split("_")[0] if args.prompt.startswith("imagine_") else args.prompt
        run_root = args.out_root / f"{args.dataset}_{prompt_family}"
    else:
        run_root = args.out_root / f"{args.dataset}_sensory_encoders"

    if kind == "lm":
        suffix = "no_think" if not args.enable_thinking else ""
        pieces = [
            model_name,
            f"tokens{args.max_new_tokens}",
            prompt_key,
            suffix,
        ]
        leaf = "_".join(piece for piece in pieces if piece)
    else:
        leaf = model_name
    out_dir = run_root / leaf
    return run_root, out_dir


def embed_lm(args: argparse.Namespace, rows) -> None:
    hf_token = os.getenv("HF_TOKEN")
    prompt_key, template = prompt_key_and_template(args.prompt, args.prompt_template)
    _, out_dir = output_dirs(args, prompt_key, "lm")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "output").mkdir(exist_ok=True)

    extractor = CausalLMExtractor(
        args.model,
        hf_token=hf_token,
        enable_thinking=args.enable_thinking,
    )
    for row in tqdm(rows, desc=f"embedding {out_dir.name}"):
        stem = sanitize_stem(row.sample_id)
        out_path = out_dir / f"{stem}.pt"
        text_path = out_dir / "output" / f"{stem}.txt"
        if out_path.exists() and text_path.exists() and not args.overwrite:
            continue

        prompt = template.format(text=row.text, caption=row.text)
        if args.max_new_tokens > 0:
            emb, text = extractor.generate_token_features(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            emb, text = extractor.embed_prompt_vector(prompt)

        torch.save(emb, out_path)
        text_path.write_text(text, encoding="utf-8")


def load_image(row):
    if row.image is not None:
        return row.image
    if row.image_path is None:
        raise ValueError(f"row {row.sample_id} has no image or image_path")
    return Image.open(row.image_path).convert("RGB")


def embed_vision(args: argparse.Namespace, rows) -> None:
    hf_token = os.getenv("HF_TOKEN")
    _, out_dir = output_dirs(args, prompt_key="vision", kind="vision")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, processor, device = load_vision_model(args.model, hf_token=hf_token)

    for row in tqdm(rows, desc=f"embedding {out_dir.name}"):
        stem = sanitize_stem(row.sample_id)
        out_path = out_dir / f"{stem}.pt"
        if out_path.exists() and not args.overwrite:
            continue
        image = load_image(row)
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            emb = outputs.last_hidden_state.mean(dim=1)
        elif hasattr(outputs, "pooler_output"):
            emb = outputs.pooler_output
        else:
            raise RuntimeError("vision model output has no last_hidden_state or pooler_output")
        torch.save(emb.detach().cpu(), out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()
    if args.out_root is None:
        args.out_root = Path(f"runs_seed{args.seed}")
    seed_all(args.seed)
    rows = load_rows(
        args.dataset,
        hf_token=os.getenv("HF_TOKEN"),
        data_file=args.data_file,
        text_field=args.text_field,
        image_field=args.image_field,
        id_field=args.id_field,
        seed=args.seed,
        limit=args.limit,
    )
    kind = resolve_model_kind(args.model, args.model_kind)
    if kind == "lm":
        embed_lm(args, rows)
    elif kind == "vision":
        embed_vision(args, rows)
    else:
        raise ValueError(f"unsupported model kind: {kind}")


if __name__ == "__main__":
    main()
