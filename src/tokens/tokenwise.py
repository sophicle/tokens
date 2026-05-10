from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from tokens.io import load_lm_tokens, load_target_vector, parse_ks
from tokens.metrics import compute_metric_values

logger = logging.getLogger(__name__)


def token_at(x: np.ndarray, token: int) -> np.ndarray:
    return x[min(token, x.shape[0] - 1)]


def match_key_from_stem(stem: str, match_key: str) -> str:
    if match_key == "exact":
        return stem
    if match_key == "pdb_chain":
        pieces = stem.split("_")
        if len(pieces) < 2:
            return stem
        return "_".join(pieces[-2:])
    raise ValueError(f"unsupported match key: {match_key}")


def index_paths(paths, match_key: str, role: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    duplicate_keys: set[str] = set()
    for path in sorted(paths):
        key = match_key_from_stem(path.stem, match_key)
        if key in indexed:
            duplicate_keys.add(key)
            continue
        indexed[key] = path
    if duplicate_keys:
        logger.warning(
            "found %d duplicate %s match keys with --match-key %s; keeping first path",
            len(duplicate_keys),
            role,
            match_key,
        )
    return indexed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute tokenwise and cumulative token alignment.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--metric",
        choices=("cka", "debiased_cka", "mknn"),
        default="debiased_cka",
        help=(
            "Alignment metric. cka uses the standard biased HSIC estimator; "
            "debiased_cka uses the debiased HSIC estimator and is the default."
        ),
    )
    parser.add_argument(
        "--match-key",
        choices=("exact", "pdb_chain"),
        default="exact",
        help=(
            "How to match source and target .pt stems. Use pdb_chain for UniProt "
            "sources named ACCESSION_PDB_CHAIN and ESM targets named PDB_CHAIN."
        ),
    )
    parser.add_argument("--ks", default="10", help="Comma-separated k values for mKNN metrics.")
    parser.add_argument("--clip-q", type=float, default=0.95)
    parser.add_argument("--prefix", default="alignment")
    return parser


def save_csvs(results: dict, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("cka", "debiased_cka"):
        if metric not in results["tokenwise"]:
            continue
        pd.DataFrame(
            {
                "token": np.arange(len(results["tokenwise"][metric])),
                metric: results["tokenwise"][metric],
            }
        ).to_csv(out_dir / f"{prefix}_tokenwise_{metric}.csv", index=False)
        pd.DataFrame(
            {
                "token": np.arange(len(results["cumulative"][metric])),
                metric: results["cumulative"][metric],
            }
        ).to_csv(out_dir / f"{prefix}_tokenwise_cumulative_{metric}.csv", index=False)

    if "mknn" in results["tokenwise"]:
        for k, values in results["tokenwise"]["mknn"].items():
            pd.DataFrame(
                {
                    "token": np.arange(len(values)),
                    "mknn": values,
                }
            ).to_csv(out_dir / f"{prefix}_tokenwise_mknn_k{k}.csv", index=False)
        for k, values in results["cumulative"]["mknn"].items():
            pd.DataFrame(
                {
                    "token": np.arange(len(values)),
                    "mknn": values,
                }
            ).to_csv(out_dir / f"{prefix}_tokenwise_cumulative_mknn_k{k}.csv", index=False)

    rows = []
    for metric in ("cka", "debiased_cka"):
        if metric in results["mean"]:
            rows.append({"metric": metric, "k": "", "value": results["mean"][metric]})
    if "mknn" in results["mean"]:
        rows.extend(
            {"metric": "mknn", "k": k, "value": value}
            for k, value in results["mean"]["mknn"].items()
        )
    pd.DataFrame(rows).to_csv(out_dir / f"{prefix}_mean_summary.csv", index=False)


def metric_names(metric: str) -> tuple[str, ...]:
    return (metric,)


def empty_series(metrics: tuple[str, ...], ks: tuple[int, ...], max_tokens: int) -> dict:
    series: dict = {}
    if "cka" in metrics:
        series["cka"] = np.zeros(max_tokens)
    if "debiased_cka" in metrics:
        series["debiased_cka"] = np.zeros(max_tokens)
    if "mknn" in metrics:
        series["mknn"] = {k: np.zeros(max_tokens) for k in ks}
    return series


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()
    metrics = metric_names(args.metric)
    ks = parse_ks(args.ks) if "mknn" in metrics else ()
    source_paths = index_paths(args.source_dir.glob("*.pt"), args.match_key, "source")
    target_paths = index_paths(args.target_dir.glob("*.pt"), args.match_key, "target")
    stems = sorted(source_paths.keys() & target_paths.keys())
    if len(stems) < 4:
        raise RuntimeError(
            "need at least 4 paired samples; found {} using --match-key {} "
            "({} source files, {} target files)".format(
                len(stems),
                args.match_key,
                len(source_paths),
                len(target_paths),
            )
        )
    logger.info("found %d paired samples", len(stems))

    source_tokens: list[np.ndarray] = []
    target_vectors: list[np.ndarray] = []
    kept_pairs: list[dict[str, str]] = []
    for stem in tqdm(stems, desc="loading pairs"):
        try:
            src = load_lm_tokens(source_paths[stem])
            if src.ndim != 2 or src.shape[0] == 0:
                logger.warning("skipping %s: source shape %s", stem, src.shape)
                continue
            tgt = load_target_vector(target_paths[stem])
            source_tokens.append(src)
            target_vectors.append(tgt)
            kept_pairs.append(
                {
                    "sample_id": stem,
                    "source_id": source_paths[stem].stem,
                    "target_id": target_paths[stem].stem,
                }
            )
        except Exception as exc:
            logger.warning("skipping %s: %s", stem, exc)

    if len(source_tokens) < 4:
        raise RuntimeError("need at least 4 valid pairs")

    max_tokens = max(arr.shape[0] for arr in source_tokens)
    for idx, arr in enumerate(source_tokens):
        if arr.shape[0] < max_tokens:
            pad = np.repeat(arr[-1][None, :], max_tokens - arr.shape[0], axis=0)
            source_tokens[idx] = np.concatenate([arr, pad], axis=0)

    targets = np.stack(target_vectors, axis=0)
    mean_features = np.stack([arr.mean(axis=0) for arr in source_tokens], axis=0)
    mean = compute_metric_values(
        mean_features,
        targets,
        metrics=metrics,
        ks=ks,
        clip_q=args.clip_q,
    )

    tokenwise = empty_series(metrics, ks, max_tokens)
    cumulative = empty_series(metrics, ks, max_tokens)

    for token in tqdm(range(max_tokens), desc="token alignment"):
        xt = np.stack([token_at(arr, token) for arr in source_tokens], axis=0)
        res = compute_metric_values(
            xt,
            targets,
            metrics=metrics,
            ks=ks,
            clip_q=args.clip_q,
        )
        xc = np.stack([arr[: token + 1].mean(axis=0) for arr in source_tokens], axis=0)
        res_c = compute_metric_values(
            xc,
            targets,
            metrics=metrics,
            ks=ks,
            clip_q=args.clip_q,
        )
        if "cka" in metrics:
            tokenwise["cka"][token] = res["cka"]
            cumulative["cka"][token] = res_c["cka"]
        if "debiased_cka" in metrics:
            tokenwise["debiased_cka"][token] = res["debiased_cka"]
            cumulative["debiased_cka"][token] = res_c["debiased_cka"]
        if "mknn" in metrics:
            mknn = res["mknn"]
            mknn_c = res_c["mknn"]
            assert isinstance(mknn, dict)
            assert isinstance(mknn_c, dict)
            for k in ks:
                tokenwise["mknn"][k][token] = mknn[k]
                cumulative["mknn"][k][token] = mknn_c[k]

    save_csvs({"mean": mean, "tokenwise": tokenwise, "cumulative": cumulative}, args.out_dir, args.prefix)
    pd.DataFrame(kept_pairs).to_csv(args.out_dir / f"{args.prefix}_paired_samples.csv", index=False)
    logger.info("saved tokenwise alignment CSVs to %s", args.out_dir)


if __name__ == "__main__":
    main()
