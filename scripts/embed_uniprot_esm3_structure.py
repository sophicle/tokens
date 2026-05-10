#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from pathlib import Path

import torch
from biotite.database import rcsb
from esm.pretrained import ESM3_sm_open_v0
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain


logger = logging.getLogger(__name__)


def sanitize_stem(value: str) -> str:
    keep = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    stem = "".join(keep).strip("_")
    return stem or "sample"


def normalize_sequence(sequence: str) -> str:
    cleaned = re.sub(r"\s+", "", sequence or "").upper()
    return re.sub(r"[UZOB]", "X", cleaned)


def read_rows(
    path: Path,
    *,
    id_field: str,
    pdb_field: str,
    chain_field: str,
    limit: int | None,
) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for index, row in enumerate(reader):
            sample_id = sanitize_stem(row.get(id_field) or f"sample_{index}")
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)

            pdb_id = str(row.get(pdb_field, "")).strip().upper()
            chain_id = str(row.get(chain_field, "")).strip()
            if not pdb_id or not chain_id:
                logger.warning("skipping %s: missing pdb_id or chain_id", sample_id)
                continue
            key = sanitize_stem(f"{pdb_id}_{chain_id}")
            if key in seen_keys:
                logger.warning("skipping duplicate PDB/chain key %s from %s", key, sample_id)
                continue
            seen_keys.add(key)
            rows.append(
                {
                    "sample_id": sample_id,
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "key": key,
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def fetch_chain(pdb_id: str, chain_id: str) -> ProteinChain:
    structure_data = rcsb.fetch(pdb_id, format="pdb")
    try:
        return ProteinChain.from_pdb(structure_data, chain_id=chain_id, id=pdb_id)
    except IndexError as exc:
        logger.warning("failed to parse %s_%s exactly, retrying without chain: %s", pdb_id, chain_id, exc)
        return ProteinChain.from_pdb(structure_data, id=pdb_id)


def load_or_fetch_chain(
    row: dict[str, str],
    cache_dir: Path,
    *,
    retries: int,
    sleep: float,
) -> ProteinChain:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{row['key']}.pdb"
    if cache_path.exists():
        try:
            return ProteinChain.from_pdb(cache_path, chain_id=row["chain_id"], id=row["pdb_id"])
        except IndexError:
            return ProteinChain.from_pdb(cache_path, id=row["pdb_id"])

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            chain = fetch_chain(row["pdb_id"], row["chain_id"])
            cache_path.write_text(chain.to_pdb_string(), encoding="utf-8")
            return chain
        except Exception as exc:  # noqa: BLE001 - keep going across flaky RCSB fetches.
            last_error = exc
            logger.warning(
                "structure fetch failed for %s on attempt %d/%d: %s",
                row["key"],
                attempt,
                retries,
                exc,
            )
            if attempt < retries and sleep > 0:
                time.sleep(sleep)
    assert last_error is not None
    raise last_error


def embed_structure(model, chain: ProteinChain) -> torch.Tensor:
    sequence = normalize_sequence(str(chain.sequence))
    coordinates = torch.as_tensor(chain.atom37_positions, dtype=torch.float32)
    with torch.no_grad():
        encoded = model.encode(ESMProtein(sequence=sequence, coordinates=coordinates))
        structure_tokens = encoded.structure
        if structure_tokens is None:
            raise RuntimeError("ESM3 returned no structure tokens")
        if structure_tokens.ndim == 1:
            structure_tokens = structure_tokens.unsqueeze(0)
        output = model(structure_tokens=structure_tokens)
        return output.embeddings.mean(dim=1).to(torch.float32).cpu().squeeze(0)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed UniProt PDB/chain structures with ESM3.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--structure-cache-dir", type=Path, required=True)
    parser.add_argument("--id-field", default="sample_id")
    parser.add_argument("--pdb-field", default="pdb_id")
    parser.add_argument("--chain-field", default="chain_id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(
        args.data_file,
        id_field=args.id_field,
        pdb_field=args.pdb_field,
        chain_field=args.chain_field,
        limit=args.limit,
    )
    if len(rows) < 4:
        raise RuntimeError(f"need at least 4 rows, found {len(rows)}")

    logger.info("loaded %d unique UniProt PDB/chain rows", len(rows))
    model = ESM3_sm_open_v0(args.device).eval()
    manifest_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        out_path = args.output_dir / f"{row['key']}.pt"
        if out_path.exists() and not args.overwrite:
            manifest_rows.append({**row, "status": "exists"})
            continue
        try:
            chain = load_or_fetch_chain(
                row,
                args.structure_cache_dir,
                retries=args.retries,
                sleep=args.sleep,
            )
            embedding = embed_structure(model, chain)
            torch.save(embedding, out_path)
            manifest_rows.append({**row, "status": "saved", "sequence_length": str(len(chain.sequence))})
            if index % 25 == 0 or index == len(rows):
                logger.info("saved %d / %d ESM3 structure embeddings", index, len(rows))
        except Exception as exc:  # noqa: BLE001 - record failed chains and continue.
            logger.warning("failed %s: %s", row["key"], exc)
            failures.append({**row, "status": "failed", "error": str(exc)})

    write_manifest(args.output_dir / "manifest.csv", manifest_rows)
    write_manifest(args.output_dir / "failures.csv", failures)
    if len(manifest_rows) < 4:
        raise RuntimeError(f"only saved/found {len(manifest_rows)} structure embeddings")


if __name__ == "__main__":
    main()
