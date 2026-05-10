#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
import urllib.request
from pathlib import Path

import torch
from biotite.database import rcsb
from esm.pretrained import ESM3_sm_open_v0
from esm.utils.structure.protein_chain import ProteinChain


logger = logging.getLogger(__name__)
VALID_AA = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYX]")


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
    cleaned = re.sub(r"[UZOB]", "X", cleaned)
    return VALID_AA.sub("X", cleaned)


def read_rows(
    path: Path,
    *,
    id_field: str,
    accession_field: str,
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
                    "accession": str(row.get(accession_field, "")).strip(),
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "key": key,
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def sequence_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def fetch_sequence(pdb_id: str, chain_id: str) -> str:
    structure_data = rcsb.fetch(pdb_id, format="pdb")
    try:
        protein_chain = ProteinChain.from_pdb(structure_data, chain_id=chain_id, id=pdb_id)
    except IndexError as exc:
        logger.warning("failed to parse %s_%s exactly, retrying without chain: %s", pdb_id, chain_id, exc)
        protein_chain = ProteinChain.from_pdb(structure_data, id=pdb_id)
    return str(protein_chain.sequence)


def fetch_uniprot_sequence(accession: str) -> str:
    if not accession:
        raise ValueError("missing UniProt accession")
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith(">")]
    sequence = "".join(lines)
    if not sequence:
        raise ValueError(f"empty UniProt FASTA response for {accession}")
    return sequence


def load_or_fetch_sequence(
    row: dict[str, str],
    cache_dir: Path,
    *,
    retries: int,
    sleep: float,
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = sequence_cache_path(cache_dir, row["key"])
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return normalize_sequence(str(payload.get("sequence", "")))

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            sequence = normalize_sequence(fetch_sequence(row["pdb_id"], row["chain_id"]))
            if not sequence:
                raise ValueError("empty sequence")
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump({**row, "sequence": sequence}, handle, indent=2, sort_keys=True)
            return sequence
        except Exception as exc:  # noqa: BLE001 - keep going across flaky RCSB fetches.
            last_error = exc
            logger.warning(
                "sequence fetch failed for %s on attempt %d/%d: %s",
                row["key"],
                attempt,
                retries,
                exc,
            )
            if attempt < retries and sleep > 0:
                time.sleep(sleep)

    logger.warning(
        "falling back to UniProt accession sequence for %s after RCSB failure: %s",
        row["key"],
        last_error,
    )
    for attempt in range(1, retries + 1):
        try:
            sequence = normalize_sequence(fetch_uniprot_sequence(row.get("accession", "")))
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {**row, "sequence": sequence, "sequence_source": "uniprot"},
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            return sequence
        except Exception as exc:  # noqa: BLE001 - keep going across flaky UniProt fetches.
            last_error = exc
            logger.warning(
                "UniProt sequence fetch failed for %s on attempt %d/%d: %s",
                row["key"],
                attempt,
                retries,
                exc,
            )
            if attempt < retries and sleep > 0:
                time.sleep(sleep)
    assert last_error is not None
    raise last_error


def embed_sequence(model, sequence: str, device: str) -> torch.Tensor:
    with torch.no_grad():
        tokens = model.tokenizers.sequence.encode(sequence)
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        output = model(sequence_tokens=token_tensor)
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
    parser = argparse.ArgumentParser(description="Embed UniProt PDB/chain sequences with ESM3.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-cache-dir", type=Path, required=True)
    parser.add_argument("--id-field", default="sample_id")
    parser.add_argument("--accession-field", default="accession")
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
        accession_field=args.accession_field,
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
            manifest_rows.append({**row, "status": "exists", "sequence_length": ""})
            continue
        try:
            sequence = load_or_fetch_sequence(
                row,
                args.sequence_cache_dir,
                retries=args.retries,
                sleep=args.sleep,
            )
            embedding = embed_sequence(model, sequence, args.device)
            torch.save(embedding, out_path)
            manifest_rows.append(
                {
                    **row,
                    "status": "saved",
                    "sequence_length": str(len(sequence)),
                }
            )
            if index % 25 == 0 or index == len(rows):
                logger.info("saved %d / %d ESM3 sequence embeddings", index, len(rows))
        except Exception as exc:  # noqa: BLE001 - record failed chains and continue.
            logger.warning("failed %s: %s", row["key"], exc)
            failures.append({**row, "status": "failed", "error": str(exc)})

    write_manifest(args.output_dir / "manifest.csv", manifest_rows)
    write_manifest(args.output_dir / "failures.csv", failures)
    if failures:
        raise RuntimeError(f"failed to embed {len(failures)} / {len(rows)} rows")


if __name__ == "__main__":
    main()
