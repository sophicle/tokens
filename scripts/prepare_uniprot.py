#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from pathlib import Path


PDB_RE = re.compile(r"\b[0-9][A-Za-z0-9]{3}\b")


def first_value(row: dict[str, str], *names: str) -> str:
    lower = {key.lower(): value for key, value in row.items()}
    for name in names:
        if name in row and row[name]:
            return row[name]
        value = lower.get(name.lower())
        if value:
            return value
    return ""


def clean_function(text: str) -> str:
    text = text.strip()
    if text.startswith("FUNCTION:"):
        text = text.removeprefix("FUNCTION:").strip()
    return " ".join(text.split())


def pdb_entries_from_text(text: str) -> dict[str, list[tuple[str, str]]]:
    entries: dict[str, list[tuple[str, str]]] = {}
    for token in text.split(";"):
        match = PDB_RE.search(token)
        if not match:
            continue
        pdb_id = match.group(0).upper()
        entries.setdefault(pdb_id, []).extend(chain_ranges_from_text(token))
    return entries


def chain_ranges_from_text(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pattern = re.compile(r"([A-Za-z0-9](?:\s*[/,]\s*[A-Za-z0-9])*)\s*=\s*([0-9?]+-[0-9?]+)")
    for match in pattern.finditer(text):
        chains = re.split(r"\s*[/,]\s*", match.group(1))
        chain_range = match.group(2)
        for chain in chains:
            if chain:
                pairs.append((chain, chain_range))
    return pairs


def fetch_uniprot_pdb_chains(accession: str, sleep: float = 0.0) -> dict[str, list[tuple[str, str]]]:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    if sleep:
        time.sleep(sleep)

    chains_by_pdb: dict[str, list[tuple[str, str]]] = {}
    for ref in payload.get("uniProtKBCrossReferences", []):
        if ref.get("database") != "PDB":
            continue
        pdb_id = str(ref.get("id", "")).upper()
        if not pdb_id:
            continue
        chain_text = ""
        for prop in ref.get("properties", []):
            if prop.get("key") == "Chains":
                chain_text = prop.get("value", "")
                break
        chains_by_pdb[pdb_id] = chain_ranges_from_text(chain_text) or [("", "")]
    return chains_by_pdb


def output_text(protein: str, gene_names: str, organism: str, function: str) -> str:
    parts = []
    if protein:
        parts.append(protein)
    if gene_names:
        parts.append(f"Gene names: {gene_names}.")
    if organism:
        parts.append(f"Organism: {organism}.")
    if function:
        parts.append(f"Function: {function}")
    return " ".join(parts)


def convert(input_file: Path, output_file: Path, fetch_json_chains: bool, sleep: float) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    delimiter = "\t" if output_file.suffix == ".tsv" else ","
    rows_written = 0

    with input_file.open(newline="", encoding="utf-8") as in_handle, output_file.open(
        "w", newline="", encoding="utf-8"
    ) as out_handle:
        reader = csv.DictReader(in_handle, delimiter="\t")
        fieldnames = [
            "sample_id",
            "accession",
            "entry_name",
            "protein_name",
            "gene_names",
            "organism",
            "function",
            "pdb_id",
            "chain_id",
            "chain_range",
            "text",
        ]
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()

        for row in reader:
            accession = first_value(row, "Entry", "accession")
            entry_name = first_value(row, "Entry Name", "id")
            protein = first_value(row, "Protein names", "protein_name")
            gene_names = first_value(row, "Gene Names", "gene_names")
            organism = first_value(row, "Organism", "organism_name")
            function = clean_function(first_value(row, "Function [CC]", "cc_function"))
            pdb_entries = pdb_entries_from_text(first_value(row, "PDB", "xref_pdb"))
            pdb_ids = list(pdb_entries)
            if not accession or not pdb_ids:
                continue

            chains_by_pdb = dict(pdb_entries)
            if fetch_json_chains:
                chains_by_pdb.update(fetch_uniprot_pdb_chains(accession, sleep=sleep))

            text = output_text(protein, gene_names, organism, function)
            for pdb_id in pdb_ids:
                chain_rows = chains_by_pdb.get(pdb_id) or [("", "")]
                for chain_id, chain_range in chain_rows:
                    suffix = f"_{chain_id}" if chain_id else ""
                    writer.writerow(
                        {
                            "sample_id": f"{accession}_{pdb_id}{suffix}",
                            "accession": accession,
                            "entry_name": entry_name,
                            "protein_name": protein,
                            "gene_names": gene_names,
                            "organism": organism,
                            "function": function,
                            "pdb_id": pdb_id,
                            "chain_id": chain_id,
                            "chain_range": chain_range,
                            "text": text,
                        }
                    )
                    rows_written += 1
    return rows_written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse a UniProt TSV into protein/PDB/chain rows.")
    parser.add_argument("--input-file", "--input_file", type=Path, required=True)
    parser.add_argument("--output-file", "--output_file", type=Path, required=True)
    parser.add_argument(
        "--fetch-json-chains",
        action="store_true",
        help="Fetch UniProt JSON entries to fill chain_id and chain_range.",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between JSON requests.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = convert(args.input_file, args.output_file, args.fetch_json_chains, args.sleep)
    print(f"wrote {rows} rows to {args.output_file}")


if __name__ == "__main__":
    main()
