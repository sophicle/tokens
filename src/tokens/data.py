from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset


GPQA_FIELD_MAP = {
    "question": "Question",
    "correct_answer": "Correct Answer",
    "incorrect_answer_1": "Incorrect Answer 1",
    "incorrect_answer_2": "Incorrect Answer 2",
    "incorrect_answer_3": "Incorrect Answer 3",
    "explanation": "Explanation",
}


PROMPTS = {
    "caption": ("caption", "{text}"),
    "imagine_see": ("see", "Imagine what it would look like to see: {text}"),
    "math_solve": ("math", "Solve the following problem and give the correct answer: {text}"),
    "gpqa_solve": (
        "gpqa",
        "Solve the following problem and output answer as \\boxed{{A/B/C/D}}. {text}",
    ),
    "protein": (
        "protein",
        "Provide a thorough summary of {text}. Include its gene name, protein family, "
        "molecular weight, known structural domains, function in the cell, binding sites, "
        "any known interactions or pathways it participates in.",
    ),
}


@dataclass(frozen=True)
class Row:
    sample_id: str
    text: str
    image: object | None = None
    image_path: str | None = None


def prompt_key_and_template(name: str, override: str | None = None) -> tuple[str, str]:
    if override is not None:
        return name, override
    if name not in PROMPTS:
        known = ", ".join(sorted(PROMPTS))
        raise ValueError(f"unknown prompt {name!r}; known prompts: {known}")
    return PROMPTS[name]


def sanitize_stem(value: str) -> str:
    keep = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    stem = "".join(keep).strip("_")
    return stem or "sample"


def first_unique_rows(rows: list[Row]) -> list[Row]:
    seen: set[str] = set()
    unique: list[Row] = []
    for row in rows:
        if row.sample_id in seen:
            continue
        seen.add(row.sample_id)
        unique.append(row)
    return unique


def _text_from_wit(row: dict) -> str:
    text = row.get("text", "")
    if isinstance(text, list):
        return text[0] if text else ""
    return str(text)


def _load_wit(hf_token: str | None) -> list[Row]:
    dataset = load_dataset("minhuh/prh", revision="wit_1024", split="train", token=hf_token)
    return [
        Row(sample_id=f"sample_{idx}", text=_text_from_wit(row), image=row.get("image"))
        for idx, row in enumerate(dataset)
    ]


def _load_math500(field: str, hf_token: str | None) -> list[Row]:
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", token=hf_token)
    return [Row(sample_id=f"sample_{idx}", text=str(row[field])) for idx, row in enumerate(dataset)]


def _load_gpqa(field: str, hf_token: str | None, seed: int) -> list[Row]:
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=hf_token)
    if field != "question":
        hf_field = GPQA_FIELD_MAP[field]
        return [
            Row(sample_id=f"sample_{idx}", text=str(row[hf_field]))
            for idx, row in enumerate(dataset)
        ]

    rows: list[Row] = []
    for idx, row in enumerate(dataset):
        answers = [
            ("correct_answer", row["Correct Answer"]),
            ("incorrect_answer_1", row["Incorrect Answer 1"]),
            ("incorrect_answer_2", row["Incorrect Answer 2"]),
            ("incorrect_answer_3", row["Incorrect Answer 3"]),
        ]
        rng = random.Random(seed + idx)
        rng.shuffle(answers)
        letters = ["A", "B", "C", "D"]
        choices = [f"{letter}. {answer}" for letter, (_, answer) in zip(letters, answers)]
        text = "Question:\n{}\n\nChoices:\n{}".format(
            row["Question"],
            "\n".join(choices),
        )
        rows.append(Row(sample_id=f"sample_{idx}", text=text))
    return rows


def _load_csv(path: Path, text_field: str, image_field: str | None, id_field: str | None) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            sample_id = row[id_field] if id_field and row.get(id_field) else f"sample_{idx}"
            rows.append(
                Row(
                    sample_id=sanitize_stem(sample_id),
                    text=str(row.get(text_field, "")),
                    image_path=row.get(image_field) if image_field else None,
                )
            )
    return rows


def _load_jsonl(path: Path, text_field: str, image_field: str | None, id_field: str | None) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            obj = json.loads(line)
            sample_id = obj[id_field] if id_field and obj.get(id_field) else f"sample_{idx}"
            rows.append(
                Row(
                    sample_id=sanitize_stem(sample_id),
                    text=str(obj.get(text_field, "")),
                    image_path=obj.get(image_field) if image_field else None,
                )
            )
    return rows


def load_rows(
    dataset: str,
    *,
    hf_token: str | None = None,
    data_file: Path | None = None,
    text_field: str = "text",
    image_field: str | None = None,
    id_field: str | None = None,
    seed: int = 0,
    limit: int | None = None,
) -> list[Row]:
    if dataset == "wit_1024":
        rows = _load_wit(hf_token)
    elif dataset == "math_500_problem":
        rows = _load_math500("problem", hf_token)
    elif dataset == "math_500_solution":
        rows = _load_math500("solution", hf_token)
    elif dataset == "math_500_answer":
        rows = _load_math500("answer", hf_token)
    elif dataset.startswith("gpqa_diamond_"):
        field = dataset.removeprefix("gpqa_diamond_")
        if field not in GPQA_FIELD_MAP:
            raise ValueError(f"unknown GPQA field: {field}")
        rows = _load_gpqa(field, hf_token, seed=seed)
    elif dataset in {"csv", "jsonl"}:
        if data_file is None:
            raise ValueError(f"--data-file is required for dataset={dataset}")
        rows = (
            _load_csv(data_file, text_field, image_field, id_field)
            if dataset == "csv"
            else _load_jsonl(data_file, text_field, image_field, id_field)
        )
        rows = first_unique_rows(rows)
    else:
        raise ValueError(
            "unsupported dataset {!r}. Use wit_1024, math_500_*, "
            "gpqa_diamond_*, csv, or jsonl.".format(dataset)
        )

    if limit is not None:
        rows = rows[:limit]
    return rows
