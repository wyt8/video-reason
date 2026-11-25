"""Read STAR annotations and format multiple-choice questions."""

import json
from pathlib import Path
import re


CATEGORIES = ("Interaction", "Sequence", "Prediction", "Feasibility")
LETTERS = "ABCD"


def category(row: dict) -> str:
    value = row["question_id"].split("_", 1)[0]
    if value not in CATEGORIES:
        raise ValueError(f"Unknown question category: {value}")
    return value


def selected_rows(path: Path, limit: int) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError("STAR annotations must contain a JSON array")
    if limit == 0:
        return rows
    groups: dict[str, list[dict]] = {name: [] for name in CATEGORIES}
    for row in rows:
        groups[category(row)].append(row)
    selected = []
    depth = 0
    while len(selected) < limit and any(
        depth < len(group) for group in groups.values()
    ):
        for name in CATEGORIES:
            if depth < len(groups[name]):
                selected.append(groups[name][depth])
                if len(selected) == limit:
                    break
        depth += 1
    return selected


def gold_letter(row: dict) -> str:
    if "answer" not in row:
        raise ValueError(
            f"{row['question_id']}: annotation has no answer; use train or val"
        )
    choices = row["choices"]
    if len(choices) != 4:
        raise ValueError(f"{row['question_id']}: expected exactly four choices")
    matches = [
        index
        for index, choice in enumerate(choices)
        if choice["choice"].strip() == row["answer"].strip()
    ]
    if len(matches) != 1:
        raise ValueError(f"{row['question_id']}: answer matches {len(matches)} choices")
    return LETTERS[matches[0]]


def prompt(row: dict) -> str:
    options = "\n".join(
        f"{LETTERS[i]}. {choice['choice']}" for i, choice in enumerate(row["choices"])
    )
    return f"Watch the video clip and answer the question.\nQuestion: {row['question']}\n{options}\nReply with only one letter: A, B, C, or D."


def parse_prediction(response: str) -> str | None:
    match = re.fullmatch(r"\s*\(?([ABCD])\)?[.\s]*", response, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None
