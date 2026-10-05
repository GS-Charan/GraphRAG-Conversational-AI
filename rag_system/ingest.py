from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst"}
SUPPORTED_STRUCTURED_EXTENSIONS = {".json", ".jsonl", ".csv"}


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    text: str


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_json_file(path: Path) -> str:
    data = json.loads(_read_text_file(path))
    return json.dumps(data, ensure_ascii=False, indent=2)


def _read_jsonl_file(path: Path) -> str:
    lines: list[str] = []
    for raw_line in _read_text_file(path).splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        item = json.loads(raw_line)
        lines.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(lines)


def _read_csv_file(path: Path) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            rows.append(" | ".join(cell.strip() for cell in row if cell.strip()))
    return "\n".join(rows)


def load_documents(source_dir: Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in SUPPORTED_TEXT_EXTENSIONS:
            text = _read_text_file(path)
        elif suffix == ".json":
            text = _read_json_file(path)
        elif suffix == ".jsonl":
            text = _read_jsonl_file(path)
        elif suffix == ".csv":
            text = _read_csv_file(path)
        else:
            continue

        cleaned = text.strip()
        if cleaned:
            documents.append(SourceDocument(path=path, text=cleaned))

    return documents
