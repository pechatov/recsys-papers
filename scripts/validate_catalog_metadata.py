#!/usr/bin/env python3
from __future__ import annotations

from datetime import date
from html import unescape
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "summaries/papers.html")
ENTRY_RE = re.compile(
    r'(<ol\b[^>]*\bstart="[^"]+"[^>]*>.*?</ol>)\s*<ul>(.*?)</ul>',
    flags=re.DOTALL,
)
DATE_RE = re.compile(
    r'<li><strong>(Дата публикации|Дата добавления):</strong>\s*([^<]+)</li>'
)


def entry_title(details_prefix: str) -> str:
    match = re.search(r'<a\b[^>]*>(.*?)</a>', details_prefix, flags=re.DOTALL)
    if match is None:
        return "неизвестная статья"
    return unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()


def valid_iso_date(value: str) -> bool:
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def main() -> int:
    text = CATALOG.read_text(encoding="utf-8")
    errors: list[str] = []

    for legacy_label in ("Год", "Дата arXiv"):
        if f"<strong>{legacy_label}:</strong>" in text:
            errors.append(f"legacy field remains: {legacy_label}")

    matches = list(ENTRY_RE.finditer(text))
    entry_count = len(re.findall(r'<ol\b[^>]*\bstart="', text))
    if len(matches) != entry_count:
        errors.append(f"matched {len(matches)} metadata lists for {entry_count} catalog entries")

    for index, match in enumerate(matches, start=1):
        title = entry_title(match.group(1))
        details = match.group(2)
        fields: dict[str, list[str]] = {}
        for label, value in DATE_RE.findall(details):
            fields.setdefault(label, []).append(value.strip())

        for label in ("Дата публикации", "Дата добавления"):
            values = fields.get(label, [])
            if len(values) != 1:
                errors.append(f"entry {index} ({title}): expected one {label}, found {len(values)}")
                continue
            if not valid_iso_date(values[0]):
                errors.append(f"entry {index} ({title}): invalid {label}: {values[0]!r}")

        publication_position = details.find("<strong>Дата публикации:</strong>")
        added_position = details.find("<strong>Дата добавления:</strong>")
        if publication_position >= 0 and added_position >= 0 and publication_position > added_position:
            errors.append(f"entry {index} ({title}): publication date must precede added date")

    for error in errors:
        print("ERROR", error)
    if errors:
        return 1

    print(f"Validated catalog metadata for {entry_count} papers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
