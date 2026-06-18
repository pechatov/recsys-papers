#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "src" / "content" / "paper_summaries"
CATALOG = ROOT / "summaries" / "papers.html"
REPORT = ROOT / "docs" / "summary_markdown_audit.md"


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    _, fm, body = text.split("---", 2)
    meta: dict[str, str] = {}
    for line in fm.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"')
    return meta, body


def words(text: str) -> int:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"`[^`]*`", " ", text)
    return len(re.findall(r"[\wА-Яа-яЁё]+", text))


def images(text: str) -> list[str]:
    result = re.findall(r'<img\s+[^>]*src="([^"]+)"', text)
    result.extend(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text))
    return result


def headings(text: str) -> set[str]:
    return {match.group(1).lower() for match in re.finditer(r"^#{2,4}\s+(.+)$", text, flags=re.M)}


def has_any_heading(headings_set: set[str], needles: list[str]) -> bool:
    joined = "\n".join(headings_set)
    return any(needle in joined for needle in needles)


def main() -> int:
    rows = []
    image_prefixes: dict[str, Counter[str]] = {}
    for path in sorted(CONTENT.glob("**/*.md")):
        text = path.read_text(encoding="utf-8")
        meta, body = split_frontmatter(text)
        imgs = images(body)
        h = headings(body)
        prefix_counter: Counter[str] = Counter()
        for img in imgs:
            match = re.search(r"\.\./\.\./assets/([^/]+)/", img)
            if match:
                prefix_counter[match.group(1)] += 1
        if prefix_counter:
            image_prefixes[str(path.relative_to(ROOT))] = prefix_counter
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "title": meta.get("title", path.stem),
                "words": words(body),
                "images": len(imgs),
                "method": has_any_heading(h, ["метод", "method", "pipeline", "architecture", "механика"]),
                "experiments": has_any_heading(h, ["experiment", "эксперимент", "result", "результ", "evaluation", "оцен"]),
                "limitations": has_any_heading(h, ["limitation", "огранич", "weakness", "risk", "риск"]),
            }
        )

    short = [row for row in rows if row["words"] < 1200]
    no_images = [row for row in rows if row["images"] == 0]
    missing_core_sections = [
        row
        for row in rows
        if not (row["method"] and row["experiments"] and row["limitations"])
    ]
    mixed_prefixes = {
        path: counter
        for path, counter in image_prefixes.items()
        if len(counter) > 1
    }
    unresolved_abstracts = CATALOG.read_text(encoding="utf-8").count("Abstract не удалось")

    lines = [
        "# Markdown summary audit",
        "",
        "Generated from local Markdown content. This is a structural audit, not a full paper-reading correctness review.",
        "",
        "## Counts",
        "",
        f"- Markdown summaries: {len(rows)}",
        f"- Summaries with at least one image: {sum(1 for row in rows if row['images'] > 0)}",
        f"- Summaries without images: {len(no_images)}",
        f"- Summaries under 1200 words: {len(short)}",
        f"- Summaries missing one of method/experiments/limitations headings: {len(missing_core_sections)}",
        f"- Catalog missing-summary placeholders without resolved abstract: {unresolved_abstracts}",
        f"- Summaries with mixed image asset prefixes: {len(mixed_prefixes)}",
        "",
        "## Priority Review Queue",
        "",
        "These summaries should be checked first with `cs-paper-reading`: short files, files without figures, or files missing core technical sections.",
        "",
        "| Words | Images | Missing core sections | Summary |",
        "| ---: | ---: | :---: | --- |",
    ]

    priority = sorted(
        rows,
        key=lambda row: (
            row["images"] > 0,
            row["words"] >= 1200,
            row["method"] and row["experiments"] and row["limitations"],
            row["words"],
        ),
    )
    for row in priority[:60]:
        missing = []
        if not row["method"]:
            missing.append("method")
        if not row["experiments"]:
            missing.append("experiments")
        if not row["limitations"]:
            missing.append("limitations")
        lines.append(
            f"| {row['words']} | {row['images']} | {', '.join(missing) or '-'} | `{row['path']}` |"
        )

    lines.extend(["", "## Image Prefix Check", ""])
    if mixed_prefixes:
        for path, counter in sorted(mixed_prefixes.items()):
            prefixes = ", ".join(f"{name}={count}" for name, count in counter.items())
            lines.append(f"- `{path}`: {prefixes}")
    else:
        lines.append("- No mixed image prefixes detected among summaries with images.")

    lines.extend(["", "## Summaries Without Images", ""])
    for row in no_images:
        lines.append(f"- `{row['path']}`")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
