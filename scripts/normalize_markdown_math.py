#!/usr/bin/env python3
from __future__ import annotations

from html import unescape
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOT = ROOT / "src" / "content" / "paper_summaries"


MATH_DIV_RE = re.compile(r'<div class="math">\s*(.*?)\s*</div>', re.DOTALL)
DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
INLINE_RE = re.compile(r"\\\((.+?)\\\)")


def split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        return "", text
    end = text.find("\n---\n", 4)
    if end == -1:
        return "", text
    return text[: end + len("\n---\n")], text[end + len("\n---\n") :]


def unwrap_math_div(match: re.Match[str]) -> str:
    body = unescape(match.group(1)).strip()
    display = DISPLAY_RE.fullmatch(body)
    if display:
        body = display.group(1).strip()
    return f"\n\n$$\n{body}\n$$\n\n"


def normalize_body(text: str) -> str:
    text = MATH_DIV_RE.sub(unwrap_math_div, text)

    text = DISPLAY_RE.sub(lambda match: f"\n\n$$\n{match.group(1).strip()}\n$$\n\n", text)
    text = INLINE_RE.sub(lambda match: f"${match.group(1).strip()}$", text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> int:
    changed = 0
    for path in sorted(CONTENT_ROOT.glob("**/*.md")):
        original = path.read_text(encoding="utf-8")
        frontmatter, body = split_frontmatter(original)
        normalized = frontmatter + normalize_body(body)
        if normalized != original:
            path.write_text(normalized, encoding="utf-8")
            changed += 1

    print(f"Normalized math markup in {changed} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
