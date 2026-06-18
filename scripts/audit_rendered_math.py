#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DIST_SUMMARIES = ROOT / "dist" / "summaries" / "paper_summaries"


CHECKS = {
    "display_math_rendered_as_plain_paragraph": re.compile(r"<p>\s*\["),
    "markdown_emphasis_inside_formula": re.compile(r"\$[^<\n]*(?:<em>|</em>)|<em>\{\\(?:mathrm|text)"),
    "lost_inline_math_delimiter": re.compile(r"(?<!\\)\(\\[A-Za-z]+"),
    "malformed_raw_math_html": re.compile(r'=""'),
}

INLINE_MATH_RE = re.compile(r'<span class="tex-inline">.*?</span>')


def main() -> int:
    problems: dict[str, list[tuple[Path, int, str]]] = defaultdict(list)
    for path in sorted(DIST_SUMMARIES.glob("**/*.html")):
        rel = path.relative_to(ROOT)
        in_display_math = False
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if in_display_math:
                if "</div>" in line:
                    in_display_math = False
                continue
            if '<div class="math"' in line:
                if "</div>" not in line:
                    in_display_math = True
                continue

            line_to_check = INLINE_MATH_RE.sub("", line)
            for name, pattern in CHECKS.items():
                if pattern.search(line_to_check):
                    problems[name].append((rel, lineno, line.strip()[:240]))

    if not problems:
        print("No rendered math formatting problems found.")
        return 0

    for name, hits in problems.items():
        print(f"\n{name}: {len(hits)}")
        for rel, lineno, snippet in hits[:50]:
            print(f"  {rel}:{lineno}: {snippet}")
        if len(hits) > 50:
            print(f"  ... {len(hits) - 50} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
