#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "summaries"
TARGET = ROOT / "public" / "summaries"
PUBLIC_EXCLUDES = {
    ".DS_Store",
    "paper_summaries",
}


def ignore_metadata(_: str, names: list[str]) -> set[str]:
    return {name for name in names if name in PUBLIC_EXCLUDES}


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source directory: {SOURCE}")

    if TARGET.exists():
        shutil.rmtree(TARGET)

    shutil.copytree(SOURCE, TARGET, ignore=ignore_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
