#!/usr/bin/env python3
"""Build the multi-venue conference archive from structured proceedings TOCs.

ACM RecSys entries come from the official accepted-contribution pages collected
by ``update_recsys_conference_catalog.py``. Other published editions are read
from DBLP's structured conference tables of contents; when 2026 proceedings are
not indexed yet, configured official accepted-paper catalogs are used instead.
Every user-facing link points to the canonical paper or official program source.

The command writes a small manifest to ``src/data`` and one minified JSON file
per venue/year to ``public/data/conference-papers``. Complete proceedings are
kept for the six venues treated as core recommender-systems venues; broad venues
are reduced to papers with explicit recommender-systems evidence in the title or
track name. The split keeps the static site responsive without losing source
coverage information.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import date
import gc
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover - maintainer-facing failure path
    raise SystemExit(
        "Beautiful Soup is required: python3 -m pip install beautifulsoup4"
    ) from exc

from update_recsys_conference_catalog import INDUSTRY_RE, infer_tags, slugify


ROOT = Path(__file__).resolve().parents[1]
DBLP_CACHE_DIR = ROOT / ".cache" / "conference-archive" / "dblp"
OFFICIAL_CACHE_DIR = ROOT / ".cache" / "conference-archive" / "official"
OUTPUT_DIR = ROOT / "public" / "data" / "conference-papers"
MANIFEST_PATH = ROOT / "src" / "data" / "conference-archive.json"
RECSYS_PATH = ROOT / "src" / "data" / "recsys-conferences.json"
ABSTRACT_CACHE_PATH = ROOT / ".cache" / "conference-archive" / "abstracts.json"
DBLP_MIRRORS = (
    "https://dblp.org/",
    "https://dblp.dagstuhl.de/",
    "https://dblp.uni-trier.de/",
)
YEARS = (2024, 2025, 2026)

# These venues are intentionally kept in full. Every other venue in ``VENUES``
# is broad enough that the generated catalog applies the auditable topic filter
# below. Keep this set aligned with the policy shown on the archive page.
PROFILE_VENUE_IDS = frozenset({"recsys", "cikm", "ecir", "kdd", "sigir", "umap"})

RECSYS_CATEGORY_RE = re.compile(
    r"\b(?:rec\s*sys|recommender(?:\s+systems?)?|recommendation(?:s|\s+systems?)?)\b",
    re.IGNORECASE,
)
RECSYS_SYSTEM_RE = re.compile(
    r"\b(?:rec\s*sys|recommender(?:s|\s+systems?)?|collaborative filtering)\b",
    re.IGNORECASE,
)
RECOMMENDATION_WORD_RE = re.compile(
    r"\brecommend(?:ation(?:s)?|ed|ing|s)?\b",
    re.IGNORECASE,
)
NON_RECSYS_RECOMMENDATION_RE = re.compile(
    r"\b(?:"
    r"label recommendation|action recommendations?|policy recommendations?|"
    r"duration recommendation|to[- ]do recommendations?|index recommendation|"
    r"resource recommendation for big data queries|"
    r"(?:design|methodological|actionable) recommendations?|"
    r"(?:perspectives|insights|principles|guidelines) and recommendations?|"
    r"(?:three|four|five) recommendations?|"
    r"recommendations? for (?:design(?:ing)?|evaluation|fostering|presenting|"
    r"ageing|responsible innovation)"
    r")\b",
    re.IGNORECASE,
)
RECSYS_TITLE_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"user[- ]item|item[- ]to[- ]item|next[- ]item|next[- ]poi|"
    r"implicit feedback|rating prediction|"
    r"user preference model(?:ing|ling)?|personalized (?:ranking|retrieval|search|feed)|"
    r"click[- ]through rate|ctr prediction|conversion rate prediction|"
    r"watch time prediction|feed ranking|playlist continuation|"
    r"filter bubbles?|shilling attacks?|item cold[- ]start|cold[- ]start (?:users?|items?)"
    r")\b",
    re.IGNORECASE,
)
MATRIX_FACTORIZATION_RE = re.compile(r"\bmatrix factorization\b", re.IGNORECASE)
MATRIX_FACTORIZATION_CONTEXT_RE = re.compile(
    r"\b(?:users?|items?|ratings?|implicit feedback)\b", re.IGNORECASE
)
SEMANTIC_ITEM_ID_RE = re.compile(
    r"(?:\bsemantic (?:ids?|identifiers?)\b.*\b(?:items?|users?|generative retrieval)\b|"
    r"\b(?:items?|users?|generative retrieval)\b.*\bsemantic (?:ids?|identifiers?)\b)",
    re.IGNORECASE,
)


def edition_meta(
    website: str,
    dates: str,
    location: str,
    note: str | None = None,
    expected_publication_date: str | None = None,
) -> dict:
    result = {"website": website, "dates": dates, "location": location}
    if note:
        result["note"] = note
    if expected_publication_date:
        result["expectedPublicationDate"] = expected_publication_date
    return result


VENUES = OrderedDict(
    [
        (
            "aaai",
            {
                "name": "AAAI",
                "fullName": "AAAI Conference on Artificial Intelligence",
                "rank": "A*",
                "stream": "aaai",
                "keyPattern": r"^{year}$",
                "years": {
                    2024: edition_meta("https://aaai.org/aaai-conference/aaai-24/", "20–27 Feb 2024", "Vancouver, Canada"),
                    2025: edition_meta("https://aaai.org/conference/aaai/aaai-25/", "25 Feb–4 Mar 2025", "Philadelphia, USA"),
                    2026: edition_meta("https://aaai.org/conference/aaai/aaai-26/", "20–27 Jan 2026", "Singapore"),
                },
            },
        ),
        (
            "chi",
            {
                "name": "CHI",
                "fullName": "ACM Conference on Human Factors in Computing Systems",
                "rank": "A*",
                "stream": "chi",
                "keyPattern": r"^{year}(?:a)?$",
                "years": {
                    2024: edition_meta("https://chi2024.acm.org/", "11–16 May 2024", "Honolulu, Hawaii, USA"),
                    2025: edition_meta("https://chi2025.acm.org/", "26 Apr–1 May 2025", "Yokohama, Japan"),
                    2026: edition_meta("https://chi2026.acm.org/", "13–17 Apr 2026", "Barcelona, Spain"),
                },
            },
        ),
        (
            "cikm",
            {
                "name": "CIKM",
                "fullName": "ACM International Conference on Information and Knowledge Management",
                "rank": "A",
                "stream": "cikm",
                "keyPattern": r"^{year}$",
                "years": {
                    2024: edition_meta("https://www.cikm2024.org/", "21–25 Oct 2024", "Boise, Idaho, USA"),
                    2025: edition_meta("https://cikm2025.org/", "10–14 Nov 2025", "Seoul, South Korea"),
                    2026: edition_meta(
                        "https://cikm2026.diag.uniroma1.it/",
                        "7–11 Nov 2026",
                        "Rome, Italy",
                        expected_publication_date="2026-11-07",
                    ),
                },
            },
        ),
        (
            "ecai",
            {
                "name": "ECAI",
                "fullName": "European Conference on Artificial Intelligence",
                "rank": "A",
                "stream": "ecai",
                "keyPattern": r"^{year}$",
                "years": {
                    2024: edition_meta("https://www.ecai2024.eu/", "19–24 Oct 2024", "Santiago de Compostela, Spain"),
                    2025: edition_meta("https://ecai2025.org/", "25–30 Oct 2025", "Bologna, Italy"),
                    2026: edition_meta(
                        "https://2026.ijcai.org/",
                        "15–21 Aug 2026",
                        "Bremen, Germany",
                        "The 2026 event is organized jointly as IJCAI-ECAI; papers are listed once under IJCAI when proceedings become available.",
                        expected_publication_date="2026-08-15",
                    ),
                },
            },
        ),
        (
            "ecir",
            {
                "name": "ECIR",
                "fullName": "European Conference on Information Retrieval",
                "rank": "A",
                "stream": "ecir",
                "keyPattern": r"^{year}-\d+$",
                "years": {
                    2024: edition_meta("https://www.ecir2024.org/", "24–28 Mar 2024", "Glasgow, UK"),
                    2025: edition_meta("https://ecir2025.eu/", "6–10 Apr 2025", "Lucca, Italy"),
                    2026: edition_meta("https://ecir2026.eu/", "29 Mar–2 Apr 2026", "Delft, The Netherlands"),
                },
            },
        ),
        (
            "ecml-pkdd",
            {
                "name": "ECML/PKDD",
                "fullName": "European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases",
                "rank": "A",
                "stream": "pkdd",
                "keyPattern": r"^{year}-\d+$",
                "years": {
                    2024: edition_meta("https://ecmlpkdd.org/2024/", "9–13 Sep 2024", "Vilnius, Lithuania"),
                    2025: edition_meta("https://ecmlpkdd.org/2025/", "15–19 Sep 2025", "Porto, Portugal"),
                    2026: edition_meta(
                        "https://ecmlpkdd.org/2026/",
                        "7–11 Sep 2026",
                        "Naples, Italy",
                        expected_publication_date="2026-09-07",
                    ),
                },
            },
        ),
        (
            "icde",
            {
                "name": "ICDE",
                "fullName": "IEEE International Conference on Data Engineering",
                "rank": "A*",
                "stream": "icde",
                "keyPattern": r"^{year}$",
                "years": {
                    2024: edition_meta("https://icde2024.github.io/", "13–16 May 2024", "Utrecht, The Netherlands"),
                    2025: edition_meta("https://ieee-icde.org/2025/", "19–23 May 2025", "Hong Kong"),
                    2026: edition_meta("https://icde2026.github.io/", "4–8 May 2026", "Montréal, Canada"),
                },
            },
        ),
        (
            "icdm",
            {
                "name": "ICDM",
                "fullName": "IEEE International Conference on Data Mining",
                "rank": "A*",
                "stream": "icdm",
                "keyPattern": r"^{year}$",
                "years": {
                    2024: edition_meta("https://icdm2024.org/", "9–12 Dec 2024", "Abu Dhabi, UAE"),
                    2025: edition_meta("https://www3.cs.stonybrook.edu/~icdm2025/", "12–15 Nov 2025", "Washington, DC, USA"),
                    2026: edition_meta(
                        "http://icdm2026.neu.edu.cn/",
                        "12–15 Nov 2026",
                        "Shenyang, China",
                        expected_publication_date="2026-11-12",
                    ),
                },
            },
        ),
        (
            "icml",
            {
                "name": "ICML",
                "fullName": "International Conference on Machine Learning",
                "rank": "A*",
                "stream": "icml",
                "keyPattern": r"^{year}(?:p)?$",
                "years": {
                    2024: edition_meta("https://icml.cc/Conferences/2024/", "21–27 Jul 2024", "Vienna, Austria"),
                    2025: edition_meta("https://icml.cc/Conferences/2025/", "13–19 Jul 2025", "Vancouver, Canada"),
                    2026: edition_meta("https://icml.cc/Conferences/2026/", "6–12 Jul 2026", "Seoul, South Korea"),
                },
            },
        ),
        (
            "ijcai",
            {
                "name": "IJCAI",
                "fullName": "International Joint Conference on Artificial Intelligence",
                "rank": "A*",
                "stream": "ijcai",
                "keyPattern": r"^{year}$",
                "years": {
                    2024: edition_meta("https://ijcai24.org/", "3–9 Aug 2024", "Jeju, South Korea"),
                    2025: edition_meta("https://2025.ijcai.org/", "16–22 Aug 2025", "Montréal, Canada"),
                    2026: edition_meta("https://2026.ijcai.org/", "15–21 Aug 2026", "Bremen, Germany"),
                },
            },
        ),
        (
            "kdd",
            {
                "name": "KDD",
                "fullName": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
                "rank": "A*",
                "stream": "kdd",
                "keyPattern": r"^{year}(?:-\d+)?$",
                "partialYears": {2026},
                "years": {
                    2024: edition_meta("https://kdd2024.kdd.org/", "25–29 Aug 2024", "Barcelona, Spain"),
                    2025: edition_meta("https://kdd2025.kdd.org/", "3–7 Aug 2025", "Toronto, Canada"),
                    2026: edition_meta(
                        "https://kdd2026.kdd.org/",
                        "9–13 Aug 2026",
                        "Jeju, South Korea",
                        "DBLP currently indexes Volume 1 only; the archive is marked partial until all volumes appear.",
                    ),
                },
            },
        ),
        (
            "neurips",
            {
                "name": "NeurIPS",
                "fullName": "Conference on Neural Information Processing Systems",
                "rank": "A*",
                "stream": "nips",
                "keyPattern": r"^{year}$",
                "years": {
                    2024: edition_meta("https://neurips.cc/Conferences/2024/", "9–15 Dec 2024", "Vancouver, Canada"),
                    2025: edition_meta("https://neurips.cc/Conferences/2025/", "2–7 Dec 2025", "San Diego, USA"),
                    2026: edition_meta(
                        "https://neurips.cc/",
                        "6–12 Dec 2026",
                        "Sydney, Australia",
                        expected_publication_date="2026-12-06",
                    ),
                },
            },
        ),
        (
            "sigir",
            {
                "name": "SIGIR",
                "fullName": "International ACM SIGIR Conference on Research and Development in Information Retrieval",
                "rank": "A*",
                "stream": "sigir",
                "keyPattern": r"^{year}$",
                "years": {
                    2024: edition_meta("https://sigir-2024.github.io/", "14–18 Jul 2024", "Washington, DC, USA"),
                    2025: edition_meta("https://sigir2025.dei.unipd.it/", "13–18 Jul 2025", "Padua, Italy"),
                    2026: edition_meta("https://sigir2026.org/en-AU", "20–24 Jul 2026", "Melbourne, Australia"),
                },
            },
        ),
        (
            "umap",
            {
                "name": "UMAP",
                "fullName": "ACM Conference on User Modeling, Adaptation and Personalization",
                "rank": "A",
                "stream": "um",
                "keyPattern": r"^{year}(?:a)?$",
                "years": {
                    2024: edition_meta("https://www.um.org/umap2024/", "1–4 Jul 2024", "Cagliari, Italy"),
                    2025: edition_meta("https://www.um.org/umap2025/", "16–19 Jun 2025", "New York City, USA"),
                    2026: edition_meta("https://www.um.org/umap2026/", "8–11 Jun 2026", "Gothenburg, Sweden"),
                },
            },
        ),
        (
            "wsdm",
            {
                "name": "WSDM",
                "fullName": "ACM International Conference on Web Search and Data Mining",
                "rank": "A",
                "stream": "wsdm",
                "keyPattern": r"^{year}(?:c)?$",
                "years": {
                    2024: edition_meta("https://www.wsdm-conference.org/2024/", "4–8 Mar 2024", "Mérida, Mexico"),
                    2025: edition_meta("https://www.wsdm-conference.org/2025/", "10–14 Mar 2025", "Hannover, Germany"),
                    2026: edition_meta("https://www.wsdm-conference.org/2026/", "22–26 Feb 2026", "Boise, Idaho, USA"),
                },
            },
        ),
        (
            "www",
            {
                "name": "WWW",
                "fullName": "The Web Conference",
                "rank": "A*",
                "stream": "www",
                "keyPattern": r"^{year}(?:c)?$",
                "years": {
                    2024: edition_meta("https://www2024.thewebconf.org/", "13–17 May 2024", "Singapore"),
                    2025: edition_meta("https://www2025.thewebconf.org/", "28 Apr–2 May 2025", "Sydney, Australia"),
                    2026: edition_meta(
                        "https://www2026.thewebconf.org/",
                        "29 Jun–3 Jul 2026",
                        "Dubai, UAE",
                        "The conference was rescheduled from April to 29 June–3 July 2026.",
                    ),
                },
            },
        ),
    ]
)


def strip_dblp_suffix(name: str) -> str:
    return re.sub(r"\s+\d{4}$", "", name.strip())


def normalized_text(element) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()


def json_dump(path: Path, value: object, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    else:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    path.write_text(payload, encoding="utf-8")


class DblpDownloader:
    def __init__(self, cache_dir: Path, refresh: bool = False) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.last_request_at = 0.0
        self.request_number = 0

    def get(self, relative_url: str) -> str:
        cache_path = self.cache_dir / relative_url
        if cache_path.exists() and not self.refresh:
            return cache_path.read_text(encoding="utf-8", errors="replace")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        failures: list[str] = []
        for attempt in range(6):
            mirror = DBLP_MIRRORS[(self.request_number + attempt) % len(DBLP_MIRRORS)]
            elapsed = time.monotonic() - self.last_request_at
            if elapsed < 2.2:
                time.sleep(2.2 - elapsed)
            request = Request(
                mirror + relative_url,
                headers={"User-Agent": "recsys-papers-catalog/1.0 (conference archive updater)"},
            )
            self.last_request_at = time.monotonic()
            try:
                with urlopen(request, timeout=60) as response:
                    payload = response.read().decode("utf-8", errors="replace")
                if "429 Too Many Requests" in payload:
                    raise RuntimeError("HTTP 429 body")
                if not payload.lstrip().startswith("<bht"):
                    raise RuntimeError(f"unexpected DBLP response ({len(payload)} bytes)")
                cache_path.write_text(payload, encoding="utf-8")
                self.request_number += 1
                return payload
            except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                failures.append(f"{mirror}: {exc}")
                time.sleep(min(5 + attempt * 3, 20))
        raise RuntimeError(f"Could not download {relative_url}: {'; '.join(failures)}")


class OfficialDownloader:
    """Download and cache official accepted-paper pages and data exports."""

    def __init__(self, cache_dir: Path, refresh: bool = False) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh

    def get(
        self,
        url: str,
        cache_name: str,
        headers: dict[str, str] | None = None,
    ) -> str:
        cache_path = self.cache_dir / cache_name
        if cache_path.exists() and not self.refresh:
            return cache_path.read_text(encoding="utf-8", errors="replace")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        failures: list[str] = []
        for attempt in range(4):
            request_headers = {
                    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
                    "Accept-Encoding": "identity",
                    "User-Agent": "recsys-papers-catalog/1.0 (conference archive updater)",
                }
            request_headers.update(headers or {})
            request = Request(url, headers=request_headers)
            try:
                with urlopen(request, timeout=240) as response:
                    payload = response.read().decode("utf-8", errors="replace")
                if not payload.strip():
                    raise RuntimeError("empty response")
                cache_path.write_text(payload, encoding="utf-8")
                return payload
            except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                failures.append(str(exc))
                time.sleep(min(4 + attempt * 3, 12))
        raise RuntimeError(f"Could not download {url}: {'; '.join(failures)}")


def proceedings_from_index(html: str, venue: dict, year: int) -> list[dict]:
    soup = BeautifulSoup(html, "xml")
    pattern = re.compile(venue["keyPattern"].format(year=year))
    results: list[dict] = []
    for proceeding in soup.find_all("proceedings"):
        key = proceeding.get("key", "")
        suffix = key.rsplit("/", 1)[-1]
        if not pattern.fullmatch(suffix):
            continue
        url_element = proceeding.find("url")
        url = normalized_text(url_element)
        if not url.startswith("db/conf/") or not url.endswith(".html"):
            continue
        ee_values = [normalized_text(ee) for ee in proceeding.find_all("ee")]
        results.append(
            {
                "key": key,
                "booktitle": normalized_text(proceeding.find("booktitle")),
                "title": normalized_text(proceeding.find("title")),
                "url": url,
                "xml": url.removesuffix(".html") + ".xml",
                "publisherUrls": [value for value in ee_values if value],
            }
        )
    soup.decompose()
    return results


def category_name_for_volume(raw_name: str, volume: dict) -> str:
    booktitle = volume["booktitle"]
    if any(label in booktitle for label in ("Extended Abstracts", "Adjunct", "Companion", "Position Papers")):
        return f"{booktitle} — {raw_name}"
    return raw_name


def preferred_paper_url(entry) -> tuple[str, str | None]:
    ee_values = [normalized_text(ee) for ee in entry.find_all("ee")]
    doi_url = next(
        (value for value in ee_values if value.lower().startswith("https://doi.org/")),
        None,
    )
    if doi_url:
        return doi_url, doi_url
    open_url = next(
        (normalized_text(ee) for ee in entry.find_all("ee") if ee.get("type") == "oa"),
        None,
    )
    if open_url:
        return open_url, None
    if ee_values:
        return ee_values[0], None
    key = entry.get("key", "")
    return f"https://dblp.org/rec/{key}", None


def recsys_relevance_reason(title: str, category: str) -> str | None:
    """Return the metadata signal that makes a broad-venue paper in scope.

    The archive deliberately uses a conservative, reproducible metadata rule.
    A clearly named recommender-systems track is sufficient. Outside such a
    track, the title must name a recommender-system concept. A small exclusion
    list handles common non-system senses of "recommendation".
    """

    if RECSYS_CATEGORY_RE.search(category):
        return "recsys-track"
    if RECSYS_SYSTEM_RE.search(title):
        return "recsys-title"
    if RECOMMENDATION_WORD_RE.search(title):
        if NON_RECSYS_RECOMMENDATION_RE.search(title):
            return None
        return "recommendation-title"
    if RECSYS_TITLE_SIGNAL_RE.search(title):
        return "recsys-title-signal"
    if MATRIX_FACTORIZATION_RE.search(title) and MATRIX_FACTORIZATION_CONTEXT_RE.search(title):
        return "recsys-title-signal"
    if SEMANTIC_ITEM_ID_RE.search(title):
        return "recsys-title-signal"
    return None


def apply_topic_selection(edition: dict) -> dict:
    """Keep full profile venues and filter broad venues in place."""

    source_paper_count = sum(
        len(category["papers"]) for category in edition.get("categories", [])
    )
    edition["sourcePaperCount"] = source_paper_count
    if edition["venueId"] in PROFILE_VENUE_IDS:
        edition["selectionPolicy"] = "complete-profile-venue"
        edition["filteredOutCount"] = 0
        return edition

    selected_categories: list[dict] = []
    for category in edition.get("categories", []):
        selected_papers = [
            paper
            for paper in category["papers"]
            if recsys_relevance_reason(paper["title"], category["name"])
        ]
        if not selected_papers:
            continue
        selected_category = dict(category)
        selected_category["papers"] = selected_papers
        selected_categories.append(selected_category)

    edition["categories"] = selected_categories
    selected_count = sum(len(category["papers"]) for category in selected_categories)
    edition["selectionPolicy"] = "recsys-related-only"
    edition["filteredOutCount"] = source_paper_count - selected_count
    return edition


def load_abstract_cache() -> dict[str, dict]:
    if not ABSTRACT_CACHE_PATH.exists():
        return {}
    payload = json.loads(ABSTRACT_CACHE_PATH.read_text(encoding="utf-8"))
    return payload.get("papers", {})


def apply_cached_abstracts(edition: dict, cache: dict[str, dict]) -> dict:
    abstract_count = 0
    paper_count = 0
    for category in edition.get("categories", []):
        for paper in category["papers"]:
            paper_count += 1
            record = cache.get(paper["id"])
            paper.pop("takeaway", None)
            paper["abstract"] = record.get("abstract") if record else None
            if paper["abstract"]:
                abstract_count += 1
    edition["abstractCount"] = abstract_count
    edition["missingAbstractCount"] = paper_count - abstract_count
    return edition


def parse_volume(html: str, volume: dict, edition_id: str) -> list[tuple[str, dict]]:
    soup = BeautifulSoup(html, "xml")
    categorized_entries: set[int] = set()
    papers: list[tuple[str, dict]] = []

    def add_entries(raw_category: str, entries: list) -> None:
        category = category_name_for_volume(raw_category, volume)
        for entry in entries:
            categorized_entries.add(id(entry))
            title = normalized_text(entry.find("title")).rstrip(".")
            if not title:
                continue
            authors = [strip_dblp_suffix(normalized_text(author)) for author in entry.find_all("author")]
            paper_url, doi_url = preferred_paper_url(entry)
            tags = infer_tags(title, "")
            paper_key = entry.get("key", f"{edition_id}/{slugify(title)}")
            stable_suffix = hashlib.sha1(paper_url.encode("utf-8")).hexdigest()[:8]
            paper_id = (
                f"{edition_id}-{slugify(paper_key.rsplit('/', 1)[-1])}-{stable_suffix}"
            )
            papers.append(
                (
                    category,
                    {
                        "id": paper_id,
                        "title": title,
                        "authors": authors,
                        "affiliations": [],
                        "industryAffiliations": [],
                        "isIndustry": False,
                        "paperUrl": paper_url,
                        "doiUrl": doi_url,
                        "tags": tags,
                        "metadataSource": "DBLP proceedings TOC",
                    },
                )
            )

    for heading in soup.find_all("h2"):
        citations = heading.find_next_sibling("dblpcites")
        if citations is None:
            continue
        entries = citations.find_all("inproceedings")
        if entries:
            add_entries(normalized_text(heading), entries)

    all_entries = soup.find_all("inproceedings")
    uncategorized = [entry for entry in all_entries if id(entry) not in categorized_entries]
    if uncategorized:
        fallback = volume["booktitle"] or "Proceedings"
        add_entries(fallback, uncategorized)
    soup.decompose()
    return papers


def industry_affiliations(affiliations: list[str]) -> list[str]:
    return [affiliation for affiliation in affiliations if INDUSTRY_RE.search(affiliation)]


def convert_recsys_edition(conference: dict) -> dict:
    categories: list[dict] = []
    for track in conference.get("tracks", []):
        papers = []
        for paper in track["papers"]:
            papers.append(
                {
                    "id": paper["id"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "affiliations": paper["affiliations"],
                    "industryAffiliations": paper["industryAffiliations"],
                    "isIndustry": paper["isIndustry"],
                    "paperUrl": paper["doiUrl"],
                    "doiUrl": paper["doiUrl"],
                    "tags": paper["tags"],
                    "metadataSource": "Official RecSys accepted list",
                }
            )
        categories.append({"id": track["id"], "name": track["name"], "papers": papers})
    return {
        "schemaVersion": 1,
        "id": conference["id"],
        "venueId": "recsys",
        "conference": conference["name"],
        "year": conference["year"],
        "status": conference["status"],
        "sourceUrl": conference["sourceUrl"],
        "categories": categories,
    }


def unique_nonempty(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def make_official_paper(
    edition_id: str,
    category: str,
    title: str,
    authors: list[str],
    affiliations: list[str],
    paper_url: str,
    metadata_source: str,
    topic_text: str = "",
) -> dict:
    title = re.sub(r"\s+", " ", title).strip().rstrip(".")
    authors = unique_nonempty(authors)
    affiliations = unique_nonempty(affiliations)
    tags = infer_tags(title, topic_text)
    stable_suffix = hashlib.sha1(paper_url.encode("utf-8")).hexdigest()[:10]
    return {
        "id": f"{edition_id}-{slugify(title)}-{stable_suffix}",
        "title": title,
        "authors": authors,
        "affiliations": affiliations,
        "industryAffiliations": industry_affiliations(affiliations),
        "isIndustry": bool(industry_affiliations(affiliations)),
        "paperUrl": paper_url,
        "doiUrl": paper_url if paper_url.lower().startswith("https://doi.org/") else None,
        "tags": tags,
        "metadataSource": metadata_source,
    }


def build_icml_2026_edition(downloader: OfficialDownloader, venue_id: str, venue: dict) -> dict:
    edition_id = "icml-2026"
    source_url = "https://icml.cc/static/virtual/data/icml-2026-orals-posters.json"
    payload = json.loads(downloader.get(source_url, "icml-2026-orals-posters.json"))
    category_names = [
        "Accept (oral)",
        "Accept (spotlight poster)",
        "Accept (poster)",
        "ICML (Position Papers) — Accept (oral)",
        "ICML (Position Papers) — Accept (spotlight poster)",
        "ICML (Position Papers) — Accept (poster)",
        "ICML (Journal Track) — Poster",
    ]
    by_category: OrderedDict[str, list[dict]] = OrderedDict(
        (name, []) for name in category_names
    )
    accepted_sources = {
        "https://openreview.net/group?id=ICML.cc/2026/Conference": "",
        "https://openreview.net/group?id=ICML.cc/2026/Position_Paper_Track": (
            "ICML (Position Papers) — "
        ),
    }
    for entry in payload.get("results", []):
        if entry.get("eventtype") != "Poster":
            continue
        source = entry.get("sourceurl")
        is_journal_track = source in {"TMLR-2026", "ANN-STATS-2026"} or (
            isinstance(source, str)
            and source.startswith("https://api.github.com/repos/JmlrOrg/")
        )
        if source not in accepted_sources and not is_journal_track:
            continue
        if is_journal_track:
            category = "ICML (Journal Track) — Poster"
        else:
            if entry.get("related_events_ids"):
                decision = "Accept (oral)"
            elif "spotlight" in (entry.get("decision") or "").casefold():
                decision = "Accept (spotlight poster)"
            else:
                decision = "Accept (poster)"
            category = accepted_sources[source] + decision
        title = re.sub(r"\s+", " ", entry.get("name", "")).strip()
        paper_url = entry.get("paper_url") or (
            "https://icml.cc" + entry.get("virtualsite_url", "")
        )
        authors = [author.get("fullname", "") for author in entry.get("authors", [])]
        affiliations = [
            author.get("institution", "") for author in entry.get("authors", [])
        ]
        by_category[category].append(
            make_official_paper(
                edition_id,
                category,
                title,
                authors,
                affiliations,
                paper_url,
                "Official ICML virtual program",
                entry.get("topic") or "",
            )
        )

    categories = [
        {
            "id": f"{edition_id}-{slugify(name)}",
            "name": name,
            "papers": papers,
        }
        for name, papers in by_category.items()
        if papers
    ]
    return {
        "schemaVersion": 1,
        "id": edition_id,
        "venueId": venue_id,
        "conference": "ICML 2026",
        "year": 2026,
        "status": "published",
        "sourceUrl": "https://icml.cc/virtual/2026/papers.html",
        "volumeSources": [
            {
                "title": "ICML 2026 accepted papers and presentation decisions",
                "publisherUrls": [source_url],
            }
        ],
        "categories": categories,
    }


ICDE_2026_CATEGORIES = OrderedDict(
    [
        ("Research Papers", "accepted-papers.html"),
        ("Industry and Application Papers", "ia-papers.html"),
        ("Demo Papers", "demo-papers.html"),
        ("TKDE Posters", "tkde-papers.html"),
        ("DEFT Papers", "deft-papers.html"),
        ("Lightning Talks", "lightning-talks.html"),
        ("Ph.D. Symposium", "phd-papers.html"),
    ]
)


def build_icde_2026_edition(downloader: OfficialDownloader, venue_id: str, venue: dict) -> dict:
    edition_id = "icde-2026"
    base_url = "https://icde2026.github.io/"
    categories: list[dict] = []
    volume_sources: list[dict] = []
    for category, relative_url in ICDE_2026_CATEGORIES.items():
        page_url = base_url + relative_url
        html = downloader.get(page_url, f"icde-2026-{relative_url}")
        soup = BeautifulSoup(html, "html.parser")
        papers: list[dict] = []
        for index, item in enumerate(soup.select("li.paper-item"), start=1):
            title = normalized_text(item.select_one(".title"))
            if not title:
                continue
            authors = [
                normalized_text(author).rstrip("*").strip()
                for author in item.select(".author-name")
            ]
            affiliations = [
                normalized_text(affiliation).removeprefix("(").removesuffix(")")
                for affiliation in item.select(".affiliation")
            ]
            paper_url = (
                f"{page_url}#paper-{index:03d}-{slugify(title)[:48]}"
            )
            papers.append(
                make_official_paper(
                    edition_id,
                    category,
                    title,
                    authors,
                    affiliations,
                    paper_url,
                    "Official ICDE accepted list",
                )
            )
        soup.decompose()
        if papers:
            categories.append(
                {
                    "id": f"{edition_id}-{slugify(category)}",
                    "name": category,
                    "papers": papers,
                }
            )
        volume_sources.append(
            {"title": f"ICDE 2026 — {category}", "publisherUrls": [page_url]}
        )

    return {
        "schemaVersion": 1,
        "id": edition_id,
        "venueId": venue_id,
        "conference": "ICDE 2026",
        "year": 2026,
        "status": "published",
        "sourceUrl": base_url + "accepted-papers.html",
        "volumeSources": volume_sources,
        "categories": categories,
    }


SIGIR_2026_CATEGORIES = OrderedDict(
    [
        # The legend says 234, while the page itself contains 233 unique full
        # paper entries (and 655 rather than 656 main-conference entries).
        ("fp", ("Full Papers", 233)),
        ("pr", ("Perspective Papers", 12)),
        ("rp", ("Reproducibility Papers", 28)),
        ("rr", ("Resource Papers", 61)),
        ("sp", ("Short Papers", 151)),
        ("de", ("Demo Papers", 24)),
        ("ip", ("Industry Papers", 131)),
        ("lre", ("Low Resource Environment Papers", 15)),
        ("dc", ("Doctoral Colloquium", 12)),
    ]
)


IJCAI_2026_TRACKS = OrderedDict(
    [
        ("main-track", ("Main Track", 713)),
        ("special-track-on-ai-and-health", ("Special Track on AI and Health", 47)),
        ("special-track-on-ai-and-robotics", ("Special Track on AI and Robotics", 11)),
        ("special-track-on-ai-and-social-good", ("Special Track on AI and Social Good", 52)),
        (
            "special-track-on-ai4tech-ai-enabling-critical-technologies",
            ("Special Track on AI4Tech: AI Enabling Critical Technologies", 25),
        ),
        (
            "special-track-on-human-centred-ai",
            ("Special Track on Human-Centred AI", 10),
        ),
        ("journal-track", ("Journal Track", 6)),
        (
            "sister-conferences-best-papers-track",
            ("Sister Conferences Best Papers Track", 17),
        ),
        ("survey-track", ("Survey Track", 45)),
        ("early-career-spotlight", ("Early Career Spotlight", 14)),
        ("demonstrations-track", ("Demonstrations Track", 50)),
    ]
)


def build_ijcai_2026_edition(downloader: OfficialDownloader, venue_id: str, venue: dict) -> dict:
    edition_id = "ijcai-2026"
    page_url = "https://2026.ijcai.org/accepted-papers/"
    proxy_base = "https://2026-ijcai-org.translate.goog/accepted-papers/"
    categories: list[dict] = []
    for track_slug, (category, expected_count) in IJCAI_2026_TRACKS.items():
        # The origin currently stalls on multi-megabyte responses in some
        # networks. Google Translate is used as a transparent read-through
        # fetcher with automatic source detection and English output; all
        # user-facing and provenance links remain on the official IJCAI site.
        fetch_url = (
            f"{proxy_base}?ijtrack={track_slug}"
            "&_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en"
        )
        html = downloader.get(fetch_url, f"ijcai-2026-{track_slug}.html")
        soup = BeautifulSoup(html, "html.parser")
        papers: list[dict] = []
        for index, item in enumerate(soup.select("li.ij-paper"), start=1):
            title = normalized_text(item.select_one(".ij-ptitle"))
            if not title:
                continue
            authors = [normalized_text(author) for author in item.select(".ij-author")]
            topics = [
                keyword.get("title", normalized_text(keyword)).replace("→", " ")
                for keyword in item.select(".ij-kw")
            ]
            paper_number = normalized_text(item.select_one(".ij-pid")).lstrip("#")
            paper_url = (
                f"{page_url}?ijtrack={track_slug}"
                f"#paper-{paper_number or index:0>4}"
            )
            papers.append(
                make_official_paper(
                    edition_id,
                    category,
                    title,
                    authors,
                    [],
                    paper_url,
                    "Official IJCAI-ECAI accepted list",
                    " ".join(topics),
                )
            )
        soup.decompose()
        if len(papers) != expected_count:
            raise RuntimeError(
                f"IJCAI-ECAI 2026 {category} count differs: expected "
                f"{expected_count}, got {len(papers)}. Refusing to publish a partial page."
            )
        categories.append(
            {
                "id": f"{edition_id}-{slugify(category)}",
                "name": category,
                "papers": papers,
            }
        )

    return {
        "schemaVersion": 1,
        "id": edition_id,
        "venueId": venue_id,
        "conference": "IJCAI-ECAI 2026",
        "year": 2026,
        "status": "published",
        "sourceUrl": page_url,
        "volumeSources": [
            {
                "title": "IJCAI-ECAI 2026 accepted papers by track",
                "publisherUrls": [page_url],
            }
        ],
        "categories": categories,
    }


def build_sigir_2026_edition(downloader: OfficialDownloader, venue_id: str, venue: dict) -> dict:
    edition_id = "sigir-2026"
    page_url = "https://sigir2026.org/en-AU/pages/program/accepted-papers"
    fetch_url = (
        "https://sigir2026-org.translate.goog/en-AU/pages/program/accepted-papers"
        "?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en"
    )
    payload = downloader.get(
        fetch_url,
        "sigir-2026-accepted-papers.html",
        headers={"Accept": "text/html,*/*;q=0.8", "User-Agent": "Mozilla/5.0"},
    )
    flight_fragments: list[str] = []
    for match in re.finditer(
        r"self\.__next_f\.push\(\[1,(\"(?:\\.|[^\"\\])*\")\]\)",
        payload,
    ):
        try:
            flight_fragments.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    soup = BeautifulSoup("\n".join(flight_fragments), "html.parser")
    by_code: OrderedDict[str, list[dict]] = OrderedDict(
        (code, []) for code in SIGIR_2026_CATEGORIES
    )
    seen_papers: set[tuple[str, str, tuple[str, ...]]] = set()
    for item in soup.find_all("p"):
        title_element = item.find("i")
        if title_element is None:
            continue
        full_text = normalized_text(item)
        match = re.match(r"^\[([a-z]+)\]\s*", full_text, flags=re.I)
        if not match:
            continue
        code = match.group(1).casefold()
        if code not in by_code:
            continue
        title = normalized_text(title_element)
        remainder = full_text[match.end():]
        if remainder.startswith(title):
            remainder = remainder[len(title):].strip()
        authors = [part.strip() for part in remainder.split(",") if part.strip()]
        dedupe_key = (code, title.casefold(), tuple(authors))
        if dedupe_key in seen_papers:
            continue
        seen_papers.add(dedupe_key)
        category = SIGIR_2026_CATEGORIES[code][0]
        index = len(by_code[code]) + 1
        paper_url = f"{page_url}#paper-{code}-{index:03d}-{slugify(title)[:42]}"
        by_code[code].append(
            make_official_paper(
                edition_id,
                category,
                title,
                authors,
                [],
                paper_url,
                "Official SIGIR accepted list",
            )
        )
    soup.decompose()

    actual_counts = {code: len(papers) for code, papers in by_code.items()}
    expected_counts = {
        code: category[1] for code, category in SIGIR_2026_CATEGORIES.items()
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"SIGIR 2026 accepted-list counts differ: expected {expected_counts}, "
            f"got {actual_counts}. Refusing to publish a partial page."
        )
    categories = [
        {
            "id": f"{edition_id}-{slugify(SIGIR_2026_CATEGORIES[code][0])}",
            "name": SIGIR_2026_CATEGORIES[code][0],
            "papers": papers,
        }
        for code, papers in by_code.items()
    ]
    return {
        "schemaVersion": 1,
        "id": edition_id,
        "venueId": venue_id,
        "conference": "SIGIR 2026",
        "year": 2026,
        "status": "published",
        "statusNote": (
            "The official page legend reports 234 full papers, but its itemized "
            "list contains 233 unique full-paper entries. The archive reflects the "
            "itemized official list and records the discrepancy explicitly."
        ),
        "sourceUrl": page_url,
        "volumeSources": [
            {"title": "SIGIR 2026 accepted papers", "publisherUrls": [page_url]}
        ],
        "categories": categories,
    }


OFFICIAL_EDITION_BUILDERS = {
    ("icde", 2026): build_icde_2026_edition,
    ("icml", 2026): build_icml_2026_edition,
    ("ijcai", 2026): build_ijcai_2026_edition,
    ("sigir", 2026): build_sigir_2026_edition,
}


def edition_search_rows(edition: dict) -> list[list]:
    rows: list[list] = []
    for category_index, category in enumerate(edition["categories"]):
        for paper_index, paper in enumerate(category["papers"]):
            rows.append(
                [
                    edition["id"],
                    category_index,
                    paper_index,
                    paper["title"],
                    ", ".join(paper["authors"]),
                    category["name"],
                    paper["paperUrl"],
                    paper["isIndustry"],
                ]
            )
    return rows


def edition_manifest_entry(edition: dict, venue: dict, year_meta: dict, file_name: str | None) -> dict:
    categories = [
        {"name": category["name"], "count": len(category["papers"])}
        for category in edition["categories"]
    ]
    paper_count = sum(category["count"] for category in categories)
    result = {
        "id": edition["id"],
        "venueId": edition["venueId"],
        "conference": edition["conference"],
        "year": edition["year"],
        "status": edition["status"],
        "website": year_meta["website"],
        "dates": year_meta["dates"],
        "location": year_meta["location"],
        "paperCount": paper_count,
        "abstractCount": edition.get("abstractCount", 0),
        "missingAbstractCount": edition.get("missingAbstractCount", paper_count),
        "sourcePaperCount": edition.get("sourcePaperCount", paper_count),
        "filteredOutCount": edition.get("filteredOutCount", 0),
        "selectionPolicy": edition.get("selectionPolicy", "recsys-related-only"),
        "categories": categories,
        "file": file_name,
        "sourceUrl": edition.get("sourceUrl"),
        "volumeSources": edition.get("volumeSources", []),
    }
    note = year_meta.get("note") or edition.get("statusNote")
    if note:
        result["note"] = note
    if edition["status"] == "pending" and year_meta.get("expectedPublicationDate"):
        result["expectedPublicationDate"] = year_meta["expectedPublicationDate"]
    return result


def build_dblp_edition(
    downloader: DblpDownloader,
    venue_id: str,
    venue: dict,
    year: int,
    volumes: list[dict],
) -> dict:
    edition_id = f"{venue_id}-{year}"
    by_category: OrderedDict[str, list[dict]] = OrderedDict()
    seen: set[str] = set()
    volume_sources: list[dict] = []
    for volume in volumes:
        print(f"Reading {edition_id}: {volume['booktitle']} ({volume['xml']})", flush=True)
        payload = downloader.get(volume["xml"])
        for category, paper in parse_volume(payload, volume, edition_id):
            dedupe_key = (paper.get("doiUrl") or paper["title"]).casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            by_category.setdefault(category, []).append(paper)
        volume_sources.append(
            {
                "title": volume["title"],
                "dblpUrl": "https://dblp.org/" + volume["url"],
                "publisherUrls": volume["publisherUrls"],
            }
        )
    categories = [
        {
            "id": f"{edition_id}-{slugify(name)}",
            "name": name,
            "papers": papers,
        }
        for name, papers in by_category.items()
    ]
    status = "partial" if year in venue.get("partialYears", set()) else "published"
    return {
        "schemaVersion": 1,
        "id": edition_id,
        "venueId": venue_id,
        "conference": f"{venue['name']} {year}",
        "year": year,
        "status": status,
        "sourceUrl": f"https://dblp.org/db/conf/{venue['stream']}/",
        "volumeSources": volume_sources,
        "categories": categories,
    }


def pending_edition(venue_id: str, venue: dict, year: int) -> dict:
    note = venue["years"][year].get("note")
    if not note:
        note = (
            "A complete, category-preserving proceedings table of contents is not yet "
            "available in the structured source. No unverified paper list is shown."
        )
    return {
        "schemaVersion": 1,
        "id": f"{venue_id}-{year}",
        "venueId": venue_id,
        "conference": f"{venue['name']} {year}",
        "year": year,
        "status": "pending",
        "statusNote": note,
        "sourceUrl": venue["years"][year]["website"],
        "categories": [],
    }


def load_recsys_editions() -> dict[int, dict]:
    source = json.loads(RECSYS_PATH.read_text(encoding="utf-8"))
    return {
        conference["year"]: convert_recsys_edition(conference)
        for conference in source["conferences"]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Redownload cached DBLP pages.")
    parser.add_argument("--verified", default=date.today().isoformat())
    parser.add_argument(
        "--venues",
        nargs="*",
        choices=[*VENUES, "recsys"],
        help="Build only selected venues (useful while developing).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = set(args.venues or [*VENUES, "recsys"])
    downloader = DblpDownloader(DBLP_CACHE_DIR, refresh=args.refresh)
    official_downloader = OfficialDownloader(OFFICIAL_CACHE_DIR, refresh=args.refresh)
    manifest_editions: list[dict] = []
    search_rows: list[list] = []
    paper_total = 0
    source_paper_total = 0
    abstract_total = 0
    abstract_cache = load_abstract_cache()

    recsys_editions = load_recsys_editions() if "recsys" in selected else {}
    if "recsys" in selected:
        recsys_venue = {
            "name": "RecSys",
            "fullName": "ACM Conference on Recommender Systems",
            "rank": "A",
            "years": {
                2024: edition_meta("https://recsys.acm.org/recsys24/", "14–18 Oct 2024", "Bari, Italy"),
                2025: edition_meta("https://recsys.acm.org/recsys25/", "22–26 Sep 2025", "Prague, Czech Republic"),
                2026: edition_meta(
                    "https://recsys.acm.org/recsys26/",
                    "28 Sep–2 Oct 2026",
                    "Minneapolis, Minnesota, USA",
                    expected_publication_date="2026-09-28",
                ),
            },
        }
        for year in YEARS:
            edition = apply_cached_abstracts(
                apply_topic_selection(recsys_editions[year]), abstract_cache
            )
            file_name = None
            if edition["status"] != "pending":
                file_name = f"{edition['id']}.json"
                json_dump(OUTPUT_DIR / file_name, edition)
                search_rows.extend(edition_search_rows(edition))
            paper_total += sum(len(category["papers"]) for category in edition["categories"])
            source_paper_total += edition["sourcePaperCount"]
            abstract_total += edition["abstractCount"]
            manifest_editions.append(
                edition_manifest_entry(edition, recsys_venue, recsys_venue["years"][year], file_name)
            )
            del edition

    for venue_id, venue in VENUES.items():
        if venue_id not in selected:
            continue
        index_relative = f"db/conf/{venue['stream']}/index.xml"
        print(f"Reading venue index: {venue['name']}", flush=True)
        index_html = downloader.get(index_relative)
        for year in YEARS:
            volumes = proceedings_from_index(index_html, venue, year)
            official_builder = OFFICIAL_EDITION_BUILDERS.get((venue_id, year))
            if volumes:
                edition = build_dblp_edition(downloader, venue_id, venue, year, volumes)
            elif official_builder:
                print(f"Reading official accepted list: {venue['name']} {year}", flush=True)
                edition = official_builder(official_downloader, venue_id, venue)
            else:
                edition = pending_edition(venue_id, venue, year)
            edition = apply_cached_abstracts(apply_topic_selection(edition), abstract_cache)
            file_name = None
            if edition["status"] != "pending":
                file_name = f"{edition['id']}.json"
                json_dump(OUTPUT_DIR / file_name, edition)
                search_rows.extend(edition_search_rows(edition))
            paper_total += sum(len(category["papers"]) for category in edition["categories"])
            source_paper_total += edition["sourcePaperCount"]
            abstract_total += edition["abstractCount"]
            manifest_editions.append(
                edition_manifest_entry(edition, venue, venue["years"][year], file_name)
            )
            del edition
            gc.collect()

    venue_manifest = []
    if "recsys" in selected:
        venue_manifest.append(
            {
                "id": "recsys",
                "name": "RecSys",
                "fullName": "ACM Conference on Recommender Systems",
                "rank": "A",
                "selectionPolicy": "complete-profile-venue",
            }
        )
    venue_manifest.extend(
        {
            "id": venue_id,
            "name": venue["name"],
            "fullName": venue["fullName"],
            "rank": venue["rank"],
            "selectionPolicy": (
                "complete-profile-venue"
                if venue_id in PROFILE_VENUE_IDS
                else "recsys-related-only"
            ),
        }
        for venue_id, venue in VENUES.items()
        if venue_id in selected
    )

    manifest = {
        "schemaVersion": 1,
        "lastVerified": args.verified,
        "trackerUrl": "https://recsys.info/",
        "metadataSourceUrl": "https://dblp.org/",
        "scope": (
            "Complete accepted lists are retained for RecSys, CIKM, KDD, SIGIR, ECIR, "
            "and UMAP. Other venues include only papers with explicit recommender-systems "
            "evidence in the title or track name. Standalone workshop proceedings are excluded."
        ),
        "selectionPolicy": {
            "profileVenueIds": sorted(selected & PROFILE_VENUE_IDS),
            "broadVenueIds": sorted(selected - PROFILE_VENUE_IDS),
            "broadVenueRule": (
                "A paper is selected when its track explicitly names recommender systems or "
                "its title contains a high-precision recommender-systems term. Ambiguous "
                "non-system uses of recommendation are excluded."
            ),
        },
        "venues": venue_manifest,
        "editions": manifest_editions,
        "paperCount": paper_total,
        "sourcePaperCount": source_paper_total,
        "abstractCount": abstract_total,
        "missingAbstractCount": paper_total - abstract_total,
    }
    json_dump(MANIFEST_PATH, manifest, pretty=True)
    json_dump(
        OUTPUT_DIR / "search-index.json",
        {"schemaVersion": 1, "papers": search_rows},
    )
    print(
        f"Wrote {manifest['paperCount']} papers across {len(manifest_editions)} editions; "
        f"global search index has {len(search_rows)} rows.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
