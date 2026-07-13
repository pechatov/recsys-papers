#!/usr/bin/env python3
"""Fetch and attach verified abstracts for the conference archive.

The collector accepts an abstract only after an exact DOI match or a strong
normalized-title match on an official proceedings page. Results are cached
under ``.cache`` so routine archive rebuilds do not repeat network requests.
OpenAlex is used as a DOI-indexed metadata source; official proceedings pages
and verified open full-text copies fill the remaining gaps. PDF extraction is
limited to the explicit Abstract section of a title-matched paper.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from difflib import SequenceMatcher
import html as html_module
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover - maintainer-facing failure path
    raise SystemExit(
        "Beautiful Soup is required: python3 -m pip install beautifulsoup4"
    ) from exc

try:
    import requests
except ImportError:  # pragma: no cover - curl/urllib remain available fallbacks
    requests = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "public" / "data" / "conference-papers"
MANIFEST_PATH = ROOT / "src" / "data" / "conference-archive.json"
CACHE_PATH = ROOT / ".cache" / "conference-archive" / "abstracts.json"
ICML_CACHE_PATH = (
    ROOT
    / ".cache"
    / "conference-archive"
    / "official"
    / "icml-2026-orals-posters.json"
)
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "recsys-papers-abstracts/1.0"
)
OPENALEX_URL = "https://api.openalex.org/works"
CROSSREF_URL = "https://api.crossref.org/works"
ARXIV_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
SIGIR_2026_CONTAINER = (
    "Proceedings of the 49th International ACM SIGIR Conference on Research "
    "and Development in Information Retrieval"
)
RECSYS_ACCEPTED_URLS = {
    2024: "https://recsys.acm.org/recsys24/accepted-contributions/",
    2025: "https://recsys.acm.org/recsys25/accepted-contributions/",
}
OFFICIAL_TOC_URLS = {
    ("cikm", 2024): ["https://www.sigweb.hosting.acm.org/toc/cikm24.html"],
    ("cikm", 2025): ["https://www.sigweb.hosting.acm.org/toc/cikm25.html"],
    ("sigir", 2024): ["https://sigir-2024.github.io/proceedings.html"],
    ("sigir", 2025): ["https://sigir2025.dei.unipd.it/proceedings.html"],
    ("umap", 2024): ["https://www.sigweb.hosting.acm.org/toc/umap24.html"],
    ("umap", 2025): [
        "https://www.sigweb.hosting.acm.org/toc/umap25a.html",
        "https://www.sigweb.hosting.acm.org/toc/umap25b.html",
    ],
    ("wsdm", 2024): ["https://www.sigweb.hosting.acm.org/toc/wsdm24.html"],
    ("wsdm", 2025): ["https://www.sigweb.hosting.acm.org/toc/wsdm25.html"],
    ("wsdm", 2026): [
        "https://www.sigweb.hosting.acm.org/toc/wsdm26a.html",
        "https://www.sigweb.hosting.acm.org/toc/wsdm26b.html",
    ],
    ("www", 2024): [
        "https://www.sigweb.hosting.acm.org/toc/www24.html",
        "https://www.sigweb.hosting.acm.org/toc/www24a.html",
    ],
    ("www", 2025): [
        "https://www.sigweb.hosting.acm.org/toc/www25a.html",
        "https://www.sigweb.hosting.acm.org/toc/www25b.html",
    ],
    ("www", 2026): [
        "https://www.sigweb.hosting.acm.org/toc/www26a.html",
        "https://www.sigweb.hosting.acm.org/toc/www26b.html",
    ],
}
PDF_MAX_BYTES = 30 * 1024 * 1024


def json_dump(path: Path, payload: object, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    else:
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    path.write_text(content, encoding="utf-8")


def normalize_title(value: str) -> str:
    value = html_module.unescape(value or "")
    value = unicodedata.normalize("NFKD", value)
    value = value.translate(
        str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})
    )
    value = value.encode("ascii", "ignore").decode("ascii").casefold()
    value = value.replace("&", " and ")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def titles_match(expected: str, actual: str, threshold: float = 0.88) -> bool:
    left = normalize_title(expected)
    right = normalize_title(actual)
    if not left or not right:
        return False
    if left == right:
        return True
    return SequenceMatcher(None, left, right).ratio() >= threshold


def clean_abstract(value: str | None) -> str | None:
    if not value:
        return None
    value = BeautifulSoup(html_module.unescape(value), "html.parser").get_text(" ", strip=True)
    value = re.sub(r"^\s*abstract\s*[:.—-]?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) < 40 or value.casefold() in {"abstract", "no abstract available"}:
        return None
    return value


def author_keys(authors: list[str]) -> set[str]:
    keys: set[str] = set()
    for author in authors:
        normalized = normalize_title(author)
        if not normalized:
            continue
        keys.add(normalized)
        keys.add(normalized.split()[-1])
    return keys


def authors_match(expected: list[str], actual: list[str]) -> bool:
    if not expected or not actual:
        return True
    return bool(author_keys(expected) & author_keys(actual))


def inverted_abstract(index: dict | None) -> str | None:
    if not index:
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in index.items():
        positioned.extend((int(position), word) for position in positions)
    positioned.sort()
    return clean_abstract(" ".join(word for _, word in positioned))


def doi_from_url(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"10\.\d{4,9}/[^\s?#]+", value, flags=re.IGNORECASE)
    return match.group(0).rstrip(".,;)").casefold() if match else None


def request_text(url: str, *, attempts: int = 5, timeout: int = 90) -> str:
    failures: list[str] = []
    for attempt in range(attempts):
        request = Request(
            url,
            headers={
                "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
                "Accept-Encoding": "identity",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            failures.append(str(exc))
            if exc.code in {400, 404}:
                break
            if attempt + 1 < attempts:
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else 5.0 * (attempt + 1)
                    time.sleep(wait)
                else:
                    time.sleep(1.5 * (attempt + 1))
        except (URLError, TimeoutError) as exc:
            failures.append(str(exc))
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {'; '.join(failures)}")


def load_documents() -> tuple[dict, dict[str, dict], list[dict]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    documents: dict[str, dict] = {}
    papers: list[dict] = []
    for edition in manifest["editions"]:
        file_name = edition.get("file")
        if not file_name:
            continue
        document = json.loads((DATA_DIR / file_name).read_text(encoding="utf-8"))
        documents[edition["id"]] = document
        for category in document["categories"]:
            for paper in category["papers"]:
                papers.append(
                    {
                        "id": paper["id"],
                        "title": paper["title"],
                        "authors": paper.get("authors", []),
                        "venueId": document["venueId"],
                        "year": document["year"],
                        "paperUrl": paper["paperUrl"],
                        "doi": doi_from_url(paper.get("doiUrl") or paper.get("paperUrl")),
                    }
                )
    return manifest, documents, papers


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {
            "schemaVersion": 2,
            "papers": {},
            "resolvedDois": {},
            "fullTextCandidates": {},
            "arxivChecks": {},
        }
    payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    payload["schemaVersion"] = 2
    payload.setdefault("papers", {})
    payload.setdefault("resolvedDois", {})
    payload.setdefault("fullTextCandidates", {})
    payload.setdefault("arxivChecks", {})
    return payload


def cache_record(
    cache: dict,
    paper: dict,
    abstract: str | None,
    source: str,
    source_url: str,
    *,
    matched_doi: str | None = None,
) -> bool:
    abstract = clean_abstract(abstract)
    if not abstract:
        return False
    cache["papers"][paper["id"]] = {
        "title": paper["title"],
        "abstract": abstract,
        "source": source,
        "sourceUrl": source_url,
        "matchedDoi": matched_doi,
    }
    return True


def fetch_json(url: str) -> dict:
    return json.loads(request_text(url))


def request_html(url: str, session=None) -> str:
    """Fetch publisher HTML through curl when available.

    The collaborative development environment proxies long-lived Python TLS
    sockets. Curl handles those proxy redirects reliably, while the urllib
    implementation remains a portable fallback for other environments.
    """

    if session is not None:
        response = session.get(url, timeout=(15, 30))
        response.raise_for_status()
        return response.text

    curl = shutil.which("curl")
    if not curl:
        return request_text(url)
    try:
        result = subprocess.run(
            [
                curl,
                "--location",
                "--silent",
                "--show-error",
                "--fail",
                "--max-time",
                "90",
                "--retry",
                "3",
                "--retry-all-errors",
                "--user-agent",
                USER_AGENT,
                url,
            ],
            check=False,
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return request_text(url)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.decode("utf-8", errors="replace")
    return request_text(url)


def toc_abstract_after(title_node) -> str | None:
    """Return the abstract associated with one ACM-style TOC heading."""

    heading = title_node.find_parent("h3")
    if heading is None:
        return None
    sibling = heading.find_next_sibling()
    while sibling is not None and sibling.name != "h3":
        classes = sibling.get("class", []) if hasattr(sibling, "get") else []
        if "DLabstract" in classes:
            return clean_abstract(sibling.get_text(" ", strip=True))
        sibling = sibling.find_next_sibling()
    return None


def fetch_official_toc_abstracts(papers: list[dict], cache: dict) -> int:
    """Read abstracts from official ACM-style proceedings mirrors.

    SIGIR conference sites and ACM SIGWEB publish static, accessible tables of
    contents with the same title/author/abstract blocks as the ACM Digital
    Library. Matching uses the DOI when present and always verifies the title.
    """

    pending: dict[tuple[str, int], list[dict]] = {}
    for paper in papers:
        key = (paper["venueId"], paper["year"])
        if paper["id"] not in cache["papers"] and key in OFFICIAL_TOC_URLS:
            pending.setdefault(key, []).append(paper)

    added = 0
    pages_checked = 0
    for key, candidates in pending.items():
        by_doi = {paper["doi"]: paper for paper in candidates if paper["doi"]}
        by_title: dict[str, list[dict]] = {}
        for paper in candidates:
            by_title.setdefault(normalize_title(paper["title"]), []).append(paper)
        for url in OFFICIAL_TOC_URLS[key]:
            try:
                soup = BeautifulSoup(request_html(url), "html.parser")
            except RuntimeError as exc:
                print(f"Official TOC unavailable: {url}: {exc}", flush=True)
                continue
            pages_checked += 1
            for title_node in soup.select("a.DLtitleLink"):
                title = title_node.get_text(" ", strip=True)
                doi = doi_from_url(title_node.get("href"))
                matches = []
                if doi and doi in by_doi:
                    matches = [by_doi[doi]]
                if not matches:
                    matches = by_title.get(normalize_title(title), [])
                abstract = toc_abstract_after(title_node)
                if not matches or not abstract:
                    continue
                heading = title_node.find_parent("h3")
                authors_node = heading.find_next_sibling("ul", class_="DLauthors")
                authors = (
                    [node.get_text(" ", strip=True) for node in authors_node.select("li")]
                    if authors_node
                    else []
                )
                for paper in matches:
                    if paper["id"] in cache["papers"]:
                        continue
                    if not titles_match(paper["title"], title, threshold=0.97):
                        continue
                    if not authors_match(paper.get("authors", []), authors):
                        continue
                    if cache_record(
                        cache,
                        paper,
                        abstract,
                        "Official proceedings table of contents",
                        url,
                        matched_doi=doi or paper["doi"],
                    ):
                        added += 1
            soup.decompose()
            json_dump(CACHE_PATH, cache, pretty=True)
    if pages_checked:
        print(
            f"Official proceedings TOCs: {pages_checked} pages; abstracts added {added}",
            flush=True,
        )
    return added


def discover_sigir_2026_dois(papers: list[dict]) -> dict[str, str]:
    candidates = {
        normalize_title(paper["title"]): paper
        for paper in papers
        if paper["venueId"] == "sigir" and paper["year"] == 2026 and not paper["doi"]
    }
    if not candidates:
        return {}
    params = urlencode(
        {
            "query.container-title": SIGIR_2026_CONTAINER,
            "filter": (
                "from-pub-date:2026-01-01,until-pub-date:2026-12-31,"
                "prefix:10.1145,type:proceedings-article"
            ),
            "rows": 1000,
            "select": "DOI,title,container-title",
        }
    )
    payload = fetch_json(f"{CROSSREF_URL}?{params}")["message"]
    resolved: dict[str, str] = {}
    for item in payload.get("items", []):
        if (item.get("container-title") or [""])[0] != SIGIR_2026_CONTAINER:
            continue
        title = (item.get("title") or [""])[0]
        paper = candidates.get(normalize_title(title))
        doi = (item.get("DOI") or "").casefold()
        if paper and doi and titles_match(paper["title"], title, threshold=0.97):
            resolved[paper["id"]] = doi
    return resolved


def crossref_title_lookup(paper: dict) -> tuple[str, str] | None:
    if paper["year"] != 2026 or paper["venueId"] not in {"icde", "ijcai"}:
        return None
    params = urlencode(
        {
            "query.title": paper["title"],
            "filter": "from-pub-date:2026-01-01,until-pub-date:2026-12-31",
            "rows": 5,
            "select": "DOI,title",
        }
    )
    try:
        items = fetch_json(f"{CROSSREF_URL}?{params}")["message"].get("items", [])
    except RuntimeError:
        return None
    expected_prefix = "10.1109/" if paper["venueId"] == "icde" else "10.24963/"
    for item in items:
        title = (item.get("title") or [""])[0]
        doi = (item.get("DOI") or "").casefold()
        if doi.startswith(expected_prefix) and titles_match(paper["title"], title, threshold=0.97):
            return paper["id"], doi
    return None


def discover_other_2026_dois(papers: list[dict], workers: int) -> dict[str, str]:
    candidates = [
        paper
        for paper in papers
        if not paper["doi"] and paper["venueId"] in {"icde", "ijcai"} and paper["year"] == 2026
    ]
    resolved: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(workers, 4)) as executor:
        futures = [executor.submit(crossref_title_lookup, paper) for paper in candidates]
        for future in as_completed(futures):
            result = future.result()
            if result:
                resolved[result[0]] = result[1]
    return resolved


def openalex_pdf_url(work: dict) -> str | None:
    locations = [work.get("best_oa_location"), work.get("primary_location")]
    locations.extend(work.get("locations") or [])
    for location in locations:
        if not location:
            continue
        pdf_url = location.get("pdf_url")
        if pdf_url and (location.get("is_oa") or location is work.get("best_oa_location")):
            return pdf_url.replace("http://", "https://", 1)
        landing_url = location.get("landing_page_url") or ""
        if location.get("is_oa") and re.search(r"arxiv\.org/abs/", landing_url):
            return re.sub(r"arxiv\.org/abs/", "arxiv.org/pdf/", landing_url).replace(
                "http://", "https://", 1
            )
    return None


def cache_full_text_candidate(
    cache: dict,
    paper: dict,
    *,
    pdf_url: str,
    record_url: str,
    source: str,
    metadata_abstract: str | None = None,
    matched_doi: str | None = None,
) -> bool:
    if not pdf_url:
        return False
    cache["fullTextCandidates"][paper["id"]] = {
        "title": paper["title"],
        "pdfUrl": pdf_url,
        "recordUrl": record_url,
        "source": source,
        "metadataAbstract": clean_abstract(metadata_abstract),
        "matchedDoi": matched_doi,
    }
    return True


def fetch_openalex_abstracts(
    papers: list[dict],
    cache: dict,
    resolved_dois: dict[str, str],
    batch_size: int,
    workers: int,
) -> int:
    by_doi: dict[str, list[dict]] = {}
    for paper in papers:
        if paper["id"] in cache["papers"]:
            continue
        doi = paper["doi"] or resolved_dois.get(paper["id"])
        # Springer is queried directly below: OpenAlex does not currently carry
        # abstracts for these proceedings chapters, and skipping them saves API
        # budget as well as a large number of empty DOI lookups.
        if doi and not doi.startswith("10.1007/"):
            by_doi.setdefault(doi, []).append(paper)

    api_key = os.environ.get("OPENALEX_API_KEY")
    if not api_key:
        return fetch_openalex_singletons(by_doi, cache, workers)

    dois = sorted(by_doi)
    added = 0
    for offset in range(0, len(dois), batch_size):
        batch = dois[offset : offset + batch_size]
        values = "|".join(f"https://doi.org/{doi}" for doi in batch)
        params: dict[str, str | int] = {
            "filter": f"doi:{values}",
            "per-page": 100,
            "select": (
                "id,doi,title,abstract_inverted_index,best_oa_location,"
                "primary_location,locations"
            ),
        }
        if api_key:
            params["api_key"] = api_key
        payload = fetch_json(f"{OPENALEX_URL}?{urlencode(params)}")
        for work in payload.get("results", []):
            doi = doi_from_url(work.get("doi"))
            abstract = inverted_abstract(work.get("abstract_inverted_index"))
            if not doi:
                continue
            for paper in by_doi.get(doi, []):
                if not titles_match(paper["title"], work.get("title", "")):
                    continue
                if abstract:
                    if cache_record(
                        cache,
                        paper,
                        abstract,
                        "OpenAlex exact DOI match",
                        work.get("id") or f"https://doi.org/{doi}",
                        matched_doi=doi,
                    ):
                        added += 1
                else:
                    pdf_url = openalex_pdf_url(work)
                    if pdf_url:
                        cache_full_text_candidate(
                            cache,
                            paper,
                            pdf_url=pdf_url,
                            record_url=work.get("id") or f"https://doi.org/{doi}",
                            source="OpenAlex exact DOI match",
                            matched_doi=doi,
                        )
        print(
            f"OpenAlex DOI batches: {min(offset + batch_size, len(dois))}/{len(dois)}; "
            f"abstracts added {added}",
            flush=True,
        )
        json_dump(CACHE_PATH, cache, pretty=True)
        time.sleep(0.35)
    return added


def openalex_singleton(doi: str) -> tuple[str, dict | None]:
    try:
        return doi, fetch_json(f"{OPENALEX_URL}/doi:{quote(doi, safe='')}")
    except RuntimeError:
        return doi, None


def fetch_openalex_singletons(
    by_doi: dict[str, list[dict]], cache: dict, workers: int
) -> int:
    """Use free DOI singleton lookups when no OpenAlex API key is configured."""

    added = 0
    completed = 0
    failures = 0
    dois = sorted(by_doi)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(openalex_singleton, doi) for doi in dois]
        for future in as_completed(futures):
            completed += 1
            doi, work = future.result()
            if not work:
                failures += 1
            else:
                abstract = inverted_abstract(work.get("abstract_inverted_index"))
                for paper in by_doi.get(doi, []):
                    if not titles_match(paper["title"], work.get("title", "")):
                        continue
                    if abstract:
                        if cache_record(
                            cache,
                            paper,
                            abstract,
                            "OpenAlex exact DOI match",
                            work.get("id") or f"https://doi.org/{doi}",
                            matched_doi=doi,
                        ):
                            added += 1
                    else:
                        pdf_url = openalex_pdf_url(work)
                        if pdf_url:
                            cache_full_text_candidate(
                                cache,
                                paper,
                                pdf_url=pdf_url,
                                record_url=work.get("id") or f"https://doi.org/{doi}",
                                source="OpenAlex exact DOI match",
                                matched_doi=doi,
                            )
            if completed % 100 == 0 or completed == len(dois):
                json_dump(CACHE_PATH, cache, pretty=True)
                print(
                    f"OpenAlex DOI singletons: {completed}/{len(dois)}; "
                    f"abstracts added {added}; lookup failures {failures}",
                    flush=True,
                )
    return added


def atom_text(node: ET.Element, name: str) -> str:
    found = node.find(f"{ATOM}{name}")
    if found is None or not found.text:
        return ""
    return " ".join(found.text.split())


def arxiv_entries_for_batch(batch: list[dict]) -> list[dict]:
    clauses = [f'ti:"{normalize_title(paper["title"])}"' for paper in batch]
    params = urlencode(
        {
            "search_query": " OR ".join(clauses),
            "start": 0,
            "max_results": min(100, max(10, len(batch) * 3)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    root = ET.fromstring(request_text(f"{ARXIV_URL}?{params}", timeout=120))
    entries: list[dict] = []
    for entry in root.findall(f"{ATOM}entry"):
        pdf_url = ""
        for link in entry.findall(f"{ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
                break
        record_url = atom_text(entry, "id")
        if not pdf_url and "/abs/" in record_url:
            pdf_url = record_url.replace("/abs/", "/pdf/")
        entries.append(
            {
                "title": atom_text(entry, "title"),
                "authors": [
                    atom_text(author, "name") for author in entry.findall(f"{ATOM}author")
                ],
                "abstract": atom_text(entry, "summary"),
                "published": atom_text(entry, "published"),
                "recordUrl": record_url.replace("http://", "https://", 1),
                "pdfUrl": pdf_url.replace("http://", "https://", 1),
            }
        )
    return entries


def fetch_arxiv_full_text_candidates(
    papers: list[dict],
    cache: dict,
    *,
    batch_size: int,
    delay: float,
    checked_at: str,
    refresh: bool = False,
) -> int:
    """Discover open preprints by exact title and overlapping authors."""

    pending = [
        paper
        for paper in papers
        if paper["id"] not in cache["papers"]
        and paper["id"] not in cache["fullTextCandidates"]
        and (refresh or cache["arxivChecks"].get(paper["id"]) != checked_at)
    ]
    found = 0
    completed = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        try:
            entries = arxiv_entries_for_batch(batch)
        except (RuntimeError, ET.ParseError) as exc:
            print(f"arXiv batch {offset + 1} failed: {exc}", flush=True)
            continue
        by_title: dict[str, list[dict]] = {}
        for paper in batch:
            by_title.setdefault(normalize_title(paper["title"]), []).append(paper)
            cache["arxivChecks"][paper["id"]] = checked_at
        for entry in entries:
            entry_title = entry["title"]
            matches = by_title.get(normalize_title(entry_title), [])
            if not matches:
                matches = [
                    paper
                    for paper in batch
                    if titles_match(paper["title"], entry_title, threshold=0.97)
                ]
            published_year = int(entry["published"][:4]) if entry["published"][:4].isdigit() else 0
            for paper in matches:
                if paper["id"] in cache["fullTextCandidates"]:
                    continue
                if not titles_match(paper["title"], entry_title, threshold=0.97):
                    continue
                if published_year and abs(published_year - paper["year"]) > 3:
                    continue
                if not authors_match(paper.get("authors", []), entry["authors"]):
                    continue
                if cache_full_text_candidate(
                    cache,
                    paper,
                    pdf_url=entry["pdfUrl"],
                    record_url=entry["recordUrl"],
                    source="arXiv exact title and author match",
                    metadata_abstract=entry["abstract"],
                ):
                    found += 1
        completed += len(batch)
        json_dump(CACHE_PATH, cache, pretty=True)
        print(
            f"arXiv exact-title search: {completed}/{len(pending)}; "
            f"open full texts found {found}",
            flush=True,
        )
        if offset + batch_size < len(pending) and delay:
            time.sleep(delay)
    return found


def element_text(element: ET.Element) -> str:
    return " ".join(
        word.text.strip()
        for word in element.iter()
        if word.tag.endswith("word") and word.text and word.text.strip()
    )


def bbox_abstract(payload: str) -> tuple[str | None, str]:
    """Extract an Abstract block from pdftotext's coordinate-aware XHTML."""

    root = ET.fromstring(payload)
    title_node = next((node for node in root.iter() if node.tag.endswith("title")), None)
    document_title = title_node.text.strip() if title_node is not None and title_node.text else ""
    stop_heading = re.compile(
        r"^(?:\d+(?:\.\d+)*\s+)?(?:introduction|ccs concepts?|keywords?|"
        r"index terms?|acm reference format|categories and subject descriptors)\b",
        flags=re.IGNORECASE,
    )
    for page in (node for node in root.iter() if node.tag.endswith("page")):
        page_width = float(page.get("width", "0") or 0)
        blocks = [node for node in page.iter() if node.tag.endswith("block")]
        markers = [
            block
            for block in blocks
            if re.match(r"^abstract(?:\s|[:.—-]|$)", element_text(block), flags=re.IGNORECASE)
        ]
        for marker in markers:
            marker_text = element_text(marker)
            marker_x = float(marker.get("xMin", "0") or 0)
            marker_y = float(marker.get("yMin", "0") or 0)
            midpoint = page_width / 2
            narrow_left = sum(
                1
                for block in blocks
                if float(block.get("yMin", "0") or 0) >= marker_y
                and float(block.get("xMin", "0") or 0) < midpoint
                and float(block.get("xMax", "0") or 0) <= midpoint + 12
            )
            narrow_right = sum(
                1
                for block in blocks
                if float(block.get("yMin", "0") or 0) >= marker_y
                and float(block.get("xMin", "0") or 0) >= midpoint - 12
            )
            if marker_x < midpoint and narrow_left >= 2:
                left, right = max(0.0, marker_x - 20), midpoint - 2
            elif marker_x >= midpoint and narrow_right >= 2:
                left, right = midpoint + 2, page_width
            else:
                left, right = max(0.0, marker_x - 20), page_width

            collected: list[str] = []
            inline = re.sub(
                r"^abstract\s*[:.—-]?\s*", "", marker_text, flags=re.IGNORECASE
            ).strip()
            if inline:
                collected.append(inline)
            for block in sorted(blocks, key=lambda node: float(node.get("yMin", "0") or 0)):
                if block is marker:
                    continue
                y_min = float(block.get("yMin", "0") or 0)
                x_min = float(block.get("xMin", "0") or 0)
                x_max = float(block.get("xMax", "0") or 0)
                y_max = float(block.get("yMax", "0") or 0)
                if y_min < marker_y - 0.5 or x_min < left or x_max > right + 12:
                    continue
                if x_max - x_min < 20 and y_max - y_min > 40:
                    continue
                text_value = element_text(block)
                if not text_value:
                    continue
                if stop_heading.match(text_value):
                    break
                lines = []
                for line in (node for node in block.iter() if node.tag.endswith("line")):
                    line_text = element_text(line)
                    if line_text:
                        lines.append(line_text)
                collected.append("\n".join(lines) if lines else text_value)
            raw = "\n".join(collected)
            raw = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", raw)
            abstract = clean_abstract(raw)
            if abstract and 80 <= len(abstract) <= 6000:
                return abstract, document_title
    return None, document_title


def abstracts_match(left: str, right: str) -> bool:
    normalized_left = normalize_title(left)
    normalized_right = normalize_title(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return True
    return SequenceMatcher(None, normalized_left, normalized_right).ratio() >= 0.72


def extract_open_pdf(candidate: dict, expected_title: str) -> str | None:
    curl = shutil.which("curl")
    pdftotext = shutil.which("pdftotext")
    if not curl or not pdftotext:
        return None
    with tempfile.TemporaryDirectory(prefix="conference-abstract-") as directory:
        pdf_path = Path(directory) / "paper.pdf"
        bbox_path = Path(directory) / "paper.html"
        try:
            download = subprocess.run(
                [
                    curl,
                    "--location",
                    "--silent",
                    "--show-error",
                    "--fail",
                    "--connect-timeout",
                    "8",
                    "--max-time",
                    "25",
                    "--max-filesize",
                    str(PDF_MAX_BYTES),
                    "--user-agent",
                    USER_AGENT,
                    "--header",
                    "Accept: application/pdf",
                    "--output",
                    str(pdf_path),
                    candidate["pdfUrl"],
                ],
                check=False,
                capture_output=True,
                timeout=35,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if download.returncode != 0 or not pdf_path.exists():
            return None
        with pdf_path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                return None
        try:
            converted = subprocess.run(
                [
                    pdftotext,
                    "-f",
                    "1",
                    "-l",
                    "2",
                    "-bbox-layout",
                    "-enc",
                    "UTF-8",
                    str(pdf_path),
                    str(bbox_path),
                ],
                check=False,
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if converted.returncode != 0 or not bbox_path.exists():
            return None
        try:
            abstract, pdf_title = bbox_abstract(bbox_path.read_text(encoding="utf-8"))
        except (ET.ParseError, OSError):
            return None
        if pdf_title and titles_match(expected_title, pdf_title, threshold=0.93):
            return abstract
        page_words = normalize_title(bbox_path.read_text(encoding="utf-8"))
        if normalize_title(expected_title) not in page_words:
            return None
        return abstract


def extract_ar5iv_abstract(
    candidate: dict, expected_title: str
) -> tuple[str, str] | None:
    match = re.search(r"arxiv\.org/pdf/([^?#]+)", candidate.get("pdfUrl", ""))
    if not match:
        return None
    arxiv_id = re.sub(r"\.pdf$", "", match.group(1), flags=re.IGNORECASE)
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id, flags=re.IGNORECASE)
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    curl = shutil.which("curl")
    if not curl:
        return None
    try:
        response = subprocess.run(
            [
                curl,
                "--location",
                "--silent",
                "--show-error",
                "--fail",
                "--connect-timeout",
                "8",
                "--max-time",
                "25",
                "--user-agent",
                USER_AGENT,
                url,
            ],
            check=False,
            capture_output=True,
            timeout=35,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if response.returncode != 0 or not response.stdout.strip():
        return None
    soup = BeautifulSoup(response.stdout.decode("utf-8", errors="replace"), "html.parser")
    title_node = soup.select_one("h1.ltx_title_document")
    subtitle_node = soup.select_one(".ltx_subtitle")
    title = title_node.get_text(" ", strip=True) if title_node else ""
    subtitle = ""
    if subtitle_node:
        subtitle_fragment = BeautifulSoup(str(subtitle_node), "html.parser")
        for note in subtitle_fragment.select(".ltx_note"):
            note.decompose()
        subtitle = subtitle_fragment.get_text(" ", strip=True)
        subtitle_fragment.decompose()
    title_with_subtitle = " ".join(
        value
        for value in [title, subtitle]
        if value
    )
    if title and not (
        titles_match(expected_title, title, threshold=0.93)
        or titles_match(expected_title, title_with_subtitle, threshold=0.93)
    ):
        soup.decompose()
        return None
    abstract_node = soup.select_one("div.ltx_abstract")
    if abstract_node is None:
        soup.decompose()
        return None
    heading = abstract_node.select_one(".ltx_title_abstract")
    if heading:
        heading.decompose()
    abstract = clean_abstract(abstract_node.get_text(" ", strip=True))
    soup.decompose()
    if not abstract or not 80 <= len(abstract) <= 6000:
        return None
    return abstract, url


def full_text_job(
    paper: dict, candidate: dict
) -> tuple[dict, dict, str | None, str | None, str]:
    metadata_abstract = clean_abstract(candidate.get("metadataAbstract"))
    if candidate["source"].startswith("arXiv"):
        ar5iv_result = extract_ar5iv_abstract(candidate, paper["title"])
        if ar5iv_result:
            extracted, source_url = ar5iv_result
            if not metadata_abstract or abstracts_match(extracted, metadata_abstract):
                return (
                    paper,
                    candidate,
                    metadata_abstract or extracted,
                    "ar5iv",
                    source_url,
                )
    extracted = extract_open_pdf(candidate, paper["title"])
    if extracted and (not metadata_abstract or abstracts_match(extracted, metadata_abstract)):
        # The explicit PDF section verifies the match; the structured arXiv
        # version is preferred when present because it is free of column-order,
        # ligature, and line-break artifacts introduced by PDF text extraction.
        return paper, candidate, metadata_abstract or extracted, "pdf", candidate["pdfUrl"]
    return paper, candidate, metadata_abstract, None, candidate["recordUrl"]


def fetch_open_full_text_abstracts(papers: list[dict], cache: dict, workers: int) -> int:
    by_id = {paper["id"]: paper for paper in papers}
    jobs = [
        (by_id[paper_id], candidate)
        for paper_id, candidate in cache["fullTextCandidates"].items()
        if paper_id in by_id and paper_id not in cache["papers"]
    ]
    if not jobs:
        return 0
    added = 0
    full_text_count = 0
    with ThreadPoolExecutor(max_workers=min(workers, 4)) as executor:
        futures = [executor.submit(full_text_job, paper, candidate) for paper, candidate in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            paper, candidate, abstract, method, source_url = future.result()
            if abstract:
                if method == "ar5iv":
                    source = "Open arXiv full text (ar5iv HTML rendering)"
                    full_text_count += 1
                elif method == "pdf":
                    source = f"Open full text linked by {candidate['source']}"
                    full_text_count += 1
                else:
                    source = "Official arXiv record exact title and author match"
                if cache_record(
                    cache,
                    paper,
                    abstract,
                    source,
                    source_url,
                    matched_doi=candidate.get("matchedDoi"),
                ):
                    added += 1
            if completed % 10 == 0 or completed == len(jobs):
                json_dump(CACHE_PATH, cache, pretty=True)
            if completed % 50 == 0 or completed == len(jobs):
                print(
                    f"Open full texts: {completed}/{len(jobs)}; abstracts added {added}; "
                    f"verified in article text {full_text_count}",
                    flush=True,
                )
    return added


def icml_2026_urls() -> dict[str, str]:
    if not ICML_CACHE_PATH.exists():
        return {}
    payload = json.loads(ICML_CACHE_PATH.read_text(encoding="utf-8"))
    return {
        normalize_title(item.get("name", "")): (
            "https://icml.cc" + item["virtualsite_url"]
        )
        for item in payload.get("results", [])
        if item.get("name") and item.get("virtualsite_url")
    }


def fetch_recsys_official_abstracts(papers: list[dict], cache: dict) -> int:
    added = 0
    by_year_title = {
        (paper["year"], normalize_title(paper["title"])): paper
        for paper in papers
        if paper["venueId"] == "recsys" and paper["id"] not in cache["papers"]
    }
    for year, url in RECSYS_ACCEPTED_URLS.items():
        if not any(key[0] == year for key in by_year_title):
            continue
        try:
            soup = BeautifulSoup(request_html(url), "html.parser")
        except RuntimeError:
            continue
        for item in soup.select("ul.accordion > li"):
            anchor = item.find("a", rel=re.compile(r"accordion"))
            abstract_node = item.select_one("div p")
            if anchor is None or abstract_node is None:
                continue
            title_fragment = BeautifulSoup(str(anchor), "html.parser")
            for metadata in title_fragment.select("span.paper-type, em"):
                metadata.decompose()
            title = title_fragment.get_text(" ", strip=True)
            paper = by_year_title.get((year, normalize_title(title)))
            if not paper or not titles_match(paper["title"], title, threshold=0.97):
                continue
            if cache_record(
                cache,
                paper,
                abstract_node.get_text(" ", strip=True),
                "Official RecSys accepted list",
                url,
                matched_doi=paper["doi"],
            ):
                added += 1
        soup.decompose()
    if added:
        json_dump(CACHE_PATH, cache, pretty=True)
        print(f"Official RecSys accepted lists: abstracts added {added}", flush=True)
    return added


def direct_source_for_paper(paper: dict, icml_urls: dict[str, str]) -> tuple[str, str] | None:
    doi = paper["doi"] or ""
    host = urlparse(paper["paperUrl"]).netloc.casefold()
    if doi.startswith("10.1007/"):
        return "springer", f"https://link.springer.com/chapter/{doi}"
    if host == "proceedings.mlr.press":
        return "pmlr", paper["paperUrl"]
    if host in {"papers.nips.cc", "proceedings.neurips.cc"}:
        parsed = urlparse(paper["paperUrl"])
        return "neurips", f"https://proceedings.neurips.cc{parsed.path}"
    if host == "www.ijcai.org":
        return "ijcai", paper["paperUrl"]
    if paper["venueId"] == "icml" and paper["year"] == 2026:
        url = icml_urls.get(normalize_title(paper["title"]))
        if url:
            return "icml", url
    return None


def parse_direct_abstract(
    kind: str, url: str, expected_title: str, session=None
) -> tuple[str, str, str] | None:
    try:
        html = request_html(url, session=session)
    except (RuntimeError, OSError, requests.RequestException if requests else OSError):
        return None
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    abstract = ""
    if kind == "springer":
        title_meta = soup.select_one('meta[name="citation_title"]')
        title = title_meta.get("content", "") if title_meta else ""
        node = soup.select_one(
            'section[data-title="Abstract"] .c-article-section__content, '
            '#Abs1-content, section[aria-labelledby*="Abs"] .c-article-section__content'
        )
        abstract = node.get_text(" ", strip=True) if node else ""
    elif kind == "pmlr":
        title_meta = soup.select_one('meta[name="citation_title"]')
        title = title_meta.get("content", "") if title_meta else ""
        node = soup.select_one("#abstract, .abstract")
        abstract = node.get_text(" ", strip=True) if node else ""
    elif kind == "neurips":
        title_node = soup.select_one("h4, h1")
        title = title_node.get_text(" ", strip=True) if title_node else ""
        node = soup.select_one(".paper-abstract")
        abstract = node.get_text(" ", strip=True) if node else ""
    elif kind == "ijcai":
        title_node = soup.select_one("h1, h2")
        title = title_node.get_text(" ", strip=True) if title_node else expected_title
        nodes = soup.select("div.col-md-12")
        abstract = nodes[0].get_text(" ", strip=True) if nodes else ""
    elif kind == "icml":
        title_node = soup.select_one(".event-title, h1")
        title = title_node.get_text(" ", strip=True) if title_node else ""
        node = soup.select_one(".abstract-text-inner")
        abstract = node.get_text(" ", strip=True) if node else ""
    if title and not titles_match(expected_title, title):
        return None
    abstract = clean_abstract(abstract)
    if not abstract:
        return None
    return abstract, title or expected_title, url


def fetch_direct_abstracts(
    papers: list[dict], cache: dict, workers: int, allowed_kinds: set[str] | None = None
) -> int:
    icml_urls = icml_2026_urls()
    jobs: list[tuple[dict, str, str]] = []
    for paper in papers:
        if paper["id"] in cache["papers"]:
            continue
        source = direct_source_for_paper(paper, icml_urls)
        if source and (allowed_kinds is None or source[0] in allowed_kinds):
            jobs.append((paper, source[0], source[1]))

    added = 0
    source_names = {
        "springer": "Official Springer chapter page",
        "pmlr": "Official PMLR proceedings page",
        "neurips": "Official NeurIPS proceedings page",
        "ijcai": "Official IJCAI proceedings page",
        "icml": "Official ICML virtual program",
    }
    session = requests.Session() if requests else None
    if session is not None:
        session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    for completed, (paper, kind, url) in enumerate(jobs, start=1):
        result = parse_direct_abstract(
            kind,
            url,
            paper["title"],
            session=session if kind == "springer" else None,
        )
        if result and cache_record(
            cache,
            paper,
            result[0],
            source_names[kind],
            url,
            matched_doi=paper["doi"],
        ):
            added += 1
        if completed % 10 == 0 or completed == len(jobs):
            json_dump(CACHE_PATH, cache, pretty=True)
        if completed % 50 == 0 or completed == len(jobs):
            print(
                f"Official proceedings pages: {completed}/{len(jobs)}; abstracts added {added}",
                flush=True,
            )
    if session is not None:
        session.close()
    return added


def apply_cache(manifest: dict, documents: dict[str, dict], cache: dict) -> tuple[int, Counter]:
    abstract_total = 0
    sources: Counter = Counter()
    editions = {edition["id"]: edition for edition in manifest["editions"]}
    for edition in manifest["editions"]:
        edition["abstractCount"] = 0
        edition["missingAbstractCount"] = edition.get("paperCount", 0)
    for edition_id, document in documents.items():
        edition_abstracts = 0
        edition_papers = 0
        for category in document["categories"]:
            for paper in category["papers"]:
                edition_papers += 1
                record = cache["papers"].get(paper["id"])
                paper.pop("takeaway", None)
                paper["abstract"] = record["abstract"] if record else None
                if record:
                    edition_abstracts += 1
                    abstract_total += 1
                    sources[record["source"]] += 1
        document["abstractCount"] = edition_abstracts
        document["missingAbstractCount"] = edition_papers - edition_abstracts
        json_dump(DATA_DIR / f"{edition_id}.json", document)
        editions[edition_id]["abstractCount"] = edition_abstracts
        editions[edition_id]["missingAbstractCount"] = edition_papers - edition_abstracts
    manifest["abstractCount"] = abstract_total
    manifest["missingAbstractCount"] = manifest["paperCount"] - abstract_total
    return abstract_total, sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=100, choices=range(1, 101))
    parser.add_argument(
        "--skip-openalex",
        action="store_true",
        help="Skip DOI-index lookups and only query official proceedings pages.",
    )
    parser.add_argument(
        "--skip-official-tocs",
        action="store_true",
        help="Skip static official proceedings tables of contents.",
    )
    parser.add_argument(
        "--skip-arxiv",
        action="store_true",
        help="Skip exact-title discovery of open arXiv full texts.",
    )
    parser.add_argument(
        "--skip-full-text",
        action="store_true",
        help="Do not download and inspect verified open PDF candidates.",
    )
    parser.add_argument(
        "--refresh-arxiv",
        action="store_true",
        help="Repeat arXiv searches already completed for --verified on this date.",
    )
    parser.add_argument(
        "--arxiv-batch-size",
        type=int,
        default=20,
        choices=range(1, 26),
    )
    parser.add_argument(
        "--arxiv-delay",
        type=float,
        default=3.0,
        help="Seconds between arXiv API requests (default follows API guidance).",
    )
    parser.add_argument(
        "--direct-kinds",
        nargs="+",
        choices=("springer", "pmlr", "neurips", "ijcai", "icml"),
        help="Restrict official-page lookups to selected source types.",
    )
    parser.add_argument("--verified", default=date.today().isoformat())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest, documents, papers = load_documents()
    cache = load_cache()
    cache["lastVerified"] = args.verified

    resolved_dois = dict(cache.get("resolvedDois", {}))
    if not resolved_dois:
        resolved_dois.update(discover_sigir_2026_dois(papers))
        resolved_dois.update(discover_other_2026_dois(papers, args.workers))
        cache["resolvedDois"] = resolved_dois
        json_dump(CACHE_PATH, cache, pretty=True)
    print(f"Resolved {len(resolved_dois)} missing DOI records by exact title.", flush=True)

    if not args.skip_official_tocs:
        fetch_official_toc_abstracts(papers, cache)
    fetch_recsys_official_abstracts(papers, cache)
    if not args.skip_openalex:
        fetch_openalex_abstracts(papers, cache, resolved_dois, args.batch_size, args.workers)
    json_dump(CACHE_PATH, cache, pretty=True)
    fetch_direct_abstracts(
        papers,
        cache,
        min(args.workers, 6),
        set(args.direct_kinds) if args.direct_kinds else None,
    )
    json_dump(CACHE_PATH, cache, pretty=True)
    if not args.skip_arxiv:
        fetch_arxiv_full_text_candidates(
            papers,
            cache,
            batch_size=args.arxiv_batch_size,
            delay=max(0.0, args.arxiv_delay),
            checked_at=args.verified,
            refresh=args.refresh_arxiv,
        )
    if not args.skip_full_text:
        fetch_open_full_text_abstracts(papers, cache, args.workers)
    json_dump(CACHE_PATH, cache, pretty=True)

    abstract_total, sources = apply_cache(manifest, documents, cache)
    manifest["lastVerified"] = args.verified
    json_dump(MANIFEST_PATH, manifest, pretty=True)
    print(
        f"Attached {abstract_total}/{manifest['paperCount']} abstracts; "
        f"missing {manifest['paperCount'] - abstract_total}.",
        flush=True,
    )
    for source, count in sources.most_common():
        print(f"  {count:5}  {source}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
