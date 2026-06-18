#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from pathlib import Path
from urllib.parse import quote, urlparse
import difflib
import json
import re
import time

import requests
from bs4 import BeautifulSoup, Tag


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "summaries" / "papers.html"
CACHE_PATH = ROOT / ".cache" / "paper_abstracts.json"
USER_AGENT = "recsys-papers-catalog/0.1 (local metadata enrichment)"


@dataclass
class CatalogEntry:
    title: str
    href: str
    ol: Tag
    details: Tag | None


def normalize_title(value: str) -> str:
    value = unescape(value).lower()
    value = re.sub(r"[^a-z0-9а-яё]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def clean_abstract(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in index.items():
        for idx in idxs:
            positions.append((idx, word))
    if not positions:
        return None
    return clean_abstract(" ".join(word for _, word in sorted(positions)))


def extract_arxiv_id(url: str) -> str | None:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#/]+)", url)
    if not match:
        return None
    arxiv_id = match.group(1).removesuffix(".pdf")
    return re.sub(r"v\d+$", "", arxiv_id)


def extract_doi(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.netloc == "doi.org":
        return parsed.path.lstrip("/")
    if "doi/" in parsed.path:
        return parsed.path.split("doi/", 1)[1].lstrip("/")
    return None


def load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_entries(soup: BeautifulSoup) -> list[CatalogEntry]:
    entries: list[CatalogEntry] = []
    for ol in soup.find_all("ol"):
        link = ol.find("a", href=True)
        if link is None:
            continue
        details = ol.find_next_sibling()
        while details is not None and getattr(details, "name", None) not in {"ul", "ol", "h2"}:
            details = details.find_next_sibling()
        if getattr(details, "name", None) != "ul":
            details = None
        entries.append(CatalogEntry(title=link.get_text(" ", strip=True), href=link["href"], ol=ol, details=details))
    return entries


def fetch_arxiv_abstracts(ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    result: dict[str, str] = {}
    # arXiv accepts comma-separated id_list; keep chunks conservative.
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        url = "https://export.arxiv.org/api/query?id_list=" + ",".join(chunk) + f"&max_results={len(chunk)}"
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
        feed = BeautifulSoup(response.text, "xml")
        for entry in feed.find_all("entry"):
            raw_id = entry.find("id")
            summary = entry.find("summary")
            if raw_id is None or summary is None:
                continue
            arxiv_id = extract_arxiv_id(raw_id.get_text("", strip=True))
            if arxiv_id:
                result[arxiv_id] = clean_abstract(summary.get_text(" ", strip=True))
        time.sleep(0.5)
    return result


def openalex_by_doi(doi: str) -> str | None:
    url = "https://api.openalex.org/works/https://doi.org/" + quote(doi, safe="/")
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return abstract_from_inverted_index(response.json().get("abstract_inverted_index"))


def openalex_by_title(title: str) -> str | None:
    url = "https://api.openalex.org/works"
    response = requests.get(
        url,
        params={"search": title, "per-page": 5, "select": "title,abstract_inverted_index"},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    target = normalize_title(title)
    best: tuple[float, str | None] = (0.0, None)
    for item in response.json().get("results", []):
        candidate_title = item.get("title") or ""
        score = difflib.SequenceMatcher(None, target, normalize_title(candidate_title)).ratio()
        abstract = abstract_from_inverted_index(item.get("abstract_inverted_index"))
        if abstract and score > best[0]:
            best = (score, abstract)
    return best[1] if best[0] >= 0.82 else None


def crossref_by_doi(doi: str) -> str | None:
    url = "https://api.crossref.org/works/" + quote(doi, safe="/")
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    abstract = response.json().get("message", {}).get("abstract")
    return clean_abstract(abstract) if abstract else None


def publisher_page_abstract(url: str) -> str | None:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20, allow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for selector in [
        "section#Abs1 p",
        "div.c-article-section__content p",
        'meta[name="citation_abstract"]',
        'meta[name="dc.description"]',
        'meta[name="description"]',
        'meta[name="twitter:description"]',
    ]:
        element = soup.select_one(selector)
        if element is None:
            continue
        value = element.get("content", "") if element.name == "meta" else element.get_text(" ", strip=True)
        value = clean_abstract(value)
        if value and not value.endswith("...") and len(value.split()) >= 25:
            return value
    return None


def lookup_abstract(entry: CatalogEntry, arxiv_cache: dict[str, str], cache: dict[str, str]) -> str | None:
    key = entry.href
    if key in cache:
        return cache[key]

    arxiv_id = extract_arxiv_id(entry.href)
    if arxiv_id and arxiv_id in arxiv_cache:
        cache[key] = arxiv_cache[arxiv_id]
        return cache[key]

    doi = extract_doi(entry.href)
    abstract = None
    if doi:
        try:
            abstract = openalex_by_doi(doi)
        except requests.RequestException:
            abstract = None
        if not abstract:
            try:
                abstract = crossref_by_doi(doi)
            except requests.RequestException:
                abstract = None

    if not abstract:
        try:
            abstract = openalex_by_title(entry.title)
        except requests.RequestException:
            abstract = None

    if not abstract:
        try:
            abstract = publisher_page_abstract(entry.href)
        except requests.RequestException:
            abstract = None

    if abstract:
        cache[key] = abstract
    return abstract


def ensure_missing_badge(soup: BeautifulSoup, entry: CatalogEntry) -> None:
    if entry.details is None:
        return
    tag_li = None
    for li in entry.details.find_all("li", recursive=False):
        strong = li.find("strong")
        if strong and "Теги" in strong.get_text(" ", strip=True):
            tag_li = li
            break
    if tag_li is None:
        tag_li = soup.new_tag("li")
        strong = soup.new_tag("strong")
        strong.string = "Теги:"
        tag_li.append(strong)
        tag_li.append(" ")
        entry.details.append(tag_li)
    if not tag_li.find("span", class_="summary-missing"):
        tag_li.append(" ")
        badge = soup.new_tag("span", attrs={"class": "badge summary-missing"})
        badge.string = "саммари пропущено"
        tag_li.append(badge)


def set_takeaway_to_abstract(soup: BeautifulSoup, entry: CatalogEntry, abstract: str | None) -> None:
    if entry.details is None:
        return
    takeaway_li = None
    for li in entry.details.find_all("li", recursive=False):
        strong = li.find("strong")
        if strong and "Выжимка" in strong.get_text(" ", strip=True):
            takeaway_li = li
            break
    if takeaway_li is None:
        takeaway_li = soup.new_tag("li")
        strong = soup.new_tag("strong")
        strong.string = "Выжимка:"
        takeaway_li.append(strong)
        entry.details.append(takeaway_li)

    for child in list(takeaway_li.children):
        if isinstance(child, Tag) and child.name == "div" and "takeaway" in child.get("class", []):
            child.decompose()

    div = soup.new_tag("div", attrs={"class": "takeaway"})
    inner = soup.new_tag("div")
    if abstract:
        strong = soup.new_tag("strong")
        strong.string = "Abstract:"
        inner.append(strong)
        inner.append(" " + abstract)
    else:
        inner.string = "Abstract не удалось получить автоматически; нужен ручной перенос из статьи."
    div.append(inner)
    takeaway_li.append(div)


def clean_catalog_takeaways(soup: BeautifulSoup) -> None:
    for div in soup.select("div.takeaway div"):
        if "К общему списку статей" in div.get_text(" ", strip=True):
            div.decompose()


def add_css_for_missing_badge(soup: BeautifulSoup) -> None:
    style = soup.find("style")
    if style is None or ".badge.summary-missing" in style.get_text():
        return
    css = ".badge.summary-missing { color: var(--muted); background: var(--pill-bg); border-color: var(--border); font-weight: 700; }\n"
    style.string = (style.string or "") + "\n" + css


def main() -> int:
    soup = BeautifulSoup(CATALOG.read_text(encoding="utf-8"), "html.parser")
    entries = get_entries(soup)
    missing = [entry for entry in entries if not entry.href.startswith("paper_summaries/")]

    cache = load_cache()
    arxiv_ids = sorted({extract_arxiv_id(entry.href) for entry in missing if extract_arxiv_id(entry.href)})
    arxiv_cache = fetch_arxiv_abstracts([value for value in arxiv_ids if value])

    missing_abstracts = 0
    for idx, entry in enumerate(missing, start=1):
        abstract = lookup_abstract(entry, arxiv_cache, cache)
        if not abstract:
            missing_abstracts += 1
        ensure_missing_badge(soup, entry)
        set_takeaway_to_abstract(soup, entry, abstract)
        if idx % 20 == 0:
            save_cache(cache)

    clean_catalog_takeaways(soup)
    add_css_for_missing_badge(soup)
    save_cache(cache)
    CATALOG.write_text(str(soup), encoding="utf-8")
    print(f"Annotated {len(missing)} missing-summary catalog entries")
    print(f"Resolved abstracts: {len(missing) - missing_abstracts}; unresolved: {missing_abstracts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
