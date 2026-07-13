#!/usr/bin/env python3
"""Build the ACM RecSys conference archive from the official accepted lists.

The generated JSON intentionally stores compact, paraphrased catalog notes rather
than copying the official abstracts.  Abstracts are used only while deriving
topic tags and high-level description signals.

This is a maintainer command, not part of the site build.  It requires
Beautiful Soup (`python3 -m pip install beautifulsoup4`).
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import re
import sys
import unicodedata
from urllib.request import Request, urlopen

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover - maintainer-facing failure path
    raise SystemExit(
        "Beautiful Soup is required: python3 -m pip install beautifulsoup4"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "src" / "data" / "recsys-conferences.json"

CONFERENCES = {
    2024: {
        "name": "RecSys 2024",
        "dates": "14–18 октября 2024",
        "location": "Бари, Италия",
        "acceptedUrl": "https://recsys.acm.org/recsys24/accepted-contributions/",
        "proceedingsUrl": "https://dl.acm.org/doi/proceedings/10.1145/3640457",
        "trackNames": [
            "Long Papers",
            "Short Papers",
            "Reproducibility",
            "LBR",
            "Demos",
            "Doctoral",
            "Industry",
        ],
    },
    2025: {
        "name": "RecSys 2025",
        "dates": "22–26 сентября 2025",
        "location": "Прага, Чехия",
        "acceptedUrl": "https://recsys.acm.org/recsys25/accepted-contributions/",
        "proceedingsUrl": "https://dl.acm.org/doi/proceedings/10.1145/3705328",
        "trackNames": [
            "Full Papers",
            "Short Papers",
            "Reproducibility",
            "Industry",
            "LBR",
            "Demo",
            "Doctoral",
        ],
    },
}

RECSYS_2026 = {
    "id": "recsys-2026",
    "name": "RecSys 2026",
    "year": 2026,
    "dates": "28 сентября – 2 октября 2026",
    "location": "Миннеаполис, Миннесота, США",
    "status": "pending",
    "statusNote": (
        "Уведомления по основным трекам вышли 9 июля, но официальный список "
        "принятых материалов и DOI ещё не опубликован. Записи появятся после "
        "публикации конференцией проверяемого списка."
    ),
    "sourceUrl": "https://recsys.acm.org/recsys26/call/",
    "expectedTracks": [
        "Long Papers",
        "Short Papers",
        "Past, Present and Future",
        "Reproducibility and Replicability",
        "Resource Papers",
        "Industry",
        "Research and Practice Notes",
        "Doctoral Symposium",
        "Demos",
    ],
    "tracks": [],
}

TOPIC_RULES = [
    ("conversational recommendation", r"conversational recomm|dialog(?:ue)? recomm"),
    ("generative recommendation", r"generative recomm|recommendation generation"),
    ("LLM", r"\bllms?\b|large language model|language-model-based|foundation model"),
    ("sequential recommendation", r"sequential recomm|next-item|user sequence|sequence model"),
    ("session-based recommendation", r"session-based|session-aware|session recommendation"),
    ("multimodal recommendation", r"multi[ -]?modal|vision-language|visual and text|image and text"),
    ("graph recommendation", r"graph neural|graph-based recomm|graph contrastive|knowledge graph|\bgnns?\b"),
    ("cross-domain recommendation", r"cross[ -]?domain"),
    ("multi-behavior recommendation", r"multi[ -]?behavio|multiple behavior"),
    ("bundle recommendation", r"bundle recomm"),
    ("group recommendation", r"group recomm"),
    ("cold-start", r"cold[ -]?start|zero[ -]?shot"),
    ("fairness and bias", r"fairness|unfair|debias|bias mitigation|exposure dispar"),
    ("privacy", r"privacy|federated|differentially private"),
    ("robustness and security", r"robust|poison|attack|adversarial|shilling|security"),
    ("explainability", r"explain|interpretability|interpretable|counterfactual explanation"),
    ("causal recommendation", r"causal|counterfactual|off-policy|propensity|treatment effect"),
    ("bandits and reinforcement learning", r"bandit|reinforcement learning|\brl\b"),
    ("ranking and retrieval", r"ranking|retrieval|candidate generation|learning to rank"),
    ("CTR prediction", r"click-through|\bctr\b|conversion rate"),
    ("evaluation", r"evaluation metric|offline evaluation|online evaluation|benchmark"),
    ("reproducibility", r"reproduc|replicat"),
    ("datasets and resources", r"\bdataset\b|data set|benchmark suite|open-source framework"),
    ("user modeling", r"user model|user representation|user preference|preference elicitation"),
    ("popularity and long tail", r"popularity|long[ -]?tail|head and tail"),
    ("diversity and novelty", r"diversity|novelty|serendip"),
    ("multi-objective recommendation", r"multi[ -]?objective|multiple objectives|pareto"),
    ("music recommendation", r"music recomm|playlist|song recomm|track recomm"),
    ("news recommendation", r"news recomm|news feed"),
    ("video recommendation", r"video recomm|short-form video|short video"),
    ("e-commerce", r"e-commerce|ecommerce|product recomm|retail"),
    ("tourism recommendation", r"touris|travel recomm|route recomm"),
    ("education recommendation", r"education|learning resource|course recomm"),
    ("health recommendation", r"health|clinical|therapy"),
    ("sustainability", r"sustainab|carbon|environmental"),
    ("human-centered RecSys", r"human-centered|human centred|user stud|human-ai|co-design"),
]

INDUSTRY_RE = re.compile(
    r"(?:\b(?:Google|Amazon|Meta|Netflix|Spotify|Microsoft|Alibaba|ByteDance|TikTok|"
    r"Kuaishou|Huawei|NVIDIA|Pinterest|Snap|Apple|Adobe|IBM|Yahoo|Zalando|eBay|"
    r"Expedia|XING|IKEA|ZDF|Aampe|ASOS|NAVER|Bloomberg|Comcast|Rakuten|Deezer|"
    r"SoundCloud|Samsung|LINE|Bosch|Criteo|Accenture|Indeed|LinkedIn|Mozilla|BBC|"
    r"ARD|Schibsted|Salesforce|Tripadvisor|Walmart|Wayfair|Tencent|Baidu|Meituan|"
    r"Shopee|Grab|Airbnb|Etsy|Reddit|Intuit|Ericsson|Nokia|Vodafone|Telef[oó]nica|"
    r"Telekom|Cookpad|Kakao|Vinted|Saks)\b|\bBooking(?:\.com)?\b|\bJD\.com\b|"
    r"\bBol\.com\b|\bDelivery Hero\b|\bMercado Libre\b|\bDPG Media\b|"
    r"(?:^|\s)(?:Inc\.?|Ltd\.?|LLC|GmbH|Corp(?:oration)?\.?)$)",
    re.IGNORECASE,
)


def fetch_html(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "recsys-papers-catalog/1.0 (+https://github.com/pechatov/recsys-papers)"
        },
    )
    with urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")


def split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    index = 0
    while index < len(value):
        char = value[index]
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        elif depth == 0 and char in ",;":
            parts.append(value[start:index])
            start = index + 1
        elif depth == 0 and value[index : index + 5].lower() == " and ":
            parts.append(value[start:index])
            start = index + 5
            index += 4
        index += 1
    parts.append(value[start:])
    return [part.strip() for part in parts if part.strip()]


def parse_author_line(author_line: str) -> tuple[list[str], list[str]]:
    value = re.sub(r"^by\s+", "", author_line.strip(), flags=re.IGNORECASE)
    authors: list[str] = []
    affiliations: list[str] = []

    for part in split_top_level(value):
        first_paren = part.find("(")
        if first_paren == -1:
            name = part.strip()
            part_affiliations: list[str] = []
        else:
            name = part[:first_paren].strip()
            part_affiliations = []
            depth = 0
            content_start: int | None = None
            for index, char in enumerate(part[first_paren:], start=first_paren):
                if char == "(":
                    if depth == 0:
                        content_start = index + 1
                    depth += 1
                elif char == ")" and depth:
                    depth -= 1
                    if depth == 0 and content_start is not None:
                        affiliation = part[content_start:index].strip()
                        if affiliation:
                            part_affiliations.append(affiliation)
                        content_start = None

        name = re.sub(r"^and\s+", "", name, flags=re.IGNORECASE).strip()
        if name:
            authors.append(name)
        for affiliation_group in part_affiliations:
            for affiliation in affiliation_group.split(";"):
                affiliation = affiliation.strip()
                if affiliation and affiliation not in affiliations:
                    affiliations.append(affiliation)

    return authors, affiliations


def infer_tags(title: str, abstract: str) -> list[str]:
    haystack = f"{title}\n{abstract}".lower()
    title_lower = title.lower()
    matches: list[tuple[int, str]] = []
    for label, pattern in TOPIC_RULES:
        if re.search(pattern, haystack, flags=re.IGNORECASE):
            priority = 0 if re.search(pattern, title_lower, flags=re.IGNORECASE) else 1
            matches.append((priority, label))
    matches.sort(key=lambda item: item[0])
    return [label for _, label in matches[:4]]


def build_takeaway(abstract: str, tags: list[str], track_name: str) -> str:
    lower = abstract.lower()
    if tags:
        topic_text = ", ".join(tags[:3])
        opening = f"Работа посвящена темам: {topic_text}."
    else:
        opening = "Работа рассматривает отдельную задачу в области рекомендательных систем."

    signals: list[str] = []
    if re.search(r"\b(we propose|we introduce|we present|this paper proposes|novel (?:method|model|framework|approach|system))\b", lower):
        signals.append("предлагает новый подход")
    elif re.search(r"\b(we study|we investigate|we analyze|we analyse|this study)\b", lower):
        signals.append("проводит систематический анализ")

    if re.search(r"user stud|participant|survey|interview|controlled stud", lower):
        signals.append("включает исследование с пользователями")
    elif re.search(r"experiment|empirical|evaluat|benchmark", lower):
        signals.append("проверяет выводы экспериментально")

    if re.search(r"online experiment|online a/b|a/b test|deployed|production", lower):
        signals.append("учитывает production или online-сценарий")
    if re.search(r"github\.com|open[- ]source|code is available|publicly available", lower):
        signals.append("сопровождается открытым артефактом")

    if signals:
        detail = " и ".join(signals[:3])
        return f"{opening} По официальному описанию материал {detail}."

    track = track_name.rstrip("s")
    return f"{opening} Материал опубликован в категории {track}."


def paper_type_and_title(item) -> tuple[str, str, str, str]:
    anchor = item.find("a", rel=re.compile(r"accordion"))
    if anchor is None:
        raise ValueError("Paper entry has no accordion title link")
    paper_type = anchor.find("span", class_="paper-type")
    type_code = paper_type.get_text(" ", strip=True) if paper_type else ""
    type_name = paper_type.get("title", "") if paper_type else ""
    if paper_type:
        paper_type.extract()
    author_element = anchor.find("em")
    author_line = author_element.get_text(" ", strip=True) if author_element else ""
    if author_element:
        author_element.extract()
    title = anchor.get_text(" ", strip=True)
    return type_code, type_name, title, author_line


def parse_conference(year: int, html: str) -> dict:
    config = CONFERENCES[year]
    soup = BeautifulSoup(html, "html.parser")
    sections = soup.select("div.tabs-content")
    expected_tracks = config["trackNames"]
    if len(sections) != len(expected_tracks):
        raise ValueError(
            f"RecSys {year}: expected {len(expected_tracks)} track sections, found {len(sections)}"
        )

    tracks: list[dict] = []
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()

    for section, track_name in zip(sections, expected_tracks):
        paper_list = section.find("ul", class_="accordion")
        if paper_list is None:
            raise ValueError(f"RecSys {year} / {track_name}: paper list not found")

        papers: list[dict] = []
        for item in paper_list.find_all("li", recursive=False):
            type_code, type_name, title, author_line = paper_type_and_title(item)
            authors, affiliations = parse_author_line(author_line)

            doi_match = re.search(r'https://doi\.org/[^\"\s<]+', str(item), flags=re.IGNORECASE)
            if doi_match is None:
                raise ValueError(f"RecSys {year} / {track_name}: DOI missing for {title}")
            doi_url = doi_match.group(0).rstrip(".,;)")
            doi_url = re.sub(
                r"^https://doi\.org/(?:https://doi\.org/)+",
                "https://doi.org/",
                doi_url,
                flags=re.IGNORECASE,
            )

            abstract_container = item.find("div")
            abstract_element = abstract_container.find("p") if abstract_container else None
            abstract = abstract_element.get_text(" ", strip=True) if abstract_element else ""
            tags = infer_tags(title, abstract)
            industry_affiliations = [
                affiliation for affiliation in affiliations if INDUSTRY_RE.search(affiliation)
            ]

            doi_key = doi_url.lower()
            title_key = title.casefold()
            if doi_key in seen_dois:
                raise ValueError(f"RecSys {year}: duplicate DOI {doi_url}")
            if title_key in seen_titles:
                raise ValueError(f"RecSys {year}: duplicate title {title}")
            seen_dois.add(doi_key)
            seen_titles.add(title_key)

            papers.append(
                {
                    "id": f"recsys-{year}-{slugify(track_name)}-{slugify(title)}",
                    "title": title,
                    "authors": authors,
                    "affiliations": affiliations,
                    "industryAffiliations": industry_affiliations,
                    "isIndustry": track_name == "Industry" or bool(industry_affiliations),
                    "typeCode": type_code,
                    "typeName": type_name,
                    "doiUrl": doi_url,
                    "tags": tags,
                    "takeaway": build_takeaway(abstract, tags, track_name),
                }
            )

        tracks.append(
            {
                "id": f"recsys-{year}-{slugify(track_name)}",
                "name": track_name,
                "papers": papers,
            }
        )

    return {
        "id": f"recsys-{year}",
        "name": config["name"],
        "year": year,
        "dates": config["dates"],
        "location": config["location"],
        "status": "published",
        "sourceUrl": config["acceptedUrl"],
        "proceedingsUrl": config["proceedingsUrl"],
        "tracks": tracks,
    }


def read_source(year: int, source_dir: Path | None) -> str:
    if source_dir is None:
        return fetch_html(CONFERENCES[year]["acceptedUrl"])
    source_path = source_dir / f"recsys{year}.html"
    if not source_path.exists():
        raise SystemExit(f"Missing cached source: {source_path}")
    return source_path.read_text(encoding="utf-8")


def build_catalog(source_dir: Path | None, verified: str) -> dict:
    conferences = [dict(RECSYS_2026)]
    for year in sorted(CONFERENCES, reverse=True):
        conferences.append(parse_conference(year, read_source(year, source_dir)))

    paper_count = sum(
        len(track["papers"])
        for conference in conferences
        for track in conference["tracks"]
    )
    return {
        "lastVerified": verified,
        "scope": "Все опубликованные материалы официальных списков ACM RecSys, без тематического отбора.",
        "trackerUrl": "https://recsys.info/",
        "paperCount": paper_count,
        "conferences": conferences,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Read recsys2024.html and recsys2025.html from this directory instead of downloading.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verified", default=date.today().isoformat())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = build_catalog(args.source_dir, args.verified)
    if catalog["paperCount"] != 406:
        raise SystemExit(
            f"Completeness guard failed: expected 406 published papers, got {catalog['paperCount']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {catalog['paperCount']} papers to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
