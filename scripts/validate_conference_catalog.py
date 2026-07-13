#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

from update_conference_archive import PROFILE_VENUE_IDS, recsys_relevance_reason


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "src" / "data" / "conference-archive.json"
DATA_DIR = ROOT / "public" / "data" / "conference-papers"
EXPECTED_VENUES = {
    "recsys", "aaai", "chi", "cikm", "ecai", "ecir", "ecml-pkdd", "icde",
    "icdm", "icml", "ijcai", "kdd", "neurips", "sigir", "umap", "wsdm", "www",
}
EXPECTED_SOURCE_COUNTS = {
    "recsys-2024": 190, "recsys-2025": 216,
    "aaai-2024": 2866, "aaai-2025": 3486, "aaai-2026": 4920,
    "chi-2024": 1705, "chi-2025": 2169, "chi-2026": 2707,
    "cikm-2024": 697, "cikm-2025": 852,
    "ecai-2024": 606, "ecai-2025": 696,
    "ecir-2024": 210, "ecir-2025": 206, "ecir-2026": 209,
    "ecml-pkdd-2024": 272, "ecml-pkdd-2025": 317,
    "icde-2024": 501, "icde-2025": 398, "icde-2026": 357,
    "icdm-2024": 116, "icdm-2025": 180,
    "icml-2024": 2610, "icml-2025": 3330, "icml-2026": 6628,
    "ijcai-2024": 1048, "ijcai-2025": 1280, "ijcai-2026": 990,
    "kdd-2024": 642, "kdd-2025": 844, "kdd-2026": 256,
    "neurips-2024": 4494, "neurips-2025": 3317,
    "sigir-2024": 389, "sigir-2025": 540, "sigir-2026": 667,
    "umap-2024": 143, "umap-2025": 148, "umap-2026": 124,
    "wsdm-2024": 175, "wsdm-2025": 150, "wsdm-2026": 216,
    "www-2024": 778, "www-2025": 978, "www-2026": 1191,
}
EXPECTED_COUNTS = {
    "recsys-2024": 190, "recsys-2025": 216,
    "aaai-2024": 52, "aaai-2025": 86, "aaai-2026": 106,
    "chi-2024": 8, "chi-2025": 16, "chi-2026": 14,
    "cikm-2024": 697, "cikm-2025": 852,
    "ecai-2024": 7, "ecai-2025": 3,
    "ecir-2024": 210, "ecir-2025": 206, "ecir-2026": 209,
    "ecml-pkdd-2024": 12, "ecml-pkdd-2025": 14,
    "icde-2024": 29, "icde-2025": 13, "icde-2026": 13,
    "icdm-2024": 19, "icdm-2025": 13,
    "icml-2024": 8, "icml-2025": 17, "icml-2026": 22,
    "ijcai-2024": 19, "ijcai-2025": 27, "ijcai-2026": 21,
    "kdd-2024": 642, "kdd-2025": 844, "kdd-2026": 256,
    "neurips-2024": 23, "neurips-2025": 22,
    "sigir-2024": 389, "sigir-2025": 540, "sigir-2026": 667,
    "umap-2024": 143, "umap-2025": 148, "umap-2026": 124,
    "wsdm-2024": 52, "wsdm-2025": 43, "wsdm-2026": 49,
    "www-2024": 147, "www-2025": 143, "www-2026": 176,
}
EXPECTED_PENDING = {
    "recsys-2026", "cikm-2026", "ecai-2026", "ecml-pkdd-2026",
    "icdm-2026", "neurips-2026",
}
EXPECTED_PENDING_PUBLICATION_DATES = {
    "recsys-2026": "2026-09-28",
    "cikm-2026": "2026-11-07",
    "ecai-2026": "2026-08-15",
    "ecml-pkdd-2026": "2026-09-07",
    "icdm-2026": "2026-11-12",
    "neurips-2026": "2026-12-06",
}

FILTER_CASES = [
    ("Graph Contrastive Learning", "RecSys: Graphs", True),
    ("Collaborative Filtering with Sparse Feedback", "Research Papers", True),
    ("Learning User-Item Representations", "Research Papers", True),
    ("A Calibrated CTR Prediction Model", "Research Papers", True),
    ("Semantic IDs for Item Retrieval", "Research Papers", True),
    ("Label Recommendation for Point Cloud Segmentation", "Research Papers", False),
    ("Action Recommendations Under Data Scarcity", "Research Papers", False),
    ("Policy Recommendations for Responsible Innovation", "Research Papers", False),
    ("Design Recommendations for AI Interfaces", "Research Papers", False),
    ("Learning to Rank Documents", "Research Papers", False),
]


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    venues = manifest.get("venues", [])
    venue_ids = {venue["id"] for venue in venues}
    if venue_ids != EXPECTED_VENUES:
        errors.append(f"venue set differs: expected {sorted(EXPECTED_VENUES)}, got {sorted(venue_ids)}")
    for venue in venues:
        expected_policy = (
            "complete-profile-venue"
            if venue["id"] in PROFILE_VENUE_IDS
            else "recsys-related-only"
        )
        if venue.get("selectionPolicy") != expected_policy:
            errors.append(
                f"{venue['id']}: expected venue selection policy {expected_policy}, "
                f"got {venue.get('selectionPolicy')}"
            )

    selection_policy = manifest.get("selectionPolicy", {})
    if set(selection_policy.get("profileVenueIds", [])) != PROFILE_VENUE_IDS:
        errors.append("manifest profile venue policy differs from the configured set")
    if set(selection_policy.get("broadVenueIds", [])) != EXPECTED_VENUES - PROFILE_VENUE_IDS:
        errors.append("manifest broad venue policy differs from the configured set")

    for title, category, expected in FILTER_CASES:
        actual = recsys_relevance_reason(title, category) is not None
        if actual != expected:
            errors.append(
                f"topic filter case failed: {title!r} / {category!r}: "
                f"expected {expected}, got {actual}"
            )

    editions = manifest.get("editions", [])
    if len(editions) != len(EXPECTED_VENUES) * 3:
        errors.append(f"expected 51 venue/year editions, got {len(editions)}")
    edition_ids = {edition["id"] for edition in editions}
    if len(edition_ids) != len(editions):
        errors.append("duplicate edition ids in manifest")

    actual_pending = {edition["id"] for edition in editions if edition["status"] == "pending"}
    if actual_pending != EXPECTED_PENDING:
        errors.append(f"pending edition set differs: {sorted(actual_pending)}")
    actual_pending_publication_dates = {
        edition["id"]: edition.get("expectedPublicationDate")
        for edition in editions
        if edition["status"] == "pending"
    }
    if actual_pending_publication_dates != EXPECTED_PENDING_PUBLICATION_DATES:
        errors.append(
            "pending publication dates differ: "
            f"expected {EXPECTED_PENDING_PUBLICATION_DATES}, "
            f"got {actual_pending_publication_dates}"
        )
    partial = {edition["id"] for edition in editions if edition["status"] == "partial"}
    if partial != {"kdd-2026"}:
        errors.append(f"unexpected partial edition set: {sorted(partial)}")

    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    paper_lookup: dict[tuple[str, int, int], dict] = {}
    counted_papers = 0
    counted_abstracts = 0
    expected_files = {"search-index.json"}

    for edition in editions:
        edition_id = edition["id"]
        expected_policy = (
            "complete-profile-venue"
            if edition["venueId"] in PROFILE_VENUE_IDS
            else "recsys-related-only"
        )
        if edition.get("selectionPolicy") != expected_policy:
            errors.append(
                f"{edition_id}: expected selection policy {expected_policy}, "
                f"got {edition.get('selectionPolicy')}"
            )
        expected_source_count = EXPECTED_SOURCE_COUNTS.get(edition_id, 0)
        if edition.get("sourcePaperCount") != expected_source_count:
            errors.append(
                f"{edition_id}: expected source count {expected_source_count}, "
                f"got {edition.get('sourcePaperCount')}"
            )
        expected_filtered_count = expected_source_count - EXPECTED_COUNTS.get(edition_id, 0)
        if edition.get("filteredOutCount") != expected_filtered_count:
            errors.append(
                f"{edition_id}: expected filtered count {expected_filtered_count}, "
                f"got {edition.get('filteredOutCount')}"
            )
        file_name = edition.get("file")
        if edition["status"] == "pending":
            if file_name or edition["paperCount"] or edition["categories"]:
                errors.append(f"{edition_id}: pending edition contains published data")
            continue
        if not file_name:
            errors.append(f"{edition_id}: published edition has no data file")
            continue
        expected_files.add(file_name)
        path = DATA_DIR / file_name
        if not path.exists():
            errors.append(f"{edition_id}: missing data file {file_name}")
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("id") != edition_id:
            errors.append(f"{edition_id}: data file id differs")

        category_counts = []
        edition_count = 0
        edition_abstracts = 0
        for category_index, category in enumerate(document.get("categories", [])):
            category_counts.append({"name": category["name"], "count": len(category["papers"])})
            for paper_index, paper in enumerate(category["papers"]):
                edition_count += 1
                counted_papers += 1
                for field in ("id", "title", "authors", "paperUrl", "metadataSource"):
                    if not paper.get(field):
                        errors.append(f"{edition_id}/{category['name']}: missing {field}")
                if "takeaway" in paper:
                    errors.append(f"{edition_id}/{paper.get('id')}: legacy takeaway must not be stored")
                if "abstract" not in paper:
                    errors.append(f"{edition_id}/{paper.get('id')}: abstract field is missing")
                elif paper["abstract"]:
                    edition_abstracts += 1
                    counted_abstracts += 1
                    if len(paper["abstract"]) < 40:
                        errors.append(f"{edition_id}/{paper.get('id')}: abstract is implausibly short")
                    if re.search(r"</?[A-Za-z][^>]*>", paper["abstract"]):
                        errors.append(f"{edition_id}/{paper.get('id')}: abstract contains raw HTML")
                paper_id = paper.get("id", "")
                paper_url = paper.get("paperUrl", "").casefold()
                if paper_id in seen_ids:
                    errors.append(f"duplicate paper id: {paper_id}")
                if paper_url in seen_urls:
                    errors.append(f"duplicate canonical paper URL: {paper_url}")
                if paper.get("doiUrl") and not paper["doiUrl"].startswith("https://doi.org/"):
                    errors.append(f"invalid DOI URL: {paper['doiUrl']}")
                seen_ids.add(paper_id)
                seen_urls.add(paper_url)
                paper_lookup[(edition_id, category_index, paper_index)] = paper
                if (
                    edition["venueId"] not in PROFILE_VENUE_IDS
                    and recsys_relevance_reason(paper["title"], category["name"]) is None
                ):
                    errors.append(
                        f"{edition_id}/{paper_id}: broad-venue paper has no recsys selection signal"
                    )

        if edition_count != edition["paperCount"]:
            errors.append(f"{edition_id}: manifest count {edition['paperCount']} != file count {edition_count}")
        if edition_count != EXPECTED_COUNTS.get(edition_id):
            errors.append(f"{edition_id}: expected {EXPECTED_COUNTS.get(edition_id)}, got {edition_count}")
        if category_counts != edition["categories"]:
            errors.append(f"{edition_id}: category summary differs from data file")
        if document.get("abstractCount") != edition_abstracts:
            errors.append(
                f"{edition_id}: data abstractCount {document.get('abstractCount')} != {edition_abstracts}"
            )
        if edition.get("abstractCount") != edition_abstracts:
            errors.append(
                f"{edition_id}: manifest abstractCount {edition.get('abstractCount')} != {edition_abstracts}"
            )
        if edition.get("missingAbstractCount") != edition_count - edition_abstracts:
            errors.append(f"{edition_id}: manifest missingAbstractCount differs")

    present_files = {path.name for path in DATA_DIR.glob("*.json")}
    if present_files != expected_files:
        errors.append(
            f"conference data file set differs; missing={sorted(expected_files - present_files)}, "
            f"unexpected={sorted(present_files - expected_files)}"
        )

    search_path = DATA_DIR / "search-index.json"
    if search_path.exists():
        search = json.loads(search_path.read_text(encoding="utf-8"))
        rows = search.get("papers", [])
        if len(rows) != counted_papers:
            errors.append(f"search index has {len(rows)} rows for {counted_papers} papers")
        for row in rows:
            if len(row) != 8:
                errors.append(f"search row width is not 8: {row[:3]}")
                continue
            paper = paper_lookup.get((row[0], row[1], row[2]))
            if paper is None:
                errors.append(f"search row references missing paper: {row[:3]}")
                continue
            if row[3] != paper["title"] or row[6] != paper["paperUrl"]:
                errors.append(f"search row metadata differs for {row[:3]}")

    if counted_papers != manifest.get("paperCount"):
        errors.append(f"manifest paperCount {manifest.get('paperCount')} != counted {counted_papers}")
    if counted_papers != sum(EXPECTED_COUNTS.values()):
        errors.append(f"completeness total differs: expected {sum(EXPECTED_COUNTS.values())}, got {counted_papers}")
    if manifest.get("sourcePaperCount") != sum(EXPECTED_SOURCE_COUNTS.values()):
        errors.append(
            f"source completeness total differs: expected {sum(EXPECTED_SOURCE_COUNTS.values())}, "
            f"got {manifest.get('sourcePaperCount')}"
        )
    if manifest.get("abstractCount") != counted_abstracts:
        errors.append(
            f"manifest abstractCount {manifest.get('abstractCount')} != counted {counted_abstracts}"
        )
    if manifest.get("missingAbstractCount") != counted_papers - counted_abstracts:
        errors.append("manifest missingAbstractCount differs from paper coverage")
    if counted_papers and counted_abstracts / counted_papers < 0.80:
        errors.append(
            f"abstract coverage is below 80%: {counted_abstracts}/{counted_papers}"
        )

    for error in errors:
        print(f"ERROR {error}")
    if errors:
        return 1
    print(
        f"Conference archive valid: {len(EXPECTED_VENUES)} venues, {len(editions)} editions, "
        f"{counted_papers} papers, {counted_abstracts} abstracts, "
        f"{len(seen_ids)} unique IDs and canonical URLs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
