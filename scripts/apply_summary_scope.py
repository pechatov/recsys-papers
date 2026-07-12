#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

from bs4 import BeautifulSoup, NavigableString, Tag

sys.path.insert(0, str(Path(__file__).resolve().parent))
from update_catalog_missing_summaries import (  # noqa: E402
    CatalogEntry,
    add_css_for_missing_badge,
    clean_abstract,
    extract_arxiv_id,
    fetch_arxiv_abstracts,
    get_entries,
    load_cache,
    lookup_abstract,
    save_cache,
    set_takeaway_to_abstract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "src" / "content" / "paper_summaries"
CATALOG = ROOT / "summaries" / "papers.html"

RETAINED_SLUGS = {
    "letter_learnable_item_tokenization_summary",
    "cost_contrastive_quantization_semantic_tokenization_summary",
    "actionpiece_contextual_action_tokenization_summary",
    "diger_differentiable_semantic_id_summary",
    "mtgrec_multi_identifier_item_tokenization_summary",
    "etegrec_end_to_end_learnable_item_tokenization_summary",
    "enhancing_embedding_representation_stability_semantic_id_summary",
    "unleashing_native_recommendation_structured_term_identifiers_summary",
    "variable_length_semantic_ids_summary",
    "merge_next_generation_item_indexing_summary",
    "tiger_recommender_systems_with_generative_retrieval_summary",
    "actions_speak_louder_than_words_trillion_parameter_sequential_summary",
    "generative_retrieval_alignment_model_ecommerce_retrieval_summary",
    "semantic_ids_for_recommender_systems_at_snapchat_use_summary",
    "unifying_generative_and_dense_retrieval_summary",
    "crab_codebook_rebalancing_for_bias_mitigation_in_generative_recommendation_summary",
    "generating_long_semantic_ids_in_parallel_for_recommendation_summary",
    "r3_vae_reference_vector_guided_rating_residual_quantization_vae_for_generative_recommendation_summary",
    "orbit_preserving_foundational_language_capabilities_in_genretrieval_via_origin_regulated_mergin_summary",
    "mmq_v2_adaptive_behavior_mining_summary",
    "closing_performance_gap_collaborative_tokenization_efficient_modeling_summary",
    "rq_gmm_residual_quantized_gaussian_mixture_model_summary",
    "order_agnostic_identifier_for_large_language_model_based_summary",
    "learning_multi_aspect_item_palette_summary",
    "pinrec_outcome_conditioned_multi_token_generative_retrieval_summary",
}


@dataclass
class SummaryMeta:
    path: Path
    title: str
    category: str
    slug: str
    catalog_id: str
    paper_url: str | None

    @property
    def href(self) -> str:
        return f"paper_summaries/{self.category}/{self.slug}.html"


def parse_frontmatter(path: Path) -> SummaryMeta:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"Missing frontmatter: {path}")
    _, fm, _ = text.split("---", 2)
    data: dict[str, str] = {}
    for line in fm.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return SummaryMeta(
        path=path,
        title=data["title"],
        category=data["category"],
        slug=data["slug"],
        catalog_id=data["catalogId"],
        paper_url=data.get("paperUrl") or None,
    )


def all_summaries() -> dict[str, SummaryMeta]:
    return {
        meta.slug: meta
        for meta in (parse_frontmatter(path) for path in sorted(CONTENT.glob("**/*.md")))
    }


def find_tag_li(details: Tag) -> Tag | None:
    for li in details.find_all("li", recursive=False):
        strong = li.find("strong")
        if strong and "Теги" in strong.get_text(" ", strip=True):
            return li
    return None


def remove_missing_badge(details: Tag) -> None:
    tag_li = find_tag_li(details)
    if tag_li is None:
        return
    for badge in tag_li.find_all("span", class_="summary-missing"):
        badge.decompose()


def ensure_badge(soup: BeautifulSoup, details: Tag) -> None:
    tag_li = find_tag_li(details)
    if tag_li is None:
        tag_li = soup.new_tag("li")
        strong = soup.new_tag("strong")
        strong.string = "Теги:"
        tag_li.append(strong)
        tag_li.append(" ")
        details.append(tag_li)
    if not tag_li.find("span", class_="summary-missing"):
        tag_li.append(" ")
        badge = soup.new_tag("span", attrs={"class": "badge summary-missing"})
        badge.string = "саммари пропущено"
        tag_li.append(badge)


def remove_article_link(details: Tag) -> None:
    for li in list(details.find_all("li", recursive=False)):
        strong = li.find("strong")
        if strong and "Статья" in strong.get_text(" ", strip=True):
            li.decompose()


def ensure_article_link(soup: BeautifulSoup, details: Tag, url: str | None) -> None:
    if not url:
        return
    for li in details.find_all("li", recursive=False):
        strong = li.find("strong")
        if strong and "Статья" in strong.get_text(" ", strip=True):
            link = li.find("a", href=True)
            if link:
                link["href"] = url
            return
    li = soup.new_tag("li")
    strong = soup.new_tag("strong")
    strong.string = "Статья:"
    li.append(strong)
    li.append(" ")
    link = soup.new_tag("a", attrs={"class": "paper-link", "href": url})
    link.string = "открыть статью"
    li.append(link)
    details.insert(3 if len(details.find_all("li", recursive=False)) >= 3 else len(details.contents), li)


def set_takeaway(soup: BeautifulSoup, details: Tag, lines: list[str]) -> None:
    takeaway_li = None
    for li in details.find_all("li", recursive=False):
        strong = li.find("strong")
        if strong and "Выжимка" in strong.get_text(" ", strip=True):
            takeaway_li = li
            break
    if takeaway_li is None:
        takeaway_li = soup.new_tag("li")
        strong = soup.new_tag("strong")
        strong.string = "Выжимка:"
        takeaway_li.append(strong)
        details.append(takeaway_li)
    for child in list(takeaway_li.children):
        if isinstance(child, Tag) and child.name == "div" and "takeaway" in child.get("class", []):
            child.decompose()
    div = soup.new_tag("div", attrs={"class": "takeaway"})
    for line in lines:
        inner = soup.new_tag("div")
        inner.string = line
        div.append(inner)
    takeaway_li.append(div)


def update_details_basics(
    soup: BeautifulSoup,
    details: Tag,
    *,
    publication_date: str,
    added_date: str,
    affiliations: str,
    tags: list[str],
) -> None:
    for li in details.find_all("li", recursive=False):
        strong = li.find("strong")
        if strong and strong.get_text(" ", strip=True).rstrip(":") in {"Год", "Дата arXiv"}:
            li.decompose()

    desired = {
        "Дата публикации": publication_date,
        "Дата добавления": added_date,
        "Аффилиации": affiliations,
    }
    insertion_positions = {"Дата публикации": 0, "Дата добавления": 1, "Аффилиации": 2}
    for label, value in desired.items():
        target = None
        for li in details.find_all("li", recursive=False):
            strong = li.find("strong")
            if strong and strong.get_text(" ", strip=True).rstrip(":") == label:
                target = li
                break
        if target is None:
            target = soup.new_tag("li")
            strong = soup.new_tag("strong")
            strong.string = f"{label}:"
            target.append(strong)
            details.insert(insertion_positions[label], target)
        for child in list(target.children):
            if not (isinstance(child, Tag) and child.name == "strong"):
                child.extract()
        target.append(" " + value)

    tag_li = find_tag_li(details)
    if tag_li is None:
        tag_li = soup.new_tag("li")
        strong = soup.new_tag("strong")
        strong.string = "Теги:"
        tag_li.append(strong)
        details.insert(3, tag_li)
    for child in list(tag_li.children):
        if not (isinstance(child, Tag) and child.name == "strong"):
            child.extract()
    tag_li.append(" ")
    for tag in tags:
        badge = soup.new_tag("span", attrs={"class": "badge"})
        badge.string = tag
        tag_li.append(badge)
        tag_li.append(" ")


def link_to_slug(href: str) -> str | None:
    match = re.search(r"/([^/]+)\.html$", href)
    return match.group(1) if match else None


def existing_paper_url(details: Tag | None) -> str | None:
    if details is None:
        return None
    link = details.find("a", class_="paper-link", href=True)
    return link["href"] if link else None


def update_counts(soup: BeautifulSoup) -> None:
    for h2 in soup.find_all("h2"):
        count = 0
        node = h2.find_next_sibling()
        while node is not None and not (isinstance(node, Tag) and node.name == "h2"):
            if isinstance(node, Tag) and node.name == "ol":
                count += 1
            node = node.find_next_sibling()
        text = h2.get_text(" ", strip=True)
        text = re.sub(r"\s*\(\d+\)$", "", text)
        h2.string = f"{text} ({count})"


def retained_catalog_snippet(meta: SummaryMeta) -> list[str]:
    snippets = {
        "mmq_v2_adaptive_behavior_mining_summary": [
            "MMQ-v2 / ADC-SID показывает, как добавлять collaborative signal в semantic IDs без загрязнения long-tail item representations шумными behavioral embeddings.",
            "Метод сочетает shared/specific experts, adaptive behavior-content alignment и dynamic behavioral weighting; online A/B сообщает +3.50% revenue в retrieval scenario.",
        ],
        "closing_performance_gap_collaborative_tokenization_efficient_modeling_summary": [
            "COSETTE добавляет collaborative contrastive signal в tokenizer, а MARIUS разделяет timeline modeling и item-code decoding, чтобы закрыть gap с сильным SASRec++ baseline.",
            "Главный takeaway: generative recommender должен конкурировать с хорошо настроенным ID baseline, а не с ослабленной реализацией.",
        ],
        "learning_multi_aspect_item_palette_summary": [
            "LAMIA заменяет one-vector RQ-VAE tokenization на multi-aspect item palette: item получает несколько параллельных semantic aspects вместо одной residual hierarchy.",
            "Метод особенно важен для доменов, где один item релевантен пользователю по разным независимым причинам.",
        ],
        "rq_gmm_residual_quantized_gaussian_mixture_model_summary": [
            "RQ-GMM использует Gaussian Mixture Model внутри residual quantization, чтобы строить multimodal semantic IDs для CTR prediction.",
            "Это пример более безопасного внедрения semantic IDs: не заменять retrieval на generator, а добавить discrete multimodal features в ranker/CTR stack.",
        ],
    }
    return snippets.get(meta.slug, [f"Подробное markdown-саммари для статьи: {meta.title}."])


def build_entry(
    soup: BeautifulSoup,
    meta: SummaryMeta,
    start: int,
    *,
    publication_date: str,
    added_date: str,
    affiliations: str,
    tags: list[str],
) -> tuple[Tag, Tag]:
    ol = soup.new_tag("ol", attrs={"id": meta.catalog_id, "start": str(start)})
    li = soup.new_tag("li")
    strong = soup.new_tag("strong")
    a = soup.new_tag("a", href=meta.href)
    a.string = meta.title
    strong.append(a)
    li.append(strong)
    ol.append(li)

    ul = soup.new_tag("ul")
    update_details_basics(
        soup,
        ul,
        publication_date=publication_date,
        added_date=added_date,
        affiliations=affiliations,
        tags=tags,
    )
    ensure_article_link(soup, ul, meta.paper_url)
    set_takeaway(soup, ul, retained_catalog_snippet(meta))
    return ol, ul


def insert_new_semantic_entries(soup: BeautifulSoup, metas: dict[str, SummaryMeta]) -> None:
    semantic_h2 = soup.find("h2", id="category-semantic-ids")
    if semantic_h2 is None:
        raise RuntimeError("Cannot find semantic category")

    existing_ids = {ol.get("id") for ol in soup.find_all("ol")}
    to_add = [
        (
            "closing_performance_gap_collaborative_tokenization_efficient_modeling_summary",
            "2025-08-12",
            "2026-06-19",
            "CRITEO AI Lab; LIGM, École Nationale des Ponts et Chaussées",
            ["generative recommendation", "collaborative tokenization", "efficient modeling"],
        ),
        (
            "mmq_v2_adaptive_behavior_mining_summary",
            "2025-10-29",
            "2026-06-19",
            "Alibaba Group; academic collaborators",
            ["semantic IDs", "behavior-content alignment", "long-tail"],
        ),
    ]

    last_ol = None
    node = semantic_h2.find_next_sibling()
    while node is not None and not (isinstance(node, Tag) and node.name == "h2"):
        if isinstance(node, Tag) and node.name == "ol":
            last_ol = node
        node = node.find_next_sibling()
    if last_ol is None:
        raise RuntimeError("Semantic category has no entries")
    start = max(int(ol.get("start", "1")) for ol in soup.find_all("ol") if ol.find_previous("h2") == semantic_h2)

    insertion_anchor = last_ol.find_next_sibling("ul")
    for slug, publication_date, added_date, affiliations, tags in to_add:
        meta = metas[slug]
        if meta.catalog_id in existing_ids:
            continue
        start += 1
        ol, ul = build_entry(
            soup,
            meta,
            start,
            publication_date=publication_date,
            added_date=added_date,
            affiliations=affiliations,
            tags=tags,
        )
        insertion_anchor.insert_after(ul)
        insertion_anchor.insert_after(NavigableString("\n"))
        insertion_anchor.insert_after(ol)
        insertion_anchor.insert_after(NavigableString("\n"))
        insertion_anchor = ul


def main() -> int:
    metas = all_summaries()
    retained = {slug: metas[slug] for slug in RETAINED_SLUGS}
    soup = BeautifulSoup(CATALOG.read_text(encoding="utf-8"), "html.parser")

    # Replace old STORE v1 entry with the current LAMIA arXiv version.
    store_ol = soup.find("ol", id="paper-store_streamlining_semantic_tokenization_and_generative_recommendation_with_summary")
    if store_ol is not None:
        lamia = retained["learning_multi_aspect_item_palette_summary"]
        store_ol["id"] = lamia.catalog_id
        link = store_ol.find("a", href=True)
        if link:
            link["href"] = lamia.href
            link.string = lamia.title
        details = store_ol.find_next_sibling("ul")
        if details:
            update_details_basics(
                soup,
                details,
                publication_date="2026-06-09",
                added_date="2026-05-15",
                affiliations="Hong Kong Polytechnic University; Huawei Noah's Ark Lab; Zhejiang University",
                tags=["semantic tokenization", "multi-aspect item palette", "generative recommendation"],
            )
            ensure_article_link(soup, details, lamia.paper_url)
            remove_missing_badge(details)
            set_takeaway(soup, details, retained_catalog_snippet(lamia))

    insert_new_semantic_entries(soup, retained)

    entries = get_entries(soup)
    cache = load_cache()
    arxiv_ids = sorted(
        {
            extract_arxiv_id((metas.get(link_to_slug(entry.href) or "") or SummaryMeta(Path(), "", "", "", "", None)).paper_url or entry.href)
            for entry in entries
            if extract_arxiv_id((metas.get(link_to_slug(entry.href) or "") or SummaryMeta(Path(), "", "", "", "", None)).paper_url or entry.href)
        }
    )
    arxiv_cache = fetch_arxiv_abstracts([value for value in arxiv_ids if value])

    missing_count = 0
    abstract_missing = 0
    retained_count = 0

    for entry in get_entries(soup):
        link = entry.ol.find("a", href=True)
        if link is None or entry.details is None:
            continue
        slug = link_to_slug(link["href"])
        retained_meta = metas.get(slug or "")
        if retained_meta and retained_meta.slug in RETAINED_SLUGS:
            link["href"] = retained_meta.href
            link.string = retained_meta.title if retained_meta.slug in {
                "learning_multi_aspect_item_palette_summary",
                "rq_gmm_residual_quantized_gaussian_mixture_model_summary",
            } else link.get_text(" ", strip=True)
            entry.ol["id"] = retained_meta.catalog_id
            remove_missing_badge(entry.details)
            ensure_article_link(soup, entry.details, retained_meta.paper_url)
            retained_count += 1
            if retained_meta.slug in {
                "learning_multi_aspect_item_palette_summary",
                "rq_gmm_residual_quantized_gaussian_mixture_model_summary",
                "closing_performance_gap_collaborative_tokenization_efficient_modeling_summary",
                "mmq_v2_adaptive_behavior_mining_summary",
            }:
                set_takeaway(soup, entry.details, retained_catalog_snippet(retained_meta))
            continue

        # RQ-GMM exists as an external entry in the catalog: make it local.
        if "RQ-GMM: Residual Quantized Gaussian Mixture Model" in link.get_text(" ", strip=True):
            rq = retained["rq_gmm_residual_quantized_gaussian_mixture_model_summary"]
            entry.ol["id"] = rq.catalog_id
            link["href"] = rq.href
            link.string = rq.title
            remove_missing_badge(entry.details)
            ensure_article_link(soup, entry.details, rq.paper_url)
            set_takeaway(soup, entry.details, retained_catalog_snippet(rq))
            retained_count += 1
            continue

        paper_url = None
        if slug and slug in metas:
            paper_url = metas[slug].paper_url
        paper_url = paper_url or existing_paper_url(entry.details) or link["href"]
        if not paper_url or paper_url.startswith("paper_summaries/"):
            paper_url = link.get_text(" ", strip=True)

        link["href"] = paper_url
        ensure_badge(soup, entry.details)
        remove_article_link(entry.details)
        abstract = lookup_abstract(CatalogEntry(entry.title, paper_url, entry.ol, entry.details), arxiv_cache, cache)
        if not abstract:
            abstract_missing += 1
        set_takeaway_to_abstract(soup, CatalogEntry(entry.title, paper_url, entry.ol, entry.details), abstract)
        missing_count += 1

    add_css_for_missing_badge(soup)
    update_counts(soup)
    CATALOG.write_text(str(soup), encoding="utf-8")
    save_cache(cache)

    deleted = 0
    for slug, meta in metas.items():
        if slug not in RETAINED_SLUGS:
            meta.path.unlink()
            deleted += 1

    print(f"Retained local summaries: {retained_count}")
    print(f"Marked abstract-only entries: {missing_count}; unresolved abstracts: {abstract_missing}")
    print(f"Deleted markdown summaries: {deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
