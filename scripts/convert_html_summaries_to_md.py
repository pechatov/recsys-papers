#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import re
import textwrap

from bs4 import BeautifulSoup, NavigableString, Tag


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "summaries" / "paper_summaries"
TARGET = ROOT / "src" / "content" / "paper_summaries"


SKIP_CLASSES = {"nav"}
SKIP_TAGS = {"script", "style", "button"}


@dataclass
class SummaryMeta:
    title: str
    paper_url: str | None
    category: str
    slug: str
    catalog_id: str
    source_html: str


def frontmatter_value(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_node(value: str) -> str:
    if not value:
        return ""
    prefix = " " if value[0].isspace() else ""
    suffix = " " if value[-1].isspace() else ""
    normalized = re.sub(r"\s+", " ", value).strip()
    return f"{prefix}{normalized}{suffix}" if normalized else (" " if prefix or suffix else "")


def inline(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        return text_node(str(node))
    if not isinstance(node, Tag):
        return ""

    if node.name in SKIP_TAGS:
        return ""

    text = "".join(inline(child) for child in node.children)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([])}])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if node.name in {"strong", "b"}:
        return f"**{text}**" if text else ""
    if node.name in {"em", "i"}:
        return f"*{text}*" if text else ""
    if node.name == "code":
        return f"`{text}`" if text else ""
    if node.name == "a":
        href = node.get("href", "")
        return f"[{text}]({href})" if href and text else text
    if node.name == "br":
        return "  \n"
    if node.name == "img":
        src = node.get("src", "")
        alt = node.get("alt", "")
        if not src:
            return ""
        attrs = f' src="{escape(src, quote=True)}"'
        if alt:
            attrs += f' alt="{escape(alt, quote=True)}"'
        return f"<img{attrs}>"
    return text


def table_html(node: Tag) -> str:
    html = str(node)
    return f'<div class="table-scroll">\n{html}\n</div>\n\n'


def figure_html(node: Tag) -> str:
    img = node.find("img")
    if img is None:
        return block_children(node)
    src = img.get("src", "")
    alt = img.get("alt", "")
    caption = ""
    figcaption = node.find("figcaption")
    if figcaption is not None:
        caption = clean_text(figcaption.decode_contents())
    attrs = f' src="{escape(src, quote=True)}"'
    if alt:
        attrs += f' alt="{escape(alt, quote=True)}"'
    result = [f'<figure class="paper-figure">', f'  <img{attrs}>']
    if caption:
        result.append(f"  <figcaption>{caption}</figcaption>")
    result.append("</figure>")
    return "\n".join(result) + "\n\n"


def list_block(node: Tag, ordered: bool) -> str:
    lines: list[str] = []
    for item in node.find_all("li", recursive=False):
        item_parts: list[str] = []
        nested_blocks: list[str] = []
        for child in item.children:
            if isinstance(child, Tag) and child.name in {"ul", "ol"}:
                nested_blocks.append(block(child).strip())
            else:
                rendered = inline(child) if not isinstance(child, Tag) or child.name not in BLOCK_TAGS else block(child).strip()
                if rendered:
                    item_parts.append(rendered)
        marker = "1." if ordered else "-"
        item_text = clean_text(" ".join(item_parts))
        lines.append(f"{marker} {item_text}".rstrip())
        for nested in nested_blocks:
            if nested:
                lines.extend("  " + line if line else "" for line in nested.splitlines())
    return "\n".join(lines) + "\n\n"


BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "details",
    "div",
    "dl",
    "figure",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "ul",
}


def block_children(node: Tag) -> str:
    return "".join(block(child) for child in node.children)


def block(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        value = clean_text(str(node))
        return value + "\n\n" if value else ""
    if not isinstance(node, Tag):
        return ""
    if node.name in SKIP_TAGS:
        return ""
    if any(cls in SKIP_CLASSES for cls in node.get("class", [])):
        return ""

    if node.name and re.fullmatch(r"h[1-6]", node.name):
        level = int(node.name[1])
        title = inline(node)
        return f"\n{'#' * level} {title}\n\n" if title else ""
    if node.name == "p":
        text = inline(node)
        return text + "\n\n" if text else ""
    if node.name == "blockquote":
        inner = block_children(node).strip()
        if not inner:
            return ""
        return "\n".join("> " + line if line else ">" for line in inner.splitlines()) + "\n\n"
    if node.name == "pre":
        text = node.get_text("\n").strip("\n")
        return f"```\n{text}\n```\n\n" if text else ""
    if node.name == "ul":
        return list_block(node, ordered=False)
    if node.name == "ol":
        return list_block(node, ordered=True)
    if node.name == "figure":
        return figure_html(node)
    if node.name == "table":
        return table_html(node)
    if node.name == "hr":
        return "\n---\n\n"
    if node.name in {"div", "section", "article", "main"}:
        return block_children(node)
    if node.name == "img":
        return inline(node) + "\n\n"

    text = inline(node)
    return text + "\n\n" if text else ""


def extract_meta(path: Path, soup: BeautifulSoup) -> SummaryMeta:
    main = soup.find("main") or soup
    h1 = main.find("h1")
    title = clean_text(h1.get_text(" ", strip=True)) if h1 else clean_text((soup.title or "").get_text(" ", strip=True))
    paper_url = None
    if h1:
        link = h1.find("a")
        if link and link.get("href"):
            paper_url = link["href"]
    nav = main.find("a", href=re.compile(r"#paper-"))
    catalog_id = ""
    if nav and nav.get("href"):
        catalog_id = nav["href"].split("#", 1)[1]
    slug = path.stem
    if not catalog_id:
        catalog_id = f"paper-{slug}"
    return SummaryMeta(
        title=title,
        paper_url=paper_url,
        category=path.parent.name,
        slug=slug,
        catalog_id=catalog_id,
        source_html=str(path.relative_to(ROOT)),
    )


def convert_file(path: Path) -> tuple[Path, str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    meta = extract_meta(path, soup)
    main = soup.find("main") or soup

    for tag in main.find_all(SKIP_TAGS):
        tag.decompose()
    for tag in main.find_all(class_=lambda value: value and any(cls in SKIP_CLASSES for cls in str(value).split())):
        tag.decompose()
    for link in list(main.find_all("a", href=lambda value: value and "papers.html#paper-" in value)):
        parent = link.find_parent(["p", "div"])
        if parent is not None:
            parent.decompose()
        else:
            link.decompose()
    first_h1 = main.find("h1")
    if first_h1:
        first_h1.decompose()

    body = block_children(main).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)

    frontmatter = [
        "---",
        f"title: {frontmatter_value(meta.title)}",
        f"category: {frontmatter_value(meta.category)}",
        f"slug: {frontmatter_value(meta.slug)}",
        f"catalogId: {frontmatter_value(meta.catalog_id)}",
        f"sourceHtml: {frontmatter_value(meta.source_html)}",
        "generatedFromHtml: true",
    ]
    if meta.paper_url:
        frontmatter.append(f"paperUrl: {frontmatter_value(meta.paper_url)}")
    frontmatter.append("---")

    content = "\n".join(frontmatter) + "\n\n" + textwrap.dedent(body).strip() + "\n"
    target = TARGET / meta.category / f"{meta.slug}.md"
    return target, content


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source directory: {SOURCE}")
    converted = 0
    for path in sorted(SOURCE.glob("**/*.html")):
        target, content = convert_file(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        converted += 1
    print(f"Converted {converted} HTML summaries into {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
