#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'summaries' / 'semantic_ids_manifest.json'
PAPERS = ROOT / 'summaries' / 'papers.html'


def ensure_summary_nav() -> None:
    items = json.loads(MANIFEST.read_text(encoding='utf-8'))
    by_path = {str(ROOT / it['summary_path']): it for it in items}
    for file_s, it in by_path.items():
        path = Path(file_s)
        if not path.exists():
            continue
        html = path.read_text(encoding='utf-8')
        title = it['title']
        url = it['url']
        linked_h1 = f'<h1><a href="{escape(url, quote=True)}">{escape(title)}</a></h1>'
        html = re.sub(r'<h1>.*?</h1>', linked_h1, html, count=1, flags=re.S)
        nav = '<p class="nav"><a href="../../papers.html">← К общему списку статей</a></p>'
        if 'К общему списку статей' not in html:
            html = html.replace(linked_h1, linked_h1 + '\n' + nav, 1)
        if '.nav {' not in html:
            html = html.replace('.industry { display:inline-block;', '.nav { margin-top: -.25rem; }\n.nav a { color: var(--link); }\n.industry { display:inline-block;', 1)
        path.write_text(html, encoding='utf-8')


def ensure_papers_links() -> None:
    items = json.loads(MANIFEST.read_text(encoding='utf-8'))
    html = PAPERS.read_text(encoding='utf-8')
    for it in items:
        href = it['summary_href']
        if href in html:
            continue
        title_pat = re.escape(f'<ol start="{it["start"]}"><li><strong><a href="{it["url"]}">{it["title"]}</a></strong></li></ol>')
        m = re.search(title_pat + r'\s*<ul>\s*', html)
        if not m:
            continue
        insert = f'<li><strong>Саммари:</strong> <a href="{escape(href, quote=True)}">подробное саммари</a></li>\n'
        # place after affiliations when possible inside this ul
        start = m.end()
        next_ul_end = html.find('</ul>', start)
        aff = re.search(r'(<li><strong>Аффилиации:</strong>.*?</li>\n)', html[start:next_ul_end], flags=re.S)
        if aff:
            pos = start + aff.end()
            html = html[:pos] + insert + html[pos:]
        else:
            html = html[:start] + insert + html[start:]
    PAPERS.write_text(html, encoding='utf-8')


if __name__ == '__main__':
    ensure_summary_nav()
    ensure_papers_links()
