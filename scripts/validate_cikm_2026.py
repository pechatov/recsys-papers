#!/usr/bin/env python3
"""Validate the approved CIKM selection, reviews and optionally built pages."""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
HEADINGS = ['Коротко', 'Задача и мотивация', 'Как работает метод', 'Эксперименты и результаты',
            'Ограничения и практический смысл', 'Вопросы авторам', 'Источники']

class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = []
        self.h1_count = 0
        self.paper_ids = []
        self.visible_paper_ids = []
        self.selected_basis = None
        self.in_basis_select = False
    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if 'id' in data:
            self.ids.append(data['id'])
        if 'data-paper' in data:
            self.paper_ids.append(data.get('id'))
            if 'hidden' not in data:
                self.visible_paper_ids.append(data.get('id'))
        if tag == 'select' and data.get('id') == 'basis':
            self.in_basis_select = True
        if tag == 'option' and self.in_basis_select and 'selected' in data:
            self.selected_basis = data.get('value')
        if tag == 'h1':
            self.h1_count += 1
        if tag == 'a' and 'href' in data:
            self.links.append(data['href'])
    def handle_endtag(self, tag):
        if tag == 'select':
            self.in_basis_select = False

def main():
    errors = []
    selection = json.loads((ROOT / 'docs/cikm-2026-selection.json').read_text())
    expected = {p['selectionId']: p for p in selection['papers'] if p['scope'] == 'main'}
    metadata = json.loads((ROOT / 'src/data/cikm-2026-papers.json').read_text())
    actual = {p['selectionId']: p for p in metadata}
    files = {p.stem: p for p in (ROOT / 'src/content/cikm_2026').glob('*.md')}
    if len(expected) != 99 or len(metadata) != 99 or set(expected) != set(actual) or set(files) != set(expected):
        errors.append('Expected exactly the approved 99 metadata records and review files')
    questions_total = 0
    basis_count = Counter()
    full_text_ids = set()
    for sid, path in files.items():
        text = path.read_text()
        if not text.startswith('---\n') or text.count('\n---\n') < 1:
            errors.append(f'{sid}: missing frontmatter')
            continue
        front, body = text[4:].split('\n---\n', 1)
        def field(key):
            match = re.search(rf'^{key}:\s*(.+)$', front, re.M)
            value = match[1].strip() if match else ''
            if value.startswith('"'):
                try:
                    return json.loads(value)
                except ValueError:
                    return value.strip('"')
            return value.strip("'")
        if field('selectionId') != sid or field('title') != expected[sid]['title']:
            errors.append(f'{sid}: identity differs from approved selection')
        basis = field('basis')
        basis_count[basis] += 1
        if basis == 'full_text':
            full_text_ids.add(sid)
        if basis not in {'abstract', 'full_text'}:
            errors.append(f'{sid}: invalid source basis')
        if basis == 'full_text' and 'cikm2026.diag.uniroma1.it/cikm-program' in field('sourceUrl'):
            errors.append(f'{sid}: program alone is not full text')
        if not field('takeaway') or not field('sourceLabel') or not field('sourceUrl').startswith('https://'):
            errors.append(f'{sid}: missing source metadata or takeaway')
        headings = re.findall(r'^## (.+?)\s*$', body, re.M)
        if headings != HEADINGS:
            errors.append(f'{sid}: unexpected section structure: {headings}')
        match = re.search(r'^## Вопросы авторам\s*\n(.*?)(?=^## |\Z)', body, re.M | re.S)
        questions = re.findall(r'^\s*(?:\d+[.)]|[-*])\s+(.+)', match[1], re.M) if match else []
        questions_total += len(questions)
        if len(questions) != 4 or len(set(questions)) != 4:
            errors.append(f'{sid}: expected 4 distinct author questions, found {len(questions)}')
        if len(body.split()) < 250:
            errors.append(f'{sid}: review too short')
        if re.search(r'\b(?:TODO|TBD|FIXME)\b', body):
            errors.append(f'{sid}: placeholder remains')
        if expected[sid]['sessionUrl'] not in body and selection['sourceUrl'] not in body:
            errors.append(f'{sid}: official program citation missing')
    if '--built' in sys.argv:
        dist = ROOT / 'dist'
        paths = [dist / 'cikm-2026.html'] + [dist / 'cikm-2026' / f'{sid.lower()}.html' for sid in expected]
        base = next((arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--base=')), '/')
        for path in paths:
            if not path.exists():
                errors.append(f'Missing built page {path}')
                continue
            parser = Page()
            parser.feed(path.read_text())
            if path == dist / 'cikm-2026.html' and (len(parser.paper_ids) != 99 or set(parser.paper_ids) != set(expected)):
                errors.append('Conference index must display exactly the 99 approved paper cards')
            if path == dist / 'cikm-2026.html':
                if parser.selected_basis != 'full_text' or set(parser.visible_paper_ids) != full_text_ids:
                    errors.append('Conference index must initially show every full-text review and hide abstract-only overviews')
            if parser.h1_count != 1:
                errors.append(f'{path.name}: expected one page heading')
            if len(parser.ids) != len(set(parser.ids)):
                errors.append(f'{path.name}: duplicate HTML IDs')
            for link in parser.links:
                url = urlsplit(link)
                if url.scheme or url.netloc:
                    continue
                link_path = unquote(url.path)
                if not link_path:
                    target = path
                elif link_path.startswith('/'):
                    if not link_path.startswith(base):
                        errors.append(f'{path.name}: link escapes deployment base: {link}')
                        continue
                    target = dist / link_path[len(base):]
                else:
                    target = path.parent / link_path
                if not target.exists():
                    errors.append(f'{path.name}: broken local link {link}')
                    continue
                if url.fragment and target.suffix == '.html':
                    target_page = Page()
                    target_page.feed(target.read_text())
                    if unquote(url.fragment) not in target_page.ids:
                        errors.append(f'{path.name}: broken anchor {link}')
    for error in errors:
        print('ERROR', error)
    if errors:
        return 1
    print(f'Validated 99 CIKM reviews, {questions_total} author questions; source basis: {dict(basis_count)}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
