#!/usr/bin/env python3
"""Validate the approved CIKM selection, reviews and optionally built pages."""
from __future__ import annotations
import json
import csv
import hashlib
import re
import sys
from collections import Counter
from datetime import date
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
        self.paper_data = {}
        self.visible_paper_ids = []
        self.selected_basis = None
        self.in_basis_select = False
    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if 'id' in data:
            self.ids.append(data['id'])
        if 'data-paper' in data:
            self.paper_ids.append(data.get('id'))
            self.paper_data[data.get('id')] = data
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
    with (ROOT / 'docs/cikm-2026-fulltext-access.csv').open() as stream:
        sources = list(csv.DictReader(stream))
    if len(sources) != len(expected) or {row['selectionId'] for row in sources} != set(expected):
        errors.append('Source report must cover exactly the approved selection')
    for row in sources:
        path = files.get(row['selectionId'])
        if path is None:
            continue
        front = path.read_text()[4:].split('\n---\n', 1)[0]
        for report_key, content_key in [('reviewBasis', 'basis'), ('reviewSourceUrl', 'sourceUrl'), ('reviewSourceLabel', 'sourceLabel')]:
            match = re.search(rf'^{content_key}:\s*(.+)$', front, re.M)
            if match is None or json.loads(match[1]) != row[report_key]:
                errors.append(f'{row["selectionId"]}: source report differs from review {content_key}')
    with (ROOT / 'docs/cikm-2026-missing-fulltexts.csv').open() as stream:
        missing = list(csv.DictReader(stream))
    if len(missing) != len(expected) - len(full_text_ids) or {row['selectionId'] for row in missing} != set(expected) - full_text_ids:
        errors.append('Missing-fulltext report differs from review coverage')
    audit = json.loads((ROOT / 'docs/cikm-2026-review-audit.json').read_text())
    coverage = audit['coverage']
    if (len(coverage) != len(full_text_ids) or {row['selectionId'] for row in coverage} != full_text_ids
            or audit['fullTextReviewsChecked'] != len(full_text_ids)
            or audit['remainingAbstractOverviews'] != len(expected) - len(full_text_ids)):
        errors.append('Content audit coverage differs from full-text reviews')
    for row in coverage:
        path = files.get(row['selectionId'])
        if path and row['reviewSha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            errors.append(f'{row["selectionId"]}: content changed after its recorded audit')
    ratings = json.loads((ROOT / 'src/data/cikm-2026-ratings.json').read_text())
    rubric = json.loads((ROOT / 'src/data/cikm-2026-rating-rubric.json').read_text())
    ratings_by_id = {row['selectionId']: row for row in ratings}
    schedule = json.loads((ROOT / 'src/data/cikm-2026-schedule.json').read_text())
    visits = {row['selectionId']: row for row in schedule['papers']}
    if len(ratings) != 99 or set(ratings_by_id) != set(expected):
        errors.append('Ratings must cover exactly the 99 approved papers')
    if len(schedule['papers']) != 99 or set(visits) != set(expected) or schedule['timezone'] != 'Europe/Rome':
        errors.append('Schedule must cover exactly the 99 approved papers in Europe/Rome')
    for row in ratings:
        scores = [row[key] for key in ('interest', 'evidence', 'importance')]
        if any(type(score) is not int or not rubric['scale']['min'] <= score <= rubric['scale']['max'] for score in scores):
            errors.append(f'{row["selectionId"]}: ratings must be integers from 1 to 5')
        total = sum(scores)
        group = ('strong' if total >= rubric['strongTotal'] and row['evidence'] >= rubric['strongMinEvidence']
                 else 'weaker' if total <= rubric['weakerTotal'] or (row['evidence'] <= rubric['weakerMaxEvidence'] and row['interest'] <= rubric['weakerMaxInterest'])
                 else 'solid')
        if row['group'] != group or not row['reason'].strip():
            errors.append(f'{row["selectionId"]}: rating group or rationale differs from rubric')
    boards = set()
    for sid, visit in visits.items():
        kinds = {item['kind'] for item in visit['appearances']}
        presentation = {frozenset({'oral'}): 'oral_only', frozenset({'poster'}): 'poster_only', frozenset({'oral', 'poster'}): 'talk_and_poster'}.get(frozenset(kinds))
        if visit['presentation'] != presentation:
            errors.append(f'{sid}: presentation format differs from actual appearances')
        for item in visit['appearances']:
            if item['date'] not in {'2026-11-09', '2026-11-10', '2026-11-11'} or date.fromisoformat(item['date']).weekday() not in {0, 1, 2}:
                errors.append(f'{sid}: invalid conference day')
            if not all(re.fullmatch(r'\d{2}:\d{2}', item[key]) for key in ('start', 'end')) or item['start'] >= item['end']:
                errors.append(f'{sid}: invalid session interval')
            if not all(item.get(key) for key in ('room', 'track', 'sessionCode', 'sessionTitle')) or item['sourceUrl'] != expected[sid]['sessionUrl']:
                errors.append(f'{sid}: incomplete or mismatched program session')
            if item['timePrecision'] != 'session' or item.get('individualStart') is not None or item.get('individualEnd') is not None:
                errors.append(f'{sid}: individual times are not published in this program snapshot')
            if item['kind'] == 'poster':
                key = (item['date'], item['board'])
                if not item['board'] or key in boards:
                    errors.append(f'{sid}: missing or duplicate poster board within day')
                boards.add(key)
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
                for sid, attrs in parser.paper_data.items():
                    if (sid in ratings_by_id and attrs.get('data-grade') != ratings_by_id[sid]['group']
                            or sid in visits and set(attrs.get('data-days', '').split()) != {a['date'] for a in visits[sid]['appearances']}):
                        errors.append(f'{sid}: rendered rating/day differs from visit data')
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
