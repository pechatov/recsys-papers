#!/usr/bin/env python3
"""Extract the approved papers' published appearances from the CIKM program.

Run with a downloaded program snapshot, for example:
  PYTHONPATH=.cache/python-deps python3 scripts/extract_cikm_2026_schedule.py \
    --program .cache/cikm-2026/program-current.html

The program publishes session intervals, not individual oral start times. Listing
order and the separate presentation-duration guidelines are never used to invent
individual timestamps. Research-paper type is not used to infer presentation type.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
PROGRAM_URL = 'https://cikm2026.diag.uniroma1.it/cikm-program/program.html'
DAYS = {
    'mon': ('2026-11-09', 'Понедельник', 'Monday'),
    'tue': ('2026-11-10', 'Вторник', 'Tuesday'),
    'wed': ('2026-11-11', 'Среда', 'Wednesday'),
}
TRACKS = {
    'FULL': 'Full Research', 'SHORT': 'Short Research',
    'APPLIED': 'Applied Research', 'RESOURCE': 'Resources',
}
VENUE = 'Auditorium Parco della Musica Ennio Morricone'
INTERVAL = re.compile(r'(\d{2}:\d{2})\s*[–−-]\s*(\d{2}:\d{2})')


def normalized(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def doi(url: str) -> str:
    parts = urlsplit(url)
    if parts.hostname not in {'doi.org', 'www.doi.org'}:
        raise ValueError(f'Expected program DOI link, got {url}')
    return parts.path.lstrip('/').casefold()


def extract(snapshot: bytes, metadata: list[dict], checked_at: str) -> dict:
    date.fromisoformat(checked_at)
    soup = BeautifulSoup(snapshot, 'html.parser')
    assert len(metadata) == 99, 'Expected the approved 99-paper collection'
    by_program_id = {p['programId']: p for p in metadata}
    assert len(by_program_id) == len(metadata), 'Duplicate approved program IDs'
    appearances: dict[str, list[dict]] = {p['selectionId']: [] for p in metadata}
    used_locations: set[tuple[str, str]] = set()
    for node in soup.select('.paper[data-pid]'):
        paper = by_program_id.get(node['data-pid'])
        if paper is None:
            continue
        session = node.find_parent(class_='sessblock')
        assert session is not None
        session_id = session['id']
        assert session_id.startswith(('s-', 'p-')), f'Unexpected presentation for {paper["selectionId"]}'
        location = (paper['selectionId'], session_id)
        assert location not in used_locations, f'Duplicate appearance {location}'
        used_locations.add(location)
        title = node.select_one('a.ptitle')
        assert title is not None
        assert normalized(title.get_text(' ', strip=True)) == normalized(paper['title']), paper['selectionId']
        assert doi(title['href']) == doi(paper['url']), f'DOI differs: {paper["selectionId"]}'
        head = session.select_one('.shead')
        assert head is not None
        header = head.get_text(' ', strip=True)
        day = session['data-day']
        event_date, label, english_day = DAYS[day]
        assert english_day in header, f'Day mismatch in {session_id}'
        interval = INTERVAL.search(header)
        assert interval is not None
        start, end = interval.groups()
        assert start < end
        is_poster = session_id.startswith('p-')
        badge = node.select_one('.badge')
        assert badge is not None
        badge_code = badge.get_text(strip=True)
        assert badge_code == paper['track'], f'Paper track differs: {paper["selectionId"]}'
        track = TRACKS[badge_code]
        sname = head.select_one('.sname').get_text(' ', strip=True)
        board = node.select_one('.board')
        if is_poster:
            assert board is not None, f'Poster board missing for {paper["selectionId"]}'
            slot = session.select_one('.sloth')
            assert slot and 'Poster Area' in slot.get_text(), session_id
            room = 'Poster Area'
            code = 'P' + session_id.split('-')[1]
        else:
            assert board is None
            chips = [x.get_text(' ', strip=True) for x in head.select('.mchip')]
            rooms = [x for x in chips if x not in TRACKS.values() and not INTERVAL.search(x) and not re.search(r'\btalks?\b', x)]
            assert len(rooms) == 1, f'Cannot identify oral room in {session_id}: {rooms}'
            room = rooms[0]
            code = session_id.removeprefix('s-')
        record = {
            'kind': 'poster' if is_poster else 'oral',
            'date': event_date, 'dayLabel': label, 'start': start, 'end': end,
            'timePrecision': 'session', 'individualStart': None, 'individualEnd': None,
            'venue': VENUE, 'room': room, 'sessionCode': code,
            'sessionTitle': sname.split(' · ', 1)[1] if not is_poster else sname,
            'track': track, 'sourceUrl': PROGRAM_URL + '#' + session_id,
            'board': board.get_text(strip=True) if board else None,
        }
        if not is_poster:
            record['orderInSession'] = next(i for i, p in enumerate(session.select('.paper'), 1) if p is node)
        appearances[paper['selectionId']].append(record)
    rows = []
    for paper in sorted(metadata, key=lambda p: p['selectionId']):
        entries = sorted(appearances[paper['selectionId']], key=lambda e: (e['date'], e['start'], e['kind']))
        assert entries, f'Missing program match: {paper["selectionId"]}'
        kinds = {entry['kind'] for entry in entries}
        presentation = 'talk_and_poster' if kinds == {'oral', 'poster'} else 'oral_only' if kinds == {'oral'} else 'poster_only'
        note = {
            'talk_and_poster': 'В программе указаны доклад и постер.',
            'oral_only': 'В программе указан доклад; отдельный постер не указан.',
            'poster_only': 'В программе указан только постер; доклад не указан.',
        }[presentation]
        rows.append({'selectionId': paper['selectionId'], 'presentation': presentation, 'presentationNote': note, 'appearances': entries})
    counts = Counter(paper['presentation'] for paper in rows)
    return {
        'checkedAt': checked_at, 'timezone': 'Europe/Rome', 'sourceUrl': PROGRAM_URL,
        'programSha256': hashlib.sha256(snapshot).hexdigest(),
        'guidelinesUrls': [
            'https://cikm2026.diag.uniroma1.it/oral-presentation-guidelines/',
            'https://cikm2026.diag.uniroma1.it/poster-instructions-and-printing/',
        ],
        'notes': [
            'Время местное для Рима (Europe/Rome; CET, UTC+1 на даты конференции).',
            'Для докладов опубликованы интервалы сессий. Индивидуальное время начала доклада не опубликовано; порядок выступлений не преобразуется в точные часы.',
            'Формат определяется по фактической записи в устной или постерной сессии, а не по типу FULL/APPLIED/SHORT.',
            f"Из {len(rows)} выбранных статей в программе указаны: только доклад — {counts['oral_only']}, только постер — {counts['poster_only']}, оба формата — {counts['talk_and_poster']}. Отсутствие отдельной постерной записи не доказывает запрета на постер.",
            'Для постеров сохранён номер стенда, привязанный к конкретному дню: номера могут повторяться в разные дни.',
        ],
        'papers': rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'src/data/cikm-2026-schedule.json')
    parser.add_argument('--checked-at', default='2026-11-09')
    args = parser.parse_args()
    metadata = json.loads((ROOT / 'src/data/cikm-2026-papers.json').read_text())
    data = extract(args.program.read_bytes(), metadata, args.checked_at)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print('Verified schedule:', len(data['papers']), 'papers;', dict(Counter(p['presentation'] for p in data['papers'])))


if __name__ == '__main__':
    main()
