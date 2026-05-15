#!/usr/bin/env python3
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

class P(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.imgs=[]
    def handle_starttag(self, tag, attrs):
        d=dict(attrs)
        if tag == 'a' and 'href' in d: self.links.append(d['href'])
        if tag == 'img' and 'src' in d: self.imgs.append(d['src'])

def main() -> int:
    missing=[]
    for f in ROOT.glob('summaries/**/*.html'):
        p=P(); p.feed(f.read_text(encoding='utf-8'))
        for kind, vals in [('img', p.imgs), ('href', p.links)]:
            for v in vals:
                if v.startswith(('http://','https://','#','mailto:')):
                    continue
                path = v.split('#',1)[0]
                if path and not (f.parent / path).resolve().exists():
                    missing.append((str(f.relative_to(ROOT)), kind, v))
    for item in missing:
        print('MISSING', *item)
    return 1 if missing else 0

if __name__ == '__main__':
    sys.exit(main())
