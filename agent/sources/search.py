"""Candidate videos for the Strategist (agent/roles/strategist.md): Bilibili video search, full-length games of the current set.

    python agent/sources/search.py [--query "<words>" ...] [--since 2026-09-01] [--min-minutes 15]
        -> agent/sources/candidates.jsonl (merged with what is there; one line per video)

Without --query it runs the default searches below. The Data agent may run its own (a player's name, a comp, a
patch). Search only: no video is downloaded here.
"""
import argparse
import http.cookiejar
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'candidates.jsonl'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36'
DEFAULT = ['云顶之弈 完整对局', '云顶之弈 第一视角', '云顶之弈 宗师 上分', '云顶之弈 大师 实战', '云顶之弈 国服第一',
           '云顶之弈 直播录像', '云顶之弈 S18 上分', '慎独y', '云顶之弈 千分', '云顶之弈 韩服 王者']


def opener():
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    op.addheaders = [('User-Agent', UA), ('Referer', 'https://search.bilibili.com/')]
    op.open('https://www.bilibili.com/', timeout=20).read()          # sets the buvid cookie the API asks for
    return op


def seconds(d):
    s = 0
    for p in str(d).split(':'):
        s = s * 60 + int(p or 0)
    return s


def search(op, q, order, page):
    url = ('https://api.bilibili.com/x/web-interface/search/type?search_type=video&keyword=' + urllib.parse.quote(q) +
           f'&order={order}&page={page}')
    j = json.loads(op.open(url, timeout=20).read().decode('utf-8'))
    return (j.get('data') or {}).get('result') or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', action='append')
    ap.add_argument('--since', default='2026-09-01')
    ap.add_argument('--min-minutes', type=float, default=15)
    ap.add_argument('--pages', type=int, default=2)
    a = ap.parse_args()
    since = datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc).timestamp()
    have = {}
    if OUT.exists():
        for line in OUT.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                have[r['bvid']] = r
    op = opener()
    strip = lambda s: re.sub(r'<[^>]+>', '', s or '').replace('&quot;', '"').replace('&amp;', '&')
    new = 0
    for q in a.query or DEFAULT:
        for order in ('pubdate', 'totalrank'):
            for page in range(1, a.pages + 1):
                try:
                    rows = search(op, q, order, page)
                except Exception as ex:
                    print(f'{q} {order} {page}: {ex}')
                    continue
                for v in rows:
                    s = seconds(v.get('duration', 0))
                    if v['bvid'] in have or s < a.min_minutes * 60 or v.get('pubdate', 0) < since:
                        continue
                    have[v['bvid']] = {'bvid': v['bvid'], 'title': strip(v.get('title')), 'uploader': v.get('author'),
                                       'mid': v.get('mid'), 'duration_s': s,
                                       'published': datetime.fromtimestamp(v['pubdate'], timezone.utc).strftime('%Y-%m-%d'),
                                       'views': v.get('play'), 'description': strip(v.get('description'))[:300], 'query': q}
                    new += 1
                time.sleep(0.4)
    OUT.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in have.values()), encoding='utf-8')
    print(f'{new} new candidates, {len(have)} in {OUT}')


if __name__ == '__main__':
    main()
