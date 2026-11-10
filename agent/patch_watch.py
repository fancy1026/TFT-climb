"""The check after each patch (the owner, 2026-11-05: "check the new patch notes, see if some things should be updated,
make this an auto check after each patch"). Patch 18.6 went live on 11-03/04 while the reference files and the game
data stayed on 18.5 (the game-data download only ran when no snapshot existed), so games 60-64 played 18.6 on 18.5
numbers.

Before each game the loop reads Riot's TFT news (at most every CHECK_EVERY hours). A new patch article, or a changed
one (a mid-patch update is added to the same article), makes the step run: the notes' text is saved for the Editor
(data/sources/patch-notes-<version>.txt), today's Community Dragon game data is downloaded, and a P0 request goes to
the Editor at once: the condensed change list (data/sources/patch-<version>.md, as for the earlier patches), the
reference files rebuilt, and every mechanic change checked against the harness with tests. The harness change then
plays the live check before the next ranked game."""
import hashlib
import html
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / 'agent' / 'patch_watch.json'
SOURCES = ROOT / 'data' / 'sources'
NEWS = 'https://teamfighttactics.leagueoflegends.com/en-us/news/game-updates/'
GAME_DATA = 'https://raw.communitydragon.org/latest/cdragon/tft/en_us.json'
CHECK_EVERY = 3                    # hours between reads of the news page
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}


def get(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def newest(page):
    """(version, url) of the newest 'Teamfight Tactics patch X.Y' article on the news page's HTML."""
    found = {(int(a), int(b), c): f'{NEWS}teamfight-tactics-patch-{a}-{b}{c}/'
             for a, b, c in re.findall(r'teamfight-tactics-patch-(\d+)-(\d+)([a-z]?)\b', page)}
    if not found:
        return None, None
    a, b, c = max(found)
    return f'{a}.{b}{c}', found[(a, b, c)]


def text_of(page):
    """The readable text of an article's HTML (scripts, styles and tags removed)."""
    page = re.sub(r'(?is)<(script|style|svg)[^>]*>.*?</\1>', ' ', page)
    page = re.sub(r'(?i)<br\s*/?>|</(p|li|h\d|div|tr)>', '\n', page)
    t = html.unescape(re.sub(r'<[^>]+>', ' ', page))
    t = re.sub(r'[ \t\r\f\v]+', ' ', t)
    return '\n'.join(l.strip() for l in t.splitlines() if l.strip())


def load():
    try:
        return json.loads(STATE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def save(state):
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def due(now=None):
    """The news was last read CHECK_EVERY hours ago or more (no network)."""
    return (now or time.time()) - float(load().get('checked') or 0) >= CHECK_EVERY * 3600


def look(now=None):
    """Read the news; -> {'version', 'url', 'text', 'hash', 'new'} with new = 'patch' | 'update' | None, or None when
    it could not be read."""
    now = now or time.time()
    st = load()
    try:
        version, url = newest(get(NEWS).decode('utf-8', 'replace'))
        if not version:
            return None
        text = text_of(get(url).decode('utf-8', 'replace'))
    except Exception:
        return None
    # only the change lines (old => new): the page's other text - the related articles below it - changes order
    # between reads (11-06 00:10: a false 'update' of 18.6 that started the Editor)
    changes = '\n'.join(l for l in text.splitlines() if '⇒' in l or '=>' in l)
    h = hashlib.sha1(changes.encode('utf-8')).hexdigest()[:16]
    new = 'patch' if version != st.get('version') else ('update' if h != st.get('hash') else None)
    st['checked'] = now
    save(st)
    return {'version': version, 'url': url, 'text': text, 'hash': h, 'new': new}


def game_data():
    """Today's Community Dragon game data (data/sources/cdragon-tft-<date>.json). -> its path."""
    raw = get(GAME_DATA, timeout=180)
    json.loads(raw)                                      # a broken download fails here
    path = SOURCES / f'cdragon-tft-{time.strftime("%Y-%m-%d")}.json'
    path.write_bytes(raw)
    return path


def request_text(seen, notes, data):
    v = seen['version']
    what = 'a new patch' if seen['new'] == 'patch' else 'a mid-patch update to its notes'
    return (f"[P0, patch {v}] Riot posted {what} ({seen['url']}). Its text is {notes.relative_to(ROOT).as_posix()}; "
            f"today's game data is {data.relative_to(ROOT).as_posix()}. (1) Write or update "
            f"data/sources/patch-{v}.md like data/sources/patch-18.5.md: every game-play change, old => new, the "
            f"newest first; no cosmetics. (2) Rebuild the reference files (tools/build_reference.py) and check that "
            f"the README marks this patch's file as current, not history. (3) Check every mechanic change against the "
            f"harness and fix what it touches, with tests: shop and Wisp prices and kinds, the carousel's timing, the "
            f"item bench and anvils, augments (the e147 table: reassess changed cards), emblems, units, traits and "
            f"items added or removed, the round structure. List what needed nothing.")


def step(cfg):
    """The patch check, as a loop task. -> what happened, in a few words."""
    import climb
    seen = look()
    if not seen or not seen['new']:
        return 'no new patch notes'
    v = seen['version']
    SOURCES.mkdir(parents=True, exist_ok=True)
    notes = SOURCES / f'patch-notes-{v}.txt'
    notes.write_text(seen['text'], encoding='utf-8')
    try:
        data = game_data()
    except Exception as e:
        climb.say(f'patch check: the game data download failed ({e.__class__.__name__}: {e}): next check')
        return 'game data download failed'
    made = climb.add_requests('editor', [request_text(seen, notes, data)], len(climb.records()), f'Patch check {v}')
    st = load()
    st.update(version=v, hash=seen['hash'], url=seen['url'], when=time.strftime('%Y-%m-%d %H:%M'))
    save(st)
    climb.commit_paths(f"Patch {v}: {seen['new']} found, notes and game data saved, request {made[0]['id']}",
                       STATE.relative_to(ROOT), notes.relative_to(ROOT), data.relative_to(ROOT),
                       climb.REQUESTS.relative_to(ROOT))
    climb.say(f"patch check: {seen['new']} {v} - notes and game data saved; the Editor on {made[0]['id']}")
    climb.editor_on_requests(made, cfg)
    return f"{seen['new']} {v}: {made[0]['id']}"
