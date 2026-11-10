"""Stage strategies (the owner's master plan, 2026-11-01): each comp has one strategy per stage - 1 (1-1..1-4), 2, 3, 4,
and 5 for stages 5-7 - sent with every model call of that stage, marked as learned from strong players' games and our
earlier games with the comp. The comp coach writes version 1; the Strategist writes later versions, each checked by the
Judge on the stage's recorded decisions before it ships (agent/judge.py stage). Every version keeps its reason, its
sources and its watch, so the diff report can tell which version each game played and revert one.

    python agent/stages.py show "<comp>"                      the current version of each stage
    python agent/stages.py history "<comp>" <stage>           every version of a stage
    python agent/stages.py revert "<comp>" <stage> --why ...  the previous version back, as a new version
    python agent/stages.py heldout "<comp>"                   the comp's strong players' games kept for the Judge

The file: agent/comps/<comp>/stages.json - {"comp", "stages": {"1": [version, ...], ...}}; a version is
{"v", "text", "why", "sources", "by", "when", "status": current|replaced|rejected, "watch", "judge"}.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
COMPS = AGENT / 'comps'
STAGES = ('1', '2', '3', '4', '5')
NAMES = {'1': 'stage 1 (1-1 to 1-4)', '2': 'stage 2', '3': 'stage 3', '4': 'stage 4', '5': 'stages 5 to 7'}
MAX_CHARS = 1500                     # one stage's text: it goes with every call of the stage
HELDOUT_SHARE = 0.3                  # a comp's strong players' games kept from the coach and the Strategist


def slug(comp):
    return re.sub(r'[^A-Za-z0-9]+', '_', str(comp)).strip('_')


def stage_of(rnd):
    """'3-5' -> '3'; stages 5, 6 and 7 share '5'; '' for anything else."""
    m = re.match(r'\s*(\d+)', str(rnd or ''))
    if not m:
        return ''
    n = int(m.group(1))
    return str(min(n, 5)) if n >= 1 else ''


def path(comp):
    return COMPS / slug(comp) / 'stages.json'


def load(comp):
    try:
        data = json.loads(path(comp).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        data = {}
    data.setdefault('comp', comp)
    stages = data.setdefault('stages', {})
    for s in STAGES:
        stages.setdefault(s, [])
    return data


def save(comp, data):
    p = path(comp)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def current(comp):
    """{stage: the version in use, or None}."""
    data = load(comp)
    return {s: next((v for v in reversed(data['stages'][s]) if v.get('status') == 'current'), None) for s in STAGES}


def frozen(comp):
    """What one game plays: {stage: {'v', 'text'}} for the stages that have a strategy (read once, at game start)."""
    return {s: {'v': v['v'], 'text': v['text']} for s, v in current(comp).items() if v}


def block(comp, stage, ver):
    """The text the player gets with every call of the stage."""
    return (f"STAGE {stage} STRATEGY for {comp} ({NAMES.get(stage, 'stage ' + stage)}, version {ver['v']}) - learned "
            f"from strong players' games and our earlier games with this comp; it applies to every decision of this "
            f"stage, and where it disagrees with the comp's FIRST RULES, the FIRST RULES hold:\n{ver['text'].strip()}")


def check(text, sources=()):
    """(ok, why not) for a version's text and sources."""
    text = str(text or '').strip()
    if not text:
        return False, 'no text'
    if len(text) > MAX_CHARS:
        return False, f'{len(text)} characters; a stage strategy is at most {MAX_CHARS}'
    sys.path.insert(0, str(AGENT))
    from comp_coach import check_source
    bad = [s for s in sources or [] if not (isinstance(s, dict) and check_source(s))]
    if bad:
        return False, 'sources that do not check (a game we hold and a round it had): ' + json.dumps(bad[:3])
    return True, ''


def add_version(comp, stage, text, why, sources=(), by='', watch=None, judge=None):
    """A new version in use for the stage (the one before it is 'replaced'). -> (version, None) or (None, why not)."""
    stage = str(stage)
    if stage not in STAGES:
        return None, f'no stage {stage!r}'
    ok, why_not = check(text, sources)
    if not ok:
        return None, why_not
    data = load(comp)
    vs = data['stages'][stage]
    for v in vs:
        if v.get('status') == 'current':
            v['status'] = 'replaced'
    ver = {'v': max([v['v'] for v in vs] or [0]) + 1, 'text': str(text).strip(), 'why': str(why or '').strip(),
           'sources': list(sources or []), 'by': by, 'when': datetime.now().isoformat(timespec='seconds'),
           'status': 'current', 'watch': watch, 'judge': judge}
    vs.append(ver)
    save(comp, data)
    return ver, None


def reject(comp, stage, text, why, by='', judge=None):
    """A candidate the Judge did not pass, kept with its verdict so the Strategist sees what was tried."""
    data = load(comp)
    vs = data['stages'][str(stage)]
    vs.append({'v': max([v['v'] for v in vs] or [0]) + 1, 'text': str(text or '').strip(), 'why': str(why or ''),
               'by': by, 'when': datetime.now().isoformat(timespec='seconds'), 'status': 'rejected', 'judge': judge})
    save(comp, data)


def revert(comp, stage, why, by=''):
    """The version before the current one back in use, as a new version. -> (version, None) or (None, why not)."""
    vs = [v for v in load(comp)['stages'][str(stage)] if v.get('status') in ('current', 'replaced')]
    if len(vs) < 2:
        return None, 'no earlier version to go back to'
    prev = vs[-2]
    return add_version(comp, stage, prev['text'], f"revert to version {prev['v']}: {why}", prev.get('sources') or [],
                       by=by, watch=None, judge={'revert_of': prev['v']})


def heldout(comp, games=None):
    """The comp's strong players' games kept for the Judge: no role reads them (the material leaves them out). Chosen
    once, by name hash, about HELDOUT_SHARE of the games known then (at least one when there are 3+); later games
    are kept in the same share as they come. -> [game names]."""
    f = COMPS / slug(comp) / 'heldout.json'
    try:
        kept = json.loads(f.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        kept = {'heldout': [], 'seen': []}
    if games is not None:
        new = [g for g in sorted(set(games)) if g not in kept['seen']]
        if new:
            kept['seen'] += new
            want = round(len(kept['seen']) * HELDOUT_SHARE) if len(kept['seen']) >= 3 else 0
            for g in sorted(new, key=lambda g: hashlib.sha1(g.encode()).hexdigest()):
                if len(kept['heldout']) < want:
                    kept['heldout'].append(g)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(kept, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return list(kept['heldout'])


def is_heldout(comp, game):
    base = re.sub(r'-g\d+$', '', str(game))
    return any(re.sub(r'-g\d+$', '', h) == base for h in heldout(comp))


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('show')
    s.add_argument('comp')
    h = sub.add_parser('history')
    h.add_argument('comp')
    h.add_argument('stage')
    r = sub.add_parser('revert')
    r.add_argument('comp')
    r.add_argument('stage')
    r.add_argument('--why', required=True)
    k = sub.add_parser('heldout')
    k.add_argument('comp')
    a = ap.parse_args()
    if a.cmd == 'show':
        for st, v in current(a.comp).items():
            print(f"== {NAMES[st]}: " + (f"version {v['v']} ({v['by']}, {v['when'][:10]})\n{v['text']}\n" if v else 'none'))
    elif a.cmd == 'history':
        for v in load(a.comp)['stages'][a.stage]:
            print(f"v{v['v']} {v['status']} {v['by']} {v['when'][:16]}: {v['why'][:200]}")
    elif a.cmd == 'revert':
        v, why = revert(a.comp, a.stage, a.why, by='owner')
        print(f"version {v['v']} in use" if v else f'not reverted: {why}')
    else:
        print('\n'.join(heldout(a.comp)) or 'none')


if __name__ == '__main__':
    main()
