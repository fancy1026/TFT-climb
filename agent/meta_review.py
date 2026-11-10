"""The meta review (the owner, 2026-11-03: "keep the current design, just review after every game, and then a meta
review every 5-10 games, additionally"): each game keeps its own review; every META_EVERY reviewed games one Reviewer
session reads the batch together (Job 5) - each game's review, grades, diff and change verdicts, the strategy text and
comp notes each game played, the requests - and names what recurs across games, what the batch says about each
change and what the per-game steps miss. Code sends its actions to the roles that own them.

    python agent/meta_review.py packet       the next batch's packet (agent/reviews/meta-<a>-<b>.md), no session
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
sys.path.insert(0, str(AGENT))

META_EVERY = 5                      # reviewed games per meta review
META = AGENT / 'model' / 'reviews' / 'meta'
TO = ('editor', 'comp_coach', 'strategist', 'owner')


def records():
    f = AGENT / 'games.jsonl'
    return [json.loads(l) for l in f.read_text(encoding='utf-8').splitlines() if l.strip()] if f.exists() else []


def reviewed(rows=None):
    """The games whose steps after the game ran (a review), in order."""
    return [r for r in (records() if rows is None else rows) if r.get('placement') and 'reviewer' in r]


def metas():
    """The meta reviews written, oldest first (meta/<first>-<last>.json)."""
    out = []
    for f in sorted(META.glob('*.json'), key=lambda p: [int(x) for x in p.stem.split('-')]):
        try:
            out.append(json.loads(f.read_text(encoding='utf-8')) | {'file': f.relative_to(ROOT).as_posix()})
        except (OSError, ValueError):
            pass
    return out


def batch(rows=None):
    """The reviewed games after the last meta review's last game; with none yet, the latest META_EVERY."""
    games = reviewed(rows)
    done = metas()
    if not done:
        return games[-META_EVERY:]
    last = max(done[-1].get('games') or [0])
    return [r for r in games if r['n'] > last]


def due(rows=None):
    return len(batch(rows)) >= META_EVERY


def name(rows):
    return f"{rows[0]['n']}-{rows[-1]['n']}"


def git(*a):
    r = subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return r.stdout.strip()


def started(row):
    """The game's start, from its run name (YYYYmmdd-HHMMSS)."""
    try:
        return datetime.strptime(row['run'], '%Y%m%d-%H%M%S')
    except (KeyError, ValueError):
        return None


def versions_played(row):
    """{stage: version} the game played."""
    import stage_review
    return stage_review.played(row['run'])


def notes_before(rows):
    """{game n: [commit lines]}: the comp-note rewrites that landed before each game (after the one before it)."""
    out = {}
    before = [r for r in records() if r.get('n', 0) < rows[0]['n'] and started(r)]
    for prev, r in zip([before[-1] if before else None] + rows[:-1], rows):
        t0, tp = started(r), started(prev) if prev else None
        if not t0:
            continue
        args = ['log', '--format=%h %ad %s', '--date=format:%m-%d %H:%M', f'--until={t0:%Y-%m-%d %H:%M:%S}']
        args.append(f'--since={tp:%Y-%m-%d %H:%M:%S}' if tp else '--max-count=3')
        out[r['n']] = [l for l in git(*args, '--', 'agent/memory/comps').splitlines() if l.strip()]
    return out


def changes_in(rows):
    """{(change, part): [(game, verdict, effect, why)]} from agent/changes.jsonl for the batch's games."""
    ns = {r['n'] for r in rows}
    f = AGENT / 'changes.jsonl'
    out = {}
    for l in (f.read_text(encoding='utf-8').splitlines() if f.exists() else []):
        try:
            c = json.loads(l)
        except ValueError:
            continue
        if c.get('game') in ns:
            out.setdefault((c.get('change'), c.get('part') or c.get('subject') or ''), []).append(
                (c['game'], c.get('verdict'), c.get('effect'), str(c.get('why') or '')[:160]))
    return out


def requests_in(rows):
    f = AGENT / 'model' / 'requests.jsonl'
    ns = {r['n'] for r in rows}
    rs = [json.loads(l) for l in f.read_text(encoding='utf-8').splitlines() if l.strip()] if f.exists() else []
    return [r for r in rs if r.get('game') in ns]


def packet(rows):
    """agent/reviews/meta-<a>-<b>.md: the batch in one file, with the paths to read for more. -> path."""
    a = name(rows)
    md = [f"# Meta review packet: games {a}\n",
          "Each game was reviewed on its own after it was played. This packet puts the batch side by side: read across "
          "the games, not one at a time. Places carry luck: judge decisions and changes, not places.\n"]
    notes = notes_before(rows)
    for r in rows:
        rv = r.get('reviewer') if isinstance(r.get('reviewer'), dict) else {}
        au = r.get('audit') or {}
        rk = r.get('rank_after') or {}
        md.append(f"\n## Game {r['n']}: {r['run']}, {r.get('comp_played') or r.get('comp')}, place {r.get('placement')}, "
                  f"LP {r.get('lp_change')} -> {rk.get('tier', '?')} {rk.get('division', '')} {rk.get('lp', '')}; "
                  f"player {r.get('model')} {r.get('effort', '')}; code {r.get('code')} -> {r.get('code_after')}")
        vp = versions_played(r)
        md.append(f"- stage strategy versions played: "
                  + (', '.join(f'stage {s} v{v}' for s, v in sorted(vp.items())) or 'none recorded'))
        md.append(f"- comp notes rewritten before this game: " + ('; '.join(notes.get(r['n']) or []) or 'none'))
        md.append(f"- grades: {au.get('graded')} decisions, {au.get('bad')} bad, {au.get('very_bad')} very bad; "
                  f"by area {json.dumps(au.get('by_area') or {})}; by cause {json.dumps(au.get('by_cause') or {})}")
        for x in (rv.get('placement_reasons') or [])[:4]:
            md.append(f"- decided the place ({x.get('kind')}, {x.get('rounds')}): {x.get('reason')}")
        for x in (rv.get('worst') or [])[:6]:
            md.append(f"- worst: {x.get('round')} {x.get('area')} {x.get('grade')} ({x.get('cause')}): "
                      f"{x.get('decision')} -> better: {x.get('better')}")
        md += [f"- pattern: {p}" for p in (rv.get('patterns') or [])[:5]]
        md += [f"- harness fault named: {h}" for h in (rv.get('harness') or [])[:5]]
        fx = r.get('harness_fixer') if isinstance(r.get('harness_fixer'), dict) else {}
        md += [f"- the Editor fixed: {x}" for x in (fx.get('fixed') or [])[:5]]
        md += [f"- the Editor left: {x}" for x in (fx.get('left') or [])[:5]]
        for s, d in sorted((r.get('diffs') or {}).items()):
            md.append(f"- diff stage {s}: {(d or {}).get('verdict')} - {str((d or {}).get('why') or '')[:220]}")
        for s, d in sorted((r.get('stage_strategies') or {}).items()):
            md.append(f"- stage {s} strategy after this game: {str((d or {}).get('done'))[:200]}")
        cr = r.get('change_review') or {}
        if cr:
            md.append(f"- change review: {cr.get('keep', 0)} keep, {cr.get('remove', 0)} remove, "
                      f"{cr.get('monitor', 0)} watched")
        md.append(f"- read: agent/model/reviews/{r['run']}.md (summary and round grades), agent/model/reviews/"
                  f"{r['run']}-audit/, agent/reviews/{r['run']}-digest.md")
    ch = changes_in(rows)
    if ch:
        md.append("\n## Each change across the batch (the change reviews' verdicts, game by game)")
        for (c, part), vs in sorted(ch.items(), key=lambda kv: -len(kv[1])):
            md.append(f"- {c} {part[:90]}: " + '; '.join(f"g{g} {v} ({e})" for g, v, e, _ in vs)
                      + (f" - last why: {vs[-1][3]}" if vs[-1][3] else ''))
    rq = requests_in(rows)
    if rq:
        md.append("\n## Requests made in the batch")
        md += [f"- [{q['id']}] {q.get('kind')} from {q.get('from')} after game {q.get('game')}, {q.get('status')}"
               f"{' (' + str(q.get('done'))[:60] + ')' if q.get('done') else ''}: {str(q.get('text'))[:200]}" for q in rq]
    try:                                                 # the round cases (agent/round_cases.py, the owner 2026-11-10)
        import round_cases
        cs, fail = round_cases.load(), round_cases.failing_text(limit=20)
        if cs:
            md.append(f"\n## Round cases: {len(cs) - len(round_cases.failing())} of {len(cs)} pass on the current code")
            md += [f"- {x}" for x in fail] or ['- none failing']
    except Exception as e:
        md.append(f"\n## Round cases: not read ({e.__class__.__name__}: {e})")
    prev = metas()
    if prev:
        p = prev[-1]
        md.append(f"\n## The last meta review ({p['file']}): its actions")
        md += [f"- to {x.get('to')}: {str(x.get('text'))[:200]}" for x in p.get('actions') or []]
    out = AGENT / 'reviews' / f'meta-{a}.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(md) + '\n', encoding='utf-8')
    return out


def result(rows):
    f = META / f'{name(rows)}.json'
    try:
        return json.loads(f.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def gaps(meta):
    """What a meta review left out, for one more pass."""
    if not isinstance(meta, dict):
        return ['no meta/<a>-<b>.json written']
    miss = [k for k in ('recurring', 'changes', 'luck', 'framework', 'actions') if k not in meta]
    miss += [f"action {i + 1}: 'to' is not one of {', '.join(TO)}" for i, x in enumerate(meta.get('actions') or [])
             if not isinstance(x, dict) or x.get('to') not in TO]
    return miss


def strategist_note():
    """The latest meta review's points for the Strategist, until a Strategist run has had them. -> (text, stages)."""
    done = metas()
    if not done:
        return '', set()
    m = done[-1]
    acts = [x for x in m.get('actions') or [] if isinstance(x, dict) and x.get('to') == 'strategist']
    if not acts or m.get('strategist_had'):
        return '', set()
    stages = {str(x.get('stage')) for x in acts if x.get('stage')}
    text = (f"The meta review of games {m['games'][0]}-{m['games'][-1]} ({m['file'][:-5]}.md) asks you: "
            + ' '.join(f"[stage {x.get('stage') or '?'}] {x.get('text')} (evidence: {x.get('evidence') or '-'})"
                       for x in acts) + ' ')
    return text, stages


def mark_strategist_had(n):
    done = metas()
    if done:
        f = ROOT / done[-1]['file']
        m = json.loads(f.read_text(encoding='utf-8'))
        m['strategist_had'] = n
        f.write_text(json.dumps(m, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.argv[1:2] == ['packet']:
        rows = batch()
        print(packet(rows) if rows else 'no reviewed games', f'({len(rows)} games; due: {due()})')
