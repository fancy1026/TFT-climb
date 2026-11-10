"""The stage report's code side (the owner's master plan, 2026-11-01): what the Reviewer reads besides its stage's
rounds, and what code makes from its grades.

- timeline: the whole game in numbers, given to every stage's audit (a stage-3 choice pays off or fails in stage 4,
  which the stage-3 session would not see otherwise);
- luck: measured, for the model to reason over - copies of the wanted units each roll-down's shops showed against what
  the shop odds give, and each augment's MetaTFT tier;
- scores: each part of a stage out of 10, worked out from the round grades (very good 10, good 8, ok 6, bad 3,
  very bad 0; 'uncertain' left out), so scores compare across games;
- the diff packet: this stage against the same stage of earlier games of the comp, with the strategy version, the
  player model and the code each game ran, and the watches due;
- the report page for the owner: every stage's screenshots, scores, summary, strong-player differences and diff
  verdict, and the running average of places.

    python agent/stage_review.py runs/<id>               the timeline, luck and scores of a reviewed game
    python agent/stage_review.py runs/<id> --page        its report page
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
REVIEWS = AGENT / 'model' / 'reviews'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AGENT))

PARTS = {'augment': ('augment',), 'Wisps': ('wisp',), 'carousel': ('carousel',),
         'board and fight goal': ('goal', 'comp', 'board'), 'items': ('items', 'container'),
         'gold and levels': ('economy', 'levelling', 'rolling', 'buys_sells'), 'positions': ('positioning',)}
POINTS = {'very good': 10, 'good': 8, 'ok': 6, 'bad': 3, 'very bad': 0}
DIFF_VERDICTS = ('keep', 'revise', 'revert', 'not enough evidence')
WATCH_VERDICTS = ('as expected', 'against', 'not enough evidence')
EARLIER = 8                          # earlier games of the comp in a diff packet


def stage_of(rnd):
    import stages
    return stages.stage_of(rnd)


def records():
    f = AGENT / 'games.jsonl'
    return [json.loads(l) for l in f.read_text(encoding='utf-8').splitlines() if l.strip()] if f.exists() else []


def rounds_of(run):
    """The rounds agent/rounds.py wrote for a game (its packets), or built now."""
    f = AGENT / 'reviews' / f'{Path(run).name}-rounds' / 'rounds.json'
    if f.exists():
        return json.loads(f.read_text(encoding='utf-8'))
    import rounds as rounds_mod
    return rounds_mod.game_rounds(ROOT / 'runs' / Path(run).name)


def timeline(rounds):
    """The whole game, one line a round: level, gold at the start of planning and at combat, rerolls, the board's
    trait bonuses, the fight goal and result, augments."""
    out = []
    for r in rounds:
        bits = [r['stage']]
        if r.get('level') is not None:
            bits.append(f"level {r['level']}")
        g0, g1 = r.get('gold_at_start'), r.get('gold')
        if g0 is not None or g1 is not None:
            bits.append(f"gold {g0 if g0 is not None else '?'} -> {g1 if g1 is not None else '?'}")
        if r.get('rolls'):
            bits.append(f"{r['rolls']} rerolls")
        if r.get('trait_bonuses'):
            bits.append(f"traits {r['trait_bonuses']}")
        if r.get('board') is not None:
            bits.append(f"{len(r['board'])} fielded")
        if r.get('augments'):
            bits.append('augment ' + ', '.join(r['augments']))
        if r.get('goal') or r.get('result'):
            bits.append(f"goal {r.get('goal') or '?'}, {r.get('result') or 'no player fight'}")
        out.append(' · '.join(bits))
    return out


def _odds(text):
    m = re.findall(r'(\d+)%', str(text or ''))
    return [int(x) / 100 for x in m[:5]] if len(m) >= 5 else None


def _pool_sizes():
    import rounds as rounds_mod
    rounds_mod.cost_of('x')
    per = {}
    for name, cost in rounds_mod._COST.items():
        try:
            c = int(cost or 0)
        except (TypeError, ValueError):
            continue
        if 1 <= c <= 5:
            per[c] = per.get(c, 0) + 1
    return per


def luck(run, rounds):
    """{stage: [line]}: each roll-down's wanted copies seen against the expected count from the shop odds (a shop
    slot shows one champion of a cost at the cost's odds, each champion of that cost alike; copies already taken from
    the pool are not known, so this is an upper bound on what to expect), and each augment's MetaTFT tier."""
    import metrics
    import rounds as rounds_mod
    ev = metrics.events(ROOT / 'runs' / Path(run).name)
    shop = json.loads((AGENT / 'model' / 'mechanics.json').read_text(encoding='utf-8'))['shop']
    by_level = {int(k): [x / 100 for x in v] for k, v in shop['odds_by_level'].items()}
    odds_at = {r['stage']: _odds(r.get('shop_odds')) or by_level.get(r.get('level') or 0) for r in rounds}
    sizes = {int(k): v for k, v in shop['champions_per_cost'].items()} or _pool_sizes()
    out, done = {}, set()
    for e in ev:
        if e.get('ev') != 'rolldown' or not e.get('shops'):
            continue
        key = (e.get('stage'), json.dumps(e['shops']))
        if key in done:                                  # the same roll-down logged twice
            continue
        done.add(key)
        st, want = e.get('stage'), [w for w in e.get('want') or [] if rounds_mod.cost_of(w)]
        odds = odds_at.get(st)
        if not want or not odds:
            continue
        shops = e['shops']
        seen = sum(1 for shop in shops for name in shop if name in want)
        expect = len(shops) * 5 * sum(odds[rounds_mod.cost_of(w) - 1] / max(1, sizes.get(rounds_mod.cost_of(w), 1))
                                      for w in want)
        verdict = 'about as expected' if abs(seen - expect) <= max(1.0, 0.35 * expect) else \
            'more than expected' if seen > expect else 'fewer than expected'
        out.setdefault(stage_of(st), []).append(
            f"{st}: {len(shops)} shops looked for {', '.join(want)}: {seen} copies shown, about {expect:.1f} expected "
            f"from the odds ({verdict})")
    try:
        from tft_player.meta import META
        tiers = {n: (a or {}).get('tier') for n, a in META.augments.items()}
    except Exception:
        tiers = {}
    for e in ev:
        if e.get('ev') == 'augment' and e.get('name'):
            out.setdefault(stage_of(e.get('stage')), []).append(
                f"{e.get('stage')}: augment {e['name']}, MetaTFT tier {tiers.get(e['name']) or 'not listed'}")
    return out


def scores(audit):
    """{part: {'score', 'decisions', 'uncertain'}} for one stage's audit."""
    got = {}
    for x in (audit or {}).get('rounds') or []:
        for d in (x or {}).get('decisions') or [] if isinstance(x, dict) else []:
            if not isinstance(d, dict):
                continue
            area, grade = str(d.get('area') or '').lower(), str(d.get('grade') or '').lower()
            part = next((p for p, areas in PARTS.items() if area in areas), None)
            if part:
                got.setdefault(part, []).append(grade)
    out = {}
    for part, grades in got.items():
        pts = [POINTS[g] for g in grades if g in POINTS]
        out[part] = {'score': round(sum(pts) / len(pts), 1) if pts else None, 'decisions': len(pts),
                     'uncertain': sum(g == 'uncertain' for g in grades)}
    return out


def score_line(sc):
    return ', '.join(f"{p} {v['score']}/10 ({v['decisions']})" + (f" +{v['uncertain']} uncertain" if v['uncertain'] else '')
                     for p, v in sc.items() if v['score'] is not None) or 'no graded parts'


def played(run):
    """{stage: version} the game played (runs/<run>/stage_strategies.json), {} for games before stage strategies. A
    game whose run folder is gone (2026-11-03: games 52-55) played the versions current when it started, from the
    comp's history - the diff packets had shown them as "none"."""
    from datetime import datetime
    run = Path(run).name
    f = ROOT / 'runs' / run / 'stage_strategies.json'
    try:
        return {s: v.get('v') for s, v in json.loads(f.read_text(encoding='utf-8')).get('stages', {}).items()}
    except (OSError, ValueError):
        pass
    row = next((r for r in records() if r.get('run') == run), None)
    if (ROOT / 'runs' / run).exists() or not row:
        return {}
    import stages
    try:
        t0 = datetime.strptime(run, '%Y%m%d-%H%M%S')
        hist = stages.load(row.get('comp_played') or row.get('comp'))['stages']
    except (ValueError, OSError, KeyError):
        return {}
    out = {}
    for s, vs in hist.items():
        before = [v for v in vs if v.get('when') and v.get('status') != 'rejected'
                  and datetime.fromisoformat(v['when']) <= t0]
        if before:
            out[s] = before[-1]['v']
    return out


def audit_of(run, stage):
    """One stage's audit, with the owner's grades of that game in place of the Reviewer's (agent/expert.py apply: the
    domain expert's feedback comes first) and the owner's words on the stage as "owner"."""
    import expert
    f = REVIEWS / f'{Path(run).name}-audit' / f'stage-{stage}.json'
    try:
        a = json.loads(f.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    owner, words = expert.grades(Path(run).name)
    if not owner and not words.get(str(stage)):
        return a
    # an owner grade belongs to this file when its round is in this game stage (audit files are one per game stage)
    have = {x.get('round') for x in a.get('rounds') or [] if isinstance(x, dict)}
    keep = have | {g['round'] for g in owner if str(g['round']).split('-')[0] == str(stage)}
    rounds = [x for x in expert.apply(Path(run).name, a.get('rounds') or []) if x.get('round') in keep]
    return dict(a, rounds=rounds, **({'owner': words[str(stage)]} if words.get(str(stage)) else {}))


def diff_of(run, stage):
    f = REVIEWS / f'{Path(run).name}-audit' / f'diff-{stage}.json'
    try:
        return json.loads(f.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def stage_audits(run, stage):
    """The audit files of one stage key ('5' covers the audits of stages 5, 6 and 7)."""
    s = str(stage)
    keys = ['5', '6', '7'] if s == '5' else [s]
    return [a for a in (audit_of(run, k) for k in keys) if a]


def merged(audits):
    return {'rounds': [x for a in audits for x in (a.get('rounds') or [])]}


def packet_extra(run, stage, rounds, luck_lines):
    """The lines added to a stage's audit packet: the whole game in numbers and the measured luck."""
    md = ["\n## The whole game in numbers (every stage: what this stage's choices led to later)"]
    md += [f"- {line}" for line in timeline(rounds)]
    lk = luck_lines.get(stage_of(f'{stage}-1')) or []
    if lk:
        md.append(f"\n## Luck measured in this stage (reason over these numbers; they are not a grade)")
        md += [f"- {line}" for line in lk]
    md.append("\nGrade each decision against the best one the state allowed, not against any strategy text the "
              "player's prompts show: whether a strategy was followed is not a grade.")
    return '\n'.join(md) + '\n'


def diff_packet(run, stage, comp, watches=()):
    """agent/reviews/<run>-diff/stage-<s>.md: this stage of this game against the same stage of the comp's earlier
    reviewed games. -> (path, the strategy version this game played or None)."""
    import stages
    run = Path(run).name
    rows = [r for r in records() if r.get('run') and (r.get('comp_played') or r.get('comp')) == comp]
    me = next((r for r in rows if r['run'] == run), {'run': run})
    earlier = [r for r in rows if r['run'] < run and stage_audits(r['run'], stage)][-EARLIER:]
    md = [f"# Diff report packet: {comp}, {stages.NAMES.get(stage, 'stage ' + stage)}, game {me.get('n', '?')} ({run})\n",
          "Each game below: its place, player model, code, the stage strategy version it played, this stage's scores "
          "(code, from the round grades), the stage in numbers and the luck measured. Places carry luck: judge the "
          "decisions, and say when the evidence is not enough.\n"]
    hist = stages.load(comp)['stages'][stage]
    if hist:
        md.append("## This stage's strategy versions")
        for v in hist:
            md.append(f"- version {v['v']} ({v['status']}, {v.get('by') or '?'}, {str(v.get('when'))[:10]}): "
                      f"{v.get('why') or ''}\n  text: {v['text']}" + (f"\n  its watch: {json.dumps(v['watch'], ensure_ascii=False)}"
                                                                     if v.get('watch') else ''))
    else:
        md.append("## This stage's strategy versions\n- none yet (the games played without a stage strategy)")
    for r in earlier + [me]:
        aud = merged(stage_audits(r['run'], stage))
        rnds = [x for x in rounds_of(r['run']) if stage_of(x['stage']) == stage] if (AGENT / 'reviews' / f"{r['run']}-rounds").exists() else []
        lk = luck(r['run'], rnds).get(stage, []) if rnds else []
        tag = 'THIS GAME' if r['run'] == run else f"game {r.get('n', '?')}"
        md.append(f"\n## {tag}: {r['run']}, place {r.get('placement', '?')}, player {r.get('model', '?')} "
                  f"{r.get('effort', '')}, code {r.get('code', '?')}, stage strategy version "
                  f"{played(r['run']).get(stage) or 'none'}")
        md.append(f"- scores: {score_line(scores(aud))}")
        md += [f"- the owner (domain expert; comes first) on this stage: {a['owner']}"
               for a in stage_audits(r['run'], stage) if a.get('owner')]
        md += [f"- {line}" for line in timeline(rnds)]
        md += [f"- luck: {line}" for line in lk]
    if watches:
        md.append("\n## Watches due in this stage")
        for w in watches:
            md.append(f"- {json.dumps(w, ensure_ascii=False)}")
    folder = AGENT / 'reviews' / f'{run}-diff'
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f'stage-{stage}.md'
    out.write_text('\n'.join(md) + '\n', encoding='utf-8')
    return out, played(run).get(stage)


def due_watches(comp, stage, run):
    """The watches of this stage's versions played since they shipped (each with the games it has had)."""
    import stages
    vs = [v for v in stages.load(comp)['stages'][stage] if v.get('watch') and v.get('status') in ('current', 'replaced')]
    if not vs:
        return []
    rows = [r for r in records() if (r.get('comp_played') or r.get('comp')) == comp and r.get('run')]
    out = []
    for v in vs:
        games = [r['run'] for r in rows if played(r['run']).get(stage) == v['v'] and r['run'] <= Path(run).name]
        if games:
            out.append({'version': v['v'], 'watch': v['watch'], 'games_played': len(games)})
    return out


def diff_gaps(d):
    """What a diff report leaves out."""
    if not isinstance(d, dict):
        return ['no diff report']
    out = []
    if str(d.get('verdict') or '') not in DIFF_VERDICTS:
        out.append(f"verdict must be one of {', '.join(DIFF_VERDICTS)}")
    for k in ('what_changed', 'play_changed', 'why'):
        if not str(d.get(k) or '').strip():
            out.append(f'{k} missing')
    if d.get('verdict') in ('revise', 'revert') and not str(d.get('for_strategist') or '').strip():
        out.append('for_strategist missing: what to change and the evidence')
    for w in d.get('watches') or []:
        if not isinstance(w, dict) or str(w.get('verdict') or '') not in WATCH_VERDICTS:
            out.append(f"a watch's verdict must be one of {', '.join(WATCH_VERDICTS)}")
    return out


def report_page(run, n=None, place=None, comp=None, folder=None):
    """agent/model/reviews/<run>-report.html: the owner's page for one game."""
    run = Path(run).name
    rows = [r for r in records() if r.get('run')]
    me = next((r for r in rows if r['run'] == run), {})
    comp = comp or me.get('comp_played') or me.get('comp') or ''
    places = [r['placement'] for r in rows if isinstance(r.get('placement'), int)
              and (r.get('comp_played') or r.get('comp')) == comp and r['run'] <= run]
    avg = f"{sum(places) / len(places):.2f} over {len(places)} games" if places else 'no games'
    last = places[-10:]
    avg10 = f"{sum(last) / len(last):.2f}" if last else '-'
    summary = (folder or REVIEWS) / f'{run}.md'
    e = html.escape
    body = [f"<h1>Game {e(str(n or me.get('n', '?')))} · place {e(str(place or me.get('placement', '?')))}</h1>",
            f"<p class=m>{e(run)} · {e(comp)} · player {e(str(me.get('model', '?')))} {e(str(me.get('effort', '')))} · "
            f"code {e(str(me.get('code', '?')))} · running average of places {e(avg)} (last 10: {avg10})</p>"]
    stages_seen = sorted({k for k in ('1', '2', '3', '4', '5') if stage_audits(run, k)})
    for s in stages_seen:
        aud = stage_audits(run, s)
        sc = scores(merged(aud))
        d = diff_of(run, s) or {}
        body.append(f"<section><h2>Stage {e(s if s != '5' else '5-7')}</h2>")
        body.append('<table>' + ''.join(f"<tr><td>{e(p)}</td><td><b>{v['score']}</b>/10</td><td>{v['decisions']} "
                                        f"decisions{(', ' + str(v['uncertain']) + ' uncertain') if v['uncertain'] else ''}</td></tr>"
                                        for p, v in sc.items() if v['score'] is not None) + '</table>')
        for a in aud:
            if a.get('summary'):
                body.append(f"<p>{e(str(a['summary']))}</p>")
            if a.get('against_strong_players'):
                body.append(f"<p><b>Against strong players:</b> {e(str(a['against_strong_players']))}</p>")
        if d:
            body.append(f"<p><b>Diff report: {e(str(d.get('verdict')))}.</b> {e(str(d.get('why') or ''))}</p>")
        shots = []
        for a in aud:
            for x in a.get('rounds') or []:
                rnd = str((x or {}).get('round') or '')
                for kind in ('start', 'fight'):
                    p = ROOT / 'runs' / run / f'{rnd}-{kind}.jpg'
                    if p.exists():
                        shots.append(f"<figure><img loading=lazy src='../../../runs/{e(run)}/{e(p.name)}'>"
                                     f"<figcaption>{e(rnd)} {kind}</figcaption></figure>")
        if shots:
            body.append("<div class=shots>" + ''.join(shots) + '</div>')
        body.append('</section>')
    if summary.exists():
        text = summary.read_text(encoding='utf-8').split('## Round by round')[0]
        body.append(f"<section><h2>Summary</h2><pre>{e(text)}</pre></section>")
    page = ("<!doctype html><meta charset=utf-8><title>Game report</title><style>body{font:15px/1.5 system-ui;"
            "max-width:1100px;margin:24px auto;padding:0 16px;background:#f5f6f8;color:#18202b}section{background:#fff;"
            "border:1px solid #d6dce4;border-radius:10px;padding:12px 16px;margin:14px 0}.m{color:#5b6677}td{padding:2px 10px}"
            ".shots{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px}img{width:100%;"
            "border-radius:6px}figure{margin:0}figcaption{font-size:12px;color:#5b6677}pre{white-space:pre-wrap}</style>"
            + '\n'.join(body))
    out = (folder or REVIEWS) / f'{run}-report.html'
    out.write_text(page, encoding='utf-8')
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--page', action='store_true')
    a = ap.parse_args()
    run = Path(a.run).name
    if a.page:
        print(report_page(run))
        return
    rnds = rounds_of(run)
    print('\n'.join(timeline(rnds)))
    for s, lines in luck(run, rnds).items():
        print(f'== luck, stage {s}\n' + '\n'.join(lines))
    for s in ('1', '2', '3', '4', '5'):
        aud = stage_audits(run, s)
        if aud:
            print(f'== scores, stage {s}: {score_line(scores(merged(aud)))}')


if __name__ == '__main__':
    main()
