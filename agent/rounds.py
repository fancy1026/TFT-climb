"""One game round by round, for the review of every decision (the owner, 2026-10-29: "review every single round for
every single action and ask, given that information and state for that current round, whether the decision made
was the best decision. Could it be anything better?").

    python agent/rounds.py runs/<id>                 agent/reviews/<id>-rounds/<stage>.md, one packet per stage
    python agent/rounds.py runs/<id> --check <audit> which rounds and decisions a stage audit leaves out

Code gathers, per round, where the player's state and answers are in the conversation, what was executed, the fight's
goal and result, and the board's measured facts (traits and their breakpoints, units that add nothing to a trait,
units owned and not fielded, the carry, who holds the items, idle components, free rerolls). The Reviewer grades each
decision from these. No HP figure is measured or used (the owner: "HP should not be used at all for any metric").
"""
import argparse
import json
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AGENT))

import metrics                                            # noqa: E402

# the decisions every planning round is graded on (the owner's example: "2-1: augment choice: good, why; comp plan:
# very bad, why; positioning: bad, why"); augments, the carousel, Wisps and containers are added where they happened
AREAS = ('goal', 'comp', 'board', 'positioning', 'items', 'economy', 'levelling', 'rolling', 'buys_sells')
EVENT_AREAS = ('augment', 'carousel', 'wisp', 'container')
# 'uncertain': the better decision cannot be established from what the record shows (the owner's master plan,
# 2026-11-01); it says why, and the stage's score leaves it out
GRADES = ('very good', 'good', 'ok', 'bad', 'very bad', 'uncertain')
BAD = ('bad', 'very bad')
# why a decision graded bad was made: the player's reasoning, the state not showing what it needed, or the harness not
# doing what it decided
CAUSES = ('reasoning', 'information', 'execution')


def stage_key(stage):
    m = re.match(r'(\d+)-(\d+)', str(stage or ''))
    return (int(m.group(1)), int(m.group(2))) if m else (99, 99)


def conversation_ranges(run):
    """{stage: (first line, last line)} of runs/<run>/conversation.md, 1-based, from its '### Round x-y' markers."""
    f = Path(run) / 'conversation.md'
    if not f.exists():
        return {}
    lines = f.read_text(encoding='utf-8', errors='replace').splitlines()
    marks = [(i + 1, m.group(1)) for i, l in enumerate(lines) for m in [re.match(r'### Round (\d+-\d+)', l)] if m]
    return {st: (a, (marks[k + 1][0] - 1) if k + 1 < len(marks) else len(lines)) for k, (a, st) in enumerate(marks)}


def units(roster, where):
    """[(name, stars, [items], slot)] of the board ('h') or bench ('b') part of a roster string."""
    part = (roster or '').split(' bench=[')[0 if where == 'h' else -1] if ' bench=[' in (roster or '') else ''
    out = []
    for m in re.finditer(r"([A-Z?][\w'. ]*?)(\*{1,3})@" + where + r"(\d+)(?:\[([^\]]*)\])?", part):
        out.append((m.group(1).strip(), len(m.group(2)),
                    [i.strip() for i in (m.group(4) or '').split(',') if i.strip()], m.group(3)))
    return out


def catalog():
    try:
        from tft_player.catalog import CATALOG
        return CATALOG
    except Exception:
        return None


def trait_facts(board):
    """The fought board's traits: [{trait, count, active, next, short}] - active is the highest breakpoint reached,
    next the one above it and short the units still needed for it - and the traits with units that add nothing
    (a count above its active breakpoint and below the next)."""
    cat = catalog()
    if cat is None:
        return [], []
    from tft_player.state import board_traits
    fielded = [types.SimpleNamespace(name=n, traits=tuple(cat.unit(n).traits) if cat.unit(n) else (), items=items)
               for n, _, items, _ in board if n and n != '?']
    counts = board_traits(fielded)
    rows, wasted = [], []
    for t, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        bps = sorted(cat.traits.get(t) or ())
        active = max((b for b in bps if b <= n), default=0)
        nxt = min((b for b in bps if b > n), default=None)
        rows.append({'trait': t, 'count': n, 'active': active, 'next': nxt, 'short': (nxt - n) if nxt else None})
        if active and n > active and nxt:
            wasted.append(f"{t} {n}: only the {active} bonus is active, so {n - active} unit(s) add nothing to it; "
                          f"{t} {nxt} needs {nxt - n} more")
    return rows, wasted


def bonuses(fielded):
    """[(trait, count, tier)] of the traits at a breakpoint of 2 or more for [(name, items)] (tft_player.board_score)."""
    from tft_player import board_score
    return board_score.bonuses(fielded)


def trait_options(owned, team, shop=(), gold=0, limit=12):
    """The strongest boards the owned units make, and make with shop cards bought within the gold, traits first
    (tft_player.board_score: trait tiers, then unit value - the owner, 2026-10-30: "it should follow by traits").
    -> {'owned': (bonuses, units), 'with_shop': (bonuses, units, bought)}"""
    if catalog() is None:
        return {}
    from tft_player import board_score
    return board_score.best(owned, team, shop, gold, limit)


def unit_line(name, stars, items):
    cat = catalog()
    u = cat.unit(name) if cat else None
    return (f"{name} {'*' * stars} ({u.cost}g, {'/'.join(u.traits)})" if u else f"{name} {'*' * stars}") + \
           (f" [{', '.join(items)}]" if items else '')


def metatft_boards(comp):
    """{level: [board lines]} of the comp's MetaTFT brief: the board played at each level on the way (with its round
    win rate) and, from level 7, the level's boards with their average place."""
    if not comp:
        return {}
    f = ROOT / 'data' / 'metatft_shots' / re.sub(r'[^A-Za-z0-9]+', '_', comp).strip('_') / 'brief.txt'
    text = f.read_text(encoding='utf-8') if f.exists() else ''
    out = {}
    for m in re.finditer(r'^- Lvl (\d+): (.+)$', text, re.M):                 # the Early list
        out.setdefault(int(m.group(1)), []).append(f"on the way: {m.group(2).strip()}")
    for m in re.finditer(r'^Lvl (\d+) \(([^)]*)\):\n((?:- .+\n?)+)', text, re.M):
        first = m.group(3).splitlines()[0][2:].strip()
        out.setdefault(int(m.group(1)), []).append(f"most played at this level ({m.group(2)}): {first}")
    return out


def answers(ev, stage):
    """The player's answers in this round: [(kind, decision JSON text, its reasons)]."""
    out = []
    for e in ev:
        if e.get('ev') != 'model_call' or e.get('stage') != stage:
            continue
        a = e.get('answer') or ''
        why = ''
        try:
            from tft_player.coach import json_objects
            objs = [o for o in json_objects(a) if isinstance(o, dict)]
        except Exception:
            objs = []
        for o in objs:
            st = o.get('strategy')
            if isinstance(st, dict) and isinstance(st.get('why'), dict):
                why = '; '.join(f"{k}: {v}" for k, v in st['why'].items() if str(v).strip() not in ('', '-'))
                if st.get('fight_goal'):
                    why = f"fight_goal: {st['fight_goal']}; " + why
            elif o.get('why'):
                why = str(o['why'])
        plan = next((json.dumps(o, ensure_ascii=False) for o in objs if 'strategy' not in o and 'why' not in o), a[:300])
        out.append((e.get('kind'), plan[:700], why[:900]))
    return out


def passed_wisps(ev, st):
    """The Wisps the round's rolls rolled past unanswered: the rolldown's wisps_passed, or (older runs) its notice."""
    out = [w for e in ev if e.get('ev') == 'rolldown' and e.get('stage') == st for w in e.get('wisps_passed') or []]
    for e in ev:
        m = re.search(r'rolled past the Wisp\(s\) (.+?) without', e.get('what') or '')             if e.get('ev') == 'comp rule' and e.get('stage') == st else None
        if m and not out:
            out += [w.strip() for w in m.group(1).split(',')]
    return out


def game_rounds(run):
    """Every round from 1-2 to the last, with what the player saw (conversation lines, screenshots), did and decided,
    and the board's measured facts. -> [dict]."""
    run = Path(run)
    ev = metrics.events(run)
    m = metrics.game(run)
    conv = conversation_ranges(run)
    results = {f['stage']: f for f in m.get('fights') or []}
    carry = m.get('carry')
    held = {h['stage']: h for h in m.get('carry_holders') or []}
    comp = next((e.get('comp') for e in ev if e.get('ev') == 'run_config' and e.get('comp')), None)
    meta = metatft_boards(comp)
    import pro_traits as pro
    try:
        pros = pro.of_comp(comp) if comp else {}
    except Exception:                                     # no strong players' games read for the comp
        pros = {}
    stages = sorted({st for st in conv} | {e.get('stage') for e in ev if e.get('ev') == 'fight' and e.get('stage')},
                    key=stage_key)
    out = []
    for st in stages:
        if not st or stage_key(st) < (1, 2):
            continue
        fight = next((e for e in ev if e.get('ev') == 'fight' and e.get('stage') == st), None)
        first_plan = next((e for e in ev if e.get('ev') == 'model_call' and e.get('stage') == st
                           and e.get('kind') == 'plan'), None)
        prompt = (first_plan or {}).get('prompt') or ''
        free = re.search(r'Free rerolls: (\d+)', prompt)
        odds = re.search(r'Shop odds by cost 1-5: ([^\n]+)', prompt)
        gold0 = re.search(r'\nGold (\d+),', prompt)
        shop_part = prompt.split('\nShop:', 1)[1].split('\nWisp in shop', 1)[0] if '\nShop:' in prompt else ''
        shop = [(m.group(1).strip(), int(m.group(2))) for m in re.finditer(r'slot \d: (.+?) \((\d)g, ', shop_part)]
        acts = [e for e in ev if e.get('ev') == 'action' and e.get('stage') == st]
        r = {'stage': st, 'conversation': conv.get(st),
             'shots': [p.name for p in (run / f'{st}-start.jpg', run / f'{st}-fight.jpg') if p.exists()],
             'kind': 'carousel' if st.endswith('-4') and stage_key(st)[0] >= 2 else
                     'pve' if (stage_key(st)[0] == 1 or st.endswith('-7')) else 'pvp',
             'opened': (results.get(st) or {}).get('opened'),
             'free_rerolls_at_start': int(free.group(1)) if free else None,
             'shop_odds': odds.group(1).strip() if odds else None,
             'shop_at_start': [f"{n} ({c}g)" for n, c in shop],
             'gold_at_start': int(gold0.group(1)) if gold0 else None,
             'goal': (fight or {}).get('goal'),
             'result': (results.get(st) or {}).get('result'),
             'actions': [f"{e.get('act')}{'' if e.get('ok') else '  [FAILED: ' + str(e.get('cause') or e.get('why') or '') + ']'}"
                         for e in acts],
             'rolls': sum(int(e.get('rolls') or 0) for e in ev if e.get('ev') == 'rolldown' and e.get('stage') == st),
             'augments': [f"{e.get('name')} (card {e.get('pick')})" for e in ev if e.get('ev') == 'augment'
                          and e.get('stage') == st],
             'wisps': [wisp_line(o) for o in metrics.wisp_offers(ev, st)],
             'containers': metrics.choice_options(ev, st) or
                           [f"{e.get('name')} (option {e.get('option')} of {e.get('of')})" for e in ev
                            if e.get('ev') == 'choice_pick' and e.get('stage') == st],
             'carousel': [f"wanted {e.get('want')}, seen {e.get('seen')}" for e in ev if e.get('ev') == 'carousel'
                          and e.get('stage') == st][:1],
             'answers': answers(ev, st)}
        if fight:
            board, bench = units(fight.get('roster'), 'h'), units(fight.get('roster'), 'b')
            traits, wasted = trait_facts(board)
            idle = [i for i in fight.get('items') or [] if i]
            team = fight.get('team_size') or fight.get('level') or len(board)
            opts = trait_options([(n, s, i) for n, s, i, _ in board + bench], team, shop,
                                 r['gold_at_start'] or fight.get('gold') or 0)
            show = lambda b: ', '.join(f"{t} {n}" for t, n, _ in b) or 'no trait bonus'
            r['trait_bonuses'] = show(bonuses([(n, i) for n, _, i, _ in board]))
            r['metatft_board'] = meta.get(fight.get('level') or 0) or []
            tiers = lambda b: sum(t for *_, t in b)
            r['tiers_fielded'] = tiers(bonuses([(n, i) for n, _, i, _ in board]))
            r['tiers_best'] = max([r['tiers_fielded']] + [tiers(o[0]) for o in opts.values()])
            r['pro_traits'] = pro.line(pros.get(st) or [])
            r['trait_options'] = ([f"owned units alone: {show(opts['owned'][0])} ({', '.join(opts['owned'][1])})"]
                                  if opts.get('owned') else []) + \
                                 ([f"with {', '.join(opts['with_shop'][2])} bought from the shop: "
                                   f"{show(opts['with_shop'][0])} ({', '.join(opts['with_shop'][1])})"]
                                  if opts.get('with_shop') else [])
            r.update(level=fight.get('level'), gold=fight.get('gold'), team_size=fight.get('team_size'),
                     board=[unit_line(n, s, i) for n, s, i, _ in board],
                     bench=[unit_line(n, s, i) for n, s, i, _ in bench],
                     traits=[f"{t['trait']} {t['count']}" + (f" (active {t['active']})" if t['active'] else ' (inactive)')
                             + (f", next {t['next']} needs {t['short']}" if t['next'] else '') for t in traits],
                     wasted_units=wasted,
                     carry=None if not carry else (
                         'fielded' if any(n == carry for n, *_ in board) else
                         'on the bench' if any(n == carry for n, *_ in bench) else 'not owned'),
                     item_holders=[unit_line(n, s, i) for n, s, i, _ in board if i],
                     carry_items_elsewhere=', '.join(f"{u} {'+'.join(i)}" for u, i in held[st]['holders'].items())
                     if st in held else '',
                     fielded_without_items=[unit_line(n, s, i) for n, s, i, _ in board if not i],
                     item_effects=effects_line([(n, i) for n, _, i, _ in board]),
                     five_cost_benched=[n for n, *_ in bench if cost_of(n) == 5 and all(n != b[0] for b in board)],
                     item_bench=idle)
        out.append(r)
    return out


def wisp_line(o):
    """One Wisp the round showed: price, effect, the player's answers, and what happened to it."""
    fate = ('bought' if o['bought'] else 'rolled past by a roll, never answered' if o['rolled_past'] and not o['answers']
            else 'answered buy, NOT bought' if any(a.startswith('buy') for a in o['answers']) else 'not bought')
    return (f"{o['wisp']} ({o['cost'] if o['cost'] is not None else '?'}g; {o['effect'] or 'effect on the card'}): "
            f"answers {', '.join(o['answers']) or 'none'} -> {fate}")


def effects_line(board):
    """[(unit, items)] -> 'Sunder: none; Shred: Ionic Spark (Krug); ...' for each effect in metrics.EFFECTS."""
    have = {e: [] for e in metrics.EFFECTS}
    for unit, items in board:
        for it in items:
            for e in metrics.item_effects(it):
                have[e].append(f'{it} ({unit})')
    return '; '.join(f"{e}: {', '.join(v) or 'none'}" for e, v in have.items())


_COST = {}


def cost_of(name):
    if not _COST:
        _COST.update({k: v.get('Cost') for k, v in metrics.ref_table('champions').items()})
    try:
        return int(_COST.get(name) or 0)
    except ValueError:
        return 0


def required(r):
    """The decisions a round's audit must grade: every planning area, and each augment, carousel, Wisp and container
    choice that happened in it."""
    need = [] if r['kind'] == 'carousel' else list(AREAS)
    if r['augments']:
        need.append('augment')
    if r['kind'] == 'carousel' or r['carousel']:
        need.append('carousel')
    need += ['wisp'] * len(r['wisps'])
    need += ['container'] * len(r['containers'])
    return need


def packet(rounds, run_name, stage):
    """The text packet of one stage's rounds, for its auditor."""
    md = [f"# Game {run_name}, stage {stage}: the rounds to audit\n",
          "Every round below lists where the player's state and answers are in runs/" + run_name + "/conversation.md, "
          "what was executed, and the board's measured facts at combat start. Grade each listed decision.\n"]
    for r in rounds:
        if r['stage'].split('-')[0] != str(stage):
            continue
        a, b = r['conversation'] or ('?', '?')
        md.append(f"\n## {r['stage']} ({r['kind']})")
        md.append(f"- conversation.md lines {a}-{b}; screenshots: {', '.join(r['shots']) or 'none'}")
        md.append(f"- decisions to grade: {', '.join(required(r))}")
        if r.get('opened'):
            md.append(f"- opened with gold {r['opened'].get('gold')}, level {r['opened'].get('level')}"
                      + (f"; free rerolls {r['free_rerolls_at_start']} (they do not expire)"
                         if r['free_rerolls_at_start'] is not None else ''))
        if r.get('shop_at_start'):
            md.append(f"- shop at the start of planning: {', '.join(r['shop_at_start'])}"
                      + (f"; odds by cost 1-5 at this level: {r['shop_odds']}" if r.get('shop_odds') else ''))
        if r.get('board') is not None:
            md.append(f"- at combat start: level {r.get('level')}, gold {r.get('gold')}, team size {r.get('team_size')}")
            md.append(f"- board: {'; '.join(r['board']) or 'empty'}")
            md.append(f"- bench (owned, not fielded): {'; '.join(r['bench']) or 'empty'}")
            md.append(f"- traits: {'; '.join(r['traits']) or 'none'}")
            md.append(f"- trait bonuses on the board: {r['trait_bonuses']}")
            # (no 'strongest board' by trait count: the owner, 2026-11-03, "more traits does not mean stronger")
            md.append(f"- trait tiers fielded: {r.get('tiers_fielded')}")
            if r.get('pro_traits'):
                md.append(f"- strong players with this comp at {r['stage']}: {r['pro_traits']}")
            for b in r.get('metatft_board') or []:
                md.append(f"- MetaTFT, this comp at level {r.get('level')}, {b}")
            if r['wasted_units']:
                md.append(f"- units adding nothing to a trait: {'; '.join(r['wasted_units'])}")
            if r.get('carry'):
                md.append(f"- the comp's carry: {r['carry']}")
            if r.get('carry_items_elsewhere'):
                md.append(f"- the carry's items (the plan's item map) on other units: {r['carry_items_elsewhere']} - "
                          f"each holder is one more move when the carry is fielded")
            md.append(f"- items on: {'; '.join(r['item_holders']) or 'nobody'}; fielded without items: "
                      f"{'; '.join(r['fielded_without_items']) or 'none'}")
            md.append(f"- item bench: {', '.join(r['item_bench']) or 'empty'}")
            md.append(f"- enemy-weakening item effects on the board (items.md): {r['item_effects']}")
            if r.get('five_cost_benched'):
                md.append(f"- 5-cost champions on the bench with no copy fielded: {', '.join(r['five_cost_benched'])}")
        md.append(f"- fight goal the player set: {r['goal'] or 'not recorded (read its reasons)'}; result: "
                  f"{r['result'] or 'no player fight'}")
        md.append(f"- rerolls run this round: {r['rolls']}")
        for k in ('augments', 'carousel', 'wisps', 'containers'):
            if r[k]:
                md.append(f"- {k}: {' | '.join(r[k])}")
        md.append(f"- executed: {' | '.join(r['actions']) or 'nothing'}")
        for kind, plan, why in r['answers']:
            md.append(f"- player ({kind}): {plan}" + (f"\n  its reasons: {why}" if why else ''))
    return '\n'.join(md) + '\n'


def write_packets(run):
    run = Path(run)
    rounds = game_rounds(run)
    folder = AGENT / 'reviews' / f'{run.name}-rounds'
    folder.mkdir(parents=True, exist_ok=True)
    stages = sorted({int(r['stage'].split('-')[0]) for r in rounds})
    import stage_review
    try:
        luck = stage_review.luck(run, rounds)
    except Exception as e:                               # a packet without luck numbers says so
        luck = {}
        print(f'luck: not measured ({e.__class__.__name__}: {e})')
    for s in stages:
        (folder / f'stage-{s}.md').write_text(packet(rounds, run.name, s)
                                              + stage_review.packet_extra(run, s, rounds, luck), encoding='utf-8')
    (folder / 'rounds.json').write_text(json.dumps(rounds, ensure_ascii=False, indent=1), encoding='utf-8')
    return rounds, stages


def gaps(rounds, audit, stage=None):
    """What an audit leaves out: [text] - a round with no entry, a required decision with no grade, a grade that is
    not one of GRADES, a bad grade without a better option or a cause."""
    got = {str(x.get('round')): x for x in (audit or {}).get('rounds') or [] if isinstance(x, dict)}
    out = []
    if audit is not None and any(stage is None or r['stage'].split('-')[0] == str(stage) for r in rounds):
        for k, what in (('summary', 'what happened in the stage'),
                        ('against_strong_players', "the differences from strong players' play that mattered, and what "
                                                   "to do instead (or that there were none)")):
            if not str((audit or {}).get(k) or '').strip():
                out.append(f"'{k}' missing: {what}")
    for r in rounds:
        if stage is not None and r['stage'].split('-')[0] != str(stage):
            continue
        a = got.get(r['stage'])
        if not a:
            out.append(f"{r['stage']}: no entry")
            continue
        rows = [d for d in a.get('decisions') or [] if isinstance(d, dict)]
        have = [str(d.get('area') or '').lower() for d in rows]
        for area in set(required(r)):
            n_need, n_have = required(r).count(area), have.count(area)
            if n_have < n_need:
                out.append(f"{r['stage']}: {area} not graded" + (f" ({n_have} of {n_need})" if n_need > 1 else ''))
        for d in rows:
            g = str(d.get('grade') or '').lower()
            if g not in GRADES:
                out.append(f"{r['stage']} {d.get('area')}: grade {g!r} is not one of {', '.join(GRADES)}")
            elif g == 'uncertain' and not str(d.get('why') or '').strip():
                out.append(f"{r['stage']} {d.get('area')}: an uncertain grade needs 'why' (what the record cannot show)")
            elif g in BAD and not (d.get('better') and str(d.get('cause') or '').lower() in CAUSES):
                out.append(f"{r['stage']} {d.get('area')}: a {g} grade needs 'better' and a cause ({', '.join(CAUSES)})")
    return out


def render(rounds, audits):
    """The round-by-round report (the owner's format): '2-1' then one line per decision: area: grade - why."""
    got = {}
    for a in audits:
        for x in (a or {}).get('rounds') or []:
            if isinstance(x, dict):
                got[str(x.get('round'))] = x
    md = ['## Round by round\n']
    for r in rounds:
        a = got.get(r['stage'])
        md.append(f"### {r['stage']}" + (f" - goal {a.get('goal')}" if a and a.get('goal') else '')
                  + (f", result {r['result']}" if r.get('result') else ''))
        if not a:
            md.append('- not audited\n')
            continue
        for d in a.get('decisions') or []:
            if not isinstance(d, dict):
                continue
            line = f"- **{d.get('area')}**: {d.get('grade')} - {d.get('why') or ''}"
            if str(d.get('grade') or '').lower() in BAD:
                line += f" Better: {d.get('better')}" + (f" The player said: \"{d['player_said']}\"" if d.get('player_said') else '') \
                        + (f" ({'logical' if d.get('logical') else 'not logical'})" if d.get('logical') is not None else '') \
                        + f" [cause: {d.get('cause')}]"
            md.append(line)
        md.append('')
    return '\n'.join(md) + '\n'


def tally(rounds, audits):
    """The audit in numbers: decisions graded, bad ones by area and by cause, and the bad ones listed. -> dict."""
    rows = []
    for a in audits:
        for x in (a or {}).get('rounds') or []:
            for d in (x or {}).get('decisions') or [] if isinstance(x, dict) else []:
                if isinstance(d, dict):
                    rows.append(dict(d, round=x.get('round')))
    bad = [d for d in rows if str(d.get('grade') or '').lower() in BAD]
    by_area, by_cause = {}, {}
    for d in bad:
        w = 2 if str(d.get('grade')).lower() == 'very bad' else 1
        by_area[str(d.get('area'))] = by_area.get(str(d.get('area')), 0) + w
        by_cause[str(d.get('cause'))] = by_cause.get(str(d.get('cause')), 0) + w
    return {'graded': len(rows), 'bad': len(bad), 'very_bad': sum(1 for d in bad if str(d.get('grade')).lower() == 'very bad'),
            'by_area': by_area, 'by_cause': by_cause,
            'rows': [{k: d.get(k) for k in ('round', 'area', 'grade', 'why', 'better', 'player_said', 'logical', 'cause')}
                     for d in bad]}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--check')
    a = ap.parse_args()
    rounds, stages = write_packets(a.run)
    if a.check:
        audit = json.loads(Path(a.check).read_text(encoding='utf-8'))
        # the stage the audit file covers (stage-<s>.json): the other stages' rounds are not its gaps
        print('\n'.join(gaps(rounds, audit, audit.get('stage'))) or 'complete')
    else:
        print(f"{len(rounds)} rounds, stages {stages}: agent/reviews/{Path(a.run).name}-rounds/")


if __name__ == '__main__':
    main()
