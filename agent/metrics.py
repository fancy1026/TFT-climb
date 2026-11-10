"""Metrics (the owner's structure v3, section 7): what each game did, measured by code from its log - and, across
games, the patterns that start the Strategist.

    python agent/metrics.py runs/<id> [runs/<id> ...]     write agent/metrics/<run>.json for each game
    python agent/metrics.py --patterns                     agent/metrics/patterns.json from the latest games

Descriptive only: the numbers say what happened (the round each level came, the board's traits, XP that completed no
level), never what should have happened - that is the Reviewer's and the Strategist's to argue from them. No HP figure
is measured (the owner, 2026-10-29: "HP should not be used at all for any metric"; a fight's result is a fact of the
round, not a grade of its decisions). The Reviewer's round-by-round grades are attached to the game (attach_audit).
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
OUT = AGENT / 'metrics'
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

# the verdict on a game (the owner, 2026-10-29: "is it an unlucky game, or strategy wrong, or execution wrong"), from
# the causes of the decisions the Reviewer graded bad
VERDICT = {'strategy': 'reasoning', 'information': 'information', 'execution': 'execution'}
KINDS = ('strategy', 'execution', 'information', 'luck')
BAD_AREA = 3                    # a signal: bad decisions of one area weighing this much in a game (very bad counts 2)
CONSUMABLES = ('alpha mark', 'remover', 'reforger', 'duplicator', 'neeko', 'magnetic', 'golden', 'anvil', 'tome')
WINDOW = 5                      # patterns: the latest games looked at
PATTERN_GAMES = 3               # ... and how many of them must show it (the structure: "a pattern in 3+ games")


def round_index(stage):
    """'1-1' -> 1 ... '2-1' -> 5 ... (7 rounds a stage from stage 2), None for no stage."""
    m = re.match(r'(\d+)-(\d+)', str(stage or ''))
    if not m:
        return None
    s, r = int(m.group(1)), int(m.group(2))
    return r if s == 1 else 4 + (s - 2) * 7 + r


def is_pvp(stage):
    """A player fight: stage 2 on, not the carousel (x-4) or the PvE round (x-7)."""
    s, r = (int(x) for x in str(stage).split('-'))
    return s >= 2 and r not in (4, 7)


def interest(gold):
    return min(5, max(0, int(gold or 0)) // 10)


def events(run):
    out = []
    f = Path(run) / 'log.jsonl'
    for line in f.read_text(encoding='utf-8', errors='replace').splitlines() if f.exists() else []:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def board(roster):
    """[(name, stars, [items])] of the board part of a roster string."""
    part = (roster or '').split(' bench=[')[0]
    out = []
    for m in re.finditer(r"([A-Z?][\w'. ]*?)(\*{1,3})@h\d\d(?:\[([^\]]*)\])?", part):
        out.append((m.group(1).strip(), len(m.group(2)), [i.strip() for i in (m.group(3) or '').split(',') if i.strip()]))
    return out


def carry_holders(fights_ev, carry, item_map):
    """Where the carry's items (the plan's item map, his first) stood at each combat start: [{stage, fielded,
    holders: {unit: [items]}}] for the rounds where units other than the carry held them. Each extra holder is one
    more move when he comes (an Item Remover use, or a sale), and what the stand-in does with them meanwhile."""
    key = lambda n: ' '.join(str(n).lower().replace("'", '').replace('.', '').split())
    want = [key(i) for i in (item_map or {}).get(carry) or []]
    if not carry or not want:
        return []
    out = []
    for e in fights_ev:
        units = board(e.get('roster'))
        holders = {}
        for name, stars, items in units:
            if name == carry:
                continue
            mine = [i for i in items if any(w == key(i) or w.startswith(key(i) + ' ') for w in want)]
            if mine:
                holders.setdefault(name, []).extend(mine)
        if holders:
            out.append({'stage': e.get('stage'), 'fielded': any(n == carry for n, _, _ in units), 'holders': holders})
    return out


def ref_table(name):
    """data/reference/<name>.md tables -> {first cell: {header: cell}} (every table in the file, its own headers)."""
    out, head = {}, None
    f = ROOT / 'data' / 'reference' / f'{name}.md'
    for row in f.read_text(encoding='utf-8').splitlines() if f.exists() else []:
        if not row.startswith('|'):
            head = None
            continue
        cells = [c.strip() for c in row.strip().strip('|').split('|')]
        if head is None:
            head = cells
        elif not set(cells[0]) <= set('-: '):
            out[cells[0]] = dict(zip(head, cells))
    return out


EFFECTS = ('Sunder', 'Shred', 'Wound', 'Burn')             # the item effects that weaken enemies (items.md wording)
_ITEMS = {}


def item_effects(item):
    """The enemy-weakening effects an item applies (data/reference/items.md): e.g. Evenshroud -> ['Sunder']."""
    if not _ITEMS:
        _ITEMS.update(ref_table('items'))
    row = _ITEMS.get(item)
    if row is None and item:                             # a short name on the board record ("Ionic", "Sunfire")
        hits = [v for k, v in _ITEMS.items() if k.lower().startswith(str(item).lower())]
        row = hits[0] if len(hits) == 1 else None
    text = ' '.join((row or {}).values())
    return [e for e in EFFECTS if re.search(rf'\b{e}\b', text)]


def first_json(text):
    try:
        return json.loads((text or '').strip().splitlines()[0])
    except (ValueError, IndexError):
        return {}


def wisp_offers(ev, stage):
    """Every Wisp the round showed - in a Wisp question, in a state the player answered, or rolled past by a roll -
    with its price and effect, the player's answers and whether it was bought. -> [dict] (expert feedback x1, game
    50 6-1: Nature's Ally answered "buy" three times and never bought, which no packet showed)."""
    fx = {k: v.get('Effect') or v.get('Card text') or '' for k, v in ref_table('wisps').items()}
    offers = {}

    def offer(name, cost=None, effect=None):
        o = offers.setdefault(name, {'wisp': name, 'cost': cost, 'effect': effect or fx.get(name), 'answers': [],
                                     'bought': False, 'rolled_past': False})
        o['cost'] = o['cost'] or cost
        return o
    for e in ev:
        if e.get('stage') != stage:
            continue
        k = e.get('ev')
        if k == 'wisp_asked':
            offer(e.get('wisp'), e.get('cost'), (e.get('desc') or '').replace('card text: ', '').strip('"') or None)
        elif k == 'model_call' and e.get('kind') == 'wisp':
            m = re.search(r"Wisp '(.+?)', costs (\d+)", e.get('prompt') or '')
            if m:
                offer(m.group(1), int(m.group(2)))['answers'].append(
                    ('buy' if first_json(e.get('answer')).get('buy') else 'skip') + ' (its question)')
        elif k == 'model_call':
            line = re.search(r'\nWisp in shop: (slot .+)', e.get('prompt') or '')
            shown = dict(re.findall(r"slot (\d) '(.+?)'(?= -| \(|,|\.|$)", line.group(1))) if line else {}
            for name in shown.values():
                offer(name)
            plan = first_json(e.get('answer')).get('execution_plan') or {}
            for w in plan.get('wisps') or []:
                if isinstance(w, dict) and str(w.get('slot')) in shown:
                    offer(shown[str(w['slot'])])['answers'].append(('buy' if w.get('buy') else 'skip')
                                                                  + f" ({e.get('kind')})")
        elif k == 'rolldown':
            for w in e.get('wisps_passed') or []:
                offer(w)['rolled_past'] = True
        elif k == 'rolldown_wisp':
            offer(e.get('card') or '?', e.get('cost'))['bought'] = True
    for e in ev:                                         # bought: after every offer is known (a buy can log first)
        if e.get('stage') == stage and e.get('ev') in ('action', 'wisp_purchase') and e.get('ok'):
            act = str(e.get('act') or e.get('wisp') or '')
            for name, o in offers.items():
                if name and name in act and (e.get('ev') == 'wisp_purchase' or 'wisp' in act.lower()):
                    o['bought'] = True
    return [o for o in offers.values() if o['wisp']]


def choice_options(ev, stage):
    """The options of each choice screen (an anvil, a container) the round answered, each item's enemy-weakening
    effects, and the pick. -> [str] (x1, game 50 6-1: Evenshroud's 30% Sunder passed over for Bloodthirster)."""
    out = []
    for e in ev:
        if e.get('ev') == 'model_call' and e.get('kind') == 'choice' and e.get('stage') == stage:
            m = re.search(r'The options, as read off the cards: ([^\n]+)', e.get('prompt') or '')
            if not m:
                continue
            opts = [o.strip().rstrip('.') for o in re.split(r',\s*(?=\d+ )', m.group(1))]
            show = []
            for o in opts:
                fx = item_effects(re.sub(r'^\d+ ', '', o))
                show.append(o + (f" [{', '.join(fx)}]" if fx else ''))
            out.append(f"options {'; '.join(show)} - picked {first_json(e.get('answer')).get('name') or '?'}")
    return out


def metatft_levels(comp):
    """{level: stage} from the comp brief's levelling row ('2-5: 5, 3-2: 6 (1.1 rolls), ...')."""
    if not comp:
        return {}
    f = ROOT / 'data' / 'metatft_shots' / re.sub(r'[^A-Za-z0-9]+', '_', comp).strip('_') / 'brief.txt'
    if not f.exists():
        return {}
    m = re.search(r'Levelling \(stage: level\): ([^\n]+)', f.read_text(encoding='utf-8'))
    return {int(lv): st for st, lv in re.findall(r'(\d-\d): (\d+)', m.group(1))} if m else {}


def game(run):
    """Every number of one game (a run folder)."""
    run = Path(run)
    ev = events(run)
    try:
        from game_metrics import metrics as gm
        base = gm(str(run))
    except Exception as e:                               # an old or broken run: what the log alone gives
        base = {'error': str(e)}
    comp = base.get('comp')
    carry = next((m.group(1) for e in ev if e.get('ev') == 'fight'
                  for m in [re.search(r"carry='([^']+)'", e.get('plan') or '')] if m), None)

    # the game plan's versions (the climb's plan file, else the strategy events)
    plans = [e for e in ev if e.get('ev') == 'game_strategy']
    levels_plan = next((p['strategy'].get('levels') for p in reversed(plans) if isinstance(p.get('strategy'), dict)
                        and isinstance(p['strategy'].get('levels'), dict)), None) or {}
    item_map = next((p['strategy'].get('items') for p in reversed(plans) if isinstance(p.get('strategy'), dict)
                     and isinstance(p['strategy'].get('items'), dict)), None) or {}

    # each planning phase: the gold and level it opened with (the first read of that stage), the XP bought
    opened, xp_bought = {}, {}
    for e in ev:
        st = e.get('stage')
        if e.get('ev') in ('round', 'action') and st and st not in opened and e.get('gold') is not None:
            opened[st] = {'gold': e.get('gold'), 'level': e.get('level')}
        if e.get('ev') == 'action' and e.get('ok') and str(e.get('act', '')).startswith('level('):
            m = re.match(r'level\((\d+)\)', e['act'])
            xp_bought[st] = xp_bought.get(st, 0) + (int(m.group(1)) if m else 1)
    for e in ev:                                         # a phase with no action: the round event's screen line
        if e.get('ev') == 'round' and e.get('stage') and e['stage'] not in opened:
            m = re.search(r'L(\d+) .*?g=(\d+)', e.get('screen') or '')
            if m:
                opened[e['stage']] = {'level': int(m.group(1)), 'gold': int(m.group(2))}

    # each fight's result - the only use of the HP reads: a drop to the next combat start (0 after the last fight of a
    # lost game) is a loss. The result is a fact of the round, never a grade (the owner, 2026-10-29)
    hp = [(r['stage'], r['hp']) for r in base.get('rounds') or [] if r.get('hp') is not None]
    hp_next = {s: hp[i + 1][1] for i, (s, _) in enumerate(hp[:-1])}
    if hp and base.get('placement') not in (None, 1):
        hp_next.setdefault(hp[-1][0], 0)
    hp_at = dict(hp)
    said = {e.get('stage'): e.get('result') for e in ev if e.get('ev') == 'fight_result' and e.get('result')}
    try:
        from rounds import trait_facts, units
    except Exception:                                    # a stand-in module in a test
        trait_facts = units = None
    fights, seen = [], set()
    for e in ev:
        st = e.get('stage')
        if e.get('ev') != 'fight' or not st or st in seen or round_index(st) is None or round_index(st) < 5:
            continue
        seen.add(st)
        b = board(e.get('roster'))
        h = hp_at.get(st, e.get('hp'))
        result = {'lose': 'loss', 'win': 'win'}.get(said.get(st)) or (
            None if h is None or hp_next.get(st) is None else 'loss' if h > hp_next[st] else 'win')
        size = e.get('team_size')
        f = {'stage': st, 'pvp': is_pvp(st), 'gold': e.get('gold'), 'level': e.get('level'),
             'xp': e.get('xp'), 'board': len(b), 'team_size': size,
             'empty': (size - len(b)) if isinstance(size, int) and size > len(b) else 0,
             'goal': e.get('goal'), 'result': result, 'opened': opened.get(st), 'xp_bought': xp_bought.get(st, 0)}
        if units:
            on, off = units(e.get('roster'), 'h'), units(e.get('roster'), 'b')
            f['wasted_units'] = trait_facts(on)[1]
            f['bench'] = [f"{n}{'*' * s}" for n, s, _, _ in off]
            f['components_idle'] = [i for i in e.get('items') or [] if i and not any(c in i.lower() for c in CONSUMABLES)]
        fights.append(f)

    # the round each level was first seen at a combat start, against the plan and MetaTFT
    reached = {}
    for f in fights:
        if f['level'] and f['level'] not in reached:
            reached[f['level']] = f['stage']
    meta = metatft_levels(comp if isinstance(comp, str) else None)
    vs = []
    plan_by_level = {}
    for st, lv in levels_plan.items():
        try:
            lv = int(lv)
        except (TypeError, ValueError):
            continue
        if lv not in plan_by_level or round_index(st) < round_index(plan_by_level[lv]):
            plan_by_level[lv] = st
    for lv in sorted(set(plan_by_level) | set(meta) | set(reached)):
        got = reached.get(lv)
        row = {'level': lv, 'reached': got, 'plan': plan_by_level.get(lv), 'metatft': meta.get(lv)}
        for k in ('plan', 'metatft'):
            if row[k] and got:
                row[f'rounds_vs_{k}'] = round_index(got) - round_index(row[k])   # + later, - earlier
        vs.append(row)

    interest_lost = [{'stage': f['stage'], 'opened': f['opened']['gold'], 'at_combat': f['gold'],
                      'steps': interest(f['opened']['gold']) - interest(f['gold'])}
                     for f in fights if f['opened'] and f['gold'] is not None
                     and interest(f['opened']['gold']) > interest(f['gold'])]
    no_level = [{'stage': f['stage'], 'xp_buys': f['xp_bought'], 'level': f['level']} for f in fights
                if f['xp_bought'] and f['opened'] and f['opened'].get('level') == f['level']]

    def carry_items(at):
        f = next((f for f in fights if round_index(f['stage']) >= round_index(at)), None)
        if not f or not carry:
            return None
        e = next(e for e in ev if e.get('ev') == 'fight' and e.get('stage') == f['stage'])
        for name, stars, items in board(e.get('roster')):
            if name == carry:
                return {'stage': f['stage'], 'carry': carry, 'stars': stars, 'items': items}
        return {'stage': f['stage'], 'carry': carry, 'fielded': False, 'items': []}

    # completed items held at the last combat start that the plan's item map does not name (the roster shortens names:
    # "Guinsoo's" for Guinsoo's Rageblade): components built into items off the map
    key = lambda n: ' '.join(str(n).lower().replace("'", '').replace('.', '').split())
    try:
        from tft_player.items import COMPONENTS
        comps = {key(c) for c in COMPONENTS}
    except Exception:
        comps = set()
    wanted = [key(i) for items in item_map.values() for i in (items if isinstance(items, list) else [items])]
    named = lambda short: any(w == key(short) or w.startswith(key(short) + ' ') for w in wanted)
    last = next((e for e in reversed(ev) if e.get('ev') == 'fight'), None)
    off_map = []
    for name, stars, items in board(last.get('roster')) if last and wanted else []:
        for i in items:
            k = key(i)
            if k in comps or k in ('unknown', '?') or any(c in k for c in CONSUMABLES) or any(c.startswith(k + ' ') for c in comps) or named(i):
                continue
            off_map.append({'stage': last.get('stage'), 'item': i, 'on': name})
    if item_map:
        carry = next(iter(item_map))                     # the plan's item map names the carry first

    def carry_final():                                   # the carry's stars at the last combat start, board or bench
        e = next((e for e in reversed(ev) if e.get('ev') == 'fight'), None)
        if not e or not carry:
            return None
        found = [len(m.group(1)) for m in re.finditer(re.escape(carry) + r"(\*{1,3})@", e.get('roster') or '')]
        return {'stage': e.get('stage'), 'carry': carry, 'stars': max(found, default=0)}

    def five_benched():
        """Fights from stage 5 with a 5-cost champion on our bench and no copy of it fielded - a spare copy held for a
        2-star is not one (x1, game 50: Elder Dragon benched 5-2 to 6-2)."""
        cost = {k: v.get('Cost') for k, v in ref_table('champions').items()}
        out = []
        for e in ev:
            if e.get('ev') == 'fight' and round_index(e.get('stage') or '1-1') >= round_index('5-1'):
                bench = (e.get('roster') or '').split(' bench=[', 1)[1:]
                names = re.findall(r"([A-Z][\w'. ]*?)\*{1,3}@b\d", bench[0]) if bench else []
                fielded = {n for n, _, _ in board(e.get('roster'))}
                fives = sorted({n.strip() for n in names if cost.get(n.strip()) == '5' and n.strip() not in fielded})
                if fives:
                    out.append(f"{e['stage']} {'+'.join(fives)}")
        return out

    revisions = [{'stage': p.get('stage'), 'why': (p.get('strategy') or {}).get('why')} for p in plans[1:]]
    held = sum(1 for e in ev if e.get('ev') == 'game_plan_held')
    out = {'run': run.name, 'comp': base.get('comp'), 'placement': base.get('placement'),
           'last_stage': base.get('last_stage'), 'carry': carry,
           'fights': fights, 'levels': vs, 'interest_lost': interest_lost, 'xp_no_level': no_level,
           'empty_slots': [{'stage': f['stage'], 'empty': f['empty']} for f in fights if f['empty']],
           'wasted_units': [{'stage': f['stage'], 'traits': f['wasted_units']} for f in fights if f.get('wasted_units')],
           'carry_benched': [f['stage'] for f in fights if carry and any(x.rstrip('*') == carry
                                                                          for x in f.get('bench') or [])],
           'carry_items': {'4-2': carry_items('4-2'), '5-1': carry_items('5-1')}, 'carry_final': carry_final(),
           'carry_holders': carry_holders([e for e in ev if e.get('ev') == 'fight'], carry, item_map),
           'items_off_map': off_map, 'item_map': item_map,
           'wisps': [{'stage': st, 'wisp': o['wisp'], 'cost': o['cost'], 'bought': o['bought'],
                      'answered_buy': any(a.startswith('buy') for a in o['answers']), 'rolled_past': o['rolled_past']}
                     for st in sorted({e.get('stage') for e in ev if e.get('stage')}, key=round_index)
                     for o in wisp_offers(ev, st)],
           'five_cost_benched': five_benched(),
           'plans': {'late': base.get('late'), 'next_round_late': sum(1 for e in ev if e.get('ev') == 'next_round_late'),
                     'revisions': revisions, 'held': held},
           'harness': {k: base.get(k) for k in ('failed_steps', 'rejected', 'sells_failed', 'actions_failed')},
           'audit': None}
    old = OUT / f'{run.name}.json'
    if old.exists():                                     # the Reviewer's round-by-round grades stay with the game
        try:
            out['audit'] = json.loads(old.read_text(encoding='utf-8')).get('audit')
        except ValueError:
            pass
    return out


def write(run):
    OUT.mkdir(exist_ok=True)
    m = game(run)
    (OUT / f"{m['run']}.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    return m


def attach_audit(run_name, tally, reasons=None):
    """The Reviewer's round-by-round grades (agent/rounds.py tally: decisions graded, the bad ones by area and by
    cause, each listed) and its ranked placement reasons into the game's metrics. The verdict is the kind of its
    largest placement reason; without one, the cause most of the bad decisions share (luck when none was bad).
    -> the audit as stored."""
    f = OUT / f'{run_name}.json'
    m = json.loads(f.read_text(encoding='utf-8')) if f.exists() else write(ROOT / 'runs' / run_name)
    t = dict(tally or {})
    reasons = [r for r in reasons or [] if isinstance(r, dict)]
    counts = {k: (t.get('by_cause') or {}).get(c, 0) for k, c in VERDICT.items()}
    first = str((reasons[0] if reasons else {}).get('kind') or '').strip().lower()
    main = first if first in KINDS else (max(counts, key=counts.get) if any(counts.values()) else 'luck')
    t.update(verdict=dict(counts, main=main), reasons=reasons)
    m['audit'] = t
    f.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    return t


def signals(m):
    """The pattern signals one game shows: {name: evidence}. Each is a measured fact or the Reviewer's grades; the
    names are what the Strategist looks for across games. No HP."""
    s = {}
    a = m.get('audit') or {}
    for area, w in (a.get('by_area') or {}).items():
        if w >= BAD_AREA:
            rounds = [r['round'] for r in a.get('rows') or [] if str(r.get('area')) == area]
            s[f'bad:{area}'] = f"{len(rounds)} round(s): {', '.join(map(str, rounds[:8]))}"
    if m.get('wasted_units'):
        s['units_adding_nothing'] = '; '.join(f"{x['stage']} {x['traits'][0].split(':')[0]}" for x in m['wasted_units'][:6])
    if m.get('carry_benched'):
        s['carry_benched'] = ', '.join(m['carry_benched'][:8])
    last = (m.get('fights') or [None])[-1]
    if last and (m.get('placement') or 1) > 1 and (last.get('gold') or 0) >= 20:
        s['died_with_gold'] = f"{last['stage']} {last['gold']}g"
    if m.get('xp_no_level'):
        s['xp_no_level'] = ', '.join(f"{x['stage']} x{x['xp_buys']}" for x in m['xp_no_level'])
    if m.get('empty_slots'):
        s['empty_board_slot'] = ', '.join(f"{x['stage']} -{x['empty']}" for x in m['empty_slots'])
    c = (m.get('carry_items') or {}).get('5-1')
    if c and len(c.get('items') or []) < 3:
        s['carry_items_short_at_5_1'] = f"{c.get('carry')} {len(c.get('items') or [])} items at {c.get('stage')}"
    cf = m.get('carry_final')
    if cf and cf.get('stars', 0) < 2:                    # each game's review may call it luck; across games it is a pattern
        s['carry_not_upgraded'] = f"{cf['carry']} {cf['stars']}-star at {cf['stage']}, place {m.get('placement')}"
    split = [h for h in m.get('carry_holders') or [] if len(h['holders']) >= 2]
    if split:                                            # his items spread over stand-ins: a move each when he comes
        s['carry_items_split'] = '; '.join(f"{h['stage']} " + ', '.join(f"{u} {'+'.join(i)}" for u, i in h['holders'].items())
                                           for h in split[:4])
    lost = [f"{w['stage']} {w['wisp']}" for w in m.get('wisps') or [] if w['answered_buy'] and not w['bought']]
    if lost:                                             # the player said buy and the harness never bought it
        s['wisp_not_bought'] = ', '.join(lost[:6])
    if m.get('five_cost_benched'):
        s['five_cost_benched'] = ', '.join(m['five_cost_benched'][:6])
    if m.get('items_off_map'):
        s['items_off_map'] = ', '.join(f"{x['item']} on {x['on']}" for x in m['items_off_map'][:5])
    if ((m.get('plans') or {}).get('late') or 0) >= 3:
        s['late_plans'] = f"{m['plans']['late']} late"
    # (no signal for a level off the plan's forecast: a level bought earlier or later needs no revision - the owner,
    # 2026-10-28; the rounds stay in the metrics as numbers)
    return s


def patterns(n=WINDOW, need=PATTERN_GAMES):
    """The latest n climb games (agent/games.jsonl order): each signal and the games showing it; `flagged` the
    ones in `need` or more of them. -> written to agent/metrics/patterns.json."""
    rows = [json.loads(l) for l in (AGENT / 'games.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] \
        if (AGENT / 'games.jsonl').exists() else []
    runs = [r['run'] for r in rows if r.get('run') and (ROOT / 'runs' / r['run']).exists()][-n:]
    seen = {}
    for run in runs:
        f = OUT / f'{run}.json'
        m = json.loads(f.read_text(encoding='utf-8')) if f.exists() else write(ROOT / 'runs' / run)
        for name, why in signals(m).items():
            seen.setdefault(name, []).append({'run': run, 'evidence': why})
    out = {'games': runs, 'window': n, 'need': need, 'signals': seen,
           'flagged': sorted(k for k, v in seen.items() if len(v) >= need)}
    OUT.mkdir(exist_ok=True)
    (OUT / 'patterns.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--patterns', action='store_true')
    a = ap.parse_args()
    for r in a.runs:
        m = write(r)
        print(m['run'], 'place', m['placement'], '| signals:',
              json.dumps(signals(m), ensure_ascii=False))
    if a.patterns:
        p = patterns()
        print('flagged in', p['need'], 'of', len(p['games']), 'games:', ', '.join(p['flagged']) or 'none')


if __name__ == '__main__':
    main()
