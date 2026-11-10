"""Round cases: a problem seen in one of our games, kept as that round's recorded question and a check of what should
come out (the owner, 2026-11-10, after game 81: every Reviewer finding and owner point becomes a case from the game it
happened in; a fix lands only when its case flips; failing cases stay on one list until they pass; the Judge decides
stage texts on them). Each case replays one real round - the game's own state - through the current code. They are
not unit tests (the owner's 11-07 rule stands): the owner asked for the fixes to be shown on the game's real cases.

A case (agent/round_cases/cases.jsonl):
  id, run, game, comp, round, call (runs/<run>/calls/<call>-prompt.txt: the question that made the round's decision),
  mode - 'harness': the game's own recorded answer through the current plan check, comp rules and executor;
         'decision': the question (with what the current version adds to it: the stage strategy in use, the trait
         tiers each unit would complete) asked again of the game's model in a fresh conversation with the game's
         digest, SAMPLES times, each answer through the current harness - it passes on a majority,
  expect (plain words), check (a Python expression over the result fields, FIELDS), setup (optional: {"bench_add":
  ["Taric"]} a unit put on the bench first, {"board_full": true} a drop refused on a full board this phase), source
  (who found it: "owner x31.2", "quick check of game 85"), request (the Editor request it belongs to), cost (1-3: what
  the problem cost - the Editor's order), status ('open'; 'retired' with why), shown (the check on the round as played:
  the recorded answer through the code that played it - it must fail there, else the check does not catch the problem).

The last result of each case on the current code: agent/round_cases/results.json; a decision case's answers:
agent/round_cases/answers/<id>.json (re-used, free, when only the harness changed).

    python agent/round_cases.py run [ids] [--ask] [--failing] [--json]   the cases on the current code (--ask: decision
                                                                         cases asked again; else their saved answers)
    python agent/round_cases.py try --run <run> --call <n> --round 3-2 [--check "..."] [--mode decision] [--setup '{}']
                                                                         the result fields of a round, and a check on them
    python agent/round_cases.py add --run <run> --call <n> --round 3-2 --mode harness --expect "..." --check "..."
                                    --source "..." [--request e262] [--cost 2] [--setup '{}']
    python agent/round_cases.py list [--failing]
    python agent/round_cases.py retire <id> --why "..."
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
HERE = AGENT / 'round_cases'
CASES = HERE / 'cases.jsonl'
RESULTS = HERE / 'results.json'
ANSWERS = HERE / 'answers'
CACHE = Path(tempfile.gettempdir()) / 'tft-round-cases'     # the code of earlier commits, for the round as played
PY = sys.executable
SAMPLES = 3
ASK_WORKERS = 3

FIELDS = {
    'round': "the round, '4-1'", 'level': 'level at the question', 'gold': 'gold at the question',
    'board': "names of the champions fielded after the plan and the rules (owned units, and units the plan buys and "
             "places) - e.g. 'Taric' in board",
    'stars': '{name: star} of the fielded units',
    'traits': "{trait: count} of the fielded units, each champion once (the game's count)",
    'steps': "{trait: the active step} - e.g. steps.get('Riftbeast') == 7",
    'no_tier': 'fielded units outside the comp trait that complete no trait tier with the rest',
    'idle': "fielded units whose leaving changes no active step (a Riftbeast past a step, a unit whose traits stay "
            "inactive, a second copy)",
    'second_copies': 'names fielded twice', 'bench': 'names on the bench after the plan',
    'bought': 'names bought this round (the plan and the executor: auto-buy, level-8 adds)',
    'sold': 'names sold (the plan and the executor)', 'level_ups': "XP buys the plan orders (0 when none)",
    'level_held': 'the executor holds the plan\'s XP buys this round (True/False; None without XP buys)',
    'rolls': 'the plan or the executor rolls this round', 'wisp_waits': "the round's Wisp roll waits for the plan's buys",
    'executor': "the executor's own steps, in order (strings)", 'plan_steps': "the plan's steps (strings)",
    'notes': "the comp rules' notes on the plan (strings)", 'rejected': 'the plan check\'s errors (strings)',
    'swap_target': 'setup board_full: the unit the deployment swaps out (name), or None',
    'text': "the model's answer (raw)", 'answered': 'an answer was parsed into a plan',
    'has(name)': 'name in board',
}


# ----------------------------------------------------------------------------------------------- store

def load(status='open'):
    rows = [json.loads(l) for l in CASES.read_text(encoding='utf-8').splitlines() if l.strip()] if CASES.exists() else []
    return [c for c in rows if status is None or c.get('status', 'open') == status]


def _write(rows):
    HERE.mkdir(parents=True, exist_ok=True)
    CASES.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in rows), encoding='utf-8')


def results():
    try:
        return json.loads(RESULTS.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _save_results(res):
    HERE.mkdir(parents=True, exist_ok=True)
    have = results()
    have.update(res)
    RESULTS.write_text(json.dumps(have, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def saved_answers(cid):
    try:
        return json.loads((ANSWERS / f'{cid}.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def head():
    return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def game_row(run):
    try:
        for line in (AGENT / 'games.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get('run') == run:
                return r
    except (OSError, ValueError):
        pass
    return {}


def stage_of(rnd):
    """'4-7' -> '4'; stages 5-7 share '5' (agent/stages.py)."""
    s = int(str(rnd).split('-')[0])
    return str(min(max(s, 1), 5))


# ----------------------------------------------------------------------------------------------- the code

_BOOT = {}


def boot(root=None):
    """Import the code under test: the working tree, or an earlier commit's tft_player (`root`, from code_root)."""
    if _BOOT:
        return
    os.environ.setdefault('TFT_CLIMB', '1')
    os.environ['TFT_NO_KILL_SWITCH'] = '1'
    sys.path.insert(0, str(Path(root) if root else ROOT))
    sys.path.append(str(AGENT))
    import tft_player  # noqa: F401
    _BOOT['root'] = root


def code_root(commit):
    """An earlier commit's tft_player and data, extracted once to the temp folder -> its folder (None: the working
    tree's tft_player is that commit's)."""
    commit = str(commit or '').replace('+dirty', '')
    if not commit:
        return None
    same = subprocess.run(['git', 'diff', '--quiet', commit, 'HEAD', '--', 'tft_player'], cwd=ROOT).returncode == 0
    dirty = subprocess.run(['git', 'diff', '--quiet', '--', 'tft_player'], cwd=ROOT).returncode != 0
    if same and not dirty:
        return None
    full = subprocess.run(['git', 'rev-parse', commit], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = CACHE / full[:12]
    if (out / 'tft_player' / '__init__.py').exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    paths = [p for p in ('tft_player', 'data', 'agent/comps', 'agent/expert', 'agent/memory', 'agent/config.json',
                         'agent/stages.py')
             if subprocess.run(['git', 'cat-file', '-e', f'{full}:{p}'], cwd=ROOT).returncode == 0]
    tar = out / 'code.tar'
    subprocess.run(['git', 'archive', '--format=tar', '-o', str(tar), full, *paths], cwd=ROOT, check=True)
    subprocess.run(['tar', '-xf', str(tar), '-C', str(out)], check=True)
    tar.unlink()
    for old in sorted((p for p in CACHE.iterdir() if p.is_dir() and p != out), key=lambda p: p.stat().st_mtime)[:-3]:
        import shutil                                        # keep the last few commits' code (folders made here)
        shutil.rmtree(old, ignore_errors=True)
    return out


# ----------------------------------------------------------------------------------------------- the round

LINE = re.compile(r'^\s*\[id (\d+)\] (.+?) (\*+) \(.*?\) at (hex \((\d),(\d)\)|bench slot (\d))\s*(items \[(.*?)\]|no items)?')


def prompt_of(case):
    return (ROOT / 'runs' / case['run'] / 'calls' / f"{int(case['call']):04d}-prompt.txt").read_text(encoding='utf-8')


def recorded_answer(case):
    try:
        f = ROOT / 'runs' / case['run'] / 'calls' / f"{int(case['call']):04d}-result.json"
        return json.loads(f.read_text(encoding='utf-8')).get('result') or ''
    except (OSError, ValueError):
        return ''


def state_of(prompt, rnd):
    """The round's state as the question gave it."""
    from tft_player.catalog import CATALOG
    from tft_player.state import Owned, Roster
    st = SimpleNamespace(stage=rnd[:3], timer=30, gold=None, level=None, xp=None, items=[None] * 20,
                         shop=[None] * 5, shop_raw=[''] * 5, auto_buy={}, hp=None)
    m = re.search(r'^HP (\d+)\.', prompt, re.M)
    if m:
        st.hp = int(m.group(1))
    m = re.search(r'Stage (\d-\d), planning phase, ~(\d+)s left', prompt)
    if m:
        st.stage, st.timer = m.group(1), int(m.group(2))
    m = re.search(r'Gold (\d+), level (\d+), xp \((\d+), (\d+)\)', prompt)
    if m:
        st.gold, st.level, st.xp = int(m.group(1)), int(m.group(2)), (int(m.group(3)), int(m.group(4)))
    r, seen = Roster(), set()
    for line in prompt.splitlines():
        lm = LINE.match(line)
        if not lm or int(lm.group(1)) in seen:
            continue
        seen.add(int(lm.group(1)))
        loc = ('board', int(lm.group(5)), int(lm.group(6))) if lm.group(5) else ('bench', int(lm.group(7)))
        name = lm.group(2) if CATALOG.unit(lm.group(2)) else None
        u = Owned(name, len(lm.group(3)), loc, ours=True, id=int(lm.group(1)))
        u.items = [i.strip() for i in (lm.group(9) or '').split(',') if i.strip()]
        u.sure, u.items_read = bool(name), True
        if '[MISSING' in line:
            u.missing = 'no unit seen at its hex'
        r.units.append(u)
    st.roster = r
    line = next((l for l in prompt.splitlines() if l.startswith('Item bench:')), '')
    for k, name in re.findall(r"slot (\d+): ([A-Z][A-Za-z'\.\- ]*?)(?= \(| \[|, slot |$)", line):
        if int(k) < 20:
            st.items[int(k)] = name.strip()
    for k, body in re.findall(r'^  slot (\d): (.+)$', prompt.split('\nShop:\n', 1)[-1].split('\nWisp in shop', 1)[0],
                              re.M):
        w = re.match(r"WISP '([^']+)'", body)
        if w:
            st.shop_raw[int(k)] = w.group(1)
        else:
            n = body.split(' (')[0]
            st.shop[int(k)] = n if CATALOG.unit(n) else None
            st.shop_raw[int(k)] = n
    m = re.search(r'Auto-buy: (.+?)(?:\n|$)', prompt)
    for n, s in re.findall(r"([A-Z][A-Za-z' ]+?) to (\d)\*", m.group(1) if m else ''):
        st.auto_buy[n.strip()] = int(s)
    return st


def setup(case, st):
    from tft_player.state import Owned
    for k, name in enumerate((case.get('setup') or {}).get('bench_add') or []):
        free = [i for i in range(9) if not any(u.loc == ('bench', i) for u in st.roster.units)]
        if not free:
            break
        u = Owned(name, 1, ('bench', free[0]), ours=True, id=990 + k)
        u.sure, u.items_read, u.items = True, True, []
        st.roster.units.append(u)


def modernize(q, st, rnd):
    """The recorded question with what the current version sends: the stage strategy in use (a stage's first message
    carries it), the trait tiers each shop unit would complete, the trait arithmetic of the fielded and bench units.
    Lines the question already has are not added again."""
    from tft_player import coach, planner
    from tft_player.catalog import CATALOG
    tier_note, cost_lines = getattr(planner, 'tier_note', None), getattr(planner, 'bench_cost_lines', None)
    out, shop = [], False
    for line in q.splitlines():
        if line.startswith('Shop:'):
            if cost_lines and 'Benching one fielded unit' not in q:
                out += cost_lines(st.roster, st.level)
            shop = True
        elif shop and not line.startswith('  slot'):
            shop = False
        m = re.match(r'^  slot \d: ([A-Z][^(]+?) \(', line) if shop else None
        if tier_note and m and CATALOG.unit(m.group(1).strip()) and ' - fielded' not in line:
            line += tier_note(st.roster.board(), m.group(1).strip())
        out.append(line)
    strat = coach.stage_strategy_text(rnd[0])
    q = '\n'.join(out)
    if q.startswith('DECISION: augment'):
        q = modern_augment(q)
    return (strat + '\n\n' if strat else '') + q


def modern_augment(q):
    """An augment question with the boards and the answer format the current version sends (comp_reach.board_lines,
    coach.augment_explain - x32.7) in place of the recorded ones; unchanged on code without them."""
    from tft_player import coach
    comp = coach.forced_comp()
    try:
        from tft_player.comp_reach import board_lines
    except ImportError:
        return q
    if comp is None:
        return q
    m = re.search(r'^(?:Target final board of |The level-\d board - ).*?its traits: [^\n]*\n', q, re.M | re.S)
    if m:
        from collections import Counter
        from tft_player.catalog import CATALOG
        traits = Counter(t for u in comp.units for t in (CATALOG.unit(u).traits if CATALOG.unit(u) else ()))
        boards = board_lines(comp, comp.units)
        new = (boards + (f"MetaTFT's final board of {comp.name} (when everything goes right): " if boards else
                         f'Target final board of {comp.name} (MetaTFT): ')
               + f'{", ".join(comp.units)}; its traits: ' + ', '.join(f'{t} {n}' for t, n in traits.most_common())
               + '.\n')
        q = q[:m.start()] + new + q[m.end():]
    # an explanation written by the old version and repeated in the question ("did not change: your explanation ...
    # stands") is dropped: every card is explained again under the current boards
    kept = re.search(r'^Cards? \d+(?: and card \d+)* did not change: your explanation of (?:it|them) from the last question stands'
                     r' - do not write it again:\n\n.*?\n\n(?=Before deciding, explain only)', q, re.M | re.S)
    if kept:
        q = q[:kept.start()] + q[kept.end():]
        q = re.sub(r'Before deciding, explain only the new [^(]*\(just rerolled\)', 'Before deciding, explain every '
                   'offered card', q, count=1)
    if m and 'The level-' in q:
        q = q.replace('  Target final board: +15% Damage Amp', '  Level-8 board: +15% Damage Amp')
    old = re.search(r'^COMBAT: the exact bonuses on your current board and on the [^\n]*$', q, re.M)
    fmt = re.search(r'^COMBAT: [^\n]*$', coach.augment_explain('2-1', comp), re.M)
    if old and fmt:
        q = q[:old.start()] + fmt.group(0) + q[old.end():]
    return q


def digest(run, n):
    """What a new conversation of this game is given before call n (tools/owner_replay.py's method)."""
    from tft_player import coach
    coach._MEMO.clear()
    calls = ROOT / 'runs' / run / 'calls'
    for k in range(1, n):
        f = calls / f'{k:04d}-prompt.txt'
        if f.exists():
            coach._remember(prompt=f.read_text(encoding='utf-8'))
        try:
            coach._remember(answer=json.loads((calls / f'{k:04d}-result.json').read_text(encoding='utf-8')).get('result')
                            or '')
        except (OSError, ValueError):
            coach._remember(answer='')
    d = coach.stage_digest('2', with_state=False)
    first = d.split('\n\n', 1)
    return ('NEW CONVERSATION - this game continues in a new conversation; the earlier one is not repeated here; this '
            'is what you need from it.\n\n' + first[1]) if len(first) == 2 else d


def player_stub(st):
    """A Player with only what the executor's own rules read: the round's roster, item bench and orders."""
    from tft_player.player import Player
    p = Player.__new__(Player)
    p.roster, p.bench_items, p.round_plan, p.handover_rp = st.roster, list(st.items), None, None
    p.brain = SimpleNamespace(protected=set(), plan=None, status={}, failed={})
    p.wisps_bought, p.auto_buy, p.auto_sell, p.auto_tries = {}, dict(st.auto_buy), False, {}
    p.wisp_flag = {'combat': 'buy', 'economy': 'skip'} if st.stage and st.stage >= '4-2' else None
    p._wisp_flag_set = bool(p.wisp_flag)
    p.last_level, p._level_floor = st.level, 1
    p.plan_pending, p.item_finals, p.named_next = False, [], set()
    p.log = lambda **k: None
    p.news_event = lambda *a, **k: None
    p.handover_action = lambda scr: 'SUBPLAN'
    p.reader = SimpleNamespace(shop_cost=lambda frame, i: None)
    return p


def executor_steps(st, plan):
    """The executor's own steps with no model decision, in its order, then the round plan's: [(kind, what)] twice."""
    from tft_player.catalog import CATALOG
    from tft_player.player import Player
    scr = SimpleNamespace(stage=st.stage, timer=st.timer, gold=st.gold, level=st.level, shop=list(st.shop),
                          shop_raw=list(st.shop_raw), shop_visible=True, augments=None, items=list(st.items),
                          team_size=st.level, hp=getattr(st, 'hp', None), xp=st.xp, xp_per_buy=4)
    p = player_stub(st)
    has = lambda name: callable(getattr(Player, name, None))     # noqa: E731  (an earlier commit may lack a rule)
    steps = []
    if has('riftbeast_shop'):
        Player.riftbeast_shop(p, scr)
    a = Player.flag_wisp(p, scr, None) if has('flag_wisp') else None
    if a:
        steps.append(('wisp', a.why))
        p.wisps_bought[st.stage] = 1
        scr.shop_raw[a.arg] = ''
    a = Player.outside_comp_sale(p, scr) if has('outside_comp_sale') else None
    while a and len(steps) < 12:
        steps.append(('sell', f'{a.why.split(":")[0]}: sell {a.arg.name} (unit {a.arg.id})'))
        p.roster.units = [u for u in p.roster.units if u is not a.arg]
        a = Player.outside_comp_sale(p, scr)
    if has('level_adds_buy'):
        Player.level_adds_buy(p, scr)
    for i, n in enumerate(scr.shop):
        if n and CATALOG.unit(n) and CATALOG.unit(n).cost <= (scr.gold or 0) and p.auto_need(n) > 0:
            steps.append(('buy', f'auto-buy {n}'))
            scr.gold -= CATALOG.unit(n).cost
            scr.shop[i], scr.shop_raw[i] = None, ''
    if has('carry_arrival_action') and Player.carry_arrival_action(p, scr) == 'SUBPLAN' and p.round_plan:
        sp = p.round_plan.p
        steps += [('sell', f'carry arrival: sell unit {i}') for i in sp.get('sell') or []]
        steps += [('move', f"carry arrival: unit {e.get('id')} -> {tuple(e.get('hex'))}") for e in sp.get('board') or []]
        steps += [('give', f"carry arrival: {e.get('component')} -> unit {e.get('id')}") for e in sp.get('items') or []]
        p.round_plan, p.handover_rp = None, None
    a = Player.executor_level(p, scr) if has('executor_level') else None
    if a:
        steps.append(('xp', a.why))
    a = Player.executor_roll(p, scr) if has('executor_roll') else None
    if a:
        steps.append(('roll', a.why))
    elif has('wisp_must') and Player.wisp_must(p, scr) and not any(scr.shop_raw[i] and not scr.shop[i]
                                                                      for i in range(5)):
        steps.append(('wait', "the round's Wisp roll waits for the plan's buys"))
    model = []
    if plan:
        unit = {u.id: u for u in st.roster.units}
        name = lambda i: f'{unit[i].name}{"*" * (unit[i].star or 1)} (unit {i})' if i in unit else f'unit {i}'  # noqa
        model += [('sell', f'plan: sell {name(i)}') for i in plan.get('sell') or [] if isinstance(i, int)]
        model += [('buy', f'plan: buy {b}') for b in plan.get('buy') or [] if isinstance(b, str)]
        if plan.get('level_ups'):
            model.append(('xp', f"plan: {plan['level_ups']} XP buys"))
        for e in plan.get('board') or []:
            u = unit.get(e.get('id')) if isinstance(e, dict) else None
            if isinstance(e, dict) and u is None and e.get('name'):
                u = next((x for x in unit.values() if x.name == e['name'] and not x.on_board), None)
                if u is None:
                    continue
            if isinstance(e, dict) and u is not None and u.loc[1:] != tuple(e.get('hex') or ()):
                model.append(('move', f"plan: {name(u.id)} -> {tuple(e.get('hex') or ())}"))
        model += [('give', f"plan: {e.get('component')} -> {name(e.get('id'))}") for e in plan.get('items') or []
                  if isinstance(e, dict)]
        for k in ('roll_n', 'fast_rr', 'roll'):
            if plan.get(k):
                model.append(('roll', f'plan: {k} {json.dumps(plan[k])[:120]}'))
    return steps, model


def _steps(units):
    from tft_player.catalog import CATALOG
    from tft_player.state import board_traits
    n = board_traits(units)
    out = {t: max((x for x in CATALOG.traits.get(t) or () if x <= c), default=0) for t, c in n.items()}
    return {t: x for t, x in out.items() if x}


def _gained(base, u):
    before, after = _steps(base), _steps(list(base) + [u])
    return [f'{t} {after[t]}' for t in sorted(after) if after[t] > before.get(t, 0)]


def outcome(case, st, answer):
    """The answer through the code under test -> the result fields (FIELDS)."""
    from tft_player import planner
    from tft_player.catalog import CATALOG
    rnd = case['round'][:3]
    # a question with no game state (an augment, a Wisp): its answer is no round plan - only its text is checked
    plan = planner.parse(answer) if answer and st.level is not None else None
    errs = []
    if plan:
        try:
            errs = planner.validate(plan, st.roster, list(st.items), planner.PlanNames(st.roster.units),
                                    level=(st.level, st.xp), gold=st.gold, stage=rnd, hp=st.hp)
        except TypeError:                         # (an earlier commit: no hp, or no level and gold)
            try:
                errs = planner.validate(plan, st.roster, list(st.items), planner.PlanNames(st.roster.units),
                                        level=(st.level, st.xp), gold=st.gold, stage=rnd)
            except TypeError:
                errs = planner.validate(plan, st.roster, list(st.items), planner.PlanNames(st.roster.units))
        if hasattr(planner, 'prepare_comp'):
            planner.prepare_comp(plan, st.roster, st.level, list(st.items))
        if hasattr(planner, 'combine_components'):
            planner.combine_components(plan, st.roster, list(st.items), rnd)
        for rule in ('_carry_corner', '_main_tank_side', '_keep_corner_free'):
            fn = getattr(planner, rule, None)
            if fn:
                planner._guarded(fn, plan, plan, st.roster)
    exe, model = executor_steps(st, plan)
    entries = planner.owned_entries(plan or {}, st.roster) if hasattr(planner, 'owned_entries') else (plan or {})
    _, after_b, at = planner.board_after(entries, st.roster, where=True)
    fielded = [u for u in after_b if getattr(u, 'id', None) is not None and at.get(u.id, ('',))[0] == 'board'
               and not getattr(u, 'missing', None) and getattr(u, 'name', None)]
    buys = [b for b in (plan or {}).get('buy') or [] if isinstance(b, str)]
    auto = [re.sub(r'^auto-buy ', '', w) for k, w in exe if k == 'buy']
    owned = {u.name for u in st.roster.units}
    placed = [e.get('name') for e in (plan or {}).get('board') or [] if isinstance(e, dict) and e.get('name')
              and not e.get('when_owned') and e['name'] not in owned and e['name'] in set(buys) | set(auto)]
    units = fielded + [SimpleNamespace(name=n, traits=CATALOG.unit(n).traits, items=(), star=1, id=None)
                       for n in placed if CATALOG.unit(n)]
    from tft_player.state import board_traits
    try:
        from tft_player.comp_rules import rules
        trait = (rules(case.get('comp') or '') or {}).get('trait')
    except Exception:
        trait = None
    steps = _steps(units)
    no_tier = [u.name for u in units if not (trait and trait in (CATALOG.unit(u.name).traits if CATALOG.unit(u.name)
                                                                  else ())) and not _gained([x for x in units if x is not u], u)]
    idle = [u.name for u in units if _steps([x for x in units if x is not u]) == steps]
    names = [u.name for u in units]
    held = None
    n = (plan or {}).get('level_ups')
    if isinstance(n, int) and n > 0 and st.xp:
        try:
            rp = planner.RoundPlan(dict(plan), st.stage, gold=st.gold)
            scr = SimpleNamespace(level=st.level, xp=st.xp, xp_per_buy=4, timer=st.timer)
            held = bool(rp._inexact_level_waits(n, st.gold, scr, st.roster))
        except Exception:
            held = None
    swap = None
    if (case.get('setup') or {}).get('board_full'):
        from tft_player.player import Player
        p = player_stub(st)
        p._board_full_at = st.stage
        fn = getattr(Player, 'deployment_swap_target', None)
        t = fn(p, SimpleNamespace(stage=st.stage, timer=st.timer, gold=st.gold, level=st.level)) if fn else None
        swap = getattr(t, 'name', None) if t is not None else None
    bench_names = [u.name for u in st.roster.units if u.name and at.get(u.id, ('',))[0] == 'bench']
    return {
        'round': rnd, 'level': st.level, 'gold': st.gold, 'board': names,
        'stars': {u.name: getattr(u, 'star', 1) or 1 for u in units}, 'traits': dict(board_traits(units)), 'steps': steps,
        'no_tier': no_tier, 'idle': idle, 'second_copies': sorted({x for x in names if names.count(x) > 1}),
        'bench': bench_names, 'bought': buys + auto,
        'sold': [w for k, w in exe + model if k == 'sell'], 'level_ups': n if isinstance(n, int) else 0,
        'level_held': held, 'rolls': any(k == 'roll' for k, _ in exe + model),
        'wisp_waits': any(k == 'wait' for k, _ in exe), 'executor': [w for _, w in exe],
        'plan_steps': [w for _, w in model], 'notes': [x for _, x in (plan or {}).get('_notes') or []],
        'rejected': [str(e) for e in errs or []], 'swap_target': swap, 'text': answer or '', 'answered': bool(plan)}


SAFE = {'any': any, 'all': all, 'len': len, 'set': set, 'sorted': sorted, 'min': min, 'max': max, 'sum': sum,
        'abs': abs, 're': re, 'str': str, 'int': int, 'True': True, 'False': False, 'None': None}


def check(expr, res):
    """The case's check on a result -> True / False (a check that raises is False, with the error kept)."""
    env = dict(res)
    env['has'] = lambda name: name in res.get('board', [])
    return bool(eval(compile(expr, '<check>', 'eval'), {'__builtins__': {}, **SAFE}, env))


def summary(res):
    """One line of what the round came out as (for the lists the Editor and the owner read)."""
    if 'error' in res:
        return 'error: ' + res['error'].strip().splitlines()[-1][:200]
    star = res.get('stars') or {}
    return (f"board {', '.join(f'{n}' + '*' * star.get(n, 1) for n in res.get('board') or []) or '-'}; "
            f"steps {', '.join(f'{t} {s}' for t, s in sorted((res.get('steps') or {}).items())) or '-'}; "
            f"bought {', '.join(res.get('bought') or []) or '-'}; XP {res.get('level_ups')}"
            + (' (held)' if res.get('level_held') else '') + ('; rolls' if res.get('rolls') else '')
            + (f"; no tier: {', '.join(res['no_tier'])}" if res.get('no_tier') else '')
            + (f"; second copies: {', '.join(res['second_copies'])}" if res.get('second_copies') else '')
            + (f"; swap {res['swap_target']}" if (res.get('swap_target') is not None) else ''))


# ----------------------------------------------------------------------------------------------- the model

def use_game_model():
    """The game's model and effort (agent/config.json, agent/backends.py: the fallback while Codex is used up)."""
    from tft_player import coach
    import backends
    cfg = json.loads((AGENT / 'config.json').read_text(encoding='utf-8'))
    model, effort = backends.choose(cfg.get('model', 'codex:gpt-6.1-terra'), cfg.get('effort', 'high'), cfg)
    why = coach.use_model(model, effort)
    if why:
        raise RuntimeError(f'the game model {model} cannot answer: {why}')
    return f'{model} {effort}'


def ask_once(text):
    from tft_player import coach
    kind, exe = coach._cli()
    session = coach._CodexSession() if kind == 'codex' else coach._Session()
    return session.ask(exe, [{'type': 'text', 'text': text}], timeout=300, raw=True) or ''


def stage_texts(comp, stage=None, text=None):
    """The stage strategies a replay reads: the comp's current ones, with `stage` replaced by `text` (a candidate);
    -> None for the current ones unchanged."""
    from tft_player import coach
    coach._STRATS.clear()
    os.environ.pop('TFT_STAGE_STRATEGIES', None)
    if stage is None:
        return None
    import stages
    data = stages.frozen(comp)
    data[str(stage)] = {'v': 'candidate', 'text': text or ''}
    f = Path(tempfile.gettempdir()) / f'tft-round-cases-stages-{os.getpid()}.json'
    f.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    os.environ['TFT_STAGE_STRATEGIES'] = str(f)
    return f


def ask_case(case, samples=SAMPLES):
    """The case's question asked again `samples` times, each in a fresh conversation -> [answer]."""
    q0 = prompt_of(case)
    st = state_of(q0, case['round'])
    text = digest(case['run'], int(case['call'])) + '\n\n=== THE QUESTION ===\n' + modernize(q0, st, case['round'])
    with ThreadPoolExecutor(max_workers=min(ASK_WORKERS, samples)) as pool:
        return list(pool.map(lambda _: ask_once(text), range(samples)))


# ----------------------------------------------------------------------------------------------- running

def run_one(case, answers=None):
    """One case on the code under test. answers: the decision case's answers (None: harness = the recorded answer)
    -> {'status': 'pass'|'fail'|'error', 'samples': [...], 'detail': str}."""
    os.environ['TFT_FORCE_COMP'] = case.get('comp') or os.environ.get('TFT_FORCE_COMP', '')
    got = []
    for ans in (answers if answers is not None else [recorded_answer(case)]):
        try:
            st = state_of(prompt_of(case), case['round'])
            setup(case, st)
            res = outcome(case, st, ans)
            ok = check(case['check'], res)
            got.append({'pass': ok, 'detail': summary(res)})
        except Exception as e:
            got.append({'pass': False, 'error': f'{e.__class__.__name__}: {e}',
                        'detail': 'error: ' + traceback.format_exc()[-600:]})
    if not got:
        return {'status': 'error', 'samples': [], 'detail': 'no answers'}
    n = sum(g['pass'] for g in got)
    errors = sum('error' in g for g in got)
    status = 'pass' if n * 2 > len(got) else 'error' if errors == len(got) else 'fail'
    return {'status': status, 'passed': f'{n}/{len(got)}', 'samples': got,
            'detail': next((g['detail'] for g in got if g['pass'] == (status == 'pass')), got[0]['detail'])}


def run_cases(cases, ask=False, record=True):
    """The cases on the working tree's code -> {id: result}. Decision cases: asked again with --ask (answers saved),
    else their saved answers (a decision case never asked runs once with --ask)."""
    boot()
    out, model = {}, None
    for c in cases:
        answers, how = None, 'recorded'
        if c['mode'] == 'decision':
            how = 'saved'
            saved = saved_answers(c['id'])
            if ask or not saved:
                if model is None:
                    model = use_game_model()
                stage_texts(c.get('comp'))
                t0 = time.time()
                answers = ask_case(c)
                ANSWERS.mkdir(parents=True, exist_ok=True)
                (ANSWERS / f"{c['id']}.json").write_text(json.dumps(
                    {'when': datetime.now().isoformat(timespec='seconds'), 'code': head(), 'model': model,
                     'seconds': round(time.time() - t0, 1), 'answers': answers}, ensure_ascii=False, indent=1) + '\n',
                    encoding='utf-8')
                how = 'asked'
            else:
                answers = saved['answers']
        r = run_one(c, answers)
        r.update(when=datetime.now().isoformat(timespec='seconds'), code=head(), answers=how)
        out[c['id']] = r
    if record:
        _save_results(out)
    return out


def as_played(case):
    """The check on the round as it was played: the game's recorded answer through the code that played it (an
    earlier commit runs in its own process) -> result dict."""
    root = code_root(game_row(case['run']).get('code'))
    if root is None:
        boot()
        return run_one(case)
    cmd = [PY, str(Path(__file__)), '_as_played', '--root', str(root), json.dumps(case)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {'status': 'error', 'detail': (p.stdout + p.stderr)[-600:]}


def add(case, run_now=True):
    """A new case: its check must compile, and fail on the round as played (else it does not catch the problem). ->
    (case or None, why)."""
    case = dict(case)
    case.setdefault('status', 'open')
    case.setdefault('added', datetime.now().strftime('%Y-%m-%d %H:%M'))
    case.setdefault('cost', 1)
    row = game_row(case['run'])
    case.setdefault('game', row.get('n'))
    case.setdefault('comp', row.get('comp_played') or row.get('comp'))
    case['id'] = case.get('id') or re.sub(r'[^\w-]+', '-', f"g{case.get('game')}-{case['round']}-{case['mode']}"
                                                          f"-{len(load(None)) + 1}")
    for k in ('run', 'call', 'round', 'mode', 'expect', 'check', 'source'):
        if not case.get(k):
            return None, f'missing {k}'
    if case['mode'] not in ('harness', 'decision'):
        return None, f"mode must be harness or decision, not {case['mode']!r}"
    try:
        compile(case['check'], '<check>', 'eval')
    except SyntaxError as e:
        return None, f'the check does not compile: {e}'
    if not (ROOT / 'runs' / case['run'] / 'calls' / f"{int(case['call']):04d}-prompt.txt").exists():
        return None, f"no question {case['call']} in runs/{case['run']}/calls"
    if run_now:
        shown = as_played(case)
        case['shown'] = {'status': shown.get('status'), 'detail': shown.get('detail'),
                         'code': str(row.get('code') or '')}
        if shown.get('status') == 'pass':
            return None, ("the check passes on the round as played (the game's answer through the code that played "
                          f"it: {shown.get('detail')}) - it does not catch the problem")
    rows = [c for c in load(None) if c['id'] != case['id']]
    _write(rows + [case])
    return case, 'added'


def retire(cid, why):
    rows = load(None)
    for c in rows:
        if c['id'] == cid:
            c.update(status='retired', retired=why, retired_at=datetime.now().strftime('%Y-%m-%d %H:%M'))
    _write(rows)


def failing(cases=None):
    """Open cases whose last result on the current code is not a pass, costliest first."""
    res = results()
    cs = [c for c in (cases or load()) if (res.get(c['id']) or {}).get('status') != 'pass']
    return sorted(cs, key=lambda c: (-int(c.get('cost') or 1), c.get('added', '')))


def failing_text(limit=12, cases=None):
    """The failing list as the Editor, the Strategist and the owner read it."""
    res = results()
    out = []
    for c in failing(cases)[:limit]:
        r = res.get(c['id']) or {}
        out.append(f"[{c['id']}] game {c.get('game')} {c['round']} ({c['mode']}, run {c['run']} call {c['call']}"
                   + (f", request {c['request']}" if c.get('request') else '') + f"): expect {c['expect']} | check "
                   f"`{c['check']}` | now: {r.get('status', 'not run')} {r.get('passed', '')} - {r.get('detail', '')}")
    return out


# ----------------------------------------------------------------------------------------------- the gate and the Judge

def run_sub(ids=None, ask=False):
    """The cases run in a fresh process (the code as it is on disk now) -> {id: result} (empty when none)."""
    if not load():
        return {}
    cmd = [PY, str(Path(__file__)), 'run', '--json'] + (['--ask'] if ask else []) + list(ids or [])
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {'_error': {'status': 'error', 'detail': (p.stdout + p.stderr)[-1500:]}}


def add_sub(specs):
    """Cases added in a fresh process (agent/round_cases.py add-json) -> [{'id', 'added', 'why', 'as_played'}]."""
    if not specs:
        return []
    f = Path(tempfile.gettempdir()) / f'tft-round-cases-add-{os.getpid()}.json'
    f.write_text(json.dumps(specs, ensure_ascii=False), encoding='utf-8')
    p = subprocess.run([PY, str(Path(__file__)), 'add-json', str(f)], cwd=ROOT, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return [{'id': c.get('id'), 'added': False, 'why': (p.stdout + p.stderr)[-300:]} for c in specs]


def compare(before, after):
    """-> (broken: passed before and not now, flipped: failed before and pass now)."""
    broken = sorted(k for k, v in after.items() if (before.get(k) or {}).get('status') == 'pass'
                    and v.get('status') != 'pass')
    if '_error' in after:
        broken.append('_error')
    flipped = sorted(k for k, v in after.items() if v.get('status') == 'pass'
                     and (before.get(k) or {}).get('status') != 'pass')
    return broken, flipped


def request_state(request_id, res=None):
    """The cases of an Editor request -> (ids, all pass)."""
    res = res or results()
    cs = [c['id'] for c in load() if c.get('request') == request_id]
    return cs, bool(cs) and all((res.get(i) or {}).get('status') == 'pass' for i in cs)


def stage_compare(comp, stage, base_text, cand_text, samples=SAMPLES):
    """The Judge on a stage text: the comp's decision cases of that stage asked with the current and the new text ->
    {'cases': n, 'base': {id: status}, 'candidate': {id: status}, 'broken': [...], 'fixed': [...]}."""
    cs = [c for c in load() if c['mode'] == 'decision' and c.get('comp') == comp and stage_of(c['round']) == str(stage)]
    out = {'cases': len(cs), 'base': {}, 'candidate': {}}
    if not cs:
        out.update(broken=[], fixed=[])
        return out
    boot()
    use_game_model()
    for name, text in (('base', base_text), ('candidate', cand_text)):
        for c in cs:
            stage_texts(comp, stage, text)
            r = run_one(c, ask_case(c, samples))
            out[name][c['id']] = {'status': r['status'], 'passed': r.get('passed'), 'detail': r.get('detail')}
    stage_texts(comp)
    out['broken'] = [i for i in out['base'] if out['base'][i]['status'] == 'pass'
                     and out['candidate'][i]['status'] != 'pass']
    out['fixed'] = [i for i in out['base'] if out['base'][i]['status'] != 'pass'
                    and out['candidate'][i]['status'] == 'pass']
    return out


# ----------------------------------------------------------------------------------------------- command line

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('ids', nargs='*')
    r.add_argument('--ask', action='store_true')
    r.add_argument('--failing', action='store_true')
    r.add_argument('--json', action='store_true')
    t = sub.add_parser('try')
    for k in ('run', 'call', 'round'):
        t.add_argument('--' + k, required=True)
    t.add_argument('--check')
    t.add_argument('--mode', default='harness')
    t.add_argument('--setup', default='{}')
    t.add_argument('--samples', type=int, default=1)
    a = sub.add_parser('add')
    for k in ('run', 'call', 'round', 'mode', 'expect', 'check', 'source'):
        a.add_argument('--' + k, required=True)
    a.add_argument('--id')
    a.add_argument('--request')
    a.add_argument('--cost', type=int, default=1)
    a.add_argument('--setup', default='{}')
    ls = sub.add_parser('list')
    ls.add_argument('--failing', action='store_true')
    rt = sub.add_parser('retire')
    rt.add_argument('id')
    rt.add_argument('--why', required=True)
    g = sub.add_parser('stage', help="the Judge: a stage's decision cases with the current and a new text")
    g.add_argument('--comp', required=True)
    g.add_argument('--stage', required=True)
    g.add_argument('--base', required=True, help='a file with the current text')
    g.add_argument('--candidate', required=True, help='a file with the new text')
    g.add_argument('--samples', type=int, default=SAMPLES)
    aj = sub.add_parser('add-json', help='cases from a JSON file (a list): how the climb adds the Reviewer cases')
    aj.add_argument('file')
    p = sub.add_parser('_as_played')
    p.add_argument('--root', required=True)
    p.add_argument('case')
    x = ap.parse_args()
    if x.cmd == '_as_played':
        boot(x.root)
        print(json.dumps(run_one(json.loads(x.case)), ensure_ascii=False))
        return
    if x.cmd == 'stage':
        out = stage_compare(x.comp, x.stage, Path(x.base).read_text(encoding='utf-8'),
                            Path(x.candidate).read_text(encoding='utf-8'), x.samples)
        print(json.dumps(out, ensure_ascii=False))
        return
    if x.cmd == 'add-json':
        got = []
        for c in json.loads(Path(x.file).read_text(encoding='utf-8')):
            try:
                case, why = add(c)
            except Exception as e:
                case, why = None, f'{e.__class__.__name__}: {e}'
            got.append({'id': (case or {}).get('id') or c.get('id'), 'added': case is not None, 'why': why,
                        'as_played': (case or {}).get('shown')})
        print(json.dumps(got, ensure_ascii=False))
        return
    if x.cmd == 'run':
        cs = [c for c in load() if not x.ids or c['id'] in x.ids]
        if x.failing:
            cs = failing(cs)
        res = run_cases(cs, ask=x.ask)
        if x.json:
            print(json.dumps(res, ensure_ascii=False))
        else:
            for k, v in res.items():
                print(f"{v['status']:5} {v.get('passed', ''):4} {k}: {v.get('detail')}")
        return
    if x.cmd == 'try':
        boot()
        c = {'run': x.run, 'call': int(x.call), 'round': x.round, 'mode': x.mode, 'check': x.check or 'True',
             'setup': json.loads(x.setup)}
        row = game_row(x.run)
        c['comp'] = row.get('comp_played') or row.get('comp')
        os.environ['TFT_FORCE_COMP'] = c['comp'] or ''
        answers = [recorded_answer(c)]
        if x.mode == 'decision':
            use_game_model()
            stage_texts(c['comp'])
            answers = ask_case(c, x.samples)
        for ans in answers:
            st = state_of(prompt_of(c), c['round'])
            setup(c, st)
            res = outcome(c, st, ans)
            show = {k: v for k, v in res.items() if k != 'text'}
            print(json.dumps(show, ensure_ascii=False, indent=1))
            if x.check:
                print('check:', check(x.check, res))
        return
    if x.cmd == 'add':
        c = {k: getattr(x, k) for k in ('run', 'round', 'mode', 'expect', 'check', 'source')}
        c.update(call=int(x.call), cost=x.cost, setup=json.loads(x.setup))
        if x.id:
            c['id'] = x.id
        if x.request:
            c['request'] = x.request
        case, why = add(c)
        print(why if case is None else f"added {case['id']} (as played: {case.get('shown', {}).get('status')})")
        return
    if x.cmd == 'retire':
        retire(x.id, x.why)
        return
    res = results()
    for c in (failing() if x.failing else load()):
        r = res.get(c['id']) or {}
        print(f"{r.get('status', 'not run'):7} {c['id']} ({c['mode']}, game {c.get('game')} {c['round']}): {c['expect']}")


if __name__ == '__main__':
    main()
