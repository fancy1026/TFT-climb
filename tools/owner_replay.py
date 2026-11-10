"""The owner's game 65/66 points asked again of the game's model, from the saved calls.

Each case is one recorded question (runs/<run>/calls/<n>-prompt.txt) sent to a fresh conversation of the game's model
(Codex, gpt-6.1-terra, effort high), with the game's system prompt as it is now, a digest of the game before it
(briefing, trait picks, game_strategy and notes from the recorded answers) and the comp notes as they are now (the
owner's FIRST RULES). An augment question has its ask rebuilt with the current code's text (x16.1 / x16.2 changed it).
The answer is checked against the owner's point. Answers go to experiments/owner-replays/<case>-<k>.json.

usage: python tools/owner_replay.py [case ...] [--samples 2]
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'agent'))
os.environ.setdefault('TFT_CLIMB', '1')
os.environ.setdefault('TFT_FORCE_COMP', 'Riftbeast Brambleback')
os.environ['TFT_COACH_BACKEND'] = 'codex'
from tft_player import coach, planner  # noqa: E402

coach.CODEX_MODEL, coach.CODEX_EFFORT = 'gpt-6.1-terra', 'high'
G65, G66 = ROOT / 'runs' / '20261106-004229', ROOT / 'runs' / '20261106-072157'
OUT = ROOT / 'experiments' / 'owner-replays'


def answer_of(run, n):
    try:
        return json.loads((run / 'calls' / f'{n:04d}-result.json').read_text(encoding='utf-8')).get('result') or ''
    except (OSError, ValueError):
        return ''


def digest(run, n):
    """What a new conversation of this game is given before call n: coach.stage_digest from the recorded calls."""
    coach._MEMO.clear()
    for k in range(1, n):
        f = run / 'calls' / f'{k:04d}-prompt.txt'
        if f.exists():
            coach._remember(prompt=f.read_text(encoding='utf-8'))
        coach._remember(answer=answer_of(run, k))
    d = coach.stage_digest('2', with_state=False)
    first = d.split('\n\n', 1)
    return ('NEW CONVERSATION - this game continues in a new conversation; the earlier one is not repeated here; this '
            'is what you need from it.\n\n' + first[1]) if len(first) == 2 else d


def ask_rebuilt(prompt, rerollable):
    """An augment question with its ask written by the current coach.Coach.ask (the cards and context stay)."""
    head, _, rest = prompt.partition('\n\n')
    cards_end = prompt.find('\nTarget final board of')
    tail = prompt.find('\n\n', prompt.find('Take a card now') if 'Take a card now' in prompt else cards_end)
    why = '"why": "<one sentence>"'
    if rerollable:
        ask = (f'Cards you can still reroll (each once): {", ".join(map(str, rerollable))}'
               + (' - the card you kept still has its own reroll' if 3 - len(rerollable) == 2 else '')
               + '. Reroll one of them, or take a card. Taking a card while another one still has its reroll: that '
                 'reroll is used first (it is free) and you are asked again with the new card.' + coach.ARTIFACT_ASK)
        decision = ('{"reroll": [<card>]} or {"pick": <0-2>, "name": "<supplied card name, or unknown>", ' + why
                    + coach.ARTIFACT_KEY + '}')
    else:
        ask = 'No rerolls left. Which augment do you take?' + coach.ARTIFACT_ASK
        decision = '{"pick": <0-2>, "name": "<supplied card name, or unknown>", ' + why + coach.ARTIFACT_KEY + '}'
    m = re.search(r'Take a card now:[^\n]*\n', prompt)
    if m:
        prompt = prompt[:m.start()] + ask + '\n' + prompt[m.end():]
    prompt = re.sub(r'Then end your reply with one JSON object:\n\{.*\}\s*$', 'Then end your reply with one JSON '
                    'object:\n' + decision, prompt, flags=re.S)
    return prompt


def plan_of(text):
    try:
        return planner.parse(text) or {}
    except Exception:
        return {}


def first_obj(text):
    objs = [o for o in coach.json_objects(text) if isinstance(o, dict)]
    return objs[0] if objs else {}


def no_roll(text):
    p = plan_of(text)
    rolls = [k for k in ('roll', 'roll_n', 'fast_rr', 'rolldown')
             if p.get(k) and (not isinstance(p.get(k), dict) or int(p[k].get('times') or 1) > 0)]
    return not rolls, f'roll keys: {rolls or "none"}'


def sells(uid, name):
    def check(text):
        p = plan_of(text)
        return uid in (p.get('sell') or []), f'sell {p.get("sell")} ({name} is unit {uid})'
    return check


def gives(to_ok, to_bad, items):
    def check(text):
        p = plan_of(text)
        got = [(e.get('give') or e.get('component'), e.get('to') if 'to' in e else e.get('id'))
               for e in p.get('items') or [] if isinstance(e, dict)]
        mine = [(i, t) for i, t in got if i in items]
        bad = [g for g in mine if g[1] in to_bad]
        good = [g for g in mine if g[1] in to_ok]
        return bool(good) and not bad, f'gives {got}'
    return check


def no_buy(name):
    def check(text):
        p = plan_of(text)
        buys = [b if isinstance(b, str) else (b or {}).get('name') for b in p.get('buy') or []]
        board = [e.get('name') for e in p.get('board') or [] if isinstance(e, dict)]
        return name not in buys and name not in board, f'buy {buys}, board names {board}'
    return check


def rerolls_or_picks(text):
    o = first_obj(text)
    o = o.get('augment', o) if isinstance(o.get('augment'), dict) else o
    did = 'reroll' if o.get('reroll') not in (None, []) or isinstance(o.get('keep'), int) else \
        'pick' if isinstance(o.get('pick'), int) else None
    return did is not None, f'answer: {did} {json.dumps(o)[:160]}'


def forge_list(forge):
    def check(text):
        o = first_obj(text)
        if isinstance(o.get('pick'), int) and o['pick'] == forge:
            arts = o.get('artifacts')
            return isinstance(arts, list) and len(arts) >= 1, f'picked the Forge, artifacts {arts}'
        return True, f'did not pick the Forge: {json.dumps(o)[:160]}'
    return check


def item_holders(others):
    """x20: the player's own item steps go to the main tank, Brambleback, Elder Dragon or a 3-star Pebbles only - none
    to the other units (others: their unit ids). Brambleback's carry items held by others are moved to him by the
    controller's comp rule (agent/comps/Riftbeast_Brambleback/rules.json), not by the player's plan."""
    def check(text):
        p = plan_of(text)                                # planner.parse: gives as items (id), moves as transfer
        gives = [e for e in p.get('items') or [] if isinstance(e, dict) and not e.get('use')]
        moves = [e for e in p.get('transfer') or [] if isinstance(e, dict)]
        bad = [e for e in gives if e.get('id') in others] + [e for e in moves if e.get('to') in others]
        return not bad, f'gives {gives}; transfers {moves}'
    return check


G72T = ROOT / 'runs' / '20261107-023517'     # the Normal test game of flex (2026-11-07): Brambleback from 2-5
# its unit ids: Krug 32 (main tank), Brambleback 81, Murkwolf 44, Sentinel 105, Cinderling 9/61, Mama Beak 31/87
X20 = item_holders({105, 9, 61, 44, 31, 87, 22})

G72 = ROOT / 'runs' / '20261107-070659'     # ranked game 72 (2026-11-07), Brambleback from 2-1, 5th


def buys_any(names):
    """x24.1: the plan buys or rolls for one of these (MetaTFT's level-8 non-Riftbeasts)."""
    def check(text):
        p = plan_of(text)
        rolls = [p.get(k) for k in ('roll_n', 'fast_rr') if isinstance(p.get(k), dict)]
        want = {n for r in rolls for n in (r.get('buy') or {})} | {b for b in p.get('buy') or [] if isinstance(b, str)}
        return bool(want & names), f'buys/rolls for {sorted(want)}'
    return check


def unfielded(uid, name):
    """x24.2: the unit leaves the board this plan (sold, or benched without a condition)."""
    def check(text):
        p = plan_of(text)
        benched = [e for e in p.get('bench') or [] if isinstance(e, dict) and e.get('id') == uid]
        ok = uid in (p.get('sell') or []) or any(not e.get('when_owned') for e in benched)
        return ok, f'{name} (unit {uid}): sell {p.get("sell")}, bench {benched}'
    return check


def all_in(text):
    """x24.7: at 1 HP the roll keeps no gold."""
    p = plan_of(text)
    r = next((p[k] for k in ('fast_rr', 'roll_n') if isinstance(p.get(k), dict)), None)
    floor = None if r is None else r.get('gold_floor', r.get('floor'))
    return r is not None and not floor, f'roll {r}'


def round_runs(text):
    """x24.8: the plan runs through the executor (no tick error) and the round acts: an anvil opened or a roll."""
    p = plan_of(text)
    try:
        planner.RoundPlan(dict(p), '6-1')._container()
    except Exception as e:
        return False, f'the executor fails: {e!r}'
    acts = [k for k in ('open_container', 'roll_n', 'fast_rr', 'buy', 'level_ups') if p.get(k)]
    return bool(acts), f'steps: {acts}'


G74 = ROOT / 'runs' / '20261107-203531'     # ranked game 74 (2026-11-07): flex, Vanguard Morgana from 2-7, F8 at 4-3


def objs_of(text, key):
    return [o[key] for o in coach.json_objects(text) if isinstance(o, dict) and isinstance(o.get(key), dict)]


def gives_up_items(holder, to):
    """x25.1: the holder (an off-comp 1-star with an item) is sold, or its items are moved to the 2-star `to`."""
    def check(text):
        p = plan_of(text)
        moved = [t for t in p.get('transfer') or [] if t.get('from') == holder and t.get('to') == to]
        return holder in (p.get('sell') or []) or bool(moved), f"sell {p.get('sell')}, transfers {p.get('transfer')}"
    return check


def not_committed(name):
    """x25.2: with Titan's Resolve built, the commit is not this comp."""
    def check(text):
        st = (objs_of(text, 'strategy') or [{}])[0]
        return st.get('commit') != name, f"commit {st.get('commit')!r}, P {st.get('comps')}"
    return check


def fight_xp(text):
    """x25.4: in the fight with 82 gold, the plan buys its XP now (execution_plan), not later."""
    p = plan_of(text)
    return int(p.get('level_ups') or 0) > 0, f"xp now {p.get('level_ups')}; next_round {objs_of(text, 'next_round')}"


def no_split(holder):
    """x25.5: the 2-star holder's items are not moved off it to another unit (items concentrate)."""
    def check(text):
        ts = [t for o in [plan_of(text)] + [planner.parse(json.dumps({'execution_plan': n})) for n in
                                            objs_of(text, 'next_round')] for t in o.get('transfer') or []]
        split = [t for t in ts if t.get('from') == holder and t.get('to') != holder]
        return not split, f'transfers {ts}'
    return check


def no_wisp_no_roll(slot):
    """x25.6 / x25.8: at level 7 the plan neither buys the shop-changing Wisp nor rolls."""
    def check(text):
        p = plan_of(text)
        wisp = [w for w in p.get('wisps') or [] if isinstance(w, dict) and w.get('slot') == slot and w.get('buy')]
        rolls = [k for o in [p] + [planner.parse(json.dumps({'execution_plan': n})) for n in objs_of(text, 'next_round')]
                 for k in ('roll_n', 'fast_rr') if o.get(k)] + (['roll'] if (p.get('roll') or {}).get('down_to') else [])
        return not wisp and not rolls, f'wisp buy {wisp}, rolls {rolls}'
    return check


def keeps(uid, name):
    """x25.7: the Coven unit (the 2-star; a spare copy may go) is kept and fielded while the game still pivots."""
    def check(text):
        p = plan_of(text)
        fielded = any(e.get('id') == uid or e.get('unit') == uid for e in p.get('board') or [] if isinstance(e, dict))
        return uid not in (p.get('sell') or []) and fielded, f'sell {p.get("sell")}, board {p.get("board")} ({name} = unit {uid})'
    return check


G74B = ROOT / 'runs' / '20261107-213309'    # ranked game 74b (2026-11-07): Brambleback, 2nd


def no_inexact_level(text):
    """x26.10: at 3-1's fight, level 5 with 49 gold (0/20 XP): no XP for level 6 now or in next_round (2/36 over)."""
    p = plan_of(text)
    nxt = [n.get('xp') or n.get('level_ups') for n in objs_of(text, 'next_round')]
    return not int(p.get('level_ups') or 0) and not any(nxt), f"xp now {p.get('level_ups')}, next_round xp {nxt}"


def benches_weaker(strong, weak):
    """x26.22: for Elder Dragon's slot the 2-star Pebbles (weak) leaves the board, not the 3-star Cinderling (strong)."""
    def check(text):
        p = plan_of(text)
        out = [e.get('id') for e in p.get('bench') or [] if isinstance(e, dict)] + list(p.get('sell') or [])
        return strong not in out, f'benched/sold {out} (strong unit {strong}, weak {weak})'
    return check


FLEX = None                                 # a case's comp: None = before a commit (the flex set)
CASES = {
    # x26 (game 74b): an inexact level-up, the weaker unit off the board
    'x26.10-3-1': (G74B, 34, None, no_inexact_level, 'Riftbeast Brambleback'),
    'x26.22-5-5': (G74B, 104, None, benches_weaker(61, 3), 'Riftbeast Brambleback'),
    # x25 (game 74): items on the 2-star, Titan's means Brambleback, XP in the fight, the Wisp and the level-7 rolls,
    # Coven kept while pivoting
    'x25.1-2-3': (G74, 17, None, gives_up_items(9, 11), FLEX),
    'x25.2-2-7': (G74, 29, None, not_committed('Vanguard Morgana'), FLEX),
    'x25.4-3-2': (G74, 39, None, fight_xp, 'Vanguard Morgana'),
    'x25.5-3-2': (G74, 39, None, no_split(11), 'Vanguard Morgana'),
    'x25.6-3-6': (G74, 51, None, no_wisp_no_roll(4), 'Vanguard Morgana'),
    'x25.8-4-1': (G74, 59, None, no_wisp_no_roll(None), 'Vanguard Morgana'),
    'x25.7-3-7': (G74, 54, None, keeps(3, 'Camille 2-star'), 'Vanguard Morgana'),
    # x24.8 6-1: a round that ran no step (the plan's "open_container" as a number crashed every tick)
    'x24.8-6-1': (G72, 117, None, round_runs),
    # x24 (game 72): the eighth slot at level 8, Ashe, the last life
    'x24.1-4-5': (G72, 79, None, buys_any({'Sett', 'Rakan'})),
    'x24.2-5-2': (G72, 94, None, unfielded(61, 'Ashe')),
    'x24.7-5-6': (G72, 111, None, all_in),
    # x20.1 the item build: 3 on the main tank, 3 on Brambleback, the rest on Elder Dragon or a 3-star Pebbles
    'x20.1-4-5': (G72T, 76, None, X20),
    'x20.1-4-7': (G72T, 82, None, X20),
    'x20.1-5-2': (G72T, 88, None, X20),
    # x18.2 never roll in stage 2: game 66's stage-2 questions whose answers rolled (all free rerolls)
    'x18.2-2-1': (G66, 14, None, no_roll),
    'x18.2-2-2': (G66, 19, None, no_roll),
    'x18.2-2-6': (G66, 31, None, no_roll),
    'x18.2-2-7': (G66, 35, None, no_roll),
    # x18.4 Elise (unit 10) sold at 2-3 for the 10-gold interest
    'x18.4-2-3': (G66, 21, None, sells(10, 'Elise')),
    # x18.5 Edge of Night's components (B.F. Sword + Chain Vest) on Murkwolf (11), not Cinderling (14)
    'x18.5-2-5': (G66, 27, None, gives({11}, {14, 16}, {'B.F. Sword', 'Chain Vest'})),
    # x18.6 3-1: the carry items on Murkwolf (11), not Cinderling (14)
    'x18.6-3-1': (G66, 39, None, gives({11}, {14, 16}, {'Needlessly Large Rod', 'Recurve Bow'})),
    # x18.7 3-2: no Alistar bought or fielded
    'x18.7-3-2': (G66, 45, None, no_buy('Alistar')),
    # x16.2 game 65 3-2: the rerolls offered (the game said "Take a card now"); the answer may reroll
    'x16.2-3-2': (G65, 42, [0, 1, 2], rerolls_or_picks),
    # x16.1 game 65 2-1: Portable Forge (card 1) - picked with its wanted artifacts, best first
    'x16.1-2-1': (G65, 12, [0], forge_list(1)),
    # x16.1 the same hand with no reroll left (a card must be taken)
    'x16.1-2-1-take': (G65, 12, [], forge_list(1)),
}


def run_case(name, k):
    run, n, reroll, check, *comp = CASES[name]
    if comp and comp[0] is None:
        os.environ.pop('TFT_FORCE_COMP', None)              # before a commit: the flex set
    elif comp:
        os.environ['TFT_FORCE_COMP'] = comp[0]
    q = (run / 'calls' / f'{n:04d}-prompt.txt').read_text(encoding='utf-8')
    if reroll is not None:
        q = ask_rebuilt(q, reroll)
    text = digest(run, n) + '\n\n=== THE QUESTION ===\n' + q
    kind, exe = coach._cli()
    s = coach._CodexSession()
    t0 = time.time()
    ans = s.ask(exe, [{'type': 'text', 'text': text}], timeout=300, raw=True) or ''
    ok, why = check(ans)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{name}-{k}.json').write_text(json.dumps({'case': name, 'run': run.name, 'call': n, 'ok': ok, 'check': why,
                                                      'seconds': round(time.time() - t0, 1), 'question': q,
                                                      'answer': ans}, ensure_ascii=False, indent=1), encoding='utf-8')
    return ok, why, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cases', nargs='*')
    ap.add_argument('--samples', type=int, default=1)
    a = ap.parse_args()
    for name in a.cases or list(CASES):
        for k in range(1, a.samples + 1):
            try:
                ok, why, secs = run_case(name, k)
            except Exception as e:
                ok, why, secs = False, f'error {e.__class__.__name__}: {e}', 0
            print(json.dumps({'case': name, 'sample': k, 'ok': ok, 'seconds': secs, 'check': why[:300]},
                             ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
