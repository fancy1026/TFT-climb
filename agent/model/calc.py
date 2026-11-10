"""The calculator: the game's arithmetic, from agent/model/mechanics.json only. Whatever can be computed is computed
here and never learned as a lesson (the owner, 2026-10-25: interest steps, levels and streak gold are math).

    python agent/model/calc.py facts   --stage 2-3 --level 4 --xp 4/10 --gold 14 [--hp 90] [--streak -2]
    python agent/model/calc.py xp      --stage 2-3 --level 4 --xp 4/10 --gold 14
    python agent/model/calc.py streaks --stage 3-1 --gold 30 [--streak 0] --a LLLLLLLLLL --b LLLLLWLLLL [--alive 2]
    python agent/model/calc.py roll    --level 8 --cost 4 --rolls 10 [--have 1] [--others 2] [--want 2]
    python agent/model/calc.py pve     --stage 3-7

Timing (measured, mechanics.json "timing"): income and the passive XP arrive at the start of each planning phase;
the carousel round (x-4) gives neither.
"""
import argparse
import json
import math
from pathlib import Path

M = json.loads((Path(__file__).resolve().parent / 'mechanics.json').read_text(encoding='utf-8'))
MEASURED = json.loads((Path(__file__).resolve().parent / 'measured.json').read_text(encoding='utf-8'))


# --- rounds ---------------------------------------------------------------------------------------------------

def key(stage):
    a, b = stage.split('-')
    return int(a), int(b)


def label(a, b):
    return f'{a}-{b}'


def next_phase(stage):
    """The next planning phase after this one (the carousel x-4 has none)."""
    a, b = key(stage)
    if a == 1:
        return label(1, b + 1) if b < 4 else '2-1'
    if b == 7:
        return label(a + 1, 1)
    return label(a, 5 if b == 3 else b + 1)


def phases_after(stage, n):
    out, s = [], stage
    for _ in range(n):
        s = next_phase(s)
        out.append(s)
    return out


def is_pvp(stage):
    a, b = key(stage)
    return a >= 2 and b in (1, 2, 3, 5, 6)


def is_pve(stage):
    a, b = key(stage)
    return a == 1 or b == 7


# --- gold -----------------------------------------------------------------------------------------------------

def interest(gold):
    i = M['income']['interest']
    return min(i['max'], max(0, gold) // i['per'])


def base(stage):
    t = M['income']['base_by_round']
    return t.get(stage, t['other'])


def streak_gold(streak):
    n = abs(streak)
    t = M['income']['streak_gold']
    if n < 2:
        return 0
    return t.get(str(n), t['6+'])


def income(next_stage, gold_held, streak_after, won):
    """Gold arriving at the start of next_stage: base + interest on the gold held + streak gold (+1 for a PvP win)."""
    return base(next_stage) + interest(gold_held) + streak_gold(streak_after) + (M['income']['pvp_win_gold'] if won else 0)


def interest_slack(gold):
    """(interest now, gold you can spend without lowering it, gold to the next interest step or None at the cap)."""
    i = interest(gold)
    cap = M['income']['interest']['max'] * M['income']['interest']['per']
    if gold >= cap:
        return i, gold - cap, None
    per = M['income']['interest']['per']
    return i, gold - i * per, (i + 1) * per - gold


# --- XP -------------------------------------------------------------------------------------------------------

def to_next(level):
    return M['xp']['to_next_level'].get(str(level))


def add_xp(level, xp, amount):
    xp += amount
    while level < 10 and to_next(level) and xp >= to_next(level):
        xp -= to_next(level)
        level += 1
    return level, xp


def next_level_at(stage, level, xp, target=None):
    """The planning phase at whose start `target` (default level+1) arrives with passive XP only; (stage, phases)."""
    target = target or level + 1
    s, lv, x = stage, level, xp
    for n in range(1, 40):
        s = next_phase(s)
        lv, x = add_xp(lv, x, M['xp']['passive_per_round'])
        if lv >= target:
            return s, n
    return None, None


def held(stage, gold, phases=2):
    """Gold after holding everything for `phases` planning phases: base income and interest only (no streak, win gold
    or loot) - what spending now costs later, interest compounding included."""
    s, g = stage, gold
    for _ in range(phases):
        s = next_phase(s)
        g += base(s) + interest(g)
    return s, g


def xp_options(stage, level, xp, gold, per=None):
    """Rows (buys, cost, gold left, interest lost this round, when the next level arrives, gold two phases later if
    the rest is held): no buys, then every number of buys that brings the next level one phase earlier, up to now.
    per: XP a buy gives, when an augment changed it (seen from our own buys)."""
    buy = dict(M['xp']['buy'], **({'xp': per} if per else {}))
    rows, seen = [], None
    for k in range(0, gold // buy['gold'] + 1):
        lv, x = add_xp(level, xp, k * buy['xp'])
        left = gold - k * buy['gold']
        when = ('now', 0) if lv > level else next_level_at(stage, lv, x, level + 1)
        if when != seen:
            rows.append({'buys': k, 'cost': k * buy['gold'], 'gold_left': left,
                         'interest_lost': interest(gold) - interest(left), 'next_level': when[0], 'phases': when[1],
                         'held_at': held(stage, left)})
            seen = when
        if lv > level:
            break
    return rows


# --- streaks --------------------------------------------------------------------------------------------------

def streak_path(stage, gold, streak, results, alive=2):
    """Hold all gold and play `results` (W/L, one per PvP round from `stage` on). -> gold at the end, HP lost, the
    stage the path ends after, per-round log. PvE rounds pass without a result and keep the streak; PvE loot and
    spending are left out."""
    s, g, st, hp, log = stage, gold, streak, 0, []
    queue = list(results)
    while queue:
        won = None
        if is_pvp(s):
            won = queue.pop(0) == 'W'
            st = (max(st, 0) + 1) if won else (min(st, 0) - 1)
            if not won:
                hp += M['damage']['stage_damage'][str(key(s)[0])] + alive * M['damage']['per_surviving_enemy_unit']
        nxt = next_phase(s)
        inc = income(nxt, g, st, bool(won))
        log.append({'round': s, 'result': '-' if won is None else ('W' if won else 'L'), 'streak': st,
                    'gold_held': g, 'income_next': inc})
        g += inc
        s = nxt
    return {'gold_end': g, 'hp_lost': hp, 'ends_before': s, 'log': log}


# --- shop odds ------------------------------------------------------------------------------------------------

def roll_odds(level, cost, rolls, have=0, others=0, gone=None, want=1):
    """Chance to see at least `want` copies of ONE champion of this cost in `rolls` rerolls (about 4.5 champion slots a
    shop: the rightmost slot is a Wisp every other round). Independent slots; a simplification: buying shrinks the
    pool, and the current shop is not counted."""
    s = M['shop']
    pct = s['odds_by_level'][str(level)][cost - 1] / 100
    copies = s['copies_per_champion'][str(cost)]
    pool = copies * s['champions_per_cost'][str(cost)] - (have + others if gone is None else gone)
    left = copies - have - others
    if pct == 0 or left <= 0 or pool <= 0:
        return {'p_at_least': 0.0, 'per_slot': 0.0, 'expected_seen': 0.0, 'gold': 2 * rolls}
    p = pct * left / pool
    n = int(4.5 * rolls)
    at_least = 1 - sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(want))
    return {'p_at_least': round(at_least, 3), 'per_slot': round(p, 4), 'expected_seen': round(n * p, 2),
            'gold': s['reroll_gold'] * rolls}


def xp_gold_to(level, xp, target, per=None):
    """Gold of XP buys from (level, xp) to `target` now (no free XP)."""
    need = (sum(to_next(l) or 0 for l in range(level, target)) - xp)
    return max(0, -(-need // (per or M['xp']['buy']['xp']))) * M['xp']['buy']['gold']


def roll_by_level(level, xp, gold, missing, up_to=2, per=None):
    """For the units still missing - [(name, cost, copies held)] - at the current level and the next ones: the chance
    a shop shows at least one of them, the XP gold to get there now, and the copies all the gold left would see.
    Where to roll is where the comp's missing units show most for the gold (the owner, 2026-10-26: level 8 for
    4-costs is one example of this, not the rule). Independent slots, ~4.5 champion slots a shop, others' copies
    unknown (counted as 0)."""
    s = M['shop']
    rows = []
    for lv in range(level, min(10, level + up_to) + 1):
        cost = xp_gold_to(level, xp, lv, per) if lv > level else 0
        if cost > gold:
            break
        odds = s['odds_by_level'][str(lv)]
        miss_none, per_roll, by_cost = 1.0, 0.0, {}
        for _, c, held in missing:
            left = s['copies_per_champion'][str(c)] - held
            pool = s['copies_per_champion'][str(c)] * s['champions_per_cost'][str(c)]
            p = odds[c - 1] / 100 * max(0, left) / pool
            miss_none *= (1 - p) ** 4.5
            per_roll += 4.5 * p
            by_cost[c] = by_cost.get(c, 0.0) + 4.5 * p
        rolls = (gold - cost) // s['reroll_gold']
        rows.append({'level': lv, 'xp_gold': cost, 'p_shop_shows_one': round(1 - miss_none, 3), 'rolls': rolls,
                     'copies_expected': round(rolls * per_roll, 1),
                     'copies_by_cost': {c: round(rolls * v, 1) for c, v in sorted(by_cost.items())}})
    return rows


# --- facts for one planning phase -----------------------------------------------------------------------------

def typical_loss(stage):
    """The median HP of a lost fight in this stage, measured in our games (measured.json)."""
    t = MEASURED['our_games']['hp_lost_per_lost_fight_by_stage']
    a = str(min(key(stage)[0], max(int(k) for k in t)))
    return t[a]['median'] if a in t else M['damage']['stage_damage'][str(key(stage)[0])] + 1


def hp_runway(stage, hp, loss, until):
    """HP left at the start of `until` if every PvP fight from `stage` on costs `loss` (None if out before), and how
    many such losses the HP pays for. loss None: each stage's typical loss."""
    s, h, n = stage, hp, 0
    while key(s) < key(until):
        if is_pvp(s):
            h -= loss if loss is not None else typical_loss(s)
            n += 1
            if h <= 0:
                return None, n - 1
        s = next_phase(s)
    return round(h), max(0, (hp - 1) // max(1, round(loss if loss is not None else typical_loss(stage))))


def losing_cost(stage, until):
    """HP lost if every PvP fight from `stage` until `until` is lost at each stage's typical loss, and how many."""
    s, h, n = stage, 0.0, 0
    while key(s) < key(until):
        if is_pvp(s):
            h += typical_loss(s)
            n += 1
        s = next_phase(s)
    return round(h), n


def facts(stage, level, xp, gold, hp=None, streak=0, recent_loss=None, missing=None, per=None):
    """The arithmetic of this moment, as short lines for a model to reason with (no advice). recent_loss: the average
    HP of the last lost fights, for the HP runway (g25: banked 32 -> 52 at level 6 while losing 10-12 a round, six fights
    before the planned 4-2 spike - about 10 HP would have been left)."""
    L = []
    i, slack, to_step = interest_slack(gold)
    if to_step is None:
        L.append(f"Interest: {gold} gold is at the cap (+{i}); the {slack} above 50 earns nothing, and holding it costs nothing.")
    else:
        L.append(f"Interest: {gold} gold earns +{i} next round. Spending up to {slack} keeps that; {to_step} more reaches +{i + 1}.")
    nxt = next_phase(stage)
    if streak is None:                                   # not known here: the player reads it from the screen
        t = M['income']['streak_gold']
        steps = ', '.join(f"{k} in a row +{v}" for k, v in t.items())
        L.append(f"Income at {nxt} if you hold your gold: {base(nxt) + interest(gold)} (base + interest)"
                 + (f", +{M['income']['pvp_win_gold']} after a win" if is_pvp(stage) else ' (PvE round; plus its loot)')
                 + f", plus streak gold after the fight ({steps}; win or loss streaks alike; the streak is on the screen).")
    elif is_pvp(stage):
        win = income(nxt, gold, max(streak, 0) + 1, True)
        lose = income(nxt, gold, min(streak, 0) - 1, False)
        L.append(f"Income at {nxt} if you hold your gold: {win} after a win, {lose} after a loss (streak now {streak:+d}).")
    else:
        L.append(f"Income at {nxt} if you hold your gold: {income(nxt, gold, streak, False)} (PvE round; plus its loot).")
    if level < 10 and to_next(level):
        rows = xp_options(stage, level, xp, gold, per)
        free = rows[0]
        free_at = (f"arrives at {free['next_level']} without buying" if (free['phases'] or 99) <= 8 else
                   f"does not arrive within 8 phases without buying")          # (not "at 8-2": no such stage)
        parts = [f"Level {level}, XP {xp}/{to_next(level)}: with the free +{M['xp']['passive_per_round']} a phase, level "
                 f"{level + 1} {free_at}."]
        now_cost = xp_gold_to(level, xp, level + 1, per)
        parts.append(f"Level {level + 1} now costs {now_cost} gold of XP (you have {gold})"
                     + (f"; a buy gives {per} XP here (seen from our own buys)." if per and per != M['xp']['buy']['xp']
                        else "."))
        useful = [r for r in rows[1:] if r['next_level'] == 'now' or (r['phases'] or 99) <= 3]
        if useful:
            parts.append(f"Holding all gold, you have {free['held_at'][1]} at {free['held_at'][0]} (base + interest "
                         f"only).")
        for r in useful:
            parts.append(f"{r['buys']} XP buy(s) ({r['cost']} gold, {r['gold_left']} left, interest -{r['interest_lost']}"
                         f" this round; {r['held_at'][1]} at {r['held_at'][0]} if the rest is held) -> level {level + 1} "
                         f"{'now' if r['next_level'] == 'now' else 'at ' + r['next_level']}.")
        L.append(' '.join(parts))
        odds = M['shop']['odds_by_level']
        L.append(f"Shop odds (1/2/3/4/5-cost %): level {level} {'/'.join(map(str, odds[str(level)]))}; level {level + 1} "
                 f"{'/'.join(map(str, odds[str(level + 1)]))}.")
    if missing:
        rows = roll_by_level(level, xp, gold, missing, per=per)
        names = ', '.join(f"{n} ({c}-cost, {h} held)" for n, c, h in missing)
        parts = [f"level {r['level']}" + (f" (+{r['xp_gold']} gold of XP now)" if r['xp_gold'] else ' (now)')
                 + f": a shop shows one {round(100 * r['p_shop_shows_one'])}% of the time; all your gold left = "
                 f"{r['rolls']} rolls, about {r['copies_expected']} copies ("
                 + ', '.join(f"{v} {c}-cost" for c, v in r['copies_by_cost'].items()) + ')' for r in rows]
        if parts:
            L.append(f"Rolling for the units your comp still needs to 2-star - {names}: " + '; '.join(parts) + '.')
    elif level < 10:                                     # no comp list here: one specific unit of each cost
        s = M['shop']
        per = []
        for lv in range(level, min(10, level + 2) + 1):
            odds = s['odds_by_level'][str(lv)]
            per.append(f"level {lv} " + ', '.join(
                f"{c}-cost {10 * 4.5 * odds[c - 1] / 100 / s['champions_per_cost'][str(c)]:.1f}" for c in (2, 3, 4, 5)))
        L.append("Copies of one specific unit seen per 10 rolls (none of its copies taken by others): "
                 + '; '.join(per) + '.')
    a = key(stage)[0]
    if is_pvp(stage):
        L.append(f"A lost fight here costs {M['damage']['stage_damage'][str(a)]} + 1 per surviving enemy unit"
                 + (f" (HP {hp})." if hp is not None else '.'))
    marks_for = []
    if level < 10 and to_next(level):
        when, n_ph = next_level_at(stage, level, xp)
        if when and n_ph <= 6:                           # a free level further off than that is no plan
            marks_for.append((when, f"{when} (level {level + 1} with the free XP)"))
    # and the next stage's start, a horizon that moves with the game (it was 4-2 every game until 2026-10-29: no level
    # is tied to a round - the owner, 2026-10-28)
    nxt_stage = label(key(stage)[0] + 1, 1)
    if nxt_stage not in [m for m, _ in marks_for]:
        marks_for.append((nxt_stage, nxt_stage))
    marks_for.sort(key=lambda m: key(m[0]))
    loss = max(1, round(recent_loss)) if recent_loss else None
    how = f"your recent {loss} HP a loss" if loss else (
        "each stage's typical loss (median of our lost fights: " + ', '.join(
            f"stage {k} {v['median']:g}" for k, v in MEASURED['our_games']['hp_lost_per_lost_fight_by_stage'].items())
        + ")")
    if hp is not None:
        marks = []
        for until, name in marks_for:
            left, _ = hp_runway(stage, hp, loss, until)
            marks.append(f"at {name} {left if left is not None else 'out'}")
        _, pays = hp_runway(stage, hp, loss, '9-1')
        L.append(f"HP runway: losing every fight at {how}, HP left " + '; '.join(marks)
                 + f"; your HP pays for {pays} more such losses.")
    elif marks_for:
        marks = []
        for until, name in marks_for:
            cost, n = losing_cost(stage, until)
            marks.append(f"by {name} about {cost} HP ({n} fight{'s' if n != 1 else ''})")
        L.append(f"HP runway: losing every fight from here at {how} costs " + '; '.join(marks)
                 + "; compare with your HP on the screen.")
    upcoming = phases_after(stage, 4)
    marks = []
    for s in [stage] + upcoming:
        if s in M['rounds']['augments']:
            marks.append((key(s), f'{s} augment'))
        if s in M['pve_loot_expected']:
            e = M['pve_loot_expected'][s]
            marks.append((key(s), f"{s} PvE (expected {e['coins']} coins, {e['gold_value']} gold value with units, "
                                  f"{e['components']} components)"))
    for c in M['rounds']['carousels']:
        if key(stage) < key(c) <= key(upcoming[-1]):
            marks.append((key(c), f'{c} carousel (no gold, no XP)'))
    if marks:
        L.append('Coming up: ' + '; '.join(t for _, t in sorted(marks)) + '.')
    return L


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('facts', 'xp'):
        p = sub.add_parser(name)
        p.add_argument('--stage', required=True)
        p.add_argument('--level', type=int, required=True)
        p.add_argument('--xp', required=True, help='x/N as on the HUD')
        p.add_argument('--gold', type=int, required=True)
        p.add_argument('--hp', type=int)
        p.add_argument('--streak', type=int, default=0, help='+ wins in a row, - losses in a row')
    p = sub.add_parser('streaks')
    p.add_argument('--stage', required=True)
    p.add_argument('--gold', type=int, required=True)
    p.add_argument('--streak', type=int, default=0)
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--alive', type=int, default=2, help='enemy units left alive after a lost fight')
    p = sub.add_parser('roll')
    for a, t in (('--level', int), ('--cost', int), ('--rolls', int)):
        p.add_argument(a, type=t, required=True)
    for a in ('--have', '--others', '--want'):
        p.add_argument(a, type=int, default=1 if a == '--want' else 0)
    p = sub.add_parser('pve')
    p.add_argument('--stage', required=True)
    x = ap.parse_args()
    if x.cmd in ('facts', 'xp'):
        cur = int(x.xp.split('/')[0])
        if x.cmd == 'facts':
            print('\n'.join(facts(x.stage, x.level, cur, x.gold, x.hp, x.streak)))
        else:
            print(json.dumps(xp_options(x.stage, x.level, cur, x.gold), indent=1))
    elif x.cmd == 'streaks':
        A = streak_path(x.stage, x.gold, x.streak, x.a, x.alive)
        B = streak_path(x.stage, x.gold, x.streak, x.b, x.alive)
        print(json.dumps({'a': {k: A[k] for k in ('gold_end', 'hp_lost', 'ends_before')},
                          'b': {k: B[k] for k in ('gold_end', 'hp_lost', 'ends_before')},
                          'gold_difference_a_minus_b': A['gold_end'] - B['gold_end'],
                          'hp_difference_a_minus_b': A['hp_lost'] - B['hp_lost'],
                          'note': 'all gold held; PvE loot left out; streak gold assumed to keep paying across PvE rounds'},
                         indent=1))
    elif x.cmd == 'roll':
        print(json.dumps(roll_odds(x.level, x.cost, x.rolls, x.have, x.others, want=x.want), indent=1))
    else:
        print(json.dumps(M['pve_loot_expected'].get(x.stage, 'no PvE expectation for this round'), indent=1))


if __name__ == '__main__':
    main()
