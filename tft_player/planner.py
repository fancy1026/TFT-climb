"""The model plans every round; local code carries the plan out and checks it.

Planning phase starts -> one model call with the exact state -> a RoundPlan
(wisps, sells, buys, levels, roll-down, lineup with hexes, items) ->
the executor turns it into one action per tick, re-checking the roster each
time, so a failed or changed step is simply retried or skipped.

Combat starts -> the controller reports what it did with the plan and shows
the board we fight with. The model uses two conversations per game, switching at 4-3 (coach._stage_turnover).

Prompts carry the decision, what changed, the state and the reply format -
no strategy advice (the model has reference files for game/MetaTFT data).
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter

from tft_player.brain import Action
from tft_player.catalog import CATALOG
from tft_player.emblems import BY_PAIR as EMBLEM_PAIRS, EMBLEMS, MAKERS, board_line, is_emblem_item, trait_of
from tft_player.items import BY_PAIR, COMPONENTS, RECIPES, consumable_note, is_consumable, is_reforger, \
    is_remover, is_reusable, item_desc, radiant_twin
from tft_player.meta import META

THIEF = "Thief's Gloves"            # takes all 3 item slots of its holder: no other item goes on it

PLAN_REPLY = 'Reply with {"execution_plan": ...} first, then {"strategy": ...} (format in your instructions).'

REVIEW_REPLY = ('Reply with ONLY this JSON: {"strategy": {...same fields as in planning...}, '
                '"problems": ["anything wrong in the reported board or state"]}')


def round_data_lines(scr, roster, items, hp_log, slots=None):
    """TFT_ROUND_DATA (experiments/README.md section 9, variant T1): numbers for the round's board decision, all
    computed from the state itself - what the components on the item bench make now, empty team slots, owned units not
    fielded, what spending does to interest, HP lost this stage. Nothing is added unless the variant is played."""
    import os
    if not os.environ.get('TFT_ROUND_DATA'):
        return []
    out = []
    comps = [it for it in (items or []) if it in COMPONENTS]
    if comps:
        counts = Counter(comps)
        makes = sorted({it for it, (a, b) in RECIPES.items()
                        if a in counts and b in counts and (a != b or counts[a] >= 2)})
        out.append(f"Components on the item bench: {len(comps)} ("
                   + ', '.join(f'{c} x{n}' if n > 1 else c for c, n in counts.items()) + '). Completed items they make '
                   f"now, two on one unit: {', '.join(makes) or 'none'}. An item on the item bench adds nothing to a "
                   'fight; on a unit it does from the next fight on.')
    if any(it and is_remover(it) for it in (items or [])):
        out.append('An Item Remover is on the item bench: items given now can be moved to another unit later.')
    used = roster.slots_used()
    if slots and used < slots:
        out.append(f'Team slots: {used} of {slots} used - {slots - used} empty.')
    bench = [u for u in roster.bench() if u.name]
    if bench:
        out.append('Owned units not fielded: ' + ', '.join(
            f"{u.name}{'*' * u.star} ({CATALOG.unit(u.name).cost if CATALOG.unit(u.name) else '?'}g)" for u in bench) + '.')
    g = scr.gold
    if g is not None and os.environ.get('TFT_ROUND_DATA') != 'T6':      # (T6: no interest table - E9.2/E9.3)
        steps = [f'spend {g - keep} -> {keep}g, interest {min(5, keep // 10)}' for keep in (50, 40, 30, 20, 10, 0)
                 if keep <= g][:5]
        out.append(f'Gold {g}, interest now {min(5, g // 10)} a round: ' + '; '.join(steps) + '.')
    stage = str(scr.stage or '').split('-')[0]
    if hp_log and stage:
        this = sum(a - b for s, a, b in hp_log if str(s).startswith(stage + '-') and a is not None and b is not None)
        out.append(f"HP {scr.hp}: lost {this} so far in stage {stage}; per round: "
                   + ', '.join(f'{s} -{a - b}' for s, a, b in hp_log[-4:] if a is not None and b is not None) + '.')
        # T4: the last three player rounds' loss carried forward - how many rounds that HP lasts at the same rate
        pvp = [(s, a - b) for s, a, b in hp_log if a is not None and b is not None and is_pvp(s)][-3:]
        rate = sum(d for _, d in pvp) / len(pvp) if pvp else 0
        if os.environ.get('TFT_ROUND_DATA') in ('T4', 'T5', 'T6') and scr.hp and rate > 0:
            out.append(f'HP lost per player round over the last {len(pvp)}: {rate:.1f} on average; at that rate '
                       f'{scr.hp} HP lasts about {math.ceil(scr.hp / rate)} more player rounds.')
    return (['ROUND DATA (computed from the state above):'] + ['- ' + l for l in out]) if out else []


_UNIT_ITEMS = {}


def item_fit_lines(roster, items):
    """TFT_ROUND_DATA=T3 (experiments/README.md section 9): for each completed item the item bench's components make
    now, how often MetaTFT's final boards (data/meta/unit_items.json, tools/fetch_unit_items.py) put it on each champion
    we own and on the comp's carries, and how those boards placed - the data for items that suit the board now and the
    comp later. Nothing without the variant or without the data file."""
    import os
    from pathlib import Path
    if os.environ.get('TFT_ROUND_DATA') not in ('T3', 'T4', 'T5', 'T6'):  # (T4 = T3 + the HP projection, T5 = T4 +
        # levelling, T6 = T5 without the interest table)
        return []
    if not _UNIT_ITEMS:
        p = Path(__file__).resolve().parent.parent / 'data' / 'meta' / 'unit_items.json'
        try:
            _UNIT_ITEMS.update(json.loads(p.read_text(encoding='utf-8'))['units'])
        except (OSError, ValueError, KeyError):
            return []
    counts = Counter(it for it in (items or []) if it in COMPONENTS)
    makes = sorted({it for it, (a, b) in RECIPES.items() if a in counts and b in counts and (a != b or counts[a] >= 2)})
    if not makes:
        return []
    owned = list(dict.fromkeys(u.name for u in roster.units if u.name and not u.missing))
    comp = _fixed_comp()
    carries = list(getattr(comp, 'carries', None) or []) if comp else []

    def on(unit, item):
        row = next((r for r in _UNIT_ITEMS.get(unit, {}).get('items', []) if r[0] == item), None)
        return f'{unit} ({round(row[3] * 100)}%, avg {row[1]:.2f})' if row and row[3] >= 0.05 else None
    out = ["ITEM FIT (MetaTFT final boards, Platinum+: % = share of that champion's boards with the item, avg = their "
           'average place):']
    for it in makes:
        mine = ', '.join(x for x in (on(u, it) for u in owned) if x) or 'none of yours'
        line = f'- {it} ({" + ".join(RECIPES[it])}): on your champions {mine}'
        if comp:
            line += f"; on the comp's carries {', '.join(x for x in (on(u, it) for u in carries) if x) or 'none'}"
        out.append(line + '.')
    return out


def bench_makes_lines(items):
    """The completed items two components on the item bench make, one line (E9.2: with what the components make in the
    question, components were given in 84% of recorded questions instead of 65%). Run 20261023-045645 3-1: a Giant's
    Belt and a Negatron Cloak (Evenshroud) went onto two units "for Morgana and Sentinel" - two half-items through a
    stage 3 that cost 81 -> 53. (The TFT_ROUND_DATA variants say it in their own block.)"""
    import os
    comps = [it for it in (items or []) if it in COMPONENTS]
    if len(comps) < 2 or os.environ.get('TFT_ROUND_DATA'):
        return []
    counts = Counter(comps)
    # emblems too (a Spatula or Frying Pan + a component; the owner's note 2026-10-23: g10 kept a Spatula on the item
    # bench from 3-1 to the end next to a Sparring Gloves, a Giant's Belt and later a Tear - nothing said what it makes)
    makes = sorted(f'{a} + {b} = {it}' for it, (a, b) in list(RECIPES.items()) + [(e, r) for e, r in EMBLEMS.items() if r]
                   if a in counts and b in counts and (a != b or counts[a] >= 2))
    return [f"The item bench's components make now (both on one unit): {'; '.join(makes)}."] if makes else []


def round_index(stage):
    """A round's place in the game counting planning phases (stage 1: 1-1 to 1-4; from stage 2 the carousel, x-4, is
    no round of ours), or None."""
    try:
        a, b = (int(x) for x in str(stage).split('-'))
    except ValueError:
        return None
    return b if a == 1 else 4 + (a - 2) * 6 + (b if b < 4 else b - 1)


def component_idle(items, ages, stage):
    """{slot: ' (N rounds on the item bench; pairs with X (slot k) -> Item)'} for each component on the item bench
    (request e23, game 40: a Needlessly Large Rod and a Sparring Gloves sat there from 3-1 to 3-7 - "we hold the Rod
    and Gloves" every round, "out of habit"). ages: {component: [the stage each copy came, oldest first]}."""
    comps = [(k, it) for k, it in enumerate(items or []) if it in COMPONENTS]
    now = round_index(stage)
    out, seen = {}, Counter()
    for k, it in comps:
        first = (ages or {}).get(it) or []
        came = first[seen[it]] if seen[it] < len(first) else None
        seen[it] += 1
        n = now - round_index(came) if now is not None and round_index(came) is not None else None
        pairs = [f'{b} (slot {j}) -> {BY_PAIR[tuple(sorted((it, b)))]}' for j, b in comps
                 if j != k and tuple(sorted((it, b))) in BY_PAIR]
        parts = ([f'{n} round{"s" if n != 1 else ""} on the item bench (since {came})' if n else
                  'came this round'] if n is not None else []) + ([f"pairs with {'; '.join(pairs)}"] if pairs else [])
        if parts:
            out[k] = f" ({'; '.join(parts)})"
    return out


def completes_lines(roster, items):
    """Fielded units holding a lone component, and the completed item each component on the item bench makes there
    (run 20261022-090637: Cassiopeia's lone Tear waited from 3-6 to 5-5 and Elise's from 3-1 to 5-2, while 4-1 sent a
    B.F. Sword - her Spear of Shojin - to a 1-star Caitlyn and a Chain Vest to Rakan)."""
    bench = {}
    for k, it in enumerate(items or []):
        if it in COMPONENTS:
            bench.setdefault(it, k)
    out = []
    for u in roster.board() if bench else []:
        held = [i for i in u.items or [] if i]
        lone = next((i for i in held if i in COMPONENTS), None)
        if lone is None or THIEF in held:
            continue
        makes = [f'{it} (slot {k}) = {BY_PAIR[tuple(sorted((lone, it)))]}' for it, k in bench.items()
                 if tuple(sorted((lone, it))) in BY_PAIR]
        # an emblem only on a unit without its trait (the holder gains the trait; a Lunar unit gains nothing from it)
        traits = CATALOG.unit(u.name).traits if u.name and CATALOG.unit(u.name) else ()
        makes += [f'{it} (slot {k}) = {em}' for it, k in bench.items()
                  for em in [EMBLEM_PAIRS.get(tuple(sorted((lone, it))))] if em and trait_of(em) not in traits]
        if makes:
            out.append(f'  [id {u.id}] {u.name} holds a lone {lone}: + ' + '; + '.join(makes))
    return ['Items a give completes on a fielded unit (its lone component + one from the item bench):'] + out         if out else []


def emblem_lines(roster, items):
    """While an emblem, a Spatula or a Frying Pan is on the item bench or alone on a fielded unit: which emblems add a
    trait level to the target board (the fixed comp's; else the board) and how each is made. g13 kept a Blossom, a
    Rapidfire and a Primal Emblem (all three from stage 1) and two Reforgers on the item bench to the end ("three
    emblems we can't use"), g18 an Inferno, a Blackthorn and an Executioner Emblem and a Reforger, g10 a Spatula and a
    Reforger from 3-1."""
    def lone_maker(u):
        held = [i for i in u.items or [] if i]
        return len(held) == 1 and held[0] in MAKERS
    if not any(is_emblem_item(it) for it in items or []) and not any(map(lone_maker, roster.board())):
        return []
    comp = _fixed_comp()
    units = list(comp.units) if comp else [u.name for u in roster.board() if u.name]
    line = board_line(units, 'your target board' if comp else 'your board')
    return [line] if line else []


def mechanics_lines(roster):
    """Board places with a trait effect, every round while the trait is active (the user, 2026-10-11: the model
    never used the Blackthorn hex or picked the BFF rider - who stands where must be in the state)."""
    from tft_player.state import BFF, board_traits
    fielded = roster.fielded()
    count = lambda trait: board_traits(fielded)[trait]
    out = []
    if count('Blackthorn') >= 2:
        from tft_player.trait_text import level_effect
        occ = roster.at(('board', 1, 3))
        gain = level_effect('Blackthorn', count('Blackthorn'))
        out.append('Blackthorn hex [1, 3]: ' + (
            f'{occ.name or "?"}{"*" * occ.star} (id {occ.id}) stands there - sacrificed at combat start'
            if occ is not None and not occ.slotless else
            'EMPTY - nothing is sacrificed at combat start, so the team gets NO Blackthorn bonus this round'
            + (f" (Blackthorn {count('Blackthorn')}: {gain} - it comes only from a sacrifice)" if gain else '')))
    bff = next((u for u in roster.units if u.name == BFF and not u.missing), None)
    snack = " The BFF's ability (Snacktime!): it eats enemies it damages under 16% max Health."
    if bff is not None:
        r, c = bff.loc[1:]
        out.append(f'BFF: at hex [{r}, {c}], no rider yet (not a champion, no team slot). A link from the BFF to one '
                   f'of your fielded Sprykin makes that one its rider (link: {{"from": [{r}, {c}], "to": <unit id>}}); '
                   'with no link, the game picks the rider when combat starts. From its first rider on, the BFF has '
                   'no hex of its own.' + snack)
    elif count('Sprykin') >= 3:
        rider = next((u for u in fielded if getattr(u, 'riding', None)), None)
        out.append('BFF: ' + (f'ridden by {rider.name or "?"}{"*" * rider.star} (id {rider.id}) at hex '
                              f'[{rider.loc[1]}, {rider.loc[2]}]' if rider else
                              'its rider is not known (one of your fielded Sprykin, shown on the BFF)')
                   + '. To change the rider: a link from the rider to another fielded Sprykin (link: {"from": '
                     '<rider id or [row, col]>, "to": <unit id>}) - the BFF goes to that champion; neither moves.'
                   + snack)
    from tft_player.trait_text import pair_hosts
    for host in [u for u in fielded if u.name in pair_hosts()]:
        ally = roster.resolve(getattr(host, 'paired', None)) if getattr(host, 'paired', None) else None
        out.append(f'{host.name} (id {host.id}): ' + (
            f'paired with {ally.name or "?"}{"*" * ally.star} (id {ally.id}) - the pair holds through moves and '
            f'rounds' if ally is not None and not ally.missing else
            f'not paired. To pair an ally: link {{"from": <ally id>, "to": {host.id}}} (the ally is dragged onto '
            f'{host.name}; a line then joins them)'))
    raw = getattr(roster, 'marks', [])
    marks = [(i, roster.resolve(uid), st) for i, uid, st in raw]
    if raw and marks[0][1] is None:
        # the latest Alpha is no longer ours (sold): no line at all left the Mark looking unused-by-choice (game 41
        # 4-1..4-5: Cinderling, the Alpha, sold at 3-7; the game's "Don't forget to use your Alpha Mark!" three phases)
        uid, st = raw[0][1], raw[0][2]
        name = (getattr(roster, 'mark_names', None) or {}).get(uid) or 'a unit'
        out.append(f'Alpha Mark (stays on the item bench): your last Alpha, {name} (id {uid}, marked at {st}), is no '
                   f'longer yours (sold) - no champion you own is your Alpha now'
                   + ''.join(f'; marked before it: {u.name or "?"}{"*" * u.star} (id {u.id}, stage {s_}) - still an '
                             f'Alpha only with Omega Riftbeast' for _, u, s_ in marks[1:2] if u is not None))
        marks = []
    marks = [(i, u, st) for i, u, st in marks if u is not None and not u.missing]
    if marks:
        out.append('Alpha Mark (stays on the item bench) last used on: ' + '; '.join(
            f'{u.name or "?"}{"*" * u.star} (id {u.id}, stage {st})' for _, u, st in marks[:2])
            + ' - without Omega Riftbeast only the latest is your Alpha')
    return out


# traits that pay only while fielded, and pay more the earlier: what their payoff is, for the comp's line
GATHER = {'Coven': 'loot at 40/85/130/185/250 Essence, worth ~4/10/16/24/40 gold'}


def gather_lines(scr, roster):
    """For the fixed comp's gathering trait (Coven), every planning state from stage 2: how many are fielded, what it
    gathers now, and the trait's units owned off the board and in the shop. Run 20261022-074359 (5th): Coven 3 only
    at 4-6 - two Camilles benched and sold as "off-comp", Caitlyn skipped at 3-5 with 41 gold - so no Essence through
    eight lost fights; in all four top-2 games of the account Coven 3 was fielded by 3-1."""
    comp = _fixed_comp()
    try:
        stage = int(str(scr.stage).split('-')[0])
    except (TypeError, ValueError):
        stage = 0
    if comp is None or stage < 2:
        return []
    from tft_player.state import board_traits
    from tft_player.trait_text import level_effect
    fielded = roster.fielded()
    count = board_traits(fielded)
    out = []
    for trait, payoff in GATHER.items():
        members = [n for n in CATALOG.units if trait in CATALOG.unit(n).traits]
        if not any(trait in (CATALOG.unit(n).traits if CATALOG.unit(n) else ()) for n in comp.units):
            continue
        th = CATALOG.traits.get(trait) or ()
        n = count[trait]
        on = sorted({u.name for u in fielded if u.name in members})
        now = max((t for t in th if t <= n), default=None)
        nxt = next((t for t in th if t > n), None)
        gets = (f'gathering {level_effect(trait, now)}' if now else
                f'NOTHING gathered - {th[0] - n} more fielded makes {trait} {th[0]} ({level_effect(trait, th[0])})')
        more = f'; {trait} {nxt}: {level_effect(trait, nxt)}' if now and nxt else ''
        off = [f"{u.name}{'*' * u.star}" for u in roster.bench() if u.name in members]
        shop = [f'{name} (slot {i}, {CATALOG.unit(name).cost}g)' for i, name in enumerate(scr.shop or [])
                if name in members]
        out.append(f"{trait} {n} fielded ({', '.join(on) or 'none'}): {gets}{more}; {payoff}. Owned, not fielded: "
                   f"{', '.join(off) or 'none'}. In the shop: {', '.join(shop) or 'none'}.")
    return out


def _unit_line(u):
    if getattr(u, 'slotless', False):
        from tft_player.state import BFF, board_objects
        where = f'hex ({u.loc[1]},{u.loc[2]})' if u.loc[0] == 'board' else f'bench slot {u.loc[1]}'
        from tft_player.state import PLACEABLES
        what = ("the Sprykin trait's Big Furry Friend: not a champion, uses no team slot, gone during combat"
                if u.name == BFF else
                f'not a champion; recorded as using no team slot (not confirmed): {PLACEABLES[u.name]}'
                if u.name in PLACEABLES else f'not a champion, uses no team slot: {board_objects().get(u.name, "")}')
        return f"  [id {u.id}] {u.name} ({what}) at {where}"
    traits = '/'.join(u.traits) if u.info else '?'
    if getattr(u, 'evolved', None):
        traits += f" (evolved: +{', +'.join(u.evolved)})"
    if u.info and u.slots > 1:
        traits += f', uses {u.slots} team slots'
    held = [('an unidentified item' if i == 'unknown' else i) for i in u.items]
    if getattr(u, 'items_read', False):
        items = f" items [{', '.join(held)}]" if held else ' no items'
        if THIEF in u.items:
            items += " (Thief's Gloves: full - the game refuses any other item on it; its 2 items are random each round)"
        elif len([i for i in u.items if i != 'unknown']) >= 3:
            items += ' (item slots full: 3/3)'
    else:
        items = f" items not read yet{f' (last known: {held})' if held else ''}"
    where = f'hex ({u.loc[1]},{u.loc[2]})' if u.loc[0] == 'board' else f'bench slot {u.loc[1]}'
    row = f', {ROW_NAMES.get(u.loc[1], "?")} row' if u.loc[0] == 'board' else ''
    if getattr(u, 'riding', None) and u.loc[0] == 'board':
        where += ', riding the BFF'
    kind = ('melee' if u.info.melee else 'ranged') if u.info else '?'
    # a unit only under a silent look check (re-read in idle time) is not flagged to the model: only
    # a read that finds another champion is reported (run 033507: 38 false alarms led to ~15 board checks)
    doubt = getattr(u, 'unverified', None) and not str(u.unverified).startswith('looks different')
    flag = (f' [MISSING: {u.missing}]' if getattr(u, 'missing', None) else
            f' [UNVERIFIED: {u.unverified}]' if doubt else '')
    badge = getattr(u, 'badge_star', None)
    if badge and badge[0] != u.star and not getattr(u, 'missing', None):
        flag += (f' [STAR: its badge on the screen reads {badge[0]}-star since {badge[1]}, the records '
                 f'{u.star}-star; its panel is read to settle it]')
    return (f"  [id {u.id}] {u.name or '? (unidentified)'} {'*' * u.star} "
            f"({u.info.cost if u.info else '?'}g {kind}, {traits}) at {where}{items}{row}{flag}")


# the board's rows by hex row (request e37, game 44: Leona at (0,0) behind an empty front row at 2-2, ranged Gromp at
# (3,2) in front of melee Murkwolf at 3-5, and Krug at (2,4) called "the front row" at 4-1). The row goes after the
# items: agent/model/plan_board.py reads "at hex (r,c) items [...]" from the recorded states
ROW_NAMES = {3: 'front', 2: 'second', 1: 'third', 0: 'back'}


def rows_line(board):
    """Who stands in each board row, front (row 3, facing the enemy) to back (row 0); '' with no unit fielded."""
    units = [u for u in board if u.loc[0] == 'board' and not getattr(u, 'slotless', False)]
    if not units:
        return ''
    kind = lambda u: ('melee' if u.info.melee else 'ranged') if u.info else '?'
    part = lambda r: ', '.join(f"{u.name or '?'} ({kind(u)}, {u.loc[1]},{u.loc[2]})"
                               for u in sorted(units, key=lambda u: u.loc[2]) if u.loc[1] == r) or 'empty'
    return ('Board rows (row 3 = front, facing the enemy): '
            + '; '.join(f'{ROW_NAMES[r]} (row {r}): {part(r)}' for r in (3, 2, 1, 0)))


def tier_note(board, name):
    """What fielding this champion adds to the board's traits (arithmetic, as the game counts them): the tiers it
    completes, or none - the owner, game 81 3-3 (x31.2): "why field alistar? it has 0 traits with the current board. it
    should look for a units with the traits available"."""
    from types import SimpleNamespace
    from tft_player.state import board_traits
    info = CATALOG.unit(name or '')
    units = [u for u in board if u.name and not getattr(u, 'slotless', False) and not getattr(u, 'missing', None)]
    if info is None:
        return ''
    if any(u.name == name for u in units):
        return ' - fielded already: another copy on the board completes no trait tier'
    now = board_traits(units)
    new = board_traits(units + [SimpleNamespace(name=name, traits=info.traits, items=())])
    got = [f'{t} {new[t]}' for t in sorted(new) if any(now[t] < x <= new[t] for x in CATALOG.traits.get(t) or ())]
    return f" - fielded, it completes {', '.join(got)}" if got else ' - fielded, it completes no trait tier'


def bench_cost_lines(roster, cap):
    """What each fielded unit's trait count holds up: the breakpoints lost if it alone leaves the board, and the
    free team slots a benched 2-slot unit needs (request e253: game 80 6-2 freed Elder Dragon's two slots by benching
    Pebbles**, Invoker 2 -> 1, while Cinderling***, whose traits lost nothing, stayed; games 79 6-3 and 80 6-3 alike)."""
    from tft_player.state import board_traits
    fielded = [u for u in roster.board() if u.name and not u.slotless]
    if not fielded:
        return []
    now = board_traits(roster.board())
    free, holds = [], []
    for u in fielded:
        after = board_traits([x for x in roster.board() if x is not u])
        lost = [f'{t} {now[t]} -> {after[t]}' for t in sorted(now)
                if any(after[t] < x <= now[t] for x in CATALOG.traits.get(t) or ())]
        tag = f"{u.name}{'*' * u.star} (unit {u.id}{', 2 slots' if u.slots > 1 else ''})"
        (holds.append(f"{tag}: {', '.join(lost)}") if lost else free.append(tag))
    out = ['Benching one fielded unit alone (trait arithmetic, the units only): no trait tier lost - '
           + (', '.join(free) or 'none') + '; tiers lost - ' + ('; '.join(holds) or 'none') + '.']
    benched = [u for u in roster.bench() if u.name and not u.missing]
    if benched:
        out.append('Fielding one bench unit (trait arithmetic): ' + '; '.join(
            f"{u.name}{'*' * u.star} (unit {u.id}){tier_note(roster.board(), u.name)}" for u in benched) + '.')
    for d in roster.bench():
        if d.name and d.slots > 1 and d.name not in {u.name for u in fielded}:
            # its own traits count with it: the units it makes redundant lose nothing once it is in
            base = roster.board() + [d]
            now_d = board_traits(base)
            free_d = [f"{u.name}{'*' * u.star} (unit {u.id})" for u in fielded
                      if not any(board_traits([x for x in base if x is not u])[t] < x <= now_d[t]
                                 for t in now_d for x in CATALOG.traits.get(t) or ())]
            out.append(f'With {d.name} (unit {d.id}) fielded as well: no trait tier lost by benching one of - '
                       f"{', '.join(free_d) or 'none'}.")
    open_ = (cap - roster.slots_used()) if isinstance(cap, int) else None
    for u in roster.bench():
        if u.name and u.slots > 1 and open_ is not None:
            out.append(f'{u.name} (unit {u.id}, bench) takes {u.slots} team slots: {max(open_, 0)} free now'
                       + (f' - {u.slots - max(open_, 0)} fielded unit(s) must be benched or sold first (it cannot be '
                          f'swapped in for one unit)' if open_ < u.slots else ''))
    return out


# held augments whose bonus needs a unit alone in its board row (request e44, game 45: Solo Plate held from 2-1, its
# Stoneplate holder Rakan shared row 3 from 3-1 to 3-6 and the state never said so): name -> the item its holder
# holds, or None for any unit
ALONE_IN_ROW = {'Solo Plate': 'Gargoyle Stoneplate', 'Stand Alone': None}


def placement_lines(board, augments):
    """Per held augment with a row condition (ALONE_IN_ROW): whether the board now meets it - objective, no advice."""
    units = [u for u in board if u.loc[0] == 'board' and not getattr(u, 'slotless', False)]
    row = lambda u: [v for v in units if v is not u and v.loc[1] == u.loc[1]]
    star = lambda u: f"{u.name or '?'}{'*' * (u.star or 1)}"
    out = []
    for a in augments or ():
        key = next((k for k in ALONE_IN_ROW if re.sub(r'(\s+I+)?\+?$', '', str(a).split(' (')[0].strip()) == k), None)
        if key is None:
            continue
        item = ALONE_IN_ROW[key]
        if item:
            hold = [u for u in units if any(str(i).split(' (')[0] == item for i in u.items or ())]
            if not hold:
                out.append(f'{a}: no fielded unit holds {item} - inactive')
                continue
            parts = [f'{item} holder {star(u)} ' + (f"shares row {u.loc[1]} with {', '.join(star(v) for v in row(u))} "
                                                    f'- inactive' if row(u) else f'is alone in row {u.loc[1]} - active')
                     for u in hold]
            out.append(f'{a} (bonus only for a unit alone in its row): ' + '; '.join(parts))
        else:
            alone = [u for u in units if not row(u)]
            out.append(f'{a} (bonus only for a unit alone in its row): units alone in their row now: '
                       + (', '.join(f'{star(u)} (row {u.loc[1]})' for u in alone) or 'none'))
    return out


def active_traits(units):
    from tft_player.state import board_traits
    counts = board_traits(units)          # the BFF, a Training Dummy: no traits; Elder Dragon: +2 Riftbeast
    out = []
    for t, c in counts.most_common():
        th = CATALOG.traits.get(t) or ()
        nxt = next((x for x in th if x > c), None)
        out.append(f"{t} {c}{'/' + str(nxt) if nxt else ''}{' (active)' if th and c >= th[0] else ''}")
    return ', '.join(out) or 'none'


def next_level_lines(scr, roster, lvl):
    """CLIMB: the next slot's XP cost, gold, interest and distinct owned comp-trait count."""
    import os
    if os.environ.get('TFT_CLIMB') != '1':
        return []
    from tft_player import board_score
    team = getattr(scr, 'team_size', None) or lvl
    nxt = board_score.next_slot_line(roster, scr.gold, team, getattr(scr, 'level', None), getattr(scr, 'xp', None),
                                     getattr(scr, 'xp_per_buy', None) or 4)
    return [nxt] if nxt else []


def board_after(plan, roster, where=False):
    """The units fielded after the plan's board moves, sells and benchings, as the executor makes them: a unit placed
    on an occupied hex swaps with its occupant (the occupant takes the unit's old place, the bench for a bench unit),
    unless the occupant has a planned hex of its own; a champion named by "board" (a buy) is fielded too. -> (units
    fielded now, units fielded after), and with where=True {unit id: its place after} too."""
    from types import SimpleNamespace
    now = [u for u in roster.units if u.on_board]
    if not isinstance(plan, dict):
        return (now, now, {u.id: u.loc for u in now}) if where else (now, now)
    gone = {e.get('id') for e in plan.get('bench') or [] if isinstance(e, dict)} |         {v for v in plan.get('sell') or [] if isinstance(v, int)}
    at = {u.id: u.loc for u in roster.units if not u.missing and u.id not in gone}
    by_id = {u.id: u for u in roster.units}
    entries = [e for e in plan.get('board') or [] if isinstance(e, dict) and isinstance(e.get('hex'), (list, tuple))
               and len(e['hex']) == 2]
    planned = {e.get('id') for e in entries if e.get('id') is not None}
    extra = []
    for e in entries:
        h = ('board', *(int(x) for x in e['hex']))
        uid = e.get('id')
        if uid is not None and uid not in at:
            continue                             # a unit sold, benched or unknown here
        occ = next((k for k, l in at.items() if l == h and k != uid), None)
        old = at.get(uid, ('bench', -1))
        if occ is not None:
            at[occ] = old if occ not in planned else ('bench', -1)   # (its own hex comes with its entry)
        for x in extra:
            if x.loc == h:
                x.loc = ('bench', -1)
        if uid is not None:
            at[uid] = h
        elif e.get('name') and CATALOG.unit(e['name']):
            extra.append(SimpleNamespace(name=e['name'], traits=CATALOG.unit(e['name']).traits, items=[], loc=h))
    after = [by_id[k] for k, l in at.items() if l[0] == 'board'] + [x for x in extra if x.loc[0] == 'board']
    return (now, after, at) if where else (now, after)


def board_after_line(plan, roster):
    """CLIMB request e2 (game 35 6-1: the player's own next_round put Taric on Scuttlecrab's hex, Riftbeast fell 7 ->
    6 and 17 HP were lost): the traits the board has after this plan's moves, when they differ from now. -> (line,
    a breakpoint is lost) or (None, False)."""
    from tft_player.state import board_traits
    now, after = board_after(plan, roster)
    a, b = board_traits(now), board_traits(after)
    if a == b and {getattr(u, 'id', u.name) for u in now} == {getattr(u, 'id', u.name) for u in after}:
        return None, False
    changed, lost, gained = [], False, []
    for t in sorted(set(a) | set(b), key=lambda t: (-max(a[t], b[t]), t)):
        if a[t] != b[t]:
            th = CATALOG.traits.get(t) or ()
            drop = [x for x in th if b[t] < x <= a[t]]
            gained += [f'{t} {x}' for x in th if a[t] < x <= b[t]]
            lost = lost or bool(drop)
            changed.append(f"{t} {a[t]} -> {b[t]}" + (f" (the {drop[-1]} breakpoint is lost)" if drop else ''))
    return (f"board after this plan's moves: {len(after)} units; {active_traits(after)} (changed from now: "
            f"{', '.join(changed) or 'no trait count change'}; gained tiers: {', '.join(gained) or 'none'}; lost tiers: "
            + (', '.join(f'{t} {x}' for t in sorted(a) for x in CATALOG.traits.get(t) or ()
                         if b[t] < x <= a[t]) or 'none') + ')'), lost


def owned_entries(plan, roster):
    """The plan with only the board and bench entries that hold now: an entry waiting on a buy ("when_owned" a
    champion not owned yet, or a champion named and not owned) is left out."""
    owned = {u.name for u in roster.units if not u.missing}
    holds = lambda e: isinstance(e, dict) and RoundPlan.arrival_ready(e, roster) and (
        'id' in e or not e.get('name') or e['name'] in owned)
    return dict(plan, board=[e for e in plan.get('board') or [] if holds(e)],
                bench=[e for e in plan.get('bench') or [] if holds(e)])


def prepare_comp(plan, roster, team, item_bench, combat=False, augments=()):
    """Reapply the owner's mechanical comp rules to the ACTUAL roster after purchases/moves (e57).
    No model call; unavailable units and unread holders never become assumed power."""
    from tft_player.comp_rules import carry_swap, rules
    from tft_player.state import board_traits
    r = rules(getattr(_fixed_comp(), 'name', None))
    if not r.get('entry_hexes'):
        return
    note = lambda text: plan.setdefault('_notes', []).append(('comp rule', text))

    def enter(u, victim, default):
        _, after, at = board_after(plan, roster, where=True)
        old = list(at.get(victim.id, victim.loc)[1:]) if victim else None
        occupied = {tuple(loc[1:]): uid for uid, loc in at.items() if loc[0] == 'board'}
        occupied.update({tuple(v.loc[1:]): None for v in after if not hasattr(v, 'id') and v.loc[0] == 'board'})
        dest = list(default)
        # Respect a deliberate position, otherwise use this champion's approach/front hex.
        explicit = next((e for e in plan.get('board') or [] if e.get('id') == u.id
                         or e.get('name') == u.name), None)
        if explicit:
            dest = list(explicit['hex'])
        # Keep a row-isolated tank's condition: Sentinel takes the second row instead.
        if u.name == 'Sentinel' and any(str(a).split(' (')[0] in ALONE_IN_ROW for a in augments):
            dest = [2, 4]
        occupant_id = occupied.get(tuple(dest))
        occ = next((v for v in after if (occupant_id is not None and getattr(v, 'id', None) == occupant_id)
                    or (not hasattr(v, 'id') and v.loc == ('board', *dest))), None)
        entries = list(plan.get('board') or [])
        removed = {u.id} | ({victim.id} if victim else set())
        entries = [e for e in entries if e.get('id') not in removed and e.get('name') != u.name]
        if occ is not None and occ is not victim and occ is not u:
            free = old or next(([row, col] for row in range(4) for col in range(7)
                                if (row, col) not in occupied), None)
            if free is None:
                return
            if hasattr(occ, 'id'):
                entries = [e for e in entries if e.get('id') != occ.id]
                entries.append({'id': occ.id, 'hex': free})
            else:
                entries = [e for e in entries if e.get('name') != occ.name]
                entries.append({'name': occ.name, 'hex': free})
        if victim:
            slot = u.loc[1] if not u.on_board else next(iter(roster.free_bench()), None)
            if slot is None:
                return
            plan['bench'] = [e for e in plan.get('bench') or [] if e.get('id') not in removed]
            plan['bench'].append({'id': victim.id, 'slot': slot})
        else:
            plan['bench'] = [e for e in plan.get('bench') or [] if e.get('id') != u.id]
        plan['board'] = entries + [{'id': u.id, 'hex': dest}]
        plan['sell'] = [uid for uid in plan.get('sell') or [] if uid != u.id]
        note(f'{u.name} fielded' + (f' for {victim.name}' if victim else ' in a free slot')
             + f' at ({dest[0]},{dest[1]})')

    after = board_after(plan, roster)[1]
    # no comp rule moves units while a fielded unit is unidentified - its traits are not counted, so no count is the
    # board's (request e194 D2: the '?' 3-item carry at (0,6), and carry_swap fielded the bench copy for Scuttlecrab,
    # Riftbeast 7 -> 6 on the real board) - and none takes a paired unit off the board (e194 D: Taric's ally)
    unknown = any(getattr(x, 'name', 1) is None for x in after)
    pair = lambda v: v is not None and bool(getattr(v, 'paired', None))
    if not combat and not unknown:
        excluded = set(plan.get('sell') or []) | {e.get('id') for e in plan.get('bench') or []}
        for out in list(after):
            if not hasattr(out, 'id') or out.missing or out.unverified or pair(out) or plan.get('keep_board'):
                continue
            copies = [u for u in roster.bench() if u.name == out.name and u.star > out.star
                      and not u.missing and not u.unverified and u.id not in excluded and u not in after]
            if not copies or out.items and not any(is_remover(i) for i in item_bench or () if i):
                continue
            inn = max(copies, key=lambda u: u.star)
            _, _, at = board_after(plan, roster, where=True)
            enter(inn, out, at.get(out.id, out.loc)[1:])
            # Preserve the job and its incoming equipment across the copy swap.
            for e in plan.get('items') or []:
                if e.get('id') == out.id and not e.get('use'):
                    e['id'] = inn.id
            after = board_after(plan, roster)[1]
        if not plan.get('bench_carry') and not any(u.name == 'Brambleback' for u in after):
            sw = carry_swap([u for u in after if hasattr(u, 'id')], roster.units,
                            getattr(_fixed_comp(), 'name', None)) if all(hasattr(u, 'id') for u in after) else None
            if sum(getattr(u, 'slots', 1) for u in after) < team:
                copies = [u for u in roster.bench() if u.name == 'Brambleback' and not u.unverified]
                if copies:
                    enter(max(copies, key=lambda u: u.star), None, r['entry_hexes']['Brambleback'])
            elif sw:
                enter(*sw, r['entry_hexes']['Brambleback'])
        after = board_after(plan, roster)[1]
        if not plan.get('keep_board') and not any(u.name == 'Sentinel' for u in after):
            copies = [u for u in roster.bench() if u.name == 'Sentinel' and not u.unverified]
            if copies:
                u = max(copies, key=lambda u: u.star)
                a = board_traits(after)
                free = sum(getattr(v, 'slots', 1) for v in after) < team
                victims = [None] if free else [v for v in after if getattr(v, 'id', None) is not None
                    and not pair(v) and v.name not in {'Pebbles', 'Krug', 'Murkwolf', 'Brambleback', 'Elder Dragon', 'Taric'}
                    and not any(i not in COMPONENTS for i in v.items or ())]
                for v in victims:
                    b = board_traits([x for x in after if x is not v] + [u])
                    if (not any(a[t] < 2 <= b[t] for t in ('Invoker', 'Vanguard'))
                            or any(b[t] < th <= n for t, n in a.items() for th in CATALOG.traits.get(t) or ())):
                        continue
                    enter(u, v, r['entry_hexes']['Sentinel'])
                    break
        if not plan.get('keep_board'):
            from tft_player import board_score as BS
            _, after, at = board_after(plan, roster, where=True)
            # The plan's own choices hold (game 57 3-1 to 3-6: Mama Beak, benched by the player, was fielded on
            # Scuttlecrab's front hex five fights running): a unit the plan benches is not fielded, a unit it places
            # is not benched, and a ranged unit never takes a front-row hex.
            benched = {e.get('id') for e in plan.get('bench') or [] if isinstance(e, dict)}
            placed = {e.get('id') for e in plan.get('board') or [] if isinstance(e, dict)}
            # ... and a unit the plan sells is not fielded, a unit its gives go to is not benched (game 61 2-2, log
            # 254-263: the update sold LeBlanc and gave B.F. Sword + Chain Vest to Cinderling at (0,3); "LeBlanc
            # fielded for Cinderling" dropped the sale, Edge of Night was built on the bench, LeBlanc fought)
            benched |= {i for i in plan.get('sell') or [] if isinstance(i, int)}
            placed |= {e.get('id') for e in plan.get('items') or [] if isinstance(e, dict) and e.get('id') is not None}
            front = lambda v: at.get(v.id, v.loc)[0] == 'board' and at.get(v.id, v.loc)[1] == 3
            if all(hasattr(u, 'id') for u in after):
                triplets = lambda units: [(u.name, u.star, u.items) for u in units]
                tiers = lambda units: {t: k for t, _, k in BS.bonuses([(u.name, u.items) for u in units])}
                now_tiers, now_score = tiers(after), BS.score(triplets(after))
                marks = {m[1] for m in (getattr(roster, 'marks', None) or [])[:1]}
                best = None
                free = sum(v.slots for v in after) < team
                for u in roster.bench():
                    if (u in after or u.missing or u.unverified or not u.info or u.slots != 1
                            or u.name == 'Brambleback' and plan.get('bench_carry') or u.id in benched):
                        continue
                    ranged = not (CATALOG.unit(u.name) and CATALOG.unit(u.name).melee)
                    for v in ([None] if free else after):
                        if v is not None and (v.id in placed or ranged and front(v) or pair(v)
                                or v.name == 'Brambleback' or v.id in marks or v.id in (plan.get('keep') or [])
                                or any(i not in COMPONENTS for i in v.items or ()) or v.slots != 1
                                or v.name in {'Murkwolf', 'Krug', 'Pebbles', 'Sentinel', 'Taric'} and v.name != u.name
                                or any(x is not v and x.name == u.name for x in after)):
                            continue
                        candidate = [x for x in after if x is not v] + [u]
                        score = BS.score(triplets(candidate))
                        if tiers(candidate) == now_tiers and score > now_score and (best is None or score > best[0]):
                            best = (score, u, v)
                if best:
                    _, u, v = best
                    # A new defensive melee body gets a front hex; a carry gets its own approach hex.
                    default = r['entry_hexes'].get(u.name, (3, 3) if BS.body_key([(u.name, u.star, u.items)])[1]
                                                   else v.loc[1:] if v else (0, 0))
                    enter(u, v, default)
    if not combat or any(u.name == 'Brambleback' for u in roster.board()):
        _carry_items_move(plan, roster, item_bench)
    _equip_comp_defense(plan, roster, item_bench, combat)
    # the Alpha Mark is used at once: its holder follows the board of units we own, not entries that wait for a buy
    # (game 57 5-6, log 2478-2538: "bench Scuttlecrab when_owned Elder Dragon" and "Elder Dragon -> (1,3)" by name
    # put the Mark on Brambleback at 20 s; the roll bought no Elder Dragon, the fight's give to Scuttlecrab missed, and
    # the 8-HP fight was played with the Alpha off the owner's holder)
    after = roster.board() if combat else board_after(owned_entries(plan, roster), roster)[1]
    carry = next((u for u in after if u.name == 'Brambleback' and getattr(u, 'id', None) is not None
                  and not u.unverified), None)
    # owner-alpha-crab: seven distinct Riftbeasts default to Scuttlecrab. Count the actual fighting
    # board in combat and the submitted board in planning, never a purchased but benched champion.
    # owner-alpha-on-crab (x18.5, game 66: "the Alpha Mark should be on the crab") came after it: a fielded Scuttlecrab
    # holds the Mark whatever the count (the Reviewer, game 78 H8: the Mark went to Brambleback at 3-5, to the Crab at
    # 3-6, back to Brambleback at 3-7 - 14 s of the phases, and the Crab's order refused while it was in transit)
    crab = next((u for u in after if u.name == 'Scuttlecrab' and getattr(u, 'id', None) is not None
                 and not u.unverified), None)
    if crab is not None:
        carry = crab
    if carry and not plan.get('keep_items') and any(str(i or '').startswith('Alpha Mark') for i in item_bench or ()):
        omega = any(str(a).startswith('Omega Riftbeast') for a in augments)
        marks = (getattr(roster, 'marks', None) or [])[:2 if omega else 1]
        if carry.name == 'Scuttlecrab' and not omega:
            plan['items'] = [e for e in plan.get('items') or [] if e.get('component') != 'Alpha Mark'
                             or e.get('id') == carry.id]
        if not any(m[1] == carry.id for m in marks) and not any(
                e.get('component') == 'Alpha Mark' and e.get('id') == carry.id for e in plan.get('items') or []):
            entries = [e for e in plan.get('items') or [] if e.get('component') != 'Alpha Mark'
                       or omega and e.get('id') != carry.id]
            k = max([e.get('k', -1) for e in entries] + [-1]) + 1
            plan['items'] = entries + [{'component': 'Alpha Mark', 'id': carry.id, 'use': True, 'k': k}]
            note(f'Alpha Mark assigned to fielded {carry.name} (unit {carry.id}) this phase')


def _equip_comp_defense(plan, roster, item_bench, combat=False):
    """Use an unreserved defensive component on a retained tank; spare Vow/Helm follows a functional main tank. Not for
    a comp with an item plan: its executor gives the tank items (game 72: Evenshroud and a Chain Vest on Scuttlecrab)."""
    from tft_player import item_plan
    if item_plan.plan_for(getattr(_fixed_comp(), 'name', None)):
        return
    if plan.get('keep_items') or item_bench is None or 'unknown' in item_bench:
        return
    defense = {"Warmog's Armor", 'Sunfire Cape', 'Bramble Vest', "Dragon's Claw", 'Gargoyle Stoneplate',
               'Spirit Visage', "Protector's Vow", 'Adaptive Helm', 'Redemption', 'Steadfast Heart'}
    base = lambda it: str(it or '').split(' (')[0]
    available = Counter(base(it) for it in item_bench if it)
    available.subtract(base(e.get('component')) for e in plan.get('items') or []
                       if 'from' not in e and not e.get('_given'))
    fielded = roster.board() if combat else board_after(plan, roster)[1]
    tanks = [u for u in fielded if getattr(u, 'id', None) is not None and u.name in ('Krug', 'Sentinel', 'Scuttlecrab')
             and not u.unverified and u.items_read and THIEF not in u.items]
    forecast = {u.id: after for _, u, _, after, _ in gives_held(plan, roster)}
    held = lambda u: forecast.get(u.id, u.items)
    main = next((u for u in tanks if u.name == 'Krug'), None)
    functional = main and any(it in defense for it in held(main))
    tanks.sort(key=lambda u: (u.name != ('Sentinel' if functional else 'Krug'), -u.star))
    spare = next((it for it in ("Protector's Vow", 'Adaptive Helm') if available[it] > 0), None) if functional else None
    parts = [it for it, n in available.items() for _ in range(max(0, n)) if it in COMPONENTS]
    item = spare or (parts[0] if len(parts) == 1 and parts[0] in ("Giant's Belt", 'Chain Vest', 'Negatron Cloak') else None)
    if not item:
        return
    for u in tanks:
        if len(held(u)) >= 3:
            continue
        lone = next((i for i in held(u) if i in COMPONENTS), None)
        if lone and item in COMPONENTS and BY_PAIR.get(tuple(sorted((lone, item)))) not in defense:
            continue
        k = max([e.get('k', -1) for e in plan.get('items') or []] + [-1]) + 1
        plan.setdefault('items', []).append({'component': item, 'id': u.id, 'k': k})
        plan.setdefault('_notes', []).append(('comp rule', f'unreserved {item} assigned to retained {u.name} (unit {u.id})'))
        return


def _two_step_moves(plan, roster):
    """A unit named in two "board" entries is a swap and then a move: the first entry drops it on an occupied hex,
    which sends that hex's unit to its old place, and the last is where it ends (games 80 1-4, 81 2-1 and 82 2-3:
    "Gromp replaces Ornn ... in the back", "the first placements swap Murkwolf for Varus and Shen for Rakan; their
    subsequent moves put them on their intended hexes" - each was rejected whole as "one unit cannot occupy two board
    entries", and the swaps never ran). The unit keeps its last entry; each earlier hex's unit goes where the swap
    would have put it - the unit's old hex, or the bench - unless another entry already places it or its hex."""
    entries = [e for e in plan.get('board') or [] if isinstance(e, dict)]
    by_unit = {}
    for e in entries:
        u = roster.resolve(e['id']) if isinstance(e.get('id'), int) and not isinstance(e.get('id'), bool) else None
        if u is not None and not e.get('when_owned') and isinstance(e.get('hex'), (list, tuple)) and len(e['hex']) == 2:
            by_unit.setdefault(u.id, (u, []))[1].append(e)
    if not any(len(es) > 1 for _, es in by_unit.values()):
        return
    placed = {by_unit[k][0].id for k in by_unit}
    targets = {tuple(es[-1]['hex']) for _, es in by_unit.values()}
    targets |= {tuple(e['hex']) for e in entries if 'id' not in e and isinstance(e.get('hex'), (list, tuple))}
    benched = {e.get('id') for e in plan.get('bench') or [] if isinstance(e, dict)}
    slots = [s for s in roster.free_bench()
             if s not in {e.get('slot') for e in plan.get('bench') or [] if isinstance(e, dict)}]
    drop, extra_board, extra_bench = set(), [], []
    for u, es in by_unit.values():
        for e in es[:-1]:
            drop.add(id(e))
            x = roster.at(('board', *e['hex']))
            if x is None or x is u or x.id in placed or x.id in benched or tuple(e['hex']) in targets:
                continue
            if u.on_board and u.loc[1:] not in targets:
                extra_board.append({'id': x.id, 'hex': list(u.loc[1:])})
                targets.add(u.loc[1:])
            elif slots:
                extra_bench.append({'id': x.id, 'slot': slots.pop(0)})
            else:
                # no free bench slot for it: the swap alone runs (the unit stays on that first hex)
                drop |= {id(o) for o in es}
                drop.discard(id(e))
                break
            placed.add(x.id)
    plan['board'] = [e for e in plan.get('board') or [] if id(e) not in drop] + extra_board
    if extra_bench:
        plan['bench'] = list(plan.get('bench') or []) + extra_bench


def _reject_duplicate_drops(plan, roster, errors):
    """A "board" entry that fields a second copy of a champion already fielded, when the board's traits then lose a
    breakpoint, is dropped before it runs (game 39 4-5: Brambleback's second copy swapped onto Cinderling's hex,
    "so we keep Riftbeast 7" - a champion counts once, Riftbeast fell 7 -> 6 and the warning came after the move had
    run; 12 HP). "dup": true on the entry fields it anyway."""
    from tft_player.state import board_traits
    entries = [e for e in plan.get('board') or [] if isinstance(e, dict) and not e.get('dup')]
    if not entries:
        return
    now, after = board_after(plan, roster)
    a, b = board_traits(now), board_traits(after)
    lost = [f"{t} {a[t]} -> {b[t]} (the {[x for x in CATALOG.traits.get(t) or () if b[t] < x <= a[t]][-1]} "
            f"breakpoint is lost)" for t in sorted(a) if any(b[t] < x <= a[t] for x in CATALOG.traits.get(t) or ())]
    if not lost:
        return
    fielded = {u.id for u in now}
    bad = []
    for e in entries:
        u = roster.resolve(e['id']) if e.get('id') is not None else None
        name = u.name if u is not None else e.get('name')
        if not name or (u is not None and u.id in fielded):
            continue
        copies = [x for x in after if getattr(x, 'name', None) == name]
        if len(copies) >= 2:
            bad.append((e, name))
    for e, name in bad:
        held = [f"unit {u.id} at ({u.loc[1]},{u.loc[2]})" for u in now if u.name == name and u.loc[0] == 'board']
        name = f"{name} ({', '.join(held)})" if held else name     # which unit holds it (a misread shows here)
        errors.append(f"board: {str({k: v for k, v in e.items() if k != 'k'})}: {name} is already fielded and a "
                      f"champion counts once for its traits, so this move makes {', '.join(lost)}; the move was not "
                      f'made. To field the second copy anyway, repeat the entry with "dup": true')
        plan['board'] = [x for x in plan['board'] if x is not e]


def _trait_count_after(t, n, after, at=None):
    """The units the comp guard counted for its trait on the plan's final board, by champion, with each unit's id and
    place and the extra count some give (game 46 6-1: the plan benched Elder Dragon for Sentinel and Pebbles, and the
    refusal said only "Riftbeast falls 7 -> 6" - unit 37 at (3,2), Scuttlecrab, was misread as Pebbles, so Pebbles
    counted once for two units and the player could not see why)."""
    from tft_player.trait_text import trait_bonus
    per = {}
    for x in after:
        if getattr(x, 'name', None) and (t in (getattr(x, 'traits', None) or ())
                                         or f'{t} Emblem' in (getattr(x, 'items', None) or ())):
            per.setdefault(x.name, []).append(x)
    parts = []
    for name in sorted(per):
        where = [f"unit {x.id} at " if getattr(x, 'id', None) is not None else 'a buy at ' for x in per[name]]
        locs = [(at or {}).get(getattr(x, 'id', None)) or x.loc for x in per[name]]
        where = [w + (f"({h[1]},{h[2]})" if h and h[0] == 'board' else 'the board') for w, h in zip(where, locs)]
        extra = trait_bonus(name).get(t, 0)
        parts.append(name + (f" (counts {1 + extra})" if extra else '') + f" [{', '.join(where)}]"
                     + (' (copies count once)' if len(per[name]) > 1 else ''))
    return f"{t} counted on the board after the plan's moves: {n} - " + ('; '.join(parts) or 'none')


def _keep_comp_breakpoint(plan, roster, errors):
    """The fixed comp's own trait step (comp_rules: Riftbeast 7) is not dropped by a plan's sell, bench or swap: the
    entry that takes a champion of it off the board while no other copy is fielded is dropped before it runs (request
    e20 - games 33, 35 and 39 lost the 7 step by a swap or a sale and paid 12-17 HP; game 35 6-1 put Taric on
    Scuttlecrab's hex). "drop_breakpoint": true in the execution_plan makes the moves anyway."""
    from tft_player.comp_rules import rules
    from tft_player.state import board_traits
    r = rules(getattr(_fixed_comp(), 'name', None))
    t, keep = r.get('trait'), r.get('keep')
    if not t or not keep or plan.get('drop_breakpoint'):
        return
    now, after, at = board_after(plan, roster, where=True)
    a, b = board_traits(now)[t], board_traits(after)[t]
    if a < keep or b >= keep:
        return
    stay = {getattr(x, 'name', None) for x in after}
    leaving = [u for u in now if u.name and t in (u.info.traits if u.info else ()) and u.name not in stay]
    why = (f"it takes {{}} off the board and {t} falls {a} -> {b} (the {keep} breakpoint is lost; a champion counts "
           f"once, so no other unit keeps it); the move was not made. {_trait_count_after(t, b, after, at)}. To make it "
           f'anyway, send the plan again with "drop_breakpoint": true in execution_plan')
    for u in leaving:
        if u.id in (plan.get('sell') or []):
            errors.append(f'sell {u.id}: ' + why.format(u.name))
            plan['sell'] = [v for v in plan['sell'] if v != u.id]
        for e in [e for e in plan.get('bench') or [] if isinstance(e, dict) and e.get('id') == u.id]:
            errors.append(f"bench: unit {u.id} -> slot {e.get('slot')}: " + why.format(u.name))
            plan['bench'] = [x for x in plan['bench'] if x is not e]
        for e in [e for e in plan.get('board') or [] if isinstance(e, dict) and e.get('id') != u.id
                  and isinstance(e.get('hex'), (list, tuple)) and ('board', *(int(x) for x in e['hex'])) == u.loc]:
            errors.append(f"board: {str({k: v for k, v in e.items() if k != 'k'})}: " + why.format(u.name))
            plan['board'] = [x for x in plan['board'] if x is not e]


def _reach_comp_breakpoint(plan, roster):
    """The fixed comp's trait step (comp_rules 'keep': Riftbeast 7) the owned units can field is fielded: when the
    board after the plan's moves is short of it and benched champions of the trait not on that board would reach it,
    each goes onto the hex of a fielded unit without the trait - never the carry, the Alpha, a holder of a completed
    item, or a two-slot unit (request e52, the comp coach's rule "a plan's board must name every one of the 7"; game
    47 3-3: an earlier update sent Pebbles to the bench, the plan's board did not put him back, and the fight was
    Riftbeast 6 with Akali while all 7 were owned). "drop_breakpoint": true in the execution_plan skips it. Told in the
    next state as 'comp rule' (plan['_notes']). -> the swaps made, [(unit in, unit out)]."""
    from tft_player.comp_rules import rules
    from tft_player.state import board_traits
    r = rules(getattr(_fixed_comp(), 'name', None))
    t, keep = r.get('trait'), r.get('keep')
    if not t or not keep or not isinstance(plan, dict) or plan.get('drop_breakpoint'):
        return []
    _, after, at = board_after(plan, roster, where=True)
    if any(getattr(x, 'name', 1) is None for x in after):
        # a fielded unit not identified: its traits are not counted, so the count is not the board's (run
        # 20261103-233832 6-1: the itemized Brambleback at (0,6) was '?', "Riftbeast 6" fielded the bench copy for
        # Taric at the start of 6-2 and Vanguard 2 was lost)
        return []
    b = board_traits(after)[t]
    if b >= keep:
        return []
    has = lambda x: t in (getattr(x, 'traits', None) or ()) or f'{t} Emblem' in (getattr(x, 'items', None) or ())
    carry = (r.get('carry') or (None,))[0]
    marks = {k[1] for k in getattr(roster, 'marks', None) or [] if k[0] == 'Alpha Mark'}
    done_item = lambda x: any(i and i != 'unknown' and i not in COMPONENTS for i in getattr(x, 'items', None) or ())
    sold = set(plan.get('sell') or [])
    # the plan's own choices hold, as for the board-upgrade rule (81f33f8; request e189): a unit the plan benches is
    # not fielded, a unit it places is not taken off for one
    benched = {e.get('id') for e in plan.get('bench') or [] if isinstance(e, dict)}
    placed = {e.get('id') for e in plan.get('board') or [] if isinstance(e, dict)}
    have = {getattr(x, 'name', None) for x in after}
    ins = []
    for u in sorted(roster.units, key=lambda x: (-(x.star or 1), x.id)):
        if (u.name and u.name not in have and has(u) and at.get(u.id, ('bench',))[0] != 'board' and u.id not in sold
                and u.id not in benched
                and not u.missing and not u.unverified and getattr(u, 'slots', 1) == 1):
            have.add(u.name)
            ins.append(u)
    # never a unit paired with another (Taric and his ally: request e177, run 20261103-233832 6-1/6-2 - Taric, paired
    # with Elder Dragon at 5-5, was benched for a second Brambleback copy and Vanguard 2 was lost)
    outs = [v for v in after if getattr(v, 'id', None) is not None and not has(v) and v.name != carry
            and v.id not in marks and not done_item(v) and getattr(v, 'slots', 1) == 1
            and not getattr(v, 'paired', None) and v.id not in placed]
    outs.sort(key=lambda v: ((v.info.cost if v.info else 0) * 3 ** ((v.star or 1) - 1), v.id))
    pairs = list(zip(ins, outs))

    def swapped(ps):
        gone = [v for _, v in ps]
        return [x for x in after if x not in gone] + [u for u, _ in ps]
    # the highest step of the trait the owned champions reach (Riftbeast 3, 5, 7): the owner, game 78 (x29.3) - "the 5
    # riftbeast should have been fielded much earlier when we had it available"; past a step, one more champion of
    # the trait short of the next adds nothing (x29.1: 5 Riftbeasts and another trait, not 6)
    from tft_player.comp_rules import trait_steps
    steps = trait_steps(getattr(_fixed_comp(), 'name', None)) or (keep,)
    reach = board_traits(swapped(pairs))[t] if pairs else b
    goal = max((s for s in steps if b < s <= reach), default=None)
    if not pairs or goal is None:
        return []                                # the bench cannot reach a step: the plan's board stands
    while len(pairs) > 1 and board_traits(swapped(pairs[:-1]))[t] >= goal:
        pairs = pairs[:-1]                       # no more swaps than the step needs
    star = lambda u: f"{u.name}{'*' * (u.star or 1)}"
    parts = []
    for u, v in pairs:
        h = list(at.get(v.id, v.loc)[1:])
        plan['board'] = [e for e in plan.get('board') or [] if not (isinstance(e, dict) and (
            list(e.get('hex') or ()) == h or e.get('id') in (u.id, v.id)))] + [{'id': u.id, 'hex': h}]
        if plan.get('bench'):
            plan['bench'] = [e for e in plan['bench'] if not (isinstance(e, dict) and e.get('id') == u.id)]
        parts.append(f'fielded {star(u)} (unit {u.id}) for {star(v)} (unit {v.id}) at hex ({h[0]},{h[1]})')
    n = board_traits(board_after(plan, roster)[1])[t]
    plan['_notes'] = list(plan.get('_notes') or []) + [('comp rule', (
        f"{'; '.join(parts)}: the plan's board had {t} {b} while the owned {t} champions make {goal}; {t} {n} after "
        f'this - "drop_breakpoint": true in execution_plan skips it for a plan'))]
    return pairs


def _keep_distinct_sells(plan, roster, gold=None):
    """A sell of the only copy of one of the fixed comp's trait champions (comp_rules 'trait': Riftbeast counts each
    champion once) is made on a spare copy instead - a benched second copy of another champion, with no items, that
    the plan does not field or keep - when that still pays for the plan's buys and XP (request e52, the comp coach's
    rule "to pay for a buy, sell a duplicate copy, never a distinct Riftbeast"; game 47 1-4: "sell Gromp" (2 g) for
    Pebbles (1 g) at 0 gold while a second Cinderling sold for the 1 g: 2 distinct Riftbeasts kept instead of 3). With a
    roll in the plan, or the gold not read, only a spare that sells for as much. "drop_breakpoint": true skips it.
    Told in the next state as 'comp rule' (plan['_notes'])."""
    from tft_player.comp_rules import rules
    r = rules(getattr(_fixed_comp(), 'name', None))
    t = r.get('trait')
    if not t or not isinstance(plan, dict) or plan.get('drop_breakpoint') or not plan.get('sell'):
        return
    from tft_player.state import board_traits
    # Game 49: Gromp was benched while seven other Riftbeasts already fought. Keeping him
    # cannot preserve a lost breakpoint, but substituting a Sentinel/Dragon sale breaks a pair.
    if r.get('keep') and board_traits(board_after(plan, roster)[1])[t] >= r['keep']:
        return
    roll = plan.get('roll_n') or plan.get('fast_rr') or {}
    targets = dict(roll.get('targets') or {}) if isinstance(roll, dict) else {}
    buy = roll.get('buy') if isinstance(roll, dict) else None
    targets.update(buy if isinstance(buy, dict) else {n: 2 for n in buy or []})
    if isinstance(plan.get('auto_buy'), dict):
        targets.update(plan['auto_buy'])
    sells = list(plan['sell'])
    by_id = {u.id: u for u in roster.units}
    alive = lambda u: u is not None and bool(u.name) and u.name != '?' and not u.missing
    fielded = {e.get('id') for e in plan.get('board') or [] if isinstance(e, dict)}
    kept = {k for k in plan.get('keep') or [] if isinstance(k, int)}
    spend = sum(CATALOG.unit(n).cost for n in plan.get('buy') or [] if isinstance(n, str) and CATALOG.unit(n))
    spend += 4 * int(plan.get('level_ups') or 0)
    rolls = bool(plan.get('roll_n') or plan.get('fast_rr') or (plan.get('roll') or {}).get('down_to') is not None)
    got = sum(sell_value(by_id[v]) for v in sells if alive(by_id.get(v)))
    others = lambda x: [y for y in roster.units if y is not x and y.name == x.name and y.id not in sells and alive(y)]
    parts = []
    for k, uid in enumerate(list(sells)):
        u = by_id.get(uid)
        if not alive(u) or t not in (u.info.traits if u.info else ()) or others(u):
            continue                             # not the trait's, or another copy stays and the champion still counts
        spares = []
        for x in roster.units:
            if (not alive(x) or x.on_board or x.id in sells or x.id in fielded or x.id in kept or x.unverified
                    or any(x.items or ()) or x.name == u.name or not others(x)):
                continue
            goal = targets.get(x.name)
            if isinstance(goal, int) and max(y.star or 1 for y in [x] + others(x)) < goal:
                continue                         # a copy of an unfinished upgrade is not a spare
            diff = sell_value(x) - sell_value(u)
            if diff < 0 and (rolls or gold is None or gold + got + diff < spend):
                continue
            spares.append((-diff, x.id, x))
        if not spares:
            continue
        x = min(spares)[2]
        sells[k] = x.id
        got += sell_value(x) - sell_value(u)
        parts.append(f"sold the spare {x.name}{'*' * (x.star or 1)} (unit {x.id}, {sell_value(x)} g) in place of "
                     f"{u.name}{'*' * (u.star or 1)} (unit {u.id}, {sell_value(u)} g), the only {u.name}: a {t} "
                     f'champion counts once, so a second copy adds nothing')
    if parts:
        plan['sell'] = sells
        plan['_notes'] = list(plan.get('_notes') or []) + [('comp rule', '; '.join(parts) + (
            ' - "drop_breakpoint": true in execution_plan skips it for a plan'))]


def _carry_items_move(plan, roster, item_bench):
    """The fixed comp's carry items go to the fielded carry (comp_rules 'carry_items', request e49 - game 45 4-5 and
    4-6: Brambleback fought bare while Guinsoo's Rageblade sat on Kog'Maw and Edge of Night on Murkwolf, the Item
    Remover at 5 uses, "next round they move"; game 40 4-6 to 5-2 Gromp held them "until Brambleback is 2-star"; game
    43 3-2 on Diana). When the carry is fielded (after the plan's moves) with a free slot and another unit holds one of
    his items, the plan's item step starts with the move, his items in the rule's order, up to his free slots: a benched
    holder the plan does not field, keep or need (not on the comp's final board) is sold (its items return to the item
    bench), any other holder takes one Item Remover use and gets its other items back. Never off a copy of the carry,
    a holder of Thief's Gloves (its items are not its own) or of an unread item (it could not be given back), never
    without a Remover use left on the item bench, never on an unread item bench. Its gives go first in the plan's
    items, so the plan's own gives see the moved items. "keep_items": true skips it. Told in the next state as
    'comp rule' (plan['_notes']).

    A comp with an item plan (item_plan.PLANS: a main carry and a main tank) runs the item plan instead: the plan's
    gives of components and of the carry's and tank's items are dropped, the carry's and the tank's items move to them
    (this move, once per holder), and the components make their next items (item_plan.build)."""
    from tft_player import item_plan
    from tft_player.comp_rules import rules
    comp = _fixed_comp()
    ip = item_plan.plan_for(getattr(comp, 'name', None))
    if ip is None:
        carry, order = rules(getattr(comp, 'name', None)).get('carry_items') or (None, ())
        _role_items_move(plan, roster, item_bench, carry, order)
        return
    if not isinstance(plan, dict) or item_bench is None or plan.get('keep_items'):
        return                                  # (as its docstring says: an item handover's own plan is not redone)
    board = board_after(owned_entries(plan, roster), roster)[1]
    provisional = item_plan.holders(board, ip, require_usable=False)
    waiting = [h.id for h in provisional.values() if h is not None and not item_plan._usable(h)]
    for entry in (plan.get('items') or []) + (plan.get('strip') or []):
        if isinstance(entry, dict):
            u = roster.resolve(entry.get('id'))
            if u is not None and (not u.name or u.unverified or getattr(u, 'items_check', False)):
                waiting.append(u.id)
    waiting = list(dict.fromkeys(waiting))
    plan['_item_reconcile'] = waiting
    if waiting:
        return                                  # retain the requested orders until actual holders are readable
    # a position with no holder fielded yet (no tank before Scuttlecrab or Krug) no longer hands the components to
    # the plan's own gives: its components wait, and the carry gets no more than his three (the owner, game 78 x29.4:
    # "there are 7 carry items, and 1 tank items, this severely deviates from the general rules, which target the 3
    # carry + 3 tanks items first" - the plan's free gives had built Titan's Resolve and a second Hand of Justice)
    # A give can have landed while an answer repeated it. Its already-equipped
    # item must not reserve the recipient's last slot in the new forecast (77
    # 5-5: a repeated Sterak's displaced the requested Rageblade to Gnar).
    requested_items, requested_transfers = list(plan.get('items') or []), list(plan.get('transfer') or [])
    already, remaining = [], []
    for e in plan.get('items') or []:
        u = roster.resolve(e.get('id')) if isinstance(e, dict) else None
        it = item_plan.base(e.get('component')) if isinstance(e, dict) else None
        if (u is not None and u.on_board and u.items_read and not getattr(u, 'items_predicted', False)
                and not e.get('_rule') and 'from' not in e and it in item_plan.COMPLETED and it in u.items):
            already.append(f'{it} -> unit {u.id} (already equipped)')
        else:
            remaining.append(e)
    plan['items'] = remaining
    before_drop = list(plan.get('items') or [])
    dropped = already + item_plan.drop_gives(plan, ip, board)
    # (at 5-1 the plan's gives of components too: every item is the model's decision that round - x28.8; game 79
    # 5-1: its component gives were dropped, nothing could take them, and the 5-1 check sent the plan back again and
    # again)
    dropped_done = [e for e in before_drop if e not in (plan.get('items') or []) and isinstance(e, dict)
                    and (item_plan.base(e.get('component')) in item_plan.COMPLETED or plan.get('_items_all'))]
    # the builds first, knowing what will move to the holder (item_plan.build), then the moves fill his other slots
    made = item_plan.repackage(plan, roster, item_bench, ip, board)
    if plan.get('_item_reconcile'):
        plan['items'] = requested_items
        plan['transfer'] = requested_transfers
        return
    made += item_plan.build(plan, roster, item_bench, ip, board, gives_held)
    hs = item_plan.holders(board, ip)
    for role in item_plan.ROLES[::-1]:            # each move goes first in the items: the carry's run first
        # onto the position's holder, whoever he is (the strongest suitable unit - item_plan.holders; the owner, game
        # 74b 4-5: "after the bramble is fielded, the next immediate action is to remover the items on the wolf to the
        # bramble"), from the weaker suitable units only
        h = hs[role]
        if h is None:
            continue
        unit, order = h.name, ip[role][1]
        allow = lambda u, h=h, role=role: item_plan.weaker(u, h, ip, role)
        # the tank takes one item a stat he lacks (x24.4: armor and magic resist, health, durability)
        steps = [(order, None)]
        if role == 'tank':
            tank = h
            held = (next((after for _, u, _, after, _ in reversed(gives_held(plan, roster)) if tank and u.id == tank.id),
                         None) or [str(i).split(' (')[0] for i in (tank.items if tank else []) if i])
            open_ = item_plan.open_items(ip, 'tank', held)
            steps = [(tuple(i for i in order if i in open_), None)]
        for its, limit in steps:
            if not its:
                continue
            before = {id(e) for e in plan.get('items') or []}
            _role_items_move(plan, roster, item_bench, unit, its, keep_ok=False, limit=limit, allow=allow,
                             group=item_plan.tank_group if role == 'tank' else None)
            for e in plan.get('items') or []:
                if id(e) not in before and isinstance(e, dict):
                    e['_rule'] = 'item plan'

    # a completed item of the carry's or the tank's that the item plan leaves on the item bench has no holder: the
    # plan's own give of it stands (the owner, game 78 x29.5: "all items made should have a carrier ... when current
    # holder is empty (cait), it should at least let the model know next turn and let it decide who to equip it")
    left = Counter(item_plan.base(i) for i in item_bench if i)
    left.subtract(item_plan.base(e.get('component')) for e in plan.get('items') or []
                  if isinstance(e, dict) and 'from' not in e and not e.get('_given'))
    for e in dropped_done:
        it = item_plan.base(e.get('component'))
        if left[it] > 0:
            left[it] -= 1
            plan['items'] = list(plan.get('items') or []) + [e]
            dropped = [d for d in dropped if d != f"{it} -> unit {e.get('id')}"]
    if dropped or made:
        plan['_notes'] = list(plan.get('_notes') or []) + [('item plan', (
            f"the executor builds the items of {ip['carry'][0]} (main carry) and {ip['tank'][0]} (main tank), three "
            f"each, by priority" + (f"; predicted deliveries: {'; '.join(made)}" if made else '')
            + (f"; your plan's gives of these items and components were not made: {', '.join(dropped)}"
               if dropped else '')))]


def _eighth_rule(level, plan=None):
    """The comp's 'eighth' rule when the level - after the plan's XP buys - reaches its level."""
    from tft_player.comp_rules import rules
    r = rules(getattr(_fixed_comp(), 'name', None)).get('eighth')
    lvl, xp = level if isinstance(level, (tuple, list)) else (level, None)
    if not r or not isinstance(lvl, int):
        return None
    n = (plan or {}).get('level_ups') if isinstance(plan, dict) else None
    pts = (xp[0] if isinstance(xp, (tuple, list)) and xp and isinstance(xp[0], int) else 0) + 4 * (n if isinstance(n, int) else 0)
    while lvl in XP_TO_NEXT and pts >= XP_TO_NEXT[lvl]:
        pts -= XP_TO_NEXT[lvl]
        lvl += 1
    return dict(r, _level=lvl) if lvl >= r['level'] else None


def aim_xp(plan, seen):
    """A reply's XP aims at the level its buys reach from the level and XP the model saw (seen: (level, (xp, need))):
    a level the executor bought while it was written (Player.executor_level, x32.3) is not bought again - "xp 8" written
    at level 6 5/36 is level 7, already there."""
    n = plan.get('level_ups') if isinstance(plan, dict) else None
    lvl, xp = seen if isinstance(seen, (tuple, list)) and len(seen) == 2 else (None, None)
    if (not isinstance(n, int) or n <= 0 or plan.get('_level_target') or not isinstance(lvl, int)
            or not isinstance(xp, (tuple, list)) or len(xp) != 2 or not all(isinstance(v, int) for v in xp) or not xp[1]):
        return
    pts, need, level = xp[0] + 4 * n, xp[1], lvl
    while need and pts >= need:
        pts, level = pts - need, level + 1
        need = XP_TO_NEXT.get(level)
    if level > lvl:
        plan['_level_target'] = level


def _level_seven_hold(plan, roster, level, gold=None, hp=None):
    """The owner, game 78 3-5 (x29.2): "it should not level to 7, because we do not have 7 riftbeast, we only have 6,
    what will happen if you level to 7 is that, it is not much stronger and will lose the interest". The plan's XP
    buys that end the phase at the comp's hold level (comp_rules 'level_hold_seven': 7), or are bought at the level
    below it toward it, while the champions of its trait owned (each once, Elder Dragon +2; the plan's named buys
    included) are fewer than that level are not made - XP that goes on to the next level is. Told in the next state as
    'comp rule'."""
    from types import SimpleNamespace
    from tft_player.comp_rules import rules
    from tft_player.state import board_traits
    r = rules(getattr(_fixed_comp(), 'name', None))
    hold, t = r.get('level_hold_seven'), r.get('trait')
    n = plan.get('level_ups') if isinstance(plan, dict) else None
    if not hold or not t or not isinstance(n, int) or n <= 0 or plan.get('_all_in_xp'):
        return                                   # (the last life's all-in XP is its own owner rule)
    from tft_player.comp_rules import low_hp_step
    if low_hp_step(hp, roster.board()):
        return                                   # HP low without Riftbeast 7: the model's level 7 (owner-hp-paths, x32.8)
    if isinstance(gold, int) and gold - 4 * n >= INTEREST_CAP:
        return                                   # it costs no interest - the owner's reason (game 81 3-6: 79 gold held)
    lvl, xp = level if isinstance(level, (tuple, list)) else (level, None)
    if not isinstance(lvl, int) or lvl >= hold:
        return
    pts = (xp[0] if isinstance(xp, (tuple, list)) and xp and isinstance(xp[0], int) else 0) + 4 * n
    end = lvl
    while end in XP_TO_NEXT and pts >= XP_TO_NEXT[end]:
        pts -= XP_TO_NEXT[end]
        end += 1
    if end > hold or end < hold and lvl != hold - 1:
        return                                   # XP on to the next level, or not toward the hold level
    owned = [u for u in roster.units if u.name and not u.missing and u.id not in set(plan.get('sell') or [])]
    names = [b for b in plan.get('buy') or [] if isinstance(b, str)]     # (a roll's list is what it looks for, not owned)
    bought = [SimpleNamespace(name=CATALOG.unit(b).name, traits=CATALOG.unit(b).traits, items=())
              for b in names if CATALOG.unit(b)]
    have = board_traits(owned + bought)[t]
    if have >= hold:
        return
    plan['level_ups'] = 0
    plan['_notes'] = list(plan.get('_notes') or []) + [('comp rule', (
        f"no XP: {n} XP buys toward level {hold} with {t} {have} owned - level {hold} waits for the "
        f"{hold}th {t} (the 7th slot adds little and the XP costs the interest)"))]


def _no_second_copy(plan, roster, errors):
    """A board entry that fields a second copy of a champion the board fields already is not made: a copy adds no
    trait (the owner, game 81, x31.4: "it fielded 2 crabs at level 8"; the 3-2 replay: a second Scuttlecrab "as a body"
    and level 6 bought for it). "keep_board": true makes it anyway."""
    if not isinstance(plan, dict) or plan.get('keep_board') or not plan.get('board'):
        return
    for e in list(plan['board']):
        if not isinstance(e, dict) or not isinstance(e.get('id'), int):
            continue
        u = next((x for x in roster.units if x.id == e['id']), None)
        if u is None or not u.name or u.on_board:
            continue
        _, after, at = board_after(owned_entries(plan, roster), roster, where=True)
        twin = next((x for x in after if getattr(x, 'id', None) not in (None, u.id) and x.name == u.name
                     and at.get(x.id, ('',))[0] == 'board'), None)
        if twin is None:
            continue
        plan['board'] = [x for x in plan['board'] if x is not e]
        errors.append(f"board {str({k: v for k, v in e.items() if k != 'k'})}: {u.name} (unit {twin.id}) is fielded "
                      f"already - a second copy adds no trait tier; field a unit that completes one (the shop and bench "
                      f"lines say which) or leave the slot. To field it anyway, \"keep_board\": true")


def tier_steps(units):
    """{trait: the highest step its count reaches} for these fielded units (the game's count: each champion once)."""
    from tft_player.state import board_traits
    n = board_traits(units)
    steps = {t: max((x for x in CATALOG.traits.get(t) or () if x <= c), default=0) for t, c in n.items()}
    return {t: x for t, x in steps.items() if x}       # active steps only: an inactive trait is no tier


def tiers_gained(base, unit):
    """The trait steps `unit` completes when fielded beside `base` -> ['Juggernaut 2', ...]."""
    from types import SimpleNamespace
    before = tier_steps(base)
    u = unit if hasattr(unit, 'traits') else SimpleNamespace(name=unit, traits=CATALOG.unit(unit).traits, items=())
    after = tier_steps(list(base) + [u])
    return [f'{t} {after[t]}' for t in sorted(after) if after[t] > before.get(t, 0)]


def _extras_add_traits(plan, roster):
    """Only a trait's steps count (Riftbeast 3, 5, 7, 10): every other fielded unit adds a trait tier to the board
    (the owner, game 81, x31.1/x31.2: "when 4, 6, 8, 9 it should be adding extra traits to the current boards ... like
    rakan (two juggernaut), alistar (0), hence not fielded"). After the plan's moves, a fielded unit whose leaving loses
    no tier - a Riftbeast past the step, a second copy, a unit whose traits all stay inactive - gives its hex to a
    benched unit that completes a tier there. Never the carry, the Alpha, a holder of a completed item, a two-slot
    unit or one of MetaTFT's level-8 units; "keep_board": true skips it. Told in the next state as 'comp rule'."""
    from tft_player.comp_rules import eighth_units, rules
    comp = _fixed_comp()
    if comp is None or not isinstance(plan, dict) or plan.get('keep_board'):
        return
    r = rules(comp.name)
    carry = (r.get('carry') or (None,))[0]
    adds = set(eighth_units(comp.name))
    marks = {k[1] for k in getattr(roster, 'marks', None) or [] if k[0] == 'Alpha Mark'}
    done_item = lambda x: any(i and i != 'unknown' and i not in COMPONENTS for i in getattr(x, 'items', None) or ())
    gone = set(plan.get('sell') or []) | {e.get('id') for e in plan.get('bench') or [] if isinstance(e, dict)}
    final = set(getattr(comp, 'units', None) or ())
    for _ in range(3):
        # the entries that hold now: a bench "when_owned" a champion not bought yet is not done (game 81 4-6 replay:
        # counted as done, it left Riftbeast 4 and benched the 3-star Pebbles)
        _, after, at = board_after(owned_entries(plan, roster), roster, where=True)
        ours = [u for u in after if getattr(u, 'id', None) is not None and at.get(u.id, ('',))[0] == 'board'
                and not u.missing and not getattr(u, 'slotless', False)]
        if any(not u.name for u in ours):
            return                               # a unit not identified: its traits are not counted
        now = tier_steps(ours)
        names = {u.name for u in ours}
        outs = [u for u in ours if u.name != carry and u.id not in marks and not done_item(u) and u.slots == 1
                and u.name not in adds and tier_steps([x for x in ours if x is not u]) == now]
        # a bridge unit before one of the comp's final board, then the cheapest (cost by stars)
        outs.sort(key=lambda u: (u.name in final, (u.info.cost if u.info else 0) * 3 ** ((u.star or 1) - 1), u.id))
        bench = [b for b in roster.units if b.name and at.get(b.id, ('',))[0] == 'bench' and not b.missing
                 and not b.unverified and b.slots == 1 and b.id not in gone]
        best = None
        for out in outs:
            rest = [x for x in ours if x is not out]
            for b in bench:
                if b.name in {x.name for x in rest}:
                    continue
                got = tiers_gained(rest, b)
                if got and (best is None or (len(got), b.star or 1, b.info.cost if b.info else 0) > best[0]):
                    best = ((len(got), b.star or 1, b.info.cost if b.info else 0), b, out, got)
            if best is not None:
                break
        if best is None:
            return
        _, b, out, got = best
        _bench_swap(plan, roster, b, out, at, f"{out.name} adds no trait tier to the board, {b.name} completes "
                                              f"{', '.join(got)} (only a trait's steps count: x31)")
        gone.add(out.id)


def _bench_swap(plan, roster, inn, out, at, why):
    """inn (benched) takes out's hex; out goes to the bench (the drag swaps them). A melee unit is never put in a back
    row (game 78 replay, 4-6 and 4-7: Sentinel took Caitlyn's (0,4), and the main-tank rule then sent Krug there): it
    takes its comp hex, else the front row's free hex nearest the middle - never the corner in front of the carry -
    and out goes to a free bench slot."""
    hexes = list(at[out.id][1:])
    info = CATALOG.unit(inn.name or '')
    where, slot = None, None
    if info is not None and info.melee and hexes[0] <= 1:
        taken = {tuple(v[1:]) for k, v in at.items() if v and v[0] == 'board' and k != out.id}
        spots = {n.strip(): (int(r), int(c)) for n, r, c in re.findall(r'([^,(]+?) \((\d),(\d)\)', target_hexes())}
        from tft_player import item_plan
        ip = item_plan.plan_for(getattr(_fixed_comp(), 'name', None))
        carry = spots.get(ip['carry'][0]) if ip else None
        corner = (3, carry[1]) if carry and carry[0] == 0 and carry[1] in (0, 6) else None
        own = spots.get(inn.name)
        cands = ([own] if own and own[0] >= 2 and own not in taken and own != corner else []) + sorted(
            ((3, c) for c in range(7) if (3, c) not in taken and (3, c) != corner), key=lambda h: abs(h[1] - 3))
        free = [k for k in roster.free_bench() if not any(isinstance(e, dict) and e.get('slot') == k
                                                          for e in plan.get('bench') or [])]
        if cands and free:
            where, slot = cands[0], free[0]
    plan['board'] = [e for e in plan.get('board') or [] if not (isinstance(e, dict) and e.get('id') in (inn.id, out.id))]
    if where is not None:
        plan['board'].append({'id': inn.id, 'hex': list(where)})
        plan['bench'] = [e for e in plan.get('bench') or [] if not (isinstance(e, dict) and e.get('id') == out.id)]
        plan['bench'].append({'id': out.id, 'slot': slot})
        plan['_notes'] = list(plan.get('_notes') or []) + [('comp rule', f"{inn.name} (unit {inn.id}) fielded at "
                                                            f"{tuple(where)} - a melee unit in the front - and {out.name} "
                                                            f"(unit {out.id}) to bench slot {slot}: {why}")]
        return
    plan['board'].append({'id': inn.id, 'hex': hexes})
    plan['_notes'] = list(plan.get('_notes') or []) + [('comp rule', f"{inn.name} (unit {inn.id}) fielded at "
                                                        f"{tuple(hexes)} for {out.name} (unit {out.id}): {why}")]


def _level_eight_extras(plan, roster, level):
    """The owner, game 72 (x24.1): "why field 8 Riftbeast? One extra does nothing, offers no value, why not field a
    brawler? Or even Rakan? In the MetaTFT it is clearly marked that level 8 should field Taric / Gnar / Sett / Rakan".
    From level 8 (comp_rules 'eighth'): every roll of the plan also looks for one copy of each of those units not
    owned, and one owned on the bench takes the place of a Riftbeast beyond the seventh (Cinderling, Gromp, Mama Beak,
    Scuttlecrab - an itemless one first) when the board keeps Riftbeast 7. Told in the next state as 'comp rule'."""
    r = _eighth_rule(level, plan)
    if not r or not isinstance(plan, dict):
        return
    from tft_player.comp_rules import eighth_units, optional_room
    from tft_player.state import board_traits
    r = dict(r, units=eighth_units(getattr(_fixed_comp(), 'name', None), r.get('_level', 8)))
    owned = {u.name for u in roster.units if not u.missing}
    for k in ('roll_n', 'fast_rr'):
        roll = plan.get(k)
        if isinstance(roll, dict) and isinstance(roll.get('buy'), (dict, list, type(None))):
            buy = roll.setdefault('buy', {})
            injected = plan.setdefault('_optional_buys', {}).setdefault(k, [])
            for n in injected:
                if isinstance(buy, dict):
                    buy.pop(n, None)
                elif n in buy:
                    buy.remove(n)
            added = [n for n in r['units'] if n not in owned and n not in buy] if optional_room(
                roster, getattr(_fixed_comp(), 'name', None), level) else []
            plan['_optional_buys'][k] = added
            for n in added:
                if isinstance(buy, dict):
                    buy[n] = 1
                else:
                    buy.append(n)                    # a list of names: one copy each
            if added:
                plan['_notes'] = list(plan.get('_notes') or []) + [('comp rule', (
                    f"the roll also looks for {', '.join(added)} (level 8: the slots beyond Riftbeast 7)"))]
    # an add MetaTFT ranks higher, on the bench, takes the place of a lower-ranked fielded add (the owner, game 81
    # 4-7, x31.5: Taric "should replace the sett")
    _, after, at = board_after(plan, roster, where=True)
    ours = [u for u in after if getattr(u, 'id', None) is not None and at.get(u.id, ('',))[0] == 'board']
    sold = set(plan.get('sell') or [])
    rank = list(r['units'])
    for name in rank:
        if any(u.name == name for u in ours):
            continue
        inn = next((u for u in roster.units if u.name == name and not u.on_board and not u.missing
                    and not u.unverified and u.id not in sold), None)
        worse = [u for u in ours if u.name in rank and rank.index(u.name) > rank.index(name)
                 and u.slots == getattr(inn, 'slots', 1)]
        if inn is None or not worse:
            continue
        out = max(worse, key=lambda u: rank.index(u.name))
        _bench_swap(plan, roster, inn, out, at, f"level 8: MetaTFT's level-8 boards rank {name} above {out.name}")
        _, after, at = board_after(plan, roster, where=True)
        ours = [u for u in after if getattr(u, 'id', None) is not None and at.get(u.id, ('',))[0] == 'board']
    from tft_player.comp_rules import trait_sharing_augment
    if trait_sharing_augment():
        return                                   # each Riftbeast past the seventh strengthens all of them (x30.18)
    _, after, at = board_after(plan, roster, where=True)
    ours = [u for u in after if getattr(u, 'id', None) is not None]
    sold = set(plan.get('sell') or [])
    for name in r['units']:
        if any(u.name == name for u in ours):
            continue
        inn = next((u for u in roster.units if u.name == name and not u.on_board and not u.missing
                    and not u.unverified and u.id not in sold), None)
        if inn is None:
            continue
        for out in sorted((u for u in ours if u.name in r['out']), key=lambda u: (len(u.items or ()), u.star or 1)):
            if board_traits([u for u in ours if u is not out])['Riftbeast'] >= 7:
                _bench_swap(plan, roster, inn, out, at, 'level 8: no eighth Riftbeast, MetaTFT\'s level-8 unit')
                _, after, at = board_after(plan, roster, where=True)
                ours = [u for u in after if getattr(u, 'id', None) is not None]
                break


def _off_comp_out(plan, roster, level):
    """The owner, game 72 (x24.2): "Ashe offers little value to this comp ... the strategy of the 7 Riftbeast: its goal
    is to last the fight as long as possible, because it gains stats with time". From level 8 (comp_rules 'eighth'), a
    fielded unit on none of the comp's boards (its final board, its trait's champions, its level-8 units) makes way for
    a benched one that is and is not fielded yet - the comp's final-board units first, the higher star first. Told in
    the next state as 'comp rule'."""
    r = _eighth_rule(level, plan)
    comp = _fixed_comp()
    if not r or not isinstance(plan, dict) or plan.get('keep_board') or comp is None:
        return
    from tft_player.comp_rules import eighth_units, rules
    r = dict(r, units=eighth_units(comp.name, r.get('_level', 8)))
    trait = rules(comp.name).get('trait')
    final = set(getattr(comp, 'units', None) or ())
    mine = lambda n: n in final or n in r['units'] or (
        trait and CATALOG.unit(n) and trait in CATALOG.unit(n).traits)
    _, after, at = board_after(plan, roster, where=True)
    ours = [u for u in after if getattr(u, 'id', None) is not None and u.name]
    sold = set(plan.get('sell') or [])
    from tft_player.state import board_traits
    keep = rules(comp.name).get('keep') or 0
    from tft_player.comp_rules import TANK_TRAITS
    tanky = lambda n: bool(CATALOG.unit(n) and set(CATALOG.unit(n).traits) & set(TANK_TRAITS))
    # an off-comp Brawler / Juggernaut / Vanguard stays: a filler for the tanky traits (the owner, game 78 x28.4)
    for out in [u for u in ours if not mine(u.name) and not tanky(u.name) and not u.missing and not u.slotless]:
        fielded = {u.name for u in ours}
        full = trait and board_traits(ours)[trait] >= keep     # a trait champion past the step adds nothing (x24.1)
        cands = [u for u in roster.units if u.name and mine(u.name) and u.name not in fielded and not u.on_board
                 and not u.missing and not u.unverified and u.id not in sold
                 and (u.name in final or u.name in r['units'] or not full)]
        if not cands:
            # none: a benched melee champion of the comp, a copy too - a body that keeps the fight going (game 72 5-2:
            # Krug 2-star for Ashe)
            cands = [u for u in roster.units if u.name and mine(u.name) and not u.on_board and not u.missing
                     and not u.unverified and u.id not in sold and CATALOG.unit(u.name) and CATALOG.unit(u.name).melee]
        if not cands:
            return
        inn = max(cands, key=lambda u: (u.name in final, u.star or 1, CATALOG.unit(u.name).cost if CATALOG.unit(u.name) else 0))
        _bench_swap(plan, roster, inn, out, at, f'{out.name} is on none of the comp\'s boards - the comp wins by making '
                                                f'the fight last')
        _, after, at = board_after(plan, roster, where=True)
        ours = [u for u in after if getattr(u, 'id', None) is not None and u.name]


def _carry_corner(plan, roster):
    """The comp's main carry stands on his hex of the comp's final board (MetaTFT's positions: Brambleback (0,6), the
    back corner) - the owner, game 76: "the brambleback back not in the bottom corner, but a bit to the left" (the
    plan's own hex (0,5)). After the plan's moves, a fielded carry elsewhere takes that hex; a unit there takes his.
    For a comp with an item plan; "keep_board": true skips it. Told in the next state as 'comp rule'."""
    from tft_player import item_plan
    from tft_player.comp_rules import formation
    comp = _fixed_comp()
    if formation(getattr(comp, 'name', None)):
        return _formation(plan, roster)          # the owner's default positions place the carry too (x29.9)
    ip = item_plan.plan_for(getattr(comp, 'name', None))
    if not ip or not isinstance(plan, dict) or plan.get('keep_board'):
        return
    name = ip['carry'][0]
    want = next(((int(r), int(c)) for n, r, c in re.findall(r'([^,(]+?) \((\d),(\d)\)', target_hexes())
                 if n.strip() == name), None)
    if want is None:
        return
    _, after, at = board_after(plan, roster, where=True)
    ours = [u for u in after if getattr(u, 'id', None) is not None and at.get(u.id, ('',))[0] == 'board'
            and not u.missing and not u.unverified]
    carry = max((u for u in ours if u.name == name), key=lambda u: (u.star or 1, len(u.items or ())), default=None)
    if carry is None or tuple(at[carry.id][1:]) == want:
        return
    here = next((u for u in ours if tuple(at[u.id][1:]) == want), None)
    c_hex = list(at[carry.id][1:])
    drop = {carry.id} | ({here.id} if here is not None else set())
    plan['board'] = [e for e in plan.get('board') or [] if not (isinstance(e, dict) and e.get('id') in drop)]
    plan['board'] += [{'id': carry.id, 'hex': list(want)}] + ([{'id': here.id, 'hex': c_hex}] if here is not None else [])
    plan['_notes'] = list(plan.get('_notes') or []) + [('comp rule', (
        f"{carry.name} (unit {carry.id}), the main carry, takes his comp hex {want} (MetaTFT's final board)"
        + (f" and {here.name} takes {tuple(c_hex)}" if here is not None else '')))]


def _main_tank_side(plan, roster):
    """The comp's main tank is the frontline's champion on the main carry's side (the owner, game 72, x24.5: "the
    strongest tank to be on the same side of the Brambleback ... if not, the opponent will attack the weak tank and
    then after finishing, the Brambleback will be attacked. Hence the frontline, the most right champion should be the
    main tank"): after the plan's moves, when the fielded main tank is not the front row's outermost champion on the
    carry's side (columns 3-6 the right, 0-3 the left), he takes that champion's hex and it takes his. For a comp with
    an item plan (item_plan.PLANS); "keep_board": true skips it. Told in the next state as 'comp rule'."""
    from tft_player import item_plan
    from tft_player.comp_rules import formation
    ip = item_plan.plan_for(getattr(_fixed_comp(), 'name', None))
    if not ip or not isinstance(plan, dict) or plan.get('keep_board') or formation(getattr(_fixed_comp(), 'name', None)):
        return                                   # (a comp with the owner's formation: _formation places the main tank)
    _, after, at = board_after(plan, roster, where=True)
    ours = [u for u in after if getattr(u, 'id', None) is not None and at.get(u.id, ('',))[0] == 'board'
            and not u.missing and not u.unverified]
    # the fighter that holds the carry's items and the tankiest frontliner - the positions' holders (item_plan.holders:
    # stars, then cost), not only Brambleback and Krug by name (the owner, game 74b 3-5: "when the fighter is at right
    # side, the front right most should be the most tanky (krug) instead of the sentinel")
    hs = item_plan.holders(ours, ip)
    carry, tank = hs['carry'], hs['tank']
    if carry is None or tank is None:
        return
    side = 1 if at[carry.id][2] >= 3 else -1
    front = [u for u in ours if at[u.id][1] == 3]
    if not front:
        return
    edge = max(front, key=lambda u: side * at[u.id][2])
    if edge is tank or side * at[edge.id][2] <= side * at[tank.id][2] and at[tank.id][1] == 3:
        return
    t_hex, e_hex = list(at[tank.id][1:]), list(at[edge.id][1:])
    if t_hex[0] != 3:
        # the tank came from a back row: the frontliner he displaces keeps to the front row, on its nearest free hex
        # (never the corner in front of the carry) - not to the tank's back-row hex (game 78 replay 4-6: Krug to (0,4))
        taken = {tuple(at[u.id][1:]) for u in ours}
        corner = (3, 6) if side > 0 else (3, 0)
        free = sorted(((3, c) for c in range(7) if (3, c) not in taken and (3, c) != corner),
                      key=lambda h: abs(h[1] - e_hex[1]))
        if free:
            t_hex = list(free[0])
    plan['board'] = [e for e in plan.get('board') or [] if not (isinstance(e, dict) and e.get('id') in (tank.id, edge.id))]
    plan['board'] += [{'id': tank.id, 'hex': e_hex}, {'id': edge.id, 'hex': t_hex}]
    plan['_notes'] = list(plan.get('_notes') or []) + [('comp rule', (
        f"{tank.name} (unit {tank.id}), the main tank, takes {edge.name}'s hex {tuple(e_hex)} - the front row's "
        f"{'right' if side > 0 else 'left'}most, {carry.name}'s side - and {edge.name} takes {tuple(t_hex)}"))]


def _guarded(rule, plan, *args):
    """One comp rule of the plan check: a fault in it is printed with its traceback (the game's output, the loop's
    monitor) and the plan runs without that rule. -> the rule's result, or None."""
    try:
        return rule(*args)
    except Exception as e:
        import traceback
        print(f'HARNESS ERROR in {rule.__name__}: {e!r}\n{traceback.format_exc()}', flush=True)
        return None


def _guarded_lines(fn, *args):
    """A state section on its own: a fault in it is printed, and the state goes out without that section."""
    return _guarded(fn, None, *args) or []          # (_guarded's second argument is the plan, unused here)


def _formation(plan, roster):
    """The comp's default positions (comp_rules 'formation'; the owner, game 78, x29.9: "tank position should default
    to, from right: 3rd position: main tank with the item, mostly krug; mid: sentinel; 5th: crab. brambleback back
    right corner, small blue left corner, and then cinderling and gromp close to the brambleback on the back"): after
    the plan's moves, each fielded unit of the formation takes its hex - 'tank' is the main tank, the tank items'
    holder (item_plan.holders); one copy each, the highest star. A unit on that hex moves to the hex the other left
    when both stand in the front rows (2-3) or both in the back rows (0-1), else to the nearest free hex of its own
    row. "keep_board": true skips it. Told in the next state as 'comp rule'."""
    from tft_player import item_plan
    from tft_player.comp_rules import formation
    comp = _fixed_comp()
    form = formation(getattr(comp, 'name', None))
    if not form or not isinstance(plan, dict) or plan.get('keep_board'):
        return
    ip = item_plan.plan_for(comp.name)
    _, after, at = board_after(plan, roster, where=True)
    if ip:
        # the carry's best copy (stars, then items) is the fielded one: a benched copy with more takes the fielded one's
        # hex, the drag swapping them (the owner, game 79 5-2/5-3, x30.8: "it left the 3 items bramble on bench, and
        # bramble with only rageblade on board")
        cn = ip['carry'][0]
        rank = lambda u: (u.star or 1, len([i for i in u.items or () if i]))
        on = [u for u in after if getattr(u, 'id', None) is not None and u.name == cn and at.get(u.id, ('',))[0] == 'board']
        off = [u for u in roster.units if u.name == cn and not u.missing and not u.unverified
               and getattr(u, 'items_read', True) and at.get(u.id, ('',))[0] == 'bench']
        if on and off:
            worst, best = min(on, key=rank), max(off, key=rank)
            if rank(best) > rank(worst):
                h = list(at[worst.id][1:])
                plan['board'] = [e for e in plan.get('board') or [] if not (isinstance(e, dict) and (
                    e.get('id') in (best.id, worst.id) or list(e.get('hex') or ()) == h))] + [{'id': best.id, 'hex': h}]
                plan['_notes'] = list(plan.get('_notes') or []) + [('comp rule', (
                    f"{cn}{'*' * (best.star or 1)} (unit {best.id}, {len(best.items or ())} items) from the bench takes "
                    f"the fielded copy's hex {tuple(h)} (unit {worst.id}, {len(worst.items or ())} items): the carry's "
                    f"best copy fights"))]
                _, after, at = board_after(plan, roster, where=True)
    ours = [u for u in after if getattr(u, 'id', None) is not None and at.get(u.id, ('',))[0] == 'board'
            and not u.missing and not u.unverified and u.name]
    if not ours:
        return
    pos = {u.id: tuple(at[u.id][1:]) for u in ours}
    start = dict(pos)
    tank = item_plan.holders(ours, ip)['tank'] if ip else None
    want, used = [], set()
    if tank is not None and 'tank' in form:
        want.append((tank, tuple(form['tank'])))
        used.add(tank.id)
    for name, hex_ in form.items():
        if name == 'tank':
            continue
        u = max((x for x in ours if x.name == name and x.id not in used),
                key=lambda x: (x.star or 1, len(x.items or ())), default=None)
        if u is not None:
            want.append((u, tuple(hex_)))
            used.add(u.id)
    front = lambda h: h[0] >= 2
    for u, hex_ in want:
        if pos[u.id] == hex_:
            continue
        other = next((x for x in ours if pos[x.id] == hex_ and x is not u), None)
        old = pos[u.id]
        if other is not None:
            if front(old) == front(hex_):
                pos[other.id] = old
            else:
                taken = set(pos.values())
                free = sorted(((hex_[0], c) for c in range(7) if (hex_[0], c) not in taken),
                              key=lambda h: abs(h[1] - hex_[1]))
                if not free:
                    continue
                pos[other.id] = free[0]
        pos[u.id] = hex_
    moved = [u for u in ours if pos[u.id] != start[u.id]]
    if not moved:
        return
    ids = {u.id for u in moved}
    # a fielded unit the plan's swap sends to the bench stays benched when this rule moves the incoming unit elsewhere -
    # else the board ends one over the team cap and the drag is refused (game 81 5-1: the player's Brambleback onto the
    # unidentified Scuttlecrab's (2,3) became a drag to the empty (0,6) on an 8/8 board, refused four times)
    gone = {e.get('id') for e in plan.get('bench') or [] if isinstance(e, dict)} | set(plan.get('sell') or [])
    swapped = [u for u in roster.board() if at.get(u.id, ('',))[0] == 'bench' and u.id not in gone]
    plan['board'] = [e for e in plan.get('board') or [] if not (isinstance(e, dict) and e.get('id') in ids)]
    plan['board'] += [{'id': u.id, 'hex': list(pos[u.id])} for u in moved]
    if swapped:
        _, _, now_at = board_after(plan, roster, where=True)
        keep = [u for u in swapped if now_at.get(u.id, ('',))[0] == 'board']
        if keep:
            plan['bench'] = list(plan.get('bench') or []) + [{'id': u.id, 'slot': at[u.id][1]} for u in keep]
    plan['_notes'] = list(plan.get('_notes') or []) + [('comp rule', (
        "the comp's formation (main tank (3,4), Sentinel (3,3), Scuttlecrab (3,2); Brambleback (0,6), Pebbles (0,0), "
        "Cinderling and Gromp beside Brambleback): " + '; '.join(f'{u.name} (unit {u.id}) -> {pos[u.id]}' for u in moved)))]


def _keep_corner_free(plan, roster):
    """The front-row hex in the carry's corner column stays empty (the owner, game 78 5-1, x28.7: "it places the krug
    to the right most front again, no frontline should be there"; x27.3: the carry's side of the front row is not
    filled up): after the plan's moves, a unit on it moves one hex in, and a unit there moves to the nearest free
    front-row hex - the main tank stays the outermost champion on the carry's side (owner-tank-on-fighter-side). For a
    comp with an item plan whose carry's hex is a back corner; "keep_board": true skips it. Told in the next state."""
    from tft_player import item_plan
    comp = _fixed_comp()
    ip = item_plan.plan_for(getattr(comp, 'name', None))
    if not ip or not isinstance(plan, dict) or plan.get('keep_board'):
        return
    spot = next(((int(r), int(c)) for n, r, c in re.findall(r'([^,(]+?) \((\d),(\d)\)', target_hexes())
                 if n.strip() == ip['carry'][0]), None)
    if not spot or spot[0] != 0 or spot[1] not in (0, 6):
        return
    col = spot[1]
    inward = -1 if col == 6 else 1
    _, after, at = board_after(plan, roster, where=True)
    ours = {tuple(at[u.id][1:]): u for u in after if getattr(u, 'id', None) is not None
            and at.get(u.id, ('',))[0] == 'board' and not u.missing}
    x = ours.get((3, col))
    if x is None:
        return
    moves = []
    nxt = (3, col + inward)
    y = ours.get(nxt)
    if y is not None:
        free = [(3, c) for c in range(7) if (3, c) not in ours and c != col]
        if not free:
            return
        f = min(free, key=lambda h: abs(h[1] - nxt[1]))
        moves.append((y, f))
    moves.append((x, nxt))
    ids = {u.id for u, _ in moves}
    plan['board'] = [e for e in plan.get('board') or [] if not (isinstance(e, dict) and e.get('id') in ids)]
    plan['board'] += [{'id': u.id, 'hex': list(h)} for u, h in moves]
    plan['_notes'] = list(plan.get('_notes') or []) + [('comp rule', (
        f"the front-row corner (3, {col}) in front of {ip['carry'][0]} stays empty: "
        + '; '.join(f'{u.name} (unit {u.id}) -> {h}' for u, h in moves)))]


def _role_items_move(plan, roster, item_bench, carry, order, keep_ok=True, limit=None, allow=None, group=None):
    """_carry_items_move for one holder (carry) and his items (order): keep_ok - "keep_items": true skips it; limit:
    at most that many items move; allow(unit): the units whose items may move (all when None)."""
    from tft_player.items import uses_left
    comp = _fixed_comp()
    if not carry or not isinstance(plan, dict) or keep_ok and plan.get('keep_items') or item_bench is None:
        return
    base = lambda n: str(n or '').split(' (')[0]
    rank = {n.lower(): k for k, n in enumerate(order)}
    current_board = board_after(owned_entries(plan, roster), roster)[1]
    on = [u for u in current_board if getattr(u, 'id', None) is not None and u.name == carry
          and not u.missing]
    if not on:
        return
    to = max(on, key=lambda u: (u.star or 1, len(u.items or ()), u.on_board))
    held = [i for i in to.items or () if i]
    if "Thief's Gloves" in held or not getattr(to, 'items_read', True):
        return
    # the plan's own gives of completed items to him take slots too (run 20261030-132522 5-5: the plan gave Brambleback
    # Hand Of Justice, Guinsoo's Rageblade and Edge of Night; the rule added Bloodthirster off Murkwolf - 4 items for
    # 3 slots, one more Remover use with 18 s left)
    forecast = [after for _, u, _, after, _ in gives_held(plan, roster) if u.id == to.id]
    own = [base(e.get('item')) for e in plan.get('transfer') or [] if isinstance(e, dict) and e.get('to') == to.id]
    room = 3 - len(forecast[-1] if forecast else held) - sum(
        1 for i in own if i not in COMPONENTS and not is_consumable(i) and i != 'Alpha Mark'
        and i not in [base(h) for h in held])
    if room <= 0:
        return
    sold, fielded = set(plan.get('sell') or []), {getattr(u, 'id', None) for u in current_board}
    stripped = {e.get('id') for e in plan.get('strip') or [] if isinstance(e, dict)} | \
        {e.get('from') for e in plan.get('transfer') or [] if isinstance(e, dict)}
    cands = []
    for u in roster.units:
        items = [base(i) for i in u.items or () if i]
        if u is to or u.name == carry or u.missing or u.unverified or u.id in stripped or not u.name or \
                "Thief's Gloves" in items or 'unknown' in items or not getattr(u, 'items_read', True) or                 allow is not None and not allow(u):
            continue
        cands += [(rank[i.lower()], u, i) for i in items if i.lower() in rank]
    # one lying on the item bench that no give of the plan names: a move whose Remover ran before an update replaced
    # its plan (its gives went with the plan), or a holder sold - a give, no Remover
    named = Counter(base(e.get('component')) for e in plan.get('items') or [] if isinstance(e, dict))
    for i in Counter(base(i) for i in item_bench if i and base(i).lower() in rank).items():
        cands += [(rank[i[0].lower()], None, i[0])] * max(0, i[1] - named[i[0]])
    if not cands:
        return
    cands.sort(key=lambda c: (c[0], c[1] is not None, c[1] is not None and not c[1].on_board))
    uses = [uses_left(i) for i in item_bench if i and is_remover(i)]
    uses = sum(1 if n is None else n for n in uses) - sum(
        1 for e in plan.get('strip') or [] if isinstance(e, dict) and e.get('tool') == 'remover')
    final = set(getattr(comp, 'units', None) or ())
    keep = set(plan.get('keep') or [])
    gives, parts, by, selected_groups = [], [], {}, set()
    for _, u, it in cands:
        if room <= 0 or limit is not None and len(gives) >= limit:
            break
        if group and group(it) in selected_groups:
            continue
        if u is None:
            room -= 1
            if group:
                selected_groups.add(group(it))
            gives.append({'component': it, 'id': to.id})
            parts.append(f'{it} given from the item bench to {carry} (unit {to.id})')
            continue
        how = by.get(u.id)
        if how is None:
            if u.id in sold or (not u.on_board and u.id not in fielded and u.id not in keep and u.name not in final):
                how = 'sell'
                if u.id not in sold:
                    plan['sell'] = list(plan.get('sell') or []) + [u.id]
            elif uses > 0:
                how, uses = 'remover', uses - 1
                plan['strip'] = list(plan.get('strip') or []) + [{'id': u.id, 'tool': 'remover'}]
            else:
                continue                          # no Remover use left: it stays where it is
            by[u.id] = how
        room -= 1
        if group:
            selected_groups.add(group(it))
        gives.append({'component': it, 'id': to.id} | ({'from': u.id} if how == 'remover' else {}))
        parts.append(f"{it} moved from {u.name}{'*' * (u.star or 1)} (unit {u.id}) to {carry} (unit {to.id}) "
                     f"({'sold: not on the board' if how == 'sell' else 'one Item Remover use'})")
    if not gives:
        return
    for uid, how in by.items():                   # a stripped holder's other items go back onto it
        if how != 'remover':
            continue
        u = next(x for x in roster.units if x.id == uid)
        rest = [base(i) for i in u.items or () if i]
        for g in gives:
            if g.get('from') == uid:
                rest.remove(g['component'])
        gives += [{'component': i, 'id': uid, 'from': uid} for i in rest]
    items = [e for e in plan.get('items') or [] if isinstance(e, dict)]
    k = max([e.get('k', -1) for e in items if isinstance(e.get('k'), int)] + [-1]) + 1
    plan['items'] = [dict(g, k=k + n) for n, g in enumerate(gives)] + items
    plan['_notes'] = list(plan.get('_notes') or []) + [(
        'comp rule', '; '.join(parts) + f' - {carry} is fielded and these are his items; "keep_items": true in '
                                         f'execution_plan skips it for a plan')]


def combine_components(plan, roster, item_bench, stage, item_map=None, done=()):
    """Owner rule owner-no-two-components (request e179; from stage 4 since game 78, x28.6): at most one single
    component is held, item bench and units together. Once per plan, in the planning phase, after the plan's own gives: any two
    single components make an item on a fielded unit - a bench component onto a fielded unit holding a lone one (no
    slot needed), two bench components onto a fielded unit with a free slot and no lone one, or a lone component off
    another unit with one Item Remover use (its other items go back onto it) onto a fielded unit holding a lone one.
    The cheapest first (no Remover, fewest gives), then the unit the plan's item map gives that item, then the unit it
    suits (items.VALUE). Spatula and Frying Pan (emblems) are left to the plan. Run 20261103-233832: a Needlessly Large
    Rod as Krug's third item (5-5) and Sparring Gloves as Elder Dragon's (6-1) were one item. Told in the next state
    as 'comp rule'. done: the running plan's steps already made ((item, unit id, k)), never redirected. -> the items
    made, [(item, unit id)]."""
    from tft_player.items import VALUE, holder_kind, uses_left
    if not isinstance(plan, dict) or plan.get('_combined') or item_bench is None:
        return []
    try:
        a, b = (int(x) for x in str(stage).split('-')[:2])
    except (TypeError, ValueError):
        return []
    if a < 3:
        # from stage 3 (the owner, game 79 x30.2: "it should make the deathblade, or IE at 3-1, it should not wait for
        # the most optimal items"; game 78 4-6, x28.6: "huge amount of the items left unused")
        return []
    base = lambda n: str(n or '').split(' (')[0]
    if any(base(i) == 'unknown' for i in item_bench if i):
        return []                                     # an unread slot may be a component: wait for its name
    pairable = lambda c: c in COMPONENTS and c not in ('Spatula', 'Frying Pan')
    from tft_player import item_plan
    banned = item_plan.never(getattr(_fixed_comp(), 'name', None))    # the comp's never-built items (owner-ad-comp)
    BY_PAIR = {k: v for k, v in globals()['BY_PAIR'].items() if v not in banned}
    items = [e for e in plan.get('items') or [] if isinstance(e, dict)
             and not e.get('_given') and (e.get('component'), e.get('id'), e.get('k')) not in done]
    bench = Counter(base(i) for i in item_bench if i and pairable(base(i)))
    bench.subtract(base(e.get('component')) for e in items if 'from' not in e and not e.get('_given')
                   and not e.get('use') and pairable(base(e.get('component'))))
    # the item plan's main carry and main tank keep one component per missing item (item_plan.reserved), and its
    # items go on them or on the extra holders only (x20: Elder Dragon or a 3-star Pebbles; no other unit keeps items)
    name = getattr(_fixed_comp(), 'name', None)
    bench.subtract(item_plan.reserved(roster, item_bench, name))
    ip = item_plan.plan_for(name)
    after_board = board_after(plan, roster)[1]
    hs_after = item_plan.holders(after_board, ip) if ip is not None else {}
    third = item_plan.third_holder(after_board, ip, hs_after) if ip is not None else None
    plan_to = None if ip is None else {getattr(u, 'id', None) for u in list(hs_after.values())
                                       + item_plan.extras(after_board, ip) + ([third[0]] if third else [])
                                       if u is not None}
    roles = {} if ip is None else {u.id: r for r, u in hs_after.items() if u}
    team = item_plan.team_items(roster, plan) if ip is not None else Counter()

    def fits(item, t):
        """The carry's open items on him, the tank's on him (one a stat), an extra holder's kind on it; an item whose
        effect does not stack once a team (x24.3, x24.6)."""
        if ip is None:
            return True
        if item_plan.nonstacking(item) and team[item] > 0:
            return False
        if t.id in roles:
            return item in item_plan.open_items(ip, roles[t.id], held.get(t.id, []))
        if third and t.id == third[0].id:             # the third set (the owner, game 76 5-2)
            return item in item_plan.third_items(ip, t.name, third[1], held.get(t.id, []), item_plan.comp_of(ip))
        return item_plan.extra_fits(t, item, ip)
    stripped = {e.get('id') for e in plan.get('strip') or [] if isinstance(e, dict)} | \
        {e.get('from') for e in items if e.get('from') is not None}
    sold = set(plan.get('sell') or [])
    fielded = {getattr(u, 'id', None) for u in board_after(plan, roster)[1]}
    forecast = {u.id: after for _, u, _, after, _ in gives_held(dict(plan, items=items), roster)}
    given = {e.get('id') for e in items}
    ok = lambda u: (u.name and not u.missing and not u.unverified and getattr(u, 'items_read', True)
                    and u.id not in stripped and u.id not in sold and "Thief's Gloves" not in (u.items or ())
                    and 'unknown' not in (u.items or ()))
    now = lambda u: [base(i) for i in u.items or () if i]
    held = {u.id: list(forecast.get(u.id, now(u))) for u in roster.units if ok(u)}
    units = {u.id: u for u in roster.units if u.id in held}
    uses = [uses_left(i) for i in item_bench if i and is_remover(i)]
    uses = sum(1 if n is None else n for n in uses) - sum(
        1 for e in plan.get('strip') or [] if isinstance(e, dict) and e.get('tool') == 'remover')
    key = lambda n: ' '.join(str(n).lower().replace("'", '').split())
    mapped = {(key(champ), key(it)) for champ, its in (item_map.items() if isinstance(item_map, dict) else ())
              for it in (its if isinstance(its, list) else [its])}

    def score(item, t):
        weight = (t.cost + 1) * {1: 1, 2: 1.8, 3: 3}.get(t.star, 1) if t.info else 1
        return (int((key(t.name), key(item)) in mapped), VALUE.get(holder_kind(t), {}).get(item, 0) * weight)

    def better(opt, best):
        return best is None or (opt[0], [-x for x in opt[1]]) < (best[0], [-x for x in best[1]])

    lone = lambda uid: [c for c in held[uid] if pairable(c)]
    made, parts, gives, strips = [], [], [], []
    for _ in range(6):
        singles = sum(max(0, n) for n in bench.values()) + sum(len(lone(uid)) for uid in held)
        if singles < 2:
            break
        best = None                                   # (cost, preference, option)
        spare = [c for c, n in bench.items() if n > 0]
        # the plan's own give of a component that stays single where it goes (6-1: the Gloves as Elder Dragon's third
        # item while Krug held a lone Rod) goes to a fielded unit whose lone component it completes instead
        moved = [(e, units[e['id']], base(e.get('component'))) for e in items
                 if 'from' not in e and not e.get('_given') and e.get('id') in held
                 and (e.get('component'), e.get('id'), e.get('k')) not in done
                 and pairable(base(e.get('component'))) and base(e.get('component')) in lone(e['id'])]
        for t in [units[uid] for uid in held if uid in fielded and (plan_to is None or uid in plan_to)]:
            mine = lone(t.id)
            for c in mine:
                for e, u, n in moved:
                    item = BY_PAIR.get(tuple(sorted((c, n)))) if u is not t else None
                    opt = ((0, 0), score(item, t), ('redirect', t, c, n, item, e, u)) if item and fits(item, t) else None
                    best = opt if opt and better(opt, best) else best
                for n in spare:                       # a bench component onto its lone one
                    item = BY_PAIR.get(tuple(sorted((c, n))))
                    opt = ((0, 1), score(item, t), ('bench', t, c, n, item)) if item and fits(item, t) else None
                    best = opt if opt and better(opt, best) else best
                if uses <= 0:
                    continue
                for s in units.values():              # a lone one off another unit, with one Remover use - only
                    if s is t or s.id in given or held[s.id] != now(s):   # what it holds now (no give lands on it)
                        continue
                    for d in lone(s.id):
                        item = BY_PAIR.get(tuple(sorted((c, d))))
                        opt = ((1, len(held[s.id])), score(item, t), ('remover', t, c, d, item, s)) if item and fits(item, t) else None
                        best = opt if opt and better(opt, best) else best
            if not mine and len(held[t.id]) < 3:
                for i, n in enumerate(spare):         # two bench components onto a free slot
                    for m in spare[i:]:
                        item = BY_PAIR.get(tuple(sorted((n, m)))) if m != n or bench[n] >= 2 else None
                        opt = ((0, 2), score(item, t), ('pair', t, n, m, item)) if item and fits(item, t) else None
                        best = opt if opt and better(opt, best) else best
        if best is None:
            break
        how, t, item = best[2][0], best[2][1], best[2][4]
        if how == 'redirect':
            c, n, e, u = best[2][2], best[2][3], best[2][5], best[2][6]
            parts.append(f"the plan's {n} for {u.name} (unit {u.id}), where it stayed a single component, onto "
                         f'{t.name} (unit {t.id}) with its {c}: {item}')
            e['id'] = t.id
            held[u.id].remove(n)
            held[t.id][held[t.id].index(c)] = item
        elif how == 'bench':
            c, n = best[2][2:4]
            bench[n] -= 1
            gives.append({'component': n, 'id': t.id})
            parts.append(f'{n} from the item bench onto {t.name} (unit {t.id}) with its {c}: {item}')
            held[t.id][held[t.id].index(c)] = item
        elif how == 'pair':
            n, m = best[2][2:4]
            bench[n] -= 1
            bench[m] -= 1
            gives += [{'component': n, 'id': t.id}, {'component': m, 'id': t.id}]
            parts.append(f'{n} and {m} from the item bench onto {t.name} (unit {t.id}): {item}')
            held[t.id].append(item)
        else:
            c, d, s = best[2][2], best[2][3], best[2][5]
            uses -= 1
            strips.append({'id': s.id, 'tool': 'remover'})
            rest = list(held[s.id])
            rest.remove(d)
            gives.append({'component': d, 'id': t.id, 'from': s.id})
            gives += [{'component': i, 'id': s.id, 'from': s.id} for i in rest]
            parts.append(f'{d} off {s.name}{"*" * (s.star or 1)} (unit {s.id}, one Item Remover use; '
                         f'{", ".join(rest) or "nothing"} back on it) onto {t.name} (unit {t.id}) with its {c}: {item}')
            held[t.id][held[t.id].index(c)] = item
            del held[s.id]
            given.add(s.id)
        given.add(t.id)
        made.append((item, t.id))
    plan['_combined'] = True
    k = max([e.get('k', -1) for e in items if isinstance(e.get('k'), int)] + [-1]) + 1
    # a component left that no holder can use, with a Reforger on the item bench: given to a spare bench unit with no
    # items, then the Reforger on that unit - a random component comes back (the owner, game 76 5-2: "it should not
    # leave 5 components ... this is why we should use reforge, we have two ... use one reforger on rod, and
    # potentially another on another rod"; how, the owner: "give, then reforge"). One a plan: the next plan sees what
    # came back
    reforge = _reforge_step(plan, roster, item_bench, bench, given, sold, ip, after_board, hs_after, third,
                            k + len(gives)) if ip is not None and not (plan.get('_reforged')) else None
    if not made and not reforge:
        return []
    plan['items'] = list(plan.get('items') or []) + [dict(g, k=k + n) for n, g in enumerate(gives)]
    if strips:
        plan['strip'] = list(plan.get('strip') or []) + strips
    if made:
        plan['_notes'] = list(plan.get('_notes') or []) + [(
            'comp rule', 'single components make items from stage 3 (owner rule: at most one single component): '
            + '; '.join(parts))]
    if reforge:
        give, strip, note = reforge
        plan['items'] = list(plan.get('items') or []) + [give]
        plan['strip'] = list(plan.get('strip') or []) + [strip]
        plan['_reforged'] = True
        plan['_notes'] = list(plan.get('_notes') or []) + [('comp rule', note)]
    return made


def _reforge_step(plan, roster, item_bench, bench, given, sold, ip, board, hs, third, k):
    """combine_components' reforge: the bench component no holder (the carry, the tank, the third set) can make an item
    of (the pairing above made all it could), given to a spare bench unit, then the Reforger on him -> (give entry,
    strip entry, note) or None. The component the holders' items use least goes first, then the one held most often;
    the spare: a readable benched unit with no items that the plan does not sell, the cheapest first."""
    from tft_player.items import is_reforger
    from tft_player import item_plan
    base = lambda n: str(n or '').split(' (')[0]
    if not any(is_reforger(i) for i in item_bench if i):
        return None
    left = [c for c, n in bench.items() if n > 0 and c in COMPONENTS and c not in ('Spatula', 'Frying Pan')]
    if not left:
        return None
    # only while a holder has room for an item: with every holder full a random component is as useless
    room = [h for h in list(hs.values()) + ([third[0]] if third else []) if h is not None
            and len(item_plan.completed([base(i) for i in h.items or () if i])) < 3]
    if not room:
        return None
    # the items the holders could still take (any partner): a component in none of them is the one to reforge first
    usable = []
    for r, h in hs.items():
        if h is not None:
            usable += [i for i in item_plan.open_items(ip, r, [base(i) for i in h.items or () if i]) if i in RECIPES]
    if third:
        usable += item_plan.third_items(ip, third[0].name, third[1], [base(i) for i in third[0].items or () if i],
                                        item_plan.comp_of(ip))
    uses = Counter(c for it in usable for c in set(RECIPES[it]))
    c = min(left, key=lambda c: (uses[c], -bench[c], c))
    spares = [u for u in roster.bench() if u.name and not u.missing and not u.unverified and not (u.items or ())
              and getattr(u, 'items_read', True) and u.id not in given and u.id not in set(sold)]
    if not spares:
        return None
    from tft_player.catalog import CATALOG
    sp = min(spares, key=lambda u: ((CATALOG.unit(u.name).cost if CATALOG.unit(u.name) else 9), u.star or 1, u.id))
    give = {'component': c, 'id': sp.id, '_rule': 'reforge', 'k': k}
    strip = {'id': sp.id, 'tool': 'reforger', 'after': [c, sp.id, k]}
    note = (f'reforge: {c} (no holder can make an item of it with the components on hand) onto {sp.name} (unit {sp.id}, '
            f'benched, no items), then the Reforger on him - a random component comes back to the item bench')
    return give, strip, note


COMBAT_NOTE = 'COMBAT is running (board locked: no moves or board sells; item gives to fielded units are still tried).'


def _fixed_comp():
    from tft_player.coach import forced_comp
    return forced_comp()


_TARGET_HEXES = {}


def _comp_formation():
    from tft_player.comp_rules import formation
    return formation(getattr(_fixed_comp(), 'name', None))


def target_hexes():
    """The fixed comp's target final board as 'Name (row,col)' from its MetaTFT page (the briefing's section 1),
    or '' - in every state, so the positions are not only in the first message (run 185805: Aphelios and
    the front line ended far from them)."""
    comp = _fixed_comp()
    if comp is None:
        return ''
    if comp.id not in _TARGET_HEXES:
        try:
            from tft_player.metatft_shots import _paths
            text = _paths(comp.name)[2].read_text(encoding='utf-8')
            sec = text.split('1. Target final board, positions', 1)[1].split('\n\n', 1)[0]
            _TARGET_HEXES[comp.id] = ', '.join(f'{n} ({r},{c})' for n, r, c in re.findall(r'^- (.+): row (\d), column (\d)$', sec, re.M))
        except Exception:
            _TARGET_HEXES[comp.id] = ''
        from tft_player.comp_rules import formation
        form = formation(comp.name)
        if form:
            # the owner's formation first (x29.9; 'tank' shown as the comp's main tank), MetaTFT's for the others - a
            # unit whose MetaTFT hex the formation takes moves to the nearest free hex of that row
            from tft_player import item_plan
            ip = item_plan.plan_for(comp.name)
            out = {(ip['tank'][0] if ip else 'Krug') if n == 'tank' else n: tuple(h) for n, h in form.items()}
            for n, r, c in re.findall(r'([^,(]+?) \((\d),(\d)\)', _TARGET_HEXES[comp.id]):
                n, h = n.strip(), (int(r), int(c))
                if n in out:
                    continue
                if h in out.values():
                    free = sorted(((h[0], k) for k in range(7) if (h[0], k) not in out.values()),
                                  key=lambda x: abs(x[1] - h[1]))
                    h = free[0] if free else h
                out[n] = h
            _TARGET_HEXES[comp.id] = ', '.join(f'{n} ({r},{c})' for n, (r, c) in out.items())
    return _TARGET_HEXES[comp.id]


def sell_value(u):
    """Gold a unit sells for: a 1-star its cost; a 2-/3-star all its copies' gold minus 1, except
    1-cost units (full amount)."""
    cost = u.info.cost if u.info else 0
    return cost * 3 ** (u.star - 1) - (1 if u.star > 1 and cost > 1 else 0)


def interest_needs(g, roster, pvp):
    """The interest arithmetic both the model and the overlay use (the user, 2026-10-16): gold now, what the bench sells
    for (unit by unit), and per fight result the gold held when interest is paid (the next round start) and how much
    of the bench must be sold to reach the next step. -> dict, or None without gold."""
    if g is None:
        return None
    named = [u for u in roster.bench() if u.name and u.info]
    worth = sum(sell_value(u) for u in named)
    results = (('win', g + 1), ('loss', g)) if pvp else (('', g),)
    out = {'gold': g, 'worth': worth, 'units': named, 'results': []}
    for label, held in results:
        nxt = min(50, (held // 10 + 1) * 10)
        need = 0 if held >= 50 else nxt - held
        out['results'].append({'label': label, 'held': held, 'next': nxt, 'need': need,
                               'reachable': held >= 50 or need <= worth})
    return out


def econ_line(scr, roster, augments=()):
    """Interest for the model, plainly (the user, 2026-10-16): '24g (interest 2). Bench sells for 8g (Elise 2g, Rakan
    3g, Varus 3g). Next interest at 30: after a win you hold 25g - sell 5g of bench units to reach it; after a loss 24g
    - sell 6g.' Interest is paid at the next round start on the gold held then; an augment can change the rule.
    TFT_NO_INTEREST (experiments/README.md section 9, a variant only): the gold and its interest, without the
    sell-to-reach-the-next-step lines - GPT-6.1-Sol banked 50-92 gold through stage 3 in every game of 2026-10-19."""
    import os
    if os.environ.get('TFT_NO_INTEREST') and scr.gold is not None:
        return f'Economy: {scr.gold}g (interest {min(5, scr.gold // 10)}).'
    try:
        stage, rnd = (int(x) for x in str(scr.stage).split('-'))
    except (TypeError, ValueError):
        stage, rnd = 0, 0
    pvp = stage >= 2 and rnd not in (4, 7)
    n = interest_needs(scr.gold, roster, pvp)
    if n is None:
        return ''
    g = n['gold']
    inv = next((str(a) for a in augments or () if str(a).startswith('Invested+')), None)
    if g >= 50 and inv:
        # Invested+ / Invested++ (card text): 1 Shop reroll at each round start per 10 gold above 50, max 80 - the gold
        # to 80 earns rerolls, not nothing (demonstration 20261023-090521: 80-99 g held in stage 3, 15 rerolls banked by 4-1)
        return (f'Economy: {g}g, interest 5 (the maximum). {inv}: 1 free reroll at each round start per 10g above 50 '
                f'(max 80g): {(min(g, 80) - 50) // 10} at this gold'
                + (f'; the {g - 80}g above 80 earn nothing.' if g >= 90 else '.'))
    if g >= 50:
        # the gold above the cap, said (run 20261021-213247: 78-94g held from 3-1 to 4-1 while HP fell 75 -> 24)
        return f'Economy: {g}g, interest 5 (the maximum)' + (f': the {g - 50}g above 50 earn nothing.' if g >= 60 else '.')
    units = ', '.join(f'[id {u.id}] {u.name}{"*" * u.star} {sell_value(u)}g' for u in n['units'])
    bench = f'Bench sells for {n["worth"]}g ({units}).' if n['units'] else 'Nothing on the bench to sell.'
    parts = []
    for r in n['results']:
        who = (f'after a win you hold {r["held"]}g' if r['label'] == 'win' else
               f'after a loss {r["held"]}g' if r['label'] == 'loss' else f'you hold {r["held"]}g (no win gold this round)')
        if r['need'] == 0:
            parts.append(f'{who} - already there')
        elif r['reachable']:
            parts.append(f'{who} - sell {r["need"]}g of bench units to reach it')
        else:
            parts.append(f'{who} - {r["need"]}g short, more than the bench sells for')
    nxt = n['results'][0]['next']
    return (f'Economy: {g}g (interest {g // 10}). {bench} Next interest at {nxt}: ' + '; '.join(parts) + '.'
            + (' Sells for one result only: execution_plan.after_fight {"win": {"sell": [ids]}, "lose": {...}}.'
               if pvp else ''))


def is_pvp(stage):
    """A player fight: stage 2 on, except the carousel (x-4) and monster (x-7) rounds."""
    try:
        s, r = (int(x) for x in str(stage).split('-'))
    except (TypeError, ValueError):
        return False
    return s >= 2 and r not in (4, 7)


def interest_check(scr, roster, augments=()):
    """The combat-start call's interest decision, in the same plain words as the planning state (econ_line): what to
    sell after a win and after a loss to reach the next step - then where the answer goes. '' without gold (and with
    the TFT_NO_INTEREST variant)."""
    import os
    if os.environ.get('TFT_NO_INTEREST'):
        return ''
    line = econ_line(scr, roster, augments)
    if not line:
        return ''
    try:
        stage, rnd = (int(x) for x in str(scr.stage).split('-'))
    except (TypeError, ValueError):
        stage, rnd = 0, 0
    pvp = stage >= 2 and rnd not in (4, 7)
    return ('INTEREST CHECK: ' + line.removeprefix('Economy: ')
            + ('' if pvp else ' This round is not a player fight: bench sells and buys in execution_plan happen at once.'))


def held_augments(extra):
    """The augments taken, from the state's extra lines ('Augments taken: A, B' - player.screen_extras)."""
    line = next((str(l) for l in extra or () if str(l).startswith('Augments taken: ')), '')
    names = [a.strip() for a in line[len('Augments taken: '):].split(',')]
    return [a for a in names if a and a != 'none']


def streak_from_hp(hp_log):
    """+ wins / - losses in a row, from the HP log ((stage seen, HP before, HP after) per round): a fight was played
    in the planning phase before the one seen, unless that was a monster round (x-7, seen at the next x-1) or the
    carousel (x-4, seen at x-5: game 36 showed "streak now +1" at 3-5 on an 8-loss streak, the carousel's unchanged
    HP read as a win)."""
    streak = 0
    for stage, before, after in reversed(hp_log or []):
        try:
            a, b = (int(x) for x in str(stage).split('-'))
        except (IndexError, ValueError):
            break
        if b in (1, 5) or a < 2 or before is None or after is None:
            continue
        lost = after < before
        if streak and (streak < 0) != lost:
            break
        streak += -1 if lost else 1
    return streak


def missing_units(roster, plan=None):
    """[(name, cost, copies held)] of the target comp's units not yet 2-star (a 2-star counts 3 copies)."""
    comp = _fixed_comp() or (META.comps.get(plan.comp) if plan is not None and getattr(plan, 'comp', None) else None)
    out = []
    for name in (list(comp.units) if comp else []):
        info = CATALOG.unit(name)
        if not info:
            continue
        held = sum(3 ** (u.star - 1) for u in roster.units if u.name == name and not getattr(u, 'missing', False))
        if held < 3:
            out.append((name, info.cost, held))
    return out


def outside_bench(roster, plan=None):
    """(gold, [units]) of the named bench units that are not in the target comp, and what they sell for. ([] when no
    comp is known: nothing is outside it.)"""
    comp = _fixed_comp() or (META.comps.get(plan.comp) if plan is not None and getattr(plan, 'comp', None) else None)
    if not comp:
        return 0, []
    units = [u for u in roster.bench() if u.name and u.info and u.name not in comp.units]
    return sum(sell_value(u) for u in units), units


def roll_gold_lines(scr, roster, plan=None):
    """Next to a roll (request e14, game 40 4-6: the third Brambleback was seen at 2 gold while a benched Mama Beak 2*
    sold for 8): per target-comp unit held but not 2-star yet, the copies a 2-star still needs and the gold on hand the
    roll that shows the last one needs (its cost + 2 for the roll), and what the bench units outside the comp sell for.
    Arithmetic only. [] without gold or a comp."""
    if scr.gold is None or roster is None:
        return []
    need = [(n, c, h) for n, c, h in missing_units(roster, plan) if h > 0]
    worth, units = outside_bench(roster, plan)
    if not need and not units:
        return []
    out = 'ROLL GOLD (arithmetic, no advice): a roll costs 2g (a free reroll 0g).'
    if need:
        out += ' Target units held, not 2-star yet: ' + '; '.join(
            f'{n} {c}g - {h} of 3 copies held, {3 - h} more make the 2-star ({(3 - h) * c}g); the roll that shows the '
            f'last one needs {c + 2}g on hand ({c}g with a free reroll)' for n, c, h in need) + '.'
    if units:
        out += (f' Bench units outside the target comp sell for {worth}g: '
                + ', '.join(f'[id {u.id}] {u.name}{"*" * u.star} {sell_value(u)}g' for u in units) + '.')
    return [out]


def calculator_lines(scr, hp_log, roster=None, plan=None):
    """CLIMB: the arithmetic of this moment from the game model's calculator (agent/model/calc.py, the owner
    2026-10-25: math is computed, not learned) - interest steps, when each level arrives and what buying moves it, the
    shop odds, what a loss costs, what comes up. [] outside the climb or without the HUD numbers."""
    import os
    if os.environ.get('TFT_CLIMB') != '1' or scr.gold is None or not scr.level or not scr.stage:
        return []
    try:
        calc = climb_calc()
        xp = scr.xp[0] if isinstance(scr.xp, (tuple, list)) and scr.xp else 0
        losses = [b - a for _, b, a in (hp_log or []) if b is not None and a is not None and a < b][-3:]
        recent = sum(losses) / len(losses) if losses else None
        if recent is not None and recent < calc.typical_loss(str(scr.stage)):
            # smaller than this stage's typical loss: the runway goes by each stage's typical loss (game 36: 7 HP a
            # loss from stage 2 projected into stages 3-4 said "pays for 8 more" at 3-3 with 60 HP; at the stage sizes
            # 10 and 11.5 it was 5, and the player banked at level 6 on it - 29 HP at 4-1, the line had said 39-43)
            recent = None
        lines = calc.facts(str(scr.stage), int(scr.level), int(xp), int(scr.gold), scr.hp, streak_from_hp(hp_log),
                           recent,
                           missing_units(roster, plan) if roster is not None else None, getattr(scr, 'xp_per_buy', None))
        k = next((i for i, line in enumerate(lines) if line.startswith('Rolling for the units')), None)
        nxt = next_level_roll(calc, str(scr.stage), int(scr.level), int(xp), int(scr.gold),
                              missing_units(roster, plan) if roster is not None else None,
                              getattr(scr, 'xp_per_buy', None))
        if k is not None and nxt and f"level {int(scr.level) + 1} (" not in lines[k]:
            lines[k] = lines[k].rstrip('.') + '; ' + nxt + '.'
        k = next((i for i, line in enumerate(lines) if line.startswith('HP runway: losing')), None)
        also = spike_loss_line(calc, str(scr.stage), int(scr.level), int(xp), scr.hp, recent)
        if k is not None and also:
            lines.insert(k + 1, also)
        return ['CALCULATOR (the arithmetic of this moment; no advice):'] + ['- ' + line for line in lines]
    except Exception as ex:                                  # the state text never fails on the calculator
        return [f'CALCULATOR: not available this round ({ex.__class__.__name__}).']


def next_level_roll(calc, stage, level, xp, gold, missing, per=None):
    """The "Rolling for" figure at the next level when its XP is more than the gold now (calc.roll_by_level shows only
    the levels affordable now): its shop odds, and the rolls left at the next planning phase if the gold is held (base +
    interest, the free XP counted) and the XP bought then. Request e26, game 41 call 83 (5-1 combat, 59 gold, level 9 =
    60 gold): the reply wrote level 9 into next_round seeing only the level-8 figure. None if nothing to add."""
    if not missing or level >= 10 or not calc.to_next(level):
        return None
    now = calc.xp_gold_to(level, xp, level + 1, per)
    if now <= gold:
        return None                                       # the calculator's own row is there
    nxt = calc.next_phase(stage)
    held = gold + calc.base(nxt) + calc.interest(gold)
    then = calc.xp_gold_to(level, xp + calc.M['xp']['passive_per_round'], level + 1, per)
    row = calc.roll_by_level(level + 1, 0, max(0, held - then), missing, up_to=0, per=per)[0]
    out = (f"level {level + 1} (+{now} gold of XP, more than the {gold} you have now): a shop shows one "
           f"{round(100 * row['p_shop_shows_one'])}% of the time")
    if held < then:
        return out + f"; at {nxt}, holding, you have {held} gold, short of its {then} gold of XP"
    return (out + f"; at {nxt}, holding ({held} gold, base + interest), its XP costs {then} and the rest = "
            f"{row['rolls']} rolls, about {row['copies_expected']} copies ("
            + ', '.join(f"{v} {c}-cost" for c, v in row['copies_by_cost'].items()) + ')')


def spike_loss_line(calc, stage, level, xp, hp, recent):
    """Next to the HP runway: the HP left at each of its points if the first fight there is lost too, at the same loss
    (request e13, game 40 review section 6: the runway showed HP at the planned spike, not what one more loss there
    leaves). The points are the runway's own (calc.facts): the free level within 6 phases and the next stage's start.
    '' without HP."""
    if hp is None:
        return ''
    marks = []
    if level < 10 and calc.to_next(level):
        when, n_ph = calc.next_level_at(stage, level, xp)
        if when and n_ph <= 6:
            marks.append((when, f'{when} (level {level + 1} with the free XP)'))
    nxt = calc.label(calc.key(stage)[0] + 1, 1)
    if nxt not in [m for m, _ in marks]:
        marks.append((nxt, nxt))
    marks.sort(key=lambda m: calc.key(m[0]))
    loss = max(1, round(recent)) if recent else None
    out = []
    for until, name in marks:
        fight = until
        while not calc.is_pvp(fight):                    # a PvE or carousel point: its first player fight
            fight = calc.next_phase(fight)
        left, _ = calc.hp_runway(stage, hp, loss, calc.next_phase(fight))
        out.append(f"at {name} {left if left is not None else 'out'}"
                   + (f' (its first player fight, {fight}, lost)' if fight != until else ''))
    return 'HP runway if the first fight there is lost too, at the same loss: HP left ' + '; '.join(out) + '.'


def climb_calc():
    """The game model's calculator (agent/model/calc.py), or None where it is not."""
    import sys
    from pathlib import Path
    model = Path(__file__).resolve().parent.parent / 'agent' / 'model'
    if not (model / 'calc.py').exists():
        return None
    if str(model) not in sys.path:
        sys.path.insert(0, str(model))
    import calc
    return calc


def last_life(stage, hp):
    """A loss ends the game (the owner's x24.7 rule: "1 HP, or less than a loss's damage"): HP 1, or no more HP than
    this stage's typical lost fight costs (the calculator's median of our games' lost fights: 11.5 in stage 4)."""
    if not isinstance(hp, (int, float)) or not stage:
        return False
    if hp <= 1:
        return True
    calc = climb_calc()
    try:
        return calc is not None and hp <= calc.typical_loss(str(stage))
    except Exception:
        return False


def bench_room_lines(scr, roster):
    """With no free bench slot, what a buy (and so a roll order) can still take: only a copy that merges at once - the
    third 1-star (two 1-stars owned). Objective data (request e10: game 37 4-1 rolled 11 times with the bench 9/9, and
    the one Brambleback it found was skipped for lack of a slot)."""
    if not hasattr(roster, 'free_bench') or roster.free_bench():
        return []
    merge = sorted({u.name for u in roster.units if u.name and u.star == 1 and len(roster.copies(u.name)) >= 2})
    seen = [n for n in (getattr(scr, 'shop', None) or []) if n and n in merge]
    return [f"Bench full (9/9 slots taken): a unit bought now lands only if it merges at once (two 1-star copies "
            f"owned: {', '.join(merge) or 'none'}); any other buy, in a roll order too, needs a bench slot freed "
            f"(a sell or a unit fielded) first." + (f" In this shop: {', '.join(seen)}." if seen else '')]


def xp_buy_line(scr):
    """Next-level arithmetic now and after one passive tick (owner e138); no level recommendation."""
    xp, level = getattr(scr, 'xp', None), getattr(scr, 'level', None)
    calc = climb_calc()
    if calc is None:
        return ''
    rules = calc.M['xp']
    need = rules['to_next_level'].get(str(level))
    if not (need and isinstance(xp, (tuple, list)) and len(xp) == 2
            and isinstance(xp[0], int) and 0 <= xp[0] < need and xp[1] == need):
        return ''
    per = getattr(scr, 'xp_per_buy', None) or rules['buy']['xp']
    cost = rules['buy']['gold']

    def buys(points):
        n = max(0, (need - points + per - 1) // per)
        over = points + n * per - need
        landing = 'exactly (0 XP over)' if over == 0 else f'with {over} XP over'
        return f'{n} buys ({n * cost} g) reach level {level + 1} {landing}'

    passive = rules['passive_per_round']
    return (f'XP buys ({per} XP for {cost} gold): now {xp[0]}/{need}: {buys(xp[0])}; '
            f'after next planning phase\'s passive +{passive} XP, {xp[0] + passive}/{need}: '
            f'{buys(xp[0] + passive)}.\n')


def copies_line(roster, name):
    """A shop card's copies held (game 57 4-5: "we own 4" counted units while 8 of Pebbles' 9 copies were held, and
    the 1-gold copy that made Pebbles 3-star was passed): ' - we hold N of 9 copies (units by star); this copy makes
    a k-star', or '' when none is held. A 1-star is 1 copy, a 2-star 3, a 3-star 9."""
    stars = Counter(min(max(u.star or 1, 1), 3) for u in roster.units if u.name == name and not u.missing)
    if not stars:
        return ''
    held = sum(n * 3 ** (s - 1) for s, n in stars.items())
    units = ', '.join(f"{n}x {s}-star" for s, n in sorted(stars.items(), reverse=True))
    after, made = Counter(stars), 0
    after[1] += 1
    for s in (1, 2):
        if after[s] >= 3:
            after[s] -= 3
            after[s + 1] += 1
            made = s + 1
    return (f" - we hold {held} of 9 copies ({units})"
            + (f"; this copy makes a {made}-star" if made else ''))


def state_text(scr, roster, plan, items, hp_log, specials, timer, extra=(), phase='planning', wisps_decided=None,
               goal=None, wisp_flag=None):
    """The exact picture for the model."""
    specials = specials or {}
    lvl = scr.level
    board, bench = roster.board(), roster.bench()
    shop = []
    for i, name in enumerate(scr.shop):
        if name and CATALOG.unit(name):
            info = CATALOG.unit(name)
            shop.append(f"  slot {i}: {name} ({info.cost}g, {'/'.join(info.traits)})"
                        f"{copies_line(roster, name)}{tier_note(board, name)}")
        elif i in specials:
            desc, cost = specials[i]
            shop.append(f"  slot {i}: WISP '{scr.shop_raw[i]}' ({cost if cost is not None else '?'}g): {desc}")
        elif scr.shop_raw[i]:
            shop.append(f'  slot {i}: name not read; raw OCR: {scr.shop_raw[i]!r}')
        else:
            shop.append(f'  slot {i}: empty')
    decided = wisps_decided or {}
    wisps = [f"slot {i} '{scr.shop_raw[i]}'" + (f" (already decided: {decided[i]})" if i in decided else '')
             for i in sorted(specials)]
    wisp_line = ('Wisp in shop: no' if not wisps else
                 f"Wisp in shop: {', '.join(wisps)}" + (
                     " - decide it in execution_plan.wisps." if any(i not in decided for i in specials) else '.'))
    if wisp_flag is not None:         # the flag in force (request e184): {} = none set
        from tft_player import wisp_flag as WF
        wisp_line += f' {WF.text(wisp_flag)}.'
    if timer == 0:                    # combat review: no shop decision
        wisp_line = ''
    def note(it):
        # what hovering the item in game shows (a player always can): consumables, emblems,
        # completed items; components are plain (their recipes are listed below)
        c = consumable_note(it)
        n = c if c is not None else ('' if it in COMPONENTS else item_desc(it))
        return f' [{n}]' if n else ''
    idle = component_idle(items, getattr(scr, 'item_ages', None), scr.stage)
    bench_line = ', '.join(f'slot {k}: ' + ('an unidentified item (being read)' if it == 'unknown'
                                            else f'{it}{note(it)}{idle.get(k, "")}'
                                            + (' (made, no holder: give it to a unit in this plan)'
                                               if needs_holder(it) else ''))
                           for k, it in enumerate(items or []) if it) or 'empty'
    comp = META.comps.get(plan.comp) if plan.comp else None
    target = f'{comp.name} ({comp.id})' if comp else 'none'
    hist = ', '.join(f'{s}: {a}->{b}' for s, a, b in hp_log[-4:]) or 'none yet'
    head = (f"Stage {scr.stage}, planning phase, ~{timer}s left.\n" if phase == 'planning' else
            f"Stage {scr.stage}. {COMBAT_NOTE}\n")
    return (head +
            f"HP {scr.hp if scr.hp is not None else 'not read yet'}"
            + (" (the last phase's: this phase's HP is not read yet; a fight may have been "
               "lost since this reading)" if scr.hp is not None and getattr(scr, 'hp_stale', False) else '')
            + '. '
            f"HP change per round (lost a fight when it drops): {hist}\n"
            + (f"Fight goal: {goal['goal']} (set at {goal['since']}; it stays until strategy.fight_goal changes it)\n"
               if goal else "Fight goal: not set (strategy.fight_goal)\n")
            + (f"Gold {scr.gold}, " if scr.gold is not None else
               "Gold not read yet, ")
            + f"level {lvl if lvl is not None else 'not read yet'}, xp {scr.xp}.\n"
            + xp_buy_line(scr)
            # (the user, 2026-10-11: they do not expire - unused ones stay for later rounds; the model spent them at
            # level 4 where its 3-cost units were rare)
            + (f"Free rerolls: {scr.free_rolls}. They do not expire: unused ones stay for later rounds. Every roll "
               f"(yours, a roll order, a roll-down) uses a free one first; the Reroll button costs 0 gold until they "
               f"are used up. A plan (or next_round) levels first when its XP completes a higher level and leaves "
               f"at least 5 gold to buy hits; otherwise its free rerolls run before XP or stay banked if no hit "
               f"is affordable or the bench cannot take one. Visible missing breakpoint units are bought first. "
               f"\"level_first\": true keeps the XP first; purchase and bench checks still apply.\n"
               # the order the executor runs (_free_first_pending, request e52), stated where the plan is written:
               # game 48 3-5's next_round meant level 7 first, and its 5 free rerolls ran at level 6 into a full bench
               if getattr(scr, 'free_rolls', None) else '')
            + (econ_line(scr, roster, held_augments(extra)) + '\n' if scr.gold is not None else '')
            # in combat too: its reply can write next_round (request e13: game 40's 3-2 to 4-1 next_round plans were
            # written with only the INTEREST CHECK in view, no HP runway)
            + ''.join(line + '\n' for line in calculator_lines(scr, hp_log, roster, plan))
            + ''.join(line + '\n' for line in roll_gold_lines(scr, roster, plan))
            + f"Board ({roster.slots_used()}/{getattr(scr, 'team_size', None) or lvl} team slots used):\n" + ('\n'.join(map(_unit_line, board)) or '  empty')
            + (f"\n{rows_line(board)}" if rows_line(board) else '')
            + ''.join('\n' + line for line in _guarded_lines(bench_cost_lines, roster,
                                                             getattr(scr, 'team_size', None) or lvl))
            + ''.join('\n' + line for line in placement_lines(board, held_augments(extra)))
            + ''.join('\n' + line for line in mechanics_lines(roster))
            + (f"\nTarget final board hexes ({'the comp formation, then MetaTFT' if _comp_formation() else 'MetaTFT'}, "
               f"row,col): {target_hexes()}" if target_hexes() else '') +
            f"\nBench ({len(bench)}/9 units"
            + (f", {len(roster.blocked)} slot(s) taken by other things" if getattr(roster, 'blocked', None) else '')
            + "):\n" + ('\n'.join(map(_unit_line, bench)) or '  no units')
            + ''.join(f"\n  bench slot {k}: {what} - occupies the slot" for k, what in sorted(getattr(roster, 'blocked', {}).items()))
            + (("\nNot located this round (dropped from the records if still not found next round):\n" +'\n'.join(map(_unit_line, roster.missing())))
               if hasattr(roster, 'missing') and roster.missing() else '') +
            f"\nActive traits on board: {active_traits(board)}\n"
            + ''.join(line + '\n' for line in (next_level_lines(scr, roster, lvl) if phase == 'planning' else []))
            + ''.join(line + '\n' for line in (gather_lines(scr, roster) if phase == 'planning' else []))
            + f"Shop:\n" + '\n'.join(shop) + ''.join('\n' + line for line in bench_room_lines(scr, roster)) +
            (f"\n{wisp_line}" if wisp_line else '') +
            f"\nItem bench: {bench_line}"
            + ''.join('\n' + line for line in bench_makes_lines(items) + completes_lines(roster, items)
                      + emblem_lines(roster, items)
                      + _item_plan_lines(roster, items, lvl))
            + ''.join('\n' + line for line in ([] if _fixed_comp() else [f"Target comp: {target}"]) + list(extra)
                      if line)
            + ''.join('\n' + line for line in ((round_data_lines(scr, roster, items, hp_log,
                                                                 getattr(scr, 'team_size', None) or lvl)
                                                + item_fit_lines(roster, items))
                                               if phase == 'planning' else [])))


def _item_plan_lines(roster, items, level=None):
    from tft_player import item_plan
    from tft_player.comp_rules import level_adds, rules
    name = getattr(_fixed_comp(), 'name', None)
    out = item_plan.state_lines(roster, items, name)
    r = rules(name).get('eighth')
    if r and isinstance(level, int) and level >= r['level'] - 1:
        # the owner, game 72 (x24.1): MetaTFT's level-8 and level-9 boards, exactly - the executor buys and fields them
        for lv in (8, 9):
            adds = level_adds(name, lv)
            if adds:
                out.append(f"LEVEL {lv} (MetaTFT's boards, average place): " + '; '.join(
                    f"{', '.join(b)} - adds {u} ({p:.2f})" for u, p, b in adds)
                    + f" - from level {lv} the executor buys {', '.join(u for u, _, _ in adds)} when seen and fields "
                      f"one in the slot beyond Riftbeast 7.")
    return out


def carry_text(leftover, protected, moves_dropped=False, continuing_moves=None):
    """What the model must know it left unfinished. moves_dropped: the carried board moves stopped when this plan
    was asked for (request e177): the board in the state is the board."""
    out = ''
    continuing = continuing_moves and any(continuing_moves.values())
    if leftover:
        doing = ('explicit placements and items part now; generated moves and last round\'s sells stopped'
                 if moves_dropped and continuing else
                 'items part now; its board moves are not made, nor its sells - your plan decides the board and '
                 'what to sell' if moves_dropped else 'board/items part now')
        out += ('\nNOT DONE LAST ROUND (the round ended first; the executor is continuing the '
                + doing + f'): {leftover}')
    if protected:
        out += f'\nProtected units (your last plan wanted them; nothing sells them unless you do): {sorted(protected)}'
    if continuing:
        out += f'\nCarried player destinations (unfinished moves continue while this reply is pending): {continuing_moves}'
    return out


def plan_prompt(state, first=True, carry=''):
    """A planning decision: what is decided, what changed, the state, and a pointer to the reply format (in full
    in the system prompt, which every stage's conversation has)."""
    changes = f'Since your last decision: {carry.strip()}\n' if carry.strip() else ''
    return f'DECISION: planning phase.\n{changes}\nSTATE\n{state}\n\n{PLAN_REPLY}'


def update_prompt(state, progress, remaining):
    """Mid-round: part of the plan is done or failed and time remains."""
    return (f'DECISION: same planning phase, update.\nSince your last decision: {progress}. '
            f'Not done: {remaining or "nothing"}.\n\nSTATE\n{state}\n\n'
            + PLAN_REPLY + ' Include whatever is still to do.')


def review_prompt(state, plan, done, failed):
    """Combat starts: what the controller did with the plan, and the board we fight with."""
    return (f'Combat is starting.\n'
            f'Controller report on your plan: executed: {done or "nothing"}; failed or not reached: '
            f'{failed or "nothing"}.\n\nSTATE\n{state}\n\n{REVIEW_REPLY}')


def once(steps):
    """Report lines, each once: a step retried with the same result is counted, not repeated."""
    n = {}
    for s in steps:
        n[s] = n.get(s, 0) + 1
    return [s + (f' (x{k})' if k > 1 else '') for s, k in n.items()]


def buy_targets(buy):
    """A roll's "buy": a list of champions (each to a 2-star's copies, or its auto_buy star) or
    {"Champion": star} -> {name: star} for the ones given a star."""
    if isinstance(buy, dict):
        return {n: int(s) for n, s in buy.items() if CATALOG.unit(n) and isinstance(s, (int, float)) and 1 <= s <= 3}
    return {}


def stamp_wisps(plan, shop_raw):
    """Each execution_plan.wisps entry gets the name of the Wisp its slot showed in the state the model answered
    (entry 'name'): the plan buys the slot only while it still shows that Wisp. A roll can put another Wisp in the
    slot before the plan runs (run 20261030-101714 5-2: "buy Treetop Archers" in slot 4 bought Wrapped In Thorns,
    which the model had declined, for 2 gold at 27 HP)."""
    for w in (plan or {}).get('wisps') or []:
        i = w.get('slot') if isinstance(w, dict) else None
        if isinstance(i, int) and 0 <= i < len(shop_raw or []) and shop_raw[i] and not w.get('name'):
            w['name'] = shop_raw[i]
    return plan


def same_wisp(a, b):
    """Two reads of a Wisp card's title name the same Wisp (OCR can drop a space or a letter)."""
    from difflib import SequenceMatcher
    k = lambda s: re.sub(r'[^a-z0-9]', '', str(s or '').lower())
    return bool(k(a)) and bool(k(b)) and (k(a) == k(b) or SequenceMatcher(None, k(a), k(b)).ratio() >= 0.8)


def parse(text):
    from tft_player.coach import json_objects
    merged = {}
    for obj in json_objects(text):
        if isinstance(obj, dict):
            merged.update(obj)             # {"execution_plan"} then {"strategy"}
    return normalize(merged) if merged else None


def normalize(ans):
    """{"execution_plan", "strategy"} (the format in the system prompt) -> the
    executor's keys. Older flat replies pass through unchanged."""
    if not isinstance(ans, dict) or 'execution_plan' not in ans:
        return ans
    ep = ans.get('execution_plan') or {}
    st = ans.get('strategy') if isinstance(ans.get('strategy'), dict) else {}
    out = {'strategy': st, 'note': st.get('now'), 'comp': st.get('target_comp')}
    out['_requested_items'] = [dict(e) for e in ep.get('items') or [] if isinstance(e, dict)]
    for k in ('sell', 'open_container', 'wisps', 'buy', 'keep', 'fast_rr', 'next_review',
              'auto_buy', 'auto_sell', 'after_fight', 'after_fight_sell', 'item_final', 'drop_breakpoint',
              'bench_carry', 'force_level', 'skip_fives', 'keep_board', 'keep_items', 'level_first',
              'hold_gold', 'wisp_flag'):
        if ep.get(k) is not None:
            out[k] = ep[k]
    # one bench slot written as a number, not a list (game 72, 5-2 to 6-1: "open_container": 5 crashed every tick, 445
    # tick errors, no step of those planning phases ran)
    if isinstance(out.get('open_container'), int) and not isinstance(out.get('open_container'), bool):
        out['open_container'] = [out['open_container']]
    if ep.get('xp'):
        out['level_ups'] = ep['xp']
    if ep.get('xp') is not None:
        out['_xp_said'] = True           # "xp": 0 too: the reply decided the XP (RoundPlan.credit keeps owed XP otherwise)
    r = ep.get('roll')
    if isinstance(r, dict) and r.get('times') is not None:
        # reroll n times: a Wisp in a new shop stops it for a question, then it goes on
        out['roll_n'] = {'times': r.get('times'), 'buy': r.get('buy') or [],
                         'sell_others': bool(r.get('sell_others', False))}
    elif isinstance(r, dict) and not isinstance(ep.get('fast_rr'), dict):
        # the one roll key: reroll down to to_gold buying the listed champions, stop early at stop_when
        out['fast_rr'] = {'buy': r.get('buy') or [], 'gold_floor': r.get('to_gold'),
                          'stop_when': r.get('stop_when') or [], 'sell_to_make_room': bool(r.get('sell_to_make_room')),
                          'sell_others': r.get('sell_others', True) is not False,
                          'keep': ep.get('keep') or []}
    if ep.get('gold_floor') is not None:
        out['keep_gold'] = ep['gold_floor']
    if ep.get('board') is not None:
        out['board'] = [({'id': e['unit']} if 'unit' in e else {'name': e.get('name')}) | {'hex': e.get('hex')}
                        | ({'when_owned': e['when_owned']} if 'when_owned' in e else {})
                        | ({'dup': True} if e.get('dup') else {}) for e in ep['board'] if isinstance(e, dict)]
    if ep.get('bench') is not None:
        out['bench'] = [{'id': e.get('unit'), 'slot': e.get('slot')}
                        | ({'when_owned': e['when_owned']} if 'when_owned' in e else {})
                        for e in ep['bench'] if isinstance(e, dict)]
    if ep.get('strip') is not None:
        out['strip'] = [{'id': e.get('unit'), 'tool': e.get('tool')} for e in ep['strip'] if isinstance(e, dict)]
    if ep.get('link') is not None:
        out['link'] = [{'from': e.get('from'), 'to': e.get('to')}
                       | ({'when_owned': e['when_owned']} if 'when_owned' in e else {})
                       for e in ep['link'] if isinstance(e, dict)]
    if ep.get('items') is not None:
        # item ACTIONS: give / use (an item or consumable from the item bench), reforge (a Reforger on
        # a unit), transfer (one item from one unit to another: a Remover on the source, then gives -
        # expanded by validate(), which checks every item on the source has a destination)
        items, strip, transfer = [], list(out.get('strip') or []), []
        for e in ep['items']:
            if not isinstance(e, dict):
                continue
            if 'transfer' in e:
                transfer.append({'item': e.get('transfer'), 'from': e.get('from'), 'to': e.get('to')})
            elif 'reforge' in e:
                strip.append({'id': e.get('reforge'), 'tool': 'reforger'})
            elif 'give' in e or 'use' in e:
                items.append({'component': e.get('give') or e.get('use'), 'id': e.get('to', e.get('on'))}
                             | ({'final': e['final'], 'final_star': e.get('final_star')} if e.get('final') else {})
                             | ({'use': True} if 'use' in e and 'give' not in e else {})
                             | ({'stack': True} if e.get('stack') else {}))
            elif 'item' in e:                                  # the older {"item", "unit"} form
                items.append({'component': e.get('item'), 'id': e.get('unit')})
        out['items'] = [dict(e, k=k) for k, e in enumerate(items)]
        if strip:
            out['strip'] = strip
        if transfer:
            out['transfer'] = transfer
    if isinstance(ep.get('item_targets'), list):
        out['item_targets'] = [str(i) for i in ep['item_targets'] if i]
    if isinstance(ans.get('next_round'), dict):
        out['next_round'] = ans['next_round']          # raw; the player normalizes it when it keeps it
    return out



def validate(plan, roster, item_bench=None, names=None, level=None, done=None, queued=(), gold=None, stage=None,
             hp=None):
    """Check every entry of a normalized plan before it runs; drop the bad ones and say exactly
    why (run26 3-5: names in id fields failed late as "item None -> unit None"). Nothing is
    auto-corrected: writing a valid plan is part of the test. Returns the error texts."""
    from tft_player.screen import ScreenReader
    if not isinstance(plan, dict):
        return []
    # ids the model saw when it wrote the plan count too: a unit merged or re-synced since then is
    # found again by name/star when the step runs (RoundPlan.unit) - run27 5-3 rejected two sells
    ids = {u.id for u in roster.units} | set(names or ()) | set(getattr(roster, 'alias', {}))
    errors = []
    is_id = lambda v: isinstance(v, int) and not isinstance(v, bool) and v in ids

    def keep_valid(key, ok, describe):
        vals = plan.get(key)
        if vals is None:
            return
        if not isinstance(vals, list):
            errors.append(f'{key}: must be a list')
            plan.pop(key, None)
            return
        good = []
        for i, v in enumerate(vals):
            why = ('when_owned must name a champion' if isinstance(v, dict) and 'when_owned' in v
                   and not (isinstance(v['when_owned'], str) and CATALOG.unit(v['when_owned'])) else ok(v))
            if why:
                errors.append(f'{key}[{i}] {describe(v)}: {why}')
            else:
                good.append(v)
        plan[key] = good

    def unit_why(v):
        if not is_id(v):
            return (f"'{v}' is a name - this field needs a unit id" if isinstance(v, str)
                    else f'no unit with id {v}')
        u = roster.resolve(v)
        meant = (names or {}).get(v)
        if u and getattr(u, 'identity_corrected', False) and meant and meant[0] != u.name:
            return f'identity corrected to {u.name}; this question saw {meant[0] or "unidentified"}'
        return None
    def sell_why(v):
        from tft_player.state import PLACEABLES
        u = roster.resolve(v) if is_id(v) else None
        if u is not None and u.name in PLACEABLES:
            return f'{u.name} is an Elderwood plant: the game does not let it be sold, only moved'
        return unit_why(v)
    keep_valid('sell', sell_why, repr)
    keep_valid('keep', unit_why, repr)

    def bench_why(e):
        if not isinstance(e.get('slot'), int) or isinstance(e.get('slot'), bool) or not 0 <= e['slot'] <= 8:
            return 'slot must be a bench slot 0-8'
        if not e.get('when_owned') and any(isinstance(b, dict) and b.get('id') == e.get('id')
                                           for b in plan.get('board') or []):
            return 'the unit is also in "board"'
        return unit_why(e.get('id'))
    keep_valid('bench', bench_why, lambda e: f"unit {e.get('id')} -> slot {e.get('slot')}")

    _two_step_moves(plan, roster)
    # one unit per hex (benchmark game 20261021-134046 3-6: Teemo and Krug both at (0, 3) - the swaps
    # ran back and forth for the rest of the planning phase)
    hexes = [list(e['hex']) for e in plan.get('board') or [] if isinstance(e, dict) and isinstance(e.get('hex'), (list, tuple))]
    board_ids = [roster.resolve(e['id']).id if roster.resolve(e['id']) is not None else e['id']
                 for e in plan.get('board') or [] if isinstance(e, dict) and isinstance(e.get('id'), int)]

    from tft_player.comp_rules import TANK_TRAITS, off_comp
    eighth = _eighth_rule(level, plan)
    from tft_player import item_plan as _ip
    _plan = _ip.plan_for(getattr(_fixed_comp(), 'name', None))
    carry_owned = bool(_plan) and any(u.name == _plan['carry'][0] and not u.missing for u in roster.units)

    def no_candidate(name):
        # from the comp's level 8 a champion on none of its boards and with no tank trait is never fielded (the owner,
        # game 78 x28.4: "why field cait at level 8? It should never be a candidate in level 8, you should field
        # brawler / juggernaut etc. to enable tanky traits")
        c = CATALOG.unit(name or '')
        if eighth and c and off_comp(c.name) and not set(c.traits) & set(TANK_TRAITS):
            return (f'{c.name} is on none of the comp\'s boards and has no tank trait ({", ".join(TANK_TRAITS)}): from '
                    f'level {eighth["level"]} it is never fielded - field a comp unit or a tank-trait one')
        return None

    def board_why(e):
        h = e.get('hex') if isinstance(e, dict) else None
        if not (isinstance(h, (list, tuple)) and len(h) == 2 and 0 <= h[0] <= 3 and 0 <= h[1] <= 6):
            return 'hex must be [row 0-3, col 0-6]'
        if hexes.count(list(h)) > 1:
            return f'{hexes.count(list(h))} "board" entries name hex {tuple(h)}: one unit per hex'
        if 'id' in e:
            u = roster.resolve(e['id']) if is_id(e['id']) else None
            if board_ids.count(u.id if u else e['id']) > 1:
                return 'one unit cannot occupy two board entries'
            return unit_why(e['id']) or (no_candidate(u.name) if u is not None and not u.on_board else None)
        if not CATALOG.unit(e.get('name') or ''):
            return f"no champion named {e.get('name')!r}"
        return no_candidate(e.get('name'))
    keep_valid('board', board_why, lambda e: str({k: v for k, v in e.items() if k != 'k'}))
    _reject_duplicate_drops(plan, roster, errors)
    # Keep the player's validated destinations separate from entries generated by
    # comp rules below. Only explicit orders may continue across a new question.
    plan.setdefault('_requested_moves', {k: [dict(e) for e in plan.get(k) or []]
                                         for k in ('board', 'bench')})
    # each comp rule on its own: a fault in one is a harness error, printed and told, never the loss of the model's
    # answer (game 78, 4-3 to 5-6: a TypeError in _level_eight_extras ended the answer thread of every plan with a
    # roll - its next_round plan, its strategy and most of its steps were gone, five rounds without a model call)
    plan['_items_all'] = stage == ITEMS_ALL_STAGE
    for rule, args in ((_keep_comp_breakpoint, (plan, roster, errors)), (_no_second_copy, (plan, roster, errors)),
                       (_keep_distinct_sells, (plan, roster, gold)),
                       (_reach_comp_breakpoint, (plan, roster)), (_off_comp_out, (plan, roster, level)),
                       (_level_eight_extras, (plan, roster, level)), (_extras_add_traits, (plan, roster)),
                       (_level_seven_hold, (plan, roster, level, gold, hp)),
                       (_carry_items_move, (plan, roster, item_bench)), (_use_temporary_items, (plan, roster, item_bench)),
                       (_carry_corner, (plan, roster)), (_main_tank_side, (plan, roster)),
                       (_keep_corner_free, (plan, roster))):
        _guarded(rule, plan, *args)

    known_items = set(ScreenReader.item_names())
    on_bench = {str(i).split(' (')[0] for i in item_bench or () if i}
    # done: the gives the running plan made while the model wrote this update, (item, unit id). The update restates
    # them from the state it saw; they are credited as done (RoundPlan.credit), not rejected for the item having left
    # the item bench (game 26 2-5 and 4-1: Recurve Bow -> Camille and B.F. Sword -> Caitlyn were given, then
    # reported to the player as "rejected" and "failed" in the same phase)
    done_left = list(done or ())

    # a champion this plan buys (its "buy", its roll's buy list) or fields by name has no id yet: an item for it may
    # name it, and goes to that champion once it is ours (game 39 4-1: Rod, Bow and the Alpha Mark "-> Brambleback",
    # bought in the same plan, were rejected as names; the Rageblade waited a round)
    def _names(v):
        v = v if isinstance(v, (list, dict)) else []
        return [CATALOG.unit(n).name for n in v if isinstance(n, str) and CATALOG.unit(n)]
    roll = plan.get('fast_rr') if isinstance(plan.get('fast_rr'), dict) else plan.get('roll_n') or {}
    arriving = set(_names(plan.get('buy')) + _names(roll.get('buy') if isinstance(roll, dict) else None)
                   + _names([e.get('name') for e in plan.get('board') or [] if isinstance(e, dict)]))

    # the items of units this plan sells come back to the item bench when they are sold: a give of one is valid and
    # waits for it to show (game 44 3-5 and 3-6: "sell Ornn, LeBlanc, Rek'Sai" with Warmog's, the Rod and Hand of
    # Justice for Krug and Gromp - each give was rejected as "not on the item bench", and Warmog's went on a round late)
    returning_ids = set(plan.get('sell') or []) | {
        e.get('id') for e in plan.get('strip') or [] if e.get('tool') == 'remover'}
    returning = [str(i).split(' (')[0] for u in roster.units if u.id in returning_ids for i in u.items or ()
                 if i and i != 'unknown']

    def item_why(e):
        it = e.get('component') if isinstance(e, dict) else None
        if not isinstance(it, str) or not it:
            return 'no item named'
        if it not in known_items and it.split(' (')[0] not in known_items:
            return f'{it!r} is not an item name'
        if 'from' not in e and (it, e.get('id')) in done_left:
            done_left.remove((it, e.get('id')))
            e['_given'] = True                  # already on the live roster: never forecast the give twice
            return None
        # an "unknown" slot may be exactly the item named (run27 2-6: a carousel Recurve Bow not yet
        # identified got a valid Rageblade rejected): only reject what the bench surely lacks. A
        # completed item is given only if it is there - never built from components on its own
        tgt = next((u for u in roster.units if u.id == e.get('id')), None)
        if (tgt is not None and 'from' not in e and not e.get('use') and tgt.name and (eighth or carry_owned)
                and off_comp(tgt.name) and CATALOG.unit(tgt.name)
                and not set(CATALOG.unit(tgt.name).traits) & set(TANK_TRAITS)):
            # its items would idle: it is sold or benched (the owner, game 78 x28.2: "the bench cait still not sold,
            # which has 2 items wasted"; the replay's 4-7 plan gave Hand of Justice to Caitlyn as she was swapped out)
            return (f'{tgt.name} (unit {tgt.id}) is on none of the comp\'s boards and has no tank trait: no item goes '
                    f'on it once the carry is owned or from level 8 - give it to a comp unit')
        from tft_player import item_plan as _ip
        _cp = _ip.plan_for(getattr(_fixed_comp(), 'name', None))
        if (tgt is not None and 'from' not in e and _cp and it.split(' (')[0].startswith("Tactician's")
                and tgt.id in _holder_ids(plan, roster, _cp)):
            # (the owner, game 79 6-2, x30.12: "one mistake was to put the crown on the sentinel, it gives no stats.
            # This should not be done")
            return (f"{it.split(' (')[0]} gives no stats: it goes on a unit that holds none of the carry's, the main "
                    f"tank's or the third set's items - not on {tgt.name} (unit {tgt.id}), whose slots are for them")
        if (tgt is not None and 'from' not in e and _cp and tgt.name == _cp['carry'][0]
                and _ip.support_item(it.split(' (')[0])):
            # (the owner, game 78 3-7, x29.6: on Gromp or Cinderling beside him, "so they attack the same opponent
            # unit to reduce the armor")
            return (f"{it.split(' (')[0]} is a support item (its hits reduce the target's armor or magic resist): it "
                    f"goes on another attacker beside {tgt.name}, who attacks the same enemies - not on {tgt.name}")
        if tgt is not None and any(str(i).split(' (')[0] == it.split(' (')[0] for i in tgt.items):
            return None                          # the unit already holds it: counted as done when it runs
        if tgt is not None and 'from' not in e and not any(s.get('id') == tgt.id for s in plan.get('strip') or []):
            from tft_player.state import item_give_error
            blocked = item_give_error(tgt, it)
            if blocked:
                return f'unit {tgt.id}: {blocked}'
        if tgt is not None and THIEF in tgt.items and not is_consumable(it):
            # its 3 slots are taken: the game answers "Item not valid on target" (g6 4-1: two Giant's Belts for
            # Diana and Elise bounced, 4 components sat on the item bench at combat start)
            return f"unit {tgt.id} holds Thief's Gloves: the game refuses any other item on it - pick another unit"
        if 'from' not in e and it.split(' (')[0] in returning and it.split(' (')[0] not in on_bench:
            returning.remove(it.split(' (')[0])  # one give per returning copy
        elif 'from' not in e and item_bench is not None and 'unknown' not in on_bench and it.split(' (')[0] not in on_bench                 and not any(radiant_twin(b, it) for b in on_bench):
            return (f'{it!r} is not on the item bench (it holds: {", ".join(sorted(on_bench)) or "nothing"})'
                    + (f'; to build it, give its components ({" + ".join(RECIPES[it])}) as separate entries'
                       if it in RECIPES else ''))
        why = None if e.get('id') in arriving else unit_why(e.get('id'))
        # (not for a "use": an Alpha Mark is used on its Riftbeast and stays on the item bench - game 12 5-5: rejected
        # twice for want of a "final", placed only at the third try)
        if why is None and 'from' not in e and not is_consumable(it) and not e.get('final') and not e.get('use') \
                and not is_reusable(it) and tgt is not None and tgt.name:
            # every item has two holders: the unit it goes on now ("to") and the champion it is meant for (the user,
            # 2026-10-14); the controller moves it to its final holder when that one is ours. Unnamed, the final holder
            # is the unit it goes on (the state lists it, to be changed any time): a give refused for it left the item
            # on the bench a round - the climb's checker, games 22, 23, 25 (Red Buff, Tear, Chain Vest)
            e['final'] = tgt.name
        return why
    keep_valid('items', item_why, lambda e: f"{e.get('component')} -> unit {e.get('id')}")
    _expand_transfers(plan, roster, item_bench, unit_why, errors)
    _allocate_plan_items(plan, roster, item_bench, errors)
    _reject_stranded_components(plan, roster, errors, queued)

    def buy_why(v):
        return None if (isinstance(v, int) and 0 <= v <= 4) or (isinstance(v, str) and CATALOG.unit(v)) else             'a shop slot 0-4 (or a champion name) is needed'
    keep_valid('buy', buy_why, repr)
    lv = plan.get('level_ups')
    if lv is not None and not (isinstance(lv, int) and 0 <= lv <= 40):
        errors.append(f'xp: {lv!r} must be a whole number of XP buys')
        plan.pop('level_ups', None)
    gap = new_slot_gap(plan, roster, level)
    if gap:
        # the XP runs and the executor fills the new slot (request e186, game 59 2-1: the refusal came 2.6 s before
        # the fight, 2-1 and 2-2 were fought with 3 units while the Pebbles the plan bought sat on the bench); the
        # error stays only when nothing can fill it
        fill = slot_fill(plan, roster, level)
        if fill:
            plan['slot_fill'] = fill
        else:
            errors.append(gap)
    if stage == ITEMS_ALL_STAGE and item_bench is not None:
        # 5-1: every item on the item bench is decided by the model in this plan (the owner, game 78 x28.8: "items
        # should not fully given. Hard code it to force the model to make decision on all of the items at 5-1")
        base = lambda n: str(n or '').split(' (')[0]
        held = Counter(base(i) for i in item_bench if i and base(i) != 'unknown' and not is_consumable(base(i))
                       and not is_reusable(base(i)) and not base(i).startswith('Alpha Mark'))
        held.subtract(base(e.get('component')) for e in plan.get('items') or []
                      if isinstance(e, dict) and 'from' not in e)
        left = _first_ask(stage, sorted((+held).elements()))
        if left:
            errors.append(f'{ITEMS_ALL_STAGE}: every item on the item bench is given in this plan - no give for '
                          f'{", ".join(left)}: give each one to a unit (two components make an item on one unit)')
    elif item_bench is not None:
        # every completed item has a holder (the owner, game 78 x29.5: "all items made should have a carrier, this
        # state is constant in the game, including the final holder and current holder"; an item that lost its holder
        # - Caitlyn sold - waits a round at most: "let the model know next turn and let it decide who to equip it")
        base = lambda n: str(n or '').split(' (')[0]
        made = Counter(base(i) for i in item_bench if i and needs_holder(base(i)))
        made.subtract(base(e.get('component')) for e in plan.get('items') or []
                      if isinstance(e, dict) and 'from' not in e)
        room = any(u.on_board and not u.missing and len([i for i in u.items or () if i]) < 3 for u in roster.units)
        left = _first_ask(stage, sorted((+made).elements())) if room else []
        if left:
            errors.append(f'every completed item has a holder: {", ".join(left)} on the item bench has none - give '
                          f'each one to a unit in this plan')
    return errors


INTEREST_CAP = 50       # the gold from which more gold earns no more interest
_ASKED = {}             # {stage: items a plan was already sent back for this planning phase}


def _first_ask(stage, items):
    """The items of a plan check that sends the plan back (the every-item and the holder checks) which it has not sent
    a plan back for in this planning phase yet: each item once a phase, then the plan runs (game 81 4-2/4-3: an
    augment's Infinity Edge and Rageblade, read late and set aside by the item plan, sent the plan back nine times -
    every model answer of those rounds). The items' names are recorded as asked."""
    if not items:
        return []
    asked = _ASKED.setdefault(stage, set())
    new = [i for i in items if i not in asked]
    asked.update(items)
    return new


def needs_holder(it):
    """A completed item (not a component, a consumable, a reusable tool or the Alpha Mark): it is on a unit, never left
    on the item bench (x29.5). A one-round item (a Wisp's potion) is the executor's (_use_temporary_items)."""
    return bool(it) and it != 'unknown' and it not in COMPONENTS and not is_consumable(it) and not is_reusable(it) \
        and not it.startswith('Alpha Mark') and not temporary_item(it)


def temporary_item(it):
    """A one-round equippable item a Wisp gives (Potioncraft: "Gain three temporary equippable Potions"): its name ends
    in "Potion"."""
    return bool(it) and str(it).split(' (')[0].endswith(' Potion')


def _holder_ids(plan, roster, ip):
    """The ids of the carry's, the main tank's and the third set's holders after the plan's moves."""
    from tft_player import item_plan
    board = board_after(owned_entries(plan, roster), roster)[1]
    hs = item_plan.holders(board, ip, require_usable=False)
    th = item_plan.third_holder(board, ip, item_plan.holders(board, ip)) if ip else None
    return {h.id for h in hs.values() if h is not None} | ({th[0].id} if th else set())


def _use_temporary_items(plan, roster, item_bench):
    """The one-round items on the item bench (temporary_item) go on fielded units with a free slot the same round (the
    owner, game 79 5-6, x30.11: "one wisp gives the one round usage consumable items, not used, this should be implemented
    to use them"): a Health Potion on the main tank's holder first, any other on the carry first, then the strongest
    fielded unit with room (stars, cost). Told in the next state as 'comp rule'."""
    if not isinstance(plan, dict) or item_bench is None:
        return
    from tft_player import item_plan
    base = lambda n: str(n or '').split(' (')[0]
    left = Counter(base(i) for i in item_bench if i and temporary_item(base(i)))
    left.subtract(base(e.get('component')) for e in plan.get('items') or [] if isinstance(e, dict) and 'from' not in e)
    if not +left:
        return
    board = [u for u in board_after(owned_entries(plan, roster), roster)[1] if getattr(u, 'id', None) is not None
             and not u.missing and u.name]
    forecast = {u.id: after for _, u, _, after, _ in gives_held(plan, roster)}
    room = {u.id: 3 - len(forecast.get(u.id, [i for i in u.items or () if i])) for u in board}
    ip = item_plan.plan_for(getattr(_fixed_comp(), 'name', None))
    hs = item_plan.holders(board, ip) if ip else {}
    strong = sorted(board, key=lambda u: item_plan.strength(u), reverse=True)
    gives, said = [], []
    for it in sorted((+left).elements()):
        first = hs.get('tank') if it.startswith('Health') else hs.get('carry')
        for u in ([first] if first is not None else []) + strong:
            if room.get(u.id, 0) > 0:
                room[u.id] -= 1
                gives.append({'component': it, 'id': u.id, '_rule': 'temporary item', 'final': u.name})
                said.append(f'{it} on {u.name} (unit {u.id})')
                break
    if gives:
        k = max([e.get('k', -1) for e in plan.get('items') or [] if isinstance(e, dict)
                 and isinstance(e.get('k'), int)] + [-1]) + 1
        plan['items'] = list(plan.get('items') or []) + [dict(g, k=k + n) for n, g in enumerate(gives)]
        plan['_notes'] = list(plan.get('_notes') or []) + [('comp rule', (
            'the one-round items are used this round: ' + '; '.join(said)))]


ITEMS_ALL_STAGE = '5-1'             # the round whose plan decides every item on the item bench (x28.8)


XP_TO_NEXT = {2: 2, 3: 6, 4: 10, 5: 20, 6: 36, 7: 56, 8: 68, 9: 68}     # xp from level L to L+1 (7 -> 8: 56 on screen)


def new_slot_gap(plan, roster, level):
    """A plan that buys XP up to a new level must say who fills the new board slot(s) when we own the units
    (run 152104 5-7: levelled to 9 in the last second with no ninth unit planned). The board after the plan:
    the units fielded now, less the ones it benches or sells, plus the ones its "board" places. -> error or None."""
    n = plan.get('level_ups')
    lvl, xp = level if isinstance(level, tuple) else (level, None)
    if not n or not isinstance(n, int) or not lvl:
        return None
    cur = xp[0] if isinstance(xp, (tuple, list)) and xp and isinstance(xp[0], int) else 0
    new, pts = lvl, cur + 4 * n
    while new in XP_TO_NEXT and pts >= XP_TO_NEXT[new]:
        pts -= XP_TO_NEXT[new]
        new += 1
    if new == lvl:
        return None
    leaving = {e.get('id') for e in plan.get('bench') or [] if isinstance(e, dict)} | set(plan.get('sell') or [])
    fielded = {u.id for u in roster.board() if not u.missing} - leaving
    placed = [e for e in plan.get('board') or [] if isinstance(e, dict)]
    from tft_player.trait_text import unit_slots
    slots = {u.id: u.slots for u in roster.units}         # team slots: an Elder Dragon fills 2
    count = sum(slots.get(i, 1) for i in fielded | {e['id'] for e in placed if e.get('id') is not None}) \
        + sum(unit_slots(e['name']) for e in placed if e.get('id') is None and e.get('name'))
    owned = sum(u.slots for u in roster.units if not u.missing) + len(plan.get('buy') or [])
    if count >= min(new, owned):
        return None
    return (f'xp: the {n} XP buy(s) take us to level {new}, but the board after this plan has {count} units for '
            f'{new} slots: place the units for the new slot(s) in "board" (an owned unit id, or a champion name '
            f'bought in this plan or its roll, or still arriving; keep it from being sold with "keep")')

def slot_fill(plan, roster, level):
    """The units that may take the new slot(s) a plan's XP opens and its "board" leaves empty (request e186): the
    plan's buys and its roll's in their order (by name: the unit bought), then the benched comp units - a Riftbeast not
    fielded first (the owner's rule: field every distinct Riftbeast) - never one the plan sells or benches, nor an
    unidentified one. -> {'n': slots to fill, 'level': the new level, 'names': [...], 'ids': [...]} or None."""
    n = plan.get('level_ups')
    lvl, xp = level if isinstance(level, tuple) else (level, None)
    if not n or not isinstance(n, int) or not lvl:
        return None
    cur = xp[0] if isinstance(xp, (tuple, list)) and xp and isinstance(xp[0], int) else 0
    new, pts = lvl, cur + 4 * n
    while new in XP_TO_NEXT and pts >= XP_TO_NEXT[new]:
        pts -= XP_TO_NEXT[new]
        new += 1
    leaving = {e.get('id') for e in plan.get('bench') or [] if isinstance(e, dict)} | set(plan.get('sell') or [])
    placed = [e for e in plan.get('board') or [] if isinstance(e, dict)]
    ids = {e.get('id') for e in placed}
    fielded = {u.id for u in roster.board() if not u.missing} - leaving
    from tft_player.trait_text import unit_slots
    count = sum(u.slots for u in roster.units if u.id in fielded | ids)         + sum(unit_slots(e['name']) for e in placed if e.get('id') is None and e.get('name'))
    want = new - count
    if want <= 0:
        return None
    names = [x for x in plan.get('buy') or [] if isinstance(x, str) and CATALOG.unit(x)]
    for key in ('roll_n', 'roll', 'fast_rr'):
        spec = plan.get(key)
        if isinstance(spec, dict):
            names += [x for x in buy_targets(spec.get('buy')) if CATALOG.unit(x) and x not in names]
    comp = _fixed_comp()
    from tft_player.comp_rules import rules
    trait = rules().get('trait')
    core = set(getattr(comp, 'units', None) or ()) | {nm for nm, _, _ in _target_list()}
    onboard = {u.name for u in roster.board() if u.id in fielded} | {e.get('name') for e in placed}
    bench = [u for u in roster.bench() if not u.missing and not u.unverified and u.name and u.id not in leaving
             and u.id not in ids and u.slots == 1 and u.name in core]
    trait_of = lambda u: bool(trait) and u.info is not None and trait in u.info.traits and u.name not in onboard
    # a unit that completes a trait step on the board first - a Riftbeast only when it reaches the next step (the
    # owner, x31.1: "only 3 5 7 10 matters ... when 4 6 8 9 it should be adding extra traits")
    base = [u for u in roster.board() if u.id in fielded]
    steps = lambda u: bool(u.name not in onboard and tiers_gained(base, u))
    bench.sort(key=lambda u: (not steps(u), not trait_of(u), -(u.star or 1), -(u.info.cost if u.info else 0), u.id))
    if not names and not bench:
        return None
    return {'n': want, 'level': new, 'names': names, 'ids': [u.id for u in bench]}


def _target_list():
    """[(name, row, col)] of the fixed comp's target final board (target_hexes)."""
    return [(n.strip(), int(r), int(c)) for n, r, c in re.findall(r'([^,(]+?) \((\d),(\d)\)', target_hexes() or '')]


def gives_held(plan, roster, queued=()):
    """Each item give of the plan, in its order, with what its unit holds before it (after the gives queued before it)
    and after it: a component completes the first lone component the unit holds, as the game combines them. queued:
    gives decided before the plan ({'component', 'id'}: a choice screen's "to", made when the item lands), counted
    first unless the plan restates them. -> [(entry, unit, held before, held after, the entry that completed the
    unit's lone component earlier or None)]."""
    base = lambda n: str(n or '').split(' (')[0]
    held, start, closed, out = {}, {}, {}, []
    reset = {e['id'] for e in plan.get('strip') or [] if e.get('_reset')}
    unstripped = {e['id'] for e in plan.get('strip') or [] if e.get('_reset') and not e.get('_stripped')}
    items = [e for e in plan.get('items') or [] if isinstance(e, dict)] if isinstance(plan, dict) else []
    said = {(base(e.get('component')), e.get('id')) for e in items}
    first = [dict(q, queued=True) for q in queued or () if (base(q.get('component')), q.get('id')) not in said]
    for e in first + items:
        if (not isinstance(e, dict) or e.get('use') or e.get('_given') or not e.get('component')
                or is_consumable(e.get('component')) or is_reusable(e.get('component'))):
            continue
        u = roster.resolve(e.get('id'))
        if u is None or not u.name:
            continue
        it = base(e['component'])
        if u.id not in held:
            held[u.id] = [] if u.id in unstripped else [base(i) for i in u.items or () if i and i != 'unknown']
            start[u.id] = [i for i in held[u.id] if i in COMPONENTS]
        before = list(held[u.id])
        if e.get('from') is not None and e.get('from') == u.id and u.id not in reset:
            continue                                  # an item put back on the unit it came off
        part = next((c for c in before if it in COMPONENTS and c in COMPONENTS and BY_PAIR.get(tuple(sorted((c, it))))),
                    None)
        after = list(before)
        if part is not None:
            after[after.index(part)] = BY_PAIR[tuple(sorted((part, it)))]
        else:
            after.append(it)
        out.append((e, u, before, after, closed.get(u.id) if part is None and it in COMPONENTS else None))
        if part is not None and part in start[u.id]:
            start[u.id].remove(part)
            if not any(c in COMPONENTS for c in after):
                closed[u.id] = e                      # the lone component it held made into an item by this give
        held[u.id] = after
    return out


def _allocate_plan_items(plan, roster, item_bench, errors):
    """One physical copy funds one give; include sales and controller transfers, then check the carry's slots (e57)."""
    if item_bench is None or 'unknown' in item_bench:
        return
    base = lambda n: str(n or '').split(' (')[0]
    bench = Counter(base(i) for i in item_bench if i)
    stripping = {e.get('id') for e in plan.get('strip') or [] if e.get('tool') == 'remover'}
    returning = set(plan.get('sell') or []) | stripping
    sources = {u.id: Counter(base(i) for i in u.items if i and i != 'unknown')
               for u in roster.units if u.id in returning}
    kept = []
    for e in plan.get('items') or []:
        it = base(e.get('component'))
        if e.get('_given') or is_reusable(it):
            kept.append(e)
            continue
        stocks = ([sources.get(e['from'], Counter())] if 'from' in e
                  else [bench] + list(sources.values()))
        target = next((u for u in roster.units if u.id == e.get('id')), None)
        funded = next(((stock, it) for stock in stocks if stock[it] > 0), None)
        if funded is None:
            funded = next(((stock, k) for stock in stocks for k, n in stock.items()
                           if n > 0 and radiant_twin(k, it)), None)
        if funded is None:
            if 'from' not in e and target and it in [base(i) for i in target.items]:
                kept.append(e)                    # an already-completed give repeated by an update
                continue
            errors.append(f'items: {it} -> unit {e.get("id")}: no unallocated copy remains after earlier gives')
            continue
        stock, allocated = funded
        if 'from' not in e:
            source = next((uid for uid, inventory in sources.items() if stock is inventory and uid in stripping), None)
            if source is not None:
                e['from'] = source
        stock[allocated] -= 1
        kept.append(e)
    plan['items'] = kept
    drop = [e for e, u, before, after, _ in gives_held(plan, roster)
            if u.name == 'Brambleback' and len(after) > 3 and not is_consumable(e.get('component'))]
    if drop:
        for e in drop:
            errors.append(f'items: {e["component"]} -> unit {e["id"]}: carry already has three item slots allocated')
            if e.get('from') is not None:
                e['id'] = e['from']              # the rejected fourth item stays with its source
        plan['items'] = [e for e in kept if e.get('from') is not None or not any(e is d for d in drop)]
        idle = {s['id'] for s in plan.get('strip') or [] if s.get('tool') == 'remover'
                and all(e.get('id') == s['id'] for e in plan['items'] if e.get('from') == s['id'])}
        plan['strip'] = [s for s in plan.get('strip') or [] if s.get('id') not in idle]
        plan['items'] = [e for e in plan['items'] if e.get('from') not in idle]


def _reject_stranded_components(plan, roster, errors, queued=()):
    """A component given to a unit whose lone component an earlier give of the same plan already completes, when no
    later give of the plan completes it, is dropped before it runs (request e38, game 44 5-1: Chain Vest and B.F.
    Sword both to Murkwolf's one Giant's Belt - Edge of Night with the Belt left lone, where Sunfire Cape plus a
    Sterak's Gage on Pebbles' lone Giant's Belt were possible - the Vest a choice screen's "to", the Sword the plan's).
    "stack": true on the give makes it anyway."""
    rows = gives_held(plan, roster, queued)
    drop = []
    for k, (e, u, before, after, closed) in enumerate(rows):
        if closed is None or e.get('stack') or e.get('queued'):
            continue
        it = str(e['component']).split(' (')[0]
        if any(e2.get('id') == u.id and BY_PAIR.get(tuple(sorted((it, str(e2['component']).split(' (')[0]))))
               for e2, *_ in rows[k + 1:]):
            continue                                  # a later give makes it an item
        c = str(closed['component']).split(' (')[0]
        by = 'a give queued from a choice screen,' if closed.get('queued') else "this plan's"
        errors.append(f"items: {it} -> unit {u.id}: {u.name}'s lone component is already completed by {by} "
                      f"{c} (it then holds {', '.join(before)}), so the {it} would sit alone on it ({', '.join(after)}) "
                      f"- the give was not made; it stays on the item bench. To give it anyway, add \"stack\": true "
                      f"to the entry")
        drop.append(e)
    if drop:
        plan['items'] = [x for x in plan['items'] if not any(x is d for d in drop)]


def gives_line(plan, roster, queued=()):
    """CLIMB request e38: per give, what its unit holds after the gives queued before it - when a unit gets two or more
    gives, or a give completes an item. -> line or None."""
    rows = gives_held(plan, roster, queued)
    many = Counter(u.id for _, u, *_ in rows)
    fielded = {getattr(u, 'id', None) for u in board_after(plan, roster)[1]}
    show = [(e, u, b, a) for e, u, b, a, _ in rows if many[u.id] > 1 or any(c in COMPONENTS for c in b)
            and len(a) == len(b) or u.id not in fielded]
    if not show:
        return None
    return ("this plan's item gives, in order (each unit's items after the gives before it): "
            + '; '.join(f"{str(e['component']).split(' (')[0]} -> {u.name} (unit {u.id})"
                        f"{' [not fielded after this plan: items idle]' if u.id not in fielded else ''}"
                        f"{' (queued from a choice)' if e.get('queued') else ''}: "
                        f"[{', '.join(b) or 'nothing'}] -> [{', '.join(a)}]" for e, u, b, a in show)
            + ' (forecast only; confirmed items are listed on the owned units)')


def _expand_transfers(plan, roster, item_bench, unit_why, errors):
    """The plan's item transfers -> a Remover on each source unit ('strip') and a give per moved item
    ('items', marked with its source). A Remover takes ALL of a unit's items off, so every item on the
    source must have a destination (another unit, the same unit, or "bench"): a source with an item
    left out is rejected whole, with the items named. Popped once expanded (validate may run twice)."""
    tr = plan.pop('transfer', None)
    if not tr:
        return
    base = lambda n: str(n or '').split(' (')[0]
    on_bench = [base(i) for i in item_bench or () if i]
    by_src = {}
    for e in tr:
        by_src.setdefault(e.get('from'), []).append(e)
    items = plan.setdefault('items', [])
    strip = plan.setdefault('strip', [])
    k = max([e.get('k', -1) for e in items] + [-1]) + 1
    for src, es in by_src.items():
        what = f"transfer from unit {src}"
        why = unit_why(src)
        u = next((x for x in roster.units if x.id == src), None)
        if not why and u is not None and getattr(u, 'items_read', False):
            # a move our Remover has made already (the model re-sent it from a state before the Remover ran - run
            # 20261023-063338 4-6: "Rakan does not hold Evenshroud", rejected while it lay on the item bench): the
            # item on its destination is done, one on the item bench is a plain give
            held, spare, rest = [base(i) for i in u.items], list(on_bench), []
            for e in es:
                name = base(e.get('item'))
                dest = next((x for x in roster.units if x.id == e.get('to')), None)
                if name in held:
                    rest.append(e)
                elif dest is not None and name in [base(i) for i in dest.items]:
                    continue
                elif name in spare and (e.get('to') == 'bench' or dest is not None):
                    spare.remove(name)
                    if dest is not None:
                        items.append({'component': e.get('item'), 'id': e.get('to'), 'k': k})
                        k += 1
                else:
                    rest.append(e)
            if not rest:
                continue
            es = rest
        if not why:
            why = next((unit_why(e.get('to')) for e in es if e.get('to') != 'bench' and unit_why(e.get('to'))), None)
        if not why and item_bench is not None and 'unknown' not in on_bench and \
                not any(i.startswith(('Item Remover', 'Magnetic Remover')) for i in on_bench):
            why = 'there is no Item Remover on the item bench'
        if not why and u is not None and getattr(u, 'items_read', False):
            held = [base(i) for i in u.items]
            moved = [base(e.get('item')) for e in es]
            missing = [i for i in moved if i not in held]
            rest = list(held)
            for i in moved:
                if i in rest:
                    rest.remove(i)
            if missing:
                why = f"{u.name} (unit {src}) does not hold {', '.join(missing)} (it holds: {', '.join(held) or 'nothing'})"
            elif rest:
                why = (f"the Remover takes ALL items off {u.name} (unit {src}); also say where each of these goes: "
                       f"{', '.join(rest)} (another unit, back to unit {src}, or \"bench\")")
        if why:
            errors.append(f'{what}: {why}')
            continue
        strip.append({'id': src, 'tool': 'remover'})
        for e in es:
            if e.get('to') != 'bench':
                items.append({'component': e.get('item'), 'id': e.get('to'), 'from': src, 'k': k})
                k += 1


class PlanNames(dict):
    """Identity and item facts captured together when a question is sent."""

    def __init__(self, units):
        super().__init__((u.id, (u.name, u.star)) for u in units)
        self.items_view = {u.id: tuple(u.items) for u in units}


_WISP_REROLLED = set()              # the planning phases (stages) whose one Wisp reroll a plan made


class RoundPlan:
    """Executes one round's plan; next_action() is called every tick."""
    LATE_S = 10          # seconds of planning left from which the board moves and item gives run before anything else
    LEVEL_LATE_S = 5     # seconds of planning left from which XP waits for the next planning phase (the owner, x18.9)
    level_held = None    # the timer when a level step was first held (told once)
    # a Remover takes ~2 s and each re-give ~2.5 s (run 20261030-132522 5-5: strip 11 -> 9, give 9 -> 6 s): one is
    # started only with the time for it, its re-gives and a margin left
    # a Remover ~2.3-2.8 s and a give at the drag's speed ~0.9-1.2 s (game 88's log: strips 2322 and 2761 ms, give_fast
    # drag + check 0.8-1.2 s); 2.5 s a re-give left the carry's two items on Mama Beak at 4-2 with 8 s ("need ~9 s")
    STRIP_S, REGIVE_S, STRIP_MARGIN_S = 2.6, 1.2, 2.0
    LATE_START_S = 20    # a next_round plan that starts with this little left (a question took the phase) does so too
    MOVE_S = 2.5         # seconds a board move takes with its checks (game 44: 1.6-3 s each)

    def __init__(self, plan: dict, stage, shop=None, carry=False, known=(), names=None, gold=None):
        self.p = plan
        self.stage = stage
        # id -> (name, star) when the plan was written: the roster can give a unit a new id
        # mid-round (re-synced with the screen), and the plan must still find it (run23: Xayah
        # and both Akalis got new ids at 2-2, so "Tear to unit 2" and "field unit 3" found nothing)
        self.names = dict(names or {})
        self.items_view = dict(getattr(names, 'items_view', {}))
        self.last_gold = gold            # a level-up/roll never waits for a gold reading
        self.idle_said = None
        self.carry = carry               # last round's unfinished board/items, run until the new plan lands
        self.source_seq = plan.get('_source_seq', 0)
        self.known = set(known)          # unit ids that existed when the plan was made
        # buys by NAME: slot numbers go stale (a misread shop, a roll), names don't
        shop = shop or [None] * 5
        self.to_buy = [shop[i] if isinstance(i, int) and 0 <= i < 5 else i for i in plan.get('buy') or []]
        self.to_buy = [n for n in self.to_buy if isinstance(n, str) and CATALOG.unit(n)]
        self.sold, self.bought = set(), set()
        self.itemised = {(e.get('component'), e.get('id'), e.get('k'))
                         for e in plan.get('items') or [] if e.get('_given')}
        self.wisps_done = set()
        self.containers_done = set()
        self.levels_done = 0
        self.level_fails = 0
        self.notes = []                  # (kind, text) the player tells in the next state (a comp rule that held)
        self.rolls = 0
        self.done, self.failed = [], []
        self.keep = int(plan.get('keep_gold') or 0)
        self.tries, self.given_up, self.refused, self.linked = {}, set(), set(), set()
        self.item_later = set()          # item gives that failed this round: kept for the next round's plan
        self.unsent_items = {}          # unsent is pending, but must not monopolise the action loop
        self.item_retry_at = {}
        self.item_waits = {}
        self.pair_waits = {}             # ticks a roll-down waited for a give that completes a held component
        self.fast_done = False
        self.rolls_left = None           # roll n times: rerolls still to do (a Wisp question pauses it)
        self.rolled = 0                  # rerolls this plan's roll-downs made
        self.rolln_done = False
        self.free_first_done = False     # the free rerolls ran before the plan's XP buys (request e52)
        self.buy_misses = {}
        self.benched = set()             # bench placements given up (failed)
        self.full_told = set()           # units the full board kept out (said once)
        self.bench_told = set()          # fielded units kept on: nothing the plan fields takes their slot (said once)
        self.late_told = set()           # swaps held late: they would take tiers off before the rest can run (said once)
        self.timer = None                # planning seconds left at the last step asked for
        self.filled = {}                 # unit id -> hex: the new slot(s) the plan's XP opened and left empty (e186)

    def unit(self, roster, uid, used=(), reread=True):
        """The plan's unit `uid`: by id, else the same champion (name, star) the id meant when
        the plan was written (a unit that got a new id), never one already taken. reread: an id read as another
        champion since the plan was written leads to the champion the plan meant, or to nothing (sells pass False:
        they keep their own stricter rule, _sell)."""
        u = next((x for x in roster.units if x.id == uid), None)
        if u is None and isinstance(uid, int):
            u = roster.resolve(uid)              # an id merged away -> the unit it became
        if u is not None and u.missing:
            return None                          # not located: nothing can be done to it
        meant = self.names.get(uid)
        if reread and u is not None and u.id == uid and meant and meant[0] and u.name and u.name != meant[0]:
            # g17 4-1: the question listed unit 24 as Diana 2-star; it was read as Murkwolf while the model wrote
            # "unit 24 -> (3,3)" and "Giant's Belt -> 24": Murkwolf was fielded with the Belt, Diana 2 stayed benched
            u = None
        if u is not None or uid not in self.names:
            return u
        name, star = self.names[uid]
        if not name:
            return None
        cands = [x for x in roster.units if x.name == name and x.id not in used and x.id not in self.names]
        cands = cands or [x for x in roster.units if x.name == name and x.id not in used]
        same = [x for x in cands if x.star == star]
        return (same or cands or [None])[0]

    def pending(self, roster):
        """Steps still to take that have not failed: units not yet placed, items not yet given,
        buys and level-ups not yet done."""
        if any(u.loc != ('board', *h) for u, h in self._lineup(roster)):
            return True
        if any((e.get('component'), e.get('id'), e.get('k')) not in self.itemised for e in self.p.get('items') or []):
            return True
        if self._bench_left(roster):
            return True
        return len(self.bought) < len(self.to_buy) or self.levels_done < int(self.p.get('level_ups') or 0)

    def late_start(self):
        """A next_round plan that started with LATE_START_S or less of the planning phase left: an augment or other
        question took the phase first (request e42, game 45 4-2: the plan started at timer 17 after the augment, an
        item move and the XP went first and Mama Beak's move was queued at timer 1 - level 8 fought 7 of 8 with
        Riftbeast 5 instead of 7). Its board moves and gives go first, as in the last LATE_S seconds."""
        t = getattr(self, 'start_timer', None)
        return getattr(self, 'pre', None) is not None and t is not None and t <= self.LATE_START_S

    def moves_pending(self, roster):
        """A step that changes the board the fight is played with is still to do: a unit to put on (or move to) its
        hex, or a fielded unit to bench. Units the full board kept out do not count (the model is told)."""
        if any(u.loc != ('board', *h) and u.id not in self.full_told for u, h in self._lineup(roster)):
            return True
        return any(u.on_board and not self._bench_waits(roster, u) for u, _ in self._bench_left(roster))

    def protected_ids(self):
        """Units this plan wants: on the planned board or explicitly kept."""
        ids = {e['id'] for e in self.p.get('board') or [] if isinstance(e.get('id'), int)}
        return ids | {i for i in self.p.get('keep') or [] if isinstance(i, int)}

    def leftover(self, roster, failed=True):
        """What the plan asked for but didn't get done (the round ran out):
        (plan dict for the executor to continue, text for the model)."""
        unbought = []
        want = self._lineup(roster, unbought)
        missing = [(u, h) for u, h in want if u.loc != ('board', *h)]
        items = [e for e in self.p.get('items') or [] if (e.get('component'), e.get('id'), e.get('k')) not in self.itemised]
        strips = [e for e in self.p.get('strip') or []
                  if ('strip', e.get('id'), e.get('tool')) not in self.itemised]
        # A completed Remover needs no second use after the boundary; its pending
        # re-gives now come from the item bench.
        items = [dict(e) for e in items]
        for e in items:
            if 'from' in e and ('strip', e['from'], 'remover') in self.itemised:
                e.pop('from')
            if e.get('_after_strip') is not None and ('strip', e['_after_strip'], 'remover') in self.itemised:
                e.pop('_after_strip')
        buys = [n for k, n in enumerate(self.to_buy) if k not in self.bought]
        levels = max(0, int(self.p.get('level_ups') or 0) - self.levels_done)
        lines = [f'{u} not yet at {h}' for u, h in missing]
        lines += [f"{e.get('component')} not yet given to unit {e.get('id')}" for e in items]
        lines += [f"{e.get('tool')} not yet used on unit {e.get('id')}" for e in strips]
        lines += [f'{n} not bought' for n in buys]
        # a board entry naming a champion the roll was to bring: placed once bought, else skipped (game 41 3-5)
        lines += [f'board {n} -> {h}: no {n} owned - placed only once one is bought' for n, h in unbought]
        lines += [f'{levels} level-up(s) not done: they run at the start of the next planning phase'] if levels else []
        # sells of board units are held during combat: they carry over to the planning phase
        sells = [uid for uid in self.p.get('sell') or [] if uid not in self.sold and self._sell_unit(roster, uid)]
        lines += [f'sell {u} not done yet ({"board locked in combat" if u.on_board else "bench action pending"})'
                  for uid in sells if (u := self._sell_unit(roster, uid)) is not None]
        lines += [f'failed: {f}' for f in once(self.failed)[-5:]] if failed else []      # else in progress()
        benches = self._bench_left(roster)
        lines += [f'{u} not yet at bench slot {k}' for u, k in benches]
        # a roll that never ran is told, not carried (g9 6-1: 8 free rerolls ordered in planning and again in combat,
        # neither ran, no prompt said so; the model bought XP at 6-2 and held the rerolls to 6-5 at 13 HP)
        lines += [f'{t} not run (send it again to run it)'
                  for t in getattr(self, 'rolls_dropped', []) + [t for k, t in self.queued(roster) if k == 'roll']]
        carry = {'board': [{'id': u.id, 'hex': list(h)} for u, h in want], 'sell': sells,
                 'items': items, 'strip': strips, 'keep': list(self.protected_ids()),
                 'bench': [{'id': u.id, 'slot': k} for u, k in benches]}             if missing or items or strips or sells or benches or levels else None
        if carry and levels:
            carry['level_ups'] = levels       # held XP (x18.9): bought at the next planning phase's start
            if self.p.get('_level_target'):
                carry['_level_target'] = self.p['_level_target']
            if self.p.get('_all_in_xp'):
                carry.update(_all_in_xp=True, force_level=True, level_first=True)
        if carry:
            carry['_source_seq'] = self.source_seq
            requested = self.p.get('_requested_moves') or {}
            board = []
            for e in requested.get('board') or []:
                u = self.unit(roster, e.get('id')) if 'id' in e else next(
                    (u for u, h in want if u.name == e.get('name') and list(h) == e.get('hex')), None)
                if u is not None and any(x.id == u.id and list(h) == e.get('hex') for x, h in want):
                    board.append({'id': u.id, 'hex': list(e['hex'])})
            bench = [dict(e, id=u.id) for e in requested.get('bench') or []
                     if (u := self.unit(roster, e.get('id'))) is not None
                     and any(x.id == u.id and k == e.get('slot') for x, k in benches)]
            carry['_requested_moves'] = {'board': board, 'bench': bench}
        return carry, '; '.join(lines)

    def drop_rolls(self, roster):
        """Combat started: the planning phase's rolls that have not run are dropped (the combat call tells the model
        "not run (send it again to run it)"; leftover keeps the text). Run 20261029-070238 5-1: "roll x12 for
        Brambleback" was told as not run, then ran in the fight right after the plan's sell of Hecarim, at 3 gold - it
        rolled away the shop with the third Brambleback the model's combat answer was about to buy (14 HP)."""
        texts = [t for k, t in self.queued(roster) if k == 'roll']
        if texts:
            self.rolls_dropped = texts
            self.rolln_done = self.fast_done = True
        return texts

    def queued(self, roster):
        """The steps still to run, in the order the executor takes them (see _next): [(kind, text)] for the overlay."""
        need = self._board_names(roster)
        sells = [self._sell_unit(roster, uid) for uid in self.p.get('sell') or [] if uid not in self.sold]
        sells = [u for u in sells if u is not None]
        buys = [(k, n) for k, n in enumerate(self.to_buy) if k not in self.bought]
        out = []
        for w in self.p.get('wisps') or []:
            if isinstance(w, dict) and w.get('buy') and isinstance(w.get('slot'), int) and w['slot'] not in self.wisps_done:
                out.append(('wisp', f"buy the Wisp in shop slot {w['slot']}"))
        fielding = any(not u.on_board for u, _ in self._lineup(roster))
        out += [('sell', f'sell {u}') for u in sells if u.on_board and not fielding]
        out += [('buy', f'buy {n}') for k, n in buys if n in need]
        levels = max(0, int(self.p.get('level_ups') or 0) - self.levels_done)
        if levels:
            out.append(('level', f'level up x{levels}'))
        out += [('move', f'{u} -> bench {k}') for u, k in self._bench_left(roster)]
        out += [('move', f'{u} -> {h}') for u, h in self._lineup(roster) if u.loc != ('board', *h)]
        out += [('sell', f'sell {u}') for u in sells if u.on_board and fielding]
        for e in self.p.get('items') or []:
            if (e.get('component'), e.get('id'), e.get('k')) not in self.itemised:
                u = self.unit(roster, e.get('id'))
                out.append(('item', f"{e.get('component') or e.get('tool') or 'item'} -> {u or 'unit ' + str(e.get('id'))}"))
        out += [('sell', f'sell {u}') for u in sells if not u.on_board]
        out += [('buy', f'buy {n}') for k, n in buys if n not in need]
        if self.roll_pending():
            rn, fr = self.p.get('roll_n'), self.p.get('fast_rr')
            if isinstance(rn, dict) and not self.rolln_done:
                left = self.rolls_left if self.rolls_left is not None else rn.get('times')
                out.append(('roll', f"roll x{left}" + (f" for {', '.join(rn.get('buy') or [])}" if rn.get('buy') else '')))
            else:
                floor = (fr or {}).get('gold_floor') if isinstance(fr, dict) else (self.p.get('roll') or {}).get('down_to')
                buy = (fr or {}).get('buy') if isinstance(fr, dict) else None
                out.append(('roll', f'roll to {floor or 0}g' + (f" for {', '.join(buy)}" if buy else '')))
        return out

    def progress(self):
        """What happened to the plan so far, for the model: steps done (rolls left out - the
        model doesn't need their count, the user) and steps that failed, with their reasons."""
        done = [d for d in self.done if not d.startswith('roll(')]
        return f"done: {'; '.join(once(done)) or 'nothing'}. failed: {'; '.join(once(self.failed)) or 'nothing'}"

    def record(self, act, ok, cause=None):
        if act.kind == 'buy' and act.why.startswith('plan: wisp'):
            (self.__dict__.setdefault('wisp_bought', set()) if ok
             else self.__dict__.setdefault('wisp_failed', set())).add(act.arg)       # _wisp_gate
        if act.kind == 'rolldown':
            self.roll_ran = True                 # the board is checked again once the plan is done (request e53)
        if act.kind == 'rolldown' and isinstance(act.arg, dict) and act.arg.get('times') is not None                 and not act.arg.get('one_roll'):         # (the one reroll for a Wisp is not the plan's roll)
            self.rolled += int(act.arg.get('done') or 0)
            self.rolls_left = max(0, (self.rolls_left or 0) - int(act.arg.get('done') or 0))
            # (the free rerolls run ahead of the XP: the rest of a roll n times runs after it)
            if (act.arg.get('stopped') not in ('wisp', 'field', 'choice', 'time_slice') and not getattr(act, 'free_first', False))                     or not self.rolls_left:
                self.rolln_done = True
        if act.kind == 'rolldown' and act.arg.get('stopped') in ('field', 'time_slice'):
            self.fast_done = False
        if getattr(act, 'all_in', False):
            # the last life's roll-down goes on after the hit it stopped for is fielded, or the choice row is answered
            self.all_in_done = act.arg.get('stopped') not in ('field', 'choice', 'time_slice')
            self.all_in_field = act.arg.get('stopped') == 'field'
            self.fast_done = True
        if act.kind == 'level' and isinstance(act.arg, int) and getattr(act, 'landed', None) is not None:
            self.levels_done -= act.arg - act.landed           # presses that did not level (gold ran out)
        # (not one that landed elsewhere - its holder records move it - or that no unit shows: it is not on the bench)
        later = (not ok and act.kind == 'item' and getattr(act, 'item_key', None) is not None
                 and not (cause or '').startswith('the game refused it')
                 and ' landed on ' not in (cause or '') and ': missing' not in (cause or ''))
        if later and 'still on the item bench after the drop' in (cause or '') and not getattr(self, 'combat', False):
            # a drop that missed is given again at once, at the unit's other drop point (Player.drop_point), and only
            # after a second miss waits for the next round (the owner, game 88 3-5, x32.2: "cloak give failed, should
            # give again immediately" - the Negatron Cloak for Scuttlecrab missed with 8 s left and waited for 3-6)
            misses = self.__dict__.setdefault('give_misses', Counter())
            misses[act.item_key] += 1
            if misses[act.item_key] < 2:
                self.itemised.discard(act.item_key)
                later = False
                cause = f'{cause} - given again now'
        if later:
            # a give that did not land is no big deal (the user, 2026-10-12): not retried now, carried to the next
            # planning phase with the rest of this plan's unfinished steps (leftover)
            self.itemised.discard(act.item_key)
            self.item_later.add(act.item_key)
            if 'not found among our units in the fight' in (cause or ''):
                # the unit was not found in this fight: its other gives would search for it again, ~3 s each, and
                # miss the same way (game 84 6-2: Sentinel searched for three gives, 6-1..6-3 six misses; game 86
                # 4-5/4-6: Elder Dragon three) - they wait with this one for the next planning phase
                uid = act.item_key[1]
                self.item_later |= {(e.get('component'), e.get('id'), e.get('k')) for e in self.p.get('items') or []
                                    if e.get('id') == uid and 'from' not in e
                                    and (e.get('component'), e.get('id'), e.get('k')) not in self.itemised}
        step = f'{act.kind}({act.why})'
        if ok:
            if act.kind == 'strip' and getattr(act, 'strip_key', None):
                for entry in self.p.get('strip') or []:
                    if ('strip', entry.get('id'), entry.get('tool')) == act.strip_key:
                        entry['_stripped'] = True
            if act.kind == 'item' and getattr(act, 'item_key', None) is not None:
                for entry in self.p.get('items') or []:
                    if (entry.get('component'), entry.get('id'), entry.get('k')) == act.item_key:
                        entry['_given'] = True
            # a retry that worked takes back the step's earlier failure (run 20261016-052413 3-5: the first E did not
            # sell, the retry did, and the model and overlay still read "failed: sell ... gold stayed 48" at 50 gold)
            self.failed = [f for f in self.failed if f != step and not f.startswith(step + ' - ')]
        (self.done if ok else self.failed).append(step + (f' - {cause}' if cause and not ok else '')
                                                  + (' - queued: it is tried again next round' if later else ''))
        if not ok:
            self.tries[repr(act)] = self.tries.get(repr(act), 0) + 1
            if act.kind == 'item' and cause and cause.startswith('the game refused it'):
                # the game said no: the same drop gets the same answer (run 20261010-113319: 3 tries a round)
                self.refused.add((act.arg[1], getattr(act.arg[2], 'id', None)))
            if act.kind == 'strip' and getattr(act, 'strip_key', None):
                # Only a confirmed strip releases its re-gives. After two failed
                # drops, carry the whole recipe to the next planning phase.
                self.itemised.discard(act.strip_key)
                if self.tries[repr(act)] >= 2:
                    self.item_later.add(act.strip_key)
                else:
                    # used again at once, before the phase's other steps (_transfer_first, the owner x32.6)
                    self.__dict__.setdefault('strip_missed', set()).add(act.strip_key)
            if act.kind == 'sell':
                # counted by the unit, not the step's text: the text changes when the failure flags the unit
                # "(unverified)" (game 82 5-5: three sells of Sett*@b5 at an empty slot in 0.6 s, each a new text)
                key = ('sell', getattr(act.arg, 'id', None))
                self.tries[key] = self.tries.get(key, 0) + 1
                if 'is empty on screen' in (cause or '') and ' found at ' not in (cause or ''):
                    self.tries[key] = 2          # no unit at its slot and none found elsewhere: the same E finds nothing
            if act.kind == 'sell' and self.tries[('sell', getattr(act.arg, 'id', None))] < 2:
                # not sold: try once more (a sell that fails twice is reported, not repeated)
                uid = next((k for k in self.p.get('sell') or [] if k in self.sold and
                            self.names.get(k, (None,))[0] == getattr(act.arg, 'name', None)), None)
                if uid is not None:
                    self.sold.discard(uid)
            if act.kind == 'level' and self.level_fails < 3 and getattr(act, 'landed', None) is None:
                # (a press count that was checked is already given back above: run 20261015-003101 3-1 gave
                # one missed press back twice and bought 2 more XP)
                self.level_fails += 1
                self.levels_done -= act.arg      # no confirmed presses: the whole action is still to do

    def snapshot(self):
        """What this plan has done so far: taken when the model is asked to update it."""
        return {'levels': self.levels_done, 'bought': set(self.bought), 'sold': set(self.sold),
                'itemised': set(self.itemised), 'wisps': set(self.wisps_done), 'rolled': self.rolled,
                'done': len(self.done)}

    def credit(self, old, snap):
        """An update replaces `old`, which kept running while the model wrote it (the action path never waits):
        the steps `old` took after `snap` - the state the model saw - count as done in this plan. The model restates
        what it saw as still to do, and run 20261015-003101 did it again: 3-1 "xp 2" ran twice (47 -> 15 gold, with
        a retry), 2-5 bought the XP twice and a Hecarim the update had dropped."""
        # the old plan's board moves and sells that ran after the model was asked: its reply was written from a board
        # without them, so it is told they are done and how to undo one (request e39, game 44 5-5: Maokai for Rakan ran
        # at 7 s, the next reply's "Rakan stays in" was not acted on and Vanguard 2 was off in the last fight)
        # ... and its XP, buys, rolls and gives: game 44 3-3 bought 8 XP (33 -> 1 gold) while the update was written
        # from the state before it, and the reply said "level 7 is delayed to the next phase" (request e41)
        ran = [d for d in old.done[snap.get('done', len(old.done)):]
               if d.startswith(('move(', 'sell(', 'strip(', 'level(', 'buy(', 'rolldown(', 'item('))]
        rolls = sum(1 for d in old.done[snap.get('done', len(old.done)):] if d.startswith('roll('))
        if rolls:
            ran.append(f'{rolls} roll(s)')
        if ran:
            self.notes.append(('already done', f"{'; '.join(once(ran))} - these ran after you were asked, before "
                                               f"your reply arrived: the board, gold and level below have them. To "
                                               f"undo a move, place the unit back with a board entry"))
        lv = max(0, old.levels_done - snap['levels'])
        # XP the old plan still owes stays ordered when this reply sets no XP of its own (the owner, game 88, x32.1: "3-3,
        # should level up, why not? It said in the AI decision level up twice from 3-2, which means it should level up
        # at the beginning of the 3-3" - 3-2's "xp 2" was held in the fight (it took the gold below 50), the two
        # updates that followed had no "xp", and the held XP went with them: level 6 came in the 3-3 fight). "xp": 0
        # in a reply cancels it.
        owed = max(0, int(old.p.get('level_ups') or 0) - old.levels_done)
        if owed and not self.p.get('_xp_said') and not int(self.p.get('level_ups') or 0):
            self.p['level_ups'] = lv + owed
            if old.p.get('_level_target'):
                self.p.setdefault('_level_target', old.p['_level_target'])
            self.notes.append(('xp kept', f'{owed} XP buy(s) of your last plan not done yet stay ordered (your reply '
                                          f'set no "xp"; "xp": 0 cancels them)'))
        self.levels_done = min(lv, int(self.p.get('level_ups') or 0))
        for k in sorted(old.bought - snap['bought']):
            name = old.to_buy[k] if k < len(old.to_buy) else None
            j = next((j for j, n in enumerate(self.to_buy) if n == name and j not in self.bought), None)
            if j is not None:
                self.bought.add(j)
        self.sold |= old.sold - snap['sold']
        for w in self.p.get('wisps') or []:
            slot = w.get('slot')
            was = next((o for o in old.p.get('wisps') or [] if o.get('slot') == slot), None)
            if (was and slot in getattr(old, 'wisp_bought', set()) and slot not in snap['wisps']
                    and (w.get('offer_id') == was.get('offer_id') if w.get('offer_id') and was.get('offer_id')
                         else bool(w.get('name') and was.get('name') and same_wisp(w['name'], was['name'])))):
                self.wisps_done.add(slot)
                self.__dict__.setdefault('wisp_bought', set()).add(slot)
        precredited = Counter((e.get('component'), e.get('id')) for e in self.p.get('items') or [] if e.get('_given'))
        for comp, uid, _ in sorted(old.itemised - snap['itemised'], key=repr):
            if precredited[(comp, uid)]:
                precredited[(comp, uid)] -= 1
                continue                         # validate and __init__ already counted this exact give
            e = next((e for e in self.p.get('items') or [] if (e.get('component'), e.get('id')) == (comp, uid)
                      and (e.get('component'), e.get('id'), e.get('k')) not in self.itemised), None)
            if e is not None:
                self.itemised.add((comp, uid, e.get('k')))
        # a give that did not verify waits for the next round (it may have landed): an update that restates it does
        # not drop it again now (run 20261015-003101 3-2: the Tear did land on Krug, the update repeated "Tear ->
        # Krug" and the second Tear built Blue Buff instead of Protector's Vow)
        for comp, uid, _ in sorted(old.item_later, key=repr):
            e = next((e for e in self.p.get('items') or [] if (e.get('component'), e.get('id')) == (comp, uid)
                      and (e.get('component'), e.get('id'), e.get('k')) not in self.item_later), None)
            if e is not None:
                self.item_later.add((comp, uid, e.get('k')))
        # a give the model was told is "queued: it is tried again next round" stays queued when the update does not
        # restate it, unless the update gives that item to another unit (game 41 3-6: Guinsoo's -> Murkwolf failed in
        # the fight, the update's empty plan replaced the queue, its "Guinsoo's goes on Murkwolf next round" never ran
        # and the Rageblade sat on the item bench through 3-7)
        given = {str(e.get('component') or '').split(' (')[0] for e in self.p.get('items') or []}
        queued = [dict(e) for e in old.p.get('items') or [] if (e.get('component'), e.get('id'), e.get('k'))
                  in old.item_later and str(e.get('component') or '').split(' (')[0] not in given]
        if queued:
            k = max([e.get('k', -1) for e in self.p.get('items') or [] if isinstance(e.get('k'), int)] + [-1]) + 1
            for e in queued:
                e['k'], k = k, k + 1
                self.item_later.add((e.get('component'), e.get('id'), e['k']))
            self.p['items'] = list(self.p.get('items') or []) + queued
        # a transfer half done: `old`'s Remover took a unit's items off after the model was asked (it saw them still
        # on the unit, "done: nothing"), so its reply may not restate the gives - they carry over unless it gives
        # those items itself (run 20261027-125046 4-6: the Remover took Bloodthirster off Diana, the update had no
        # items, Bloodthirster sat on the item bench through the fight)
        stripped = {uid for c, uid, tool in old.itemised - snap['itemised'] if c == 'strip' and tool == 'remover'}
        mine = {str(e.get('component') or '').split(' (')[0] for e in self.p.get('items') or []}
        carried = [dict(e) for e in old.p.get('items') or [] if e.get('from') in stripped
                   and (e.get('component'), e.get('id'), e.get('k')) not in old.itemised
                   and str(e.get('component') or '').split(' (')[0] not in mine]
        if carried:
            k = max([e.get('k', -1) for e in self.p.get('items') or [] if isinstance(e.get('k'), int)] + [-1]) + 1
            for e in carried:
                e['k'], k = k, k + 1
            self.p['items'] = list(self.p.get('items') or []) + carried
            self.itemised |= {('strip', uid, 'remover') for uid in {e['from'] for e in carried}}
        # A full primary repackage can include bench-funded recipe parts as
        # well as returned items. Keep its whole unfinished transaction after
        # a strip, even when the new answer did not repeat those bench parts.
        started = {s.get('_package') for s in old.p.get('strip') or []
                   if s.get('_package') is not None and s.get('_stripped')}
        pending = [e for e in old.p.get('items') or [] if e.get('_package') in started
                   and not e.get('_given') and (e.get('component'), e.get('id'), e.get('k')) not in old.itemised]
        counts = Counter((e.get('component'), e.get('id')) for e in self.p.get('items') or [] if not e.get('_given'))
        for e in pending:
            key = e.get('component'), e.get('id')
            if counts[key]:
                counts[key] -= 1
                continue
            k = max([g.get('k', -1) for g in self.p.get('items') or []] + [-1]) + 1
            self.p.setdefault('items', []).append(dict(e, k=k))
        active = {e.get('_package') for e in self.p.get('items') or [] if not e.get('_given')}
        for s in old.p.get('strip') or []:
            if s.get('_package') not in started & active:
                continue
            key = 'strip', s['id'], s['tool']
            if not any((g['id'], g['tool']) == (s['id'], s['tool']) for g in self.p.get('strip') or []):
                self.p.setdefault('strip', []).append(dict(s))
            if s.get('_stripped'):
                self.itemised.add(key)
        rn = self.p.get('roll_n')
        rolled = old.rolled - snap['rolled']
        if isinstance(rn, dict) and rolled > 0:
            self.rolls_left = max(0, int(rn.get('times') or 0) - rolled)
        return {'levels': self.levels_done, 'buys': len(self.bought), 'rolls': rolled}

    def give_back(self, act):
        """An item step handed out by _items that the caller does not run now (a give burst stops at it): the plan's
        own order hands it out again."""
        key = getattr(act, 'item_key' if act.kind == 'item' else 'strip_key', None)
        if key is not None:
            self.itemised.discard(key)

    def sell_step(self, roster, scr=None):
        """The plan's next bench sell, the first thing of a planning phase (the owner, 2026-11-08: "just sell at the
        start of the planning")."""
        self.combat, self.timer = False, getattr(scr, 'timer', None)
        a = self._sell(roster, board=False)
        if a is not None and (repr(a) in self.given_up or self.tries.get(repr(a), 0) >= 2):
            return None
        return a

    def sells_pending(self, roster):
        """A bench sell of the plan not done yet."""
        return any(uid not in self.sold and getattr(self.unit(roster, uid, reread=False), 'loc', ('',))[0] == 'bench'
                   for uid in self.p.get('sell') or [])

    def unsent(self, act, soon=False):
        """An action this plan handed out was never sent (no input possible): count it as not
        done, so it is handed out again. soon: it waits for a moment that is seconds away (a fight's first seconds, a
        planning phase's bench bars not drawn yet) - a sell then is not counted towards being put off for the phase
        (the owner, 2026-11-08: sells decided in a fight are sold in the fight; two such waits in a fight's first
        seconds used to put them off to the next round)."""
        if act.kind == 'sell' and soon and getattr(act.arg, 'id', None) in self.sold:
            self.sold.discard(act.arg.id)
            return
        if act.kind == 'move':
            counts = self.__dict__.setdefault('unsent_moves', {})
            key = (act.arg[0].id, tuple(act.arg[1]))
            counts[key] = counts.get(key, 0) + 1
        elif act.kind == 'level':
            self.levels_done = max(0, self.levels_done - act.arg)
        elif act.kind == 'roll':
            self.rolls = max(0, self.rolls - 1)
        elif act.kind == 'sell' and getattr(act.arg, 'id', None) in self.sold:
            self.sold.discard(act.arg.id)
            counts = self.__dict__.setdefault('unsent_sells', {})
            counts[act.arg.id] = counts.get(act.arg.id, 0) + 1
        elif act.kind == 'buy' and act.why.startswith('plan: buy '):
            name = act.why[len('plan: buy '):]
            k = max((k for k in self.bought if k < len(self.to_buy) and self.to_buy[k] == name), default=None)
            if k is not None:
                self.bought.discard(k)
        elif act.kind == 'buy' and act.why.startswith('plan: wisp'):
            self.wisps_done.discard(act.arg)
        elif act.kind in ('item', 'strip') and getattr(
                act, 'item_key' if act.kind == 'item' else 'strip_key', None) in self.itemised:
            key = act.item_key if act.kind == 'item' else act.strip_key
            self.itemised.discard(key)
            counts = self.__dict__.setdefault('unsent_items', {})
            counts[key] = counts.get(key, 0) + 1
            self.__dict__.setdefault('item_retry_at', {})[key] = time.monotonic() + 2.0
            if counts[key] >= 2:
                self.__dict__.setdefault('item_later', set()).add(key)
                self.__dict__.setdefault('notes', []).append(
                    ('item queued', f'{act.why}: input not sent twice; still pending for next round'))

    def next_action(self, scr, roster, cap, combat=False):
        """The next step, skipping any that already failed twice. cap: the board slots (the level). combat: the fight is running - the board is locked (no
        board moves, no selling or itemising board units), but buying, selling bench units, rerolling and XP
        work; the rest waits for the next planning phase."""
        self.combat = combat
        refused = lambda a: a.kind == 'item' and (a.arg[1], getattr(a.arg[2], 'id', None)) in self.refused
        # a step that failed once waits behind the others: they run first, it is tried again when none is left (the
        # owner, game 74b: "If there are 5 different independent actions, one failed should not block the rest")
        once = {k for k, n in self.tries.items() if n == 1}
        a = self._next(scr, roster, cap, skip=self.given_up | once) if once else None
        while a is not None and (self.tries.get(repr(a), 0) >= 2 or refused(a)):
            self.given_up.add(repr(a))
            a = self._next(scr, roster, cap, skip=self.given_up | once)
        if a is not None:
            return a
        a = self._next(scr, roster, cap, skip=self.given_up)
        while a is not None and (self.tries.get(repr(a), 0) >= 2 or refused(a)):
            self.given_up.add(repr(a))
            a = self._next(scr, roster, cap, skip=self.given_up)
        return a

    def items_waiting(self):
        """An item step not handed out yet and not given up: a transfer's gives wait for the items its Remover took
        off (or a sold holder's) to show, and be named, on the item bench."""
        if any(('strip', e.get('id'), e.get('tool')) not in self.itemised | self.item_later
               for e in self.p.get('strip') or []):
            return True
        return any((e.get('component'), e.get('id'), e.get('k')) not in self.itemised
                   and (e.get('component'), e.get('id'), e.get('k')) not in self.item_later
                   for e in self.p.get('items') or [])

    def deployment_step(self, roster, cap, scr=None):
        """Already-owned board entries only, before reads, items or questions (owner e227).

        Use the existing move/dependency checks, but do not let a reposition or a
        bench rearrangement hide a champion waiting to enter the fight.
        """
        self.combat, self.timer = False, getattr(scr, 'timer', None)
        if not any(not u.on_board for u, _ in self._lineup(roster)):
            return None
        def accept(a):
            if a.kind != 'move' or repr(a) in self.given_up or self.tries.get(repr(a), 0) >= 2:
                return None
            return a
        return self._board(roster, cap, accept)

    def board_step(self, roster, cap, scr=None, dependencies=False, gives=True):
        """Only the next board move or item hand-out, while the model rewrites an interrupted plan (a Wisp rerolled the
        shop): no step that spends is handed out, so none is counted as done (run 20261021-213247 3-5: next_action
        handed out the 8 XP buys, the caller dropped them as not a move, and the update was credited with them - level 6
        all round). Items spend no gold and do not depend on the shop: they go on (run 20261022-064841 4-2: four
        components waited the whole phase for an answer about the shop; the fight was lost 54 -> 40)."""
        self.combat = False
        self.timer = getattr(scr, 'timer', None)
        a = self._board(roster, cap, lambda a: a if repr(a) not in self.given_up and self.tries.get(repr(a), 0) < 2
                        else None)
        if a is None and scr is not None and dependencies:
            a = self.placement_dependency(scr, roster, cap)
        if a is None and scr is not None and gives:
            a = self._items(scr, roster)
            refused = a is not None and a.kind == 'item' and (a.arg[1], getattr(a.arg[2], 'id', None)) in self.refused
            if a is not None and (refused or repr(a) in self.given_up or self.tries.get(repr(a), 0) >= 2):
                a = None
        return a

    def placement_dependency(self, scr, roster, cap):
        """Only spending already ordered to open a slot for an owned placement.

        Ordinary owned moves still run first. A full board must not let unrelated
        item retries hide the XP (or its authorised funding sales) it needs.
        """
        if not roster.free_bench() and any(not u.on_board and u.slots > 1 for u, _ in self._lineup(roster)):
            return self._sell(roster, board=False)
        if not any(not u.on_board and roster.slots_used() + u.slots > cap
                   for u, _ in self._lineup(roster)):
            return None
        if int(self.p.get('level_ups') or 0) <= self.levels_done:
            return self._sell(roster, bench=False)
        gold = scr.gold if scr.gold is not None else (self.last_gold or 0)
        if scr.timer is not None and (scr.timer <= self.LATE_S or self.late_start()):
            action = self._late_level(gold, scr, roster)
        else:
            action = self._free_first(scr, gold) or self._xp(gold, scr, roster)
        # Healthy owned gives retain their priority over funding sales. Unsent
        # gives yield through their bounded retry path, then these sales can run.
        return action or (None if self.items_waiting() else self._sell(roster, board=False))

    @staticmethod
    def fast_rr_met(fr, roster):
        """Every stop_when entry has one of its champions with that star level's copies (3 for a 2-star,
        9 for a 3-star) on board + bench: a 1-star fielded and two benched count as a 2-star, merged
        or not."""
        conds = [c for c in fr.get('stop_when') or [] if isinstance(c, dict) and c.get('any')]

        def copies(n):
            return roster.proven_copies(n)
        return bool(conds) and all(any(copies(n) >= 3 ** (int(c.get('star') or 2) - 1) for n in c['any'])
                                   for c in conds)

    def _fast_rr(self, fr, scr, roster, gold):
        want = {n for n in fr.get('buy') or [] if CATALOG.unit(n)}
        floor = int(fr.get('gold_floor') or 0)
        if self.fast_rr_met(fr, roster):
            self.fast_done = True
            self.done.append('fast_rr: stop condition met')
            return None
        if fr.get('wisps') and not getattr(self, 'combat', False):
            i = next((k for k in range(5) if scr.shop_raw[k] and not scr.shop[k]), None)
            if i is not None and ('fast', self.rolls, i) not in self.wisps_done:
                self.wisps_done.add(('fast', self.rolls, i))
                return Action('buy', i, f'plan: fast_rr wisp {scr.shop_raw[i]}')
        for i, name in enumerate(scr.shop):
            if name not in want or gold - CATALOG.unit(name).cost < min(floor, gold):
                continue
            # a third copy merges on the spot: no bench space needed
            if roster.free_bench() or len(roster.copies(name, 1)) >= 2:
                return Action('buy', i, f'plan: fast_rr buy {name}')
            if fr.get('sell_to_make_room'):
                keep = {k for k in fr.get('keep') or [] if isinstance(k, int)} | self.protected_ids()
                spare = [u for u in roster.bench() if u.id not in keep and u.name not in want]
                if spare:
                    u = min(spare, key=lambda x: (x.cost * x.star, x.loc))
                    return Action('sell', u, f'plan: fast_rr make room ({u})')
        if gold - 2 >= floor:
            self.rolls += 1
            return Action('roll', None, f'plan: fast_rr roll {self.rolls} (floor {floor})')
        self.fast_done = True
        self.done.append(f'fast_rr: stopped at gold {gold} (floor {floor})')
        return None

    def _bench_left(self, roster):
        """[(unit, slot)] the plan's "bench" placements not done yet (nor given up)."""
        out = []
        for e in self.p.get('bench') or []:
            if not self.arrival_ready(e, roster):
                continue
            u = self.unit(roster, e.get('id'))
            k = e.get('slot')
            if u is not None and isinstance(k, int) and u.loc != ('bench', k) and (u.id, k) not in self.benched:
                out.append((u, k))
        return out

    LANDING_DOUBT = 'its landing slot was not seen'          # Player.note_landing: our own buy, its slot predicted

    @staticmethod
    def arrival_ready(entry, roster, located=False):
        """A dependent move waits until the named arrival is actually owned (located: and its place confirmed - a link
        drags onto it). A unit we bought whose landing slot was not seen is owned: its name is the shop card's (request
        e187, game 58 3-3: the roll bought Gromp at timer 13, its landing unseen; "bench Rammus when_owned Gromp" waited
        for the read until combat, and "Gromp -> (0,5)" failed on a full board at timer 1)."""
        name = entry.get('when_owned')
        sure = lambda u: not u.unverified or (not located and str(u.unverified).startswith(RoundPlan.LANDING_DOUBT))
        return not name or any(u.name == name and not u.missing and sure(u) for u in roster.units)

    def arrival_pending(self, roster):
        """A preplaced arrival is benched; yield rolling/questions to its board moves."""
        incoming = {u.id for u, _ in self._lineup(roster) if not u.on_board and u.id not in self.known}
        return any(self.arrival_ready(e, roster) and (
            e.get('name') and any(u.id in incoming and u.name == e['name'] for u in roster.units)
            or e.get('when_owned') and (incoming or self.moves_pending(roster)))
            for e in self.p.get('board') or [])

    def _lineup(self, roster, unresolved=None):
        """[(unit, (row, col))] resolving ids, then names of units bought this round. A name entry with no unit of
        that champion owned (the roll has not bought it, or never did) is skipped: unresolved gets (name, hex)."""
        want, used = [], set()
        # A temporary placement ends when its conditional departure is ready
        # (game 63 5-1, Krug fills the eighth slot until Elder Dragon arrives).
        departing = {u.id for e in self.p.get('bench') or []
                     if e.get('when_owned') and self.arrival_ready(e, roster)
                     for u in [self.unit(roster, e.get('id'))] if u is not None}
        for e in self.p.get('board') or []:
            if not self.arrival_ready(e, roster):
                continue
            hex_ = tuple(e.get('hex') or ())
            if len(hex_) != 2 or not (0 <= hex_[0] <= 3 and 0 <= hex_[1] <= 6):
                continue
            u = self.unit(roster, e.get('id'), used) if 'id' in e else None
            if u is None and e.get('name'):
                u = next((x for x in sorted(roster.units, key=lambda x: (-x.star, not x.on_board))
                          if x.name == e['name'] and not x.missing and not x.unverified
                          and x.id not in used and x.id not in departing), None)
            if u is not None and u.id in departing:
                continue
            if u is not None and u.id not in used:
                want.append((u, hex_))
                used.add(u.id)
            elif u is None and e.get('name') and unresolved is not None:
                unresolved.append((e['name'], hex_))
        # a benched copy fielded while a fielded copy of the same champion, holding items, stands on a hex the board
        # entries take: that one moves instead and the bare copy stays benched (run 20261022-090637 4-5: "spare
        # Rakan 34 -> (2,4), Sentinel -> (3,3)" benched Rakan 6 with Protector's Vow and Stoneplate; Sentinel
        # fought 4-6 with no items, lost 51 -> 39). Hexes of entries not resolved yet count: a buy comes later
        taken = {tuple(e.get('hex') or ()) for e in self.p.get('board') or []}
        for k, (u, hex_) in enumerate(want):
            if u.on_board or u.items:
                continue
            f = next((x for x in roster.board() if x.name == u.name and (x.star or 1) >= (u.star or 1) and x.items
                      and x.id not in used and x.id not in departing and x.loc[1:] in taken), None)
            if f is not None:
                want[k] = (f, hex_)
                used.add(f.id)
        out = want + self._slot_fill(roster, want, used)
        if getattr(self, 'mirrored', False):
            # the board took its mirror positions for the fight (Player.mirror_action, the owner's x29.10): the plan's
            # hexes, written for the normal positions, are kept on the mirror side
            out = [(u, (h[0], 6 - h[1])) for u, h in out]
        return out

    def _fill_waits(self):
        """A slot fill (e186) waits for the plan's XP: the new slot opens with it."""
        return bool(self.p.get('slot_fill')) and self.levels_done < int(self.p.get('level_ups') or 0)

    def _slot_fill(self, roster, want, used):
        """The new slot(s) the plan's XP opened and its "board" left empty (request e186): each takes a unit the plan
        or its roll bought (in its buy order), else a benched comp unit (planner.slot_fill), on a free hex - its target
        hex when free. Only once the XP is bought; a unit chosen keeps its hex. Told once: 'the new slot took <unit>'.
        -> [(unit, hex)]"""
        fill = self.p.get('slot_fill')
        if not isinstance(fill, dict) or self._fill_waits():
            return []
        leaving = {e.get('id') for e in self.p.get('bench') or [] if isinstance(e, dict)} | set(self.p.get('sell') or [])
        mine = {u.id for u, _ in want}
        count = sum(u.slots for u in roster.board() if u.id not in leaving or u.id in mine)             + sum(u.slots for u, _ in want if not u.on_board)
        out = []
        for uid, hex_ in list(self.filled.items()):
            u = next((x for x in roster.units if x.id == uid and not x.missing), None)
            if u is None or u.id in used:
                del self.filled[uid]
                continue
            out.append((u, hex_))
            used.add(u.id)
            if not u.on_board:
                count += u.slots
        taken = {tuple(e.get('hex') or ()) for e in self.p.get('board') or [] if isinstance(e, dict)}             | {h for _, h in want} | {h for _, h in out}
        while count < int(fill.get('level') or 0):
            free = lambda u: (not u.on_board and not u.missing and not u.unverified and u.slots == 1
                              and u.id not in used and u.id not in leaving)
            u = next((x for n in fill.get('names') or [] for x in roster.units
                      if x.name == n and free(x) and x.id not in self.known), None)
            u = u or next((x for i in fill.get('ids') or [] for x in [self.unit(roster, i, used)]
                           if x is not None and free(x)), None)
            if u is None:
                break
            hex_ = self._free_hex(roster, u, taken)
            if hex_ is None:
                break
            self.filled[u.id] = hex_
            out.append((u, hex_))
            used.add(u.id)
            taken.add(hex_)
            count += u.slots
            self.notes.append(('slot fill', f'the new slot took {u.name} (unit {u.id}) at hex {hex_} (your plan '
                                            f'bought the XP and left it empty)'))
        return out

    @staticmethod
    def _free_hex(roster, u, taken):
        """A hex for a unit the plan did not place: its comp target hex when free, else the first free hex of the rows
        its range suits (melee from the front row, ranged from the back)."""
        free = lambda h: h not in taken and roster.at(('board', *h)) is None
        for n, r, c in _target_list():
            if n == u.name and free((r, c)):
                return r, c
        rows = (3, 2, 1, 0) if (u.info is not None and u.info.melee) else (0, 1, 2, 3)
        return next(((r, c) for r in rows for c in range(7) if free((r, c))), None)

    def _next(self, scr, roster, cap, skip=()):
        if scr.gold is not None:
            self.last_gold = scr.gold
        self.timer = getattr(scr, 'timer', None)
        gold = scr.gold if scr.gold is not None else (self.last_gold or 0)
        ok = lambda a: a if repr(a) not in skip else None
        # little time left in the planning phase: the board moves and item gives first, before the wisps, buys and
        # rolls - a move cannot run in the fight and a give there must find a moving unit (request e34, run
        # 20261030-064325 4-1: the update came with ~5 s left, the Blast Potion wisp went first, one of five gives
        # ran, Krug's two were tried in the fight and not found). Units still to buy are fielded after their buy; the
        # plan's level-up goes first when the board is full and a unit is still to field (it opens the slot)
        if getattr(self, 'start_timer', None) is None and scr.timer is not None and not getattr(self, 'combat', False):
            self.start_timer = scr.timer
        if getattr(scr, 'hp', None) is not None:
            self.last_hp = scr.hp
        if getattr(scr, 'level', None):
            self.level_now = scr.level
        self.board_now = roster.board()                  # (the roll level's low-HP opening: _roll_level_stop)
        if any(getattr(scr, 'shop_raw', None) or ()):
            self.shop_seen = list(scr.shop_raw)
        if not getattr(scr, 'hp_disputed', False) and self.all_in(getattr(self, 'last_hp', None)):
            # the last life (all_in): the moves that field a hit the roll stopped for, the plan's bench sells, the XP
            # that completes the next level, then the roll-down of all the gold - before the board, the items and any
            # anvil
            act = ((self._board(roster, cap, ok) if getattr(self, 'all_in_field', False) else None)
                   or self._sell(roster, board=False) or self._all_in_level(scr, gold))
            if act:
                return act
            if self.p.get('_all_in_xp') and self.levels_done < int(self.p.get('level_ups') or 0):
                return self._level(gold, scr, roster) or self._board(roster, cap, ok)
            return self._all_in_roll()
        if not getattr(self, 'combat', False):
            act = self._transfer_first(scr, roster)
            if act:
                return act
            act = self._wisp_reroll(scr, roster, gold)
            if act:
                return act
        if (not getattr(self, 'combat', False) and scr.timer is not None and (scr.timer <= self.LATE_S or self.late_start())):
            full = roster.slots_used() >= cap and (any(not u.on_board for u, _ in self._lineup(roster))
                                                   or self._fill_waits())
            act = ((self._late_level(gold, scr, roster) if full else None) or self._board(roster, cap, ok)
                   or self._links(roster) or self._items(scr, roster))
            if act:
                return act
        # 0. wisps the model chose to buy, before anything else (a Wisp can reroll the shop or give gold) - but a
        #    listed buy this shop already shows comes first when the gold pays it with the Wisp's cost left (game 83
        #    3-6: the Jungling Wisp's new shop took the listed Mama Beak away). A wisp slot is text that isn't a
        #    champion. Never in combat: Wisps leave the shop when it starts
        act = None if getattr(self, 'combat', False) else self._buys_before_wisp(scr, roster, gold)
        if act:
            return act
        for w in ([] if getattr(self, 'combat', False) else self.p.get('wisps') or []):
            i = w.get('slot') if isinstance(w, dict) else None
            if not w.get('buy') or not isinstance(i, int) or not 0 <= i < 5 or i in self.wisps_done:
                continue
            if w.get('name') and not same_wisp(w['name'], scr.shop_raw[i]):
                continue                         # the slot shows another Wisp now: the answer was for the one before
            if w.get('cost') is not None and gold < w['cost']:
                continue                         # explicit bench sales can fund this accepted offer
            # (only now is the slot's step spent: game 50 6-1, the first update's "buy Petrify Shields" in slot 4,
            # rolled away, marked slot 4 done, and the next update's "buy Nature's Ally" there was never tried)
            self.wisps_done.add(i)
            if scr.shop_raw[i] and not scr.shop[i]:
                action = Action('buy', i, f"plan: wisp {scr.shop_raw[i]} ({w.get('why') or ''})")
                action.wisp_key = (self.stage, i, w.get('name') or scr.shop_raw[i])
                action.wisp_offer = w.get('offer_id')
                return action
        # 0b. a choice container (an anvil) first while the choice can still be answered: the model needs
        #    ~9 s, and opened last it came with ~8 s left (run 152203 4-3: the pick was lost, the game chose).
        #    Later than that it waits for the fight, where opening works too.
        if not getattr(self, 'combat', False) and scr.timer is not None and scr.timer >= 11:
            act = self._container()
            if act:
                return act
        if not getattr(self, 'combat', False):
            # the plan's bench sells at once (the owner, game 74b 2-6: "it should be sold instantly. Do not make actions
            # sequential and block the next actions"), then the already-owned placements (e187). The item gives come
            # after the plan's roll (below): they land in the fight too (the owner, game 74b 4-6: "should reroll and
            # buy wisp first and then give items ... you could give items even planning stage ends")
            act = (self._sell(roster, board=False) or self._board(roster, cap, ok)
                   or self.placement_dependency(scr, roster, cap))
            if act:
                return act
        act = self._comp_buy(scr, roster, gold)
        if act:
            return act
        if getattr(self, 'combat', False):
            # the fight: the board is locked - bench sells, buys, XP and rolls work; board moves wait for the next
            # planning phase, items for the countdown before the fight (countdown_action) or the next planning phase.
            # In an away fight the bench is elsewhere: its sells wait for the next planning phase and do not hold up
            # what comes after them (run 20261023-063338 4-1: "sell LeBlanc, buy Hecarim" - the sell, deferred again
            # every tick, kept Hecarim unbought all fight; 4-2 was fought without Vanguard 4, 52 -> 40)
            away = getattr(self, 'away', None) == scr.stage
            return ((None if away else self._sell(roster, board=False)) or self._buy(scr, roster, gold)
                    or self._free_first(scr, gold) or self._xp(gold, scr, roster) or self._rolldown(gold)
                    or self._container())
        # planning, by priority (the user, 2026-10-10): the board the fight is played with, then the items, then the
        # bench's sells and buys (they also work in the fight). What the board needs comes first: selling board
        # units, buying the units it fields (or a copy of one), the level-ups
        need = self._board_names(roster)
        bench_first = self._needed_buy_blocked(scr, roster, gold, need)
        # a board unit the plan sells waits while units the plan fields are still on the bench: each is dragged onto
        # the hex it takes (a swap), so the board is never short when the round ends, and the benched unit is sold
        # from the bench (that works in the fight too). A sale that frees a team slot comes right after the moves;
        # first only when a buy the board needs is short of gold or bench room (run 20261019-234104 4-6: four board
        # sells first, one unit fielded with 3 s left - the fight was played on half a board; 20261019-230743 4-5)
        fielding = any(not u.on_board for u, _ in self._lineup(roster))
        # free rerolls ahead of the XP: right away when the level opens the slot a unit waits for, else after the
        # board, items and other buys (request e52)
        slot = roster.slots_used() >= cap and fielding
        return (self._sell(roster, board=bench_first or not fielding, bench=bench_first)
                or self._buy(scr, roster, gold, only=need)
                or (self._free_first(scr, gold) if slot else None)
                or self._xp(gold, scr, roster) or self._board(roster, cap, ok) or self._sell(roster, bench=False)
                or self._links(roster)
                or self._sell(roster) or self._buy(scr, roster, gold, exclude=need)
                or self._free_first(scr, gold) or (None if self._pair_hold(roster) else self._rolldown(gold))
                or self._items(scr, roster)
                or self._container())

    def _transfer_first(self, scr, roster):
        """A Remover's transfer runs as one step, before anything else of the planning phase: a Remover that missed is
        used again at once, and the items it took off go onto their new holders right after it (the owner, game 88
        4-6, x32.6: "It removes 3 items and did not give it back ... there is no time left to give them. this should not
        happen, it should give two items to the sett" - Scuttlecrab's Remover missed at 14 s and was used again at 4 s
        after a move, two gives and a roll; Evenshroud and Crownguard for Sett stayed on the item bench all fight).
        -> the strip or re-give, or None."""
        if scr.items is None or getattr(self, 'combat', False):
            return None
        pending = self.itemised | self.item_later
        key = lambda g: (g.get('component'), g.get('id'), g.get('k'))
        regive = any(('from' in g and ('strip', g['from'], 'remover') in self.itemised
                      or g.get('_after_strip') is not None and ('strip', g['_after_strip'], 'remover') in self.itemised)
                     and key(g) not in pending for g in self.p.get('items') or [])
        missed = any(('strip', e.get('id'), e.get('tool')) in getattr(self, 'strip_missed', set())
                     and ('strip', e.get('id'), e.get('tool')) not in pending for e in self.p.get('strip') or [])
        if not (regive or missed):
            return None
        return self._items(scr, roster, transfers_only=True)

    def _buys_before_wisp(self, scr, roster, gold):
        """With a Wisp of this plan still to buy: the plan's next listed buy that this shop shows, when the gold pays it
        and still leaves the Wisp's cost (3 when unknown) and the plan's floor. Nothing is marked failed here - the
        plan's own buy step judges the rest after the Wisp."""
        want = [w for w in self.p.get('wisps') or [] if isinstance(w, dict) and w.get('buy')
                and isinstance(w.get('slot'), int) and w['slot'] not in self.wisps_done]
        if not want or not getattr(scr, 'shop_visible', True) or gold is None:
            return None
        reserve = max(3 if w.get('cost') is None else int(w['cost']) for w in want)
        for k, name in enumerate(self.to_buy):
            info = CATALOG.unit(name)
            i = next((j for j in range(5) if scr.shop[j] == name), None)
            if k in self.bought or info is None or i is None:
                continue
            if gold - info.cost < max(reserve, min(self.keep, gold)):
                continue
            if not roster.free_bench() and len(roster.copies(name, 1)) < 2:
                continue
            self.bought.add(k)
            return Action('buy', i, f'plan: buy {name} (before the Wisp: its effect may refresh the shop)')
        return None

    PAIR_HOLD = 3        # ticks a roll-down waits for such a give's slot to read

    def _pair_hold(self, roster):
        """A give of this plan that completes an item with a component its unit already holds is still to go: the
        roll-down (one action of up to ~20 s) waits a few ticks for it. Right after a give the item bench shifts and
        its slots read unsteady for a tick, so the give finds no slot and the roll-down ran first (game 56 5-1, log
        1921-1928: Recurve Bow went on Brambleback at 9 s, Hand of Justice at 5 s, then 12 rolls for 20 s; the Rod
        that made Guinsoo's was tried in the fight and not found - Brambleback fought with a lone Recurve Bow)."""
        for e in self.p.get('items') or []:
            key = (e.get('component'), e.get('id'), e.get('k'))
            if key in self.itemised or key in self.item_later or 'from' in e or e.get('component') not in COMPONENTS:
                continue
            u = self.unit(roster, e.get('id')) if isinstance(e.get('id'), int) else None
            if u is None or not any(tuple(sorted((i, e['component']))) in BY_PAIR for i in u.items or []):
                continue
            n = self.pair_waits.get(key, 0)
            if n < self.PAIR_HOLD:
                self.pair_waits[key] = n + 1
                return True
        return False

    def _comp_buy(self, scr, roster, gold):
        """Buy a funded first carry/Sentinel or missing core-trait champion before XP or a refresh (e57)."""
        from tft_player.comp_rules import rules, trait_count
        r = rules()
        if not r.get('entry_hexes') or not getattr(scr, 'shop_visible', True):
            return None
        have = {u.name for u in roster.units if not u.missing}
        short = trait_count(roster.board(), r['trait']) < r['keep']
        for i, name in sorted(enumerate(scr.shop or []), key=lambda x: x[1] not in r['entry_hexes']):
            info = CATALOG.unit(name) if name else None
            if (info is None or name in have or roster.pending(name) or info.cost > gold
                    or not (name in r['entry_hexes'] or short and r['trait'] in info.traits)
                    or not roster.free_bench()):
                continue
            a = Action('buy', i, f'comp rule: buy first {name} before XP/refresh')
            if self.tries.get(repr(a), 0) < 2:
                return a
        return None

    def fight_give(self, scr, roster):
        """The fight is running: the plan's hand-outs not done yet go on now - the executor finds the moving unit by
        reading it off its panel first (the user, 2026-10-14: items can be given in combat; run 20261014-063145 4-6:
        Aphelios fought without the items the plan gave him, they went on at the next planning phase)."""
        self.combat = True
        a = self._items(scr, roster, gives_only=True)
        if a is not None:
            a.fight = True
        return a

    def countdown_action(self, scr, roster):
        """The 5-4-3-2-1 countdown before the fight: units stand still on their hexes (in an away fight they leave at
        2) and only items can be given - the plan's items not given in planning go on now (the user, 2026-10-10)."""
        self.combat = True
        return self._items(scr, roster)

    def _board_names(self, roster):
        """Champions the plan's board fields (by name or unit) and the ones fielded now: buying one of them changes
        the board (a unit to field, or a copy toward its next star)."""
        names = {u.name for u in roster.board() if u.name}
        for e in self.p.get('board') or []:
            if e.get('name'):
                names.add(e['name'])
            elif 'id' in e:
                u = self.unit(roster, e['id'])
                names.add((self.names.get(e['id']) or (None,))[0] or getattr(u, 'name', None))
        names.discard(None)
        return names

    def _needed_buy_blocked(self, scr, roster, gold, need):
        """A buy the board needs is in the shop but the bench is full or the gold short: the sells go first."""
        cost = 0
        for k, name in enumerate(self.to_buy):
            if k in self.bought or name not in need or name not in (scr.shop or []):
                continue
            cost += CATALOG.unit(name).cost
            if not roster.free_bench() and len(roster.copies(name, 1)) < 2:
                return True
        return cost > 0 and gold - cost < min(self.keep, gold)

    def _merged_away(self, roster, uid):
        """The unit a sell id leads to only through a merge the plan never knew of (the id was not a unit when the
        plan was written): selling it would sell the merged unit, not the copy meant. None if not so."""
        if uid in self.names or any(x.id == uid for x in roster.units):
            return None
        return roster.resolve(uid) if isinstance(uid, int) else None

    def _sell_unit(self, roster, uid):
        """The unit a sell id would sell, by _sell's own rules - or None: an id merged away, a unit now read as another
        champion, or one of another star level than meant. (Run 20261022-225248 7-1: "sell 65, 66" for two benched
        1-star copies, already sold by the plan before; the ids led by name to the board's Morgana 2 with three items and
        Camille 2, which _sell refused - but the model was told "sell Morgana** not done yet" and spent its last reply
        of the final duel cancelling it.)"""
        u = self.unit(roster, uid, reread=False)
        if u is None or self._merged_away(roster, uid) is not None or self._stand_in(uid, u):
            return None
        meant = self.names.get(uid)
        if meant and ((meant[0] and u.name != meant[0]) or u.star != meant[1]):
            return None
        return u

    def _stand_in(self, uid, u):
        """A sell id that is gone (sold, merged) leads by name to another unit the model knew under its own id when
        it planned: that is a different unit, never the one meant. (Run 20261028-042121 5-6: "sell 43", the bench Mama
        Beak sold at the start of the phase, was repeated by the next plan; it led by name to the board Mama Beak,
        unit 22, and sold it with Last Whisper, Red Buff and Guinsoo's Rageblade.)"""
        return u is not None and u.id != uid and (u.id in self.names or u.on_board or bool(u.items))

    def explicit_spare_sale(self, u):
        """A positively identified, itemless bench copy sold by its own id in this question."""
        return (u.id in (self.p.get('sell') or []) and self.names.get(u.id) == (u.name, u.star)
                and u.id not in self.protected_ids() and not u.on_board and not u.missing
                and not u.unverified and getattr(u, 'sure', False) and not u.items)

    def _sell(self, roster, board=True, bench=True):
        """The plan's next sell: of a board unit (board), of a bench unit (bench)."""
        for uid in self.p.get('sell') or []:
            if getattr(self, 'unsent_sells', {}).get(uid, 0) >= 2:
                continue                        # kept pending, but another step can use this phase
            u = self.unit(roster, uid, reread=False)
            meant = self.names.get(uid)
            if self._stand_in(uid, u) and self._merged_away(roster, uid) is None and uid not in self.sold:
                self.sold.add(uid)
                self.failed.append(f'sell unit {uid}: no unit has that id now (sold or merged before) - '
                                   f'{u} (unit {u.id}) is another unit, not sold')
                continue
            merged = self._merged_away(roster, uid)
            if merged is not None and uid not in self.sold:
                # run 20261022-010349 5-2: the Duplicator's copy (unit 37) merged at once into Sentinel 2; "sell the spare
                # copy 37" resolved through the merge and sold the 2-star with its Sunfire Cape
                self.sold.add(uid)
                self.failed.append(f'sell unit {uid}: that copy merged into {merged} (unit {merged.id}) - not sold')
                continue
            if u and meant and meant[0] and u.name != meant[0] and uid not in self.sold:
                # the unit was re-read as another champion since the plan was written (run 221257 5-3:
                # "sell the spare Leona" sold Aphelios, re-read as Aphelios after the plan): not sold
                self.sold.add(uid)
                self.failed.append(f'sell unit {uid}: it is now read as {u.name}{"*" * u.star}, '
                                   f'not {meant[0]} as when you planned it - not sold')
                continue
            if u and meant and not meant[0] and u.name and u.id == uid and uid not in self.sold:
                # unidentified when the model planned, named since: the plan was made on '?' (run 20261103-233832
                # 6-2: "sell 55, misidentified" - unit 55 was read as Brambleback with Rageblade, Quicksilver and
                # Edge of Night 7 s before the sell, and the carry was sold with all three)
                self.sold.add(uid)
                why = (f'sell unit {uid}: it is now read as {u.name}{"*" * u.star}'
                       f'{" holding " + ", ".join(u.items) if u.items else ""}, unidentified ("?") when you planned it '
                       f'- not sold')
                self.failed.append(why)
                self.notes.append(('unit_corrected', why))      # told at once: the plan was made on '?'
                continue
            if u and u.star != (meant or (None, u.star))[1]:
                u = None                         # never sell a different star level than meant
            if u is None or uid in self.sold or not (board if u.on_board else bench):
                continue                         # (a fighting unit can't be sold: after the fight)
            from tft_player.comp_rules import plan_targets, upgrade_keep
            if (not u.on_board and uid in upgrade_keep(roster, plan_targets(self.p))
                    and not self.explicit_spare_sale(u)):
                self.sold.add(uid)
                self.notes.append(('comp rule', f'kept {u}: this copy advances an unfinished star upgrade'))
                continue
            self.sold.add(uid)
            return Action('sell', u, f'plan: sell {u}')
        return None

    def _board_sell_pending(self, roster):
        """A board unit the plan sells, not sold yet: its sale frees the team slot a unit to field waits for."""
        return any(uid not in self.sold and getattr(self._sell_unit(roster, uid), 'on_board', False)
                   for uid in self.p.get('sell') or [])

    def _buy(self, scr, roster, gold, only=None, exclude=()):
        """The plan's next buy, by name, in the order given (each entry once). A buy is done once clicked. A missing
        read is not an answer: without a gold or shop reading, wait; give it up only when the champion is absent from
        4 shop reads in a row, or truly unaffordable (before, one blank read dropped it for good - 2 buys a game)."""
        for k, name in enumerate(self.to_buy):
            if k in self.bought or (only is not None and name not in only) or name in exclude:
                continue
            if not getattr(scr, 'shop_visible', True) or (scr.gold is None and self.last_gold is None):
                return None
            i = next((j for j in range(5) if scr.shop[j] == name), None)
            if i is None:
                self.buy_misses[k] = self.buy_misses.get(k, 0) + 1
                if self.buy_misses[k] >= 4:
                    self.bought.add(k)
                    self.failed.append(f'buy {name}: not in the shop')
                continue
            if gold - CATALOG.unit(name).cost < min(self.keep, gold):
                self.bought.add(k)
                self.failed.append(f'buy {name}: gold {gold} is below the floor')
                continue
            if not roster.free_bench() and len(roster.copies(name, 1)) < 2:
                # the game refuses it (a 3rd copy would merge and is fine): say so at once instead of
                # clicking and waiting 2.5 s for nothing - with Cluttered Mind the bench is kept full
                self.bought.add(k)
                self.failed.append(f'buy {name}: the bench is full (9/9) - sell a unit first in the same plan')
                continue
            self.bought.add(k)
            return Action('buy', i, f'plan: buy {name}')
        return None

    def _level(self, gold, scr=None, roster=None):
        """Level-ups: all of them in ONE action - F pressed n times in a row (0.79 s per level-up with a screen read
        each in run26; a player just presses F n times).
        An order the gold cannot pay in full waits for the plan's own sells, then buys only the XP that completes a
        level (the climb's game 37 4-3: "xp 12" at 10/56 with 44 gold ran level(11) -> 54/56 and 3 gold - no level 8
        for the fight, every interest step gone; R11: a partial XP buy that completes no level buys nothing yet).
        Buys the gold pays in full run as ordered (gold above 50 into XP stays the player's call)."""
        left = int(self.p.get('level_ups') or 0) - self.levels_done
        target = self.p.get('_level_target')
        level, xp = getattr(scr, 'level', None), getattr(scr, 'xp', None)
        if level and xp and xp[1] > 0:
            from tft_player.pilot import XP_TO_NEXT
            per = getattr(scr, 'xp_per_buy', None) or 4
            if target is None and left * per + xp[0] >= xp[1]:
                target, points, required = level, xp[0] + left * per, xp[1]
                while required and points >= required:
                    points -= required
                    target += 1
                    required = XP_TO_NEXT.get(target)
                self.p['_level_target'] = target
            if target:
                needed = xp[1] - xp[0] + sum(XP_TO_NEXT.get(lv, 0) for lv in range(level + 1, target))
                if level >= target:
                    self.levels_done = int(self.p.get('level_ups') or 0)
                    return None
                left = min(left, max(0, math.ceil(needed / per)))
        if left <= 0 or gold < 4:                     # explicit: not held to keep_gold
            return None
        # the owner (x16.3, x18.9; games 65 3-2 and 66 3-5): no XP in a fight, nor in the last LEVEL_LATE_S s of planning
        # (no unit can be fielded after it) - it runs at the next planning phase's start, after that round's income
        timer = getattr(scr, 'timer', None)
        # ... except in the fight with the gold above the interest cap: XP from the gold over 50 loses no interest (the
        # owner, game 74 x25.4: "3-2 during combat, it should buy xp and level up, because it does not affect interest,
        # since have 79 gold")
        rich = getattr(self, 'combat', False) and gold - 4 * left >= 50
        if not rich and (getattr(self, 'combat', False) or (timer is not None and timer <= self.LEVEL_LATE_S)):
            if self.level_held is None:
                self.level_held = timer
                when = 'the fight is on' if getattr(self, 'combat', False) else f'{timer} s of planning left'
                self.notes.append(('level held', f'{left} XP buy(s) held: {when} - they run at the start of the next '
                                                 f'planning phase'))
            return None
        n = min(left, gold // 4)
        if n < left:
            if roster is not None and self._sells_pending(roster):
                return None                           # the sells come first: their gold pays for the rest
            n = self._levelling_buys(scr, n)
            if n == 0:
                msg = (f'XP held: {gold} gold pays {gold // 4} of the {left} XP buys ordered, and they complete no '
                       f'level - nothing bought (sell first, or send the XP again)')
                if msg not in self.failed:
                    self.failed.append(msg)
                return None
        n = self._comp_level_hold(n, gold, scr, roster)
        if n <= 0:
            return None
        if self._inexact_level_waits(n, gold, scr, roster):
            return None
        self.levels_done += n
        return Action('level', n, f'plan: level up x{n}')

    def _inexact_level_waits(self, n, gold, scr, roster):
        """The owner, game 74b 3-2 (x26.10): "there is no reason to level to 6, it wasted 2 gold interest doing this.
        First, the xp is not perfect, it is 2/36, even you want to level up, do it next round" - XP buys that leave XP
        over the level they reach and take the gold below an interest step wait for the next planning phase (its passive
        XP), unless a unit this plan fields waits for the slot. Told once. -> True when held."""
        level, xp = getattr(scr, 'level', None), getattr(scr, 'xp', None)
        if not level or not isinstance(xp, (tuple, list)) or len(xp) < 2 or not xp[1] or getattr(self, 'combat', False):
            return False
        from tft_player.pilot import XP_TO_NEXT
        per = getattr(scr, 'xp_per_buy', None) or 4
        points, lvl, need = xp[0] + n * per, level, xp[1]
        while need and points >= need:
            points -= need
            lvl += 1
            need = XP_TO_NEXT.get(lvl)
        if lvl == level or points == 0:
            return False                          # reaches no level, or exactly
        from tft_player.comp_rules import roll_level, rules
        low, r = roll_level(), rules(getattr(_fixed_comp(), 'name', None))
        if low and lvl >= low and r.get('trait') and roster is not None and tier_steps(roster.board()).get(
                r['trait'], 0) < (r.get('keep') or 0):
            # the comp's roll level without its last step (Riftbeast 7) is never held for interest: level 8 and the
            # roll are the priority (the owner, game 81 x31.3: "it should immediately level to 8 at 4-1")
            return False
        if min(5, (gold - 4 * n) // 10) >= min(5, gold // 10):
            return False                          # no interest step lost
        if self.roll_pending():
            # the plan rolls in this phase: the roll spends the interest anyway, and the comp's roll waits for the level
            # (no reroll below level 8) - held, neither ran (game 81 4-2 to 4-6: level 8 held five phases "an interest
            # step lost" while each plan's roll x9-x20 stayed "not run - no reroll below level 8"; 79 -> 23 gold only
            # at 4-6, HP 49 -> 17)
            return False
        if roster is not None and any(not u.on_board for u, _ in self._lineup(roster)) and                 roster.slots_used() >= int(getattr(scr, 'level', 0) or 0):
            return False                          # a unit of this plan waits for the slot
        if not getattr(self, 'inexact_told', False):
            self.inexact_told = True
            self.notes.append(('level held', f'{n} XP buy(s) held: level {lvl} with {points} XP over and '
                                             f'{gold - 4 * n} gold (an interest step lost) - at the next planning phase '
                                             f'the passive XP makes it closer (the owner rule)'))
        return True

    def _xp(self, gold, scr, roster):
        """The plan's XP buys, after its free rerolls when it has both (_free_first)."""
        return None if self._free_first_pending(scr) else self._level(gold, scr, roster)

    def _free_first_pending(self, scr):
        """Free-first only when XP would starve purchase money or complete no higher level (e57).
        Game 47 3-3 spent 36 -> 4 before nine free shops; game 48 instead had funded level-7 odds.
        Explicit level_first remains supported; the roll executor still checks money and capacity."""
        if self.free_first_done or self.p.get('level_first') or not (getattr(scr, 'free_rolls', None) or 0) > 0:
            return False
        left = int(self.p.get('level_ups') or 0) - self.levels_done
        gold = getattr(scr, 'gold', None)
        xp = getattr(scr, 'xp', None)
        if (isinstance(gold, int) and xp and self._levelling_buys(scr, left) > 0
                and gold - 4 * left >= 5):
            return False                          # game 48: funded level 7 before rolling at its better odds (e57)
        return left > 0 and self.roll_pending()

    def _free_first(self, scr, gold):
        """The plan's roll limited to the free rerolls on hand, before its XP buys: the gold buys what they show,
        the XP takes what is left (_level buys only XP that completes a level); a paid rest of the roll runs after
        the XP, as the plan ordered it."""
        if not self._free_first_pending(scr):
            return None
        fast = self.fast_done
        a = self._rolldown(gold)
        if a is None:
            return None                          # a Wisp question is open: the XP waits for its answer too
        self.free_first_done = True
        self.fast_done = fast                    # a roll-down to a floor: its paid part comes after the XP
        free, times = int(scr.free_rolls), a.arg.get('times')
        a.arg['times'] = min(free, int(times)) if times is not None else free
        a.free_first = True
        xp = int(self.p.get('level_ups') or 0) - self.levels_done
        self.notes.append(('comp rule', f"the plan's {a.arg['times']} free rerolls ran before its {xp} XP buys, with "
                                        f"{gold} gold to buy what they show; the XP got the gold left after them "
                                        f'("level_first": true in execution_plan keeps the XP first)'))
        return a

    def _late_level(self, gold, scr, roster):
        """The XP that opens the slot a unit waits for, in the last LATE_S seconds: it stays first (the board before
        anything else), so a roll n times the free rerolls on hand cover is kept for later instead of rolling with the
        gold the XP leaves (request e52: free rerolls do not expire; game 47 3-3 rolled its 9 at 2 gold).
        Told as 'comp rule' and in the plan's "not run" line."""
        first = self._free_first_pending(scr)
        a = self._level(gold, scr, roster)
        if a is None or not first:
            return a
        self.free_first_done = True
        rn = self.p.get('roll_n')
        left = self.rolls_left if self.rolls_left is not None else int((rn or {}).get('times') or 0)
        if isinstance(rn, dict) and not self.rolln_done and 0 < left <= int(scr.free_rolls):
            text = f"roll x{left}" + (f" for {', '.join(rn.get('buy') or [])}" if rn.get('buy') else '')
            self.rolln_done = True
            self.rolls_dropped = list(getattr(self, 'rolls_dropped', [])) + [text]
            self.notes.append(('comp rule', f"{text}: kept - the XP went first with {scr.timer} s left (it opens the "
                                            f"slot a unit waits for), and these {left} free rerolls are not rolled "
                                            f"with the gold it leaves; they do not expire (send the roll again to run "
                                            f"it)"))
        return a

    def _comp_level_hold(self, n, gold, scr, roster):
        """The comp's level rule (comp_rules.level_hold, request e29; game 41 5-2 bought level 9 with 56 of 66 gold and
        Brambleback 1-star, no roll): the XP buys that would complete level 9 are taken off the plan, their gold stays
        for its roll at level 8; "force_level": true skips it. Told in the next state (self.notes). -> the buys to run."""
        from tft_player.comp_rules import level_hold, rules
        if self.p.get('force_level') or roster is None or scr is None:
            return n
        run, kept = level_hold(getattr(scr, 'level', None), getattr(scr, 'xp', None), n, gold, roster.units,
                               getattr(scr, 'xp_per_buy', None) or 4)
        if not kept:
            return n
        carry = (rules().get('level_hold') or ('the carry',))[0]
        star = '1-star' if any(u.name == carry and not u.missing for u in roster.units) else 'not owned'
        self.p['level_ups'] = self.levels_done + run
        self.notes.append(('comp rule', f'level 9 held ({carry} {star}), {kept} gold kept to roll at level 8 - '
                                        f'{n - run} of the plan XP buys not made ("force_level": true in '
                                        f'execution_plan buys them)'))
        return run

    def _sells_pending(self, roster):
        """The plan's sells not done yet that can run now (a fielded unit is not sold in a fight)."""
        for uid in self.p.get('sell') or []:
            if uid in self.sold:
                continue
            u = self._sell_unit(roster, uid)
            if u is not None and not (getattr(self, 'combat', False) and u.on_board):
                return True
        return False

    @staticmethod
    def _levelling_buys(scr, n):
        """Of n affordable XP buys, the most that end on a completed level (n when the XP bar is not read)."""
        from tft_player.pilot import XP_TO_NEXT
        lvl, xp = getattr(scr, 'level', None), getattr(scr, 'xp', None)
        if not (isinstance(lvl, int) and isinstance(xp, (tuple, list)) and len(xp) == 2
                and all(isinstance(v, int) for v in xp) and xp[1] > 0):
            return n
        per = getattr(scr, 'xp_per_buy', None) or 4
        pts, need, last = xp[0], xp[1], 0
        for b in range(1, n + 1):
            pts += per
            while need and pts >= need:
                pts, lvl, last = pts - need, lvl + 1, b
                need = XP_TO_NEXT.get(lvl)
        return last

    def _board(self, roster, cap, ok):
        accept = ok

        def ok(action):
            unit, destination = action.arg
            if not unit.field_ready(self.stage) or getattr(self, 'unsent_moves', {}).get((unit.id, tuple(destination)), 0) >= 2:
                return None
            return accept(action)
        # a full board, a fielded unit to bench and a bench unit to field: ONE drag of the bench unit onto the
        # fielded unit's hex swaps the two (the user, 2026-10-10: fielding comes first - W then a drag took two
        # actions, and run 20261010-064944 2-3 fielded Lillia with 1 s left). The benched unit lands on the other's
        # old bench slot (a slot the plan names follows below), the fielded unit's own hex follows as a board move.
        # The executor reads who stands on the hex after a swap.
        # (team slots, not units. An Elder Dragon is never swapped in: the game takes him only onto a board with 2
        # slots free - the user, 2026-10-13 - so the unit leaving is benched first and he goes onto an empty hex)
        used = roster.slots_used()
        if any(not u.on_board and u.slots > 1 and used + u.slots > cap for u, _ in self._lineup(roster)):
            a = self._bench_moves(roster, ok, fielded=True)
            if a:
                return a                        # open the arrival's slots before a single-slot swap can consume them
        ins = [(u, h) for u, h in self._lineup(roster) if not u.on_board and u.slots == 1 and used + 1 > cap]
        if ins:
            outs = [u for u, k in self._bench_left(roster) if u.on_board and k not in roster.blocked]
            pairs = [(o, u, h) for o in outs for u, h in ins if used - o.slots + u.slots <= cap]
            if pairs:
                o, u, _ = next(((o, u, h) for o, u, h in pairs if o.loc == ('board', *h)), pairs[0])
                a = None if self._late_hold(roster, u, o) else ok(
                    Action('move', (u, o.loc), f'plan: {u} -> {tuple(o.loc[1:])} (swap with {o})'))
                if a:
                    return a
        # bench slots the plan names (a fielded unit named here leaves the board; an effect can depend on the
        # slot): a unit already in the slot is moved to another free slot first (a drop onto a unit swaps
        # the two, which the screen can't confirm). Fielded units first: they free board slots; a bench unit's
        # slot change waits until the lineup is done (it does not change the fight)
        a = self._bench_moves(roster, ok, fielded=True)
        if a:
            return a
        # lineup: the units "board" lists go to their hexes - only the model's placements. Units it does not
        # list stay where they are (the user, 2026-10-09): it benches a unit with "bench", or places a unit
        # on an occupied hex, which moves the occupant off it
        want = self._lineup(roster)
        for u, hex_ in want:
            if u.loc == ('board', *hex_):
                continue
            occupant = roster.at(('board', *hex_))
            if occupant is None and not u.on_board and roster.slots_used() + u.slots > cap:
                # (not while a board unit the plan sells still stands: its sale, next, frees the slot)
                if u.id not in self.full_told and not self._board_sell_pending(roster):
                    # the game refuses a unit over the team size: said once
                    self.full_told.add(u.id)
                    self.failed.append(f'{u} -> {tuple(hex_)}: the board is full ({roster.slots_used()}/{cap} team '
                                       f'slots{f"; {u.name} needs {u.slots}" if u.slots > 1 else ""}) - '
                                       + (f'bench units ("bench") until {u.slots} slots are free: {u.name} cannot be '
                                          f'swapped in for a fielded unit' if u.slots > 1 else
                                          f'bench a unit ("bench") or place {u.name or "it"} on an occupied hex'))
                continue
            if occupant is not None and occupant is not u:
                # an occupant with a free planned hex of its own goes there first; otherwise ONE drag of the unit onto
                # the occupant swaps the two - the occupant takes the unit's old place (the user, 2026-10-16: click the
                # bench unit and drag it onto the board unit it replaces). The executor reads who stands there after.
                # (It used to bench the occupant first, onto a "free" slot: run 20261016-011010 2-2 dragged Rakan onto
                # the component anvil, and the Coven units were fielded late.)
                dest = next((('board', *h) for o, h in want if o is occupant
                             and roster.at(('board', *h)) is None), None)
                if u.slots > 1 and not u.on_board:
                    # an Elder Dragon is never swapped in (2 team slots): the hex is cleared first
                    if dest is None and roster.free_bench():
                        dest = ('bench', roster.free_bench()[0])
                    if dest is None:
                        continue
                if dest is not None:
                    a = ok(Action('move', (occupant, dest), f'plan: clear {hex_} - {occupant} -> {dest}'))
                    if a:
                        return a
                    if u.slots > 1 and not u.on_board:
                        continue
                if self._late_hold(roster, u, occupant):
                    continue
                a = ok(Action('move', (u, ('board', *hex_)), f'plan: {u} -> {hex_} (swap with {occupant})'))
                if a:
                    return a
                continue
            a = ok(Action('move', (u, ('board', *hex_)), f'plan: {u} -> {hex_}'))
            if a:
                return a
        return self._bench_moves(roster, ok, fielded=False)

    def _late_hold(self, roster, u, occupant):
        """A bench unit swapped in for a fielded one late in the phase, when the plan's moves left need more time than
        is left and this swap alone lowers the board's trait breakpoints: it is held, so the fight is not played on a
        half-made swap (game 44, 2-7: a 5-unit Riftbeast swap answered with 4 s left ran only Cinderling for Xayah and
        took off Elderwood 3; 3-3: Scuttlecrab into Ornn's hex at 7 s left, the 5 other moves never ran - Elderwood 3
        and Defender 2 off, Brawler 2 alone). A swap that keeps or raises them runs. -> True when held (told once)."""
        from tft_player.state import board_traits
        t = self.timer
        if t is None or t > self.LATE_S or u.on_board or occupant is None or getattr(self, 'combat', False):
            return False
        unresolved = []
        left = sum(1 for x, h in self._lineup(roster, unresolved) if x.loc != ('board', *h)) + len(unresolved)
        if left < 2 or t >= self.MOVE_S * left:
            return False
        reached = lambda units: sum(sum(1 for th in CATALOG.traits.get(tr) or () if n >= th)
                                    for tr, n in board_traits(units).items())
        now = roster.board()
        after = [x for x in now if x is not occupant] + [u]
        if reached(after) >= reached(now):
            return False
        if u.id not in self.late_told:
            self.late_told.add(u.id)
            a, b = board_traits(now), board_traits(after)
            off = [f'{tr} {a[tr]} -> {b[tr]}' for tr in sorted(a)
                   if any(b[tr] < th <= a[tr] for th in CATALOG.traits.get(tr) or ())]
            self.failed.append(f'{u} -> {tuple(occupant.loc[1:])} (swap with {occupant}) held: {left} board moves '
                               f'left need ~{round(self.MOVE_S * left)} s and {t} s are left, and this swap alone '
                               f'takes {", ".join(off) or "a trait breakpoint"} off before the rest could run - the '
                               f'board fights as it stands')
        return True

    def _bench_waits(self, roster, u):
        """A fielded unit the plan benches stays on while no unit the plan fields (owned, off the board) takes its
        team slot: benching it would leave the board a unit short (run 20261023-063338 4-2: "bench Xayah, field
        Hecarim", Hecarim never bought - the move failed twice in the phase's last 6 s, and the three components the
        plan gave sat on the item bench through the fight, lost 52 -> 40)."""
        return u.on_board and sum(x.slots for x, _ in self._lineup(roster) if not x.on_board) < u.slots

    def _bench_moves(self, roster, ok, fielded):
        """The plan's "bench" placements of fielded units (fielded=True) or of bench units (False)."""
        for u, k in self._bench_left(roster):
            if u.on_board != fielded:
                continue
            if fielded and self._bench_waits(roster, u):
                if u.id not in self.bench_told:
                    self.bench_told.add(u.id)
                    self.failed.append(f'bench {u}: kept on the board - no unit the plan fields is owned to take its '
                                       f'slot (the board would fight a unit short); it is benched once one is')
                continue
            if k in roster.blocked:
                self.benched.add((u.id, k))
                self.failed.append(f'bench {u} to slot {k}: slot {k} holds {roster.blocked[k]}')
                continue
            occ = roster.at(('bench', k))
            free = [i for i in roster.free_bench() if i != k]
            if (fielded and occ is not None and not free and any(
                    not x.on_board and x.slots > 1 for x, _ in self._lineup(roster))):
                continue                        # a swap would keep the board full and displace the incoming Dragon
            if (occ is not None and occ is not u and (free or not u.on_board)
                    and any(x is occ for x, _ in self._lineup(roster))):
                # the slot holds a unit this plan fields: it leaves the bench once this one leaves the board, so the
                # departure takes a free slot no plan entry names instead of moving the entering unit aside (game 55
                # 6-1: "clear bench slot 1" dragged the bought Elder Dragon b1 -> b5 with 3 s left; Cinderling's and
                # Scuttlecrab's moves were dropped at combat start and Dragon never fought - out the next round)
                if not u.on_board:
                    continue                     # benched already: it takes the slot once the occupant is fielded
                named = {e.get('slot') for e in self.p.get('bench') or []}
                dest = next((i for i in free if i not in named), free[0])
                a = ok(Action('move', (u, ('bench', dest)), f'plan: {u} -> bench slot {dest} (slot {k} holds {occ}, '
                                                            f'which this plan fields)'))
                if a:
                    return a
                continue
            if occ is not None and occ is not u and occ.id in (self.p.get('sell') or []) and occ.id not in self.sold:
                continue                         # the occupant is sold first (the sells run first): no clearing drag
            if occ is not None and occ is not u and free:
                a = ok(Action('move', (occ, ('bench', free[0])), f'plan: clear bench slot {k} - {occ}'))
                if a:
                    return a
            a = ok(Action('move', (u, ('bench', k)), f'plan: {u} -> bench slot {k}'))
            if a:
                return a
        return None

    def _links(self, roster):
        """The plan's links (drag from the source, hover on the target ~1 s, release - the BFF to its rider, Taric
        to his ally), each once, after the board is placed. from: a unit id or a board hex [row, col]; to: a unit."""
        for k, e in enumerate(self.p.get('link') or []):
            if not self.arrival_ready(e, roster, located=True):
                continue
            if ('link', k) in self.linked:
                continue
            self.linked.add(('link', k))
            src, to = e.get('from'), self.unit(roster, e.get('to'))
            if isinstance(src, (list, tuple)) and len(src) == 2 and all(isinstance(v, int) for v in src):
                loc = ('board', src[0], src[1])
            else:
                su = self.unit(roster, src)
                loc = su.loc if su is not None and su.on_board else None
            if loc is None or to is None or not to.on_board or to.loc == loc:
                self.failed.append(f'link {src} -> {e.get("to")}: needs a board unit or hex to drag from and '
                                   f'another unit on the board to link to')
                continue
            return Action('link', (loc, to), f'plan: link {loc[1:]} -> {to}')
        return None

    def _items(self, scr, roster, gives_only=False, transfers_only=False):
        # a disputed holder (unread items) holds only the steps that involve him, never the whole plan (the Reviewer,
        # game 78 S3-H1: an unread Hand of Justice on Murkwolf froze every give of the plan for rounds - 2-5 to 3-2,
        # 3-5 on - Bow and Gloves for a readable Cinderling among them)
        wait = set(self.p.get('_item_reconcile') or [])
        involved = lambda e: bool(wait) and isinstance(e, dict) and (e.get('id') in wait or e.get('from') in wait)
        # first the Removers / Reforgers (they return items to the bench), then the hand-outs. gives_only: in the
        # fight, where a give finds its unit by reading it, but a Remover/Reforger waits for the units to stand still
        if scr.items is None:
            # no item bench reading this tick (a hidden shop, the fight before the tracked bench stood in): nothing is
            # found, nothing is counted towards "not on the item bench" (benchmark game 20261021-013833)
            return None
        gives_of = lambda src: [g for g in self.p.get('items') or [] if 'from' in g and g['from'] == src
                                and (g.get('component'), g.get('id'), g.get('k')) not in self.itemised
                                and (g.get('component'), g.get('id'), g.get('k')) not in self.item_later]
        # a Remover's re-gives come before the next Remover (run 20261030-132522 5-5: three strips in a row, then
        # the six re-gives with no time left - all were tried in the fight and not found)
        regiving = any(gives_of(e.get('id')) for e in self.p.get('strip') or []
                       if ('strip', e.get('id'), e.get('tool')) in self.itemised and e.get('tool') == 'remover')
        for e in ([] if gives_only or regiving else self.p.get('strip') or []):
            key = ('strip', e.get('id'), e.get('tool'))
            if key in self.itemised or key in self.item_later or involved(e):
                continue
            if e.get('after') and tuple(e['after']) not in self.itemised:
                continue                         # the reforge's component goes onto the unit first
            if time.monotonic() < self.item_retry_at.get(key, 0):
                continue
            u = self.unit(roster, e.get('id'))
            want = is_reforger if e.get('tool') == 'reforger' else is_remover
            slot = next((k for k, it in enumerate(scr.items or []) if want(it)), None)
            if not u or not u.items:
                self.itemised.add(key)           # nothing on it to take off
                continue
            regives = gives_of(e.get('id')) if e.get('tool') == 'remover' else []
            steps = max(len(regives), e.get('_delivery_steps', 0))
            strips_left = 1
            started = False
            if e.get('_package') is not None:
                pending = self.itemised | self.item_later
                steps = sum(g.get('_package') == e['_package'] and not g.get('_given')
                            and (g.get('component'), g.get('id'), g.get('k')) not in pending
                            for g in self.p.get('items') or [])
                strips_left = sum(s.get('_package') == e['_package'] and
                                  ('strip', s['id'], s['tool']) not in pending for s in self.p.get('strip') or [])
                started = any(s.get('_package') == e['_package'] and s.get('_stripped')
                              for s in self.p.get('strip') or [])
            need = self.STRIP_S * strips_left + self.REGIVE_S * steps + self.STRIP_MARGIN_S
            if steps and not started and isinstance(self.timer, (int, float)) and self.timer < need:
                # too late for the Remover and its re-gives: the items stay on the unit for the fight rather than
                # on the item bench (5-5: Krug stripped at 9 s fought with none, his three items on the bench)
                self.item_later.add(key)
                for g in regives:
                    self.item_later.add((g.get('component'), g.get('id'), g.get('k')))
                if e.get('_package') is not None:
                    self.item_later |= {('strip', s['id'], s['tool']) for s in self.p.get('strip') or []
                                        if s.get('_package') == e['_package']}
                    self.item_later |= {(g.get('component'), g.get('id'), g.get('k')) for g in self.p.get('items') or []
                                        if g.get('_package') == e['_package']}
                self.failed.append(f'{e.get("tool")} on {u}: not used with {self.timer} s left - it and its '
                                   f'{len(regives)} re-gives need ~{need:.0f} s; its items stay on it for this fight')
                continue
            if slot is None:
                self.item_waits[key] = self.item_waits.get(key, 0) + 1
                if self.item_waits[key] > 30:
                    self.itemised.add(key)
                    self.failed.append(f'{e.get("tool")}: none on the item bench')
                continue
            self.itemised.add(key)
            a = Action('strip', (slot, scr.items[slot], u), f'plan: {e.get("tool")} on {u}')
            a.strip_key = key
            return a
        entries = list(self.p.get('items') or [])
        if gives_only:
            # Game 49 5-1: an optional duplicator's failed lookup consumed the fight
            # before Sett's Belt and Scuttlecrab's Gloves could be equipped.
            from tft_player.items import is_duplicator
            entries.sort(key=lambda e: is_duplicator(e.get('component')))
        for e in entries:
            if e.get('_after_strip') is not None and (
                    'strip', e['_after_strip'], 'remover') not in self.itemised:
                continue
            if transfers_only and 'from' not in e and e.get('_after_strip') is None:
                continue                         # _transfer_first: only a Remover's re-gives
            key = (e.get('component'), e.get('id'), e.get('k'))
            if key in self.itemised or key in self.item_later or involved(e):
                continue
            if time.monotonic() < self.item_retry_at.get(key, 0):
                continue
            u = self.unit(roster, e.get('id'))
            if u is None and isinstance(e.get('id'), str):
                # a champion bought in this plan, named (validate let it through): its copy once it is ours - the
                # fielded one first, the highest star
                u = next((x for x in sorted(roster.units, key=lambda x: (not x.on_board, -x.star))
                          if x.name == e['id'] and not x.missing), None)
                if u is None:
                    self.item_waits[key] = self.item_waits.get(key, 0) + 1
                    if self.item_waits[key] > 80:
                        self.itemised.add(key)
                        self.failed.append(f"item {e.get('component')} -> {e['id']}: {e['id']} was not bought")
                    continue
                e['id'] = u.id                   # from now on an id, as every other entry (refusals, credits)
                key = (e.get('component'), u.id, e.get('k'))
                if key in self.itemised:
                    continue
            if gives_only and (u is None or u.loc[0] != 'board'):
                continue                         # in the fight only fielded units can be found (bench units show no bar)
            if gives_only and u.id in getattr(self, 'fight_unfound', ()):
                # its search already failed in this fight (dead, or not readable): a plan written since (next_round,
                # an update) gives to it again, each a ~3 s search that misses the same way (game 84 6-2: Sentinel
                # three searches after next_round plans, 4-2 Brambleback twice) - given in the next planning phase
                continue
            name = e.get('component')
            if gives_only and str(name or '').startswith('Alpha Mark'):
                continue                        # the game only accepts changing the Alpha in planning
            bench = list(scr.items or [])
            base = lambda n: re.sub(r'\s*\(.*$', '', n or '')
            if 'from' in e and ('strip', e['from'], 'remover') not in self.itemised:
                continue                         # a transfer: its source's items come off first
            slot = next((k for k, it in enumerate(bench) if it and base(it) == base(name)), None)
            if slot is None:                     # its Radiant twin, named either way (items.radiant_twin)
                slot = next((k for k, it in enumerate(bench) if radiant_twin(base(it), base(name))), None)
            if slot is None and u and 'from' not in e and any(base(i) == base(name) for i in u.items):
                # the unit already holds it (given earlier, or a plan repeating its lineup): done
                self.itemised.add(key)
                self.done.append(f'{name} already on {u}')
                continue
            if not u or not name:
                self.itemised.add(key)           # nothing to do for this entry
                self.failed.append(f'item {name} -> unit {e.get("id")}: no such unit or item name')
                continue
            from tft_player.state import item_give_error
            blocked = item_give_error(u, name)
            if blocked:
                self.itemised.add(key)
                self.failed.append(f'item {name} -> {u}: {blocked}')
                continue
            if slot is None:
                # Never substituted (a completed item that is not there is NOT given as its two
                # components: building one is the model's own instruction, one entry per
                # component). An unidentified slot may be the item: it is read now, then retried.
                # The item bench also reads empty for a moment after each drop.
                unread = any(it == 'unknown' for it in bench)
                if unread:
                    self.need_identify = True
                self.item_waits[key] = self.item_waits.get(key, 0) + 1
                if self.item_waits[key] > (80 if unread else 30):
                    self.itemised.add(key)
                    held = ', '.join(('an unidentified item' if i == 'unknown' else str(i)) for i in bench if i)
                    self.failed.append(f'item {name} -> {u}: not on the item bench (it holds: {held or "nothing"})'
                                       + (f' - {name} is a completed item: to build it, give its components '
                                          f'({" + ".join(RECIPES[name])}) as separate entries'
                                          if name in RECIPES else ''))
                continue
            self.itemised.add(key)
            a = Action('item', (slot, name, u), f'plan: {name} -> {u}')
            a.item_key = key
            return a
        return None

    def roll_pending(self):
        """The plan has a roll that has not run (or not finished) yet."""
        roll = self.p.get('roll') or {}
        if isinstance(self.p.get('roll_n'), dict) and not self.rolln_done:
            return True
        return not self.fast_done and (isinstance(self.p.get('fast_rr'), dict) or
                                       isinstance(roll.get('down_to'), (int, float)))

    def all_in(self, hp):
        """The owner's x24.7 rule (game 72): when a loss ends the game, all the gold is rolled for the comp's units, no
        gold floor, selling what the board does not need - first in the planning phase (game 73 4-6: 4 HP, 57 gold; the
        plan's four rolls came last, after a move, an item give and an anvil's choice screen, and were dropped when the
        fight began). -> True while that roll-down is still to run in this planning phase."""
        if getattr(self, 'combat', False) or self.carry or getattr(self, 'all_in_done', False):
            return False
        return last_life(self.stage, hp)

    def _all_in_level(self, scr, gold):
        """The last life's XP first when the gold on hand completes the next level (the owner, game 74b 5-6: "why
        started to waste money on reroll? ... the play should be level to 9, it is 36/68 already ... when you get to 9,
        you have chance to two star elder dragon, and one more unit") - once; then the roll-down of the rest."""
        if getattr(self, 'all_in_levelled', False):
            return None
        level, xp = getattr(scr, 'level', None), getattr(scr, 'xp', None)
        if not level or level >= 10 or not isinstance(xp, (tuple, list)) or len(xp) < 2 or not xp[1]:
            return None
        n = math.ceil(max(0, xp[1] - xp[0]) / (getattr(scr, 'xp_per_buy', None) or 4))
        if n <= 0 or 4 * n > gold:
            return None
        self.all_in_levelled = True
        self.p.update(level_ups=self.levels_done + n, _level_target=level + 1, _all_in_xp=True,
                      level_first=True, force_level=True)
        self.levels_done += n
        return Action('level', n, f'plan: last life (a loss ends the game): level {level + 1} first ({xp[0]}/{xp[1]} XP, '
                                  f'{4 * n} gold) - a slot and the next level odds - then all the gold rolls')

    WISP_PAIRS = 3       # comp units two copies short of their next star that make one level-7 reroll worth it

    def _wisp_reroll(self, scr, roster, gold):
        """One reroll at once, once a planning phase, when the shop shows no Wisp (the owner, game 74b): from stage 6
        or with the comp's final board fielded ("when the comp is finished built or in stage 6, when there is no wisp in
        the shop, the instant action is to reroll once to get wisp, not wait for 10 s for other things"), and at the
        level below the comp's roll level from stage 4 when WISP_PAIRS or more comp units are two copies short of their
        next star (4-2: "should reroll once, to 1. get wisp, 2. a lot of the champions are two copies ... reroll once
        during this situation is the best play") - the one reroll comp_rules.roll_level lets through. -> the roll or
        None."""
        if getattr(self, 'wisp_rerolled', False) or self.carry or gold < 2 or not getattr(scr, 'shop_visible', True):
            return None
        raw, names = list(getattr(scr, 'shop_raw', None) or []), list(getattr(scr, 'shop', None) or [])
        if not any(raw) or any(r and not (names[i] if i < len(names) else None) for i, r in enumerate(raw)):
            return None                          # unread, or a Wisp is there already
        try:
            stage_n = int(str(self.stage).split('-')[0])
        except ValueError:
            return None
        from tft_player.comp_rules import roll_level
        from tft_player import item_plan
        comp = _fixed_comp()
        level, low = getattr(scr, 'level', None), roll_level()
        if low and isinstance(level, int) and level >= low or self.stage in _WISP_REROLLED:
            # from the roll level the executor's Wisp roll is the round's (Player.executor_roll, the owner's x29.7 and
            # x30.14); and one a planning phase, not one a plan (game 79 7-1: each update rolled again - the shop
            # with Taric and Gnar was rolled away after the round's Wisp was bought)
            return None
        fielded = {u.name for u in roster.board()}
        done = comp is not None and bool(getattr(comp, 'units', None)) and set(comp.units) <= fielded
        why = None
        if stage_n >= 6 or done:
            why = 'stage 6' if stage_n >= 6 else 'the final board fielded'
        elif low and level == low - 1 and stage_n >= 4:
            ip = item_plan.plan_for(getattr(comp, 'name', None)) or {}
            copies = Counter((u.name, u.star or 1) for u in roster.units if u.name and not u.missing
                             and (u.star or 1) < 3 and (comp is None or item_plan.comp_unit(u.name, ip)))
            pairs = [n for (n, _), c in copies.items() if c == 2]
            if len(pairs) >= self.WISP_PAIRS:
                why = f"level {level} with {len(pairs)} pairs ({', '.join(sorted(pairs))})"
        if why is None:
            return None
        self.wisp_rerolled = True
        _WISP_REROLLED.add(self.stage)
        buy = [n for n in (getattr(comp, 'units', None) or ()) if CATALOG.unit(n)]
        spec = {'buy': buy, 'targets': {}, 'floor': 0, 'floor_set': False, 'times': 1, 'stop_when': [],
                'wisps': 'ask', 'keep': self.protected_ids(), 'sell_others': False, 'one_roll': True,
                'skip_fives': bool(self.p.get('skip_fives'))}
        return Action('rolldown', spec, f'one reroll for a Wisp (none in the shop; {why}) - the owner rule')

    def _all_in_roll(self):
        """The last life's roll-down (all_in): the plan's rolls in one, to 0 gold - its buy lists (the comp's auto-buy
        units are bought as in every roll), its kept units kept, bench units outside them sold."""
        self.all_in_done, self.all_in_field = True, False
        rolls = [r for r in (self.p.get('roll_n'), self.p.get('fast_rr'), self.p.get('roll')) if isinstance(r, dict)]
        self.rolln_done = self.fast_done = True          # the plan's own rolls are this one
        try:
            stage_n = int(str(self.stage).split('-')[0])
        except ValueError:
            stage_n = 0
        targets = {}
        for r in rolls:
            targets.update(buy_targets(r.get('buy')))
        spec = {'buy': list(dict.fromkeys(n for r in rolls for n in r.get('buy') or [] if CATALOG.unit(n))),
                'targets': targets, 'floor': 0, 'floor_set': True, 'stop_when': [],
                'wisps': 'auto' if stage_n >= 4 else 'first', 'sell_others': True,
                'keep': {k for r in rolls for k in r.get('keep') or [] if isinstance(k, int)} | self.protected_ids(),
                'skip_fives': bool(self.p.get('skip_fives')), 'ahead': getattr(self, 'pre', None) is not None,
                'all_in': True}
        a = Action('rolldown', spec, f"last life (a loss ends the game): roll-down of all the gold for "
                                     f"{', '.join(spec['buy']) or 'the comp units'}, no floor")
        a.all_in = True
        return a

    def _roll_level_stop(self):
        """Why no roll of the plan runs now (the owner, game 74 x25.8: "set it as hard rule, no reroll at level 7" for a
        level-8 comp - comp_rules.roll_level; Player.run_rolldown holds the same line), or None."""
        from tft_player.comp_rules import low_hp_step, roll_level
        low, now = roll_level(), getattr(self, 'level_now', None)
        e = low_hp_step(getattr(self, 'last_hp', None), getattr(self, 'board_now', None))
        if e and now and now >= e['level']:
            return None                          # HP low without Riftbeast 7: the model's rolls at level 7 (owner-hp-paths)
        if low and now and now < low:
            return (f'roll: not run - no reroll below level {low} for this comp, free rerolls included (level {now} '
                    f'now; the comp rule)')
        return None

    def _wisp_gate(self):
        """The Wisps this plan buys before its roll (the owner, game 74 x25.6: "If wisp not bought, the following actions
        (reroll) should not follow" - 3-6's plan bought Search Party and rolled; it came after the fight began, the
        Wisp was gone and 8 rolls ran): 'go' (none, or all bought), 'wait' (still to buy), 'lost' (the fight began, the
        slot shows another card, or the buy failed) -> (state, the Wisps not bought)."""
        bought = getattr(self, 'wisp_bought', set())
        want = [w for w in self.p.get('wisps') or [] if isinstance(w, dict) and w.get('buy')
                and isinstance(w.get('slot'), int) and 0 <= w['slot'] < 5 and w['slot'] not in bought]
        if not want:
            return 'go', []
        names = [str(w.get('name') or f"in shop slot {w['slot']}") for w in want]
        seen = getattr(self, 'shop_seen', None) or [None] * 5
        for w in want:
            card = seen[w['slot']]
            if (getattr(self, 'combat', False) or w['slot'] in getattr(self, 'wisp_failed', set())
                    or card and w.get('name') and not same_wisp(w['name'], card)):
                return 'lost', names
        return 'wait', names

    def _rolldown(self, gold):
        """Roll-downs run as ONE scripted action (player.run_rolldown): D, wait for the cards, read the
        shop only, buy targets, repeat - no full screen read between rolls (run26: 1.9 s per roll; ~0.5 s now)."""
        if getattr(self, 'hold_roll', False):
            return None                          # a Wisp question is open: its answer may change the shop
        if self.roll_pending():
            gate, wisps = self._wisp_gate()
            stop = self._roll_level_stop()
            if stop is None and gate == 'lost':
                stop = (f"roll: not run - it came after the Wisp {', '.join(wisps)} in your plan, which was not bought "
                        f"(a plan's roll follows its Wisp buy, the owner's rule)")
            if stop is not None:
                self.rolln_done = self.fast_done = True
                if stop not in self.failed:
                    self.failed.append(stop)
                return None
            if gate == 'wait':
                return None                      # the plan's Wisp first: its roll comes after the buy
        combat = getattr(self, 'combat', False)
        rn = self.p.get('roll_n')
        if isinstance(rn, dict) and not self.rolln_done:
            # reroll n times; a Wisp in a new shop stops it (the model is asked), then the rest goes on
            left = self.rolls_left if self.rolls_left is not None else int(rn.get('times') or 0)
            if left <= 0:
                self.rolln_done = True
            else:
                self.rolls_left = left
                # the plan's gold_floor stops its paid rerolls (never free rerolls, nor buying a unit rolled into: the
                # executor's floor stops only paid rolling). The climb's game 24 4-2: "roll 10, gold_floor 50 - using
                # the free rerolls while keeping 50" ran a second 10 after the free ones were gone, 61 -> 32
                spec = {'buy': [n for n in rn.get('buy') or [] if CATALOG.unit(n)],
                        'floor': int(self.p.get('keep_gold') or 0), 'floor_set': self.p.get('keep_gold') is not None,
                        'targets': buy_targets(rn.get('buy')),
                        'times': left, 'stop_when': [], 'wisps': False if combat else 'ask',
                        'keep': self.protected_ids(), 'sell_others': bool(rn.get('sell_others')),
                        'skip_fives': bool(self.p.get('skip_fives'))}
                return Action('rolldown', spec, f"plan: roll x{left} for {', '.join(spec['buy']) or 'nothing'}")
        try:
            stage_n = int(str(self.stage).split('-')[0])
        except ValueError:
            stage_n = 0
        # fast roll-down: every Wisp bought on the spot, no question - from stage 4 (the user: the harmful
        # ones do not appear from then on); earlier, only the first Wisp a roll brings up (the user, 2026-10-12:
        # "auto buy the first one" - run 20261012-072956 3-6 rolled past five)
        auto = False if combat else 'auto' if stage_n >= 4 else 'first'
        roll = self.p.get('roll') or {}
        fr = self.p.get('fast_rr')
        if not self.fast_done and (isinstance(fr, dict) or isinstance(roll.get('down_to'), (int, float))):
            self.fast_done = True
            if isinstance(fr, dict):
                spec = {'buy': [n for n in fr.get('buy') or [] if CATALOG.unit(n)],
                        'targets': buy_targets(fr.get('buy')),
                        'floor': int(fr.get('gold_floor') or 0), 'floor_set': fr.get('gold_floor') is not None,
                        'stop_when': fr.get('stop_when') or [],
                        'wisps': auto, 'sell_others': fr.get('sell_others', True) is not False,
                        'keep': {k for k in fr.get('keep') or [] if isinstance(k, int)} | self.protected_ids(),
                        'ahead': getattr(self, 'pre', None) is not None}
            else:
                spec = {'buy': [n for n in roll.get('buy') or [] if CATALOG.unit(n)],
                        'targets': buy_targets(roll.get('buy')),
                        'floor': int(roll['down_to']), 'floor_set': True, 'stop_when': [], 'wisps': auto,
                        'keep': self.protected_ids(), 'sell_others': roll.get('sell_others', True) is not False,
                        'ahead': getattr(self, 'pre', None) is not None}
            spec['skip_fives'] = bool(self.p.get('skip_fives'))
            return Action('rolldown', spec, f"plan: roll-down for {', '.join(spec['buy']) or 'nothing'} "
                                            f"to {spec['floor']} gold")
        return None

    def _container(self):
        """The next choice container (an anvil: no health bar, not a unit) to open - selling opens it."""
        slots = self.p.get('open_container') or []
        for slot in [slots] if isinstance(slots, int) else slots:
            if isinstance(slot, int) and 0 <= slot <= 8 and slot not in self.containers_done:
                self.containers_done.add(slot)
                return Action('container', slot, f'plan: open the container on bench slot {slot}')
        return None
