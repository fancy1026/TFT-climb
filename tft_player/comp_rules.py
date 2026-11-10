"""What the executor keeps for one comp (CLIMB request e20, the comp coach's notes for Riftbeast Brambleback,
agent/memory/comps/<comp>.md): the comp's own trait breakpoint, which a plan's sell, bench or swap may not drop, and the
trait's champions bought on sight.

Riftbeast counts each champion once; its power is the 7 step (the stats grow every 5 s), and the 7 needs Cinderling,
Gromp, Scuttlecrab and Mama Beak beside the final board MetaTFT lists - auto_buy starts from that board only, so game 40
rolled past a Mama Beak at 3-6 (not in the buy list) and the 7th unit came at 4-1."""
import os

from tft_player.catalog import CATALOG

RULES = {
    'Riftbeast Brambleback': {
        'trait': 'Riftbeast',      # the comp's trait: counts distinct champions
        'keep': 7,                 # a plan's sell, bench or swap may not take the board below it ("drop_breakpoint")
        'buy_stars': {1: 2, 2: 2, 3: 1, 4: 1},   # its champions by cost -> auto_buy star: the 1-2 costs paired
        # the comp's carry, fielded at any star in place of one of these when the plan leaves him benched (request
        # e24; "bench_carry": true skips it). Pebbles, Krug, Murkwolf, Sentinel, Taric, Elder Dragon are never swapped
        'carry': ('Brambleback', ('Cinderling', 'Scuttlecrab', 'Gromp')),
        # request e29: at level 8 the XP buys that complete level 9 wait while no copy of the carry is 2-star and they
        # would leave under 20 gold ("force_level": true skips it) - (carry, level, star, gold)
        'level_hold': ('Brambleback', 8, 2, 20),
        # request e32: from this level every roll also buys one copy of each 5-cost on the comp's final board while
        # none is owned ("skip_fives": true skips it) - a roll's list names the 4-costs it rolls for
        'roll_fives': 8,
        'entry_hexes': {'Brambleback': (0, 6), 'Sentinel': (3, 3)},      # (3, 3): the owner's formation, x29.9
        'upgrade_stars': {'Brambleback': 2, 'Sentinel': 2},
        # request e49: the carry's items held by other units move to the fielded carry at the start of a plan's item
        # step ("keep_items": true skips it) - (carry, his items in the order they move) - planner._carry_items_move
        'carry_items': ('Brambleback', ("Guinsoo's Rageblade", 'Edge of Night', 'Hand Of Justice', 'Bloodthirster',
                                        'Quicksilver')),
        # request e50 (a roll does not stop for a Wisp mid-roll, game 45 3-5) is off: a roll replaces the shop, so a
        # Wisp rolled past is lost, not asked later - game 47 5-6, the last round, rolled past Backrow Star and Stand
        # Alone down to 1 gold. A roll late in the phase (Player.WISP_LATE) still rolls on without asking.
        'wisp_after_roll': False,
        # request e55: at level 7+, with 5 or 6 distinct champions of the trait owned and no roll in the plan, the
        # executor rolls after the plan's sells, buys and XP - from 24 gold or more down to 20 (the plan's floor when
        # lower) - buying each one not owned, until 7 are owned ("hold_gold": true skips it) - Player.comp_seventh_action
        'roll_seventh': {'level': 7, 'owned': (5, 6), 'want': 7, 'start': 24, 'floor': 20,
                         'order': ('Mama Beak', 'Krug', 'Sentinel', 'Brambleback', 'Elder Dragon')},
        # the owner, game 72 (x24.1): past Riftbeast 7 another Riftbeast adds nothing - from level 8 the slots beyond
        # the seven go to the units MetaTFT's level-8 and level-9 boards add (level_adds: Taric, Gnar, Sett, by their
        # boards' average place): auto-bought (Player.level_adds_buy), looked for in every roll, and one owned takes
        # an eighth Riftbeast's place (planner._level_eight_extras); a fielded unit on none of the comp's boards (Ashe,
        # x24.2) makes way for one that is (planner._off_comp_out)
        # (the order of those units is MetaTFT's ranking of its level-8 boards, then level 9 - the owner, 2026-11-10:
        # "the level 8 extra units should be determined based on the meta tft level 8 board ranking", no fixed list)
        'eighth': {'level': 8, 'out': ('Cinderling', 'Gromp', 'Mama Beak', 'Scuttlecrab')},
        # the owner, game 78 (x29.9): the board's default positions - the front row from the right: the 3rd hex the
        # main tank (the tank items' holder, mostly Krug), the middle Sentinel, the 5th Scuttlecrab; Brambleback in the
        # back right corner, Pebbles in the back left corner, Cinderling and Gromp on the back row beside Brambleback.
        # 'tank' is the main tank, whoever holds the tank's items (item_plan.holders) - planner._formation
        'formation': {'tank': (3, 4), 'Sentinel': (3, 3), 'Scuttlecrab': (3, 2), 'Brambleback': (0, 6),
                      'Pebbles': (0, 0), 'Cinderling': (0, 5), 'Gromp': (0, 4)},
        # the owner, game 78 (x29.2): no level 7 while the seventh Riftbeast is not owned - the 7th slot then adds
        # little and the XP costs the interest (planner._level_seven_hold)
        'level_hold_seven': 7,
    },
}


HELD_AUGMENTS = []    # the game's augments taken, in order (Player.augments_taken: the same list)


def trait_sharing_augment():
    """An augment held whose bonus grows with each ally sharing a trait (Verticality: "Champions gain ... for each ally
    that shares a trait with them"): every more champion of the comp's trait strengthens all of them - the owner, game
    79 (x30.18): "we have 2 verticality augments, which makes the fielding of the extra 3 star cinderling very strong,
    it is better than a single gnar or taric"; "verticality is the best and strongest combat augment for this comp"."""
    try:
        from tft_player.coach import augment_text
    except Exception:
        return False
    return any('for each ally that shares a trait' in (augment_text(str(a)) or '') for a in HELD_AUGMENTS)


def formation(name=None):
    """The comp's default positions {name or 'tank': (row, col)} (the owner's, x29.9), or {}."""
    return dict(rules(name).get('formation') or {})


def trait_steps(name=None):
    """The comp's trait's steps up to its 'keep' step (Riftbeast: 3, 5, 7) -> tuple, or ()."""
    r = rules(name)
    t, keep = r.get('trait'), r.get('keep')
    info = CATALOG.traits.get(t) if t else None
    steps = tuple(info) if isinstance(info, (tuple, list)) else tuple(getattr(info, 'breakpoints', ()) or ())
    return tuple(s for s in steps if isinstance(s, int) and (not keep or s <= keep))


def rules(name=None):
    """The rules of the comp named, else of the fixed comp (TFT_FORCE_COMP), or {}."""
    name = name or os.environ.get('TFT_FORCE_COMP') or ''
    return RULES.get(name) or {}


# the owner, game 74 (x25.8): "NO reroll at level 7 if the comp is level 8 comp (BOTH brambleback and morgana are 4 cost
# champion, and level 8 comp)" - the level a comp's paid rerolls start at (Player.run_rolldown; the last life's roll-down,
# x24.7, and free rerolls excepted)
ROLL_LEVEL = {'Riftbeast Brambleback': 8, 'Vanguard Morgana': 8}

# the owner, 2026-11-10 on game 88 (x32.8, revised): "let model decides, but let it aware, two paths: 1. HP healthy and
# board lucky, hit 7 riftbeast, could accumulate interest 2. HP not healthy, <50, still not 7 rift, need to get 7 rift
# immediately. it is all decisions based on the HP and gold etc." - the model chooses (comp rule owner-hp-paths). While
# path 2's state holds (HP at or under `hp`, the trait's `step` not active on the board), the no-reroll-below-8 rule and
# the level-7 hold do not stop the model's own XP to `level` and its rolls there (low_hp_step); the executor does neither
LOW_HP_STEP = {'Riftbeast Brambleback': {'hp': 50, 'step': 7, 'level': 7}}


def low_hp_step(hp, board):
    """The committed comp's LOW_HP_STEP while path 2's state holds - HP at or under its 'hp' and the trait's 'step' not
    active on `board` (the fielded units) - else None."""
    e = LOW_HP_STEP.get(os.environ.get('TFT_FORCE_COMP') or '')
    if not e or not isinstance(hp, int) or hp <= 0 or hp > e['hp']:
        return None
    trait = rules(os.environ.get('TFT_FORCE_COMP')).get('trait')
    if not trait:
        return None
    from tft_player.planner import tier_steps
    return None if tier_steps(list(board or ())).get(trait, 0) >= e['step'] else e


def roll_level():
    """The level below which this game's comp rolls no gold: the committed comp's (TFT_FORCE_COMP), else in a flex game
    on a set the lowest of its comps' when each has one - or None."""
    name = os.environ.get('TFT_FORCE_COMP')
    if name:
        return ROLL_LEVEL.get(name)
    from tft_player.coach import flex_set
    fs = flex_set()
    levels = [ROLL_LEVEL.get(c.name) for c in (fs or {}).get('comps') or []]
    return min(levels) if levels and None not in levels else None


_ADDS = {}


def level_adds(name=None, level=8):
    """The units MetaTFT's boards of this level add beyond the comp's trait (the owner, game 72: "in the MetaTFT, it
    shows clearly the win rate and what champions could be added ... at level 8 ... just follow that"), best first by
    the average place of the best board that has them -> [(unit, avg place, the board)]. From the comp's page data
    (data/metatft_shots/<comp>/pages.json, its "Lvl N" tab); [] without it."""
    name = name or os.environ.get('TFT_FORCE_COMP') or ''
    key = (name, level)
    if key not in _ADDS:
        out = {}
        try:
            import json
            from tft_player.metatft_shots import _paths, option_rows
            pages = json.loads((_paths(name)[0] / 'pages.json').read_text(encoding='utf-8'))
            trait = rules(name).get('trait')
            for _, units, n, _ in option_rows(pages.get(f'Lvl {level}')):
                names = [u.rsplit(' ', 1)[0] if u.endswith('*') else u for u in units]
                for u in names:
                    info = CATALOG.unit(u)
                    if info and trait and trait not in info.traits:
                        place = float(n)
                        if u not in out or place < out[u][0]:
                            out[u] = (place, names)
        except (OSError, ValueError, KeyError, ImportError):
            out = {}
        _ADDS[key] = sorted(((u, p, b) for u, (p, b) in out.items()), key=lambda x: x[1])
    return _ADDS[key]


def eighth_units(name=None, level=8):
    """The units for the slots beyond the trait's step from level 8: MetaTFT's level-8 adds, then its level-9 ones (the
    owner, game 72: "4-5, why not buy Sett?" - Sett is on a level-9 board, Sett + Taric 3.33) -> [unit]."""
    out = []
    for lv in (8, 9):
        out += [u for u, _, _ in level_adds(name, lv) if u not in out]
    return out


def trait_buys(name=None):
    """{champion: star} auto_buy starts with for the comp's trait: every champion of it the rules give a star for."""
    r = rules(name)
    if not r.get('trait'):
        return {}
    stars = r.get('buy_stars') or {}
    return {u.name: stars[u.cost] for u in CATALOG.units.values()
            if r['trait'] in u.traits and u.cost in stars}


# the set's frontline traits: an off-comp unit with one of them can fill a board slot for the comp's tanky traits (the
# owner, game 78 x28.4: "field brawler / juggernaut etc. to enable tanky traits"); one with none is never a candidate
TANK_TRAITS = ('Brawler', 'Juggernaut', 'Vanguard', 'Defender')


def off_comp(name, comp_name=None):
    """True for a champion on none of the comp's boards (its units, MetaTFT's early and final boards, the level-8 and
    level-9 adds) - None when there is no fixed comp."""
    from tft_player.coach import forced_comp
    comp = forced_comp()
    if comp is None or not name:
        return None
    want = set(comp.units)
    for section in ('early', 'final'):
        for boards in (comp.guide.get(section) or {}).values():
            for board in boards:
                want.update(board.get('units') or ())
    for lv in (8, 9):
        want.update(eighth_units(comp_name or comp.name, lv))
    return name not in want


def optional_room(roster, name, cap, unit=None):
    """Unclaimed useful slots for the level-add job, using actual champion slots. cap: the team size, or the plan
    check's (level, xp) - game 78 (4-3 to 5-6): the tuple compared with a number crashed the answer thread of every
    plan with a roll, so its strategy, next_round plan and most of its steps were lost. unit: the add being bought -
    an add after it in MetaTFT's ranking (eighth_units) does not claim a slot from it (game 79 7-1: Taric and Gnar sat
    in the shop, Sett and Ivern held the room)."""
    from tft_player.state import board_traits
    if isinstance(cap, (tuple, list)):
        cap = cap[0] if cap else None
    r = rules(name)
    if not cap or cap < (r.get('eighth') or {}).get('level', 99):
        return 0
    adds = set(eighth_units(name, cap))
    board = roster.board()
    if any(not u.name for u in board):
        return 0                                     # (an unverified unit keeps its recorded name: game 79 6-2's Sett)
    # the slots the trait's step still needs are kept for it (the owner, game 81 x31.3: "7 riftbeast must be
    # activated"), the others are the adds' - before, any board short of Riftbeast 7 had no room at all, so at 4-7
    # Taric was never ordered (Riftbeast 6, a free slot, 32 gold). A second copy of a fielded champion adds no trait:
    # its slot counts as free (the owner, x31.4: "it fielded 2 crabs at level 8")
    seen, base = set(), []
    for u in sorted((u for u in board if u.name not in adds), key=lambda u: -(u.star or 1)):
        if u.name not in seen:
            seen.add(u.name)
            base.append(u)
    missing = max(0, r.get('keep', 0) - board_traits(board)[r.get('trait')])
    for u in list(base):
        if (u.name in (r.get('eighth') or {}).get('out', ()) and not u.items
                and board_traits([v for v in base if v is not u])[r['trait']] >= r['keep']):
            base.remove(u)
    room = max(0, cap - sum(u.slots for u in base) - missing)
    claimed = {u.name for u in roster.units if u.name in adds and not u.missing}
    claimed |= {n for n in adds if roster.pending(n)}
    if unit in adds:
        order = eighth_units(name, cap)
        claimed = {n for n in claimed if n == unit or order.index(n) < order.index(unit)}
    return max(0, room - len(claimed))


def _tiers(counts):
    from tft_player.catalog import CATALOG
    return {t: sum(1 for b in CATALOG.traits.get(t) or () if b <= n) for t, n in counts.items()}


def carry_swap(board, owned, name=None):
    """The comp's carry owned and not fielded (request e24, game 40 4-5 to 5-2: Brambleback benched "until he is
    2-star" for five fights while Scuttlecrab 2* with an inactive Juggernaut 1, or Gromp, held the slot): -> (the
    highest-star benched copy, the fielded unit he takes the hex of), else None. The unit that leaves is one the rules
    name (Cinderling, then Scuttlecrab, then Gromp; an itemless one first) whose leaving lowers no trait tier and no
    count of the comp's trait - a champion counts once, so a second copy of the carry is never fielded."""
    r = rules(name)
    carry, order = r.get('carry') or (None, ())
    if not carry or any(u.name == carry for u in board):
        return None
    from tft_player.state import board_traits
    cands = [u for u in owned if u.name == carry and not u.on_board and not u.missing and not u.unverified]
    if not cands:
        return None
    c = max(cands, key=lambda u: u.star or 1)
    now = board_traits(board)
    t0 = _tiers(now)
    # never a paired unit (Taric and his ally hold the pair through moves only on the board: request e194 D)
    out = [u for u in board if u.name in order and not u.missing and not getattr(u, 'paired', None)]
    for v in sorted(out, key=lambda u: (any(u.items or ()), order.index(u.name))):
        after = board_traits([u for u in board if u is not v] + [c])
        if after[r['trait']] < now[r['trait']]:
            continue
        t1 = _tiers(after)
        if all(t1.get(t, 0) >= n for t, n in t0.items()):
            return c, v
    return None


def level_hold(level, xp, n, gold, owned, per=4, name=None):
    """Of a plan's n XP buys, how many run (request e29, game 41 5-2: 14 XP buys at level 8, 12/68, 66 -> 10 gold, with
    Brambleback 1-star and no roll - the level-8 roll bank went on the 9th slot). At the rule's level (8), while no
    copy of the carry is 2-star or better, buys that complete the next level (9) and leave under its gold (20) are
    held: the buys up to the last level completed below it run (none at 8), the gold stays for the plan's roll. Buys
    that complete no level 9 run as ordered. -> (buys that run, gold kept) - kept 0: the rule does not apply.
    xp: (points, needed) read from the bar; unread, the rule does not apply. per: XP a buy gives."""
    from tft_player.pilot import XP_TO_NEXT
    carry, at, star, bank = rules(name).get('level_hold') or (None, 0, 0, 0)
    if not carry or not n or not isinstance(level, int) or level > at or not (
            isinstance(xp, (tuple, list)) and len(xp) == 2 and all(isinstance(v, int) for v in xp) and xp[1] > 0):
        return n, 0
    if any(u.name == carry and (u.star or 1) >= star and not u.missing for u in owned):
        return n, 0
    pts, need, lvl, below, reach = xp[0], xp[1], level, 0, None
    for b in range(1, n + 1):
        pts += per
        while need and pts >= need:
            pts, lvl = pts - need, lvl + 1
            need = XP_TO_NEXT.get(lvl)
            if lvl <= at:
                below = b
            elif reach is None:
                reach = b
    if reach is None or gold - 4 * n >= bank:          # 4 gold a buy, whatever XP a buy gives
        return n, 0
    return below, 4 * (n - below)


def roll_fives(level, units, owned, name=None):
    """The final board's 5-costs a roll buys one copy of (request e32, game 42 4-3: level 8, 20 gold after the roll's
    Sentinel, Gnar in the shop twice and rolled away, the list naming only the 4-costs; game 40 4-2 passed Taric the
    same way). At the rule's level or above: each 5-cost of `units` (the comp's final board) with no copy owned.
    level unread: none. -> [names]"""
    at = rules(name).get('roll_fives')
    if not at or not isinstance(level, int) or level < at:
        return []
    have = {u.name for u in owned if not u.missing}
    return [n for n in units or () if CATALOG.unit(n) and CATALOG.unit(n).cost == 5 and n not in have]


def distinct(units, trait):
    """The distinct champions of `trait` among `units` (a champion counts once; unnamed or missing units do not)."""
    return {u.name for u in units if u.name and not getattr(u, 'missing', False) and CATALOG.unit(u.name)
            and trait in CATALOG.unit(u.name).traits}


def upgrade_keep(roster, targets=None):
    """Copies still needed for named star targets, including the comp's held Sentinel/Dragon pairs (e57).
    Traits count champions; upgrades count copies. Retain the fielded/itemed copies before genuine extras."""
    goals = dict(rules().get('upgrade_stars') or {})
    goals.update(targets or {})
    for name, cap in (rules().get('upgrade_stars') or {}).items():
        goals[name] = min(goals[name], cap)
    if rules() and len(roster.copies('Elder Dragon', 1)) >= 2:
        goals['Elder Dragon'] = max(2, goals.get('Elder Dragon', 1))
    keep = set()
    for name, star in goals.items():
        left = 3 ** (star - 1)
        copies = sorted((u for u in roster.units if u.name == name and not u.missing),
                        key=lambda u: (not u.on_board, -(u.star or 1), not bool(u.items), u.id))
        for u in copies:
            if left <= 0:
                break
            keep.add(u.id)
            left -= 3 ** ((u.star or 1) - 1)
    return keep


def plan_targets(plan):
    """Named star targets across normalized roll forms; used by bench clearing outside the roll too."""
    from tft_player.planner import buy_targets
    out = {}
    for key in ('roll_n', 'roll', 'fast_rr'):
        spec = (plan or {}).get(key) or {}
        if isinstance(spec, dict):
            if isinstance(spec.get('buy'), list):
                out.update({n: max(out.get(n, 1), 2) for n in spec['buy'] if CATALOG.unit(n)})
            for name, star in buy_targets(spec.get('buy')).items():
                out[name] = max(out.get(name, 1), star)
            for c in spec.get('stop_when') or []:
                for name in c.get('any') or []:
                    out[name] = max(out.get(name, 1), int(c.get('star') or 2))
    return out


def trait_count(units, trait):
    """Distinct fielded contributions, including Dragon's two TOTAL Riftbeasts."""
    from tft_player.state import board_traits
    return board_traits(units)[trait]


def seventh_roll(level, gold, owned, floor=None, name=None):
    """The roll for the comp trait's 7th champion (request e55, game 48 4-2 and 4-3: level 7, 42 and 55 gold, 6
    distinct Riftbeasts owned, Mama Beak missing, no roll - "level 8 once the gold covers the XP plus a roll"; game 45
    4-1 and game 40 3-6 the same). At the rule's level or above, with 5 or 6 distinct champions of the trait owned
    (board + bench) and at least its start gold: -> {'buy': the trait's champions not owned, the rule's order first,
    'floor': 20 or the plan's floor when lower, 'have': the distinct count, 'want': 7}, else None. floor: the plan's
    gold_floor (None: not set). level or gold unread: None."""
    r = rules(name)
    cfg, trait = r.get('roll_seventh'), r.get('trait')
    if not cfg or not trait or not isinstance(level, int) or not isinstance(gold, int):
        return None
    have = distinct(owned, trait)
    lo, hi = cfg['owned']
    count = trait_count(owned, trait)
    if level < cfg['level'] or not lo <= count <= hi or gold < cfg['start']:
        return None
    order = list(cfg.get('order') or ())
    rest = sorted((u for u in CATALOG.units.values() if trait in u.traits and u.name not in order),
                  key=lambda u: (u.cost, u.name))
    buy = [n for n in order + [u.name for u in rest] if n not in have and CATALOG.unit(n)]
    low = cfg['floor'] if floor is None else min(cfg['floor'], int(floor))
    return {'buy': buy, 'floor': max(0, low), 'have': count, 'want': cfg['want'], 'trait': trait}
