"""The item plan of a comp with a main carry and a main tank: three items on each, built by the executor (the owner,
2026-11-07: "3 tanks + 3 main carry items must have", "just hard code the rules", "this is a general rule, not just
for this comp: if the comp has a main carry and a main tank, the 6 items should be built like this", and "this
changes the priority of the carousel"; x21-x23: about 6 completed items by the start of stage 5, the carry's first,
then the tank's, in turns - a priority, not a fixed order).

The executor, not the player's plan, decides these items (planner._carry_items_move):
- the plan's gives and moves of components and of the carry's and the tank's items are dropped, except onto the
  comp's extra holders; consumables and other items stay the plan's;
- the carry's and the tank's items held by other units or lying on the item bench go to the fielded carry and tank
  (the carry-items move, one Item Remover use per holder): his best three by his list, of those and of the items the
  components on hand make - an item his stand-in took early (Titan's Resolve) does not keep his Rageblade out;
- components make the carry's or the tank's next item as soon as they can: a bench component that completes the lone
  component the holder has, or two bench components onto a free slot. When both can get one, the carry's comes first
  while he is not ahead of the tank, then the tank's, in turns; an item only one of them can get now is built now.
  Of the items a holder can get, the one first in its list; one that leaves the other holder an item first. A single
  component waits for its partner (after the 4-7 monster round planner.combine_components pairs what is left);
- until the carry or the tank is fielded, its stand-in holds its items (they move when he is fielded);
- the main tank's three cover the frontline's stats, one item each (the owner, game 72: "for frontline it should have
  1. armor and MR, 2. health, 3. durability"): TANK_GROUPS, by the components' stats (Chain Vest armor, Negatron
  Cloak magic resist, Giant's Belt health) and the item's passive;
- an item whose effect does not stack (Sunder, Shred, Burn, Wound: Evenshroud, Last Whisper...) is built once a team
  (the owner, game 72: "why get two Evenshrouds? One is enough, its effect does not stack");
- the carry's and the tank's items beyond their three go on the extra holders, each only an item of its kind (Elder
  Dragon damage items, a 3-star Pebbles AP items - the owner, game 72: "why Crownguard on small blue? It is a tank
  item, not AP"); a tank item has no extra holder, and no component goes on any other unit;
- the rest of their items, which neither can take (three items, or the tank's stat covered), go on another fielded
  unit of the comp (comp_unit) MetaTFT's final boards give that item (fits: data/meta/unit_items.json, the best average place first; a tank
  item only on a melee unit) - the owner, 2026-11-07: "Yes, field them";
- the carousel ranks first what the plan wants next (wanted): for the holder whose turn it is, its items, then the
  components that complete one with a component on hand, then its items' components - with three carry items done,
  tank components first.
Each comp's holders are in PLANS; their items, best first, follow MetaTFT's priority (the owner, 2026-11-07: "it should
also follow the prio of the items on the meta tft"): the comp page's Unit + Items rows for the holder, by share of his
games, then the items of his Full Builds (data/metatft_shots/<comp>/pages.json) - PLANS' own lists come after them,
for the components MetaTFT's items do not use (a tank still gets three). A comp not in PLANS has no item plan."""
from collections import Counter

from tft_player.items import BY_PAIR, COMPONENTS, RECIPES, is_consumable

COMPLETED = set(RECIPES)

# the frontline's three stats, one item each (the owner, game 72), best first
RESIST = ('Gargoyle Stoneplate', 'Evenshroud', 'Adaptive Helm', 'Bramble Vest', "Dragon's Claw")    # armor and MR
HEALTH = ("Warmog's Armor", 'Sunfire Cape', 'Spirit Visage', "Protector's Vow", 'Crownguard')       # health, shields
DURABILITY = ('Steadfast Heart',)
TANK_GROUPS = (('armor and magic resist', RESIST), ('health', HEALTH), ('durability', DURABILITY))
TANK = RESIST + HEALTH + DURABILITY
AP = ('Void Staff', "Nashor's Tooth", 'Spear of Shojin', 'Blue Buff')    # and every non-tank item with a Rod
PLANS = {
    'Riftbeast Brambleback': {
        # MetaTFT: Edge of Night + Rageblade + Quicksilver 30.2%, + Hand of Justice 21.5%, + Titan's Resolve 2.7%
        # (a damage item: Attack Damage and Ability Power stacks, then Damage Amp - the owner, 2026-11-07)
        'carry': ('Brambleback', ("Guinsoo's Rageblade", 'Edge of Night', 'Quicksilver', 'Hand Of Justice',
                                  'Bloodthirster', "Titan's Resolve")),
        'carry_standins': (),                    # the strongest suitable unit holds them (holders; owner x26)
        # Krug with Gargoyle 53.7%, Warmog's 37.3%, Spirit Visage 32.6%; no AP item (owner-ad-comp: no Ionic Spark)
        'tank': ('Krug', TANK),
        'tank_standins': (),
        'extra': {'Elder Dragon': 'damage'},     # MetaTFT: Infinity Edge, Last Whisper, Striker's Flail
        'extra_3star': {'Pebbles': 'ap'},        # owner-ad-comp: an AP item we hold goes on a 3-star Pebbles
        'never': ('Ionic Spark', "Archangel's Staff"),     # owner-ad-comp: no AP item, no AP penetration
        # the third set, once the carry's and the tank's three are done: Sentinel's tank items first - the second batch
        # of tank items - then Elder Dragon's damage items (the owner, game 78 5-1, x28.10: "the priority is to make
        # the second batch of tank items on sentinel"; game 76 5-2: "another set of items for the Sentinel, or elder
        # dragon if we have it")
        'third': (('Sentinel', 'tank'), ('Elder Dragon', 'damage')),
    },
    'Vanguard Morgana': {
        # Morgana with Void Staff 60.1%, Morellonomicon 55.6%; builds + Adaptive Helm / Rabadon's / Gunblade
        'carry': ('Morgana', ('Void Staff', 'Morellonomicon', 'Adaptive Helm', "Rabadon's Deathcap",
                              'Hextech Gunblade')),
        'carry_standins': ('Cassiopeia',),       # the comp notes: Morgana's items parked on the best other caster
        # Sentinel with Gargoyle 53.4%, Protector's Vow 40.6%, Warmog's 28.6%
        'tank': ('Sentinel', TANK),
        'tank_standins': ('Diana', 'Rakan'),     # the comp notes: Rakan/Diana in front in stage 2
        'extra': {'Cassiopeia': 'ap'},
        'extra_3star': {},
    },
}
ROLES = ('carry', 'tank')


_METATFT = {}


def metatft_items(comp, unit):
    """MetaTFT's items for a unit of the comp, best first: its Unit + Items rows by share, then its Full Builds' items
    -> [item] ([] without the comp's page data)."""
    key = (comp, unit)
    if key not in _METATFT:
        out = []
        try:
            import json
            from tft_player.metatft_shots import _flat, _paths, _rows
            pages = json.loads((_paths(comp)[0] / 'pages.json').read_text(encoding='utf-8'))
            rows = _rows(_flat(pages.get('Unit + Items')), 'Unit + Item Play Rate', 2)
            share = lambda n: float(str(n[1]).rstrip('%') or 0)
            out = [im[1] for im, n in sorted(rows, key=lambda r: -share(r[1])) if im[0] == unit and len(im) > 1]
            for im, n in _rows(_flat(pages.get('Full Builds')), 'Unit Build Play Rate', 2):
                if im[0] == unit:
                    out += [x for x in im[1:] if x not in ('Empty Slot', "Thief's Gloves")]
        except (OSError, ValueError, KeyError, ImportError):
            out = []
        _METATFT[key] = list(dict.fromkeys(i for i in out if i in RECIPES or i in COMPLETED))
    return _METATFT[key]


# {champion: [completed items that reached him as their final holder]}: they stay his - one knocked off him (a merge
# past three items, a Remover) goes back first (the owner, game 79 5-5, x30.10: "it knocks the artifact item off, and
# then it is given to cinderling. The remedy is to use remover to replace it back to the bramble"). Player.note_final
HOME = {}


def plan_for(name):
    """The comp's plan, its holders' items in MetaTFT's order (metatft_items) and then PLANS' own, never its banned
    items -> dict or None. The carry's list starts with the items that reached him (HOME)."""
    p = PLANS.get(name or '')
    if not p:
        return None
    ban = set(p.get('never') or ())
    out = dict(p)
    for role in ROLES:
        unit, own = p[role]
        home = [i for i in HOME.get(unit, ()) if i in RECIPES or i not in COMPONENTS] if role == 'carry' else []
        its = list(dict.fromkeys(home + metatft_items(name, unit) + list(own)))
        if role == 'tank' and FIRST_TANK in TANK:
            its = [FIRST_TANK] + [i for i in its if i != FIRST_TANK]
        out[role] = (unit, tuple(i for i in its if i not in ban))
    return out


def never(name):
    """The items the comp never builds (its owner rules) -> set."""
    return set((PLANS.get(name or '') or {}).get('never') or ())


def base(n):
    return str(n or '').split(' (')[0]


def tank_group(item):
    """The frontline stat an item gives the main tank (TANK_GROUPS) -> its name or None."""
    return next((g for g, its in TANK_GROUPS if item in its), None)


# the owner, game 74b (x26.19): "It should make evershround first! (there is armor penetration in this game, ever shroud
# should be the priority)" - the tank's first item, whatever stats his other items give (one a team: nonstacking)
FIRST_TANK = 'Evenshroud'


def open_items(ip, role, held):
    """The role's items it can still take: the carry any of his not held yet (the owner, x26.8: an item obtained -
    an augment's Guinsoo's Rageblade - is not built again); the tank Evenshroud (FIRST_TANK), then only those of a stat
    none of his completed items gives yet."""
    items, have_items = ip[role][1], set(completed(held))
    if role != 'tank':
        return tuple(i for i in items if i not in have_items)
    have = {tank_group(i) for i in have_items}
    return tuple(i for i in items if i not in have_items and (i == FIRST_TANK or tank_group(i) and tank_group(i) not in have))


def kind(item):
    """'tank' (TANK_GROUPS), 'ap' (a Rod, or AP) or 'damage'."""
    if tank_group(item):
        return 'tank'
    return 'ap' if item in AP or 'Needlessly Large Rod' in RECIPES.get(item, ()) else 'damage'


_DESC = {}


def nonstacking(item):
    """Its effect does not stack (its passive Sunders, Shreds, Burns or Wounds): one a team."""
    if item not in _DESC:
        try:
            from tft_player.planner import item_desc
            d = item_desc(item) or ''
        except Exception:
            d = ''
        _DESC[item] = any(w in d for w in ('Sunder', 'Shred', 'Burn', 'Wound', 'does not stack'))
    return _DESC[item]


def support_item(item):
    """A damage item whose hits reduce the target's armor or magic resist (its passive Sunders or Shreds: Last
    Whisper) - a support item: it goes on another attacker beside the carry, not on him (the owner, game 78 3-7,
    x29.6: "last whisper ideally should not be on the brambleback, because it is a support item, it reduces the armor,
    like a support")."""
    if kind(item) == 'tank':
        return False
    try:
        from tft_player.planner import item_desc
        d = item_desc(item) or ''
    except Exception:
        d = ''
    return 'Sunder' in d or 'Shred' in d


def team_items(roster, plan=None):
    """The completed items our units hold, and those the plan's gives make -> Counter."""
    out = Counter(base(i) for u in getattr(roster, 'units', []) if not getattr(u, 'missing', False)
                  for i in u.items or () if i and base(i) in RECIPES)
    if isinstance(plan, dict):
        from tft_player.planner import gives_held
        for _, u, before, after, _ in gives_held(plan, roster):
            out.update(i for i in after if i in RECIPES and after.count(i) > before.count(i))
    return out


def extra_fits(u, item, ip):
    k = (ip.get('extra') or {}).get(u.name)
    if k is None and (u.star or 1) >= 3:
        k = (ip.get('extra_3star') or {}).get(u.name)
    return k is not None and kind(item) == k


_UNIT_ITEMS = {}


def fits(u, item, ip):
    """Another fielded unit takes one of the carry's or the tank's items they have no room for when MetaTFT's final
    boards give it that item (data/meta/unit_items.json: on 2% of its boards or more) -> those boards' average place,
    lower first; None when it does not, and a tank item only on a melee unit, an extra holder only its kind (the owner,
    game 72: "why Crownguard on small blue? It is a tank item, not AP")."""
    if not _UNIT_ITEMS:
        try:
            import json
            from pathlib import Path
            p = Path(__file__).resolve().parent.parent / 'data' / 'meta' / 'unit_items.json'
            _UNIT_ITEMS.update(json.loads(p.read_text(encoding='utf-8'))['units'])
        except (OSError, ValueError, KeyError):
            return None
    from tft_player.catalog import CATALOG
    info = CATALOG.unit(u.name)
    if kind(item) == 'tank' and not (info and info.melee):
        return None
    extra = u.name in (ip.get('extra') or {}) or (u.star or 1) >= 3 and u.name in (ip.get('extra_3star') or {})
    if extra and not extra_fits(u, item, ip):
        return None
    row = next((r for r in _UNIT_ITEMS.get(u.name, {}).get('items', []) if r[0] == item), None)
    return float(row[1]) if row else None


def comp_unit(name, ip):
    """On the comp's boards (as planner._off_comp_out tests it): its final board, its trait's champions, MetaTFT's
    level-8 and level-9 adds, the plan's stand-ins and extra holders -> bool (game 72's Ashe is not)."""
    from tft_player.catalog import CATALOG
    from tft_player.comp_rules import eighth_units, rules
    from tft_player.planner import _fixed_comp
    comp = _fixed_comp()
    if comp is None:
        return True
    trait, info = rules(comp.name).get('trait'), CATALOG.unit(name)
    return (name in set(getattr(comp, 'units', None) or ()) or name in eighth_units(comp.name, 9)
            or bool(trait and info and trait in info.traits)
            or any(name in (ip.get(k) or ()) for k in ('carry_standins', 'tank_standins', 'extra', 'extra_3star')))


def role_items(ip):
    return {it for r in ROLES for it in ip[r][1]}


def _usable(u):
    items = [base(i) for i in getattr(u, 'items', None) or () if i]
    return (getattr(u, 'id', None) is not None and getattr(u, 'name', None) and not getattr(u, 'missing', False)
            and not getattr(u, 'unverified', False) and getattr(u, 'items_read', True)
            and not getattr(u, 'items_check', False)
            and "Thief's Gloves" not in items and 'unknown' not in items)


SUIT = 0.05      # a unit suits a position when MetaTFT's boards give it the position's items this often (a share)
FULL = 0.15      # ... and is one of the position's own kind from this share: such units go first, by strength


def suit_share(name, role, ip):
    """The share of MetaTFT's boards of this champion that hold the position's items (data/meta/unit_items.json)."""
    fits(types_unit(name), '', ip)                     # loads MetaTFT's unit items
    return sum(r[3] for r in (_UNIT_ITEMS.get(name) or {}).get('items') or [] if r[0] in ip[role][1])


def tier(name, role, ip):
    """1: the position's own kind (its unit, its stand-ins, FULL or more of MetaTFT's boards with its items) - 0: a
    unit that only suits it (game 74b 2-3: Guinsoo's Rageblade on Cinderling while no fighter is fielded; 4-6: the
    fielded Brambleback, not the 2-star Cinderling)."""
    if name == ip[role][0] or name in (ip.get(role + '_standins') or ()):
        return 1
    return 1 if suit_share(name, role, ip) >= FULL else 0


def strength(u):
    """The owner's item priority for one position (x26.20, every comp): more stars first, then the higher cost - all
    1-star: 5-cost > 4-cost > ... > 1-cost; a 2-star 4-cost over a 1-star 5-cost."""
    from tft_player.catalog import CATALOG
    info = CATALOG.unit(getattr(u, 'name', None))
    return (getattr(u, 'star', None) or 1, info.cost if info else 0)


def suits(name, role, ip):
    """Whether a champion is one for this position's items (the owner, x26: "the pivot items should be on the
    strongest and most suitable champion on board"): the position's own unit and its stand-ins; else a comp unit
    (comp_unit) whose MetaTFT boards hold the position's items SUIT of the time or more (data/meta/unit_items.json) -
    a tank only melee, a carry not one whose most-played item is a tank item; never an extra holder (Elder Dragon)."""
    if name == ip[role][0] or name in (ip.get(role + '_standins') or ()):
        return True
    if name in (ip.get('extra') or {}) or name in (ip.get('extra_3star') or {}) or not comp_unit(name, ip):
        return False
    from tft_player.catalog import CATALOG
    info = CATALOG.unit(name)
    if role == 'tank' and not (info and info.melee):
        return False
    fits(types_unit(name), '', ip)                     # loads MetaTFT's unit items
    rows = (_UNIT_ITEMS.get(name) or {}).get('items') or []
    if role == 'carry' and rows and kind(max(rows, key=lambda r: r[3])[0]) == 'tank':
        return False
    return suit_share(name, role, ip) >= SUIT


def types_unit(name):
    import types
    return types.SimpleNamespace(name=name, star=1)


def holders(board, ip, bench=(), require_usable=True):
    """{role: the unit that holds the position's items, or None} - the strongest suitable fielded unit: the position's
    own kind first (tier), then strength (stars, then cost), the position's own unit first among equals. The fielded
    comp carry overrides stand-ins; tanks continue to follow strength. One that cannot be read now makes it
    None until it can: the items wait, never a weaker unit (game 74b 5-2: a Bloodthirster went on Murkwolf "until
    Brambleback is fielded" with Brambleback fielded). bench: the benched units, taken when none is fielded (for
    counting what a holder has, not for giving)."""
    out, used, waiting = {}, set(), set()
    for pool in (board, bench):
        live = [u for u in pool or () if getattr(u, 'id', None) is not None and getattr(u, 'name', None)
                and not getattr(u, 'missing', False)]
        for role in ROLES:
            if out.get(role) is not None or role in waiting:
                continue
            cands = [u for u in live if u.id not in used and suits(u.name, role, ip)]
            # the comp's own carry, fielded, holds the carry's items whatever the stand-ins' stars (the owner, game 77:
            # "why titan at the end not on the brambleback?" - a 2-star Mama Beak kept them; game 74b 4-5: "after the
            # bramble is fielded, the next immediate action is to remover the items on the wolf to the bramble")
            own = [u for u in cands if role == 'carry' and pool is board and u.name == ip[role][0]]
            best = max(own or cands, key=lambda u: (tier(u.name, role, ip), strength(u), u.name == ip[role][0],
                                                    len(u.items or ())), default=None)
            out[role] = best if best is not None and (not require_usable or _usable(best)) else None
            if best is not None:
                used.add(best.id)
                if out[role] is None and pool is board:
                    waiting.add(role)                    # the fielded holder cannot be read now: no bench one
    return out


def weaker(u, h, ip, role):
    """Whether the position's items held by u move to its holder h: u is another suitable unit, weaker than h, never
    the comp's carry to a stand-in, nor an extra holder."""
    key = lambda x: (tier(x.name, role, ip), strength(x))
    return (u is not h and u.name != h.name and suits(u.name, role, ip)
            and (role == 'carry' and h.name == ip[role][0]
                 or (role != 'carry' or u.name != ip[role][0]) and key(u) < key(h)))


def extras(board, ip):
    return [u for u in board if _usable(u) and (u.name in (ip.get('extra') or {}) or
                                                (u.name in (ip.get('extra_3star') or {}) and (u.star or 1) >= 3))]


def comp_of(ip):
    """The comp name whose PLANS entry this plan is (for its MetaTFT data) -> name or None."""
    return next((n for n, q in PLANS.items() if q.get('carry') and q['carry'][0] == ip['carry'][0]
                 and q.get('third') == ip.get('third')), None)


def third_holder(board, ip, hs):
    """(unit, kind) of the third set's holder: the first of PLANS' 'third' units fielded and readable, the highest
    star, never the carry's or the tank's holder -> or None."""
    taken = {h.id for h in hs.values() if h is not None}
    for name, k in ip.get('third') or ():
        us = [u for u in board if u.name == name and _usable(u) and u.id not in taken]
        if us:
            return max(us, key=strength), k
    return None


def third_items(ip, unit, k, held, comp_name=None, team=None):
    """The third holder's items, best first: MetaTFT's for him, then every item of his kind ('tank': one a frontline
    stat, as the main tank's; 'damage'); never the comp's banned ones, one a team of a non-stacking effect, none he
    holds."""
    have = set(completed(held))
    stats = {tank_group(i) for i in have}
    ban = set(ip.get('never') or ())
    pool = list(dict.fromkeys((metatft_items(comp_name, unit) if comp_name else [])
                              + list(TANK if k == 'tank' else RECIPES)))
    out = []
    for it in pool:
        if it not in RECIPES or it in have or it in ban or kind(it) != k:
            continue
        if team is not None and nonstacking(it) and team[it] > 0:
            continue
        if k == 'tank' and tank_group(it) in stats:
            continue
        out.append(it)
    return out


def completed(held):
    return [i for i in held if i not in COMPONENTS and not is_consumable(i) and i not in ('Alpha Mark', 'unknown')]


def drop_gives(plan, ip, board):
    """The plan's gives and moves of components and of the carry's and tank's items, except a component or an item of
    its kind onto an extra holder (game 72: the plan's Chain Vest onto Cinderling's Recurve Bow made a second Titan's
    Resolve) -> what was dropped, ['item -> unit id']."""
    ours = role_items(ip)
    extra = {u.id: u for u in extras(board, ip)}
    mine = lambda it, to: (it in COMPONENTS or it in ours) and not (
        to in extra and (it in COMPONENTS or extra_fits(extra[to], it, ip)))
    out, keep = [], []
    for e in plan.get('items') or []:
        if (isinstance(e, dict) and not e.get('_given') and not e.get('use') and not e.get('_rule')
                and 'from' not in e and not is_consumable(e.get('component'))
                and mine(base(e.get('component')), e.get('id'))):
            out.append(f"{base(e.get('component'))} -> unit {e.get('id')}")
        else:
            keep.append(e)
    plan['items'] = keep
    # a transfer takes all items off its unit (one Remover use): when one of them is ours, the unit's other
    # transfers (its put-backs) go too - the carry-items move strips it itself if it has to
    moves = [e for e in plan.get('transfer') or [] if isinstance(e, dict)]
    off = {e.get('from') for e in moves if e.get('to') != e.get('from') and mine(base(e.get('item')), e.get('to'))}
    tr = []
    for e in plan.get('transfer') or []:
        if isinstance(e, dict) and e.get('from') in off:
            out.append(f"{base(e.get('item'))} from unit {e.get('from')} -> unit {e.get('to')}")
        else:
            tr.append(e)
    if 'transfer' in plan:
        plan['transfer'] = tr
    return out


def repackage(plan, roster, item_bench, ip, board):
    """A full primary holder: reserve an entire better package before stripping it.

    All inputs are confirmed physical stocks. This does not remove equipment in
    anticipation of a roll, a future component, or an unread donor.
    """
    from itertools import combinations, product
    from tft_player.items import uses_left, is_remover
    hs = holders(board, ip)
    strips = {e['id'] for e in plan.get('strip') or []}
    uses = sum(uses_left(i) or 1 for i in item_bench if i and is_remover(i)) - sum(
        e.get('tool') == 'remover' for e in plan.get('strip') or [])
    notes = []
    for role in ROLES:
        h = hs[role]
        if (h is None or not h.on_board or getattr(h, 'items_predicted', False) or h.id in strips
                or len(completed([base(i) for i in h.items])) != 3 or uses < 1
                or any(e.get('id') == h.id and not e.get('_given') for e in plan.get('items') or [])):
            continue
        stocks = {None: Counter(base(i) for i in item_bench if i)}
        for e in plan.get('items') or []:
            if not e.get('_given'):
                stocks[None][base(e.get('component'))] -= 1
        donors = [u for u in roster.units if u.id not in strips and _usable(u)
                  and (u is h or u not in hs.values() and weaker(u, h, ip, role))]
        stocks.update({u.id: Counter(base(i) for i in u.items if i) for u in donors})
        total = sum(stocks.values(), Counter())
        rank = {it: k for k, it in enumerate(ip[role][1])}
        old = [base(i) for i in h.items]

        def quality(package):
            order = tuple(sorted(rank.get(i, 999) for i in package))
            return ((-len({tank_group(i) for i in package if tank_group(i)}),
                     -int(FIRST_TANK in package), order) if role == 'tank' else order)

        candidates = []
        team = team_items(roster)
        for it in dict.fromkeys(list(ip[role][1]) + old):
            forms = [(it,)] if total[it] > 0 else []
            recipe = RECIPES.get(it)
            if (recipe and it not in (ip.get('never') or ()) and not (nonstacking(it) and team[it])
                    and all(total[c] >= recipe.count(c) for c in recipe)):
                forms.append(tuple(recipe))
            if forms:
                candidates.append((it, forms))
        packages = sorted((quality([x[0] for x in combo]), combo) for combo in combinations(candidates, 3)
                          if quality([x[0] for x in combo]) < quality(old))
        chosen = None
        for _, combo in packages:
            for forms in product(*(x[1] for x in combo)):
                need = Counter(c for form in forms for c in form)
                if any(total[c] < n for c, n in need.items()):
                    continue
                left = {uid: stock.copy() for uid, stock in stocks.items()}
                used, deliveries = {h.id}, []
                for form in forms:
                    for c in form:
                        sources = [uid for uid, stock in left.items() if stock[c] > 0]
                        src = min(sources, key=lambda uid: (uid != h.id, uid is not None,
                                                           uid not in used, uid or -1))
                        left[src][c] -= 1
                        if src is not None:
                            used.add(src)
                        deliveries.append(dict(component=c, id=h.id, _rule='primary package',
                                               _after_strip=h.id, _package=h.id,
                                               **({'from': src} if src is not None else {})))
                if len(used) <= uses:
                    chosen = used, deliveries, left, [x[0] for x in combo]
                    break
            if chosen:
                break
        if chosen is None:
            continue
        used, deliveries, left, package = chosen
        waiting = [u.id for u in donors if u.id in used and getattr(u, 'package_confirmed', None)
                   != (u.loc, u.name, u.star, tuple(u.items))]
        if waiting:
            plan['_item_reconcile'] = waiting
            return []                           # positive panels before reserving a destructive repackage
        # Other donors keep their unused equipment. The primary holder's rejected
        # items remain on the bench until the primary package is actually equipped.
        for uid in sorted(used - {h.id}):
            deliveries += [dict(component=c, id=uid, **{'from': uid}, _rule='primary package',
                                _after_strip=h.id, _package=h.id)
                           for c, n in left[uid].items() for _ in range(max(0, n))]
        k = max([e.get('k', -1) for e in plan.get('items') or []] + [-1]) + 1
        plan['items'] = list(plan.get('items') or []) + [dict(e, k=k + n) for n, e in enumerate(deliveries)]
        plan['strip'] = list(plan.get('strip') or []) + [dict(id=uid, tool='remover',
            _reset=uid == h.id, _package=h.id, _delivery_steps=len(deliveries))
            for uid in [h.id, *sorted(used - {h.id})]]
        strips |= used
        uses -= len(used)
        notes.append(f"repackage {h.name} (unit {h.id}): predicted {', '.join(package)}")
        return notes                            # at most one full repackage per compilation
    return notes


def build(plan, roster, item_bench, ip, board, gives_held):
    """The carry's and the tank's items from the item bench, by the plan's priority (the module's doc), appended to
    plan['items'] -> ['what was built']. gives_held: planner.gives_held (the holders' items after the plan's gives)."""
    hs = holders(board, ip)
    forecast = {u.id: after for _, u, _, after, _ in gives_held(plan, roster)}
    held = {r: list(forecast.get(h.id, [base(i) for i in h.items or () if i])) if h else None for r, h in hs.items()}
    # the fielded holder's items on other units, which the carry-items move brings him after this (a Remover use each
    # unit; a benched unit is sold): their places in his list, best first
    remover = any(str(i or '').startswith('Item Remover') for i in item_bench)
    moving = {}
    for r, h in hs.items():
        rank = {it: k for k, it in enumerate(ip[r][1]) if h is None or it in open_items(ip, r, held[r])}
        found = sorted((rank[base(i)], base(i)) for u in roster.units if h is not None and weaker(u, h, ip, r)
                       and _usable(u) and (remover or not u.on_board)
                       for i in u.items or () if base(i) in rank) if h is not None else []
        if r == 'tank':                               # one a stat
            found = [x for k, x in enumerate(found) if tank_group(x[1]) not in {tank_group(y[1]) for y in found[:k]}]
        moving[r] = [k for k, _ in found]
    bench = Counter(base(i) for i in item_bench if i and not is_consumable(base(i)) and base(i) != 'Alpha Mark')
    bench.subtract(base(e.get('component')) for e in plan.get('items') or [] if isinstance(e, dict)
                   and 'from' not in e and not e.get('_given') and not e.get('use'))
    team = team_items(roster, plan)
    never_items = set(ip.get('never') or ())

    def options(role, bench):
        """(list index, [items to give], the item it makes) for the holder of `role`, best first."""
        h, s = hs[role], held[role]
        if h is None or len(completed(s)) >= 3:
            return []
        rank = {it: k for k, it in enumerate(ip[role][1]) if it in open_items(ip, role, s)
                and not (nonstacking(it) and team[it] > 0)}
        lone = [c for c in s if c in COMPONENTS]
        slots, mv = 3 - len(s), moving[role]
        better = lambda k: len(mv) < slots or k < mv[slots - 1]     # a new slot only for one of his best three
        out = [(rank[it], [it], it) for it in rank if bench[it] > 0 and len(s) < 3 and better(rank[it])]
        for c in lone:
            out += [(rank[BY_PAIR[tuple(sorted((c, n)))]], [n], BY_PAIR[tuple(sorted((c, n)))])
                    for n in COMPONENTS if bench[n] > 0 and BY_PAIR.get(tuple(sorted((c, n)))) in rank]
            # a lone component on the holder becomes a full item of his kind, after his own list's (the owner, game
            # 74b 5-3: "1 component on krug, why not build the full item, there is a perfect item existing, belt +
            # tear on krug" - Spirit Visage, with his health stat covered)
            for n in COMPONENTS:
                it = BY_PAIR.get(tuple(sorted((c, n))))
                if (bench[n] > 0 and it and it not in rank and it not in completed(s) and it not in never_items
                        and not (nonstacking(it) and team[it] > 0)
                        and (kind(it) == 'tank' if role == 'tank' else it in ip[role][1])):
                    out.append((100 + len(out), [n], it))
        if not lone and len(s) < 3:
            out += [(rank[it], list(RECIPES[it]), it) for it in rank if it in RECIPES and better(rank[it])
                    and all(bench[c] >= list(RECIPES[it]).count(c) for c in RECIPES[it])]
        return sorted(out, key=lambda o: o[0])

    made, gives = [], []
    for _ in range(6):
        done = {r: len(completed(held[r])) + min(len(moving[r]), 3 - len(held[r])) if held[r] is not None else 0
                for r in ROLES}
        order = ROLES if done['carry'] <= done['tank'] else ROLES[::-1]
        pick = None
        for role in order:
            opts = options(role, bench)
            if not opts:
                continue
            other = order[1] if role == order[0] else order[0]
            if options(other, bench):                 # one that leaves the other holder an item, if one does
                left = [o for o in opts if options(other, bench - Counter(o[1]))]
                opts = left or opts
            pick = (role, opts[0])
            break
        if pick is None:
            break
        role, (_, give, item) = pick
        h, s = hs[role], held[role]
        bench.subtract(give)
        if len(give) == 1 and give[0] in COMPONENTS:
            s[s.index(next(c for c in s if c in COMPONENTS and BY_PAIR.get(tuple(sorted((c, give[0])))) == item))] = item
        else:
            s.append(item)
        gives += [{'component': g, 'id': h.id, '_rule': 'item plan'} for g in give]
        team[item] += 1
        made.append(f"{item} on {h.name} (unit {h.id}, the {role}"
                    + ('' if h.name == ip[role][0] else f"'s stand-in until {ip[role][0]} is fielded") + ')')
    # a component another fielded unit holds, with a bench component, makes the next item of the carry or the tank:
    # it is made now on that unit, its final holder the position's holder - the handover moves it (the owner, game 78
    # 3-5, x29.4: "we have a cloak and belt available, should make the evenshroud" - the Belt sat on Murkwolf). The
    # position further behind goes first; one an item a position each pass.
    count = lambda r: len(completed(held[r])) + min(len(moving[r]), 3 - len(held[r])) if held[r] is not None else 0
    elsewhere = Counter()
    for role in sorted(ROLES, key=lambda r: count(r)):
        h = hs[role]
        if count(role) + elsewhere[role] >= 3:
            continue
        open_ = open_items(ip, role, held[role]) if held[role] is not None else ip[role][1]
        rank = {it: k for k, it in enumerate(ip[role][1]) if it in open_ and it not in never_items
                and not (nonstacking(it) and team[it] > 0)}
        given = {g['id'] for g in gives}
        # (the other position's holder too: the component turns into the item in its own slot - the Belt on the carry's
        # stand-in Murkwolf at 3-5 - never the position's own holder, whose components options() completes)
        opts = sorted((rank[it], u.id, c, n, it, u) for u in board if _usable(u) and (h is None or u.id != h.id)
                      and u.id not in given and not (u.name in (ip.get('extra') or {}))
                      for c in [base(i) for i in u.items or () if i] if c in COMPONENTS
                      for n in COMPONENTS if bench[n] > 0
                      for it in [BY_PAIR.get(tuple(sorted((c, n))))] if it in rank)
        if not opts:
            continue
        _, _, c, n, it, u = opts[0]
        final = h.name if h is not None else ip[role][0]
        bench[n] -= 1
        team[it] += 1
        elsewhere[role] += 1
        gives.append({'component': n, 'id': u.id, '_rule': 'item plan', 'final': final})
        made.append(f"{it} on {u.name} (unit {u.id}: its {c} with the item bench's {n}) - the {role}'s item, made now; "
                    f"its final holder {final}")
    # An allocation is a prediction. Extras wait for the primary packages to
    # be equipped on the actual fielded holders, rather than forecast as full.
    primary_equipped = all(h is not None and h.on_board and not getattr(h, 'items_predicted', False)
                           and len(completed(h.items or ())) >= 3 for h in hs.values())
    tank = hs['tank']
    primary_equipped = primary_equipped and {tank_group(base(i)) for i in tank.items or ()} >= {
        g for g, _ in TANK_GROUPS}
    # the carry's and tank's items beyond their three go on an extra holder of their kind
    for it in [i for i in ip['carry'][1] + ip['tank'][1] for _ in range(max(0, bench[i]))]:
        if primary_equipped:
            room = [u for u in extras(board, ip) if extra_fits(u, it, ip) and not (nonstacking(it) and team[it] > 0)
                    and len(forecast.get(u.id, [base(i) for i in u.items or () if i])
                            + [g['component'] for g in gives if g['id'] == u.id]) < 3]
            if room:
                bench[it] -= 1
                gives.append({'component': it, 'id': room[0].id, '_rule': 'item plan'})
                team[it] += 1
                made.append(f'{it} on {room[0].name} (unit {room[0].id}, beyond the carry\'s and the tank\'s three)')
    # the rest of their items, which neither can take (three items, or the tank's stat covered), go on another fielded
    # unit (the owner, 2026-11-07: "Yes, field them"): one MetaTFT's final boards give that item (fits); a second
    # non-stacking item stays on the bench, and so does an item while its holder is not fielded
    full = lambda r: len(completed(held[r])) + min(len(moving[r]), 3 - len(held[r])) >= 3
    mine = {h.id for h in hs.values() if h is not None}
    for it in dict.fromkeys(ip['carry'][1] + ip['tank'][1]):
        roles = [r for r in ROLES if it in ip[r][1]]
        while primary_equipped and bench[it] > 0 and not (nonstacking(it) and team[it] > 0) and all(
                held[r] is not None and (full(r) or it not in open_items(ip, r, held[r])) for r in roles):
            # a unit already holding items first: two items on one champion, not one each on two (the owner, game 77:
            # "why each mama beak and cinderling have 1 item, one champion should have two")
            room = sorted((0 if n else 1, f, u.id, u) for u in board if _usable(u) and u.id not in mine
                          and comp_unit(u.name, ip)
                          for n in [len(forecast.get(u.id, [base(i) for i in u.items or () if i])
                                        + [g['component'] for g in gives if g['id'] == u.id])] if n < 3
                          for f in [fits(u, it, ip)] if f is not None)
            if not room:
                break
            _, f, _, u = room[0]
            bench[it] -= 1
            gives.append({'component': it, 'id': u.id, '_rule': 'item plan'})
            team[it] += 1
            made.append(f"{it} on {u.name} (unit {u.id}: no room on the carry or the tank; MetaTFT's boards with it on "
                        f"{u.name} place {f:.2f} on average)")
    # the third set: with the carry's and the tank's three done, the components build the third holder's items
    th = third_holder(board, ip, hs) if primary_equipped else None
    if th is not None:
        u, k = th
        s = list(forecast.get(u.id, [base(i) for i in u.items or () if i])) + [g['component'] for g in gives
                                                                               if g['id'] == u.id]
        for _ in range(3):
            if len(completed(s)) >= 3:
                break
            ranked = third_items(ip, u.name, k, s, comp_of(ip), team)
            rank = {it: n for n, it in enumerate(ranked)}
            lone = [c for c in s if c in COMPONENTS]
            opts = [(rank[BY_PAIR[tuple(sorted((c, n)))]], [n], BY_PAIR[tuple(sorted((c, n)))], c)
                    for c in lone for n in COMPONENTS if bench[n] > 0 and BY_PAIR.get(tuple(sorted((c, n)))) in rank]
            if len(s) < 3:
                opts += [(rank[it], list(RECIPES[it]), it, None) for it in ranked if it in RECIPES
                         and all(bench[c] >= list(RECIPES[it]).count(c) for c in RECIPES[it])]
            if not opts:
                break
            _, give, it, c = min(opts, key=lambda o: o[0])
            bench.subtract(give)
            if c is not None:
                s[s.index(c)] = it
            else:
                s.append(it)
            team[it] += 1
            gives += [{'component': g, 'id': u.id, '_rule': 'item plan'} for g in give]
            made.append(f"{it} on {u.name} (unit {u.id}, the third set: the carry's and the tank's three are done)")
    if gives:
        items = [e for e in plan.get('items') or [] if isinstance(e, dict)]
        k = max([e.get('k', -1) for e in items if isinstance(e.get('k'), int)] + [-1]) + 1
        plan['items'] = list(plan.get('items') or []) + [dict(g, k=k + n) for n, g in enumerate(gives)]
    return made


def role_items_held(roster, ip, hs):
    """{role: [(item, unit)]}: the role's completed items on its holder, then those of its list on other units (they
    move to the fielded holder - the carry-items move), three at most."""
    out = {}
    for role in ROLES:
        h = hs[role]
        mine = [(i, h) for i in completed([base(i) for i in h.items or () if i])] if h is not None else []
        others = [] if h is None or h.name != ip[role][0] else [
            (base(i), u) for u in roster.units if u is not h and not getattr(u, 'missing', False)
            and u.name != ip[role][0] and u not in hs.values() for i in u.items or () if base(i) in ip[role][1]]
        out[role] = (mine + others)[:3]
    return out


def wanted(roster, item_bench, name):
    """What the plan wants next (the carousel's first ranks): for the holder whose turn it is, then the other, its
    items, the components that complete one with a component on hand (its lone one or the item bench's), then its
    items' components -> [(item or component, the item it makes or None)]; [] when both have their three or the comp
    has no plan."""
    ip = plan_for(name)
    if not ip or roster is None:
        return []
    hs = holders(roster.board(), ip, roster.bench())
    held = {r: [base(i) for i in h.items or () if i] if h else [] for r, h in hs.items()}
    have = role_items_held(roster, ip, hs)
    done = {r: len(have[r]) for r in ROLES}
    order = ROLES if done['carry'] <= done['tank'] else ROLES[::-1]
    bench = [base(i) for i in item_bench or () if i and base(i) in COMPONENTS]
    # an item obtained is not wanted again (the owner, game 74b 2-4: "there is 0 reason to get another rod, we already
    # have the guinsoo" - the augment's Rageblade on the item bench, no holder fielded yet)
    obtained = [base(i) for i in item_bench or () if i and base(i) in COMPLETED]
    out = []
    for role in order:
        if done[role] >= 3 and hs[role] is not None and hs[role].name == ip[role][0]:
            continue
        its = [i for i in open_items(ip, role, held[role] + [i for i, _ in have[role]] + obtained) if i in RECIPES]
        on_hand = set(bench) | {c for c in held[role] if c in COMPONENTS}
        out += [(i, None) for i in its]
        out += [(c, it) for it in its for k, c in enumerate(RECIPES[it]) if RECIPES[it][1 - k] in on_hand]
        out += [(c, it) for it in its for c in RECIPES[it]]
    seen, first = set(), []
    for n, it in out:
        if n not in seen:
            seen.add(n)
            first.append((n, it))
    return first


def reserved(roster, item_bench, name):
    """The components kept for the carry's and the tank's missing items (planner.combine_components does not pair
    them away after the 4-7 monster round): per missing item, one bench component of that holder's items, the one in
    most of them -> Counter. Run 20261107-023517 5-1: B.F. Sword and the Cloak made Murkwolf a Bloodthirster while Krug
    lacked his third; the second Cloak came in the same round (Dragon's Claw)."""
    ip = plan_for(name)
    if not ip or roster is None:
        return Counter()
    hs = holders(roster.board(), ip, roster.bench())
    have = role_items_held(roster, ip, hs)
    bench = Counter(base(i) for i in item_bench or () if i and base(i) in COMPONENTS)
    keep = Counter()
    for role in ROLES:
        held = [base(i) for i in hs[role].items or () if i] + [i for i, _ in have[role]] if hs[role] else []
        uses = Counter(c for it in open_items(ip, role, held) if it in RECIPES and it not in never(name)
                       for c in set(RECIPES[it]))
        for _ in range(3 - len(have[role])):
            left = [c for c in bench if bench[c] - keep[c] > 0 and uses[c]]
            if not left:
                break
            keep[max(left, key=lambda c: uses[c])] += 1
    return keep


def first(want, ranked, comp_units=(), live=False):
    """The carousel's ranking: a champion of the comp the model ranks (Elder Dragon) first - the owner, game 74b 4-4:
    "the priority should be elder dragon, even it is picked up by another player, but it is the priority if we have the
    chance" (the plan's item wants had put Chain Vest before him) - then the plan's wants, then the rest of the model's
    ranking."""
    from tft_player.catalog import CATALOG
    champs = [r for r in ranked or [] if CATALOG.unit(r) and r in set(comp_units or ())]
    names = [n for n, _ in want if n not in champs]
    if live:
        # The offer-specific answer has considered what is actually selectable.
        # Keep its order; the item plan fills omissions instead of silently
        # replacing that answer with the earlier inventory's component priority.
        return champs + [r for r in ranked or [] if r not in champs] + [n for n in names if n not in ranked]
    return champs + names + [r for r in ranked or [] if r not in names and r not in champs]


def state_lines(roster, item_bench, name):
    """The item plan in the state's item section: the main tank's and the main carry's items, the components on hand,
    and what the carousel takes first."""
    ip = plan_for(name)
    if not ip or roster is None:
        return []
    hs = holders(roster.board(), ip, roster.bench(), require_usable=False)
    have = role_items_held(roster, ip, hs)
    out = [f"ITEM PLAN (the executor builds it): 3 items on the main tank, 3 on the main carry, the carry's next item "
           f"first when the components make both, an item the components make built at once; beyond those six: "
           + ', '.join([f'{n} ({k} items)' for n, k in (ip.get('extra') or {}).items()]
                       + [f'a 3-star {n} ({k} items)' for n, k in (ip.get('extra_3star') or {}).items()])
           + ", then another fielded unit of the comp MetaTFT's boards give the item (a tank item only on a melee unit)"
           + "; the main tank's three: one armor and magic resist, one health, one durability item."]
    for role, label in (('tank', 'main tank'), ('carry', 'main carry')):
        unit = ip[role][0]
        h = hs[role]
        where = (f" (actual holder {h.name}, unit {h.id}, {'fielded' if h.on_board else 'benched'})"
                 if h is not None else ' (no suitable holder located)')
        confirmed = h is not None and _usable(h) and not getattr(h, 'items_predicted', False)
        held = [base(i) for i in h.items or () if i] if h is not None else []
        equipment = (f"{', '.join(held) or 'none'} - {len(completed(held))} of 3 completed"
                     if confirmed else 'unconfirmed')
        if h is not None and getattr(h, 'items_predicted', False):
            equipment += f"; predicted: {', '.join(held) or 'none'}"
        elsewhere = [f'{i} on {u.name}, unit {u.id}' for i, u in have[role] if u is not h
                     and _usable(u) and not getattr(u, 'items_predicted', False)]
        stats = ''
        if role == 'tank' and confirmed:
            got = {tank_group(i): i for i in held if tank_group(i)}
            stats = ' (' + '; '.join(f'{g}: {got.get(g) or "none yet"}' for g, _ in TANK_GROUPS) + ')'
        out.append(f"- {label} {unit}{where}: equipped: {equipment}{stats}"
                   + (f"; available elsewhere: {'; '.join(elsewhere)}" if elsewhere else ''))
    comps = [base(i) for i in item_bench or () if i and base(i) in COMPONENTS]
    lone = [(base(i), u) for u in roster.units if not getattr(u, 'missing', False) for i in u.items or ()
            if i and base(i) in COMPONENTS]
    out.append('- components: ' + (', '.join([f'{c} (item bench)' for c in comps]
                                            + [f'{c} (on {u.name}, unit {u.id})' for c, u in lone]) or 'none'))
    want = wanted(roster, item_bench, name)
    if want:
        done = [n for n, it in want if it is None]
        comps = [f'{n} ({it})' for n, it in want if it]
        out.append('- the carousel takes first, in this order: completed ' + ', '.join(done[:5])
                   + (' ...' if len(done) > 5 else '') + '; components ' + ', '.join(comps))
    return out
