"""Trait-count analysis for offline reviews, plus factual trait and next-slot helpers.

The historical search in best/score orders boards by trait tiers, then unit value; this is not a measure of board
strength. agent/rounds.py still uses it for review comparisons. Live state and execution no longer search for boards
by tier count (o148/e149, owner 2026-11-03). Bonuses, body comparisons at equal tiers, and XP arithmetic remain.
"""
import itertools
from collections import Counter

from tft_player.catalog import CATALOG

SLOTS = {'Elder Dragon': 2}          # team slots a unit takes


def _traits(name, items=()):
    from tft_player.trait_text import trait_bonus
    u = CATALOG.unit(name)
    t = set(u.traits) if u else set()
    t |= {i[:-len(' Emblem')] for i in items or () if isinstance(i, str) and i.endswith(' Emblem')}
    return t, trait_bonus(name)


def bonuses(fielded):
    """[(trait, count, tier)] of the traits at a breakpoint of 2 or more (tier 1 = its first such breakpoint) for
    [(name, items)], each champion counted once, highest tier first."""
    per = {}
    for n, items in fielded:
        if n:
            per.setdefault(n, [set(), Counter()])
            t, extra = _traits(n, items)
            per[n][0] |= t
            per[n][1] = extra
    counts = Counter()
    for t, extra in per.values():
        counts.update(t)
        counts.update(extra)
    return _tiers(counts)


def _tiers(counts):
    out = []
    for t, n in counts.items():
        tier = sum(1 for b in sorted(CATALOG.traits.get(t) or ()) if b >= 2 and b <= n)
        if tier:
            out.append((t, n, tier))
    return sorted(out, key=lambda x: (-x[2], -x[1]))


def score(fielded):
    """(trait tiers, unit value) of [(name, stars, items)]: the key boards are compared by."""
    return (sum(t for *_, t in bonuses([(n, i) for n, _, i in fielded])), *body_key(fielded))


def body_key(fielded):
    """Stars/cost first; at equal value, a melee defensive-trait body and usable equipped items (e57)."""
    from tft_player.items import is_consumable, is_reusable
    tanks = any(CATALOG.unit(n) and CATALOG.unit(n).melee and
                set(CATALOG.unit(n).traits) & {'Brawler', 'Vanguard', 'Juggernaut', 'Defender'} for n, _, _ in fielded)
    items = sum(1 for _, _, held in fielded for i in held or ()
                if i and i != 'unknown' and not is_consumable(i) and not is_reusable(i))
    return value(fielded), int(tanks), items


def value(fielded):
    seen = {}
    for n, s, _ in fielded:
        u = CATALOG.unit(n)
        if u:
            seen[n] = max(seen.get(n, 0), u.cost * 3 ** (s - 1))
    return sum(seen.values())


def best(owned, team, shop=(), gold=0, limit=12, shop_limit=6, keep=None):
    """The strongest boards: {'owned': (bonuses, [name*stars]), 'with_shop': (bonuses, [name*stars], [bought])} -
    from the owned units [(name, stars, items)], and from them with shop cards [(name, cost)] bought within the gold
    (shown only when it scores higher). Each champion fields once (its highest star). keep (trait, count): the comp's
    kept step (comp_rules: Riftbeast 7) - when the owned units can field it, only boards that keep it are offered
    (request e22: game 40 5-2 offered Riftbeast 5 + Morgana and Master Yi over the fielded Riftbeast 7)."""
    if not team:
        return {}
    best_of = {}
    for n, s, i in owned:
        if n and n != '?' and CATALOG.unit(n) and (n not in best_of or
                body_key([(n, s, i)]) > body_key([(n, *best_of[n])])):
            best_of[n] = (s, list(i or ()))
    val = lambda n, s: CATALOG.unit(n).cost * 3 ** (s - 1)
    kt = keep[0] if keep else None
    # the kept trait's units first: a cut to `limit` never drops the step
    base = sorted(((n, s, i, 0) for n, (s, i) in best_of.items()),
                  key=lambda c: (kt not in CATALOG.unit(c[0]).traits, -val(c[0], c[1])))[:limit]
    extra = sorted({n: (n, 1, [], c) for n, c in shop if n not in best_of and CATALOG.unit(n) and c <= gold}.values(),
                   key=lambda c: -c[3])[:shop_limit]
    info = {c[0]: _traits(c[0], c[2]) for c in base + extra}

    def search(cands, need=0):
        top = None
        for k in range(min(team, len(cands)), 0, -1):
            for combo in itertools.combinations(cands, k):
                if sum(SLOTS.get(c[0], 1) for c in combo) > team or sum(c[3] for c in combo) > gold:
                    continue
                counts = Counter()
                for c in combo:
                    t, more = info[c[0]]
                    counts.update(t)
                    counts.update(more)
                if counts[kt] < need:
                    continue
                b = _tiers(counts)
                key = (sum(t for *_, t in b), *body_key([(c[0], c[1], c[2]) for c in combo]))
                if top is None or key > top[0]:
                    top = (key, b, [c[0] + '*' * c[1] for c in combo], [c[0] for c in combo if c[3]])
            if top:                       # a unit added never lowers a tier: fewer units score less
                break
        return top

    out = {}
    need = keep[1] if keep else 0
    own = search(base, need) if need else None
    if own is None:
        need = 0                          # the owned units cannot field the step: every board counts
        own = search(base)
    if own:
        out['owned'] = (own[1], own[2])
    if extra:
        both = search(base + extra, need)
        if both and both[3] and (not own or both[0] > own[0]):
            out['with_shop'] = (both[1], both[2], both[3])
    return out


def show(b):
    return ', '.join(f"{t} {n}" for t, n, _ in b) or 'no trait bonus'


def interest(gold):
    return min(max(gold, 0) // 10, 5)


def buy_cost(with_shop, tiers_now, shop, gold):
    """What the shop board's buys cost against now (request e30, the comp coach: a 3-cost that adds a tier within the
    same 10-gold interest step is bought - game 41 2-6, Diana: 4 tiers against 3, 24 -> 21 gold, +2 interest kept)."""
    cost = sum(dict(shop).get(n, 0) for n in with_shop[2])
    left = gold - cost
    i0, i1 = interest(gold), interest(left)
    return (f"{sum(t for *_, t in with_shop[0])} trait tiers against {tiers_now} fielded now; the buy costs {cost}g: "
            f"{gold} -> {left} gold, interest +{i0} -> +{i1}"
            f"{' (the same 10-gold step)' if i0 == i1 else ''}")


def change(now, new, fielded, names):
    """What a board changes against the one fielded now (request e36, the comp coach: "when a swap trades one tier
    away count every tier it turns on before calling it even" - game 43 5-3: Xayah in for Kog'Maw made Rapidfire 2 +
    Fae 2 and only lowered Invoker 3 to 2, 9 tiers against 8, and the player called it even)."""
    was = {t: (n, k) for t, n, k in now}
    to = {t: (n, k) for t, n, k in new}
    up = [f"{t} {n}" for t, n, k in new if k > was.get(t, (0, 0))[1]]
    down = [f"{t} {was[t][0]} -> {to[t][0]}" for t, _, k in now if t in to and k > to[t][1]]
    off = [f"{t} {n}" for t, n, _ in now if t not in to]
    have = {n for n, _, _ in fielded}
    plain = [x.rstrip('*') for x in names]
    ins = [n for n in plain if n not in have]
    outs = [n for n in dict.fromkeys(n for n, _, _ in fielded) if n not in plain]
    parts = []
    if ins:
        parts.append(f"fields {', '.join(ins)}")
    if outs:
        parts.append(f"benches {', '.join(outs)}")
    parts.append(f"turns on {', '.join(up)}" if up else "turns on no tier")
    if down:
        parts.append(f"lowers {', '.join(down)}")
    if off:
        parts.append(f"turns off {', '.join(off)}")
    return "against the board fielded now it " + '; '.join(parts)


def next_slot_line(roster, gold, team, level, xp, per=4):
    """Next slot's XP cost, remaining gold/interest and owned comp-trait count; no board ranking.
    None when the level, XP bar or gold is unread, or at level 10."""
    from tft_player.comp_rules import rules
    from tft_player.pilot import XP_TO_NEXT
    if not (isinstance(level, int) and isinstance(team, int) and isinstance(gold, int) and XP_TO_NEXT.get(level)
            and isinstance(xp, (tuple, list)) and len(xp) == 2 and all(isinstance(v, int) for v in xp) and xp[1] > 0):
        return None
    per = per or 4
    cost = 4 * -(-max(0, xp[1] - xp[0]) // per)
    left = gold - cost
    r = rules()
    out = (f"Level {level + 1} ({team + 1} team slots): its XP costs {cost} gold, {gold} -> {left} gold"
           + (f", interest +{interest(gold)} -> +{interest(left)}" if left >= 0 else " (more than you have)") + '.')
    if r.get('trait'):
        from tft_player.state import board_traits
        mine = [u for u in roster.units if u.name and not getattr(u, 'missing', None)]
        names = sorted({u.name for u in mine if r['trait'] in u.traits})
        n = board_traits(mine)[r['trait']]
        out += (f" {r['trait']}: {n} distinct owned ({', '.join(names) or 'none'})"
                + (f", {r['keep'] - n} short of {r['trait']} {r['keep']}." if r.get('keep') and n < r['keep'] else '.'))
    return out
