"""This Set's emblems: what makes each one (a Spatula or a Frying Pan + a component; some come only from augments,
orbs and encounters) and which ones add a trait level to a board - for the augment question and the planning state.

The owner (2026-10-23, agent/history/owner-notes.md): augments work together - U.R.F (holders of Spatula and Frying
Pan items +20% Attack Speed and 3 Mana Regen) with Branching Out and every augment that gives emblems. In g1-g21 an
emblem card was taken once in 18 offers ("random emblems seldom fit this comp"; g7 4-2: U.R.F "needs a second
component and only helps one unit"), and Spatulas or emblems sat on the item bench all game in g10, g13 and g18:
neither the reference files nor the state said what a Spatula makes or which emblem adds a trait level.
Data: data/emblems-<set>.json (tools/build_reference.py, from the game data)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tft_player.catalog import CATALOG

DATA = Path(__file__).resolve().parent.parent / 'data'
MAKERS = ('Spatula', 'Frying Pan')
EMBLEMS: dict = {}        # 'Lunar Emblem' -> ('Spatula', 'Tear of the Goddess'), or None (augments, orbs and encounters)
BY_PAIR: dict = {}        # ('Spatula', 'Tear of the Goddess') sorted -> 'Lunar Emblem'


def reload():
    """Read data/emblems-<set>.json (the newest one) into EMBLEMS and BY_PAIR."""
    EMBLEMS.clear()
    BY_PAIR.clear()
    files = sorted(DATA.glob('emblems-*.json'))
    try:
        d = json.loads(files[-1].read_text(encoding='utf-8')) if files else {}
    except (OSError, ValueError):
        d = {}
    for name, recipe in (d.get('emblems') or {}).items():
        EMBLEMS[name] = tuple(recipe) if recipe else None
        if recipe:
            BY_PAIR[tuple(sorted(recipe))] = name


reload()


def trait_of(item):
    """'Lunar Emblem' -> 'Lunar'; None for anything else."""
    item = str(item or '')
    return item[:-len(' Emblem')] if item.endswith(' Emblem') else None


def is_emblem_item(item):
    """An emblem, or a Spatula / Frying Pan that makes one (U.R.F and Cooking Pot reward their holders)."""
    return item in MAKERS or trait_of(item) is not None


def recipe(emblem):
    r = EMBLEMS.get(emblem)
    return ' + '.join(r) if r else 'no recipe'


def steps(units):
    """[(emblems needed, trait, units with it, the trait level it reaches, emblem)] for the traits that one or two
    emblems raise to their next level on a board of these units (each name counted once; a trait no unit has counts
    only when one emblem turns it on), fewest first."""
    counts = Counter(t for u in dict.fromkeys(units or ()) for t in (CATALOG.unit(u).traits if CATALOG.unit(u) else ()))
    out = []
    for e in EMBLEMS:
        t = trait_of(e)
        n = counts.get(t, 0)
        nxt = next((b for b in CATALOG.traits.get(t) or () if b > n), None)
        if nxt is not None and (nxt - n == 1 or (nxt - n == 2 and n)):
            out.append((nxt - n, t, n, nxt, e))
    return sorted(out, key=lambda s: (s[0], EMBLEMS.get(s[4]) is None, -s[2], s[1]))     # craftable ones first


def board_line(units, whose='your target board'):
    """One line: which emblems add a trait level to that board and how each is made; '' without units or data."""
    st = steps(units) if units else []
    if not st:
        return ''
    part = lambda k: ', '.join(f'{t} {n} -> {b} ({recipe(e)})' for need, t, n, b, e in st if need == k)
    one, two = part(1), part(2)
    # (a replay of g9's 2-1 question read "augments, orbs and encounters only" as: the craftable ones never come from a
    # random-emblem augment - g17's Branching Out+ gave a Ravager Emblem, g13's and g18's stage-1 emblems were craftable)
    return (f'Emblems for {whose} (an emblem gives its holder the trait - a holder without it; a Spatula or Frying '
            f'Pan + a component makes one, augments and orbs give random ones of any trait): '
            + (f'one adds a trait level: {one}' if one else '')
            + ('; ' if one and two else '') + (f'two: {two}' if two else '') + '. Any other emblem adds no trait '
            'level there: a Reforger on its holder re-rolls it (with all that unit\'s items).')
