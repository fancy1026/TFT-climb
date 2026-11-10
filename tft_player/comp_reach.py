"""How far our games with a comp got - the board an augment is judged on (the owner, 2026-11-10, after game 88, x32.7:
"the metatft showed the ultimate game comp, which assumes everything went right, but if we force it, most of the time
we never reach the level 9 2 star elder dragon final phase, mostly just died during level 8, or sometimes 9 but still
ended with 1 star elder; hence we should not aim for the final comp, but aim for the picking the augment for the level 8
comp, because it is where important"). From the climb's game records (agent/games.jsonl) and each game's fights
(runs/<run>/log.jsonl): the highest level reached, the 5-costs fielded and the units its level-8 board held. Kept in
agent/comps/<comp>/reach.json and counted again when the comp has more games; the preparation warms it."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEVEL = 8                                  # the level whose board an augment is judged on
_CACHE = {}


def _slug(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')


def _games(comp):
    f = ROOT / 'agent' / 'games.jsonl'
    rows = [json.loads(line) for line in f.read_text(encoding='utf-8').splitlines() if line.strip()] if f.exists() else []
    return [r for r in rows if (r.get('comp_played') or r.get('comp')) == comp and r.get('placement')]


def count(comp, rows=None):
    """-> {'games', 'levels': {max level: games}, 'level8_games', 'level8': {unit: games}, 'fives': {unit: [games
    fielded, games 2-star+]}} from the comp's recorded games."""
    from tft_player.catalog import CATALOG
    rows = _games(comp) if rows is None else rows
    levels, at8, fives, n8, n = Counter(), Counter(), {}, 0, 0
    for r in rows:
        f = ROOT / 'runs' / r['run'] / 'log.jsonl'
        if not f.exists():
            continue
        top, boards, five = 0, [], {}
        for line in f.open(encoding='utf-8'):
            if '"ev": "fight"' not in line:
                continue
            try:
                e = json.loads(line)
                lv = int(e.get('level') or 0)
            except (ValueError, TypeError):
                continue
            board = (e.get('roster') or '').split('bench=')[0]
            units = re.findall(r"([A-Z][A-Za-z' ]+?)(\*+)", board)
            top = max(top, lv)
            if lv == LEVEL:
                boards.append({u.strip() for u, _ in units})
            for u, stars in units:
                info = CATALOG.unit(u.strip())
                if info is not None and info.cost == 5:
                    five[u.strip()] = max(five.get(u.strip(), 0), len(stars))
        if not top:
            continue
        n += 1
        levels[top] += 1
        if boards:
            n8 += 1
            held = Counter(u for b in boards for u in b)
            at8.update(u for u, k in held.items() if k >= len(boards) / 2)
        for u, s in five.items():
            got = fives.setdefault(u, [0, 0])
            got[0] += 1
            got[1] += s >= 2
    return {'games': n, 'levels': dict(sorted(levels.items())), 'level8_games': n8, 'level8': dict(at8.most_common()),
            'fives': fives}


def reach(comp):
    """The comp's counts, from agent/comps/<comp>/reach.json while it covers every recorded game, else counted again
    (and kept). -> dict, or None with no games."""
    if not comp:
        return None
    rows = _games(comp)
    if not rows:
        return None
    got = _CACHE.get(comp)
    if got and got.get('records') == len(rows):
        return got
    f = ROOT / 'agent' / 'comps' / _slug(comp) / 'reach.json'
    try:
        got = json.loads(f.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        got = None
    if not got or got.get('records') != len(rows):
        got = dict(count(comp, rows), records=len(rows))
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(got, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        except OSError:
            pass
    _CACHE[comp] = got
    return got


def board_lines(comp, final_units):
    """The augment question's boards: the level-8 board our games fought with (each card is judged on it), how far
    they got, then MetaTFT's final board as the best case. -> text ('' with no games)."""
    r = reach(getattr(comp, 'name', comp))
    if not r or not r.get('games'):
        return ''
    name = getattr(comp, 'name', comp)
    n, n8 = r['games'], r.get('level8_games') or 0
    lv = {int(k): v for k, v in (r.get('levels') or {}).items()}
    low, eight, high = (sum(v for k, v in lv.items() if k < LEVEL), lv.get(LEVEL, 0),
                        sum(v for k, v in lv.items() if k > LEVEL))
    held = r.get('level8') or {}
    board = ', '.join(f'{u} {k}' for u, k in held.items() if k * 2 >= n8)
    rare = ', '.join(f'{u} {k}' for u, k in held.items() if max(2, n8 // 10) <= k and k * 2 < n8)
    at8 = board + (f'; less often: {rare}' if rare else '')
    from tft_player.catalog import CATALOG
    got = r.get('fives') or {}
    fives = '; '.join(f'{u} fielded in {got.get(u, [0, 0])[0]} ({got.get(u, [0, 0])[1]} at 2 stars)'
                      for u in final_units or () if CATALOG.unit(u) is not None and CATALOG.unit(u).cost == 5)
    try:                                       # the executor's own rule on what it fields (planner: "never a board candidate")
        from tft_player.comp_rules import TANK_TRAITS
        fields = (f'From level {LEVEL} the executor fields only the comp\'s units and units with a tank trait '
                  f'({", ".join(TANK_TRAITS)}): a random unit a card gives outside these is not fielded then.\n')
    except ImportError:
        fields = ''
    return (f'The level-{LEVEL} board - judge each card on it (and on your current board): our {n8} games with {name} '
            f'that reached level {LEVEL} fought there with (units, in how many of those games): {at8}.\n'
            f'How far our {n} games with {name} got: highest level {LEVEL - 1} or lower in {low}, {LEVEL} in {eight}, '
            f'{LEVEL + 1} or higher in {high}' + (f'; of the 5-costs of the final board: {fives}' if fives else '')
            + '.\n' + fields)
