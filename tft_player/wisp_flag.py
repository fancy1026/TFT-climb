"""The player's standing Wisp flag (request e184, the owner's design x11.2: "buy wisp without model call ... it could
work like a flag, like avoid all combat wisp, or accept all combat wisp, accept all eco wisps").

strategy.wisp_flag (or a plan's / next_round's "wisp_flag") stays until the player changes it, like fight_goal:
    {"combat": "buy"|"skip"|"ask", "economy": "buy"|"skip"|"ask", "buy": [Wisp names], "skip": [Wisp names]}
A named exception wins over its kind. Each Wisp's kind is the "Kind" column of data/reference/wisps.md (combat:
changes our fights; economy: gold, XP, rerolls, the shop, champions, items kept). A Wisp the flag says buy or skip is
bought (when the gold covers it) or left at once, in any roll and at any time left, with no question; "ask" (and no
flag) asks as before."""
from pathlib import Path

KINDS = ('combat', 'economy')
CHOICES = ('buy', 'skip', 'ask')
PATH = Path(__file__).resolve().parent.parent / 'data' / 'reference' / 'wisps.md'
_KINDS = {}


def wisp_kinds(path=PATH):
    """{Wisp name: 'combat' | 'economy'} from the Kind column of wisps.md."""
    key = str(path)
    if key not in _KINDS:
        out, col = {}, None
        rows = Path(path).read_text(encoding='utf-8').splitlines() if Path(path).exists() else []
        for row in rows:
            if not row.startswith('|'):
                continue
            cells = [c.strip() for c in row.strip().strip('|').split('|')]
            if cells[0] == 'Wisp':
                col = cells.index('Kind') if 'Kind' in cells else None
            elif col is not None and len(cells) > col and not cells[0].startswith('---'):
                out[cells[0]] = cells[col]
        _KINDS[key] = out
    return _KINDS[key]


def kind_of(name):
    """A Wisp's kind, or None (not in the table)."""
    return wisp_kinds().get(name) if name else None


def _names(v):
    from tft_player.pilot import wisp_names
    from tft_player.screen import match_name
    known = wisp_names()
    out = []
    for n in v if isinstance(v, list) else [v] if isinstance(v, str) else []:
        m = match_name(str(n), known, cutoff=0.8) if n else None
        if m and m not in out:
            out.append(m)
    return out


def parse(spec, old=None):
    """The flag a reply sets, merged over the one in force (a key it leaves out stays). -> (flag or None, [errors])."""
    if not isinstance(spec, dict):
        return old, ([] if spec is None else [f'wisp_flag: {spec!r} is not an object'])
    flag, errs = dict(old or {}), []
    for k in KINDS:
        if k in spec:
            v = str(spec[k] or '').strip().lower()
            if v in CHOICES:
                flag[k] = v
            else:
                errs.append(f'wisp_flag.{k}: {spec[k]!r} is not buy, skip or ask')
    for k in ('buy', 'skip'):
        if k in spec:
            flag[k] = _names(spec[k])
            bad = [n for n in (spec[k] if isinstance(spec[k], list) else [spec[k]]) if n and not _names(n)]
            if bad:
                errs.append(f'wisp_flag.{k}: no Wisp named {", ".join(map(str, bad))}')
    for k, o in (('buy', 'skip'), ('skip', 'buy')):
        if k in spec and o not in spec:     # a name listed anew under one leaves the other list
            flag[o] = [n for n in flag.get(o) or [] if n not in flag[k]]
    return flag, errs


def decide(flag, name):
    """'buy' | 'skip' by the flag for the Wisp `name`, or None (ask)."""
    if not flag or not name:
        return None
    if name in (flag.get('buy') or ()):
        return 'buy'
    if name in (flag.get('skip') or ()):
        return 'skip'
    v = flag.get(kind_of(name))
    return v if v in ('buy', 'skip') else None


def text(flag):
    """The flag in force, for the state's Wisp line."""
    if not flag:
        return ('Your Wisp flag: none set (a Wisp in the shop is asked; in a roll none is asked - set '
                'strategy.wisp_flag before rolling, or the roll passes the Wisps it brings up)')
    parts = [f"{k} {flag.get(k) or 'ask'}" for k in KINDS]
    for k in ('buy', 'skip'):
        if flag.get(k):
            parts.append(f"always {k}: {', '.join(flag[k])}")
    return 'Your Wisp flag: ' + '; '.join(parts) + ' (in a roll a Wisp the flag leaves at ask is passed, not asked)'
