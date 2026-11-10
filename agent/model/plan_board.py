"""The board a reply's plan would make, rebuilt from the recorded question's state text - for the Judge's checks on
it: "board.units" (the fielded champions), "board.trait.<Trait>" (its count), "board.active_traits" (traits at a
breakpoint of 2 or more), "board.items.<Champion>" (the items it would hold).

The state lists every unit as `[id 25] Gromp ** (2g ranged, Riftbeast/Adaptor) at hex (0,0) items [Guinsoo's
Rageblade]` and the shop as `slot 3: Vi (3g, Primal/Juggernaut)`. The plan's sells, buys, bench and board moves and
item gives and transfers are applied in the controller's order; what a roll would find is not known, so a roll adds
nothing.
"""
import re
import types

UNIT = re.compile(r"^\s*\[id (\d+)\] (.+?) (\*{1,3}) \((\d)g[^,]*, ([^)]*)\) at (?:hex \((\d),(\d)\)|bench slot (\d))"
                  r"(?: items \[(.*?)\]| no items)?", re.M)
SHOP = re.compile(r"^\s*slot (\d): (.+?) \((\d)g, ", re.M)


def parse(text):
    """-> {'units': {id: {...}}, 'shop': {slot: name}} of the state's board, bench and shop."""
    units = {}
    for m in UNIT.finditer(text or ''):
        loc = ('board', int(m.group(6)), int(m.group(7))) if m.group(6) is not None else ('bench', int(m.group(8)))
        units[int(m.group(1))] = {'name': m.group(2).strip(), 'star': len(m.group(3)), 'cost': int(m.group(4)),
                                  'traits': [t.strip() for t in m.group(5).split('/') if t.strip()], 'loc': loc,
                                  'items': [i.strip() for i in (m.group(9) or '').split(',') if i.strip()]}
    shop_part = (text or '').split('\nShop:', 1)[1].split('\n\n', 1)[0] if '\nShop:' in (text or '') else ''
    shop = {int(m.group(1)): m.group(2).strip() for m in SHOP.finditer(shop_part)}
    return {'units': units, 'shop': shop}


def apply(state, plan):
    """The units after the plan (sell, buy, bench, board, items). -> {id or 'new<n>': unit}."""
    units = {k: dict(v, items=list(v['items'])) for k, v in state['units'].items()}
    plan = plan if isinstance(plan, dict) else {}
    for i in plan.get('sell') or []:
        units.pop(i if isinstance(i, int) else int(i) if str(i).isdigit() else i, None)
    new = 0
    for b in plan.get('buy') or []:
        name = state['shop'].get(b) if isinstance(b, int) else str(b)
        if name:
            new += 1
            units[f'new{new}'] = {'name': name, 'star': 1, 'cost': 0, 'traits': [], 'loc': ('bench', None), 'items': []}
    for e in plan.get('bench') or []:
        u = units.get(e.get('unit')) if isinstance(e, dict) else None
        if u:
            u['loc'] = ('bench', e.get('slot'))
    for e in plan.get('board') or []:
        if not isinstance(e, dict) or not isinstance(e.get('hex'), (list, tuple)) or len(e['hex']) != 2:
            continue
        key = e.get('unit') if e.get('unit') in units else next(
            (k for k, u in units.items() if e.get('name') and u['name'] == e['name'] and u['loc'][0] == 'bench'), None)
        if key is None:
            continue
        hexx = ('board', int(e['hex'][0]), int(e['hex'][1]))
        there = next((k for k, u in units.items() if u['loc'] == hexx and k != key), None)
        if there is not None:                            # a unit placed on an occupied hex swaps with it
            units[there]['loc'] = units[key]['loc']
        units[key]['loc'] = hexx
    for it in plan.get('items') or []:
        if not isinstance(it, dict):
            continue
        if it.get('give') and it.get('to') in units:
            units[it['to']]['items'].append(str(it['give']))
        elif it.get('transfer') and it.get('from') in units:
            src = units[it['from']]['items']
            name = str(it['transfer'])
            hit = next((x for x in src if x.lower().startswith(name.lower()[:6])), None)
            if hit:
                src.remove(hit)
                if it.get('to') in units:
                    units[it['to']]['items'].append(hit)
    return units


def view(question, plan):
    """The board after the plan, as the checks read it."""
    from tft_player.catalog import CATALOG
    from tft_player.state import board_traits
    units = [u for u in apply(parse(question), plan).values() if u['loc'][0] == 'board']
    fielded = [types.SimpleNamespace(name=u['name'], items=u['items'],
                                     traits=tuple(CATALOG.unit(u['name']).traits) if CATALOG.unit(u['name'])
                                     else tuple(u['traits'])) for u in units]
    counts = dict(board_traits(fielded))
    active = 0
    for t, n in counts.items():
        if max((b for b in CATALOG.traits.get(t) or () if b <= n), default=0) >= 2:
            active += 1
    items = {}
    for u in units:
        items.setdefault(u['name'], []).extend(u['items'])
    return {'units': [u['name'] for u in units], 'trait': counts, 'active_traits': active, 'items': items}
