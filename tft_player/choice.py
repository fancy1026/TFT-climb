"""Bounded item-choice fallback using the comp's frozen item data (owner e227)."""
import json
import math
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'


@lru_cache(maxsize=1)
def item_data():
    units = json.loads((DATA / 'meta/unit_items.json').read_text(encoding='utf-8'))['units']
    source = sorted((DATA / 'sources').glob('cdragon-tft-*.json'))[-1]
    effects = {}
    for it in json.loads(source.read_text(encoding='utf-8'))['items']:
        api, eff = it.get('apiName', ''), it.get('effects') or {}
        if api.startswith('TFT_Item_') and eff and not any(x in api for x in ('Free', 'Corrupted')):
            effects[it['name']] = eff
    return units, effects


def rank_items(names, comp, planned=None):
    """Return a pick and auditable evidence. Unknown effects never get invented.

    Prefer requested items, then observed items on comp units. For artifacts
    without observations, compare their base stats with the comp's recommended
    builds; this is a deadline tie-break, not a claim about their unique effects.
    """
    units, effects = item_data()
    builds = getattr(comp, 'builds', {}) or {}
    targets = planned or {}
    desired = set((getattr(comp, 'guide', {}) or {}).get('items') or [])
    desired.update(i for rows in builds.values() for row in rows for i in row.get('items', []))
    desired.update(i for items in targets.values() if isinstance(items, list) for i in items)
    scales = {'AD': 10, 'AP': 10, 'AS': 10, 'Armor': 20, 'MagicResist': 20,
              'Health': 200, 'Mana': 15, 'ManaRegen': 2}
    def vector(name):
        return [max(0, effects.get(name, {}).get(k, 0)) / scale for k, scale in scales.items()]
    def similarity(a, b):
        norm = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
        return sum(x*y for x, y in zip(a, b)) / norm if norm else 0
    templates = [vector(i) for i in desired]
    rows = []
    for k, name in enumerate(names):
        if not name:
            continue
        observed = [r for u in getattr(comp, 'units', ()) for r in units.get(u, {}).get('items', [])
                    if r[0] == name and r[2] >= 15]
        fit = max((similarity(vector(name), v) for v in templates), default=0)
        score = (int(name in desired), int(bool(observed)), -min(r[1] for r in observed) if observed else 0, fit)
        rows.append((score, -k, name))
    if not rows:
        return None
    score, neg_k, name = max(rows)
    return (-neg_k, name), {'requested': bool(score[0]), 'observed': bool(score[1]),
                            'avg': -score[2] if score[1] else None, 'base_stat_fit': round(score[3], 4)}
