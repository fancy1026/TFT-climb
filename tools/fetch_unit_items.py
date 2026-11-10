"""MetaTFT's unit + item statistics: for every champion, how its final boards placed with each item on it
(tft-explorer-api, the same filter as tools/fetch_meta.py: Ranked, Platinum+, last 3 days). Final boards, so an item's
number says how it does on that champion by the end of a game, not in stage 3.

    python tools/fetch_unit_items.py          -> data/meta/unit_items.json

{champion: {"avg": its average placement, "games": n,
            "items": [[item, avg placement with it, games, share of the champion's games, avg - champion avg], ...]}}
Items are kept when they are on at least 2% of the champion's games; sorted by the difference (best first).
"""
import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import fetch_meta as F  # noqa: E402

EXPLORER = 'https://api-hc.metatft.com/tft-explorer-api/'
OUT = ROOT / 'data' / 'meta' / 'unit_items.json'
MIN_SHARE = 0.02


def main():
    lookup = F.get(f'https://data.metatft.com/lookups/{F.get(F.COMPS + "latest_cluster_id")["tft_set"]}_latest_en_us.json')
    unit = {}
    for u in lookup['units']:
        for a in [u['apiName'], *(u.get('assetNames') or [])]:
            unit[a] = u['name']
    item = {i['apiName']: i['name'] for i in lookup['items']}
    units = {r['units']: r for r in F.get(EXPLORER + 'units?' + F.FILTER) if r.get('units')}
    # the endpoint returns its 600 most played pairs; filtered to boards with one champion it returns that champion's
    # whole item list (Xayah: 30 items instead of 6) - one request per champion
    pairs = []
    for u in units:
        try:
            pairs += [r for r in F.get(EXPLORER + 'unit_items?' + F.FILTER + f'&unit={u}')
                      if (r.get('unit_items') or '').startswith(u + '&')]
        except Exception as e:
            print('skipped', u, repr(e)[:80])
        time.sleep(0.3)
    out = {}
    for r in pairs:
        u, _, it = (r.get('unit_items') or '').partition('&')
        if u not in units or not it:
            continue
        base = units[u]
        name = unit.get(u, u)
        e = out.setdefault(name, {'avg': round(base['avg_placement'], 3), 'games': base['total_games'], 'items': []})
        share = r['total_games'] / max(1, base['total_games'])
        if share >= MIN_SHARE:
            e['items'].append([item.get(it, it), round(r['avg_placement'], 3), r['total_games'], round(share, 3),
                               round(r['avg_placement'] - base['avg_placement'], 3)])
    for e in out.values():
        e['items'].sort(key=lambda x: x[4])
    OUT.write_text(json.dumps({'source': 'MetaTFT explorer unit_items', 'filter': F.FILTER, 'units': out}, indent=1))
    print(f'{len(pairs)} unit+item rows, {len(out)} champions -> {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
