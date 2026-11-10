"""Top players' games: recent Ranked matches of the Challenger and Grandmaster ladder, read through the League
client (its match history shows every player's final board: units with stars and items, traits, level, gold left,
place, last round). What the agent studies to see how the best players play.

    python agent/top_games.py fetch [--players 40] [--matches 10]   new matches -> agent/history/top/<matchId>.json
    python agent/top_games.py summary [--units Morgana,Sentinel] [--out file.md]
                                                                    boards with those units, by place: levels, gold
                                                                    left, stars, items per unit, traits; and all boards
Lobbies of Challenger players are Master+ players throughout, so every participant of a fetched match counts.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = ROOT / 'agent' / 'history' / 'top'
sys.path.insert(0, str(ROOT))


def _norm(s):
    return re.sub(r'[^a-z]', '', s.lower())


def unit_name(cid, names={}):
    """'DA_18_Morgana' / 'DA_Taric18' / 'DA_KogMaw18_AD' -> the catalog's champion name."""
    if not names:
        from tft_player.catalog import CATALOG
        names.update({_norm(n): n for n in CATALOG.units})
    base = re.sub(r'^(DA_|TFT\d*_)', '', cid or '')
    base = re.sub(r'(^18_|18$|18_|_A[PD]$)', '', base)
    return names.get(_norm(base), base)


def item_name(iid):
    """'DA_RabadonsDeathcap' -> 'Rabadons Deathcap'; 'DA_18_EmblemHunter' -> 'Hunter Emblem'."""
    s = re.sub(r'^(DA_|TFT\d*_Item_|TFT_Item_)', '', iid or '')
    s = re.sub(r'^18_', '', s)
    m = re.match(r'Emblem(\w+)', s)
    if m:
        return f'{m.group(1)} Emblem'
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)


def trait_name(tid):
    s = re.sub(r'^(DA_18_|DA_|TFT\d*_)', '', tid or '')
    return re.sub(r'(18$|18_|_UniqueTrait$)', '', s)


def fetch(players, matches):
    from tft_player.lobby import LeagueClient
    TOP.mkdir(parents=True, exist_ok=True)
    c = LeagueClient()
    ladder = []
    for tier in ('CHALLENGER', 'GRANDMASTER'):
        st, body = c.request('GET', f'/lol-ranked/v1/apex-leagues/RANKED_TFT/{tier}')
        for d in (body or {}).get('divisions', []) if st == 200 else []:
            ladder += [(tier, s.get('leaguePoints') or 0, s['puuid']) for s in d.get('standings', []) if s.get('puuid')]
    ladder.sort(key=lambda x: -x[1])
    new = seen = 0
    for tier, lp, puuid in ladder[:players]:
        st, h = c.request('GET', f'/lol-match-history/v1/products/tft/{puuid}/matches?begin=0&count={matches}')
        for g in (h or {}).get('games', []) if st == 200 and isinstance(h, dict) else []:
            info = (g.get('json') or g)
            info = info.get('info') or info
            if info.get('queueId', info.get('queue_id')) != 1100:
                continue
            mid = str(info.get('matchId') or info.get('gameId') or info.get('game_id'))
            f = TOP / f'{mid}.json'
            seen += 1
            if not f.exists():
                info['_fetched_from'] = {'tier': tier, 'lp': lp}
                f.write_text(json.dumps(info), encoding='utf-8')
                new += 1
        time.sleep(0.2)
    c.close()
    print(f'{len(ladder)} players on the ladder; {min(players, len(ladder))} read; {seen} Ranked matches seen, {new} new; '
          f'{len(list(TOP.glob("*.json")))} saved in {TOP}')


def boards():
    for f in sorted(TOP.glob('*.json')):
        info = json.loads(f.read_text(encoding='utf-8'))
        for p in info.get('participants', []):
            units = [(unit_name(u.get('character_id')), u.get('tier') or 1, [item_name(i) for i in u.get('itemNames') or []])
                     for u in p.get('units', [])]
            traits = [(trait_name(t.get('name')), t.get('num_units')) for t in p.get('traits', []) if (t.get('tier_current') or 0) > 0]
            yield {'match': f.stem, 'place': p.get('placement'), 'level': p.get('level'), 'gold_left': p.get('gold_left'),
                   'last_round': p.get('last_round'), 'units': units, 'traits': traits}


def summary(want, out=None):
    B = list(boards())
    L = [f'# Top players\' boards: {len(B)} boards from {len({b["match"] for b in B})} Master+ Ranked matches', '']
    by_place = defaultdict(list)
    for b in B:
        by_place[b['place']].append(b)
    L.append('| place | boards | avg level | avg gold left | avg 2-star+ units | avg items on board |')
    L.append('|---|---|---|---|---|---|')
    for pl in sorted(by_place):
        bs = by_place[pl]
        avg = lambda xs: sum(xs) / len(xs) if xs else 0
        L.append(f'| {pl} | {len(bs)} | {avg([b["level"] or 0 for b in bs]):.1f} | {avg([b["gold_left"] or 0 for b in bs]):.0f} | '
                 f'{avg([sum(1 for u in b["units"] if u[1] >= 2) for b in bs]):.1f} | {avg([sum(len(u[2]) for u in b["units"]) for b in bs]):.1f} |')
    if want:
        sel = [b for b in B if all(any(u[0] == w for u in b['units']) for w in want)]
        L += ['', f'## Boards with {", ".join(want)}: {len(sel)} (average place '
                  f'{sum(b["place"] for b in sel) / len(sel):.2f})' if sel else f'## Boards with {", ".join(want)}: none', '']
        for half, cond in (('top 4', lambda p: p <= 4), ('bottom 4', lambda p: p > 4)):
            bs = [b for b in sel if cond(b['place'])]
            if not bs:
                continue
            units = Counter(u[0] for b in bs for u in b['units'])
            L.append(f'### {half} ({len(bs)} boards): level {sum(b["level"] for b in bs) / len(bs):.1f}, gold left '
                     f'{sum(b["gold_left"] or 0 for b in bs) / len(bs):.0f}')
            L.append('Units (share of boards, average star): ' + ', '.join(
                f'{n} {c / len(bs):.0%} {sum(u[1] for b in bs for u in b["units"] if u[0] == n) / c:.1f}*'
                for n, c in units.most_common(14)))
            items = defaultdict(Counter)
            for b in bs:
                for n, _, its in b['units']:
                    items[n].update(its)
            L.append('Items by holder: ' + '; '.join(f'{n}: ' + ', '.join(f'{i} {k}' for i, k in items[n].most_common(4))
                                                     for n, _ in units.most_common(10) if items[n]))
            tr = Counter(f'{t} {k}' for b in bs for t, k in b['traits'])
            L.append('Traits: ' + ', '.join(f'{t} {c}' for t, c in tr.most_common(10)))
            L.append('')
        firsts = [b for b in sel if b['place'] == 1][:5]
        if firsts:
            L.append('### Winning boards')
            for b in firsts:
                L.append(f'- level {b["level"]}, {b["gold_left"]} gold left: ' + '; '.join(
                    f'{n} {s}*' + (f' [{", ".join(i)}]' if i else '') for n, s, i in b['units']))
    text = '\n'.join(L) + '\n'
    if out:
        Path(out).write_text(text, encoding='utf-8')
        print(f'{out}: {len(text) // 1000} kB')
    else:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('fetch', 'summary'))
    ap.add_argument('--players', type=int, default=40)
    ap.add_argument('--matches', type=int, default=10)
    ap.add_argument('--units', default='')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.what == 'fetch':
        fetch(a.players, a.matches)
    else:
        summary([u.strip() for u in a.units.split(',') if u.strip()], a.out)


if __name__ == '__main__':
    main()
