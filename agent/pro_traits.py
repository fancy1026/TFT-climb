"""The traits strong players fielded, round by round, read from their games' keyframes (the owner, 2026-10-30: "it
should follow by traits ... if you watch all the pro plays, this is the same pattern"). The keyframe of each round
(agent/vod_read.keyframes: the board as the planning phase ends) shows the game's trait panel; it is read with the
player's own reader (tft_player.screen.ScreenReader.trait_panel), so the review and the comp coach can set a board
beside what strong players fielded at that round.

    python agent/pro_traits.py runs/vod-*          read the keyframes' trait panels -> read/traits.json
    python agent/pro_traits.py --comp "<comp>"     the comp's strong players' trait bonuses by round
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AGENT))


def read(folder, reader=None):
    """{keyframe stem: {trait: count}} of a read game's keyframes, cached in read/traits.json (new keyframes only)."""
    folder = Path(folder)
    keys = sorted((folder / 'read' / 'keyframes').glob('*.jpg'))
    out_f = folder / 'read' / 'traits.json'
    try:
        got = json.loads(out_f.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        got = {}
    todo = [k for k in keys if k.stem not in got]
    if todo:
        import numpy as np
        from PIL import Image
        if reader is None:
            from tft_player import screen
            reader = screen.ScreenReader()
        for k in todo:
            try:
                got[k.stem] = reader.trait_panel(np.asarray(Image.open(k).convert('RGB')))
            except Exception:
                got[k.stem] = {}
        out_f.write_text(json.dumps(got, ensure_ascii=False, indent=0), encoding='utf-8')
    return got


def bonuses(counts):
    """[(trait, count, tier)] of a panel's traits at a breakpoint of 2 or more."""
    from tft_player import board_score
    return board_score._tiers(counts or {})


def tiers(counts):
    return sum(t for *_, t in bonuses(counts))


def show(counts):
    from tft_player import board_score
    return board_score.show(bonuses(counts))


def extra_counts():
    """{trait: the most a champion adds to it beyond its own count} (tft_player.trait_text.trait_bonus)."""
    from tft_player.catalog import CATALOG
    from tft_player.trait_text import trait_bonus
    out = {}
    for name in CATALOG.units:
        for t, n in trait_bonus(name).items():
            out[t] = max(out.get(t, 0), n)
    return out


def own_board(row):
    """The keyframe shows the player's own board: its HUD (level and a shop) was read. A player scouting another board
    sees that board's trait panel and no shop (vod-yt-OdFTSRHjm_0 4-6: another player's Invoker 4, Blossom 3)."""
    return bool(row.get('level')) and sum(1 for s in row.get('shop') or () if s) >= 1


def of_comp(comp):
    """{stage: [(game, {trait: count})]} of the comp's strong players' read games (comp_coach.games_of), each stream
    split into its games as the comp coach splits them."""
    import comp_coach
    out = {}
    for g in comp_coach.games_of(comp):
        folder = ROOT / 'runs' / g['game']
        panels = read(folder)
        reads = comp_coach.jsonl(folder / 'read' / 'reads.jsonl')
        spans = [(min(r['t'] for r in rows), max(r['t'] for r in rows)) for rows in comp_coach.split_games(reads)]
        at = {int(r['t']): r for r in reads if r.get('t') is not None}
        for stem, counts in panels.items():
            t, _, stage = stem.partition('-')
            i = next((k for k, (a, b) in enumerate(spans) if a - 2 <= int(t) <= b + 2), None)
            row = at.get(int(t)) or {}
            if i is None or not counts or not own_board(row):
                continue
            # a count above the units fielded (the level, plus what a champion adds to a trait beyond itself: Elder
            # Dragon) is a misread - a name tag or a bar over the row (vod-yt-OdFTSRHjm_0 3-3 read 'Adaptor 7' at 6)
            counts = {k: v for k, v in counts.items() if v <= row['level'] + extra_counts().get(k, 0)}
            out.setdefault(stage, []).append((g['game'] + (f'-g{i + 1}' if i else ''), counts))
    return out


def line(entries):
    """One round's strong players' boards in words: their trait bonuses each, and how many on average."""
    if not entries:
        return ''
    per = [(game, bonuses(c)) for game, c in entries]
    avg = sum(sum(t for *_, t in b) for _, b in per) / len(per)
    from tft_player import board_score
    return (f"{avg:.1f} trait tiers on average (each breakpoint of 2+ reached is one tier) over {len(per)} strong players' games: "
            + '; '.join(f"{board_score.show(b)} ({game})" for game, b in per[:6]))


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('folders', nargs='*')
    ap.add_argument('--comp')
    a = ap.parse_args()
    for f in a.folders:
        got = read(f)
        print(f, len(got), 'keyframes read')
    if a.comp:
        import comp_coach
        by = of_comp(a.comp)
        for st in sorted(by, key=lambda s: comp_coach.stage_key(s) or (99, 99)):
            print(st, line(by[st]))


if __name__ == '__main__':
    main()
