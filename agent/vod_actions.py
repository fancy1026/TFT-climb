"""A read video's actions, phase by phase: rerolls, XP buys, units bought and sold, from the HUD and shop read 4 times
a second during each planning phase (agent/vod_read.py read the phases; this reads inside them).

    python agent/vod_actions.py runs/vod-<id> [--fps 4]

Inferred from state changes, the way the game shows them (no clicks are seen in a video):
- reroll: three or more shop cards change at once and gold drops by 2 (0 with a free reroll);
- XP buy: XP rises by 4 (or the level rises) and gold drops by 4;
- buy: one shop card empties and gold drops by 1-5 (its cost);
- sell: gold rises within the phase (no round start between).
Writes runs/vod-<id>/read/actions.jsonl (one row per planning phase) and prints the table.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'agent'))


def phases(rows):
    """[(stage, t_start, t_end)]: each stage's longest stretch in the 2-s reads, its planning part = from the first HUD
    frame while the countdown runs down (to the first frame where it rises again, or 32 s)."""
    cur, runs = None, []
    for r in rows:
        cur = r['stage'] or cur
        if cur:
            if runs and runs[-1][0] == cur:
                runs[-1][1].append(r)
            else:
                runs.append((cur, [r]))
    best = {}
    for st, rs in runs:
        if len(rs) > len(best.get(st, [])):
            best[st] = rs
    out = []
    for st, rs in sorted(best.items(), key=lambda kv: kv[1][0]['t']):
        hud = [r for r in rs if r['level'] is not None]
        if len(hud) < 3:
            continue
        t0, end, last = hud[0]['t'], hud[0]['t'] + 32, None
        for r in rs:
            if r['t'] < t0 or r['timer'] is None:
                continue
            if last is not None and r['timer'] > last + 3:
                end = r['t']
                break
            last = r['timer']
        out.append((st, max(0.0, t0 - 1), end + 1))
    return out


def same(a, b):
    return (a or '').strip() == (b or '').strip()


def stable(reads):
    """Only states the screen held for two reads in a row (gold, level, XP and the shop the same): one frame's OCR
    flicker or a card covered by an overlay is not a change (the first version counted 12 rerolls at 2-1 on 2 gold)."""
    key = lambda r: (r['gold'], r['level'], tuple(r['xp'] or ()), tuple((s or '').strip() for s in r['shop_raw']))
    out = []
    for a, b in zip(reads, reads[1:]):
        if a['gold'] is not None and a['shop_raw'] and key(a) == key(b) and (not out or key(out[-1]) != key(a)):
            out.append(a)
    return out


def infer(reads):
    """Actions between consecutive stable states of one phase, and how much of the gold change they account for."""
    acts = {'rerolls': 0, 'xp_buys': 0, 'buys': [], 'sells': 0, 'unexplained': 0}
    st = stable(reads)
    for prev, r in zip(st, st[1:]):
        dg = r['gold'] - prev['gold']
        changed = sum(not same(a, b) for a, b in zip(prev['shop_raw'], r['shop_raw']))
        emptied = [i for i, (a, b) in enumerate(zip(prev['shop_raw'], r['shop_raw'])) if (a or '').strip() and not (b or '').strip()]
        xp_up = (r['level'] or 0) > (prev['level'] or 0) or bool(r['xp'] and prev['xp'] and r['xp'][0] > prev['xp'][0])
        if xp_up and dg < 0 and dg % 4 == 0:
            acts['xp_buys'] += -dg // 4
        elif changed >= 3 and dg < 0 and dg % 2 == 0 and not emptied:
            acts['rerolls'] += -dg // 2           # several rerolls between two stable reads
        elif emptied and -5 * len(emptied) <= dg <= -1:
            acts['buys'] += [prev['shop_raw'][i].strip() for i in emptied]
        elif dg > 0:
            acts['sells'] += 1
        elif dg < 0:
            acts['unexplained'] -= dg             # spent, but no pattern above fits (a buy and a reroll at once, ...)
    acts['stable_reads'] = len(st)
    # XP buys from the XP gained over the phase (its first and last reads, the screen's own level steps), which a
    # level-up in a flickering frame cannot hide: 4 XP per buy (the passive XP comes before the phase's first read)
    gained = xp_gained(reads)
    if gained is not None and gained // 4 > acts['xp_buys']:
        acts['unexplained'] = max(0, acts['unexplained'] - 4 * (gained // 4 - acts['xp_buys']))
        acts['xp_buys'] = gained // 4
    return acts


def xp_gained(reads):
    """XP gained between the phase's first and last reads with an XP bar, from the level steps the screen shows."""
    xs = [r for r in reads if r['xp'] and r['level']]
    if len(xs) < 2:
        return None
    step = {}
    for r in xs:
        step[r['level']] = r['xp'][1]
    a, b = xs[0], xs[-1]
    if b['level'] < a['level']:
        return None
    if a['level'] == b['level']:
        return max(0, b['xp'][0] - a['xp'][0])
    total = step[a['level']] - a['xp'][0]
    for lv in range(a['level'] + 1, b['level']):
        if lv not in step:
            return None
        total += step[lv]
    return total + b['xp'][0]


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--fps', type=float, default=4)
    a = ap.parse_args()
    folder = Path(a.folder)
    rows = [json.loads(l) for l in (folder / 'read' / 'reads.jsonl').read_text(encoding='utf-8').splitlines()]
    from tft_player import screen
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    screen.save_hud_dx = lambda *args: None
    reader = screen.ScreenReader()
    # the same plateau offset as agent/vod_read.py found (recorded in its first line of output; -102 for BV1QVnH6vEgM)
    dx = json.loads((folder / 'read' / 'hud.json').read_text(encoding='utf-8'))['dx'] if (folder / 'read' / 'hud.json').exists() else -102
    screen.set_hud_dx(dx)
    out = []
    tmp = Path(tempfile.mkdtemp(prefix='vodact-'))
    try:
        for st, t0, t1 in phases(rows):
            d = tmp / st
            d.mkdir()
            subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-ss', f'{t0:.2f}', '-t', f'{t1 - t0:.2f}',
                            '-i', str(folder / 'game.mp4'), '-vf', f'fps={a.fps},scale=-2:1080', '-q:v', '3',
                            str(d / 'f%04d.jpg')], check=True)
            reads = []
            for p in sorted(d.glob('*.jpg')):
                s = reader.read(np.asarray(Image.open(p).convert('RGB')))
                if s.level is not None:
                    reads.append({'gold': s.gold, 'level': s.level, 'xp': s.xp, 'shop_raw': s.shop_raw})
            acts = infer(reads)
            golds = [r['gold'] for r in reads if r['gold'] is not None]
            row = {'stage': st, 't0': round(t0, 1), 't1': round(t1, 1), 'frames': len(reads),
                   'gold_start': golds[0] if golds else None, 'gold_end': golds[-1] if golds else None,
                   'level_start': reads[0]['level'] if reads else None, 'level_end': reads[-1]['level'] if reads else None, **acts}
            out.append(row)
            print(f"{st}: {row['frames']} frames, gold {row['gold_start']} -> {row['gold_end']}, level {row['level_start']} -> "
                  f"{row['level_end']}, rerolls {acts['rerolls']}, XP buys {acts['xp_buys']}, buys {acts['buys']}, sells {acts['sells']}, unexplained {acts['unexplained']} g ({acts['stable_reads']} stable)",
                  flush=True)
            shutil.rmtree(d, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    (folder / 'read' / 'actions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out), encoding='utf-8')


if __name__ == '__main__':
    main()
