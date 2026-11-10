"""A recorded game video (a streamer's VOD) read by the bot's own screen reader, frame by frame - the first step of
learning from other players' games (does our perception read their screen?).

    python agent/vod_read.py <video> [--every 2] [--out <dir>]

Frames every N seconds at 1080 high (ffmpeg), each read with tft_player.screen.ScreenReader: stage, timer, level, XP,
gold, HP, shop (names matched and the raw text). Writes <out>/reads.jsonl (one row per frame: t = seconds into the
video) and prints how often each field was read and the stage sequence, with the gaps a cut would leave.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract(video, out, every):
    out.mkdir(parents=True, exist_ok=True)
    if not any(out.glob('f*.jpg')):
        subprocess.run([ffmpeg(), '-hide_banner', '-loglevel', 'error', '-i', str(video), '-vf',
                        f'fps=1/{every},scale=-2:1080', '-q:v', '4', str(out / 'f%05d.jpg')], check=True)
    return sorted(out.glob('f*.jpg'))


def stage_key(s):
    a, b = s.split('-')
    return int(a), int(b)


def read_chunk(args, reader=None):
    """Rows for a run of frames: (paths, index of the first, seconds between frames, HUD offset). A worker process
    makes its own reader with the offset the calibration found."""
    paths, first, every, dx = args
    sys.path.insert(0, str(ROOT))
    from tft_player import screen
    if reader is None:
        screen.save_hud_dx = lambda *x: None
        screen.set_hud_dx(dx)
        reader = screen.ScreenReader()
    rows = []
    for i, p in enumerate(paths, first):
        scr = reader.read(np.asarray(Image.open(p).convert('RGB')))
        rows.append({'t': round(i * every, 1), 'frame': Path(p).name, 'stage': scr.stage, 'timer': scr.timer,
                     'level': scr.level, 'xp': scr.xp, 'gold': scr.gold, 'hp': scr.hp,
                     'shop': scr.shop if scr.shop_visible else None,
                     'shop_raw': scr.shop_raw if scr.shop_visible else None})
    return rows


def keyframes(rows, frames, dest, width=1280):
    """One picture a round, kept when the video's frames are deleted: the frame of each stretch of one stage with the
    least planning time left (the board, bench, items and trait panel as the round is fought), for the comp coach to
    look at (agent/comp_coach.py). Named <seconds into the video>-<stage>.jpg. -> how many."""
    runs, cur = [], []
    for r in rows:
        if r.get('stage') and cur and cur[-1]['stage'] != r['stage']:
            runs.append(cur)
            cur = []
        if r.get('stage'):
            cur.append(r)
    if cur:
        runs.append(cur)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for run in runs:
        timed = [r for r in run if isinstance(r.get('timer'), int) and r['timer'] > 0]
        pick = min(timed, key=lambda r: r['timer']) if timed else run[-1]
        src = frames / pick['frame']
        if not src.exists():
            continue
        im = Image.open(src).convert('RGB')
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)))
        im.save(dest / f"{int(pick['t']):06d}-{pick['stage']}.jpg", quality=80)
        n += 1
    return n


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--every', type=float, default=2.0)
    ap.add_argument('--out')
    ap.add_argument('--workers', type=int, default=4, help='reader processes (each loads its own OCR models)')
    ap.add_argument('--keep-frames', action='store_true', help='keep the extracted pictures (default: deleted after reading)')
    a = ap.parse_args()
    video = Path(a.video)
    out = Path(a.out) if a.out else video.parent / 'read'
    frames = extract(video, out / 'frames', a.every)
    from tft_player import screen
    reader = screen.ScreenReader()
    # the bottom HUD's offset for this video (a streamer's client and HUD scale): found on the first frames that
    # show it, set for this run only - never saved to data/hud-dx.json, which the live player reads
    # (calibrate_hud's middle-of-run pick suits "Lvl. 7": the shorter "7级" read only at its edge - 2 of 682 frames):
    # the middle of the widest plateau of shifts at which level, XP and gold all read on 24 sample frames
    screen.save_hud_dx = lambda *args: None
    sample = [np.asarray(Image.open(p).convert('RGB')) for p in frames[len(frames) // 25::max(1, len(frames) // 25)][:24]]
    score = {}
    for dx in range(-180, 61, 6):
        screen.set_hud_dx(dx)
        reads = [reader.read(f) for f in sample]
        score[dx] = sum(s.level is not None and s.xp is not None and s.gold is not None for s in reads)
    top = max(score.values())
    best, run = [], []
    for dx in sorted(score):
        run = run + [dx] if score[dx] == top else []
        best = run if len(run) > len(best) else best
    dx = best[len(best) // 2] if top else 0
    screen.set_hud_dx(dx)
    (out / 'hud.json').write_text(json.dumps({'dx': dx, 'plateau': [best[0], best[-1]] if best else None}) + '\n',
                                  encoding='utf-8')          # agent/vod_actions.py reads the same offset
    print(f'HUD offset: {dx} ({top} of {len(sample)} sample frames read; plateau {best[0]}..{best[-1]})' if top
          else 'HUD offset: none found')
    t0 = time.time()
    if a.workers > 1 and len(frames) > 200:
        # contiguous chunks, one reader per process (each loads its own OCR models); order kept by chunk
        from multiprocessing import Pool
        k = a.workers
        size = (len(frames) + k - 1) // k
        chunks = [([str(p) for p in frames[i:i + size]], i, a.every, dx) for i in range(0, len(frames), size)]
        with Pool(k) as pool:
            rows = [r for part in pool.map(read_chunk, chunks) for r in part]
    else:
        rows = read_chunk(([str(p) for p in frames], 0, a.every, dx), reader)
    with open(out / 'reads.jsonl', 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    n_key = keyframes(rows, out / 'frames', out / 'keyframes')
    print(f'{n_key} keyframes (one a round: the board as the planning phase ends) -> {out / "keyframes"}')
    try:                                                   # the trait panel of each round (agent/pro_traits.py)
        sys.path.insert(0, str(ROOT / 'agent'))
        import pro_traits
        print(f'{len(pro_traits.read(out.parent, reader))} trait panels read -> {out / "traits.json"}')
    except Exception as e:
        print(f'trait panels not read ({e.__class__.__name__}: {e})')
    if not a.keep_frames:
        shutil.rmtree(out / 'frames', ignore_errors=True)   # the reads are kept; the pictures are the video's
    n = len(rows)
    print(f'read in {time.time() - t0:.0f} s with {a.workers} worker(s)')
    print(f'{n} frames, one every {a.every:g} s -> {out / "reads.jsonl"}')
    sys.path.insert(0, str(ROOT / 'agent'))
    from channel import post
    stages = [r['stage'] for r in rows if r['stage']]
    post('Data', f"I read {video.name}: {n} frames ({n * a.every / 60:.0f} min of video) in {time.time() - t0:.0f} s; "
                 f"the HUD (level, XP, gold, shop) read in {sum(r['level'] is not None for r in rows)} frames, stages "
                 f"{stages[0] if stages else '-'} to {stages[-1] if stages else '-'}. Next: the case builder.",
         refs=[str(out / 'reads.jsonl')])
    for k in ('stage', 'timer', 'level', 'xp', 'gold', 'hp', 'shop'):
        print(f'  {k}: read in {sum(r[k] is not None for r in rows)} of {n}')
    seen = []
    for r in rows:
        if r['stage'] and (not seen or seen[-1][0] != r['stage']):
            seen.append((r['stage'], r['t']))
    print('stages in order (first second seen):', ', '.join(f'{s}@{t:.0f}' for s, t in seen))
    order = [s for s, _ in seen]
    missing = []
    for x, y in zip(order, order[1:]):
        (a1, b1), (a2, b2) = stage_key(x), stage_key(y)
        if (a2, b2) > (a1, b1) and not (a2 == a1 and b2 == b1 + 1) and not (a2 == a1 + 1 and b2 == 1):
            missing.append(f'{x} -> {y}')
    print('jumps (a cut, or a round the reader missed):', ', '.join(missing) or 'none')


if __name__ == '__main__':
    main()
