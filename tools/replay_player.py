"""Run the player loop over recorded frames with no input sent.

Shows what the brain would do at each moment of a recorded game. Frames
don't react to our actions, so this checks perception + decisions, not
outcomes.

    python tools/replay_player.py <frames_dir> [--step 1]
"""
import argparse
import glob
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os  # noqa: E402
os.environ['TFT_COACH_BACKEND'] = 'none'   # no model calls in replay: local rules decide
os.environ['TFT_REPLAY'] = '1'            # clicks do nothing here: no info-panel reads
from tft_player.player import Player  # noqa: E402


class FrameFeed:
    """Recorded frames, scaled to the 1080-high frames the live capture gives.
    One recorded frame per loop step (step() at each tick): every read within the step sees it, as live reads a
    few ms apart see the same picture. (Advancing on every read made each read a new picture - no OCR cache hit,
    3.8 s per step.) Background threads see the current frame."""

    def __init__(self, files):
        self.files, self.i, self.cur, self.fresh = files, 0, None, True

    def step(self):
        self.fresh = True

    def read(self):
        if threading.current_thread() is not threading.main_thread() or (self.cur is not None and not self.fresh):
            return self.cur
        self.fresh = False
        if self.i >= len(self.files):
            raise KeyboardInterrupt
        im = Image.open(self.files[self.i]).convert('RGB')
        self.i += 1
        if im.height != 1080:
            im = im.resize((round(im.width * 1080 / im.height), 1080))
        self.cur = np.asarray(im), time.time()
        return self.cur


class NoInput:
    def __init__(self):
        self.log = []

    def click(self, p, button='left'):
        self.log.append(('click', p))

    def key(self, k, at=None):
        self.log.append(('key', k, at))

    def drag(self, a, b, duration=0.12):
        self.log.append(('drag', a, b))

    def hover(self, p, *a, **k):
        self.log.append(('hover', p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames')
    ap.add_argument('--step', type=int, default=1)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    files = sorted(glob.glob(str(Path(args.frames) / '*.png')) + glob.glob(str(Path(args.frames) / '*.jpg')))[::args.step]
    # replay runs as fast as possible - the game loop's waits only: background threads (the Watcher, the live
    # page) keep their sleeps, else they spin without pause and starved the loop (3 s per step)
    real_sleep = time.sleep
    time.sleep = lambda s: None if threading.current_thread() is threading.main_thread() else real_sleep(s)
    feed = FrameFeed(files)
    p = Player(feed, NoInput(), Path('runs/replay'))
    tick = p._tick

    def stepped(*a, **k):                # a new recorded frame for each loop step
        feed.step()
        return tick(*a, **k)
    p._tick = stepped
    p.coach.exe = None                    # no real model calls during replay
    try:
        p.run()
    except KeyboardInterrupt:
        pass
    print('\nlatency:', p.latency_report())


if __name__ == '__main__':
    main()
