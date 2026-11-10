"""Read the TFT HUD from one frame, fast.

Regions are measured on a 1920x1080 client. The HUD scales with client height
and is anchored to the horizontal centre: x' = w/2 + (x-960)*s, y' = y*s,
s = h/1080.

OCR is RapidOCR (PP-OCR recognition only; the regions are fixed so text
detection is skipped). Each crop gets a tiny pixel fingerprint and is only
re-recognised when it changes, so a typical tick OCRs one or two crops.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tft_player.catalog import CATALOG
from tft_player.meta import META

# ---- regions in 1080p pixels -------------------------------------------------
STAGE = [(755, 4, 815, 32), (810, 4, 870, 32)]     # normal bar, narrow stage-1 bar
TIMER = [(1125, 4, 1175, 32), (1070, 4, 1120, 32)]
LEVEL = (345, 880, 420, 910)
XP = (465, 882, 530, 908)
GOLD = (1015, 882, 1080, 912)
REROLL_COST = (372, 1036, 412, 1066)    # the gold cost on the Reroll button: 0 while free rerolls last
REROLL_FREE = (456, 1006, 494, 1040)    # the count badge on the Reroll button (free rerolls left)
SHOP_X0, SHOP_STEP = 555, 201.5
AUGMENT_TITLE = (830, 198, 1090, 248)               # "Choose One"
BOARD_COUNT = (880, 210, 1150, 315)                  # big "6/7" watermark shown in planning
BOARD_COUNT_WIDE = (800, 210, 1230, 315)             # its two-digit form ("10/10" spans x 827-1198)
AUGMENT_POSITIONS = ['left card', 'middle card', 'right card']     # the augment picker, by position only
AUGMENT_NAMES = [(415, 526, 700, 560), (815, 526, 1105, 560), (1220, 526, 1510, 560)]
HP_DIGITS_X = (1778, 1838)      # the digits inside our (big) HP box


# The bottom HUD row (level, XP, shop odds, gold, streak, the shop cards, the Reroll button) as measured above, shifted
# HUD_DX 1080p px. TFT 16.21 (2026-10-21) moved the row ~108 px left on a 2560x1600 client and every read failed
# (benchmark game 20261021-094200: gold, level and XP None from 1-2, the shop "closed"). HUD_DX is per client size:
# found by calibrate_hud when the level stops reading, kept in data/hud-dx.json.
HUD_DX = 0
HUD_FILE = Path(__file__).resolve().parent.parent / 'data' / 'hud-dx.json'
_HUD_BASE = {}


# the level label: "Lvl. 5" (English client) or "5级" (Chinese client: streamers' games, agent/vod_read.py)
LEVEL_TEXT = re.compile(r'l\w{0,2}\W*(\d{1,2})|(\d{1,2})\s*级', re.I)


def level_number(text):
    """The level a level-label OCR text shows (1-10), or None."""
    m = LEVEL_TEXT.search(text or '')
    n = int(m.group(1) or m.group(2)) if m else None
    return n if n and 1 <= n <= 10 else None


def set_hud_dx(dx):
    """Shift every bottom-HUD box (screen.py) and the shop's click points (board.py) to HUD_DX = dx."""
    global HUD_DX, LEVEL, XP, GOLD, REROLL_COST, REROLL_FREE, SHOP_X0
    from tft_player import board as B
    if not _HUD_BASE:
        _HUD_BASE.update(LEVEL=LEVEL, XP=XP, GOLD=GOLD, REROLL_COST=REROLL_COST, REROLL_FREE=REROLL_FREE,
                         SHOP_X0=SHOP_X0, STREAK_NUM=ScreenReader.STREAK_NUM, STREAK_ICON=ScreenReader.STREAK_ICON,
                         ODDS_BOX=ScreenReader.ODDS_BOX, SHOP_CLICK=list(B.SHOP_CLICK))
    sh = lambda b: (b[0] + dx, b[1], b[2] + dx, b[3])
    LEVEL, XP, GOLD = sh(_HUD_BASE['LEVEL']), sh(_HUD_BASE['XP']), sh(_HUD_BASE['GOLD'])
    REROLL_COST, REROLL_FREE = sh(_HUD_BASE['REROLL_COST']), sh(_HUD_BASE['REROLL_FREE'])
    SHOP_X0 = _HUD_BASE['SHOP_X0'] + dx
    ScreenReader.STREAK_NUM, ScreenReader.STREAK_ICON = sh(_HUD_BASE['STREAK_NUM']), sh(_HUD_BASE['STREAK_ICON'])
    ScreenReader.ODDS_BOX = sh(_HUD_BASE['ODDS_BOX'])
    B.SHOP_CLICK = [(x + dx, y) for x, y in _HUD_BASE['SHOP_CLICK']]
    HUD_DX = dx


def hud_key(w, h):
    """The HUD offset's key: the frame's aspect ratio (the offset is in 1080p px, the HUD scales with the height - the
    live player sees its frames downscaled to 1080 high, 1728x1080 for a 2560x1600 client)."""
    return f'{w / h:.3f}'


def load_hud_dx(w, h):
    """The HUD offset kept for this aspect ratio (data/hud-dx.json), set; None when there is none."""
    try:
        d = json.loads(HUD_FILE.read_text(encoding='utf-8'))
        dx = d.get(hud_key(w, h), d.get(f'{w}x{h}'))
    except (OSError, ValueError):
        dx = None
    if dx is not None:
        set_hud_dx(int(dx))
    return dx


def save_hud_dx(w, h, dx):
    try:
        d = json.loads(HUD_FILE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        d = {}
    d[hud_key(w, h)] = int(dx)
    try:
        HUD_FILE.write_text(json.dumps(d, indent=1), encoding='utf-8')
    except OSError:
        pass


# the Reroll badge can read as a circled digit: '①'.isdigit() is True, int('①') fails (a tick error twice in benchmark
# game 20261021-125639)
CIRCLED = str.maketrans({chr(0x2460 + k): str(k + 1) for k in range(9)})


def shop_name_box(i):
    x0 = SHOP_X0 + SHOP_STEP * i
    return (x0 + 2, 1043, x0 + 160, 1072)


@dataclass
class Screen:
    t: float = 0.0
    stage: str | None = None          # "3-2"
    timer: int | None = None
    level: int | None = None
    xp: tuple[int, int] | None = None
    gold: int | None = None
    hp: int | None = None
    shop: list[str | None] = field(default_factory=lambda: [None] * 5)
    shop_raw: list[str] = field(default_factory=lambda: [''] * 5)   # OCR text: wisps aren't champions
    shop_visible: bool = False
    augments: list[str] | None = None
    items: list | None = None          # item bench, filled by the player in planning
    board_count: int | None = None     # team slots used, from the game's own "n/m" watermark
    team_size: int | None = None       # its m, once two reads agree (the level unless something adds slots)
    free_rolls: int | None = None      # free rerolls left (Reroll button at 0 gold, its badge); 0 = rolls cost gold
    ms: float = 0.0

    @property
    def round(self):
        if not self.stage:
            return None
        a, b = self.stage.split('-')
        return int(a), int(b)

    def short(self):
        shop = ','.join(s or '-' for s in self.shop) if self.shop_visible else 'closed'
        aug = f' AUG={self.augments}' if self.augments else ''
        return (f"{self.stage or '?'} t={self.timer} L{self.level} xp={self.xp} "
                f"g={self.gold} hp={self.hp} shop=[{shop}]{aug} ({self.ms:.0f}ms)")


def read_board_count(text, level, wide=False):
    """The game's 'n/m' over the board: n team slots used of m, the team size. -> (n, m) or None.
    m is the level unless something adds slots (a +team-size augment, a Tactician's Shield, Riftbeast 10) or
    sets it (an augment that makes it 1). OCR often reads the slash as '7' ('474') or drops it ('54'): without
    a slash, m is the level or one more, the level first: a unit standing over the m changes its digit ("3/3" at
    level 3 read "375" - run 20261030-064325 1-4: team size 5, two units "fielded" that the game kept on the bench), so
    2+ slots over the level count only with the slash read. `wide`: read from BOARD_COUNT_WIDE (a two-digit m)."""
    t = re.sub(r'[^0-9/]', '', text or '')
    if not level:
        return None
    m = re.fullmatch(r'(\d{1,2})/(\d{1,2})', t)
    if m and wide and len(m.group(1)) == 2 and int(m.group(1)) > int(m.group(2)) + 2:
        m = re.fullmatch(r'\d(\d)/(\d{1,2})', t)      # the wide crop's left edge adds a '1' ("18/8" at 8/8)
    if m and 1 <= int(m.group(2)) <= level + 4 and int(m.group(1)) <= int(m.group(2)) + 2:
        size = int(m.group(2))
        if not wide and size < level and any(str(k).startswith(m.group(2)) for k in range(level, level + 3)):
            return None          # "10" cut to "1" by the narrow crop (run 20261029-051106 6-5: "0/1" at 9/10)
        return int(m.group(1)), size
    for size in range(level, level + 2):
        m = re.fullmatch(rf'(\d{{1,2}}?)[/7]?{size}', t)
        if m and 0 <= int(m.group(1)) <= size + 2:
            return int(m.group(1)), size
    return None


def parse_board_count(text, level):
    """'6/7' -> 6 at level 7: the n of the game's 'n/m' (read_board_count)."""
    got = read_board_count(text, level)
    return got[0] if got else None


def match_name(text, names, cutoff=0.75):
    t = re.sub(r"[^a-z'+]", '', text.lower())       # keep '+': "Trade Sector" != "Trade Sector+"
    if len(t) < 2:
        return None
    keys = {}
    for n in names:
        keys.setdefault(re.sub(r"[^a-z'+]", '', n.lower()), n)
    if t in keys:
        return keys[t]
    hit = difflib.get_close_matches(t, keys, n=1, cutoff=cutoff)
    return keys[hit[0]] if hit else None


MEMO_FILE = Path(__file__).resolve().parent.parent / 'data' / 'ocr_memo.npz'


def same_pixels(a, b):
    """Two crops of one box show the same thing: under 0.4% of the pixels differ by more than 40 steps - or by more
    than a third of the crops' brightness range, when that is lower. Text dimmed by a panel over it changes too few
    pixels by 40: TFT 16.21's level sits under our overlay's EXECUTION panel (crop mean 26, max 54), "Lvl. 2" against
    1-2's junk read 'A' differed in 0 of 2250 pixels by 40+, and the level stayed unread (benchmark game
    20261021-110650)."""
    if a.shape != b.shape:
        return False
    a16, b16 = a.astype(np.int16), b.astype(np.int16)
    rng = int(max(a16.max(), b16.max())) - int(min(a16.min(), b16.min()))
    thr = min(40, max(8, rng // 3))
    return (np.abs(a16 - b16).max(axis=2) > thr).mean() < 0.004


class OcrMemo:
    """Crops we've OCR'd before, per field: exact pixel match -> text in ~0.1 ms.
    HUD text renders identically every time, so after a game or two nearly
    every read is a lookup. Saved to disk."""

    def __init__(self, path=MEMO_FILE):
        self.path = path
        self.stacks: dict[str, np.ndarray] = {}
        self.sigs: dict[str, np.ndarray] = {}
        self.texts: dict[str, list[str]] = {}
        self.dirty = 0
        self.saved_at = time.time()
        try:
            with np.load(path, allow_pickle=False) as z:
                for k in z.files:
                    if k.endswith('|img'):
                        name = k[:-4]
                        stack, texts = z[k], list(z[name + '|txt'])
                        if name.split('@')[0] in self.NO_EMPTY:
                            keep = [i for i, t in enumerate(texts) if str(t).strip()]
                            stack, texts = stack[keep], [texts[i] for i in keep]
                            if not len(texts):
                                continue
                        self.stacks[name] = stack
                        self.sigs[name] = np.stack([self._sig(c) for c in stack])
                        self.texts[name] = texts
        except (FileNotFoundError, OSError, ValueError):
            pass

    @staticmethod
    def _sig(crop):
        """Small grayscale signature for a fast first pass."""
        return crop[::3, ::3].astype(np.int16).sum(axis=2)

    def get(self, key, crop):
        st = self.stacks.get(key)
        if st is None or st.shape[1:] != crop.shape:
            return None
        d = np.abs(self.sigs[key] - self._sig(crop)).mean(axis=(1, 2))
        i = int(d.argmin())
        if d[i] > 12:                       # summed over 3 channels
            return None
        # verify: essentially pixel-identical (one changed digit alters far more than 0.4%)
        return self.texts[key][i] if same_pixels(st[i], crop) else None

    # only fixed HUD text repeats pixel-for-pixel; crops over the moving board or
    # art (board counter, panel names, picker titles) never hit and just bloat it
    KEEP = ('level', 'xp', 'gold', 'stage', 'timer', 'stage1', 'timer1', 'hp', 'hpbox',
            'shop0', 'shop1', 'shop2', 'shop3', 'shop4', 'rrcost', 'rrbadge')
    CAP = 400
    # fields whose empty read is not kept: the level's absence closes the HUD (shop_visible), and TFT 16.21's level sits
    # under our overlay's EXECUTION panel - the dimmed "Lvl.5" matched dark crops kept as '' (benchmark game
    # 20261021-102523: no level, gold or shop read with the right boxes)
    NO_EMPTY = ('level',)
    NO_MEMO = ('level',)                  # fields never looked up in, nor added to, the memo (ScreenReader._texts)

    def add(self, key, crop, text):
        if key.split('@')[0] not in self.KEEP:
            return
        if key.split('@')[0] in self.NO_EMPTY and not str(text).strip():
            return
        st = self.stacks.get(key)
        if st is not None and st.shape[1:] != crop.shape:
            return
        if st is not None and len(st) >= self.CAP:          # oldest out
            self.stacks[key], self.sigs[key] = st[1:], self.sigs[key][1:]
            self.texts[key] = self.texts[key][1:]
            st = self.stacks[key]
        self.stacks[key] = crop[None].copy() if st is None else np.concatenate([st, crop[None]])
        sig = self._sig(crop)[None]
        self.sigs[key] = sig if st is None else np.concatenate([self.sigs[key], sig])
        self.texts.setdefault(key, []).append(text)
        self.dirty += 1
        if time.time() - self.saved_at > 60 and not getattr(self, '_saving', False):
            self.saved_at = time.time()
            self._saving = True
            import threading
            threading.Thread(target=self.save, daemon=True).start()   # never block the game loop

    def save(self):
        """Write the memo (uncompressed: compressing a large memo took ~5s)."""
        try:
            if not self.dirty:
                return
            arrays = {}
            for k in list(self.stacks):
                img, txt = self.stacks[k], list(self.texts[k])      # snapshot: OCR adds while we save
                n = min(len(img), len(txt))
                arrays[k + '|img'] = img[:n]
                arrays[k + '|txt'] = np.array(txt[:n])
            tmp = self.path.with_suffix(f'.{os.getpid()}.tmp.npz')
            np.savez(tmp, **arrays)
            tmp.replace(self.path)
            self.dirty = 0
        except (OSError, RuntimeError, ValueError):
            pass                    # best effort: a failed save must never cost a game tick
        finally:
            self.saved_at = time.time()
            self._saving = False


def _tune_onnx():
    """Every OCR model (each reader has its own) took all 16 CPU threads and busy-waited between runs: with the game,
    the video encoder and readers in other threads, a 30 ms read took 2-8 s (run 20261015-161625: 1,027 s of slow
    loop steps). 4 threads and no busy-waiting: p50 188 / p95 207 ms under load, against 479 / 1,968 ms (offline
    measure, 2 concurrent detections)."""
    from rapidocr.inference_engine.onnxruntime import main as M
    if getattr(M.OrtInferSession, '_tft_tuned', False):
        return
    orig = M.OrtInferSession._init_sess_opts

    def tuned(cfg):
        o = orig(cfg)
        o.intra_op_num_threads, o.inter_op_num_threads = 4, 1
        o.add_session_config_entry('session.intra_op.allow_spinning', '0')
        return o
    M.OrtInferSession._init_sess_opts = staticmethod(tuned)
    M.OrtInferSession._tft_tuned = True


class ScreenReader:
    def __init__(self):
        _tune_onnx()
        from rapidocr import RapidOCR
        from rapidocr.ch_ppocr_rec.typings import TextRecInput
        self._engine = RapidOCR(params={'Global.log_level': 'error'})
        self._input = TextRecInput
        self._cache: dict[tuple, tuple[np.ndarray, str]] = {}
        self.memo = OcrMemo()
        self.unit_names = list(CATALOG.units)
        # current augments first: the catalogue misses many live ones (e.g. Patient Study)
        self.augment_names = list(dict.fromkeys([*META.augments, *CATALOG.augments]))

    # -- OCR with change cache -------------------------------------------------
    def _texts(self, frame: np.ndarray, boxes: list[tuple], keys=None) -> list[str]:
        """Text per box: unchanged crop -> cached; seen-before crop -> memo;
        otherwise one batched OCR call (memoised when confident)."""
        out: list[str | None] = []
        todo = []
        h = frame.shape[0]
        for j, box in enumerate(boxes):
            crop = frame[box[1]:box[3], box[0]:box[2]]
            fp = crop.astype(np.int16)
            hit = self._cache.get(box)
            # a one-digit change touches only a few % of a crop: compare pixels, not means
            if hit is not None and same_pixels(hit[0], fp):
                out.append(hit[1])
                continue
            # (the level is never taken from the disk memo: its absence closes the HUD, and TFT 16.21 put it under our
            # overlay's panel, where crops of different levels look alike)
            key = f'{keys[j]}@{h}' if keys and keys[j] not in OcrMemo.NO_MEMO else None
            txt = self.memo.get(key, crop) if key else None
            if txt is not None:
                out.append(txt)
                self._cache[box] = (fp, txt)
                continue
            out.append(None)
            todo.append((len(out) - 1, box, crop, fp, key))
        if todo:
            res = self._engine.text_rec(self._input(img=[np.ascontiguousarray(t[2]) for t in todo]))
            for (i, box, crop, fp, key), txt, sc in zip(todo, res.txts, res.scores):
                txt = txt.strip()
                out[i] = txt
                self._cache[box] = (fp, txt)
                if key and (sc >= 0.9 or not txt):     # OCR is deterministic: same pixels, same text
                    self.memo.add(key, crop, txt)
        return out

    def calibrate_hud(self, frame, span=(-240, 120), step=6):
        """Find the bottom HUD's offset on this frame: the x shifts (1080p px) at which both the level ("Lvl. 5") and
        the gold read; the middle of that run is set and kept for this client size. -> the offset, or None (no HUD on
        screen: a picker, a carousel). ~1-3 s of OCR, once per client size."""
        h, w = frame.shape[:2]
        s = h / 1080
        base_level, base_gold = _HUD_BASE.get('LEVEL', LEVEL), _HUD_BASE.get('GOLD', GOLD)     # (unshifted)
        S = lambda b, dx: (round(w / 2 + (b[0] + dx - 960) * s), round(b[1] * s), round(w / 2 + (b[2] + dx - 960) * s),
                           round(b[3] * s))
        good = []
        for dx in range(span[0], span[1] + 1, step):
            level, gold = self._texts(frame, [S(base_level, dx), S(base_gold, dx)], [None, None])
            if level_number(level) and re.fullmatch(r'\d{1,3}', (gold or '').strip()):
                good.append(dx)
        if not good:
            return None
        run = [good[0]]                        # the longest run of neighbouring good shifts
        best = run
        for dx in good[1:]:
            run = run + [dx] if dx - run[-1] == step else [dx]
            best = run if len(run) > len(best) else best
        dx = (best[0] + best[-1]) // 2
        set_hud_dx(dx)
        save_hud_dx(w, h, dx)
        return dx

    def read_shop(self, frame):
        """Only what a roll-down needs: (gold, shop names, raw texts) - one batched OCR call on
        six small crops, memoised (a full read() is ~0.35 s; this is the inner loop of rolling)."""
        h, w = frame.shape[:2]
        s = h / 1080
        S = lambda b: (round(w / 2 + (b[0] - 960) * s), round(b[1] * s),
                       round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        gold, *names = self._texts(frame, [S(GOLD), *[S(shop_name_box(i)) for i in range(5)]],
                                   ['gold', *[f'shop{i}' for i in range(5)]])
        g = int(gold) if re.fullmatch(r'\d{1,3}', gold) else None
        return g, [match_name(n, self.unit_names) for n in names], [n.strip() for n in names]

    def shop_sig(self, frame):
        """The shop cards' name strips, coarse pixels: changes when the shop rolls, settles when the
        new cards are drawn (no OCR - for waiting on a roll)."""
        h, w = frame.shape[:2]
        s = h / 1080
        out = []
        for i in range(5):
            b = shop_name_box(i)
            x0, y0, x1, y1 = (round(w / 2 + (b[0] - 960) * s), round(b[1] * s),
                              round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
            out.append(frame[y0:y1:3, x0:x1:3].astype(np.int16))
        return out

    def read_clock(self, frame):
        """Stage/countdown only, for a nested action that must yield at the phase boundary."""
        for k in (0, 1):
            stage, timer = self._texts(frame, [self._centred(frame, STAGE[k]), self._centred(frame, TIMER[k])],
                                      [f'action_stage{k}', f'action_timer{k}'])
            m = re.search(r'([1-9])\s*-\s*([1-9])', stage)
            if m:
                n = re.search(r'\d{1,2}', timer)
                return f'{m[1]}-{m[2]}', int(n[0]) if n else None
        return None, None

    def read(self, frame) -> Screen:
        """frame: RGB numpy array (H, W, 3) or PIL image."""
        start = time.perf_counter()
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame.convert('RGB'))
        h, w = frame.shape[:2]
        s = h / 1080

        def S(b):
            return (round(w / 2 + (b[0] - 960) * s), round(b[1] * s),
                    round(w / 2 + (b[2] - 960) * s), round(b[3] * s))

        scr = Screen(t=time.time())
        level, stage, timer = self._texts(frame, [S(LEVEL), S(STAGE[0]), S(TIMER[0])],
                                          ['level', 'stage', 'timer'])
        if level_number(level):
            scr.level = level_number(level)
            scr.shop_visible = True

        if not re.search(r'[1-9]\s*-\s*[1-9]', stage):       # stage-1 bar is narrower
            stage, timer = self._texts(frame, [S(STAGE[1]), S(TIMER[1])], ['stage1', 'timer1'])
        m = re.search(r'([1-9])\s*-\s*([1-9])', stage)
        if m:
            scr.stage = f'{m.group(1)}-{m.group(2)}'
            m = re.search(r'\d{1,2}', timer)
            scr.timer = int(m.group()) if m else None

        if scr.shop_visible:
            # HUD open: xp, gold and the five shop names (one batched call at most)
            xp, gold, *names = self._texts(
                frame, [S(XP), S(GOLD), *[S(shop_name_box(i)) for i in range(5)]],
                ['xp', 'gold', *[f'shop{i}' for i in range(5)]])
            m = re.fullmatch(r'(\d{1,2})\D(\d{1,2})', xp.replace(' ', ''))
            if m:
                scr.xp = (int(m.group(1)), int(m.group(2)))
            if re.fullmatch(r'\d{1,3}', gold):
                scr.gold = int(gold)
            scr.shop = [match_name(n, self.unit_names) for n in names]
            # free rerolls (an encounter, Invested++): the Reroll button costs 0 and its badge counts them - the
            # model saw no count and spent them early (run 20261010-235723). A fixed HUD crop: cached, no OCR
            # unless it changes
            cost, badge = self._texts(frame, [S(REROLL_COST), S(REROLL_FREE)], ['rrcost', 'rrbadge'])
            cost, badge = cost.strip(), badge.strip().translate(CIRCLED)
            if cost == '0' and badge.isdecimal():
                scr.free_rolls = int(badge)
            elif re.fullmatch(r'\d', cost) and cost != '0':
                scr.free_rolls = 0
            scr.shop_raw = [n.strip() for n in names]
            if scr.level:
                # the big "n/level" over the board: units moving behind it change its pixels every tick, so its
                # OCR ran on almost every read - 9.8% of the game loop's time in run 20261010-235723's profile.
                # Its only user (the roster check) acts on a difference that lasts 1.5 s: once a second is enough
                last = getattr(self, '_board_count', None)
                if last is None or time.time() - last[1] > 1.0 or last[2] != scr.level:
                    got = self.board_count(frame, scr.level)
                    # the team size counts once two reads in a row agree (one OCR slip can't change it)
                    size = got[1] if got and last and last[3] == got[1] else None
                    last = (got[0] if got else None, time.time(), scr.level, got[1] if got else None, size)
                    self._board_count = last
                scr.board_count, scr.team_size = last[0], last[4]
        elif self._maybe_title(frame, S(AUGMENT_TITLE)):
            # HUD hidden: maybe the augment picker
            aug_title, = self._texts(frame, [S(AUGMENT_TITLE)], ['augtitle'])
            if re.search(r'choose', aug_title, re.I):
                # Detect the picker here; Player.read_augment_hand reads names after the flip
                # and requires two matching reads before exposing a name.
                scr.augments = AUGMENT_POSITIONS

        scr.hp = self._hp(frame, S)
        scr.ms = (time.perf_counter() - start) * 1000
        return scr

    def board_count(self, frame, level):
        """The game's 'n/m' over the board -> (n, m) or None (read_board_count). The narrow crop cuts a two-digit m
        ("9/10" read "0/1": team size 1 from 6-3 to 6-7, moves refused at "9/1 team slots" - run 20261029-051106):
        when it gives nothing or less than the level, the wide crop is read."""
        h, w = frame.shape[:2]
        s = h / 1080

        def S(b):
            return (round(w / 2 + (b[0] - 960) * s), round(b[1] * s),
                    round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        got = read_board_count(self._texts(frame, [S(BOARD_COUNT)], ['boardcnt'])[0], level)
        if got is None or got[1] < level:
            got = read_board_count(self._texts(frame, [S(BOARD_COUNT_WIDE)], ['boardcntw'])[0], level,
                                   wide=True) or got
        return got

    def shop_cost(self, frame, i):
        """The gold cost printed on shop card i (Wisps have varying costs), or None."""
        h, w = frame.shape[:2]
        s = h / 1080
        x0 = SHOP_X0 + SHOP_STEP * i
        b = (x0 + 160, 1043, x0 + 196, 1072)
        box = (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        m = re.search(r'\d{1,2}', self._texts(frame, [box], [f'shopcost{i}'])[0] or '')
        return int(m.group()) if m else None

    def read_augments(self, frame):
        """The three augment cards' names (known names, None where unreadable). Read once the
        cards have finished flipping; Player confirms two reads before asking the model."""
        h, w = frame.shape[:2]
        s = h / 1080
        S = lambda b: (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        txts = self._texts(frame, [S(b) for b in AUGMENT_NAMES], None)
        return [match_name(t, self.augment_names, cutoff=0.7) for t in txts], txts

    def special_card(self, frame, i):
        """A Wisp in the shop (any non-champion card: Coin Flip, Bronze Spoon, ...): (description, cost) read
        from the card, cached by its pixels. The description is cream text on a dark card: its lines are
        found from the pixels and read recognition-only (~0.05-0.25 s; detection OCR took 1.3-1.7 s)."""
        h, w = frame.shape[:2]
        s = h / 1080
        x0 = SHOP_X0 + SHOP_STEP * i
        S = lambda b: (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        a = S((x0 + 6, 950, x0 + 192, 1040))
        crop = np.ascontiguousarray(frame[a[1]:a[3], a[0]:a[2]])
        key = crop[::6, ::6].tobytes()
        hit = getattr(self, '_cards', {}).get(key)
        if hit:
            return hit
        lines = self._card_lines(crop, s)
        desc = ' '.join(t.strip() for t in self._engine.text_rec(self._input(img=lines)).txts) if lines else ''
        desc = re.sub(r'(?<![A-Za-z])I(?![A-Za-z])', '1', desc)     # the card font draws 1 like I
        out = (desc.strip(), self.shop_cost(frame, i))
        self._cards = getattr(self, '_cards', {})
        self._cards[key] = out
        return out

    @staticmethod
    def _card_lines(crop, s):
        """The text lines of a Wisp card's description: runs of rows with cream pixels, each padded
        with the card's background, the rarity badge (drawn over the first line) painted out."""
        cream = crop.min(axis=2) > 140
        prof = cream.sum(axis=1)
        rows = prof >= max(3, round(4 * s))
        runs, k = [], 0
        while k < len(rows):
            if not rows[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(rows) and (rows[j + 1] or (j + 2 < len(rows) and rows[j + 2])):
                j += 1
            if j - k + 1 >= round(6 * s) and prof[k:j + 1].max() >= round(8 * s):
                runs.append((k, j))
            k = j + 1
        if not runs or cream.all():
            return []
        bg = np.median(crop[~cream].reshape(-1, 3), axis=0).astype(crop.dtype)
        pad, out = round(3 * s), []
        for y0, y1 in runs:
            line = crop[max(0, y0 - pad):y1 + pad + 1].copy()
            line[(line.max(axis=2).astype(int) - line.min(axis=2)) > 90] = bg
            side = np.tile(bg, (line.shape[0], round(10 * s), 1))
            out.append(np.ascontiguousarray(np.hstack([side, line, side])))
        return out

    PANEL_BOX = (1690, 150, 1915, 420)      # portrait, stars, name, HP of the unit details panel

    def panel_sig(self, frame):
        """Pixels of the unit details panel, to tell when it shows a different unit."""
        h, w = frame.shape[:2]
        s = h / 1080
        b = self.PANEL_BOX
        x0, y0, x1, y1 = (round(w - (1920 - b[0]) * s), round(b[1] * s), round(w - (1920 - b[2]) * s), round(b[3] * s))
        return frame[y0:y1:3, x0:x1:3].astype(np.int16)

    TRAIT_ROWS = [274 + 53 * k for k in range(10)]      # the trait panel's rows (left edge, 1080p)
    TRAIT_SHIFT = 55          # the panel moves right when the item bench grows a second column (11+ items)

    def trait_shift(self, frame):
        """0, or TRAIT_SHIFT when the item bench has a second column (items.bench_columns). Without it,
        rows read at the one-column place gave 2-5 garbled traits on 11 of 137 recorded planning starts
        (221257 4-2..5-1, 152203 4-1..5-3)."""
        from tft_player.items import bench_columns
        return self.TRAIT_SHIFT if bench_columns(frame) == 2 else 0

    def trait_panel(self, frame):
        """The game's trait panel: {trait: units counted} for the rows shown (active traits
        first; a '4+' box hides the rest, all inactive). Left-anchored like the item bench."""
        h, w = frame.shape[:2]
        s = h / 1080
        dx = self.trait_shift(frame)
        B = lambda b: tuple(round(v * s) for v in b)
        names = self._texts(frame, [B((136 + dx, y - 13, 230 + dx, y + 11)) for y in self.TRAIT_ROWS], None)
        counts = self._texts(frame, [B((108 + dx, y - 13, 132 + dx, y + 12)) for y in self.TRAIT_ROWS], None)
        out = {}
        for n, c in zip(names, counts):
            t = match_name(n, list(CATALOG.traits), cutoff=0.7)
            m = re.fullmatch(r'\D{0,1}(\d)\D{0,1}', (c or '').strip())
            if t and m:
                out[t] = int(m.group(1))
        return out

    PANEL_NAME = (1685, 350, 1870, 377)
    PANEL_STARS = (1756, 188, 1844, 205)
    PANEL_HP = (1680, 382, 1905, 398)                 # the green health bar under the name

    def read_panel(self, frame):
        """Unit details panel (opened by right-clicking a unit): (name, stars) or (None, None)."""
        h, w = frame.shape[:2]
        s = h / 1080

        def S(b):   # the panel is anchored to the right edge
            return (round(w - (1920 - b[0]) * s), round(b[1] * s), round(w - (1920 - b[2]) * s), round(b[3] * s))
        txt, = self._texts(frame, [S(self.PANEL_NAME)], ['panelname'])
        self.last_panel_text = txt                     # kept raw: what an unexpected unit's panel says (data)
        x0, y0, x1, y1 = S(self.PANEL_HP)
        hp = frame[y0:y1, x0:x1].astype(np.int16)
        r, g, b = hp[..., 0], hp[..., 1], hp[..., 2]
        green = (g > 150) & (g - r > 50) & (g - b > 30)
        if hp.size and float(green.mean()) < 0.25:
            # a unit below ~25% health: its bar is green only from the left edge (run 20261027-120031 2-1: Shen at
            # 250/900 read 0.29, then lower - every fight read of him was "no panel", two gives "not found"). A
            # popup's green never starts at the bar's left edge (measured: player-name popups, 0.01-0.02)
            cols = green.mean(axis=0) > 0.5
            run = int(np.argmin(cols)) if not cols.all() else cols.size
            drained = run >= max(2, round(2 * s)) and float(cols[run:].mean() if run < cols.size else 0) < 0.05
        else:
            drained = False
        if hp.size == 0 or (float(green.mean()) < 0.25 and not drained):
            # no unit panel is open (its green health bar sits under the name): text there is something else -
            # run 20261010-235723 4-7: an opponent's "lucky choice: Ornn" popup was read as an Ornn*** on our
            # board when a click on the BFF opened no panel. Unit panels measured 0.53-0.90, popups 0.0
            self.last_panel_text = f'(no unit panel) {txt}'
            return None, None
        if re.search(r'FURRY|\bBFF\b', (txt or '').upper()):
            # the Sprykin trait's Big Furry Friend (run 20261010-113319 4-2: its panel was matched to "Alistar")
            from tft_player.state import BFF
            return BFF, 1
        name = match_name(txt, self.unit_names)
        if not name:
            # not a champion: a board object (Training Dummy) is recorded by its own name, one star
            from tft_player.state import board_objects
            obj = match_name(txt, list(board_objects()))
            return (obj, 1) if obj else (None, None)
        x0, y0, x1, y1 = S(self.PANEL_STARS)
        strip = frame[y0:y1, x0:x1, 0] > 120          # bronze/silver/gold stars; the teal border has low red
        cols = strip.any(axis=0).astype(np.int8)
        runs = np.diff(np.concatenate(([0], cols, [0])))
        widths = np.where(runs == -1)[0] - np.where(runs == 1)[0]
        stars = int((widths >= round(6 * s)).sum()) or None
        return name, (stars if stars and stars <= 3 else None)

    SETTINGS_TITLE = (425, 145, 600, 185)

    SETTINGS_BUTTONS = (700, 898, 1420, 932)          # "EXIT TFT  SURRENDER  RESTORE DEFAULTS"
    SURRENDER_TITLE = (780, 418, 1140, 456)           # "SURRENDER GAME" confirmation dialog

    def _centred(self, frame, b):
        h, w = frame.shape[:2]
        s = h / 1080
        return (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))

    def settings_open(self, frame):
        """Is the in-game Settings menu open? Its button row (EXIT TFT / SURRENDER / RESTORE
        DEFAULTS), or its title. The title alone failed at 16:10: our MODEL THINKING panel
        covers it (run22: the surrender never saw the menu)."""
        title, buttons = self._texts(frame, [self._centred(frame, self.SETTINGS_TITLE),
                                             self._centred(frame, self.SETTINGS_BUTTONS)], ['settingstitle', None])
        buttons = buttons.upper()
        return 'SETTING' in title.upper() or 'RESTORE' in buttons or ('EXIT' in buttons and 'SURRENDER' in buttons)

    def surrender_dialog(self, frame):
        """Is the 'SURRENDER GAME' confirmation open?"""
        txt, = self._texts(frame, [self._centred(frame, self.SURRENDER_TITLE)], None)
        return 'SURRENDER' in txt.upper()

    # the game's encounter, shown in the 1-1 carousel as a centred dark banner under the stage bar,
    # one or two lines ("Lux begins your journey with 3 random emblems.", "It's a calm day ... No
    # encounter this game.")
    ENCOUNTER_BOX = (610, 74, 1310, 142)

    def encounter(self, frame):
        """The 1-1 encounter banner's text, or None while it is not shown (~2-5 s: run it off the
        tick loop). Only lines centred on the screen count: player names float beside it."""
        b = self._centred(frame, self.ENCOUNTER_BOX)
        crop = frame[b[1]:b[3], b[0]:b[2]]
        mid = crop.shape[1] / 2
        lines = sorted((y, t.strip()) for x, y, _, t in self._detect(crop, 'encounter')
                       if abs(x - mid) < crop.shape[1] * 0.2 and t.strip())
        text = ' '.join(t for _, t in lines)
        return text if len(text.split()) >= 4 and text.endswith('.') else None

    def find_text(self, frame, width=1280):
        """Full-frame text detection + recognition (~2s; background use only).
        Returns [(text, (x, y) centre in frame pixels)]."""
        h, w = frame.shape[:2]
        k = max(1, w // width)
        small = np.ascontiguousarray(frame[::k, ::k])
        res = self._engine(small, use_cls=False)
        if res.boxes is None:
            return []
        return [(t, (float(b[:, 0].mean() * k), float(b[:, 1].mean() * k))) for t, b in zip(res.txts, res.boxes)]

    @staticmethod
    def _maybe_title(frame, box):
        """Cheap check before OCR: the picker title is large cream text."""
        c = frame[box[1]:box[3], box[0]:box[2]]
        bright = (c[..., 0] > 200) & (c[..., 1] > 190) & (c[..., 2] > 160)
        return 0.04 < bright.mean() < 0.5

    # ---- more of what the screen shows, read when the model is asked (not every tick) ----
    PLAYERS_BOX = (1560, 150, 1920, 830)     # the player list, right-anchored (8 rows)
    # the game's Damage Dealt panel slides over the player list (its sword button, top right); the people button
    # under the sword brings the list back (user, run 20261014-032220: open from 1-3 on, the 2-4 carousel misread
    # our row as the top one and waited 19 s at an open gate)
    DAMAGE_BOX = (1730, 266, 1880, 300)      # "<name> / Damage Dealt" under our portrait, right-anchored
    STANDINGS_BUTTON = (1898, 146, 'R')

    def damage_panel(self, frame):
        """Is the Damage Dealt panel covering the player list?"""
        h, w = frame.shape[:2]
        s = h / 1080
        b = self.DAMAGE_BOX
        txt, = self._texts(frame, [(round(w - (1920 - b[0]) * s), round(b[1] * s), round(w - (1920 - b[2]) * s),
                                    round(b[3] * s))], ['damagepanel'])
        return 'DAMAGE' in (txt or '').upper()
    STREAK_NUM = (1135, 876, 1180, 908)      # the number beside the streak flame
    STREAK_ICON = (1108, 878, 1134, 906)     # the flame: orange/red = win streak, blue = loss streak
    ODDS_BOX = (545, 878, 850, 910)          # shop odds per cost, above the shop

    def _detect(self, crop, key):
        """Detection + recognition on one region, cached by its pixels."""
        sig = crop[::8, ::8].tobytes()
        cache = self.__dict__.setdefault('_det_cache', {})
        if cache.get(key, (None,))[0] == sig:
            return cache[key][1]
        r = self._engine(np.ascontiguousarray(crop), use_cls=False)
        out = [] if r.boxes is None else [((b[0][0] + b[2][0]) / 2, (b[0][1] + b[2][1]) / 2, b[2][1] - b[0][1], t)
                                          for b, t in zip(r.boxes, r.txts)]
        cache[key] = (sig, out)
        return out

    def players(self, frame, our_hp=None):
        """The player list top to bottom (= standing): [(name or None, hp or None)].
        Our row is the big HP number without a name."""
        h, w = frame.shape[:2]
        s = h / 1080
        b = self.PLAYERS_BOX
        x0, y0, x1, y1 = round(w - (1920 - b[0]) * s), round(b[1] * s), w, round(b[3] * s)
        items = self._detect(frame[y0:y1, x0:x1], 'players')
        rows = []
        for cx, cy, ht, t in sorted(items, key=lambda i: i[1]):
            t = t.strip()
            if not t or t.upper() in ('X', 'LEVEL', 'LEVEL5!') or t.lower().startswith('level'):
                continue
            row = next((r for r in rows if abs(r['y'] - cy) < 22 * s), None)
            if row is None:
                row = {'y': cy, 'name': None, 'hp': None, 'big': False}
                rows.append(row)
            if re.fullmatch(r'\d{1,3}', t) and int(t) <= 100:
                row['hp'] = int(t)
                row['big'] = row['big'] or ht > 26 * s
            elif re.search(r'[A-Za-z]', t) and len(t) > len(row['name'] or ''):
                row['name'] = t              # a name split in two pieces: keep the longest
        # the small HP box right of a name the list read missed: recognition only, all in
        # one batch (a detection call per box cost ~1.4 s each, 6-9 s per plan prompt)
        miss = [r for r in rows if r['hp'] is None and r['name']]
        boxes = [(round(w - (1920 - 1818) * s), round(y0 + r['y'] - 13 * s),
                  round(w - (1920 - 1866) * s), round(y0 + r['y'] + 13 * s)) for r in miss]
        for r, t in zip(miss, self._texts(frame, boxes) if boxes else []):
            m = re.fullmatch(r'\D*(\d{1,3})\D*', t or '')
            if m and int(m.group(1)) <= 100:
                r['hp'] = int(m.group(1))
        out = []
        for r in rows:
            if r['name'] is None and r['hp'] is None:
                continue
            you = r['big'] or (r['name'] is None and our_hp is not None and r['hp'] == our_hp)
            out.append(('YOU' if you else r['name'], r['hp']))
        return out

    # the other players' small HP boxes: gold top and bottom borders ~20 px apart (1080p) across x 1828-1852, where
    # our own big box has none; the row's name label left of the box, the portrait right of it (grey = eliminated)
    LOBBY_BAND = (1828, 1852)

    def lobby(self, frame):
        """The player list without detection (recognition only, one batch; detection took 1-11 s before every plan):
        [{'name', 'hp', 'out', 'y'}] for the other players top to bottom (= by HP), and our row's index among them,
        or None when the list is covered (a unit's info panel, the Damage Dealt panel). Rows are found by their HP
        box borders, not fixed positions: the rows next to our big entry sit 6-12 px off a fixed grid (game 35)."""
        h, w = frame.shape[:2]
        s = h / 1080
        X = lambda x: round(w - (1920 - x) * s)
        if self.unit_panel_open(frame):
            return None
        y0 = round(150 * s)
        band = frame[y0:round(850 * s), X(self.LOBBY_BAND[0]):X(self.LOBBY_BAND[1])].astype(np.int16)
        r, g, b = band[..., 0], band[..., 1], band[..., 2]
        ys = np.where(((r > 120) & (g > 90) & (b < 110) & (r - b > 50)).mean(axis=1) > 0.8)[0]
        if not len(ys):
            return None
        lines = [float(gp.mean()) / s + 150 for gp in np.split(ys, np.where(np.diff(ys) > 2)[0] + 1)]
        cs = []
        for a, c in zip(lines, lines[1:]):
            if 15 <= c - a <= 25 and (not cs or (a + c) / 2 - cs[-1] > 40):
                cs.append((a + c) / 2)
        if not cs:
            return None
        boxes = []
        for y in cs:
            boxes.append((X(1640), round((y - 14) * s), X(1820), round((y + 14) * s)))      # name label
            boxes.append((X(1826), round((y - 10) * s), X(1858), round((y + 10) * s)))      # HP digits
        txt = self._texts(frame, boxes)
        champs = {n.lower() for n in CATALOG.units} | {t.lower() for t in CATALOG.traits}
        rows = []
        for i, y in enumerate(cs):
            por = frame[round((y - 15) * s):round((y + 15) * s), X(1862):X(1900)].astype(np.int16)
            grey = float((por.max(axis=2) - por.min(axis=2)).mean()) < 20
            m = re.fullmatch(r'\D{0,2}(\d{1,3})\D{0,2}', (txt[2 * i + 1] or '').strip())
            hp = int(m.group(1)) if m and int(m.group(1)) <= 100 else None
            # names in other scripts come out garbled, and a unit's name tag on the board can sit over the label:
            # a name only when it reads as one (Latin letters, not a champion or trait)
            # (a tag reads glued to a stray mark or star digit: "2Kayle", "aLeona" at 3-3/3-6 of game 35)
            name = re.sub(r'^[^A-Za-z]+', '', (txt[2 * i] or '').strip())
            low = name.lower()
            if (not re.fullmatch(r'[A-Za-z][\w .\-]{2,}', name, re.A) or low.startswith('level')
                    or any(low.endswith(c) and len(low) - len(c) <= 2 for c in champs)):
                name = None
            rows.append({'name': name, 'hp': hp, 'out': grey or hp == 0, 'y': round(y)})
        # our big entry: the gap in the rows (80 px apart), else above the first row or below the last
        gap = next((i + 1 for i in range(len(cs) - 1) if cs[i + 1] - cs[i] > 120), None)
        ours = gap if gap is not None else 0 if cs[0] > 240 else len(cs)
        return rows, ours

    def lobby_line(self, frame, our_hp=None):
        """The player list as one state line, or None when it is covered."""
        got = self.lobby(frame)
        if not got:
            return None
        rows, ours = got
        alive = [r for r in rows if not r['out']]
        if any(r['out'] for r in rows[:ours]):
            return None                          # an eliminated row above ours: misread
        shown = [f"{r['name'] or '?'} {r['hp'] if r['hp'] is not None else '?'}" for r in alive]
        shown.insert(ours, f"YOU {our_hp if our_hp is not None else '?'}")
        out = [r['name'] or '?' for r in rows if r['out']]
        n = len(alive) + 1
        return (f'Players (the list on the right, top = most HP): ' + ' | '.join(shown)
                + (f"; eliminated: {', '.join(out)}" if out else '')
                + f'. {n} alive; you are {ours + 1} of {n}.')

    ITEM_NAMES = None

    @classmethod
    def item_names(cls):
        """Every item name in the game data (components, completed, emblems, consumables...)."""
        if cls.ITEM_NAMES is None:
            import json as _json
            p = sorted((Path(__file__).resolve().parent.parent / 'data' / 'sources').glob('cdragon-tft-*.json'))[-1]   # newest
            from tft_player.catalog import CATALOG
            from tft_player.items import COMPONENTS, RECIPES
            names = set(COMPONENTS) | set(RECIPES)
            traits = set(CATALOG.traits)
            try:
                # this Set's items only: standard items, Set-18 items and every emblem - never a
                # bare trait name (the trait panel sits right beside the item lane)
                for it in _json.loads(p.read_text(encoding='utf-8'))['items']:
                    n, api = it.get('name') or '', it.get('apiName') or ''
                    n = re.sub(r'\s*<.*$', '', n)          # "Reforger <rules>(3 uses left!)</rules>"
                    if '_' in n:
                        continue                               # an untranslated key, not a name
                    if len(n) > 2 and not n.startswith('@') and n not in traits and (
                            n.endswith('Emblem') or api.startswith('TFT_Item_') or 'TFT18' in api
                            or (api.startswith('DA_') and 'Item' in api) or '_Consumable_' in api
                            # this Set's artifacts (run 20261016-052413 4-2: Hullcrusher, DA_Artifact_Hullcrusher, was
                            # missing, so the 4-card choice row fell to text detection and blocked the loop 9.4 s)
                            or api.startswith('DA_Artifact_')
                            or api.startswith('DA_Consumable') or api == 'DA_Reforger'
                            # Radiant items (TFT5_Item_*Radiant: run 221257, a Radiant Giant Slayer's
                            # tooltip was read 4 times and matched nothing - "unknown" for 5 rounds)
                            or (n.startswith('Radiant ') and api.endswith('Radiant'))
                            # this Set's own items (run27: the Knick-Knack reward "Attack Speed
                            # Booster", DA_SpeedBooster18, stayed unknown all game - its tooltip was
                            # read, but the name was not in this list)
                            or re.match(r'DA_.*18', api)):
                        names.add(n)
            except Exception:
                pass
            from tft_player.items import MISSING_ITEMS
            names |= set(MISSING_ITEMS)              # this set's items the game data lacks (the Alpha Mark)
            from tft_player.items import RADIANT_ICONS
            names |= set(RADIANT_ICONS)              # every current-set Radiant, including shared-art consumables
            cls.ITEM_NAMES = sorted(names)
        return cls.ITEM_NAMES

    def item_tooltip(self, frame, slot_xy, before=None, title_only=False):
        """The name in the tooltip shown while an item-bench slot is hovered, or None.
        slot_xy: the slot's point in left-anchored 1080p coordinates. before: the frame just before the hover - the
        tooltip is then the part that changed, and nothing else (the trait panel beside the item bench is never
        read as an item's tooltip); nothing changed = no tooltip yet, read again later."""
        h, w = frame.shape[:2]
        s = h / 1080
        x, y = slot_xy
        # the title sits on a fixed line beside the hovered slot (run27 frames: 110-470 px right of
        # the slot point, 30 px above to 4 below it): recognition on that line alone takes tens
        # of ms, where detection over the whole tooltip took 7-14 s under load
        line = (max(0, round((x + 110) * s)), max(0, round((y - 30) * s)),
                min(w, round((x + 470) * s)), min(h, round((y + 4) * s)))
        # never the cached text: a remover's count is one digit on a long line, under the cache's change threshold
        # (run 20261014-063145: "2 uses left" read again at 3-1 and 4-1 while the slot showed 5 and 6)
        self._cache.pop(line, None)
        if before is not None and before.shape == frame.shape:
            # the line must have changed since before the hover: text that stood there already is not a tooltip (run
            # 20261015-115044 3-2: an item slot with no tooltip read our tactician's name label as its item)
            x0, y0, x1, y1 = line
            if (np.abs(frame[y0:y1, x0:x1].astype(np.int16) - before[y0:y1, x0:x1].astype(np.int16)).max(axis=2)
                    > 40).mean() < 0.05:
                self.tooltip_text = ''
                return None
        t = self._texts(frame, [line], None)[0] or ''
        self.tooltip_text = t.strip()            # what the title line says, also when no name matches
        name = match_name(re.sub(r'\s*\(.*$', '', t), self.item_names(), 0.82)
        if name:
            uses = re.search(r'\((\d+)\s*uses?', t)
            if uses and ('Remover' in name or 'Reforger' in name):
                name = f'{name} ({uses.group(1)} uses left)'
            return name
        if title_only:
            return None                          # the action path: no 7-9 s detection over the whole tooltip
        x0, x1 = max(0, round((x + 22) * s)), min(w, round((x + 520) * s))
        y0, y1 = max(0, round((y - 160) * s)), min(h, round((y + 220) * s))
        # never our own overlay: its panels change while we hover (run 20261019-224321 2-1: the AI DECISION panel's
        # new text, over the top of slot 0's tooltip box, was read as the item's tooltip)
        from tft_player.overlay import panel_rects
        for px0, py0, px1, py1 in panel_rects(w, h):
            if px0 < x1 and x0 < px1:
                if py0 <= y0 < py1:
                    y0 = py1
                elif py0 < y1 <= py1:
                    y1 = py0
        if y1 - y0 < 20 * s:
            self.tooltip_text = ''
            return None
        if before is not None and before.shape == frame.shape:
            diff = (np.abs(frame[y0:y1, x0:x1].astype(np.int16) - before[y0:y1, x0:x1].astype(np.int16)).max(axis=2) > 40)
            rows, cols = np.where(diff.any(axis=1))[0], np.where(diff.any(axis=0))[0]
            if diff.mean() < 0.02 or not len(rows):
                self.tooltip_text = ''
                return None                       # no tooltip appeared: nothing to read (tried again later)
            # the tooltip's box: the rows and columns that changed
            x0, x1, y0, y1 = x0 + cols[0], x0 + cols[-1] + 1, y0 + rows[0], y0 + rows[-1] + 1
        items = self._detect(frame[y0:y1, x0:x1], f'tooltip{x},{y}')
        traits = {t.lower() for t in CATALOG.traits}
        # trait names and "2 > 4 > 6" thresholds are the trait panel's, never an item's
        items = [it for it in items if it[3].strip().lower() not in traits and not re.fullmatch(r'[\d\s>/]+', it[3].strip())]
        best = None
        for _, cy, ht, t in sorted(items, key=lambda i: i[1]):
            # consumables carry a count on the title line: "Item Remover (4 uses left!)"
            name = match_name(re.sub(r'\s*\(.*$', '', t), self.item_names(), 0.82)
            uses = re.search(r'\((\d+)\s*uses?', t)
            if name and uses and ('Remover' in name or 'Reforger' in name):
                name = f'{name} ({uses.group(1)} uses left)'
            if name and (best is None or ht > best[1]):
                best = (name, ht)                 # the name is the biggest matching text
        if not best and items:
            self.tooltip_text = ' / '.join(t for *_, t in sorted(items, key=lambda i: i[1])[:3])
        return best[0] if best else None

    def streak(self, frame):
        """(n, 'win' | 'loss') from the flame beside the gold, or None."""
        h, w = frame.shape[:2]
        s = h / 1080
        S = lambda b: (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        txt = self._texts(frame, [S(self.STREAK_NUM)], ['streak'])[0]
        m = re.search(r'\d{1,2}', txt)
        if not m:
            return None
        x0, y0, x1, y1 = S(self.STREAK_ICON)
        icon = frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.int16)
        bright = icon[icon.max(axis=1) > 120]
        if not len(bright):
            return int(m.group()), '?'
        r, g, b = bright.mean(axis=0)
        return int(m.group()), 'loss' if b > r else 'win'

    def shop_odds(self, frame):
        """Shop odds per cost 1..5 as printed above the shop, e.g. [45, 33, 20, 2, 0]."""
        h, w = frame.shape[:2]
        s = h / 1080
        b = self.ODDS_BOX
        box = (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        text = self._texts(frame, [box], ['odds'])[0]
        vals = [int(v) for v in re.findall(r'(\d{1,3})\s*%', text)]
        if len(vals) == 5:
            return ' / '.join(f'{v}%' for v in vals)
        return f'{text.strip()} (read incompletely)' if vals else None

    # where the gold border lines of our big HP box start (1080p x, right-anchored): 1782 (3 digits) - 1795 (1-2)
    # at 2560x1600 and 2560x1440 alike. Other players' small boxes start at 1822-1823; lines of the scenery behind
    # the list run past the search's left edge (1760). The earlier test - "no username text at x 1826-1850" - held
    # at 16:10 only: at 2560x1440 our own box reaches into that place, every read was refused and the model played
    # from 2-5 on with 90, then a stray 1 (real 76 -> 2; release run 20261016-141957)
    HP_BOX_START = (1772, 1808)

    @staticmethod
    def unit_panel_open(frame):
        """The unit info panel over the player list, by its health bar row: mostly green at full health, or - its unit
        hurt in a fight - a green part from the left and the rest near-black, white digits over both. (The old test,
        over half green, missed a hurt unit's panel: benchmark game 20261018-183028 took a "2" off Akali's panel at 2-1
        for our HP and the model was told "HP 2" for two rounds. 1083 recorded frames: no panel found where there was
        none; 984 of 984 panels with 10-60% of the bar left found, 32 of them by the old test.)"""
        h, w = frame.shape[:2]
        s = h / 1080
        bar = frame[round(382 * s):round(396 * s), round(w - (1920 - 1690) * s):round(w - (1920 - 1890) * s)].astype(np.int16)
        if not bar.size:
            return False
        green = (bar[..., 1] > 150) & (bar[..., 0] < 110) & (bar[..., 2] < 110)
        if float(green.mean()) > 0.5:
            return True
        cols = green.mean(axis=0) > 0.5
        cols |= (bar.max(axis=2) < 60).mean(axis=0) > 0.5          # the bar's lost part
        cols |= (bar.min(axis=2) > 170).mean(axis=0) > 0.15        # its digits
        return bool((green.mean(axis=0)[:3] > 0.5).all() and cols.mean() >= 0.8)

    def _hp(self, frame, S):
        """Our HP: the number in our own entry of the player list, the only one drawn as a big box with long gold
        border lines (top and bottom, ~37px at 1080p; other players' boxes are ~25px), found by those lines and
        confirmed by where they start (HP_BOX_START: run 010416 2-6 took a '7' for our 84 HP off another entry).
        The old rule - the tallest gold ring - read digits off a name ("RainDude87" -> 87 HP while we had 42)."""
        h, w = frame.shape[:2]
        s = h / 1080
        # the unit info panel (open after a unit read) covers the player list: its gold edges and
        # numbers were read as our HP (run32 3-3: "2" while we had 53)
        if self.unit_panel_open(frame):
            return None
        x0, x1 = round(w - (1920 - 1760) * s), round(w - (1920 - 1860) * s)
        y0, y1 = round(150 * s), round(0.85 * h)
        col = frame[y0:y1, x0:x1].astype(np.int16)
        r, g, b = col[..., 0], col[..., 1], col[..., 2]
        gold = (r > 120) & (g > 90) & (b < 110) & (r - b > 50)
        need = 30 * s
        lines, starts = [], {}
        for y in np.where(gold.sum(axis=1) >= need)[0]:
            d = np.diff(np.concatenate(([0], gold[y].view(np.int8), [0])))
            run_s, run_e = np.where(d == 1)[0], np.where(d == -1)[0]
            k = int((run_e - run_s).argmax())
            if run_e[k] - run_s[k] >= need:
                lines.append(y)
                starts[y] = run_s[k]
        if not lines:
            return None
        groups = np.split(np.array(lines), np.where(np.diff(lines) > 2)[0] + 1)
        ys = [(int(gp.mean()), min(starts[y] for y in gp)) for gp in groups]
        lo, hi = self.HP_BOX_START
        start = lambda a, b: 1920 - (w - x0 - min(a[1], b[1])) / s
        pair = next(((a[0], b[0]) for i, a in enumerate(ys) for b in ys[i + 1:]
                     if 28 * s <= b[0] - a[0] <= 40 * s and lo <= start(a, b) <= hi), None)
        if pair is None:
            return None                                   # no box of ours (another entry's, or the scenery)
        yc, half = y0 + (pair[0] + pair[1]) // 2, round(13 * s)
        # fixed size (memo hits) and quantised (the change cache holds when a line jitters by a pixel)
        box = (round(w - (1920 - HP_DIGITS_X[0]) * s), (yc - half) // 2 * 2,
               round(w - (1920 - HP_DIGITS_X[1]) * s), (yc - half) // 2 * 2 + 2 * half)
        txt = self._texts(frame, [box], ['hpbox'])[0]
        m = re.fullmatch(r'\D{0,2}(\d{1,3})\D{0,2}', (txt or '').strip())     # '4.' happens
        return int(m.group(1)) if m and 0 < int(m.group(1)) <= 100 else None
