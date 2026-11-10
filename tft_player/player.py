"""The real-time player loop.

    capture (continuous) -> read HUD + occupancy -> update roster
      -> brain.decide -> one input -> repeat

Started by tools/play.py. Stop: F8 at any time.
"""
from __future__ import annotations

import argparse
import atexit
import copy
import json
import math
from collections import Counter
import os
import re
import threading
import time
import types
from pathlib import Path

import cv2
import numpy as np

from tft_player import board as B
from tft_player.brain import Action, Brain
from tft_player.coach import Coach, local_augment, set_stage
from tft_player.loot import BeamOrbs, LootRunner, find_beams, find_orbs, find_tactician
from tft_player.carousel import (CarouselChaser, OwnTracker, find_bars,
                                 find_carousel_items, gate_ring, release_timer)
from tft_player.items import (COMPONENTS, BY_PAIR, N_SLOTS, ItemTracker, fitting, read_item_bench, read_panel_items,
                              is_remover, same_slot_item, slot_point, uses_left)
from tft_player.screen import ScreenReader
from tft_player.state import BFF
from tft_player.live import LiveView
from tft_player.catalog import CATALOG
from tft_player.state import Roster


class NotForeground(Exception):
    pass

AUGMENT_CARDS = [(555, 450), (960, 450), (1365, 450)]
AUGMENT_REROLL = [(555, 864), (960, 864), (1365, 864)]
SETTINGS_CLOSE = (1476, 163)       # X of the in-game Settings menu (1080p reference)
# where our tactician is parked each planning phase: open ground at the arena's right edge, ~95 px from the
# nearest hexes and the bench (its default spot, left of hexes (1,0)/(0,0), hid units there: run 213127 1-4
# Hecarim, 4-3 Varus); the bar scan ends at x 1480, so its own bar is out of the way too
# where the tactician waits in a planning phase: open ground above our board's top-left corner, on the half that holds
# enemy units only in combat - it covers none of our units, bars, bench slots or item rows. The ledge in the bottom-left
# corner (the user's spot of 2026-10-10, where the game puts it after every trip; feet 426, 603) hid bench slots 0-1
# and hex (1,0) behind a large tactician: panel reads clicked it, E-sells missed, a Lifebloom and Nidalee's item row
# went unseen (runs 20261015-115044, -132144, -161625). Same click-to-feet offset as there (47 px).
PARK, PARK_FEET = (690, 337), (690, 290)
# where our units' health bars can be in a fight: the board, not the HUD under it. To y 900 the shop odds' green
# 2-cost dot (~(542, 895)) read as a bar in every fight frame, and find_in_fight spent one of its 6 panel reads on a
# right-click on the shop (run 20261027-110551 5-1: the only "bar" left once all our units were dead)
FIGHT_BARS_Y = (120, 870)
# seconds of planning left from which fielding comes first: while a plan step still has to put a unit on the board
# (or bench one), loot, reads and checks wait (the user, 2026-10-10: run 064944 moved units with 1-3 s left)
FIELD_FIRST_S = 6
RUNS = Path(__file__).resolve().parent.parent / 'runs'
REPLAY = bool(os.environ.get('TFT_REPLAY'))
CODE_VERSION = None      # code_version() when the harness was loaded (tools/play_match.load_harness)


def about_other(said, item):
    """A game notice that is not the game's answer to a drop of `item`: the Alpha Mark's own banners ("Don't forget to
    use your Alpha Mark!", "Alpha Mark was used automatically. You can select a different Riftbeast after combat.")
    stand over the bench for seconds whatever is dropped. Game 36 took them for refusals of Chain Vest (4-2),
    Infinity Edge (4-3) and Evenshroud (4-5): the give stopped waiting at once and was reported "refused", the
    items stayed off the board for three fights."""
    t = (said or '').lower()
    if "forget to use" in t:
        return True
    return 'alpha mark' in t and 'alpha mark' not in re.sub(r'\s*\(.*$', '', item or '').lower()


def fill_unknown(read, known):
    """A unit's items as read, an icon not recognised named by the records: it is not another item, and a name we
    knew that is missing from the reading fills it (run 033507: Aphelios' Kraken's Fury read "unknown" for 13 reads
    after Cybernetic Implants changed the icons; benchmark game 20261020-221427 2-6: Hecarim's panel showed his Sunfire
    Cape as "unknown"). Components combining shrink the count: then no filling."""
    if 'unknown' not in read or len(read) < len(known):
        return list(read)
    spare = list(known)
    for n in read:
        if n in spare:
            spare.remove(n)
    spare = [n for n in spare if n != 'unknown']
    return [spare.pop(0) if n == 'unknown' and spare else n for n in read]


def fill_from_records(read, known, expect=None):
    """fill_unknown against what the unit should hold after a give whose landing was not seen (`expect`: the item
    left the item bench, no unit showed it) when the icons that were named agree with it; else against the records.
    Game 26 3-1: Chain Vest met Rengar's Giant's Belt, his row read empty, his panel at 3-2 showed two unknown icons
    and they were filled from the old records as Giant's Belt + Warmog's Armor - at 3-5 a Rod went on him for a
    Morellonomicon and stayed a lone component next to his Sunfire Cape."""
    if expect and not Counter(n for n in read if n != 'unknown') - Counter(expect):
        return fill_unknown(read, expect)
    return fill_unknown(read, known)


def code_version():
    """Which harness this is: the commit, and the uncommitted files under tft_player/ and tools/ with a hash of
    their diff - models are only comparable on the same code."""
    import hashlib
    import subprocess
    root = Path(__file__).resolve().parent.parent
    git = lambda *a: subprocess.run(['git', *a], cwd=root, capture_output=True, text=True, timeout=10).stdout
    # status lines are 'XY path': not stripped as a whole, or the first line loses its leading space and its path its
    # first letter (benchmark game 20261018-110145: "ools/benchmark.py")
    diff = git('diff', 'HEAD', '--', 'tft_player', 'tools')
    return {'git': git('rev-parse', '--short', 'HEAD').strip(),
            'dirty_files': [l[3:] for l in git('status', '--short', '--', 'tft_player', 'tools').splitlines()
                            if l.strip()],
            'dirty_hash': hashlib.sha1(diff.encode('utf-8', 'replace')).hexdigest()[:10] if diff else None}


def test_mode():
    """Test-game data, in runs/<ts>: the carousel frames, the per-round unit crops and the link-drag frames. Only with
    --record (a test-only option): by default a game runs exactly as the release's does (the user, 2026-10-15: the
    release is the test version, functionally and visually). Never on a replay."""
    return not REPLAY and os.environ.get('TFT_TEST_MODE') == '1'


def shots_on():
    """Screenshots go into the game's folder (round starts, fights, failed actions, model-call evidence);
    off with --no-screenshots - the log and the model conversation are kept either way."""
    return not REPLAY and os.environ.get('TFT_NO_SCREENSHOTS') != '1'
DATA_MODE = bool(os.environ.get('TFT_DATA_MODE'))   # data-collection games (tools/play_comp.py --data)
SELL_HOVER = 0.2         # s the cursor rests on a unit before E sells it (Input.key then holds E for KEY_HOLD)
MERGE_SETTLE = 1.2       # s a newly lit slot must stay lit before a merge we expected is judged not to have happened
DATA_ROLL = bool(os.environ.get('TFT_DATA_ROLL'))   # attached to the user's game at level 8: only reroll /
                                                    # buy / photograph / sell what WE bought (--data-roll)     # tools/replay_player.py: recorded frames, clicks do nothing


class Watcher(threading.Thread):
    """Slow, non-HUD perception off the main loop: tactician position every
    frame, loot orbs when looting is on. cv2 releases the GIL, so this runs in
    parallel and the loop just reads the latest results."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.want_orbs = False
        self.tac = (None, None)
        self.orbs, self.orbs_at = [], 0.0
        self.beams = BeamOrbs()
        self.running = True

    def run(self):
        while self.running:
            got = self.cap.read()
            if got is None:
                time.sleep(0.05)
                continue
            frame = got[0]
            try:
                self.tac = find_tactician(frame)
                # orbs at most 4 times a second (79 ms a scan: every frame took a whole core from the loop, run
                # 20261015-161625); the router predicts arrival between scans
                if self.want_orbs and time.time() - self.orbs_at > 0.25:
                    orbs = find_orbs(frame)
                    # gold orbs hidden among our units, found by their standing light beam
                    orbs += [o for o in self.beams.update(find_beams(frame))
                             if all(math.dist(o[:2], q[:2]) > 50 for q in orbs)]
                    self.orbs, self.orbs_at = orbs, time.time()
            except Exception:
                pass
            time.sleep(0.08)                     # ~10 looks a second: the tactician walks, it does not jump


def planner_fixed_comp():
    from tft_player.planner import _fixed_comp
    return _fixed_comp()


def is_carousel(scr):
    """A carousel round: every x-4 from stage 2. Not 1-1: this set opens with an encounter there, no carousel
    (the user, 2026-10-10; run 20261010-064944 ran the carousel loop through it)."""
    r = scr.round
    return bool(r) and r[0] >= 2 and r[1] == 4


def trait_tooltip(text):
    """An item slot's tooltip text whose first line is a trait's name: the trait panel's tooltip, not an item's."""
    first = re.split(r'[\n|/>]', str(text or ''))[0].strip().lower()
    # its first letter can be cut off (run 20261023-010206: "uggernaut" 2-1, "anguard" 4-2)
    return bool(first) and any(t.lower() == first or (len(first) >= 5 and t.lower().endswith(first))
                               for t in CATALOG.traits)


def stage_start_ask(stage):
    """The combat-start question of a stage's last fight (2-7, 3-7, 4-7, ...): the next stage's first phase is planned
    now, as next_round, so its levels and rolls start the moment it opens. Game 17: the first answer of 3-1, 4-1 and 5-1
    took 19, 24 and 25 s (1300-1800 thinking tokens, a stage's biggest decision) of a 30 s phase - the 3-1 rolls, the
    4-1 Diana swap and every 5-1 step (Elise 2 in, Morgana's items) never ran; games 15-16: 18-21 s."""
    s, _, r = str(stage or '').partition('-')
    if not s.isdigit() or int(s) < 2 or r != '7':
        return ''
    nxt = f'{int(s) + 1}-1'
    return (f'\nSTAGE {nxt} NEXT (required in this reply): after strategy, {{"next_round": {{...}}}} for {nxt}, the '
            "stage's first planning phase - its XP (your level line), its roll, the gives that complete items, its "
            f'board by name. It runs the moment {nxt} opens; you are asked again then, with this round\'s loot, while it '
            'runs.')


def comp_auto_buy(units, comp=None):
    """auto_buy's start: the comp's final-board units to 2 stars - its 1-4 costs only. A 5-cost it bought came at
    1 star and sat on the bench at levels 7-8 (game 11 4-1 to 5-1; game 12 5-3, before the model's order taking it
    off arrived); the model adds one by name when a slot is free for it. The climb's settings can start a unit at
    another star level (coach.auto_buy_stars: Elise and Cassiopeia at 3, the comp's win condition - a roll's list
    buy fills each unit to its auto_buy star, so g15 4-7 passed a Cassiopeia copy at auto_buy 2). The comp's own
    trait adds its other champions (comp_rules, request e20: Riftbeast's 7 needs Cinderling, Gromp, Scuttlecrab and
    Mama Beak, which the final board does not list - game 40 rolled past a Mama Beak at 3-6)."""
    from tft_player.coach import auto_buy_stars
    from tft_player.comp_rules import trait_buys
    stars = auto_buy_stars()
    out = {n: stars.get(n, 2) for n in units if CATALOG.unit(n) and CATALOG.unit(n).cost <= 4}
    for n, s in trait_buys(comp).items():
        out[n] = max(out.get(n, 0), stars.get(n, s))
    return out


def set_auto_buy(comps):
    """Flex on the climb's set (coach.flex_set), before a commit (the owner, 2026-11-06): every set comp's auto_buy at
    once, each unit at the highest star level any of them gives it - staying open costs no tempo."""
    out = {}
    for c in comps:
        for n, s in comp_auto_buy(c.units, c.name).items():
            out[n] = max(out.get(n, 0), s)
    return out


class Phase:
    """The stage number changing is the start of planning; within planning the countdown only goes down, so
    once it has been seen running, a rise is the next phase (combat start, 5 -> 0)."""

    def __init__(self):
        self.stage = None
        self.planning = False
        self.last_timer = None
        self.changed = False
        self.candidate = None
        self.running = False    # the planning countdown was seen going down
        self.rise = None        # a countdown rise read once, mid-planning
        self.countdown = False  # the 5-4-3-2-1 before the fight: units stand still, only items can be given
        self.last_t = None      # when last_timer was read (Screen.t)

    @staticmethod
    def _key(stage):
        a, b = stage.split('-')
        return int(a), int(b)

    @staticmethod
    def next_stage(stage):
        a, b = Phase._key(stage)
        return f'{a + 1}-1' if b >= (4 if a == 1 else 7) else f'{a}-{b + 1}'

    def update(self, scr):
        self.changed = False
        if scr.stage and scr.stage != self.stage:
            # rounds come in order: once the stage is known, only the next one can follow (anything else is a
            # misread - a dimmed 3-3 read "5-3" for a whole fight, run 20260927-151256) and counts at its first
            # read; the first stage of a run must be read twice in a row
            if self.stage:
                if scr.stage != self.next_stage(self.stage):
                    return
            elif self.candidate != scr.stage:
                self.candidate = scr.stage
                return
            self.stage, self.planning, self.changed = scr.stage, True, True
            self.running, self.rise, self.candidate, self.countdown = False, None, None, False
            self.last_timer, self.last_t = scr.timer, scr.t or None
            return
        if scr.stage:
            self.candidate = None       # only a read of the current stage cancels a candidate:
                                        # a tick that reads no stage must not (it froze us at 2-2)
        t, lt = scr.timer, self.last_timer
        if t is None:
            return
        if self.countdown and (t == 0 or (lt is not None and t > lt)):
            self.countdown = False      # the fight began (units move, or leave for an away fight)
        gap = scr.t - self.last_t if scr.t and self.last_t else None
        if self.planning and self.running and lt is not None and gap is not None and gap > lt + 1:
            # more time passed since the last read than the countdown had left: planning is over whatever the read
            # (game 56 6-1, log 2560-2565: a 20 s roll-down began at 7 s; the fight's 22 read once as a rise, and the
            # Item Remover was dropped on Murkwolf in the fight - "the items did not come back")
            self.planning, self.rise = False, None
            self.countdown = t <= 8 and gap - lt < 8
            self.last_timer, self.last_t = t, scr.t
            return
        if self.planning and lt is not None and t > lt:
            if not self.running:
                pass                    # the countdown starting (its first value), not a new phase
            elif lt <= 2 or self.rise is not None:
                # the countdown ran out and restarted: combat start (0 -> 5, or 0 -> 2 when a tick read it
                # late - the old "> 4" jump missed that and moved units into the locked board, runs 010345/185805)
                self.planning, self.rise = False, None
                self.countdown = t <= 8         # the countdown before the fight (5 s; augment rounds read 7-8)
            else:
                self.rise = t           # mid-planning: a rise counts on its second read (OCR glitch)
                return
            self.last_timer, self.last_t = t, scr.t or None
            return
        if self.planning and lt is not None and t < lt:
            self.running = True
        self.rise = None
        self.last_timer, self.last_t = t, scr.t or None




def fight_damage(hp, last, stage):
    """A drop from last to hp no bigger than a fight deals at this stage (2 + 6 per stage number)."""
    try:
        s = int(str(stage).split('-')[0])
    except (TypeError, ValueError):
        return False
    return hp < last <= hp + 2 + 6 * s


def hp_log_settle(player, stage, hp):
    """This phase's HP, confirmed after the phase opened: its HP-log entry (made at the opening with the value then)
    and the round's baseline take it - the HP change per round, the streak and the calculator's runway come from
    the log (the checker: the streak was wrong in games 23-25)."""
    log = getattr(player, 'hp_log', None)
    if log and log[-1][0] == stage and log[-1][2] != hp:
        log[-1] = (stage, log[-1][1], hp)
        player.hp_at_round = hp


def late_hp_drop(hp, last, ph):
    """A lost fight's damage that lands after the phase's HP was fixed: a drop no bigger than a fight deals at this
    stage (2 + 6 per stage number), read in a planning phase or in a monster/carousel round's combat - never in a player
    fight's combat, whose own HP drop tells the fight's result. Run 20261021-221735: 53 -> 41 showed 4 s into 4-6, after
    an unchanged 53 had fixed the phase; 41 -> 29 showed in 4-7's monster fight (its choice screen hid the HUD all
    planning phase), so 4-7's prompts said HP 53 at 29 and the player chose to "rebuild gold"."""
    from tft_player.planner import is_pvp
    return fight_damage(hp, last, ph.stage) and (ph.planning or not is_pvp(ph.stage))


def loc_point(loc):
    return B.BENCH[loc[1]] if loc[0] == 'bench' else B.hex_xy(loc[1], loc[2])


FIGHT_BODY = 80      # a fighting unit's body (and its Vanguard shield bubble) centres ~80 px under its health bar
FIGHT_CLEAR = 80     # a drop this close to another of our units' bodies can land on that unit


def fight_crowded(bars, point):
    """Our other units whose body stands within FIGHT_CLEAR of a fight drop at `point` (bars: our health bars, the
    tactician's left out; the one nearest above `point` is the target's own). In a fight the front line bunches, and
    the game gives a dropped item to whichever model or shield bubble is in front at the cursor: run 20261026-052337
    5-1 B.F. Sword for Diana went onto Elise (body 58 px away), 5-5 Adaptive Helm for Morgana onto Diana (69 px).
    Units standing apart are 105+ px from the next body."""
    if not bars or point is None:
        return []
    own = min(bars, key=lambda b: (b[0] - point[0]) ** 2 + (b[1] + 45 - point[1]) ** 2)
    return [b for b in bars if b is not own
            and math.hypot(b[0] - point[0], b[1] + FIGHT_BODY - point[1]) < FIGHT_CLEAR]


class Player:
    fight_goal = None                    # (a Player built without __init__ in a test has none set)
    wisp_flag = None                     # the player's standing Wisp flag (request e184, tft_player/wisp_flag.py)

    def __init__(self, capture, inp, log_dir: Path, until=None, overlay=None, client=None, prep=None):
        """prep: the preparation thread started as we entered the queue (play_match.start_preparation; its
        .result is coach.prepare()'s); without it the briefing goes out now."""
        self.cap, self.inp = capture, inp
        self.reader = ScreenReader()
        B.TALL_HINT = self.tall_hint     # our tall units' bars map to their own hexes (board.BAR_DY)
        from tft_player.ending import EndWatcher
        # separate reader: its OCR runs in another thread
        self.ending = EndWatcher(capture, ScreenReader(), client, log=lambda **e: self.log(**e))
        self.exit_tried = 0.0
        self.reconciled_stage = None     # stage whose board/bench the roster has checked against the screen
        self.round_plan = None           # the model's plan for this planning phase (planner.RoundPlan)
        self.announced = set()           # unit ids already reported to the model as new
        self.item_names, self.item_tries = {}, {}    # item-bench icon signature -> name read from its tooltip
        self.augments_taken = []         # names, in the order picked
        from tft_player import comp_rules as _cr
        _cr.HELD_AUGMENTS = self.augments_taken          # the same list: the comp rules read the augments held
        self.encounter = None            # this game's encounter, as the 1-1 banner says it
        self._enc_busy = False
        self.wisps_bought = {}           # stage -> Wisps bought that round
        self.plan_pending = False        # a plan call is in flight
        self.plan_calls = 0              # plan calls this round (first + updates)
        self.carry_note = ''             # what last round's plan left undone, for the model
        self.plan_asked = None           # stage the plan was requested for
        self.next_plan = None            # (stage made at, plan): the model's next_round, run when the next planning opens
        self.next_plan_names = {}
        self.carousel_stage = None
        self.last_level = 1              # the plan never waits for a level reading
        self.key_asked = set()           # key decision points the model was reminded of
        self.pivoted = set()             # stages where the model changed comp
        self.fail_cause = None           # why the last executed step failed
        self.hp_log = []                 # (stage, hp at start of previous round, hp now)
        self.fight_goal = None           # {'goal': 'win'|'lose', 'since', 'why'}: the result the player plays for
        self.wisp_flag = None            # {'combat', 'economy': buy|skip|ask, 'buy', 'skip': [names]} (request e184)
        self.hp_at_round = None
        self.roster = Roster()
        self.brain = Brain()
        self.pilot = None
        if self.pilot:
            self.brain = self.pilot
            print(f'pilot: {self.pilot.comp.name} -> {", ".join(self.pilot.final)}')
        self.phase = Phase()
        self.live = LiveView()
        self.overlay = overlay
        self.overlay_at = 0.0
        if self.live.port:
            print(f'live view: http://localhost:{self.live.port}')
        from tft_player import coach as _coach
        if _coach.CALL_DIR != log_dir / 'calls':     # not set by coach.new_game before the queue (replays, attach)
            _coach.CALL_DIR = log_dir / 'calls'      # before the briefing thread starts: its inputs are data too
            _coach._CALL_N[0] = 0
        from tft_player.coach import CLAUDE_EFFORT, CLAUDE_MODEL, _cli, warm_up
        warm_up()
        self.comp_p = {}                 # comp name -> P(final board), from the model's last reply
        self.game_strategy = {}          # the model's strategy, section -> text (game_strategy in its replies)
        self.target = None               # comp name: the fixed or committed comp (flex: None until a commit)
        self.pending_pages = None        # (comp name, [jpeg]) to show the model at the next combat review
        self.pages_sent = set()
        from tft_player.coach import forced_comp
        self.forced = forced_comp()      # TFT_FORCE_COMP (tools/play_comp.py): one comp, fixed all game
        self.committed = None            # flex: the stage the model committed to a comp (it is self.forced from then)
        if self.forced:
            self.target = self.forced.name
        # standing orders the model sets (execution_plan.auto_buy / auto_sell), kept until it changes them:
        # auto-buy (champion -> star target) starts as the fixed comp's final board at 2 stars; auto-sell is off
        # (flex on the climb's set: every set comp's units until the commit - set_auto_buy)
        from tft_player.coach import flex_set
        self.flex = None if self.forced else flex_set()
        self.auto_buy = (comp_auto_buy(self.forced.units, self.forced.name) if self.forced else
                         set_auto_buy(self.flex['comps']) if self.flex else {})
        self.auto_sell = False
        self.item_finals = []            # [{'item', 'holder' (unit id), 'final' (champion), 'star'}]: the model's
                                         # final holders (execution_plan.item_final / a give's "final")
        self.give_finals = {}            # (unit id, item given) -> (final, star): set on the item the give makes
        self.handover_rp = None          # items moving to their final holder (it runs as the round plan meanwhile)
        self.auto_done, self.auto_tries = [], {}
        kind, _ = _cli()
        from tft_player.coach import model_name
        print('model:', f'{CLAUDE_MODEL} via {kind} (effort {CLAUDE_EFFORT})'
              if kind in ('claude', 'claude-cli') else model_name() if kind else
              'NONE - set ANTHROPIC_API_KEY to connect Claude; playing on local fallback rules only')
        self.coach = Coach(self.brain, log=lambda m: (print(m), self.log(ev='coach', msg=m)),
                           on_answer=lambda ans: self.update_comps(ans, self.phase.stage),
                           on_call=self.model_call)
        self.augment_seen = None
        self.augment_read = [None] * 3   # pilot mode: the cards' names, read by OCR
        self.surveys = {}                # stage -> surveys done (pilot mode)
        self.need_survey = None          # a reason to read every unit off the screen (pilot mode)
        self._trait_bad = 0
        self._trait_left = ()            # trait differences a survey did not explain (emblems)
        self.checked_stage = None        # the stage whose board was checked against the screen
        self.surveyed_at8 = False
        self.panel_items = []            # items shown by the last unit panel read
        self.pilot_call = None           # pilot mode: the model's current / last judgement call
        self.aug_answer = None           # pilot mode: the model's answer about the augment hand
        self.wisp_answers = {}           # pilot mode: (stage, Wisp) -> 'asked' | True | False
        self.hud_seen = None             # the last screen with the HUD up (the picker hides gold)
        self.bench_objs = {}             # bench slot -> {'kind', 'cls', 'conf'}: non-unit objects seen there
        self._bench_prev = {}
        self.bench_kinds = None          # the last read of all 9 bench slots (bench.classify)
        from tft_player import bench as _bench
        _bench.forget_empty()            # empty-slot pictures are this game's own (learn_bench_baseline)
        self.item_crops = [None] * N_SLOTS    # the item bench's icon crops (to learn a name read off a tooltip)
        self.pending_events = []         # events for the model, sent with the next update
        self.auto_opened = set()         # bench slots whose container the harness opened
        from tft_player.coach import auto_open_anvil
        self.auto_open = auto_open_anvil()   # True: containers are opened by the harness; False: the model decides
        self._auto_tried = set()         # (slot, stage) already tried
        self.news = []                   # changes, failed steps, mismatches: in the next model call
        self.container_opened = None     # (class, slot, time) of the last container we opened
        self.choice = None               # an open choice screen: {'asked', 'pick', 'cards', 'seen'}
        self.augment_clicked = 0.0
        self.augment_default = ('pick', 0)
        self.reroll_stage, self.rerolled = None, set()
        self.augment_note = None         # why the next augment question comes (the last answer could not run)
        self.aug_wait, self.aug_reasks = 0, 0    # coach.finished when the hand was asked; re-asks of the hand
        self.loot = LootRunner()
        self.chaser, self.own = CarouselChaser(), OwnTracker()
        self.bench_items = None
        self.prev_items = None
        self.items_done = set()
        self.tries = {}
        self.round_started = 0.0
        self.status_extra = {}
        self.last_frame = None
        self.tick = 0
        self.watcher = Watcher(capture)
        self.watcher.start()
        self.until = tuple(int(x) for x in until.split('-')) if until else None
        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_f = (log_dir / 'log.jsonl').open('a', encoding='utf-8')
        self.recorder = None
        # the whole game as a video: only with --video (the user, 2026-10-12: off - the carousel clips are what is
        # needed; the full video also cost capture frames)
        if not REPLAY and os.environ.get('TFT_VIDEO') == '1':
            try:
                from tft_player.game import ScreenRecorder
                self.recorder = ScreenRecorder(log_dir / 'game.mp4')
                atexit.register(self.stop_recording)
                print(f'recording: {log_dir / "game.mp4"}', flush=True)
            except Exception as e:
                print(f'RECORDING FAILED: {e!r}', flush=True)
                self.log(ev='recording_error', error=repr(e))
            try:                         # where the loop's time goes (tools/profile_report.py); --profile only: it
                # slowed the game it measured (run 20261010-235723: capture 15 fps, read p50 1.3 s)
                from tft_player.game import start_profiler
                if os.environ.get('TFT_PROFILE') == '1' and start_profiler(log_dir / 'profile.speedscope.json'):
                    print(f'profiling: {log_dir / "profile.speedscope.json"} (written when the game process ends)', flush=True)
            except Exception as e:
                print(f'PROFILER FAILED: {e!r}', flush=True)
                self.log(ev='profiler_error', error=repr(e))
        if self.forced:
            self.set_comp(self.forced.id, 'forced')     # logs, so after log_f
        from tft_player.coach import model_settings, opening_prompt
        model, effort = model_settings(kind)
        self.log(ev='model_session', backend=kind, model=model, effort=effort, system_prompt=opening_prompt())
        _coach.BACKEND_LOG = self.log
        try:                             # which harness played: models are only comparable on the same code
            # (the version loaded with the harness, tools/play_match.load_harness: the working tree may have moved on)
            self.log(ev='run_config', **(CODE_VERSION or code_version()),
                     auto_open_anvil=self.auto_open, test_mode=test_mode(), comp=os.environ.get('TFT_FORCE_COMP'),
                     model=model, effort=effort, round_data=os.environ.get('TFT_ROUND_DATA'),   # (section 9 variants)
                     next_round=os.environ.get('TFT_NEXT_ROUND'), no_interest=os.environ.get('TFT_NO_INTEREST'),
                     speed_check=dict(_coach.SPEED) or None)        # the pre-game decision times (coach.speed_check)
        except Exception as e:
            self.log(ev='run_config_error', error=repr(e))
        if _cli()[0] and not self.pilot:     # after log_f: the answer is logged
            threading.Thread(target=self.send_briefing, args=(prep,), daemon=True).start()
        self.lat = []
        self.ending.start()

    def log(self, **ev):
        ev['t'] = round(ev.pop('at', None) or time.time(), 3)    # at: when it happened, if before it is logged
        self.log_f.write(json.dumps(ev, default=str) + '\n')
        self.log_f.flush()

    def combat_start(self, scr):
        """Combat started: the model is asked as soon as the shop has settled (send_combat_call), and
        the carousel prep (x-3) goes out now."""
        rp = getattr(self, 'round_plan', None)
        self.log(ev='item_delivery', status='locked', stage=scr.stage,
                 requested=(rp.p.get('_requested_items') or []) if rp else [],
                 predicted_orders=(rp.p.get('items') or []) if rp else [],
                 units=[dict(id=u.id, name=u.name, stars=u.star, loc=u.loc, missing=u.missing,
                             unverified=u.unverified,
                             equipped=(list(u.items) if u.items_read and not getattr(u, 'items_predicted', False)
                                       and not getattr(u, 'items_check', False) else None),
                             predicted=list(u.items) if getattr(u, 'items_predicted', False) else None)
                        for u in self.roster.units])
        self.combat_call = (scr.stage, time.time())
        self.prepare_carousel(scr)
        self.bff_rides(scr)
        if self.planner_on() and not REPLAY:
            from tft_player.coach import hold_stage_opening, open_second_conversation
            hold_stage_opening(scr.stage)               # only 4-2: its combat answer precedes the handover
            open_second_conversation(scr.stage, self.log)

    def bff_rides(self, scr):
        """The BFF (Sprykin 3) stands on the board as a unit of its own only until its first rider: a link from it
        to a Sprykin, or the game's own pick when combat starts. From then on it has no hex: it is under its rider,
        a champion on that champion's hex (the user, 2026-10-12). Its hex is kept: the game's pick shows there
        (run 20261012-071645 2-5: Teemo left (0, 3) and rode the BFF at (1, 4))."""
        bff = next((u for u in self.roster.units if u.name == BFF and not u.missing), None)
        if bff is None:
            return
        self.bff_hex = bff.loc
        self.roster.units.remove(bff)
        self.bff_assigned = True
        if not any(getattr(u, 'riding', None) for u in self.roster.units):
            self.news_event('bff', scr.stage, f'combat started: the game gave the BFF (at hex {bff.loc[1:]}) a rider - '
                                              f'one of the fielded Sprykin; the BFF has no hex of its own from now on')

    COMBAT_SHOP_WAIT = 2.0   # s: longest wait for a settled shop before the combat-start call goes anyway
    LAST_FIGHT_WAIT = 20.0   # s: an x-7 fight's call waits this long for a question still in flight

    def send_combat_call(self, scr):
        """The combat-start call, sent on the first combat tick whose shop names every card: when combat
        starts each Wisp leaves the shop and the champion card under it is drawn, and a frame caught in
        between read a champion as a Wisp ('1' for Ivern, 'CV' for Kog'Maw in run 152203). Never blocks:
        until then the loop goes on; after COMBAT_SHOP_WAIT it is sent with whatever is read. The model
        hears what its plan did not get done, may order what works in combat (buy, sell bench units,
        reroll, XP) and writes the next round's plan.
        A stage's last fight (x-7) asks for the next stage's first plan: a question still in flight there delays
        the call instead of dropping it (g18 3-7: the phase's plan, asked with 2 s left after a 7 s choice screen,
        was answered 3 s into the fight; no combat question went out and 4-1 had no next_round)."""
        from tft_player.coach import is_handover_round, release_stage_opening
        stage, t0 = self.combat_call
        last = bool(stage_start_ask(stage)) or is_handover_round(stage)
        if self.phase.planning or (scr.stage and scr.stage != stage):
            self.combat_call = None      # the fight is over already
            release_stage_opening(stage)
            return
        settled = scr.shop_visible and all(n or not r for n, r in zip(scr.shop, scr.shop_raw))
        if not settled and time.time() - t0 < self.COMBAT_SHOP_WAIT:
            return
        if (last and self.plan_pending and self.planner_on() and not self.pilot
                and time.time() - t0 < self.LAST_FIGHT_WAIT):
            return                       # asked once the question in flight is answered
        self.combat_call = None
        if not self.planner_on() or self.pilot or self.plan_pending:
            release_stage_opening(stage)
            return                       # a question still running is answered in combat anyway
        rp = self.round_plan
        left = rp.leftover(self.roster)[1] if rp else ''
        orbs = len(getattr(self.watcher, 'orbs', None) or [])
        change = ('COMBAT STARTED (what the planning phase did not get done is listed below as "Not done")'
                  + (f"; {orbs} loot orb(s) still on the board, not collected" if orbs else ''))
        self.log(ev='combat_start_call', stage=scr.stage, left=left, orbs=orbs, settled=settled,
                 waited=round(time.time() - t0, 2))
        self.request_plan(scr, self.last_frame, 'update', change, interrupt=False)

    def prepare_carousel(self, scr):
        """The fight before a carousel (x-3): the model ranks every item it wants, in the background, so the
        chase starts on its choice the moment the carousel opens (the user: never ask once it is up)."""
        rnd = scr.round or (0, 0)
        if rnd[0] < 2 or rnd[1] != 3 or getattr(self, '_prepared_for', None) == (rnd[0], 4):
            return
        from tft_player import carousel as CR
        from tft_player.coach import ask_model
        self._prepared_for = (rnd[0], 4)
        target_stage = f'{rnd[0]}-4'
        targets = list(self.brain.plan.item_targets or [])
        state = (f"HP {getattr(self, '_hp', None)}, {self.roster!r}, item bench: "
                 f"{[i for i in (self.bench_items or []) if i] or 'empty'}")
        prompt = CR.prep_prompt(state)
        image = self.screen_jpeg()
        self.log(ev='carousel_prep_asked', stage=scr.stage, for_stage=target_stage)

        def run():
            t0 = time.time()
            got = {}

            def first(obj):
                r = CR.parse_prep(obj)
                if r and not got:
                    got['ranked'], got['latency'] = r, time.time() - t0
                    self.carousel_prep = {'stage': target_stage, 'ranked': r}
                    self.brain.plan.item_targets = r          # the chase's own wants follow it at once
                    self.log(ev='item_targets', stage=scr.stage, targets=r, asked='carousel_prep')
            try:
                ans = ask_model(prompt, image, on_first=first, timeout=60)
                if not got:
                    from tft_player.coach import json_objects
                    for obj in json_objects(ans or ''):
                        first(obj)
                self.model_call('carousel_prep', prompt, ans, time.time() - t0, got.get('latency'))
            except Exception as e:
                self.log(ev='carousel_prep_error', error=repr(e))
        threading.Thread(target=run, daemon=True).start()

    def fill_items_in_fight(self, scr):
        """The item bench is read in planning only: in the fight (the countdown's gives, the fight's gives) the tracked
        bench stands in - the last reading, with our own gives taken off (benchmark game 20261021-013833 3-1: unread,
        the plan's gives found nothing, failed as "not on the item bench (it holds: nothing)" and the model was told
        so, while the bench held six items)."""
        if scr.items is None:
            names = list(self.item_track.names())
            scr.items = names if any(names) else (list(getattr(self, 'bench_items', None) or []) or None)

    def combat_decide(self, scr):
        """The fight is running: the board is locked, but buying, selling bench units, rerolling and XP
        work - the plan's steps of those kinds go on now (the user: they still work in combat); board
        moves wait for the next planning phase. Wisps leave the shop when combat starts. In the countdown before
        the fight (5-4-3-2-1) the units stand still: the plan's items not given yet go first (in an away fight the
        units leave at 2; a drop after that lands on no unit of ours and the item returns to the bench)."""
        Player.observe_wisp_shop(self, scr, expire=True)
        rp = self.round_plan
        self.fill_items_in_fight(scr)
        if rp is not None and self.phase.countdown and scr.shop_visible and (scr.timer or 0) >= 2:
            self.prepare_comp(scr, rp, combat=True)
            act = rp.countdown_action(scr, self.roster)
            if act:
                self.brain.status['plan'] = f'COUNTDOWN ACT {act}'
                return act
        act = self.after_fight_orders(scr)
        if act:
            return act
        if not scr.shop_visible or not scr.stage or scr.augments:
            return None
        act = self.auto_action(scr)
        if act:
            return act
        if rp is None:
            return None
        if self.pending_events and not self.plan_pending and self.planner_on() and not self.pilot                 and getattr(self, 'combat_call', None) is None:
            # an event in combat wakes the model too (what works in combat is done at once)
            ev = '; '.join(self.pending_events)
            self.log(ev='event_sent', stage=scr.stage, what=ev)
            self.request_plan(scr, self.last_frame, 'update', ev, interrupt=False)
        if scr.level:
            self.last_level = scr.level
        rp.hold_roll = False                     # no Wisps in combat
        self.prepare_comp(scr, rp, combat=True)
        act = rp.next_action(scr, self.roster, self.team_cap(scr), combat=True)
        self.tell_plan_notes(rp, scr)
        if act:
            self.brain.status['plan'] = f'COMBAT ACT {act}'
            return act
        if not self.phase.countdown and (scr.timer or 0) >= 3 and time.time() >= getattr(self, 'fight_give_wait', 0):
            unfound = getattr(self, 'fight_unfound', None)
            rp.fight_unfound = unfound[1] if unfound and unfound[0] == scr.stage else set()
            act = rp.fight_give(scr, self.roster)
            if act:
                self.brain.status['plan'] = f'FIGHT GIVE {act}'
                return act
        if rp.failed and not rp.carry:
            self.plan_idle(rp, scr, combat=True)
        return None

    def fight_result(self, scr):
        """'win' / 'lose' once this fight's result shows, else None: the streak flame's number rises or
        its colour flips (orange = win, blue = loss), or our HP drops. Read at most every 0.4 s."""
        base = getattr(self, 'fight_base', None)
        if not base or base['stage'] != scr.stage or self.last_frame is None:
            return None
        now = time.time()
        if now - base.get('checked', 0) < 0.4:
            return base.get('result')
        base['checked'] = now
        if base.get('result'):
            return base['result']
        st, st0, hp0 = self.reader.streak(self.last_frame), base['streak'], base['hp']
        res = None
        if scr.hp is not None and hp0 is not None and scr.hp < hp0:
            res = 'lose'
        elif st and st[1] in ('win', 'loss') and (st0 is None or st0[1] != st[1] or st[0] > st0[0]):
            res = 'win' if st[1] == 'win' else 'lose'
        elif self.win_gold(scr, base):
            res = 'win'
        if res:
            base['result'] = res
            self.log(ev='fight_result', stage=scr.stage, result=res, streak=st, streak0=st0, hp=scr.hp, hp0=hp0,
                     goal=(getattr(self, 'fight_goal', None) or {}).get('goal'))
        return res

    def win_gold(self, scr, base):
        """A player fight won: the gold rises by exactly 1 with no sell or buy of ours (the win gold),
        read twice in a row. The streak flame misses a win that ends a loss streak."""
        from tft_player.planner import is_pvp
        g = scr.gold
        if g is None or not is_pvp(scr.stage):
            return False
        if time.time() - getattr(self, '_acted_at', 0) < 1.5:
            base['gold_ref'], base['gold_up'] = None, 0       # our own sell/buy: a new baseline after it
            return False
        ref = base.get('gold_ref')
        if ref is None or (g != ref and g != ref + 1):
            base['gold_ref'], base['gold_up'] = g, 0
            return False
        base['gold_up'] = base.get('gold_up', 0) + 1 if g == ref + 1 else 0
        return base['gold_up'] >= 2

    def after_fight_orders(self, scr):
        """The model's orders for after this fight ({"win": {"sell": [ids], "buy": [names]}, "lose": ...}),
        one step per tick, the moment the result shows: interest is paid on the gold held when the next
        round starts, so a sell or buy made now counts. Sells (bench units) first, then buys from the
        shop on screen, each card once; what could not be done is told with the reason."""
        af = getattr(self, 'after_fight', None)
        if not af or af['stage'] != scr.stage or af.get('done'):
            return None
        res = self.fight_result(scr)
        if not res:
            return None
        o = af.get(res)
        o = o if isinstance(o, dict) else {'sell': list(o or []), 'buy': []}      # the older form: sells only
        sold, said = af.setdefault('sold', set()), af.setdefault('names', [])
        # by the unit's own id only: an id that merged away would sell the merged unit (run 20261022-010349 5-2)
        todo = [next((u for u in self.roster.units if u.id == i), None) for i in o.get('sell') or [] if isinstance(i, int)]
        todo = [u for u in todo if u is not None and u.loc[0] == 'bench' and not u.missing and u.id not in sold]
        if todo:
            u = todo[0]
            sold.add(u.id)                       # tried once here; 'sold' is said by after_fight_outcome once it lands
            return Action('sell', u, f'after-fight sell ({res}) {u}')
        if time.time() - getattr(self, '_acted_at', 0) < 0.6:
            return None                          # the gold of the last sell/buy is read first
        k = af.setdefault('bought', 0)
        buys = list(o.get('buy') or [])
        if k < len(buys) and scr.shop_visible and scr.gold is not None:
            af['bought'] = k + 1
            name = buys[k]
            slot = next((i for i, n in enumerate(scr.shop) if n == name and i not in af.setdefault('slots', set())), None)
            info = CATALOG.unit(name)
            why = ('not in the shop' if slot is None else f'{info.cost}g, gold {scr.gold}' if info.cost > scr.gold
                   else 'no bench slot' if not self.roster.free_bench() and len(self.roster.copies(name, 1)) < 2
                   else None)
            if why:
                said.append(f'not bought {name} ({why})')
                return None
            af['slots'].add(slot)
            said.append(f'bought {name}')
            return Action('buy', slot, f'after-fight buy ({res}) {name}')
        if k < len(buys):
            return None                          # the shop or gold not read on this tick
        af['done'] = res
        self.news_event('after_fight', scr.stage, f'fight result: {res}; after-fight orders: '
                                                  f'{"; ".join(said) or "none"}')
        return None

    def after_fight_outcome(self, act, ok, deferred=False):
        """What an after-fight sell did, told as it is: 'sold' only once the sell went through. A sell the fight would
        not take (an away fight, its first or last second) is carried to the next planning phase and sold as it opens.
        Game 48 (run 20261030-143543) 3-1..3-3: the away fight and the fight's last second deferred the sells of Karma,
        Veigar and Akali, the player was told "sold" three rounds running, and the 9/9 bench missed Mama Beak at 3-5."""
        if not act.why.startswith('after-fight sell'):
            return
        u, cause = act.arg, self.fail_cause or '?'
        carry = getattr(self, 'after_fight_carry', None)
        if '(carried' in act.why:                # the carried sell, in the next planning phase
            if deferred:
                return                           # still to do: tried again on the next tick
            if carry and u.id in carry['ids']:
                carry['ids'].remove(u.id)
            self.news_event('after_fight', carry.get('at') if carry else None,
                            f'after-fight order carried to this planning phase: '
                            f'{f"sold {u}" if ok else f"not sold {u} ({cause})"}')
            return
        af = getattr(self, 'after_fight', None)
        said = af.setdefault('names', []) if af else []
        if ok:
            said.append(f'sold {u}')
        elif deferred:
            if not carry or carry.get('from') != (af or {}).get('stage'):
                carry = self.after_fight_carry = {'from': (af or {}).get('stage'), 'ids': []}
            if u.id not in carry['ids']:
                carry['ids'].append(u.id)
            said.append(f'not sold yet {u} ({cause.removeprefix("queued: ")}): it is sold as the next planning '
                        f'phase opens')
        else:
            said.append(f'not sold {u} ({cause})')

    def carried_sell(self, scr):
        """The next after-fight sell the fight would not take (after_fight_outcome), at the start of the planning phase
        after it - unless the plan now fields or keeps that unit, or the phase is ending. -> Action or None."""
        carry = getattr(self, 'after_fight_carry', None)
        if not carry or not carry['ids'] or carry.get('from') == scr.stage:
            return None
        rp = self.round_plan
        keep = (rp.protected_ids() if rp else set()) | set(getattr(self.brain, 'protected', None) or ())
        for i in list(carry['ids']):
            u = next((x for x in self.roster.units if x.id == i), None)
            why = ('it is no longer ours' if u is None else 'it is not on the bench' if u.loc[0] != 'bench'
                   else 'it is set aside as missing' if u.missing else 'your plan now keeps it' if i in keep
                   else 'the planning phase is ending' if scr.timer is not None and scr.timer <= 3 else None)
            if why:
                carry['ids'].remove(i)
                if u is not None:
                    self.news_event('after_fight', scr.stage, f'after-fight order carried from {carry["from"]}: '
                                                              f'not sold {u} ({why})')
                continue
            carry['at'] = scr.stage
            return Action('sell', u, f'after-fight sell (carried from {carry["from"]}) {u}')
        return None

    # -- standing orders: auto-buy / auto-sell -------------------------------------------------
    def auto_need(self, name, target=None):
        """Copies of `name` still to buy for a star target (default: its auto_buy one), counted over board
        and bench: 3 copies make a 2-star, 9 a 3-star - a 1-star on the board and two on the bench are
        already a 2-star's worth, whether or not the merge was seen."""
        t = target or self.auto_buy.get(name, 0)
        from tft_player.comp_rules import rules
        ceiling = (rules().get('upgrade_stars') or {}).get(name)
        if ceiling and t:
            t = min(t, ceiling)
        if not t:
            return 0
        have = self.roster.proven_copies(name)
        have += self.roster.pending(name)
        return max(0, 3 ** (t - 1) - have)

    def outside_comp_sale(self, scr):
        """Owner e142: clear the opening bench from stage 3; all MetaTFT boards and planned units stay."""
        from tft_player.coach import forced_comp
        comp = forced_comp()
        stage = getattr(scr, 'stage', None)
        if (comp is None or not stage or int(stage.split('-')[0]) < 3
                or not getattr(scr, 'shop_visible', False) or getattr(scr, 'augments', None)
                or (scr.timer is not None and scr.timer <= 2)):
            return None
        # Drain the opening bench and incoming rewards one verified action per tick.
        if getattr(self, '_comp_cleanup_stage', None) != stage:
            self._comp_cleanup_stage = stage
            self._comp_cleanup_ids = {u.id for u in self.roster.bench()}
        self._comp_cleanup_ids.update(u.id for u in self.roster.bench() if getattr(u, 'arrival_source', None))
        want = set(comp.units) | set(getattr(getattr(self.brain, 'plan', None), 'units', ()) or ())
        for section in ('early', 'final'):
            for boards in (comp.guide.get(section) or {}).values():
                for board in boards:
                    want.update(board.get('units') or ())
        from tft_player.comp_rules import TANK_TRAITS, eighth_units
        for lv in (8, 9):
            want.update(eighth_units(comp.name, lv))
        # a unit on none of the comp's boards and with no tank trait is no candidate for this comp's board at all: it
        # is sold as soon as it is benched, whatever a plan says of it (the owner, game 78 x28.2: "the bench cait still
        # not sold, which has 2 items wasted. It is not in the comp, it should be sold immediately, does not need the
        # model decision"; x28.4: "field brawler / juggernaut etc. to enable tanky traits")
        useless = {u.id for u in self.roster.bench() if u.name and CATALOG.unit(u.name) and u.name not in want
                   and not set(CATALOG.unit(u.name).traits) & set(TANK_TRAITS)}
        self._comp_cleanup_ids |= useless
        rp = getattr(self, 'round_plan', None)
        keep = set(getattr(self.brain, 'protected', ()) or ())
        if rp:
            keep |= rp.protected_ids()

            def planned(value):
                if isinstance(value, str):
                    want.add(value)
                elif isinstance(value, int):
                    keep.add(value)
                elif isinstance(value, list):
                    for v in value:
                        planned(v)
                elif isinstance(value, dict):
                    for k in ('id', 'name', 'unit', 'on', 'to', 'when_owned', 'buy', 'keep'):
                        if k in value:
                            planned(value[k])
                    want.update(k for k in value if CATALOG.unit(k))

            for key in ('board', 'bench', 'keep', 'items', 'link', 'buy', 'roll_n', 'fast_rr', 'roll'):
                planned(rp.p.get(key))
        want.update(u.name for u in self.roster.units if u.id in keep and u.name)
        want |= getattr(self, 'named_next', None) or set()     # the last reply's "next" and notes
        free_items = sum(not it for it in (getattr(self, 'bench_items', None) or ()))
        tried = self.__dict__.setdefault('_comp_cleanup_tries', {})
        for u in self.roster.bench():
            key = (stage, u.id)
            if (u.id not in self._comp_cleanup_ids or (u.id not in useless and (u.id in keep or u.name in want))
                    or (u.star >= 2 and getattr(u, 'arrival_source', None))
                    or u.missing or u.unverified
                    or len(u.items) > free_items or tried.get(key, 0) >= 2):
                continue
            if not u.name:
                from tft_player import stars
                from tft_player.items import read_unit_items
                frame = getattr(self, 'last_frame', None)
                bar = self.unit_bar(frame, u.loc) if frame is not None else None
                if bar is None or stars.read_star(frame, bar) != 1 or u.items or read_unit_items(frame, bar):
                    continue
                if not self.outside_comp_crop(u, want, frame, bar):
                    continue
            elif not CATALOG.unit(u.name):
                continue
            tried[key] = tried.get(key, 0) + 1
            return Action('sell', u, f'comp cleanup: {u} is on none of the comp boards or planned units')
        return None

    def outside_comp_crop(self, unit, wanted, frame, bar):
        """Exclude an unnamed reward by crop without blocking actions on gallery/model loading."""
        from tft_player import champs
        key = (unit.loc, frozenset(wanted))
        result = getattr(unit, 'comp_crop_result', None)
        if result is not None and result[:2] == key:
            return result[2] is False
        if getattr(self, '_comp_crop_busy', False):
            return False
        crop = champs.unit_crop(frame, bar)
        if crop is None:
            return False
        self._comp_crop_busy = True

        def compare():
            try:
                vec = champs.embed([crop])[0]
                unit.comp_crop_result = (*key, champs.matches_any(vec, wanted))
            except Exception as error:
                unit.comp_crop_result = (*key, None)
                self.log(ev='comp_crop_error', unit_id=unit.id, error=repr(error))
            finally:
                self._comp_crop_busy = False

        threading.Thread(target=compare, daemon=True).start()
        return False

    def auto_sell_unit(self):
        """The next bench unit auto-sell sells, or None. auto_sell True: a champion not on auto_buy, or a
        spare copy beyond its star target; a list: only those champions. Never a board unit, a 2- or 3-star unit, a
        unit the plan fields or keeps, or one whose identity is in doubt; items come back only to free bench slots."""
        if not self.auto_sell:
            return None
        rp = self.round_plan
        keep = (rp.protected_ids() if rp else set()) | set(getattr(self.brain, 'protected', None) or ())
        from tft_player.comp_rules import plan_targets, upgrade_keep
        keep |= upgrade_keep(self.roster, plan_targets(getattr(rp, 'p', None)))
        free_items = sum(1 for i in self.bench_items if not i) if self.bench_items is not None else 0
        for u in self.roster.bench():
            if u.missing or u.unverified or not u.name or u.id in keep or len(u.items) > free_items:
                continue
            if u.star >= 2:
                # a starred unit leaves only by the model's own sell (the climb's game 27 6-1: a swap benched Rakan 3
                # for Taric and auto-sell sold him as "not on auto_buy"; the next two fights cost 34 HP)
                continue
            if isinstance(self.auto_sell, list):
                if u.name in self.auto_sell:
                    return u
                continue
            t = self.auto_buy.get(u.name, 0)
            if not t:
                return u
            same = [x for x in self.roster.units if x.name == u.name and not x.missing and x is not u]
            if any(getattr(x, 'star_assumed', False) for x in same):
                continue                         # a merge not yet seen on screen: its copies may still be needed
            others = sum(3 ** (x.star - 1) for x in same)
            if others >= 3 ** (t - 1):
                return u                         # the target is met without it
        return None

    def optional_buy_ok(self, name, gold, cap, target=1, roster=None):
        """A one-copy level-add fills a live job and leaves funds for the carry hit."""
        from tft_player.comp_rules import eighth_units, optional_room, rules
        comp = os.environ.get('TFT_FORCE_COMP') or getattr(getattr(self, 'forced', None), 'name', None)
        if target > 1 or name not in eighth_units(comp, cap or 8):
            return True
        roster = roster or self.roster
        # (no room check: every MetaTFT level-8 unit is bought when the shop shows it - the owner, game 81 x31.5:
        # "autobuy should be enabled for all those suitable units, sett, taric, gnar etc." - the board picks among them)
        carry = (rules(comp).get('carry') or (None,))[0]
        info = CATALOG.unit(carry)
        star = (rules(comp).get('upgrade_stars') or {}).get(carry, 1)
        need = max(0, 3 ** (star - 1) - roster.proven_copies(carry) - roster.pending(carry))
        return gold is not None and gold - CATALOG.unit(name).cost >= (info.cost * need if info else 0)

    def level_adds_buy(self, scr):
        """From level 8, auto_buy takes one copy of each unit MetaTFT's boards of this level add beyond the comp's
        trait (comp_rules.eighth_units: Taric, Gnar - and Sett from level 9), so the shop and every roll buy them
        with no model call (the owner, game 72, x24.1: "tell the model what to buy ... it doesn't even need the
        model"). Told in the next state once."""
        from tft_player.comp_rules import eighth_units, optional_room, rules
        name = getattr(getattr(self, 'forced', None), 'name', None) or os.environ.get('TFT_FORCE_COMP')
        r = rules(name).get('eighth')
        if not r or not isinstance(scr.level, int) or scr.level < r['level']:
            return
        cap = Player.team_cap(self, scr)
        # each add by its own room: one ranked higher by MetaTFT is ordered even while a lower one holds the slot (the
        # owner, game 81 4-7, x31.5: "there is clearly taric, why not bought? it should replace the sett")
        added = [u for u in eighth_units(name, scr.level) if not self.auto_buy.get(u)]
        for u in added:
            self.auto_buy[u] = 1
        # a 4-cost add we own is bought on to 2-star (the owner, game 79 x30.15: "it did not buy extra copy of sett, it
        # should be bought and make it two star")
        for u in eighth_units(name, scr.level):
            info = CATALOG.unit(u)
            if info and info.cost <= 4 and self.auto_buy.get(u) == 1 and self.roster.proven_copies(u) >= 1:
                self.auto_buy[u] = 2
                self.news_event('auto_orders', scr.stage, f'auto_buy takes {u} to 2-star (a fielded level-8 unit)')
        if added:
            self.news_event('auto_orders', scr.stage, f"level {scr.level}: auto_buy takes {', '.join(added)} (one each) - "
                            f"the units MetaTFT's level-{min(scr.level, 9)} boards add beyond Riftbeast 7")

    LEVEL_KEEP = 50                          # gold that earns full interest

    def executor_level(self, scr):
        """The next level, bought by the executor with no model decision at the start of a planning phase:
        - when its XP leaves 50 gold or more - no interest is lost (the owner, game 88 3-6, x32.3: "has enough gold for
          level 7 without affecting interest, should level up automatically. Instead it is late (level up after combat
          started, same problem as the 3-3)" - 85 gold at level 6, the level came with 2 s left); up to the comp's roll
          level (8 for a level-8 comp: past it the plan decides). A level that costs interest is the model's decision
          (the owner, x32.8 revised: "do not do this, let model decides" - comp rule owner-hp-paths).
        Not while the round plan's own XP is still to come, with an augment open or in the last LEVEL_LATE_S seconds; once
        per level and round. A plan written from the state before it aims its XP at the level the model saw
        (planner.aim_xp), so it is not bought twice. -> the XP buys, or None."""
        from tft_player import planner
        from tft_player.coach import forced_comp
        from tft_player.comp_rules import roll_level
        xp, comp = getattr(scr, 'xp', None), forced_comp()
        if (comp is None or REPLAY or scr.augments or not isinstance(scr.gold, int) or not isinstance(scr.level, int)
                or not isinstance(xp, (tuple, list)) or len(xp) != 2 or not all(isinstance(v, int) for v in xp)
                or not xp[1] or scr.timer is None or scr.timer <= planner.RoundPlan.LEVEL_LATE_S):
            return None
        rp = self.round_plan
        if rp is not None and rp.stage == scr.stage and int(rp.p.get('level_ups') or 0) > rp.levels_done:
            return None                          # the plan's own XP comes first
        if scr.level >= (roll_level() or 8):
            return None
        per = getattr(scr, 'xp_per_buy', None) or 4
        n = -(-(xp[1] - xp[0]) // per)
        if n <= 0 or per * n > scr.gold or scr.gold - 4 * n < self.LEVEL_KEEP:
            return None
        tried = self.__dict__.setdefault('executor_levels', set())
        if (scr.stage, scr.level) in tried:
            return None
        tried.add((scr.stage, scr.level))
        why = f'{4 * n} XP gold of {scr.gold} keeps {scr.gold - 4 * n} - full interest (the owner, x32.3)'
        self.news_event('comp rule', scr.stage, f'the executor bought level {scr.level + 1} at the phase start: {why}')
        self._executor_level_t = time.time()     # a reply asked before it aims its XP at the level it saw (aim_xp)
        return Action('level', n, f'executor: level {scr.level + 1} - {why}')

    def auto_action(self, scr):
        """One auto-sell or auto-buy step on this shop, or None. No gold reserve, no question: the model set
        these orders and changes them in any reply. Each (round, slot, champion) is tried twice at most."""
        if not scr.shop_visible or scr.gold is None or scr.augments or REPLAY:
            return None
        if scr.timer is not None and scr.timer <= 1:
            return None                          # the phase is switching (run 033507 2-2: a buy at 0 s never landed)
        Player.level_adds_buy(self, scr)
        u = self.auto_sell_unit()
        if u is not None:
            key = (scr.stage, 'sell', u.id)
            if self.auto_tries.get(key, 0) < 2:
                self.auto_tries[key] = self.auto_tries.get(key, 0) + 1
                return Action('sell', u, f'auto-sell {u}')
        for i, name in enumerate(scr.shop):
            info = CATALOG.unit(name) if name else None
            if info is None or info.cost > scr.gold or self.auto_need(name) <= 0:
                continue
            if not Player.optional_buy_ok(self, name, scr.gold, Player.team_cap(self, scr),
                                          self.auto_buy.get(name, 1)):
                continue
            if not self.roster.buy_has_room(name):
                # no room (a 3rd copy would merge and needs none) - said once per stage and champion: runs
                # 20261018-092528 and -102854 kept a full bench for rounds while their comp units passed in the shop
                said = self.__dict__.setdefault('auto_blocked', set())
                if (scr.stage, name) not in said:
                    said.add((scr.stage, name))
                    self.news_event('auto_buy_blocked', scr.stage, f'auto_buy did not buy {name} (shop slot {i}): '
                                                                  f'the bench is full')
                continue
            key = (scr.stage, i, name)
            if self.auto_tries.get(key, 0) >= 2:
                continue
            self.auto_tries[key] = self.auto_tries.get(key, 0) + 1
            a = Action('buy', i, f'auto-buy {name}')
            a.more = self.auto_burst(scr, i, name)
            return a
        return None

    def auto_burst(self, scr, first, name0):
        """The other auto-buy champions on this shop, bought in the same burst as the first (the owner, x18.9: game 66
        3-5 took 20 s to buy its wanted Riftbeasts one by one - "this should be done like in 1 second"). Within the
        gold, the bench room (a third 1-star copy merges and needs none) and each champion's copies still wanted.
        -> [(slot, name), ...]"""
        info0 = CATALOG.unit(name0)
        gold = scr.gold - (info0.cost if info0 else 0)
        predicted = copy.deepcopy(self.roster)
        predicted.bought(name0)
        need = Counter({name0: self.auto_need(name0) - 1})
        out = []
        for j, name in enumerate(scr.shop):
            info = CATALOG.unit(name) if name else None
            if j == first or info is None or info.cost > gold:
                continue
            if name not in need:
                need[name] = self.auto_need(name)
            if need[name] <= 0 or not predicted.buy_has_room(name):
                continue
            if not Player.optional_buy_ok(self, name, gold, Player.team_cap(self, scr),
                                          self.auto_buy.get(name, 1), predicted):
                continue
            key = (scr.stage, j, name)
            if self.auto_tries.get(key, 0) >= 2:
                continue
            self.auto_tries[key] = self.auto_tries.get(key, 0) + 1
            out.append((j, name))
            gold -= info.cost
            need[name] -= 1
            predicted.bought(name)
        return out

    def apply_auto(self, plan, stage):
        """execution_plan.auto_buy / auto_sell from the model: merged into the standing orders; its
        after_fight_sell is armed for this stage's fight."""
        af = plan.get('after_fight')
        if not isinstance(af, dict) and isinstance(plan.get('after_fight_sell'), dict):
            af = {k: {'sell': v} for k, v in plan['after_fight_sell'].items()}
        if isinstance(af, dict):
            orders = lambda o: ({'sell': [i for i in o.get('sell') or [] if isinstance(i, int)],
                                 'buy': [str(n) for n in o.get('buy') or [] if CATALOG.unit(str(n))]}
                                if isinstance(o, dict) else {'sell': [i for i in o or [] if isinstance(i, int)], 'buy': []})
            self.after_fight = {'stage': stage, 'win': orders(af.get('win')), 'lose': orders(af.get('lose'))}
            self.log(ev='after_fight_armed', stage=stage, win=self.after_fight['win'], lose=self.after_fight['lose'])
        ab, as_ = plan.get('auto_buy'), plan.get('auto_sell')
        bad = []
        if isinstance(ab, dict):
            for n, star in ab.items():
                info = CATALOG.unit(str(n))
                if not info or not isinstance(star, int) or isinstance(star, bool) or not 0 <= star <= 3:
                    bad.append(f'auto_buy {n!r}: {star!r} (a champion name -> star 0-3)')
                elif star == 0:
                    self.auto_buy.pop(info.name, None)
                else:
                    self.auto_buy[info.name] = star
        elif ab is not None:
            bad.append('auto_buy must be {"Champion": star, ...}')
        if isinstance(as_, bool):
            self.auto_sell = as_
        elif isinstance(as_, list):
            names = [CATALOG.unit(str(n)).name for n in as_ if CATALOG.unit(str(n))]
            bad += [f'auto_sell: no champion named {n!r}' for n in as_ if not CATALOG.unit(str(n))]
            self.auto_sell = names or False
        elif as_ is not None:
            bad.append('auto_sell must be true, false or ["Champion", ...]')
        ho = plan.get('item_final')
        if isinstance(ho, list):
            for o in ho:
                uid, item = (o.get('unit'), o.get('item')) if isinstance(o, dict) else (None, None)
                final = o.get('final') if isinstance(o, dict) else None
                info = CATALOG.unit(str(final)) if final else None
                if not isinstance(uid, int) or not item or (final and not info):
                    bad.append(f'item_final {o!r}: needs "unit" (an id), "item" and "final" (a champion, or null)')
                    continue
                self.set_final(uid, str(item), info.name if info else None, o.get('star'))
        elif ho is not None:
            bad.append('item_final must be a list of {"unit", "item", "final", "star"}')
        for e in plan.get('items') or []:
            if isinstance(e, dict) and e.get('final') and isinstance(e.get('id'), int) and CATALOG.unit(str(e['final'])):
                self.give_finals[(e['id'], str(e.get('component')))] = (CATALOG.unit(str(e['final'])).name,
                                                                      e.get('final_star'))
        if ab is not None or as_ is not None or ho is not None:
            self.log(ev='auto_orders', stage=stage, auto_buy=self.auto_buy, auto_sell=self.auto_sell,
                     item_finals=self.item_finals)
        if bad:
            self.news_event('auto_rejected', stage, '; '.join(bad))

    def set_final(self, uid, item, final, star=None):
        """The model's final holder for an item a unit holds (None clears it)."""
        self.item_finals = [r for r in self.item_finals if not (r['holder'] == uid and r['item'] == item)]
        if final:
            self.item_finals.append({'item': item, 'holder': uid, 'final': final,
                                     'star': star if isinstance(star, int) and 1 <= star <= 3 else 1})

    def note_final(self, unit, given, made):
        """A give landed on `unit`: `given` from the item bench, making `made` (the same item, or the item a component
        completed). The item records follow it: an item whose holder no longer has it (taken off by a Remover, or its
        holder sold) has a new current holder now, with the same final holder; a component this unit held is part of
        `made` now; the final holder the model named with this give goes onto `made`. An item at its final holder is
        where it was meant to be: its record ends."""
        if not hasattr(self, 'item_finals'):
            return
        f = self.give_finals.pop((unit.id, given), None)
        held = unit.items or []
        for r in self.item_finals:
            h = self.roster.resolve(r['holder'])
            if r['holder'] != unit.id and r['item'] in (given, made) and (h is None or r['item'] not in (h.items or [])):
                r['holder'] = unit.id            # moved here: a transfer, or a sold holder's item given out again
                f = f or (r['final'], r['star'])
                r['item'] = made
            elif r['holder'] == unit.id and r['item'] != made and r['item'] not in held:
                f = f or (r['final'], r['star'])                 # the component it held is part of `made` now
        self.item_finals = [r for r in self.item_finals if r['holder'] != unit.id or r['item'] in held]
        if f:
            self.set_final(unit.id, made, f[0], f[1])
        from tft_player import item_plan
        for r in self.item_finals:               # at its final holder: the record ends, and the comp's carry keeps it his
            if getattr(self.roster.resolve(r['holder']), 'name', None) == r['final'] and r['item'] not in COMPONENTS:
                home = item_plan.HOME.setdefault(r['final'], [])
                if r['item'] not in home:
                    home.append(r['item'])
        self.item_finals = [r for r in self.item_finals
                            if getattr(self.roster.resolve(r['holder']), 'name', None) != r['final']]

    def comp_units(self):
        """The comp's final board, by champion name (the fixed comp, else the target comp), or None."""
        from tft_player.meta import META
        c = self.forced or (META.by_name(self.target) if getattr(self, 'target', None) else None)
        return set(c.units) if c else None

    @staticmethod
    def unit_worth(name, star):
        """A unit's gold worth: its cost for each 1-star copy (3 a star), 0 for an unknown one."""
        info = CATALOG.unit(str(name)) if name else None
        return info.cost * 3 ** (max(1, star or 1) - 1) if info else 0

    def final_star(self, r, holder=None):
        """The star an item record's final holder needs before the item moves: the model's "star", and at least the
        gold worth of the unit holding it now (a final holder worth less waits for its next star). Run 20261022-054806
        4-6: Warmog's Armor went from Diana 2 (worth 9) to Sentinel 1 (worth 4), Elise 2's Chain Vest too; HP 70 -> 47
        over 4-5 and 4-6. A 2-star holder gives its items only to a unit worth more: run 20261022-064841 4-2 swapped
        Camille 2 (worth 3) and her Spear off the board for a 1-star Cassiopeia (worth 3); the sells failed and the
        Spear stayed benched to 4-5."""
        need = r.get('star') or 1
        holder = holder if holder is not None else self.roster.resolve(r['holder'])
        cost = getattr(CATALOG.unit(str(r['final'])), 'cost', 0)
        worth = self.unit_worth(holder.name, holder.star) if holder is not None else 0
        equal_too = holder is not None and (holder.star or 1) >= 2
        while cost and need < 3 and (cost * 3 ** (need - 1) < worth or equal_too and cost * 3 ** (need - 1) == worth):
            need += 1
        return need

    def handover_leftovers(self, hp):
        """A final-holder move ends (its 15 s are up) with gives not made - the items its Remover or sale put on the item
        bench: they go into the round plan, which gives them as they show and is not dropped at combat start (fielded
        units take gives in the fight). Run 20261023-045645 4-2: the Remover took Cassiopeia's three items off, Void
        Staff and Jeweled Gauntlet went out, Spear of Shojin stayed on the item bench for the fight (lost 53 -> 38),
        reported as done. -> the entries handed on."""
        from tft_player import planner
        stripped = {e.get('id') for e in hp.p.get('strip') or [] if ('strip', e.get('id'), e.get('tool')) in hp.itemised}
        left = [e for e in hp.p.get('items') or []
                if ((e.get('component'), e.get('id'), e.get('k')) not in hp.itemised
                    or (e.get('component'), e.get('id'), e.get('k')) in hp.item_later)
                and ('from' not in e or e['from'] in stripped)]
        if not left:
            return []
        rp = self.round_plan
        if rp is None:
            rp = self.round_plan = planner.RoundPlan({'items': []}, hp.stage, known=[x.id for x in self.roster.units],
                                                     names=self.unit_names())
        items = rp.p.setdefault('items', [])
        k = max([e.get('k', -1) for e in items if isinstance(e.get('k'), int)] + [-1]) + 1
        out = []
        for e in left:
            c = {key: v for key, v in e.items() if key != 'from'}
            c['k'] = k
            k += 1
            items.append(c)
            out.append(c)
        return out

    HANDOVER_WAIT = 'wait'                   # handover_action: its gives are waiting - nothing else acts this tick
    HANDOVER_MIN_S = 10                      # planning seconds left to start one (a sell or Remover + a give per item)

    def handover_action(self, scr):
        """Items to their final holders (the model's item_final / a give's "final"): when we own an item's final holder
        at the star the model set (name read off its panel), in a planning phase, the item moves with no question to
        the model. Its holder outside the comp's final board is sold (its items return to the item bench), the final
        holder takes its hex if it was fielded, and the item is given; a holder in the final board keeps its place - an
        Item Remover takes its items off, the item goes to the final holder, the others back onto the holder. One
        holder at a time, as a small plan of its own ahead of the round plan (set aside meanwhile, then resumed).
        -> the next action, or None."""
        from tft_player import planner
        hp = self.handover_rp
        if hp is not None:
            if self.round_plan is not hp:        # a new plan from the model replaced it
                self.handover_rp = None
                return None
            act = hp.next_action(scr, self.roster, self.team_cap(scr))
            if act:
                self.brain.status['plan'] = f'TO FINAL HOLDER {act}'
                return act
            if hp.items_waiting() and (scr.timer is None or scr.timer > 2) and \
                    time.time() - getattr(self, '_handover_t0', 0) < 15:
                # its gives wait for the items to show and be named on the item bench (run 20261021-231230 4-6: the
                # move ended after the Remover and the sell, its gives dropped - Cassiopeia, the 2-star carry, fought
                # with no items and five completed items on the bench)
                self.brain.status['plan'] = 'TO FINAL HOLDER: the gives wait for the items on the item bench'
                return self.HANDOVER_WAIT
            moved, to = self._handover
            self.round_plan, self.handover_rp = self._handover_saved, None
            if not hp.failed:
                self.item_finals = [r for r in self.item_finals if r not in moved]
            left = self.handover_leftovers(hp)
            what = (f"items to their final holder {to}: {', '.join(r['item'] for r in moved)} - done: "
                    f"{', '.join(hp.done) or 'nothing'}" + (f"; not done: {', '.join(hp.failed)}" if hp.failed else '')
                    + (f"; still on the item bench, given by the round plan as they show: "
                       f"{', '.join(e['component'] + ' -> unit ' + str(e['id']) for e in left)}" if left else ''))
            self.news_event('final_holder', scr.stage, what)
            self.log(ev='final_holder_done', stage=scr.stage, moved=moved, done=hp.done, failed=hp.failed, left=left)
            return None
        if not self.item_finals or scr.timer is None or scr.timer < self.HANDOVER_MIN_S or self.plan_pending:
            return None                          # a sell or Remover, then a give per item: not in the last seconds
        rp = self.round_plan
        if rp is not None and getattr(rp, 'late_start', None) and rp.late_start() and rp.moves_pending(self.roster):
            return None                          # a late next_round plan's board first (request e42, game 45 4-2)
        tried = self.__dict__.setdefault('handover_tried', set())
        comp = self.comp_units()
        remover = any(str(i or '').startswith(('Item Remover', 'Magnetic Remover')) for i in self.bench_items or ())
        for r in self.item_finals:
            holder = self.roster.resolve(r['holder'])
            need = self.final_star(r, holder)
            # a fielded copy first, then the higher star
            to = min((x for x in self.roster.units if x.name == r['final'] and x.star >= need
                      and not x.missing and not x.unverified),
                     key=lambda x: (x.loc[0] != 'board', -(x.star or 1)), default=None)
            if to is None or holder is None or holder is to or (scr.stage, holder.id, to.id) in tried:
                continue
            if r['item'] not in (holder.items or []):
                continue                         # not where the record says: the state shows it, the model decides
            if self.plan_moves_off(holder):
                # the round plan takes this holder's items off itself: its transfers are the same items, counted
                # twice they leave no room on the final holder (run 20261028-022436 4-3: the plan's Stoneplate and
                # Spirit Visage from Kobuko to Krug; the move gave Krug the Stoneplate and Spirit Visage back to
                # Kobuko, then the plan stripped Kobuko again - two Remover uses, the Visage idle through the fight)
                tried.add((scr.stage, holder.id, to.id))
                self.log(ev='final_holder_skipped', stage=scr.stage, holder=repr(holder), to=repr(to),
                         why='the round plan moves its items')
                continue
            if to.loc[0] != 'board' and any(x is not to and x.name == to.name and x.loc[0] == 'board' and not x.missing
                                            for x in self.roster.units):
                # a copy of the final holder is fielded (not read yet): a benched copy fielded in the holder's hex is
                # a duplicate that adds nothing (run 20261023-045645 4-1: Leona 2 sold for her Giant's Belt, a
                # second Morgana 1-star took her front-row hex next to the fielded one; 4-2 lost 53 -> 38)
                continue
            sell = comp is not None and holder.name not in comp
            if sell and holder.loc[0] == 'board' and to.loc[0] == 'board':
                # a fielded holder sold with its final holder already fielded leaves its hex empty: an Item Remover
                # instead, or the model decides (run 20261022-054806 4-5: Hecarim sold for a Chain Vest to a fielded
                # Sentinel, the fight played with 7 units of 8 and lost 70 -> 59)
                if not remover:
                    continue
                sell = False
            if not sell and to.loc[0] != 'board':
                # a holder that stays gives up its items only to a fielded final holder: on the bench they fight for
                # no one, and the Remover and two gives went ahead of the round plan that fielded it (run
                # 20261022-025740 3-6: Protector's Vow from Elise to a benched Sentinel took 15 s of the phase; the
                # plan's Sentinel, Morgana and Cassiopeia stayed on the bench, lost 64 -> 43)
                continue
            moved =[x for x in self.item_finals if self.roster.resolve(x['holder']) is holder and x['final'] == to.name
                     and x['item'] in (holder.items or []) and to.star >= self.final_star(x, holder)]
            # only what the final holder has room for (run 20261022-090637 5-2: the Remover stripped Cassiopeia 2's
            # Void Staff, Jeweled Gauntlet and Tear for Morgana 2, who held 2 items and a Cloak: the Tear made her
            # Adaptive Helm, the Void Staff had no slot, and Cassiopeia fought 5-2 with no items, 5-3 with two)
            # the round plan's gives still to come to the final holder keep their slots: the model's newer plan wins
            # (run 20261026-020606 4-6: Spear of Shojin went from Xayah to Cassiopeia's third slot while the plan's
            # Rod and Gloves for her Jeweled Gauntlet waited; both were refused and sat idle through 4-6 and 4-7)
            planned = self.planned_gives(to)
            room = fitting(to.items, planned + [x['item'] for x in moved])[len(fitting(to.items, planned)):]
            fit = Counter(room)
            moved = [x for x in moved if fit[x['item']] > 0 and not fit.subtract([x['item']])]
            if not sell and len(moved) < len(holder.items or []):
                # a holder that stays and keeps other items: its Remover takes them ALL off and each goes back by a
                # give of its own - a risk taken for a completed item, never for a component
                moved = [x for x in moved if x['item'] not in COMPONENTS]
            if not moved:
                tried.add((scr.stage, holder.id, to.id))
                self.log(ev='final_holder_skipped', stage=scr.stage, holder=repr(holder), to=repr(to))
                continue
            items = [x['item'] for x in moved]
            ep = {}
            if sell:
                # the holder's other items come off with the sale: they go onto the final holder too (their records
                # keep their own final holders), or the holder is not sold - a completed item on the item bench
                # fights for no one (run 20261023-045645 3-5: Caitlyn sold for her Jeweled Gauntlet to Cassiopeia,
                # her Void Staff sat on the item bench through the lost fight, 71 -> 61)
                rest = list(holder.items or [])
                for i in items:
                    if i in rest:                # two records of one copy (game 78 4-3: a ValueError on 3 ticks)
                        rest.remove(i)
                rest = [i for i in rest if i and i != 'unknown']
                if rest and len(fitting(to.items, items + rest)) < len(items) + len(rest):
                    tried.add((scr.stage, holder.id, to.id))
                    self.log(ev='final_holder_skipped', stage=scr.stage, holder=repr(holder), to=repr(to),
                             why=f'{to.name} has no room for {", ".join(rest)} as well')
                    continue
                ep['sell'] = [holder.id]
                if holder.loc[0] == 'board' and to.loc[0] != 'board':
                    ep['board'] = [{'unit': to.id, 'hex': [holder.loc[1], holder.loc[2]]}]
                final_of = {x['item']: x['final'] for x in self.item_finals
                            if self.roster.resolve(x['holder']) is holder}
                ep['items'] = ([{'give': i, 'to': to.id, 'final': to.name} for i in items]
                               + [{'give': i, 'to': to.id, 'final': final_of.get(i, to.name)} for i in rest])
                marks = getattr(self.roster, 'marks', None) or []
                if (marks and marks[0][1] == holder.id and 'Riftbeast' in getattr(CATALOG.unit(to.name), 'traits', ())
                        and any(str(i or '').startswith('Alpha Mark') for i in self.bench_items or ())):
                    # the sold holder is the Alpha: the Mark goes with its items, first - one drag (run
                    # 20261028-022436 6-1: Cinderling, the Alpha, sold for Brambleback; the fight that ended the
                    # game was played with no Alpha, the Mark planned for the next round)
                    ep['items'].insert(0, {'use': 'Alpha Mark', 'on': to.id})
                how = f'{holder} is not in the final board: sold'
            else:
                rest = list(holder.items or [])
                for i in items:
                    if i in rest:                # two records of one copy (game 78 4-3: a ValueError on 3 ticks)
                        rest.remove(i)
                ep['items'] = ([{'transfer': i, 'from': holder.id, 'to': to.id} for i in items]
                               + [{'transfer': i, 'from': holder.id, 'to': holder.id} for i in rest])
                how = f'{holder} is in the final board: Item Remover'
            tried.add((scr.stage, holder.id, to.id))
            ep['keep_board'] = True              # an item handover skips prepare_comp's other board rules
            ep['keep_items'] = True              # nor runs the carry's item rule (e49): it is an item move itself
            plan = planner.normalize({'execution_plan': ep})
            plan.pop('strategy', None)
            rp = planner.RoundPlan(plan, scr.stage, known=[x.id for x in self.roster.units], names=self.unit_names(),
                                   gold=scr.gold)
            # a sold holder's items reach the item bench only by the sell: not checked against the bench before it
            bad = planner.validate(rp.p, self.roster, None if 'sell' in ep else self.bench_items, rp.names)
            if bad:
                self.news_event('final_holder', scr.stage, f"items to {to.name} ({', '.join(items)} on {holder}) could not "
                                                           f"move: {'; '.join(bad)}")
                continue
            self._handover_saved, self._handover = self.round_plan, (moved, to.name)
            self.round_plan = self.handover_rp = rp
            self._handover_t0 = time.time()
            self.log(ev='final_holder_started', stage=scr.stage, items=items, holder=repr(holder), to=repr(to), how=how,
                     plan=ep)
            print(f"{scr.stage}: {', '.join(items)} -> final holder {to} ({how})")
            return self.handover_action(scr)
        return None

    CARRY_ARRIVAL_S = 8                      # planning seconds the carry's arrival steps need

    def carry_arrival_action(self, scr):
        """The comp's carry arrived (the owner, game 78 3-5, x28.1: "sell cait; field brambleback; remove items and fit
        the first batch of items to the bramble, second carry items on the cinderling; everything should be finished in
        a few seconds at most, not like 2-3 rounds"): in a planning phase - before the model's next plan, with no
        question - one small plan of its own, run like an item handover (handover_action): every unit that holds items
        and is on none of the comp's boards with no tank trait is sold (its items come back; those of the carry's items
        go onto him); a benched carry is fielded in the place of a sold fielded one, or of an empty slot, then on his
        comp hex (planner._carry_corner); the Item Remover takes his items off the units that hold them for him, their
        other items going back onto them (planner._carry_items_move: Cinderling keeps the second carry's items). Once
        per carry and round, never under CARRY_ARRIVAL_S left. -> the next action, or None."""
        from tft_player import item_plan, planner
        from tft_player.coach import forced_comp
        from tft_player.comp_rules import TANK_TRAITS, off_comp
        comp = forced_comp()
        ip = item_plan.plan_for(getattr(comp, 'name', None))
        if (ip is None or self.handover_rp is not None or scr.timer is None or scr.timer < self.CARRY_ARRIVAL_S
                or not getattr(scr, 'shop_visible', False) or scr.augments or self.bench_items is None or REPLAY):
            return None
        name, order = ip['carry'][0], ip['carry'][1]
        mine = [u for u in self.roster.units if u.name == name and not u.missing and not u.unverified]
        if not mine:
            return None
        # a fielded copy of him - read or in doubt - is the carry already: a copy that arrives later is no arrival (the
        # owner, game 79 5-2, x30.8, "fatal mistake": the Training Yard Wisp's dummies changed the look of the fielded
        # Brambleback with his three items, he was "unverified", the bench copy just bought was fielded in a dummy's
        # place and the corner rule benched the one with the items)
        fielded = [u for u in self.roster.board() if u.name == name and not u.missing]
        if fielded and not any(u.on_board for u in mine):
            return None
        c = max(mine, key=lambda u: (u.star or 1, u.on_board, len(u.items or ())))
        done = self.__dict__.setdefault('carry_arrival_done', set())
        if (scr.stage, c.id) in done:
            return None
        # (marked done only once its steps start, or are refused: game 88 4-1 - Brambleback came from the roll-down to a
        # full bench-and-board, nothing could start, the round was marked done, and when the plan fielded him at 23 s his
        # items stayed on Mama Beak; the owner, x32.4: "once bramble is fielded, next action is to use remover")
        if (not c.items and not c.items_read and c.loc[0] == 'bench' and not getattr(c, 'gift', None)
                and getattr(c, 'arrival_source', None) != 'a bench object read as a unit'):
            c.items_read = True                  # a unit from the shop or a loot orb holds nothing
        tanky = lambda n: bool(CATALOG.unit(n) and set(CATALOG.unit(n).traits) & set(TANK_TRAITS))
        offs = [u for u in self.roster.units if u is not c and u.name and not u.missing and not u.unverified
                and off_comp(u.name) and not tanky(u.name)]
        # benched ones with items; a fielded one only to make his place (else _off_comp_out swaps it for a comp unit
        # first and the bench sale sells it - a sale alone would leave its hex empty)
        useless = [u for u in offs if u.items and not u.on_board]
        his = lambda u: any(item_plan.base(i) in order for i in u.items or ())
        # his items off another unit only while he has a free slot for them
        stand_ins = [u for u in self.roster.board() if u is not c and u.name != name and his(u)] \
            if len(c.items or ()) < 3 else []
        spot = None
        if not c.on_board:
            # his place: a fielded off-comp unit's (sold, items or not - x28.1: "sell cait, field brambleback"), never
            # a Riftbeast's (3-5: he took Scuttlecrab's hex and Caitlyn stayed)
            spot = max((u for u in offs if u.on_board), key=lambda u: (len(u.items or ()), -(u.star or 1)),
                       default=None)
            if spot is not None and spot not in useless:
                useless.append(spot)
        if c.on_board and not useless and not stand_ins:
            done.add((scr.stage, c.id))
            return None
        if (not c.on_board and spot is None and not useless
                and len(self.roster.board()) >= (self.team_cap(scr) or 0)):
            return None                          # no place for him yet: the round plan fields him, then his items move
        ep = {'sell': [u.id for u in useless]}
        if not c.on_board:
            if spot is not None:
                ep['board'] = [{'unit': c.id, 'hex': [spot.loc[1], spot.loc[2]]}]
            elif len(self.roster.board()) < (self.team_cap(scr) or 0):
                free = [(r, k) for r in (0, 1, 2, 3) for k in range(7)
                        if not any(u.loc == ('board', r, k) for u in self.roster.board())]
                if free:
                    ep['board'] = [{'unit': c.id, 'hex': list(free[0])}]
        room = 3 - len(c.items or ())
        gives = []
        for u in useless:                        # the sold units' items of the carry's go onto him
            for i in u.items or ():
                b = item_plan.base(i)
                if b in order and room > 0 and b not in gives and b not in (c.items or ()):
                    gives.append(b)
                    room -= 1
        if gives and (c.on_board or ep.get('board')):
            ep['items'] = [{'give': i, 'to': c.id, 'final': name} for i in gives]
        if not ep['sell']:
            ep.pop('sell')
        plan = planner.normalize({'execution_plan': ep})
        plan.pop('strategy', None)
        fielded = bool(ep.get('board'))
        if fielded:                              # his corner, the main tank beside it (the board rules of a plan)
            for rule in (planner._carry_corner, planner._main_tank_side, planner._keep_corner_free):
                planner._guarded(rule, plan, plan, self.roster)
        planner._guarded(planner._carry_items_move, plan, plan, self.roster, self.bench_items)
        plan['keep_board'] = plan['keep_items'] = True    # its own rules ran: none again when it runs
        if not (plan.get('sell') or plan.get('board') or plan.get('items') or plan.get('strip')):
            if c.on_board:
                done.add((scr.stage, c.id))      # fielded and nothing to move (no Remover, say): not again this round
            return None
        rp = planner.RoundPlan(plan, scr.stage, known=[x.id for x in self.roster.units], names=self.unit_names(),
                               gold=scr.gold)
        bad = planner.validate(rp.p, self.roster, None if plan.get('sell') else self.bench_items, rp.names)
        done.add((scr.stage, c.id))
        if bad:
            self.log(ev='carry_arrival_refused', stage=scr.stage, carry=repr(c), errors=bad, plan=ep)
            return None
        what = (f"{name} arrived: " + '; '.join(
            ([f"sold {', '.join(repr(u) for u in useless)} (on none of the comp's boards)"] if useless else [])
            + ([f"{name} fielded"] if fielded else [])
            + ([f"his items off {', '.join(repr(u) for u in stand_ins)} by the Item Remover"] if plan.get('strip')
               else [])
            + ([f"{', '.join(gives)} onto him"] if gives else [])))
        self.news_event('comp rule', scr.stage, what)
        self._handover_saved, self._handover = self.round_plan, ([], name)
        self.round_plan = self.handover_rp = rp
        self._handover_t0 = time.time()
        self.log(ev='carry_arrival', stage=scr.stage, carry=repr(c), sell=[repr(u) for u in useless],
                 stand_ins=[repr(u) for u in stand_ins], plan={k: plan.get(k) for k in ('sell', 'board', 'items', 'strip')})
        print(f'{scr.stage}: {what}')
        return self.handover_action(scr)

    def item_ages(self, stage):
        """{component: [the stage each copy on the item bench came, oldest first]}, kept from one state to the next
        (request e23): a copy more than last time came now, a copy fewer left (the newest goes first)."""
        ages = self.__dict__.setdefault('_item_ages', {})
        now = Counter(i for i in self.bench_items or () if i in COMPONENTS)
        if self.bench_items is None:
            return ages
        for it in list(ages):
            ages[it] = ages[it][:now[it]]
            if not ages[it]:
                del ages[it]
        for it, n in now.items():
            ages.setdefault(it, []).extend([stage] * (n - len(ages.get(it, []))))
        return ages

    def tell_plan_notes(self, rp, scr):
        """What the plan's executor kept from the plan (a comp rule that held, request e29), told in the next state."""
        while getattr(rp, 'notes', None):
            kind, what = rp.notes.pop(0)
            self.news_event(kind, scr.stage, what)
        for kind, what in (rp.p.pop('_notes', None) if isinstance(getattr(rp, 'p', None), dict) else None) or []:
            self.news_event(kind, scr.stage, what)               # validation notes for a plan not told yet

    def prepare_comp(self, scr, rp, combat=False):
        """Complete comp entries from the current roster, before optional spending or another model answer (e57)."""
        from tft_player import planner
        if rp.carry:
            return
        planner.prepare_comp(rp.p, self.roster, self.team_cap(scr), scr.items, combat,
                             getattr(self, 'augments_taken', None) or ())
        if not combat:
            key = tuple((u.id, u.name, u.star, u.loc, tuple(u.items)) for u in self.roster.units)
            if getattr(rp, 'comp_board_key', None) != key:
                rp.comp_board_key = key
                planner._reach_comp_breakpoint(rp.p, self.roster)
                # the level-8 board rules again on the roster as it is now: a unit named after the plan's check (game
                # 78 5-5: Taric from the 5-4 carousel was read after it, and 8 Riftbeasts fought - x28.12)
                lvl = (getattr(scr, 'level', None) or getattr(self, 'last_level', None), getattr(scr, 'xp', None))
                planner._guarded(planner._off_comp_out, rp.p, rp.p, self.roster, lvl)
                planner._guarded(planner._level_eight_extras, rp.p, rp.p, self.roster, lvl)
            if scr.items is not None:            # owner-no-two-components (e179): once per plan, on a read item bench
                gs = getattr(self, 'game_strategy', None)
                planner._guarded(planner.combine_components, rp.p, rp.p, self.roster,
                                 getattr(self, 'bench_items', None) or scr.items, getattr(scr, 'stage', None),
                                 gs.get('items') if isinstance(gs, dict) else None, getattr(rp, 'itemised', ()))
            for rule in (planner._carry_corner, planner._main_tank_side, planner._keep_corner_free):
                planner._guarded(rule, rp.p, rp.p, self.roster)
        for u in self.roster.units:
            rp.names.setdefault(u.id, (u.name, u.star))
        self.tell_plan_notes(rp, scr)

    def comp_carry_action(self, scr, rp):
        """The comp's carry fielded when the plan's moves are done and he is still benched (comp_rules carry_swap,
        request e24; game 40 benched Brambleback for five fights, 4-5 to 5-2, "until he is 2-star"): one board entry
        added to the plan - the carry onto the hex of the unit that leaves - once per plan, never with
        "bench_carry": true, a carried-over plan or in the last 2 s. Told in the next state. -> the next action."""
        from tft_player.comp_rules import carry_swap
        if rp.carry or rp.p.get('bench_carry') or getattr(rp, 'carry_rule_done', False) or scr.timer is None                 or scr.timer < 2:
            return None
        sw = carry_swap(self.roster.board(), self.roster.units)
        if not sw:
            return None
        rp.carry_rule_done = True
        c, v = sw
        from tft_player import planner
        planner.prepare_comp(rp.p, self.roster, self.team_cap(scr), getattr(scr, 'items', None))
        h = next((e['hex'] for e in rp.p.get('board') or [] if e.get('id') == c.id), None)
        if h is None:
            return None                         # preparation retained another copy or refused this entry
        rp.names.setdefault(c.id, (c.name, c.star))
        self.news_event('comp rule', scr.stage, f'{c.name} fielded for {v.name}{"*" * (v.star or 1)} (unit {v.id}) '
                                                f'at hex ({h[0]},{h[1]}) - he was benched and the swap keeps every '
                                                f'trait tier ("bench_carry": true in execution_plan keeps him benched)')
        return rp.next_action(scr, self.roster, self.team_cap(scr))

    SEVENTH_S = 8                             # planning seconds the comp's roll for the 7th needs to start

    def comp_seventh_action(self, scr, rp):
        """The comp's roll for its trait's 7th champion (comp_rules.seventh_roll, request e55; game 48 4-2 and 4-3:
        level 7, 42 and 55 gold, 6 distinct Riftbeasts with Mama Beak missing, the plan held the gold and the board
        fought on Riftbeast 5-6). Once the first reply's plan or a next_round plan with no roll of its own is done (its
        sells, buys and XP made): a roll down to 20 gold (the plan's floor when lower), from 24 or more, buying each
        champion of the trait not owned, selling a benched itemless unit outside the trait for room, stopping at 7
        owned; a Wisp is asked after it. Once per planning phase, never with "hold_gold": true or a carried-over plan.
        The new unit is fielded by the board check after it (board_recheck_action). -> the roll, or None."""
        from tft_player.comp_rules import seventh_roll
        if (rp.carry or rp.stage != scr.stage or rp.p.get('hold_gold')
                or getattr(self, 'seventh_stage', None) == scr.stage or not (getattr(rp, 'pre', None) is not None or getattr(rp, 'kind', None) == 'plan')
                or scr.timer is None or scr.timer < Player.SEVENTH_S or getattr(rp, 'hold_roll', False)):
            return None
        if rp.p.get('roll_n') or rp.p.get('fast_rr') or (rp.p.get('roll') or {}).get('down_to') is not None:
            return None                          # the plan rolls itself
        from tft_player.comp_rules import roll_level
        if roll_level() and (scr.level or 0) < roll_level():
            return None                          # no reroll below the comp's roll level (the owner, x25.8)
        go = seventh_roll(scr.level, scr.gold, self.roster.units, rp.p.get('keep_gold'))
        if not go:
            return None
        self.seventh_stage = scr.stage
        rp.roll_ran = True                       # the board is checked after it (board_recheck_action)
        spec = {'buy': go['buy'], 'targets': {n: 1 for n in go['buy']}, 'floor': go['floor'], 'floor_set': True,
                'stop_when': [], 'wisps': 'ask', 'wisps_after': True, 'keep': rp.protected_ids(),
                'sell_others': False, 'make_room': go['trait'], 'distinct': (go['trait'], go['want']),
                'seventh': dict(go, gold=scr.gold), 'skip_fives': bool(rp.p.get('skip_fives'))}
        return Action('rolldown', spec, f"comp rule: roll for the {go['want']}th {go['trait']} "
                                        f"({go['have']} owned) to {go['floor']} gold")

    def board_recheck_action(self, scr, rp):
        """The board checked again once the plan is done, when a roll ran in it or its board names a champion that was
        never bought (request e53, game 48 4-1: the update fielded Pebbles in Akali's hex beside "Mama Beak -> (1, 4)",
        the roll missed her, and the fight was played with 3 trait tiers against the 5 the owned units made). The
        unbought names leave the plan's board, then the comp's board rules run again on the units owned now: the trait's
        7 step (planner._reach_comp_breakpoint). Once per such
        state, never with a carried-over plan or under 4 s left. Told in the next state. -> the next action, or None."""
        from tft_player import planner
        if rp.carry or rp.stage != scr.stage or scr.timer is None or scr.timer < 4:
            return None
        unbought = []
        rp._lineup(self.roster, unbought)
        names = sorted({n for n, _ in unbought})
        key = (tuple(names), bool(getattr(rp, 'roll_ran', False)))
        if not (names or key[1]) or getattr(rp, 'recheck_key', None) == key:
            return None
        rp.recheck_key = key
        p = rp.p
        before = [dict(e) for e in p.get('board') or [] if isinstance(e, dict)]
        if names:
            p['board'] = [e for e in p.get('board') or [] if not (isinstance(e, dict) and 'id' not in e
                                                                  and e.get('name') in names)]
        n0 = len(p.get('_notes') or [])
        planner._reach_comp_breakpoint(p, self.roster)
        changed = len(p.get('_notes') or []) > n0
        self.log(ev='board_recheck', stage=scr.stage, unbought=names, roll=key[1], changed=changed,
                 board=p.get('board'))
        if not changed:
            if names:
                p['board'] = before              # nothing better to field: the entries wait for a buy as before
            return None
        for e in p.get('board') or []:
            if isinstance(e, dict) and isinstance(e.get('id'), int):
                u = next((x for x in self.roster.units if x.id == e['id']), None)
                if u is not None:
                    rp.names.setdefault(u.id, (u.name, u.star))
        self.news_event('board_recheck', scr.stage, 'the board was checked again after the plan'
                        + (f": {', '.join(names)} named in its board never bought" if names else ' and its roll')
                        + ' - the comp rule below fields the owned units')
        self.tell_plan_notes(rp, scr)
        return rp.next_action(scr, self.roster, self.team_cap(scr))

    def plan_moves_off(self, holder):
        """The items the round plan still has to take off `holder` (its transfers "from" it, not done yet)."""
        rp = getattr(self, 'round_plan', None)
        out = []
        for e in getattr(rp, 'p', {}).get('items') or []:
            if not e.get('component') or e.get('from') is None or \
                    (e.get('component'), e.get('id'), e.get('k')) in rp.itemised:
                continue
            if e['from'] == holder.id or self.roster.resolve(e['from']) is holder:
                out.append(e['component'])
        return out

    def planned_gives(self, to):
        """The items the round plan still has to give `to` (not handed out yet, not refused), in its order."""
        rp = getattr(self, 'round_plan', None)
        out = []
        for e in getattr(rp, 'p', {}).get('items') or []:
            key = (e.get('component'), e.get('id'), e.get('k'))
            if not e.get('component') or key in rp.itemised or (e.get('component'), to.id) in rp.refused:
                continue
            u = next((x for x in self.roster.units if x.id == e.get('id')), None) or self.roster.resolve(e.get('id'))
            if u is to:
                out.append(e['component'])
        return out

    def auto_line(self):
        """The standing orders and what they did since the model's last decision, for the state."""
        buy = ', '.join(f'{n} to {s}* (' + (f'{self.auto_need(n)} more to buy' if self.auto_need(n) else 'done') + ')'
                        for n, s in sorted(self.auto_buy.items())) or 'nothing listed'
        sell = ('off' if not self.auto_sell else 'on - champions not on auto_buy and spare copies'
                if self.auto_sell is True else f"on - {', '.join(self.auto_sell)}")
        done, self.auto_done = self.auto_done[-12:], []
        ho = '; '.join(f"{r['item']} (holder: unit {r['holder']}, final holder: {r['final']}"
                       + (f" at {s}*" if (s := self.final_star(r)) > 1 else '') + ')'
                       for r in getattr(self, 'item_finals', [])) or 'none set'
        return (f'Standing orders (all game; change them in any reply with auto_buy / auto_sell / item_final): '
                f'Auto-buy: {buy}. Auto-sell: {sell}. Item final holders: {ho}.'
                + (f" Done by them since your last decision: {'; '.join(done)}." if done else ''))

    BURST = 50                                # frames per place (the user: 50 bench + 50 board)

    BENCH_BAR_Y = 700                         # where a bench unit's health bar sits (1080p): the crop
                                              # anchor when no bar is drawn (bench units in combat)

    def bench_pixels(self, frame):
        from tft_player import bench
        band = bench._bench_band(frame)
        return [bench.slot_crop(band, i).astype(np.int16) for i in range(9)]

    def photograph_buy(self, u):
        """Data games: a bought unit, labelled with the shop card's name, photographed 50 times on the
        bench, then sold (data_decide). Where it landed: the ONE bench slot whose pixels changed since
        just before the buy (bench units show no health bar in combat, so bars can't tell). Not
        exactly one changed slot: nothing is saved."""
        from tft_player import bench as BN
        before = getattr(self, '_bench_before', None)
        kinds = getattr(self, '_bench_before_kinds', None) or []
        slot = None
        empties = [i for i, k in enumerate(kinds) if k == 'empty']
        if empties:
            # the game's rule: a bought unit lands in the leftmost empty slot - confirmed when that
            # slot no longer reads empty (recorded buys: 60/90; the rest merged or did not land)
            i = empties[0]
            for _ in range(15):
                time.sleep(0.07)
                got = self.cap.read()
                if got is None:
                    continue
                BN._memo.clear()
                if BN.classify(got[0], B.occupancy(got[0])[0])[i]['kind'] != 'empty':
                    slot = i
                    break
            before = None                         # no fallback guess when the rule says nothing landed
        if before is not None:
            # no empty tile recognisable (an opponent's arena in combat): exactly one slot must change
            for _ in range(15):
                time.sleep(0.07)
                got = self.cap.read()
                if got is None:
                    continue
                now = self.bench_pixels(got[0])
                change = [float((np.abs(a - b).max(axis=2) > 40).mean()) for a, b in zip(before, now)]
                moved = [i for i, c in enumerate(change) if c > 0.15]
                if len(moved) == 1:
                    slot = moved[0]
                    break
        if slot is None:
            self.log(ev='look_not_taken', unit=repr(u), why='not exactly one bench slot changed after the buy')
            return
        if u.loc != ('bench', slot):
            other = self.roster.at(('bench', slot))
            if other is not None and other is not u:
                other.loc = u.loc
            u.loc = ('bench', slot)
        time.sleep(0.3)                           # the unit drops in
        from tft_player import champs
        bench_n = self.burst(u, self.BURST, 2.5)
        champs.add_progress(u.name, bench_n, 0)
        u.labelled = True
        u.photographed = True                     # bought and photographed by us: sold next (data_decide)
        self.log(ev='champ_labelled', how=f'buy: {bench_n} bench frames', unit=repr(u))
        print(f'data: {u.name} photographed - {bench_n} bench frames')

    def data_board_full(self):
        return self.roster.slots_used() >= self.team_cap()

    def move_for_photo(self, u, dst):
        src = u.loc
        try:
            self.inp.drag(loc_point(src), loc_point(dst))
        except NotForeground:
            return False
        if self.move_took(src, dst):
            self.roster.moved(u, dst)
            time.sleep(0.4)                       # the unit walks/drops into place
            return True
        return False

    def burst(self, u, frames, seconds):
        """Save up to `frames` crops of u at its place over `seconds` (each frame must show its bar)."""
        from tft_player import champs
        saved, end = 0, time.time() + seconds + 1.5     # a little slack: 50 frames are the goal
        step = seconds / frames
        while saved < frames and time.time() < end:
            t0 = time.time()
            got = self.cap.read()
            if got is not None:
                bar = self.unit_bar(got[0], u.loc)
                if bar is None and u.loc[0] == 'bench':
                    bar = (B.BENCH[u.loc[1]][0], self.BENCH_BAR_Y)    # combat: no bar drawn on the bench
                crop = champs.unit_crop(got[0], bar) if bar is not None else None
                if crop is not None:
                    champs.add_to_library(crop, u.name, u.star, burst=True, flush=False)
                    saved += 1
            time.sleep(max(0.0, step - (time.time() - t0)))
        champs.flush_labels()
        return saved

    def data_decide(self, scr, planning):
        """Data-collection games (the user, 2026-10-05): a checklist of every champion; buy only
        champions not yet crossed off (50 bench + 50 board frames), photograph, sell. Gold goes to
        levels; rerolls only from level 8. Fielding keeps one board hex free for the board photos.
        No model calls."""
        from tft_player import champs
        if not scr.shop_visible or scr.gold is None or scr.augments:
            return None
        if scr.level:
            self.last_level = scr.level
        # only what we bought and photographed is sold - never the player's own units
        # (bench or board: the board photo leaves it on the board)
        done = [u for u in self.roster.units if getattr(u, 'photographed', False) and not u.missing
                and (u.loc[0] == 'bench' or planning)]
        if done:
            return Action('sell', done[0], f'data: sell {done[0]} (photographed)')
        if not DATA_ROLL and planning and len(self.roster.board()) < max(1, (scr.level or 1) - 1):
            a = self.brain.decide(scr, self.roster, planning)
            if a and a.kind == 'move' and a.arg[1][0] == 'board':
                return a                         # field units, leaving one hex free
        gold = scr.gold
        todo = champs.still_needed()
        cards = [(i, n) for i, n in enumerate(scr.shop)
                 if n in todo and CATALOG.unit(n) and CATALOG.unit(n).cost <= gold]
        if cards and self.roster.free_bench():
            i, n = cards[0]
            return Action('buy', i, f'data: buy {n} (not yet on the list: {len(todo)} champions left)')
        if (scr.level or 0) < 8 and gold >= 4:
            return Action('level', 1, 'data: level up (rerolls only from level 8)')
        if ((scr.level or 0) >= 8 or DATA_ROLL) and gold >= 2 and planning:
            return Action('roll', None, f'data: reroll for champions still needed ({len(todo)} left)')
        return None

    def stage1_buy(self, scr):
        """Stage 1 fallback: the model decides stage-1 buys in its plan (the user, 2026-10-09: its answer comes
        2-9 s into a 14-19 s phase). Only when no plan for this round has come with 3 s left: buy every champion
        we can afford - no interest to lose below 10 gold (run23 bought nothing at 1-3 with 5 gold and 5 cards
        while the plan was late)."""
        if not scr.round or scr.round[0] != 1 or scr.gold is None or not scr.shop_visible:
            return None
        if getattr(self, 'plan_pending', False):
            return None                          # the answer is spending the bank it was shown
        rp = self.round_plan
        if not self.planner_on() or (rp is not None and rp.stage == scr.stage and not rp.carry)                 or scr.timer is None or scr.timer > 3:
            return None
        done = self.__dict__.setdefault('_stage1_bought', set())
        for i, name in enumerate(scr.shop):
            info = CATALOG.unit(name) if name else None
            if not info or (scr.stage, i) in done or info.cost > scr.gold:
                continue
            if not self.roster.free_bench() and len(self.roster.copies(name, 1)) < 2:
                continue
            if (scr.gold - info.cost) // 10 < scr.gold // 10:
                continue                         # would cost an interest step
            done.add((scr.stage, i))
            return Action('buy', i, f'stage 1 fallback: buy {name} (no plan yet; every affordable card)')
        return None

    @staticmethod
    def effect_text(name):
        """The game's description of a Wisp or augment (data/reference wisps.md / augments.md)."""
        import re as _re
        for f, col in (('wisps.md', 1), ('augments.md', 3)):
            path = Path(__file__).resolve().parent.parent / 'data' / 'reference' / f
            try:
                for row in path.read_text(encoding='utf-8').splitlines():
                    cells = [c.strip() for c in row.strip().strip('|').split('|')]
                    if len(cells) > col and cells[0].lower() == (name or '').strip().lower():
                        return cells[col]
            except OSError:
                pass
        return ''

    def unit_change(self, name, stage):
        """A Wisp/augment that TRANSFORMS units we own (Polymorph: a random 2-cost becomes a
        3-cost, in the same slot - no new bar shows it): the units it can affect are read once,
        in idle time. Units it GIVES appear in empty slots and are read as arrivals anyway."""
        import re as _re
        desc = self.effect_text(name)
        if not _re.search(r'transform', desc, _re.I) or _re.search(r'items? transform', desc, _re.I):
            return
        costs = {int(c) for c in _re.findall(r'(\d)-cost(?= you own)', desc)}   # transformed, not the result
        if 'rightmost bench' in desc:
            # the bench's own rightmost slots, filled or not ("champions on the 3 rightmost bench slots"): the three
            # rightmost units in the records were read instead (benchmark game 20261020-193454: slots 4-6 at 4-1)
            m = _re.search(r'(\d+) rightmost bench', desc)
            n = int(m.group(1)) if m else 3
            hit = [u for u in self.roster.bench() if u.loc[1] >= 9 - n]
        else:
            hit = [u for u in self.roster.units if not costs or (u.info and u.info.cost in costs)]
        self.reread = getattr(self, 'reread', set()) | {u.id for u in hit}
        self.log(ev='unit_change_event', stage=stage, source=name, reread=[repr(u) for u in hit])
        if _re.search(r'start of every round|each round|each turn', desc, _re.I):
            self.recurring_changes = getattr(self, 'recurring_changes', set()) | {name}

    def confirm_merge(self, merge, scr):
        """A buy completed 3 copies: the game merged them into one unit. Where it stands is read
        from the screen, not assumed: of the copies' places, the one still holding a unit is it
        (bench slots by their bar; a fielded copy's hex by a bar near it)."""
        from tft_player.state import _hex_near
        # looked at frame by frame through the merge animation: done once one copy's place alone holds a unit on two
        # frames in a row (it was a fixed 0.35 s wait, then one look)
        still, prev, end = [], None, time.time() + 0.6
        while time.time() < end:
            got = self.cap.read()
            if got is None:
                continue
            bench, board = B.occupancy(got[0], exclude=self.watcher.tac[1])
            still = [loc for loc in merge['locs'] if (loc[0] == 'bench' and bench[loc[1]]) or
                     (loc[0] == 'board' and any(_hex_near(loc[1:], h) for h in board))]
            if len(still) == 1 and still == prev:
                break
            prev = still
        u = merge['unit']
        if len(still) == 1 and still[0] != u.loc and self.roster.at(still[0]) in (None, u):
            self.log(ev='merge_place', stage=scr.stage, unit=repr(u), assumed=u.loc, seen=still[0])
            u.loc = still[0]
        elif len(still) != 1:
            self.reread = getattr(self, 'reread', set()) | {u.id}       # unclear: read it once
            self.log(ev='merge_unclear', stage=scr.stage, unit=repr(u), still=still)

    def plan_rejected(self, errs, stage, what=''):
        """Entries of the model's plan that were invalid and dropped: they go on the plan's failed
        list (so the model is asked again at once, with the reason) and into the news."""
        if not errs:
            return
        self.log(ev='plan_rejected', stage=stage, errors=errs)
        print(f'{stage}: {what}plan entries rejected: {"; ".join(errs)}')
        self.alert('PLAN', f'{what}entry rejected: {errs[0][:90]}' + (f' (+{len(errs) - 1})' if len(errs) > 1 else ''),
                   'model asked again')
        if self.round_plan is not None and not what:
            self.round_plan.failed += [f'rejected {e}' for e in errs]
        self.news_event('plan_rejected', stage, f'{what}plan entries rejected: ' + '; '.join(errs))

    def tell_board_after(self, plan, stage, what=''):
        """The traits the board has after the plan's moves, when they change (CLIMB request e2: game 35 6-1, the
        player's own next_round put Taric on Scuttlecrab's hex and Riftbeast fell 7 -> 6). Objective data, told with the
        next call; a lost breakpoint asks for that call at once, once per distinct line."""
        from tft_player import planner
        if plan.get('board'):
            target = Phase.next_stage(stage) if what.startswith('your next_round') else stage
            if what.startswith('your next_round') and target.endswith('-4'):
                target = Phase.next_stage(target)
            self.newest_lineup = (target, [dict(e) for e in plan['board']])
        try:
            line, lost = planner.board_after_line(plan, self.roster)
            gives = planner.gives_line(plan, self.roster, self.queued_gives())   # e38: what each give leaves it holding
        except Exception as ex:                  # the plan runs whatever this check makes of it
            self.log(ev='board_after_error', stage=stage, error=repr(ex))
            return
        if gives:
            self.news_event('items_after', stage, f'{what}{gives}')
        for kind, note in plan.pop('_notes', None) or []:          # a comp rule that changed the plan (request e40)
            self.news_event(kind, stage, f'{what}{note}')
        if not line:
            return
        told = self.__dict__.setdefault('_board_after_told', set())
        kind = 'board_after_lost' if lost and (stage, what, line) not in told else 'board_after'
        told.add((stage, what, line))
        self.news_event(kind, stage, f'{what}{line}')

    def unit_names(self):
        """id -> (name, star) of every tracked unit: a plan keeps it to find its units again."""
        from tft_player.planner import PlanNames
        return PlanNames(self.roster.units)

    def plan_idle(self, rp, scr, combat=False):
        """The plan has steps left but none can be taken now: say why, once per change (run23:
        the controller stopped silently at 2-2 and 3-1 with a level-up and a unit still to do)."""
        carry, left = rp.leftover(self.roster)
        if not left:
            return
        missing = [uid for uid in {e.get('id') for e in (rp.p.get('board') or []) + (rp.p.get('items') or [])
                                   if isinstance(e, dict) and isinstance(e.get('id'), int)}
                   if rp.unit(self.roster, uid) is None]
        why = (f"left: {left}; gold read {scr.gold} (last {rp.last_gold}), level read {scr.level} "
               f"(using {self.last_level}), board {len(self.roster.board())} units"
               + (f"; units of the plan not found: {missing}" if missing else ''))
        if why != rp.idle_said:
            rp.idle_said = why
            self.log(ev='plan_idle', stage=scr.stage, timer=scr.timer, why=why)
            print(f'{scr.stage}: PLAN IDLE - {why}')
            self.news_event('plan_idle', scr.stage, f'your plan could not be finished: {left}'
                            + (f' (units not found: {missing})' if missing else ''))
            # tell the model now, while the round runs, so it can correct the plan (sell before a
            # buy, another hex...) - not only at the next planning phase
            # at any time left, and in combat too: what still works there (buy, sell bench units, XP,
            # reroll) is done at once, the rest is kept for the planning phase (the user)
            if ((rp.failed or missing) and not self.plan_pending
                    and not getattr(rp, 'feedback_sent', False)):
                rp.feedback_sent = True
                self.log(ev='feedback_sent', stage=scr.stage, why=why)
                print(f'{scr.stage}: asking the model to correct the plan ({"combat" if combat else f"{scr.timer}s left"})')
                self.request_plan(scr, self.last_frame, 'update',            # the steps: in "failed" / "Not done"
                                  'the controller could not finish your plan'
                                  + (f' (units not found: {missing})' if missing else ''), interrupt=False)

    def input_failed(self, act, scr, before):
        """A failed action: what the input conditions were (foreground, keyboard layout, cursor and
        the window under it) and the screen just before and after, at full size - to find why
        inputs get lost (run20: 26 of 39 actions failed from 2-5 on; runs before: ~1%)."""
        if REPLAY or not hasattr(self.inp, 'diagnose'):
            return
        n = getattr(self, '_fail_n', 0) + 1
        self._fail_n = n
        diag = self.inp.diagnose()
        if re.search(r'no unit is seen at|still on the item bench after|left the item bench, but|did not move|'
                     r'not where the drop went|stands at',
                     self.fail_cause or ''):
            # failures that point at a wrong record of where our units stand. One can be a plain miss; two in a
            # round, or in two rounds running, is consistent: read the units off the screen (board_check)
            now = time.time()
            fails = [f for f in getattr(self, '_pos_fails', []) if now - f[1] < 90] + [(scr.stage, now)]
            self._pos_fails = fails[-4:]
            if len(fails) >= 2:                      # within ~90 s: this round or the one before
                self.need_survey = (f'consistent failures ({len(fails)} lately, last: '
                                    f'{(self.fail_cause or "")[:80]})')
                self._pos_fails = []
        self.log(ev='input_failed', stage=scr.stage, timer=scr.timer, act=repr(act), cause=self.fail_cause,
                 diag=diag, n=n)
        print(f'INPUT: {act.kind} failed ({self.fail_cause}) - {diag}')
        self.alert('FAILED', f'{act.kind}: {(self.fail_cause or "no effect seen")[:90]}', 'Execution')
        if n <= 40 and shots_on():        # full-size pairs: ~1 MB each
            from PIL import Image
            Image.fromarray(before).save(self.log_dir / f'fail{n:02d}-before.jpg', quality=90)
            got = self.cap.read()
            if got is not None:
                raw = getattr(self.cap, 'raw', None)
                Image.fromarray(raw if raw is not None else got[0]).save(
                    self.log_dir / f'fail{n:02d}-after.full.jpg', quality=90)

    def snap(self, frame, name, full=False, force=False):
        if not (force or shots_on()):
            return
        from PIL import Image
        im = Image.fromarray(frame)
        if full:            # native resolution (evaluation snapshots, debugging misreads)
            raw = getattr(self.cap, 'raw', None)
            (Image.fromarray(raw) if raw is not None else im).save(self.log_dir / f'{name}.full.jpg', quality=92)
        im.resize((960, 540)).save(self.log_dir / f'{name}.jpg', quality=70)

    def snap_settled(self, frame, scr, board_seen):
        """The round's picture (<stage>-start), in planning once the board has settled: 3 s or more into the round and
        as many of our health bars on the board as units tracked there (at 10 s, taken anyway). Taken at the round's
        first tick, 25 of 32 showed an empty board - the units still on their way back from the fight (game
        20261020-204746; the user, 2026-10-20: "this does not help")."""
        due = getattr(self, '_snap_due', None)
        if not due or due[0] != scr.stage:
            return
        age = time.time() - due[1]
        tracked = sum(1 for u in self.roster.board() if not u.missing)
        if age >= 3.0 and (len(board_seen) >= tracked or age >= 10.0):
            self._snap_due = None
            self.snap(frame, f'{scr.stage}-start', full=True)

    def read(self):
        got = self.cap.read()
        if got is None:
            return None, None
        frame, t_cap = got
        self._t_cap = t_cap
        last = getattr(self, '_full_scr', None)
        now = time.time()
        if last is not None and is_carousel(last) and not last.shop_visible and now - last.t < 2.5:
            # the carousel: the chase needs fresh frames, not the HUD. A full read there OCRs the hidden
            # HUD's boxes over a turning scene every tick (0.3-0.8 s: run 213127's chase ticked every
            # 0.35-0.74 s and lost its target); between full reads the timer counts down by the clock
            import copy
            scr = copy.copy(last)
            if last.timer is not None:
                scr.timer = max(0, round(last.timer - (now - last.t)))
            return frame, scr
        if not REPLAY and getattr(self, '_hud_size', None) != frame.shape[:2]:
            # this client size's bottom-HUD offset, if one was found before (screen.calibrate_hud); live games only -
            # recorded frames of earlier patches keep the measured boxes
            from tft_player import screen as SC
            self._hud_size = frame.shape[:2]
            SC.load_hud_dx(frame.shape[1], frame.shape[0])
        scr = self.reader.read(frame)
        if scr.level is None and scr.stage and scr.stage != '1-1' and scr.timer is not None and not scr.augments                 and not is_carousel(scr) and not REPLAY and time.time() - getattr(self, '_hud_cal_at', 0) > 60                 and getattr(self, '_level_seen', None) != frame.shape[:2]:
            # the stage reads but the level does not, round after round: the bottom HUD may have moved (a patch) -
            # its offset is looked for (screen.calibrate_hud; benchmark game 20261021-094200, TFT 16.21). Not once a
            # level has been read at this size: the HUD does not move within a game, so unread levels then are a
            # store/choice screen or a panel over it - the search found nothing in 22 tries over games 1-4 and held
            # the planning phase 7-13 s each time (g4 3-3: the buys came at 3 s and 1 s, a Stoneplate missed the fight)
            self._no_level = getattr(self, '_no_level', 0) + 1
            if self._no_level >= 8:
                self._hud_cal_at, self._no_level = time.time(), 0
                from tft_player import screen as SC
                was = SC.HUD_DX
                dx = self.reader.calibrate_hud(frame)
                self.log(ev='hud_calibration', stage=scr.stage, was=was, dx=dx)
                print(f'bottom HUD offset: {was} -> {dx}', flush=True)
                if dx is not None:
                    scr = self.reader.read(frame)
        elif scr.level is not None:
            self._no_level, self._level_seen = 0, frame.shape[:2]
        self._full_scr = scr
        # HP is only trustworthy with the normal HUD up (pickers, carousel and panels cover it)
        raw = scr.hp if scr.shop_visible else None
        self._hp_open_raw = raw                      # if this frame opens a planning phase: its first HP read
        scr.hp = self.hp_filter(raw, scr.round)
        scr.hp_disputed = getattr(self, '_hp_disputed', False)
        if raw is not None and raw != scr.hp and raw != getattr(self, '_hp_logged', None):
            self._hp_logged = raw
            self.log(ev='hp_rejected', raw=raw, kept=scr.hp, stage=scr.stage)
        return frame, scr

    def model_call(self, kind, prompt, answer, latency, plan_s=None, at=None, meta=None):
        """plan_s: when the decision (first answer) arrived; latency: the whole reply; at: when it was answered,
        if earlier than now (the preparation during the queue); meta: the call's, if not the last one's."""
        stage = self.phase.stage
        from tft_player.coach import active_session
        self.log(ev='model_call', kind=kind, stage=stage, latency=round(latency, 1),
                 decision_s=round(plan_s, 1) if plan_s else None, at=at, prompt=prompt, answer=answer,
                 meta=meta or getattr(active_session(), 'last_meta', None))
        self.apply_game_strategy(answer, kind)
        if getattr(self, 'strategy_due', None) == stage and kind == 'update':
            self.strategy_due = None
            if 'game_strategy' not in str(answer or ''):
                self.log(ev='game_strategy_missing', stage=stage, what='the combat-start reply has no strategy rewrite')
        self.live.model_call(kind, stage, prompt, answer, latency)
        self.live.event(stage, 'model', f'{kind} answered in {latency:.0f}s')

    def apply_game_strategy(self, answer, kind):
        """game_strategy in a model reply: the paragraph ("text"), why it changed and, in flex, the comp it sticks
        with ("comp"). The model sends it only when its strategy changes; a reply without it keeps the last one,
        and the same text again changes nothing."""
        from tft_player.coach import json_objects
        for o in json_objects(answer if isinstance(answer, str) else ''):
            if not isinstance(o, dict):
                continue
            if isinstance(o.get('trait_choices'), dict) and o['trait_choices'] != getattr(self, 'trait_choices', None):
                self.trait_choices = o['trait_choices']          # a committed comp's brief asks for them (flex)
                self.log(ev='trait_choices', choices=self.trait_choices)
            gs = o.get('game_strategy')
            if gs is None and isinstance(o.get('strategy'), dict):
                gs = o['strategy'].get('game_strategy')          # nested in strategy: the same intent
            if isinstance(gs, str):
                gs = {'text': gs}                                # a bare paragraph
            if not isinstance(gs, dict):
                continue
            new = {k: str(gs[k]).strip() for k in ('text', 'why') if gs.get(k) is not None}
            old = self.game_strategy
            plan = {k: gs[k] for k in Player.PLAN_PARTS if isinstance(gs.get(k), dict) and gs[k]}
            climb = os.environ.get('TFT_CLIMB') == '1'
            if not ((new.get('text') and new['text'] != old.get('text'))
                    or (climb and any(plan[k] != old.get(k) for k in plan))):
                continue
            stage = self.phase.stage
            if climb and old.get('text') and not Player.plan_checkpoint(stage):
                # the game plan changes only at checkpoints (the owner's structure, 2026-10-27)
                self.log(ev='game_plan_held', kind=kind, stage=stage, plan=gs)
                self.news_event('plan_held', stage, f'your game plan sent at {stage} was not applied: it changes only '
                                                    f'at checkpoints (next: {Player.next_checkpoint(stage)})')
                continue
            self.game_strategy = {'text': new.get('text') or old.get('text', ''), 'why': new.get('why', ''),
                                  'stage': stage,
                                  **{k: plan.get(k, old.get(k)) for k in Player.PLAN_PARTS if plan.get(k, old.get(k))}}
            self.log(ev='game_strategy', kind=kind, stage=stage, strategy=self.game_strategy,
                     first=not old.get('text'))
            if climb:
                self.save_plan()

    # no round -> level map (request o7, game 36: the pinned "Levels: 2-1 4, 2-5 5, 3-2 6, 3-6 7, 4-2 8" was read
    # back as a schedule - "Level 7 comes at 3-5" - and never revised while HP fell 77 -> 29; the owner, 2026-10-28:
    # no level is tied to a round)
    PLAN_PARTS = ('boards', 'items')

    @staticmethod
    def plan_checkpoint(stage):
        """Where the climb's game plan may change (the owner's structure, 2026-10-27): before the game, the first
        planning phase of each stage (2-1, 3-1, ...) and the augment rounds (2-1, 3-2, 4-2)."""
        if not stage:
            return True
        a, b = (int(x) for x in str(stage).split('-'))
        return (a >= 2 and b == 1) or str(stage) in ('3-2', '4-2')

    @staticmethod
    def next_checkpoint(stage):
        a, b = (int(x) for x in str(stage or '1-1').split('-'))
        nxt = [f'{a}-2'] if a in (3, 4) and b < 2 else []
        return (nxt + [f'{a + 1}-1'])[0]

    @staticmethod
    def render_plan(gs):
        """The game plan as one text: the paragraph, then its boards and items."""
        out = gs.get('text', '')
        if isinstance(gs.get('boards'), dict):
            out += ' Boards: ' + '; '.join(f'level {k}: {", ".join(map(str, v)) if isinstance(v, list) else v}'
                                           for k, v in gs['boards'].items()) + '.'
        if isinstance(gs.get('items'), dict):
            out += ' Items: ' + '; '.join(f'{k}: {", ".join(map(str, v)) if isinstance(v, list) else v}'
                                          for k, v in gs['items'].items()) + '.'
        return out

    def item_map_lines(self):
        """The plan's item map against where each item is now (games 31, 32, 34: the carry's items ended on other
        units - Guinsoo's Rageblade on Elder Dragon in game 34 while the map gave it to Brambleback)."""
        from tft_player.items import RECIPES
        wanted = self.game_strategy.get('items')
        roster = getattr(self, 'roster', None)
        if not isinstance(wanted, dict) or not wanted or roster is None:
            return []
        key = lambda n: ' '.join(str(n).lower().replace("'", '').split())
        units = [u for u in roster.units if not getattr(u, 'missing', False) and u.name]
        bench = [b for b in (getattr(self, 'bench_items', None) or []) if b]
        recipe = {key(k): v for k, v in RECIPES.items()}
        parts = []
        for champ, items in wanted.items():
            own = [u for u in units if key(u.name) == key(champ)]
            where = []
            for item in items if isinstance(items, list) else [items]:
                held = [u for u in units if any(key(i) == key(item) for i in u.items)]
                if any(key(u.name) == key(champ) for u in held):
                    where.append(f'{item} held')
                elif held:
                    where.append(f'{item} on {held[0].name}')
                elif key(item) in recipe:
                    have = [c for c in recipe[key(item)] if any(key(b) == key(c) for b in bench)]
                    where.append(f'{item} not built ({" + ".join(recipe[key(item)])}; on the item bench: '
                                 f'{", ".join(have) or "neither"})')
                else:
                    where.append(f'{item} not held')
            parts.append(f'{champ}{"" if own else " (not owned)"}: ' + '; '.join(where))
        return ['Your item map against now: ' + ' | '.join(parts) + '.']

    def save_plan(self):
        """runs/<run>/game_plan.json: every version of the game plan, with the round and reason it changed at; the
        stage conversations' digest carries the current one."""
        from tft_player import coach
        gs = self.game_strategy
        coach._MEMO['plan'] = Player.render_plan(gs)
        path = Path(self.log_dir) / 'game_plan.json'
        try:
            versions = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        except (OSError, ValueError):
            versions = []
        versions.append(dict(gs))
        path.write_text(json.dumps(versions, indent=1), encoding='utf-8')

    def strategy_lines(self):
        """The model's game strategy as a state line, in every planning message; in the climb its game plan, and
        whether this round is a checkpoint."""
        gs = self.game_strategy
        if not gs.get('text'):
            return ['Your game_strategy: not written yet (format in your instructions).']
        if os.environ.get('TFT_CLIMB') == '1':
            st = self.phase.stage
            return [f'Your game plan, pinned on screen (it changes only at checkpoints, with the EV reason in "why"): '
                    f'{Player.render_plan(gs)}'] + Player.item_map_lines(self) + [
                    f'CHECKPOINT ({st}): send a revised game plan in this reply if this game has shown a reason; leave '
                    f'it out otherwise.' if Player.plan_checkpoint(st) else
                    f'Next checkpoint: {Player.next_checkpoint(st)}.']
        return [f"Your game_strategy, pinned on screen - send it again when it changes, and rewrite it at every combat start: {gs['text']}"]

    def strategy_panel(self):
        """STRATEGY (bottom right): the model's strategy paragraph and its last change; in flex mode, until it
        commits to a comp ("comp"), its comp probabilities."""
        if self.pilot:
            return None
        gs = self.game_strategy
        if not self.forced and self.comp_p:
            return {'comps': sorted(self.comp_p.items(), key=lambda kv: -kv[1])[:5], 'pending': not gs.get('text')}
        foot = ''
        if gs.get('text'):
            st = gs.get('stage')
            # just when (the user, 2026-10-13): the reason was cut off and read as noise
            foot = f'last updated: {st}' if st else 'last updated: before the game'
        return {'comp': self.forced.name if self.forced else '', 'text': gs.get('text', ''),
                'foot': foot, 'pending': not gs.get('text')}

    def publish(self, scr, planning):
        """Push the current picture to the live view."""
        p = self.brain.plan
        agents = dict(self.brain.status)
        agents['loot'] = self.status_extra.get('loot', '')
        agents['carousel'] = self.status_extra.get('carousel', '')
        agents['augment'] = self.status_extra.get('augment', '')
        agents['identify'] = self.status_extra.get('identify', '')
        agents['coach'] = ('thinking...' if self.coach.busy else 'idle') + \
            (f' (last answer {self.coach.answers[-1][0]:.0f}s)' if self.coach.answers else '')
        self.live.update(
            game=dict(stage=scr.stage, timer=scr.timer, gold=scr.gold, level=scr.level, hp=scr.hp,
                      phase='planning' if planning else ('augment' if scr.augments else 'combat/other')),
            agents=agents,
            plan=dict(comp=p.comp, carry=p.carry, traits=p.traits, units=p.units, style=p.style),
            roster=repr(self.roster).replace(' bench=', '\nbench='),
            items=self.bench_items or [])
        if self.overlay and time.time() - self.overlay_at > 0.3:
            self.overlay_at = time.time()
            self.overlay.show(**self.overlay_texts(scr, planning, agents))

    def overlay_texts(self, scr, planning, agents):
        """The four overlay panels (see overlay.py): the model's decision, the target comp, the plan's
        execution and the bot's records."""
        snap = self.live.snapshot()
        if self.pilot:
            c = self.pilot_call
            recent = c and (c['answer'] is None or time.time() - c.get('t1', 0) < 90)
            old = self.pilot_call_narrative(scr, c) if recent else self.pilot_narrative(scr)
            top = {'stage': scr.stage or self.phase.stage, 'title': old.get('intent'), 'info': old.get('info', ''),
                   'thinking': bool(old.get('thinking')), 'alerts': self.alert_lines(),
                   'rows': [(t, str(v), '') for t, v in old.get('lines', [])]}
        else:
            top = self.decision_panel(scr, snap)
        if scr is not None and is_carousel(scr):
            return {}                                         # show_carousel draws the carousel's panels
        out = dict(top_left=top, top_right=self.comp_panel(scr), bottom_left=self.exec_panel(scr, planning),
                   bottom_right=self.strategy_panel())
        cards = self.augment_cards(scr, snap) if scr is not None and scr.augments and not self.pilot else None
        if cards or getattr(self, '_cards_shown', False):     # the picker closed: its explanations go too
            out['path'] = {'cards': cards} if cards else None
            self._cards_shown = bool(cards)
        return out

    def augment_cards(self, scr, snap):
        """The model's explanation of each augment card on screen, for the overlay to draw over the cards
        (the user, 2026-10-12): while it writes it (no wait for the answer), then with its choice marked
        (TAKING / REROLL / KEEP) until the picker closes. [left, middle, right] or None."""
        import re as _re
        th = self.thinking_now()
        stage = scr.stage or self.phase.stage
        earlier = [m for m in (snap.get('model') or []) if m['kind'] == 'augment' and m.get('stage') == stage]
        if th and th['what'].startswith('DECISION: augment'):
            text, a = th['text'], {}
        else:
            if not earlier:
                return None
            m, earlier = earlier[-1], earlier[:-1]
            text, a = m.get('answer') or '', self.answer_obj(m)
        tidy = lambda t: '\n'.join(ln.strip().replace('Target final board:', 'Final board:')
                                   for ln in (t or '').strip().splitlines() if ln.strip())

        def parse(txt):
            out = [None, None, None]
            # tolerant of the heading and labels a model writes: "Card 0 (left):", "Card 0: Name", "**Card 0:**",
            # "Combat:" / "**COMBAT:**" (the owner, game 74b 2-1: "augmentation explanation disappeared" - this game's
            # answers wrote "Card 0: Slammin'" and "- **Combat:**", and every card parsed as nothing)
            parts = _re.split(r'(?mi)^[ *#-]*Card (\d)\**\s*(?:\((?:left|middle|right)\))?\s*\**\s*:\s*', txt or '')
            lab = lambda w: r'(?:^|\n)[ \t*#-]*' + w + r'\s*\**\s*:\s*\**'
            for i in range(1, len(parts) - 1, 2):
                n = int(parts[i])
                if not 0 <= n <= 2:
                    continue
                body = _re.split(r'(?m)^\s*\{', parts[i + 1])[0]               # the JSON answer is not text
                co = _re.search(lab('COMBAT') + r'(.*?)(?=' + lab('ECONOMY') + '|' + lab('SYNERGY') + '|'
                                + lab('VERDICT') + r'|$)', body, _re.S | _re.I)
                ec = _re.search(lab('ECONOMY') + r'(.*?)(?=' + lab('SYNERGY') + '|' + lab('VERDICT') + r'|$)', body,
                                _re.S | _re.I)
                ve = _re.search(lab('VERDICT') + r'(.+)', body, _re.I)
                clean = lambda t: tidy(t).strip(' *')
                out[n] = {'name': body.strip().splitlines()[0].split(' - ')[0].strip(' *"') if body.strip() else '',
                          'combat': clean(co.group(1)) if co else '', 'economy': clean(ec.group(1)) if ec else '',
                          'verdict': ve.group(1).strip(' *') if ve else ''}
            return out
        cards = parse(text)
        # after a reroll the answer explains only the new card: the unchanged ones keep their earlier explanation
        for m in reversed(earlier):
            for n, c in enumerate(parse(m.get('answer') or '')):
                if cards[n] is None and c is not None:
                    cards[n] = c
        if isinstance(a.get('pick'), int) and 0 <= a['pick'] <= 2 and cards[a['pick']]:
            cards[a['pick']]['mark'] = 'TAKING'
        for i in a.get('reroll') or []:
            if isinstance(i, int) and 0 <= i <= 2 and cards[i]:
                cards[i]['mark'] = 'REROLL'
        if isinstance(a.get('keep'), int) and 0 <= a['keep'] <= 2 and cards[a['keep']]:
            cards[a['keep']]['mark'] = 'KEEP'
        return cards if any(cards) else None

    # -- overlay panels ---------------------------------------------------------------------------
    PLAN_KINDS = ('plan', 'update', 'review', 'combat', 'pilot')
    CARD_KINDS = {'augment': 'AUGMENT', 'wisp': 'WISP', 'choice': 'CHOICE', 'carousel': 'CAROUSEL · v2',
                  'carousel_prep': 'CAROUSEL · v1'}

    def plain(self, text):
        """Overlay text in plain words (the user, 2026-10-16: everything shown in natural language, never code or JSON):
        'Rakan*@h33[Chain]' -> 'Rakan★', "('bench', 1)" -> 'bench slot 1', '(3, 2)' -> 'hex 3,2', 'unit 29' -> its name,
        no list indexes, quotes, braces or JSON keys. The model and the log keep the exact text."""
        t = str(text or '')
        t = re.sub(r'execution_plan\.\w+\s*\{.*\}', '', t)                   # a reply-format hint
        t = re.sub(r'\{[^{}]*\}', '', t)                                     # any JSON left
        t = re.sub(r'\b(items|board|bench|sell|buy|wisps)\[\d+\]\s*', '', t)  # "items[0] "
        names = {u.id: u.name for u in getattr(getattr(self, 'roster', None), 'units', []) if u.name}
        t = re.sub(r'\bunits? (\d+)\b', lambda m: names.get(int(m.group(1)), m.group(0)), t)
        t = re.sub(r'(\*+)@[hb]\d\d?(\[[^\]]*\])?', lambda m: '★' * len(m.group(1)), t)   # a unit: its stars only
        t = re.sub(r'\[([^\]]*)\]', lambda m: m.group(1) or 'nothing', t)       # a list: its names
        t = re.sub(r'\bplan:\s*', '', t)
        t = re.sub(r"\('bench', (\d)\)", r'bench slot \1', t)
        t = re.sub(r"\('board', (\d), (\d)\)", r'hex \1,\2', t)
        t = re.sub(r'\((\d), (\d)\)', r'hex \1,\2', t)
        t = t.replace('(unverified)', '').replace(' -> ', ' → ').replace('->', '→')
        t = re.sub(r'["`]|(?<![A-Za-z])\'|\'(?![A-Za-z])', '', t)          # quotes, not apostrophes
        t = re.sub(r'\s+', ' ', t).strip(' ;,-.:')
        return t[:1].upper() + t[1:]

    def alert(self, tag, text, where=''):
        """A problem for the red line on top of the decision panel: shown ~25 s, newest first - in plain words."""
        al = self.__dict__.setdefault('alerts', [])
        al.append((time.time(), tag, self.plain(text), where))
        del al[:-6]

    def alert_lines(self):
        now = time.time()
        return [(tag, text, where) for t, tag, text, where in reversed(getattr(self, 'alerts', [])) if now - t < 25][:2]

    def decision_panel(self, scr, snap):
        """AI DECISION: the latest round plan as rows (its own sentences), the interest numbers, and a
        one-off choice (augment, Wisp, anvil, carousel list) as a card for a few seconds. Never raw JSON."""
        from tft_player.coach import forced_comp
        calls = snap.get('model') or []
        fc = forced_comp()
        d = {'stage': scr.stage or self.phase.stage, 'alerts': self.alert_lines(),
             'thinking': self.thinking_now(), 'took': self.last_took(),
             'title': fc.name if fc else None, 'rows': []}
        from tft_player.coach import LIVE
        if LIVE.get('backend_switch'):
            d['alerts'] = [('MODEL', LIVE['backend_switch'], '')] + d['alerts'][:1]
        plans = [m for m in calls if m['kind'] in self.PLAN_KINDS or 'execution_plan' in str(m.get('answer') or '')]
        plan = plans[-1] if plans else None
        this_round = [m for m in plans if m.get('stage') == d['stage']]
        if this_round:
            d['rows'] = self.round_rows(this_round)
            if not d['title']:
                d['title'] = self.plan_target(this_round[-1])
        if d['rows']:
            self._dec_last = (d['stage'], list(d['rows']), d['title'])
        elif getattr(self, '_dec_last', None):
            # no answer yet this round: the last plan stays up until the new one arrives (the user, 2026-10-10)
            st, d['rows'], d['title'] = self._dec_last[0], list(self._dec_last[1]), d['title'] or self._dec_last[2]
            d['stage'] = f"{d['stage']} · plan of {st}" if st != d['stage'] else d['stage']
        # while the model thinks, the header shows it; its reasoning fills the panel only before the round's
        # first answer (every update question used to swap the plan out for a few seconds)
        d['think_block'] = not d['rows']
        econ = self.econ_row(scr)
        if econ:
            at = next((k for k, r in enumerate(d['rows']) if r[0] in ('NEXT', 'NOTE')), len(d['rows']))
            d['rows'].insert(at, econ)
        # a one-off choice (augment, Wisp, anvil) shows in the same page as the plan, until a newer answer
        one = next((m for m in reversed(calls) if m['kind'] in self.CARD_KINDS), None)
        if one and (not plan or one.get('t', 0) >= plan.get('t', 0)):
            d['card'] = self.card(one)
        return d

    @staticmethod
    def thinking_now():
        """The model's question in flight, for the overlay: {what, t0, text} (text: the reply so far), or None."""
        from tft_player import coach
        lv = coach.LIVE
        if lv.get('took') is not None or not lv.get('t0') or not coach.model_busy():
            return None
        what = lv.get('what') or 'thinking'
        what = what.split(' - ')[0].rstrip('.:')[:110]
        return {'what': what, 't0': lv['t0'], 'text': (lv.get('text') or '')[-4000:]}

    @staticmethod
    def last_took():
        from tft_player import coach
        return coach.LIVE.get('took')

    @staticmethod
    def answer_obj(m):
        from tft_player.coach import json_objects
        a = {}
        for obj in json_objects(m.get('answer') or ''):
            if isinstance(obj, dict):
                a.update(obj)
        return a

    @staticmethod
    def plan_target(m):
        st = Player.answer_obj(m).get('strategy')
        return st.get('target') if isinstance(st, dict) else None

    @staticmethod
    def plan_rows(m):
        """One row per part of a round plan, in the model's own words; the action itself where it
        wrote none. -> [(LABEL, text, reason)]"""
        a = Player.answer_obj(m)
        st = a.get('strategy') if isinstance(a.get('strategy'), dict) else {}
        ep = a.get('execution_plan') if isinstance(a.get('execution_plan'), dict) else {}
        why = st.get('why') if isinstance(st.get('why'), dict) else {}
        prompt = m.get('prompt') or ''
        acts = Player.plan_parts(ep, prompt)
        rows = []
        if str(st.get('fight_goal') or '').strip():
            rows.append(('GOAL', f"{st['fight_goal']}: {why.get('goal') or ''}".rstrip(': '), ''))
        for part, tag in (('sell', 'SELL'), ('buy', 'BUY'), ('level', 'LEVEL'), ('board', 'BOARD'),
                          ('items', 'ITEMS'), ('containers', 'OPEN')):
            said = str(why.get(part) or '').strip()
            text = said if said not in ('', '-') else acts.get(part)
            if text:
                rows.append((tag, text, ''))
            if part == 'buy':
                rows += [(t, txt, '') for t, txt in Player.wisp_lines(ep, prompt)]
        if str(why.get('next') or '').strip() not in ('', '-'):
            rows.append(('NEXT', str(why['next']), ''))
        for pr in a.get('problems') or []:
            rows.append(('NOTE', str(pr), ''))
        if not rows and st.get('now'):
            rows.append(('NOW', str(st['now']), ''))
        return rows

    ROW_ORDER = ('SELL', 'BUY', 'WISP', 'LEVEL', 'BOARD', 'ITEMS', 'OPEN', 'NEXT', 'NOTE', 'NOW')
    ROW_KIND = {'SELL': 'sell', 'BUY': 'buy', 'WISP': 'wisp', 'LEVEL': 'level', 'BOARD': 'move', 'ITEMS': 'item'}
    ROW_ACTS = {'SELL': 'sell', 'BUY': 'buy', 'WISP': 'wisps', 'LEVEL': 'xp', 'BOARD': 'board', 'ITEMS': 'items',
                'OPEN': 'open_container'}
    SAYS_NOTHING = re.compile(r"^\s*(-|unchanged|no change|same|nothing|none|n/a)\b", re.I)

    def round_rows(self, calls):
        """The round's plan as rows, one per part: the round's first plan, each later answer replacing a part only
        when it says something new for it ("Unchanged nine-unit board" keeps what was there). A part the
        executor has finished is ticked, so a done step does not read as still to do."""
        rows, notes, acts = {}, [], {}
        for m in calls:
            ep = self.answer_obj(m).get('execution_plan')
            ep = ep if isinstance(ep, dict) else {}
            # a part's words change only when its actions do: the model rewords the same plan in every update
            # (and a combat update's "the board is locked" is no new board)
            changed = set()
            for label, key in self.ROW_ACTS.items():
                v = ep.get(key)
                if v not in (None, [], {}, 0, '') and json.dumps(v, sort_keys=True, default=str) != acts.get(key):
                    acts[key] = json.dumps(v, sort_keys=True, default=str)
                    changed.add(label)
            new_notes = []
            for label, text, reason in self.plan_rows(m):
                if label == 'NOTE':
                    new_notes.append((label, text, reason))
                elif label not in rows or ((label in changed or (label not in self.ROW_ACTS and changed))
                                           and not self.SAYS_NOTHING.match(str(text))):
                    rows[label] = (label, text, reason)
            notes = new_notes or notes
        rp = self.round_plan
        if rp is not None and not rp.carry and rp.stage == calls[-1].get('stage'):
            p = rp.p
            has = {'SELL': bool(p.get('sell')), 'BUY': bool(rp.to_buy), 'LEVEL': int(p.get('level_ups') or 0) > 0,
                   'BOARD': bool(p.get('board')), 'ITEMS': bool(p.get('items')),
                   'WISP': any(isinstance(w, dict) and w.get('buy') for w in p.get('wisps') or [])}
            pending = {k for k, _ in rp.queued(self.roster)}
            for label, kind in self.ROW_KIND.items():
                if label in rows and has.get(label) and kind not in pending:
                    lb, text, reason = rows[label]
                    rows[label] = (lb, '✓ ' + str(text), reason)
        out = [rows[k] for k in self.ROW_ORDER if k in rows and k != 'NOTE']
        at = next((i for i, r in enumerate(out) if r[0] in ('NEXT', 'NOW')), len(out))
        return out[:at] + notes + out[at:]

    def econ_row(self, scr):
        """ECON: gold now, after a win / a loss, the next interest step, and the model's after-fight sells
        (once the result shows, the sells for that result)."""
        from tft_player.planner import is_pvp
        base = getattr(self, 'fight_base', None) or {}
        if not self.phase.planning and base.get('stage') == scr.stage and base.get('gold') is not None:
            g = base['gold']          # in combat: from the gold at combat start (a quick win's +1 is not counted twice)
        else:
            g = scr.gold if scr.gold is not None else getattr(self, '_last_gold_seen', None)
        if g is None or g >= 50:
            return None
        self._last_gold_seen = g
        # the user, 2026-10-16: "24g" and what happens on a win / a loss - not "win -> 25 - lose -> 24 - next 30"
        from tft_player.planner import interest_needs
        n = interest_needs(g, self.roster, is_pvp(scr.stage or self.phase.stage))
        def need(r):
            return 'hold' if r['need'] == 0 else f"sell {r['need']}g" if r['reachable'] else f"{r['need']}g short"
        rs = n['results']
        text = f"{g}g · bench {n['worth']}g · " + (
            f"{rs[0]['next']}: win {need(rs[0])} · lose {need(rs[1])}" if len(rs) == 2
            else f"{rs[0]['next']}: {need(rs[0])} (no win gold)")
        af = getattr(self, 'after_fight', None)
        reason = ''
        if af and af.get('stage') == (scr.stage or self.phase.stage):
            def fmt(o):
                o = o if isinstance(o, dict) else {'sell': list(o or []), 'buy': []}
                s = ', '.join(getattr(self.roster.resolve(i), 'name', None) or f'unit {i}' for i in o.get('sell') or [])
                b = ', '.join(o.get('buy') or [])
                return ' + '.join(x for x in (f'sell {s}' if s else '', f'buy {b}' if b else '') if x) or 'nothing'
            res = (getattr(self, 'fight_base', None) or {}).get('result')
            reason = f"win: {fmt(af.get('win'))} · lose: {fmt(af.get('lose'))}"
            if res:
                reason = f"{res}: {fmt(af.get(res))}"
            text = f"{g}g · {reason}"           # the model's orders for each result are the line (the user)
        return ('ECON', text, reason)

    def card(self, m):
        """A one-off decision as a card: what was chosen and why, in words (never the JSON)."""
        import re as _re
        a = self.answer_obj(m)
        prompt = m.get('prompt') or ''
        tag = self.CARD_KINDS.get(m['kind'], m['kind'].upper())
        if m['kind'] == 'augment':
            names = dict((int(i), n.strip()) for i, n in _re.findall(r'(?m)^\s*(\d): (.+?) - ', prompt))
            name = lambda i: names.get(i, f'card {i}') if isinstance(i, int) else str(i)
            choice = (f"Keeping {name(a['keep'])}, rerolling the others" if a.get('keep') is not None else
                      'Rerolling ' + ', '.join(name(i) for i in a['reroll']) if a.get('reroll') else
                      f"Taking {a.get('name') or name(a.get('pick'))}" if a.get('pick') is not None else 'Deciding')
        elif m['kind'] == 'wisp':
            w = _re.search(r"Wisp '([^']+)'", prompt)
            choice = f"{'Buy' if a.get('buy') else 'Skip'} {w.group(1) if w else 'the Wisp'}"
        elif m['kind'] == 'choice':
            src = _re.search(r'Source: (\w+)', prompt)
            tag = 'ANVIL' if src and 'anvil' in src.group(1) else 'CHOICE'
            choice = f"Took {a.get('name') or 'option ' + str(a.get('option'))}"
        elif m['kind'] in ('carousel', 'carousel_prep'):
            pri = a.get('carousel_priority') or a.get('item_targets') or []
            choice = ' > '.join(map(str, pri[:4])) + (' …' if len(pri) > 4 else '')
        else:
            choice = str(a.get('name') or a.get('pick') or 'answered')
        return {'tag': tag, 'choice': choice, 'why': str(a.get('why') or '')}

    def comp_panel(self, scr):
        """TARGET COMP: the fixed (or target) comp's units in its own order, as owned / missing / short of
        copies, with the comp page's items; hidden during the carousel (it sits in its icon search)."""
        if scr is not None and is_carousel(scr):
            return None
        from tft_player.coach import forced_comp
        from tft_player.meta import META
        comp = forced_comp() or (META.comps.get(self.round_plan.p.get('comp')) if self.round_plan else None)
        if comp is None:
            return None
        stars = comp.guide.get('stars') or {}
        units, owned_n = [], 0
        for name in comp.units:
            info = CATALOG.unit(name)
            mine = [u for u in self.roster.units if u.name == name and not u.missing]
            goal = int(stars.get(name) or 2)
            copies = sum(3 ** (u.star - 1) for u in mine)
            held = {i for u in mine for i in u.items}
            build = ((comp.builds.get(name) or [{}])[0].get('items') or [])[:3]
            units.append({'name': name, 'cost': info.cost if info else None, 'owned': bool(mine),
                          'star': max((u.star for u in mine), default=0), 'goal': goal,
                          'shop': name in (scr.shop or []), 'items': [(it, it in held) for it in build]})
            owned_n += bool(mine)
        lvl = scr.level or self.last_level
        return {'name': comp.name, 'count': f"{owned_n}/{len(comp.units)}" + (f" · L{lvl}" if lvl else ''),
                'units': units}

    @staticmethod
    def pretty(t):
        """A plan step as the log writes it -> words: 'move(plan: Diana*@b1 -> (3, 4))' ->
        'Diana★ bench 1 → (3,4)'."""
        import re as _re
        t = str(t)
        m = _re.match(r'^([a-z_]+)\((.*)\)( - .*)?$', t)
        if m:
            t = m.group(2) + (m.group(3) or '')
        t = t.replace('plan: ', '').replace('(unverified)', '')
        t = _re.sub(r"\[[^\]]*\]", '', t)
        t = _re.sub(r"(\*+)@h(\d)(\d)", lambda k: f"{'★' * len(k.group(1))} ({k.group(2)},{k.group(3)})", t)
        t = _re.sub(r"(\*+)@b(\d)", lambda k: f"{'★' * len(k.group(1))} bench {k.group(2)}", t)
        t = _re.sub(r"\((\d), (\d)\)", r"(\1,\2)", t)
        return t.replace(' -> ', ' → ').strip()

    STEP_TAGS = {'buy': 'BUY', 'sell': 'SELL', 'level': 'XP', 'move': 'MOVE', 'item': 'ITEM', 'container': 'OPEN',
                 'roll': 'ROLL', 'rolldown': 'ROLL', 'wisp': 'WISP'}
    ROWS = {3: 'front', 2: '2nd row', 1: '3rd row', 0: 'back'}     # board rows, front (3) to back (0)

    @classmethod
    def hex_words(cls, row, col):
        return f"{cls.ROWS.get(int(row), 'row ' + str(row))} #{int(col) + 1}"

    @classmethod
    def step_row(cls, kind, why):
        """One step in words with its kind's tag: ('move', 'plan: Elise*@b7 -> (3, 4)') ->
        ('MOVE', 'Elise★ (bench 7) → front row #5')."""
        why = str(why or '')
        tag = 'WISP' if kind == 'buy' and re.sub(r'^plan:\s*', '', why.lower()).startswith('wisp') else             cls.STEP_TAGS.get(kind, kind.upper()[:5])
        t = re.sub(r'^(plan|data|body|pilot|stage \d+):\s*', '', why)
        t = re.sub(r'^auto-', '', t)
        t = re.sub(r'\s*\((?:the model[^)]*|every affordable[^)]*|\d+ crops recorded|photographed)\)$', '', t)
        if tag == 'WISP':
            t = re.sub(r'\s*\(.*$', '', t)                  # the Wisp's name, not its card text
        t = re.sub(r'^bench (\S.*)$', lambda k: k.group(1) + ' -> bench', t)    # "bench Camille*@h00": to the bench
        t = re.sub(r'\[[^\]]*\]', '', t)                     # items a unit holds
        # units only, no positions: the panel is a short list (the user, 2026-10-10)
        t = re.sub(r'(\*+)@[hb]\d\d?', lambda k: '★' * len(k.group(1)), t)
        t = re.sub(r'\(\d, \d\)', 'board', t)
        t = re.sub(r'-> bench \d', '-> bench', t)
        t = t.replace(' -> ', ' → ').replace('->', '→').replace('(unverified)', '').strip()
        return tag, t[:1].upper() + t[1:]

    def exec_panel(self, scr, planning):
        """EXECUTION: this round's actions in the order they ran (✓ done, ✗ failed + why), then the plan's steps
        still to come (○) and a stored next_round plan; every row with its kind's coloured tag."""
        rp = self.round_plan
        acts_now = getattr(self, 'round_acts', [])
        late = any(ok is None for _, _, ok, _ in acts_now)      # board moves queued in the last second
        queued = []
        for kind, text in (rp.queued(self.roster) if rp is not None else []):
            tag, words = self.step_row(kind, text)
            queued.append(('wait', tag, words, 'queued for next round' if late and kind == 'move' else ''))
        # steps the plan gave up without an action (not in the shop, board full, ...): the newest two
        given_up = []
        for f in (rp.failed if rp is not None else [])[-2:]:
            if re.match(r'^\w+\(', f):
                continue                                     # an action that ran and failed: listed below
            body, _, why = f.partition(': ')
            kind = next((k for k in ('sell', 'buy', 'item', 'bench') if body.startswith(k)), 'move')
            given_up.append(('fail',) + self.step_row(kind, body) + (self.plain(why),))
        acts = []
        listed = any(r[1] == 'MOVE' for r in queued)
        for kind, why, ok, cause in acts_now:
            if ok is None:                                   # too late this round: listed with the queued steps
                if not listed:
                    queued.append(('wait',) + self.step_row(kind, why) + ('queued for next round',))
                continue
            reason = (str(cause or '') if kind == 'rolldown' else '') if ok else str(cause or 'failed')
            acts.append(('done' if ok else 'fail',) + self.step_row(kind, why) + (self.plain(reason),))
        acts += getattr(self, 'roll_live', None) or []     # a roll-down running now: its progress, in this list
        nxt = getattr(self, 'next_plan', None)
        nxt_rows = []
        if nxt:
            p = nxt[1]
            parts = [f"sell {len(p.get('sell') or [])}" if p.get('sell') else '',
                     'buy ' + ', '.join(map(str, p.get('buy') or [])) if p.get('buy') else '',
                     f"level x{p['level_ups']}" if p.get('level_ups') else '',
                     f"board {len(p.get('board') or [])}" if p.get('board') else '',
                     f"items {len(p.get('items') or [])}" if p.get('items') else '']
            nxt_rows.append(('next', 'NEXT', 'Next round: ' + (' · '.join(x for x in parts if x) or 'nothing to do'), ''))
        room = 10                                            # rows the panel holds
        keep = max(2, room - len(queued) - len(given_up) - len(nxt_rows))
        steps = acts[-keep:] + given_up + queued + nxt_rows
        if not steps:
            # nothing yet this round: the last list stays up until the next one has rows (the user, 2026-10-10)
            return getattr(self, '_exec_last', None) or {'mode': 'PLAN' if planning else 'COMBAT', 'steps': []}
        self._exec_last = {'mode': 'PLAN' if planning else 'COMBAT', 'steps': steps[:room]}
        return self._exec_last

    def pilot_narrative(self, scr):
        """The top panel in pilot mode: the plan for this stage of the game and each part's
        current decision (the same reasons the live view lists)."""
        p, st = self.pilot, self.pilot.status
        stage = scr.round
        left = p.goal_left(self.roster)
        if stage and stage >= p.eight_at and (scr.level or 0) >= 8:
            intent = ('Level-8 roll-down: looking for ' + ', '.join(left)) if left else 'Comp complete: the 8 fielded'
        elif stage and stage >= (2, 1):
            intent = f'Econ toward the level-8 roll-down at {p.eight_at[0]}-{p.eight_at[1]} - {p.comp.name}'
        else:
            intent = f'Stage 1: units for the early board - {p.comp.name}'
        lines = []
        for tag, key in (('ECON', 'econ'), ('LEVEL', 'level'), ('BUY', 'buy'), ('EARLY', 'early'), ('ROLL', 'roll'),
                         ('SELL', 'sell'), ('BOARD', 'board'), ('ITEMS', 'items')):
            if st.get(key):
                lines.append((tag, str(st[key])))
        if self.status_extra.get('augment'):
            lines.append(('AUGMENT', self.status_extra['augment']))
        if self.status_extra.get('choice'):
            lines.append(('CHOICE', self.status_extra['choice']))
        c = self.pilot_call
        if c and c['answer'] is not None:
            d = self.pilot_call_narrative(scr, c)
            lines.insert(0, ('MODEL', f"{c['stage']}: {d['intent']} - {c.get('why') or ''}"))
        return {'head': f"Stage {scr.stage or self.phase.stage} - pilot", 'intent': intent, 'lines': lines, 'info': ''}

    def pilot_call_narrative(self, scr, c):
        """The model's judgement call for the top panel: the question while it thinks, then its
        decision and reasoning."""
        what = {'augment': 'augment', 'wisp': 'Wisp'}.get(c['kind'], c['kind'])
        head = f"Stage {c['stage']} - model: {what}"
        if c['answer'] is None:
            return {'head': head, 'intent': f"Deciding: {c['about']}", 'lines': [],
                    'info': f"{time.time() - c['t0']:.0f}s", 'thinking': True}
        a = c['answer']
        if c['kind'] == 'augment':
            names = c.get('names') or [None] * 3
            name = lambda i: (names[i] or f'card {i}') if isinstance(i, int) and 0 <= i < 3 else str(i)
            if a.get('reroll'):
                intent = 'Rerolling ' + ', '.join(name(i) for i in a['reroll'])
            elif a.get('pick') is not None:
                intent = f"Taking {name(a['pick'])}"
            else:
                intent = 'No usable answer: the tier list decides'
            if c.get('late'):
                intent = f"Too late, the tier list took {c['late']}. The model: {intent}"
        else:
            intent = f"{'Buy' if a.get('buy') else 'Skip'} {c['about']}" if a else 'No usable answer: skip'
        return {'head': head, 'intent': intent, 'lines': [('WHY', str(c.get('why') or '...'))],
                'info': f"{c.get('t1', c['t0']) - c['t0']:.0f}s"}

    @staticmethod
    def plan_parts(ep, prompt):
        """The execution plan in words, per part, with names (the prompt's state maps slots and ids)."""
        import re as _re
        shop = dict((int(i), n) for i, n in _re.findall(r'slot (\d): ([^\s(]+(?: [^\s(]+)?) \(\d+g', prompt))
        units = dict((int(i), n.strip()) for i, n in _re.findall(r'\[id (\d+)\] ([^*\n]+?) \*', prompt))
        uname = lambda e: e.get('name') or units.get(e.get('unit'), f"unit {e.get('unit')}")
        out = {}
        if ep.get('sell'):
            out['sell'] = 'sell ' + ', '.join(units.get(i, f'unit {i}') for i in ep['sell'])
        if ep.get('buy'):
            out['buy'] = 'buy ' + ', '.join(shop.get(i, str(i)) if isinstance(i, int) else str(i) for i in ep['buy'])
        level = []
        if ep.get('xp'):
            level.append(f"level up (XP x{ep['xp']})")
        if isinstance(ep.get('roll'), dict):
            # what the roll order says, in its own terms (the user, 2026-10-11: "roll down to None gold" for a
            # roll x4 order was confusing)
            r = ep['roll']
            buy = r.get('buy')
            buy = ', '.join(buy if isinstance(buy, list) else [f'{n} {"*" * int(s)}' for n, s in buy.items()]
                            if isinstance(buy, dict) else []) if buy else ''
            how = (f"roll {r['times']} times" if r.get('times') is not None else
                   f"roll down to {r['to_gold']} gold" if r.get('to_gold') is not None else 'roll')
            level.append(how + (f' for {buy}' if buy else ''))
        if isinstance(ep.get('fast_rr'), dict):
            level.append('fast roll-down for ' + ', '.join(ep['fast_rr'].get('buy') or []))
        if level:
            out['level'] = ', '.join(level)
        if ep.get('board'):
            out['board'] = 'board: ' + ', '.join(uname(e) for e in ep['board'] if isinstance(e, dict))
        if ep.get('items'):
            uid = lambda i: units.get(i, f'unit {i}')

            def item_words(e):
                if 'transfer' in e:
                    return f"{e['transfer']}: {uid(e.get('from'))} -> " + ('the item bench' if e.get('to') == 'bench'
                                                                          else uid(e.get('to')))
                if 'reforge' in e:
                    return f"reforge {uid(e['reforge'])}"
                if 'give' in e or 'use' in e:
                    return f"{e.get('give') or e.get('use')} -> {uid(e.get('to', e.get('on')))}"
                return f"{e.get('item')} -> {uname(e)}"
            out['items'] = ', '.join(item_words(e) for e in ep['items'] if isinstance(e, dict))
        if ep.get('open_container'):
            slots = ep['open_container']
            out['containers'] = 'open the container on bench slot ' + ', '.join(
                map(str, [slots] if isinstance(slots, int) else slots))
        if ep.get('item_targets'):
            out['targets'] = 'carousel/orbs: want ' + ', '.join(map(str, ep['item_targets']))
        return out

    @staticmethod
    def plan_summary(ep, prompt):
        return '; '.join(Player.plan_parts(ep, prompt).values()) or 'no actions'

    @staticmethod
    def wisp_lines(ep, prompt):
        import re as _re
        names = dict((int(k), n) for k, n in _re.findall(r"slot (\d): WISP '([^']+)'", prompt))
        return [('WISP', f"{'Buy' if w.get('buy') else 'Skip'} {names.get(w.get('slot'), 'the Wisp')} - {w.get('why') or ''}")
                for w in ep.get('wisps') or [] if isinstance(w, dict)]

    def hp_filter(self, hp, rnd=None, opening=None):
        """Our HP, from our own HP box (read only with the normal HUD up). A new value counts once
        it is read twice in a row at least 0.5 s apart (a frame can catch the number mid-change, and two
        reads of one unchanged picture are one read: run 010416 took a '7' for 84); a rise needs 6 reads
        (a rise is almost always a misread). No damage cap: since 09-30 the box read is ours (the
        cap came from reading another player's ring) - it kept the state a round behind the
        screen in run26. The time of the last good read lets the round's prompt wait for it.
        opening: the HP read on the frame that opened a new planning phase (see below)."""
        last = getattr(self, '_hp', None)
        now = time.time()
        ph = getattr(self, 'phase', None)
        log = getattr(self, 'log', None)
        log = log if callable(log) else (lambda **k: None)
        after = getattr(self, '_hp_after', None)
        if after is not None and ph is not None and ph.stage != after[1]:
            # a player fight's damage, read steady at its end (below), is the next round's HP from its first tick: the
            # planning phase's own reads came late or never - a unit's info panel over the player list after our
            # identify clicks. Every lost fight of games 1-9 started its next round's prompts with the old HP (g9: 100
            # at 3-3 to 3-5 while the screen said 91 / 84, 45 at 6-2 for 31); a planning read still corrects it
            self._hp_after = None
            if last is not None and after[0] < last:
                self._hp, self._hp_read_at, last = after[0], now, after[0]
                log(ev='hp_from_fight', hp=after[0], stage=after[1])
        if opening is not None and last is not None and ph is not None and 0 < opening <= last:
            # the frame that opens a planning phase: the last fight's damage has landed and stopped counting, so one
            # read of it is this phase's HP for the round's first question, which goes on this tick (the climb's
            # checker, games 22-27: the HP in the state was a round late in every game - 2026-10-26 run 020606 3-3
            # read 84 five seconds in, after the question had gone with 95; 4-5's said 41 at 32). It is not locked:
            # the phase's next reads confirm it or correct it.
            self._hp_seen_stage = ph.stage
            if opening < last and fight_damage(opening, last, ph.stage):
                self._hp, self._hp_read_at, self._hp_run, last = opening, now, 0, opening
                log(ev='hp_at_open', hp=opening, stage=ph.stage)
        if hp is None or not 0 < hp <= 100:
            return last
        if hp == last:
            self._hp_disputed = False
            self._hp, self._hp_run, self._hp_read_at = hp, 0, now
            if ph is not None and ph.planning:
                self._hp_seen_stage = ph.stage
                if now - getattr(self, 'round_started', now) >= 2.0:
                    self._hp_phase = ph.stage        # read unchanged in this phase: it is this phase's value
            return hp
        # HP changes only when a fight ends: one new value per planning phase - the first confirmed one after it opens;
        # a change read later in that phase or during combat is a misread (a unit's info panel left open over the player
        # list - a dead unit's shows no green bar: run 20261019-053552 3-6 read "2" for 55 and kept it at 3-7)
        locked = last is not None and ph is not None and (not ph.planning or getattr(self, '_hp_phase', None) == ph.stage)
        correcting = locked and ph.planning and hp > last
        self._hp_disputed = bool(correcting)
        if locked and not correcting and not late_hp_drop(hp, last, ph):
            if not ph.planning and fight_damage(hp, last, ph.stage):
                # a player fight's damage: not taken in its combat, kept for the next round once read steady (three
                # reads over a second; the number counts down as the damage lands)
                fv = getattr(self, '_hp_fight', None)
                fv = (hp, fv[1] + 1, fv[2]) if fv and fv[0] == hp else (hp, 1, now)
                self._hp_fight = fv
                if fv[1] >= 3 and now - fv[2] >= 1.0:
                    self._hp_after = (hp, ph.stage)
            return last
        # (the first value too: a misread first read - "4" at 1-2 of benchmark game 20261018-150046 - held, and the
        # true 100 then needed six reads to win)
        same = getattr(self, '_hp_runv', None) == hp
        run = getattr(self, '_hp_run', 0) + 1 if same else 1
        if not same:
            self._hp_run_t0 = now
        self._hp_run, self._hp_runv = run, hp
        need, span = (6, 1.0) if correcting else (3, 1.0) if locked else (2 if last is None or hp < last else 6, 0.5)
        if run >= need and now - getattr(self, '_hp_run_t0', now) >= span:
            self._hp, self._hp_run, self._hp_read_at = hp, 0, now
            self._hp_disputed = False
            if ph is not None and ph.planning:
                self._hp_phase = self._hp_seen_stage = ph.stage   # this phase has its new value: no other change
                hp_log_settle(self, ph.stage, hp)
        return getattr(self, '_hp', None)

    def run(self):
        """Play until the stage limit. A failing tick is logged and skipped;
        only F8 / Ctrl-C / the stage limit end the run."""
        import traceback
        from tft_player.game import Paused, Stopped
        print('player running - F7 to pause / continue, F8 to stop')
        self._last_print = ''
        errors = 0
        self._start_tick_watchdog()
        while True:
            try:
                self._tick_t0 = time.time()
                done = self._tick() == 'done'
                took = time.time() - self._tick_t0
                self._tick_t0 = None
                if took > 1.0:
                    self.log(ev='slow_tick', seconds=round(took, 2), stage=getattr(self.phase, 'stage', None),
                             planning=getattr(self.phase, 'planning', None), where=getattr(self, '_tick_where', None))
                self._tick_where = None
                if done:
                    return
            except (Stopped, KeyboardInterrupt):
                raise
            except Paused:
                time.sleep(0.1)                  # F7 came mid-action: the action is not sent; tracking goes on
            except Exception as e:
                errors += 1
                self.log(ev='tick_error', error=repr(e), trace=traceback.format_exc()[-1500:])
                if errors <= 5 or errors % 50 == 0:
                    print(f'tick error ({errors}): {e!r}')
                time.sleep(0.2)

    def _start_tick_watchdog(self):
        """What a tick that takes > 1 s is doing: the main thread's innermost game-code frames, sampled once per slow
        tick and logged with it (run 20261010-010345: rounds after a fight were first seen with ~20 s of their 30 s
        left and nothing in the log said why)."""
        import sys
        import traceback
        main_id = threading.main_thread().ident

        def watch():
            while True:
                time.sleep(0.25)
                t0 = getattr(self, '_tick_t0', None)
                if t0 is None or time.time() - t0 < 1.0 or getattr(self, '_tick_where', None):
                    continue
                fr = sys._current_frames().get(main_id)
                if fr is None:
                    continue
                stack = [f for f in traceback.extract_stack(fr) if 'tft_player' in f.filename]
                self._tick_where = [f'{Path(f.filename).stem}.{f.name}:{f.lineno}' for f in stack[-6:]]
        threading.Thread(target=watch, daemon=True).start()

    def _tick(self):
        return self._tick_body()

    def _tick_body(self):
        last_print = self._last_print
        if True:
            t0 = time.perf_counter()
            if hasattr(self.inp, 'ensure_front') and not self.inp.ensure_front():
                if self.ending.game_over:
                    # the game window is gone (we left after elimination) and the client says it is over:
                    # waiting for the window would never end (run17 and run19 hung here)
                    E = self.ending
                    seen = ' '.join([E.eliminated_seen or ''] + list(getattr(E, 'words', []) or []))
                    m = re.search(r'([1-8])\s*(ST|ND|RD|TH)\s*PLACE', seen.upper())
                    kept = getattr(self, 'placement_screen', None)
                    self.placement_screen = (int(m.group(1)), f'elimination screen: "{m.group(0)}"') if m else                         kept if kept and kept[0] else (None, 'eliminated, place not read')
                    print(f'game over (client phase {self.ending.phase}); the game window is gone; placement {self.placement_screen}')
                    self.log(ev='game_over', phase=self.ending.phase, stage=self.phase.stage, hp=getattr(self, '_hp', None),
                             placement_screen=self.placement_screen)
                    return 'done'
                if time.time() - getattr(self, '_nofocus_logged', 0) > 5:
                    self._nofocus_logged = time.time()
                    self.log(ev='no_focus', stage=self.phase.stage)
                    print('game window not in front: waiting')
                time.sleep(0.3)             # covered and couldn't take focus: frames would be garbage
                return
            frame, scr = self.read()
            if scr is None:
                time.sleep(0.05)
                return
            self.last_frame = frame
            stalls = getattr(self.cap, 'stalls', None)
            while stalls:
                at, waited = stalls.pop(0)
                self.log(ev='capture_stall', at=at, seconds=waited, stage=self.phase.stage)
            if scr.stage:
                self.stage_seen = scr.stage
            if not REPLAY and self.ending.greyed(frame):
                self.ending.scan_now()          # an elimination greys the screen: read it at once
            if self.ending.eliminated_seen and re.search(r'PLACE', self.ending.eliminated_seen.upper()):
                from tft_player import coach
                if not coach.STOPPED:
                    coach.stop_model()          # no more questions to the model: the game is lost
                    self.log(ev='eliminated_stop', stage=scr.stage, text=self.ending.eliminated_seen)
                    # the place the elimination screen shows, kept: the text is gone by the time the game is over
                    # (benchmark game 20261021-033239: "8THPLACE" here, "eliminated, place not read" at the end)
                    m = re.search(r'([1-8])\s*(ST|ND|RD|TH)\s*PLACE', self.ending.eliminated_seen.upper())
                    if m:
                        self.placement_screen = (int(m.group(1)), f'elimination screen: "{m.group(0)}"')
                    print(f'{scr.stage}: eliminated ({self.ending.eliminated_seen}) - model calls and actions stopped')
            end = self.handle_end(frame, scr)
            if end == 'done':
                return 'done'
            if end:
                time.sleep(0.3)
                return
            from tft_player import coach as _c
            if _c.STOPPED:                      # eliminated: nothing else is played; wait for the exit button
                time.sleep(0.2)
                return
            # the menu check is OCR (~0.85s while the carousel turns: its pixels never stay still);
            # once a tick it cut the carousel loop to one action every 1-2s (run29)
            carousel = is_carousel(scr)
            check_menu = (not scr.shop_visible
                          and time.time() - getattr(self, '_menu_at', 0) > (5.0 if carousel else 1.0))
            if check_menu:
                self._menu_at = time.time()
            from tft_player.game import PAUSED as _PAUSED
            if check_menu and not _PAUSED.is_set() and self.reader.settings_open(frame):
                # the game menu blocks everything; close it with its X button
                print(f'{scr.stage}: game menu open - closing it')
                self.log(ev='menu_closed', stage=scr.stage)
                self.inp.click(SETTINGS_CLOSE)
                time.sleep(0.4)
                return
            was_planning = self.phase.planning
            self.phase.update(scr)
            if self.phase.changed:
                set_stage(self.phase.stage)          # coach switches conversations once, at 4-3
                # the last fight's damage, read at its end or on this opening frame: this round's first prompt
                scr.hp = self.hp_filter(None, opening=getattr(self, '_hp_open_raw', None))
            from tft_player.game import PAUSED
            paused = PAUSED.is_set()
            self.roster.in_combat = not self.phase.planning      # merges with a fighting unit wait for the fight's end
            if not was_planning and self.phase.planning:
                merged = self.roster.settle_merges()             # ...and happen now
                if merged:
                    self.log(ev='merges_settled', stage=scr.stage, names=merged)
                    print(f'{scr.stage}: merges after the fight: {", ".join(merged)}')
            elif self.phase.planning and time.time() - getattr(self, '_merges_at', 0) > 1.0:
                # three copies of one star never stand side by side in planning - the game merges them at once. Copies
                # that came without our buy (a carousel unit, a panel read naming a unit, a star corrected from its
                # badge) never ran the buy's merge: run 20261010-235723 showed three 2-star Azirs for rounds
                self._merges_at = time.time()
                merged = self.roster.settle_merges()
                if merged:
                    self.log(ev='merges_inferred', stage=scr.stage, names=merged)
                    self.news_event('merged', scr.stage, f'three copies of {", ".join(merged)} are one unit of the next '
                                                         f'star level now (the game merges them)')
            # a carousel is no fight: its start saved a full-res snap and a unit gallery of the carousel and built a
            # "COMBAT STARTED" model call on the chase's loop (run 010345 2-4: a 2.2 s gap between chase ticks)
            if was_planning and not self.phase.planning and not self.phase.changed and not is_carousel(scr):
                self.combat_t0, self._autofill_prev, self._autofill_stage = time.time(), None, None
                self._autofill_reads = 0
                # the fight's result shows as a streak change (or an HP drop): its baseline, for after-fight sells
                self.fight_base = {'stage': scr.stage, 'streak': self.reader.streak(frame), 'hp': scr.hp, 'gold': scr.gold}
                if self.round_plan is not None:      # the planning phase's rolls do not run in the fight
                    dropped = self.round_plan.drop_rolls(self.roster)
                    if dropped:
                        self.log(ev='rolls_dropped', stage=scr.stage, rolls=dropped)
                if paused:
                    pass                             # F7: no model questions; continuing asks with the state then
                elif self.forced:
                    self.combat_start(scr)
                else:
                    self.review_round(scr)
                    self.prepare_carousel(scr)       # x-3: the carousel's wants (combat_start does it in fixed-comp games)
                self.snap(frame, f'{scr.stage}-fight', full=True)       # the board we actually fought with
                self.save_board_gallery(frame, scr)          # every unit's crop with the labels we hold for it
                self.log(ev='fight', stage=scr.stage, gold=scr.gold, level=scr.level, hp=scr.hp,
                         xp=getattr(scr, 'xp', None), team_size=getattr(scr, 'team_size', None),
                         goal=(getattr(self, 'fight_goal', None) or {}).get('goal'),
                         roster=repr(self.roster), plan=repr(self.brain.plan),
                         agents=dict(self.brain.status), items=self.bench_items)
            if self.phase.changed:
                self.round_started = time.time()
                self.commit_due(scr.stage)           # the climb's flex set: its commit_by round
                self.drop_stale_board_check()
                self.round_acts = []             # this round's executed actions (the overlay's list)
                if scr.hp is not None:
                    if self.hp_at_round is not None:
                        self.hp_log.append((scr.stage, self.hp_at_round, scr.hp))
                    self.hp_at_round = scr.hp
                # last round's unfinished board/items carry over: executed at once, and
                # the new plan is told about them (never sold meanwhile: protected)
                self.carry_note, self.plan_calls = '', 0
                if self.round_plan is not None:
                    from tft_player import planner
                    carry, self.carry_note = self.round_plan.leftover(self.roster)
                    self.round_plan = planner.RoundPlan(carry, scr.stage, carry=True, names=self.unit_names(),
                                                        gold=scr.gold) if carry else None
                self.brain.new_round(scr.round, self.roster)
                # a consumable's count changes without its icon changing (the game adds removers): its tooltip is read
                # again each round (run 20261014-063145: "2 uses left" all game while the slot showed 4, 5, then 6)
                for sig in [k for k, v in self.item_names.items() if 'uses left' in (v or '')]:
                    del self.item_names[sig]
                self.item_track.new_round()      # the game can change bench items at a round's start (ItemTracker)
                for name in getattr(self, 'recurring_changes', ()):
                    self.unit_change(name, scr.stage)
                self.items_done = set()
                ch = getattr(self, 'chaser', None)
                # the HUD often shows the carousel's stage a second after the carousel began:
                # that is the same carousel, keep chasing (run28 reset the chase mid-carousel)
                same_carousel = self.carousel_stage == scr.stage
                pending_carousel = getattr(self, '_carousel_pending', None)
                if ch is not None and pending_carousel and not same_carousel:
                    # what the last carousel did, for the model: what we went for, what another
                    # player took first (contested - not a pathing miss), what we ended up chasing
                    unresolved = getattr(ch, 'unresolved', [])
                    uncertain = ', '.join(dict.fromkeys(it for it, _ in unresolved))
                    went = ' -> '.join(dict.fromkeys(ch.taken + [it for it, _ in unresolved]
                                                    + ([ch.item or ch.got] if ch.item or ch.got else [])))
                    taken = ', '.join(ch.taken)
                    self.news_event('carousel', self.carousel_stage or scr.stage,
                                    f'carousel: went for {went}' + (f' ({taken} taken by other players first)'
                                                                    if taken else '')
                                    + (f' ({uncertain} pickup unverified)' if uncertain else ''))
                    self.log(ev='carousel_result', stage=scr.stage, taken=ch.taken, last_target=ch.item,
                             got=ch.got, unresolved=unresolved)
                    # confirmed by the one new unit that holds an item (check_carousel_unit): not the first new
                    # unit - loot drops arrive too (run 033507: "we got Gromp / Diana / Nidalee / Mama Beak", 4 of 4 wrong)
                    carried = getattr(self, '_carried_read', {}).get(self.carousel_stage)
                    if carried and carried[0]:
                        self.news_event('carousel', self.carousel_stage, f'carousel: we got {carried[0]} holding '
                                        f'{carried[1] or "an item not recognised"} (read off its panel as we carried it)')
                    self.carousel_confirm = {'stage': self.carousel_stage, 'went': ch.item or ch.got, 'got': ch.got,
                                             'release_choice': ch.release_choice, 'first_contact': ch.first_contact,
                                             'carried': carried,
                                             **{k: pending_carousel[k] for k in
                                                ('started_at', 'merge_index', 'before_copies') if k in pending_carousel},
                                             'before_stars': pending_carousel.get('before_stars', {}),
                                             'before': pending_carousel['before'], 'at': time.time(),
                                             'offer': sorted(getattr(self, '_car_offer', {}).get(self.carousel_stage, ()))}
                    self._carousel_pending = None
                if not same_carousel:
                    self.chaser, self.own = CarouselChaser(), OwnTracker()
                    self.chaser.champ_first = Player.carousel_champions_first(self)
                    if self.overlay:
                        self.overlay.show(path=None)
                self.bench_items = getattr(self, 'bench_items', None)
                self._snap_due = (scr.stage, time.time())     # the round's picture: once the board has settled
                self.log(ev='round', stage=scr.stage, screen=scr.short(), roster=repr(self.roster),
                         agents=dict(self.brain.status))
                self.live.event(scr.stage, 'round', scr.short())
                print(f'\n=== {scr.stage}  gold {scr.gold}  L{scr.level}  hp {scr.hp}  {self.roster}')
                # a unit not found again for a whole round is dropped from the records, and the model told once (the
                # user, 2026-10-12: the "Not located" list only grew). Still there after all? Then it shows up as a
                # unit without a record and is read off its panel like any other (read_new_unit, unit_arrived)
                for u in list(self.roster.units):
                    if not u.missing:
                        u.missing_rounds = 0
                        continue
                    u.missing_rounds = getattr(u, 'missing_rounds', 0) + 1
                    if u.missing_rounds >= 2:
                        self.roster.units.remove(u)
                        self.news_event('unit_dropped', scr.stage, f'{u.name or "?"}{"*" * u.star} (unit {u.id}) was '
                                        f'not found for a whole round ({u.missing}): removed from the records')
            if self.paused_tick(frame, scr, paused):
                return
            if getattr(self, 'combat_call', None):
                self.send_combat_call(scr)
            # stage limit: stop once the normal HUD is up (an open augment picker
            # blocks the Settings menu, so surrendering there fails)
            st = self.phase.stage and Phase._key(self.phase.stage)
            if self.until and st and st > self.until and scr.shop_visible and not scr.augments:
                print(f'reached {self.phase.stage}; stopping')
                return 'done'
            in_carousel = is_carousel(scr) and not scr.shop_visible
            if DATA_ROLL and (in_carousel or scr.augments or bench_choice_open(frame)):
                return                           # the player's own game: augments, carousels, choices are theirs
            # the owner (x18.11, game 66 4-2): on an augment round nothing is clicked or dragged before the augment is
            # picked - a drag 0.0 s into 4-2 ran under the cards as they came up and took the first one
            if self.phase.planning and not self.pilot and not DATA_MODE and not scr.augments and self.augment_due(scr):
                return
            # the owner (x16.4): the moment a planning phase starts, the units it fields go first; never under open
            # augment cards (they cover the board)
            if self.phase.planning and not self.pilot and not DATA_MODE and not is_carousel(scr) and not scr.augments:
                self.rushing = False              # every early entrant gets its normal identity read
                t1 = time.perf_counter()
                act = self.planning_first_action(scr)
                t2 = time.perf_counter()
                if act:
                    self.perform_action(act, scr, frame, t0, t1, t2)
                    return
                # the owner (x18.10): once the board changed, the trait panel is checked against the records (a wrong
                # record is read and fixed; planning_first_action then fields the missing comp unit) - game 66 showed
                # Riftbeast 6 from 3-6 11 s to the 4-1 start while the records said 7
                sig = tuple(sorted((u.id, u.name, tuple(u.loc)) for u in self.roster.board()))
                if sig != getattr(self, '_trait_board', None) and scr.timer is not None and scr.timer > 6                         and time.time() - self.round_started > 2.5:
                    self._trait_board = sig
                    self.trait_check(frame, scr)
            self.standings_shown(frame, scr)
            self.bench_reference(frame, scr)
            if not in_carousel and not scr.augments:
                self.learn_bench_baseline(frame, scr)
            # (no choice screen opens during a carousel; its OCR check would slow the chase)
            if (not in_carousel or self.choice) and self.watch_choice(frame, scr):
                return
            if scr.augments:
                self.augment_gone = None
                self.handle_augment(scr)
                self.publish(scr, False)         # the overlay too: the model's card explanations (augment_cards)
                return
            self.collect_augment_loot(frame, scr)
            if getattr(self, 'augment_open', None):
                self.augment_missed(scr)
            self.read_encounter(frame, scr)          # 1-1 opens with an encounter (no carousel in this set)
            if is_carousel(scr) and not scr.shop_visible:
                self.handle_carousel(frame, scr)
                self._car_overlay = True
                return
            if getattr(self, '_car_overlay', False):
                # the carousel is over: its ring and item labels leave the screen now, not at the next round
                # (run 185805: they stayed over the board after the carousel)
                self._car_overlay = False
                if self.overlay:
                    self.overlay.show(path=None)
            planning = self.phase.planning and scr.shop_visible
            self.tick += 1
            rp0 = self.round_plan
            self.rushing = bool(planning and not self.pilot and scr.timer is not None and scr.timer <= FIELD_FIRST_S
                                and rp0 is not None and rp0.moves_pending(self.roster))
            # a run of planned gives goes back to back: between them the tick skips the slow reads (unit items,
            # looks, board check, loot, new units) - the owner, game 74b 4-6: "It used 20 seconds to give the items,
            # this is way too slow ... it is all planned, should be like 5s at most"
            self.rushing = self.rushing or bool(planning and not self.pilot and rp0 is not None
                                                and time.time() - getattr(self, '_gave_at', 0.0) < 2.5
                                                and rp0.items_waiting())
            tac, tac_box = self.watcher.tac
            if planning:
                bench_seen, board_seen, places = B.bar_places(frame, exclude=tac_box)
                self.snap_settled(frame, scr, board_seen)
                self.watch_lights(frame, scr)       # bench units don't walk back: no settling needed
                if not REPLAY and time.time() - getattr(self, '_notice_at', 0) > 3.0:
                    # the game's own notices ("... Rider was auto assigned! ..."): logged and passed on - data on
                    # what the game does by itself (the Sprykin rider, refused drops)
                    self._notice_at = time.time()
                    said = self.game_message(frame)
                    if said and said != getattr(self, '_notice_last', None):
                        self.news_event('game_notice', scr.stage, f'the game showed: "{said}"')
                        if re.search(r'exceed the maximum team size', said, re.I):
                            self.team_full_notice(scr)
                    self._notice_last = said
                # just after a round flips units are still walking back from combat
                # units walk back for ~2.5s after a round flips; in the last second of
                # planning bench bars vanish as combat starts - neither is evidence
                settled = time.time() - self.round_started > 2.5 and scr.timer is not None and scr.timer > 1
                if not settled and scr.timer is not None and scr.timer > 1 \
                        and time.time() - getattr(self, '_bench_at', 0) > 0.3:
                    # containers from the round's first tick (a clear match counts at once; walking units do not
                    # matter to it): run 20261016-011010 2-2 moved Rakan onto a component anvil 1 s into the round,
                    # before the first bench check, so slot 1 looked free
                    self._bench_at = time.time()
                    self.bench_perception(frame, bench_seen, scr, planning=True, containers_only=True)
                if settled:
                    # every bench slot's state (unit / empty / container / unknown / obscured) before
                    # the roster is checked against it: containers occupy slots, and an unreadable
                    # slot is no evidence that a unit left it (not before the round settles: units
                    # walking back have no bar at their slot yet)
                    if time.time() - getattr(self, '_bench_at', 0) > 0.3:
                        self._bench_at = time.time()
                        self.bench_perception(frame, bench_seen, scr, planning=True)
                        if self.auto_open_container(scr):
                            return              # opened: its choice row is next
                        if self.probe_bench_object(scr):
                            return              # an object was clicked: read again from the next frame
                    self.star_check(frame, scr, places)
                    before_auto = {u.id for u in self.roster.units if getattr(u, 'auto_placed', False)}
                    self.roster.stage = scr.stage
                    self.roster.reconcile(bench_seen, board_seen, scr.board_count,
                                          bench_kinds=[o['kind'] for o in self.bench_kinds] if self.bench_kinds else None)
                    self.reconciled_stage = scr.stage
                    for u in self.roster.units:
                        if getattr(u, 'auto_placed', False) and u.id not in before_auto:
                            self.news_event('game_moved_unit', scr.stage, f'{u} was moved onto the board by the '
                                                                          f'game (empty slot at combat start)')
                            self.reread = getattr(self, 'reread', set()) | {u.id}
                    for u in self.roster.lost:
                        self.news_event('unit_missing', scr.stage, f'{u.name} (unit {u.id}) is not where it was tracked '
                                                                   f'({u.missing}); kept as missing, not deleted')
                        Player.record_roster_gap(self, frame, scr, f'missing:{u.id}', bench_seen, board_seen)
                    for u in getattr(self.roster, 'taken', []):
                        self.log(ev='unit_taken', stage=scr.stage, unit=repr(u))
                        self.news_event('unit_taken', scr.stage, f'{u.name}{"*" * u.star} (unit {u.id}) left the board '
                                        f'without our action (the game counts one unit fewer and no bar is at its hex): '
                                        f'removed from the records')
                    for u in self.roster.units:
                        if u.unverified and u.id not in getattr(self, 'reread', set()) and not u.missing                                 and u.id not in getattr(self, 'reread_asked', set()):
                            # read it once (a fresh try budget: it may have been read before)
                            self.reread = getattr(self, 'reread', set()) | {u.id}
                            self.reread_asked = getattr(self, 'reread_asked', set()) | {u.id}
                            self.tries[u.id] = 0
                    if DATA_MODE and not DATA_ROLL:
                        # data games: every unit is read off its info panel once (a labelled crop each)
                        for u in self.roster.units:
                            if not getattr(u, 'labelled', False) and not u.missing                                     and u.id not in getattr(self, 'data_asked', set()):
                                self.data_asked = getattr(self, 'data_asked', set()) | {u.id}
                                self.reread = getattr(self, 'reread', set()) | {u.id}
                                self.tries[u.id] = 0
                    # items over every unit, read off the screen (never from what a drag intended)
                    if not self.rushing and time.time() - getattr(self, '_unit_items_at', 0) > 0.6:
                        self._unit_items_at = time.time()
                        self.read_roster_items(frame, places, scr)
                        self.check_carousel_unit(scr)
                    # every unit still looks like itself (a wrong name never shows in occupancy)
                    if not self.rushing and time.time() - getattr(self, '_looks_at', 0) > 2.5:
                        self._looks_at = time.time()
                        self.check_looks(frame.copy(), places, scr)
                    bc = getattr(self, 'buy_check', None)
                    if bc and time.time() > bc:
                        self.buy_check = None
                        tracked = [u.loc[1] for u in self.roster.bench()]   # (a missing board unit is no slot)
                        seen_slots = [i for i, b in enumerate(bench_seen) if b]
                        if tracked != seen_slots:
                            if self.pilot:
                                self.need_survey = f'bench slots tracked {tracked} != on screen {seen_slots}'
                            # model player: no re-reading of known units - reconcile() settles the
                            # slots from persistent reads (arrival -> one read; a moved unit -> moved)
                            self.news_event('roster_mismatch', scr.stage,
                                            f'after a buy the tracked bench slots {tracked} != slots with a unit on '
                                            f'screen {seen_slots} (roster: {self.roster})')
                            Player.record_roster_gap(self, frame, scr, 'bench_mismatch', bench_seen, board_seen)
                    # pilot only: a survey re-reads every unit (4.5-16 s, blocking). The model player
                    # never does: its roster comes from confirmed actions + game rules + one read
                    # per new arrival (the user, 2026-10-02: a human never clicks every champion)
                    if self.pilot and self.check_roster(frame, scr):
                        return                  # the roster was rebuilt from the screen: decide afresh
                    if not self.pilot and not self.rushing and self.board_check(frame, scr, places):
                        return                  # a wrong board record was corrected from the screen
                detail = read_item_bench(frame, detail=True)
                items = [d[0] for d in detail]
                self.item_crops = [d[2] for d in detail]
                self.item_sigs = [self.item_sig(frame, i) if it else None for i, it in enumerate(items)]
                # a name read off the tooltip (text) for this very icon beats the picture match
                texts = [bool(sig) and sig in self.item_names for sig in self.item_sigs]
                said = getattr(self, 'item_names_model', {})       # the model's labels: below a tooltip read
                items = [self.item_names.get(sig, said.get(sig, Player.picture_name(self, sig, it))) if sig else it
                         for it, sig in zip(items, self.item_sigs)]
                prev = self.prev_items or [None] * len(items)
                raw_prev, self._raw_items = getattr(self, '_raw_items', None), items
                if items == raw_prev:            # a stable read: the tracker takes what arrived or was read as text
                    what = self.item_track.sync(items, texts, time.time(), scr.stage)
                    if what:
                        self.log(ev='item_bench', stage=scr.stage, change=what, now=self.item_track.names())
                tracked = self.item_track.names()
                if [n is not None for n in tracked] == [n is not None for n in items]:
                    items = tracked              # same slots filled: the tracked names (text, our own gives) hold
                # act only on slots read the same twice in a row (a consumable's uses count aside), and never
                # re-hand an item this round
                fresh = time.time() - getattr(self, '_gave_at', 0.0) < 2.5     # just after our own give
                scr.items = [it if (same_slot_item(it, pv) or fresh and it is not None and i < len(tracked)
                                    and tracked[i] == it) and (i, it) not in self.items_done else None
                             for i, (it, pv) in enumerate(zip(items, prev))]
                self.prev_items = items
                self.bench_items = items
            elif scr.shop_visible and time.time() - getattr(self, '_bench_at', 0) > 1.0:
                # combat: containers can arrive now too (rewards); only containers count here -
                # on an opponent's arena every free tile reads 'unknown'
                self._bench_at = time.time()
                self.bench_perception(frame, B.occupancy(frame, exclude=tac_box)[0], scr, planning=False)
                if self.auto_open_container(scr):
                    return
            if not planning and scr.shop_visible:
                self.check_autofill(frame, scr)
            looting = not self.rushing and self.collect_loot(frame, scr, tac)
            if looting:
                self._parked = None                  # collecting moved it: parked again when the loot is done
            elif planning and not self.rushing:
                self.park_tactician(scr)
                self.close_unit_panel(frame, scr)
            elif not self.rushing:
                self.close_unit_panel(frame, scr, combat=True)   # the fight's damage is read off the HP box it covers
            # a unit that arrived on the bench without our buy ('?', loot, a reward) is read at once: a local
            # click, not the model's time (the user, 2026-10-09); so is one on a board hex with no record
            # ... except before this round's plan is asked for: the question goes out first and the reads run
            # while the model thinks (run 20261010-010345 1-3: three loot units read first, ~6 s, and the plan came
            # after the 15 s stage-1 round had ended)
            ask_first = planning and self.planner_on() and not self.pilot and self.plan_asked != scr.stage
            # ... and once this round's plan is here, its steps come first: arrivals are read when it has nothing to do
            # right now (run 20261012-194821 3-3: twelve failed reads of three anvils, ~1.8 s each, held the plan's
            # Wisp buy and BFF link until the last second)
            rp = self.round_plan
            plan_here = planning and rp is not None and rp.stage == scr.stage
            if scr.shop_visible and not DATA_ROLL and not ask_first and not plan_here and not self.rushing and \
                    self.identify_one(scr, combat=not planning, arrivals=True):
                return
            if not DATA_ROLL and not self.rushing and self.read_new_unit(frame, scr):
                return
            t1 = time.perf_counter()
            act = self.decide(scr, planning, frame)
            t2 = time.perf_counter()
            self.publish(scr, planning)
            if act is None:
                # item tooltips (9-14 s of OCR each) only outside planning: they delayed the
                # round's plan request until the round was nearly over
                # a plan step waiting on a fresh read (an item bench settling after a drop) comes
                # first: a 1-3 s identification click in between made run25's items take 9 s
                rp = self.round_plan
                busy = rp is not None and not rp.carry and rp.pending(self.roster)
                if plan_here and scr.shop_visible and not DATA_ROLL and not self.rushing and \
                        self.identify_one(scr, arrivals=True):
                    return                      # the plan has nothing to do this tick: an arrival is read now
                if planning and not self.pilot and self.idle_survey(scr):
                    return                      # a board check, in idle time only (stops when a plan comes)
                # reads only in time nobody else needs: while the model thinks (planning), or in
                # combat; never while plan steps are waiting, never before the model is asked
                thinking = getattr(self, 'plan_pending', False) or self.pilot or (DATA_MODE and not DATA_ROLL)
                # (a hex the model asked about is read as soon as the plan's steps are done, too)
                if planning and not busy and thinking and self.identify_one(scr):
                    return
                if not planning and scr.shop_visible and not DATA_ROLL and self.identify_one(scr, combat=True):
                    return
                # unidentified item-bench icons: read in combat, while the model thinks, and AT ONCE
                # when a plan step is waiting for an item that may be one of them
                wanted = getattr(self, 'identify_items_now', False) or (rp is not None and getattr(rp, 'need_identify', False))
                if (not planning or thinking or wanted) and self.identify_item():
                    self.identify_items_now = False
                    if rp is not None:
                        rp.need_identify = False
                    return
                time.sleep(0.03 if looting else 0.1)
                return
            self.perform_action(act, scr, frame, t0, t1, t2)

    def perform_action(self, act, scr, frame, t0, t1, t2):
        last_print = self._last_print
        line = f'{scr.stage} t={scr.timer} g={scr.gold}: {act}'
        if line != last_print:
            print(line)
            self._last_print = line
            self.live.event(scr.stage, act.kind, act.why)
        if act.kind == 'move' and 'position' in act.why:
            self.brain.position_moves += 1
        self.fail_cause, self.deferred, self.retry_soon = None, False, False
        act_rp = self.round_plan         # an update installed mid-action has already credited this step
        try:
            ok = self.execute(act, scr)
        except NotForeground:
            # the input was not sent (window not in front, or the cursor could not be placed):
            # the plan step is still to do
            if act_rp and act.why.startswith('plan'):
                act_rp.unsent(act)
            self.fail_cause = 'the input was not sent'
            self.after_fight_outcome(act, False, deferred=True)
            self.log(ev='input_not_sent', stage=scr.stage, act=repr(act), diag=self.inp.diagnose())
            time.sleep(0.3)             # user tabbed out: wait, don't quit
            return
        if self.deferred:
            # a board move in the last second: nothing was sent and the step stays in the plan (carried to the
            # next planning phase) - no failure, the overlay lists it as queued (the user, 2026-10-10)
            if act.kind in ('move', 'sell', 'item', 'strip', 'buy', 'level') and act_rp is not None and act.why.startswith('plan'):
                # handed out = counted done: it is still to do (in combat, if need be)
                act_rp.unsent(act, soon=getattr(self, 'retry_soon', False))
            self.after_fight_outcome(act, False, deferred=True)
            row = (act.kind, act.why, None, self.fail_cause)
            ra = self.__dict__.setdefault('round_acts', [])
            if row not in ra:
                ra.append(row)
                unit = act.arg if act.kind == 'sell' else None
                self.log(ev='move_queued', stage=scr.stage, timer=scr.timer, act=repr(act),
                         kind=act.kind, location=getattr(unit, 'loc', None), availability=self.fail_cause)
            time.sleep(0.05)
            return
        t3 = time.perf_counter()
        self.lat.append((t1 - t0, t2 - t1, t3 - t2))
        self._acted_at = time.time()      # gold changes after this are ours, not the win gold
        self.log(ev='action', stage=scr.stage, timer=scr.timer, gold=scr.gold, level=scr.level,
                 hp=scr.hp, shop=scr.shop, act=repr(act), ok=ok, roster=repr(self.roster),
                 ms=dict(read=round((t1 - t0) * 1000), decide=round((t2 - t1) * 1000),
                         act=round((t3 - t2) * 1000)))
        self.__dict__.setdefault('round_acts', []).append(
            (act.kind, act.why, ok, act.arg.get('summary') if ok and act.kind == 'rolldown' and isinstance(act.arg, dict)
             else self.fail_cause))
        if act_rp and act.why.startswith('plan'):
            act_rp.record(act, ok, self.fail_cause)
            if not ok:
                self.news_event('step_failed', scr.stage,
                                f'{act.kind}: {act.why}' + (f' - {self.fail_cause}' if self.fail_cause else ''))
        self.note_own_action(act, ok, scr)
        self.after_fight_outcome(act, ok)
        if act.why.startswith('auto-'):
            note = act.why[len('auto-'):] + ('' if ok else f' (failed: {self.fail_cause or "?"})')
            if ok and act.kind == 'buy' and getattr(act, 'more', None):
                note += ' and ' + ', '.join(n for _, n in act.more) + ' in the same burst'
            if ok and act.kind == 'buy':
                # the unit it bought, so the model can sell it if it does not want it (the user, 2026-10-09)
                name = act.why.split('auto-buy ', 1)[-1]
                u = next((x for x in reversed(self.roster.units) if x.name == name and not x.missing), None)
                note += f' (unit {u.id}, {u.name}{"*" * u.star}' if u else ' ('
                note += ' - bought by your auto_buy order; it can be sold like any unit)'
            self.auto_done.append(note)
        if ok and act.kind == 'buy' and 'wisp ' in act.why:
            m = re.search(r'wisp (.+?)(?: \(|$)', act.why)
            if m:
                self.unit_change(m.group(1), scr.stage)
                self.expect_gifts(m.group(1), scr.stage)
        if not ok:
            self.brain.failed[(act.kind, str(act.arg))] += 1
            self.input_failed(act, scr, frame)
        if (ok and self.round_plan is not None and not getattr(act, 'fight', False) and not DATA_MODE
                and getattr(getattr(self, 'phase', None), 'planning', False) and not getattr(self, '_in_burst', False)):
            self.plan_burst(scr)
        return ok

    BURST_STEPS = 30
    BURST_SECONDS = 4.0

    def plan_burst(self, scr):
        """The next steps right after one ran, back to back: a fresh frame and its HUD read (~30-40 ms), the decision,
        the step - none of the tick's slower reads between them (the item bench's pictures, unit looks and items, the
        board check, star badges, loot, new units, the game's notices). Those run in the next tick, after the burst.
        The owner, 2026-11-08: "you do not need to check for every action, it is a waste of time ... make latency fast,
        as fast as possible" - every step of game 74b paid 0.5-1.4 s of reads and 0-1.3 s of deciding before it ran.
        The item bench is the tracker's (our own gives and returns keep it); a step that does not run, the planning
        phase ending (another stage, the countdown rising, no shop) or nothing left to do ends the burst."""
        from tft_player.game import PAUSED
        from tft_player import coach as _c
        self._in_burst = True
        stage0, timer0, t_start, ran = scr.stage, scr.timer, time.time(), []
        try:
            for _ in range(self.BURST_STEPS):
                if PAUSED.is_set() or _c.STOPPED or time.time() - t_start >= self.BURST_SECONDS:
                    break
                t0 = time.perf_counter()
                frame, s = self.read()
                if (s is None or not s.shop_visible or s.augments or is_carousel(s) or s.stage != stage0
                        or s.timer is None or s.timer <= 1
                        or timer0 is not None and s.timer > timer0 + 1):
                    break
                timer0 = s.timer
                self.last_frame = frame
                names = self.item_track.names()
                s.items = [n if (i, n) not in self.items_done else None for i, n in enumerate(names)]
                self.bench_items = names
                if not self.pilot and self.augment_due(s):
                    break                        # an augment round: nothing before the pick (x18.11)
                t1 = time.perf_counter()
                act = (None if self.pilot else self.planning_first_action(s)) or self.decide(s, True, frame)
                t2 = time.perf_counter()
                if act is None:
                    break
                ran.append(act.kind)
                if not self.perform_action(act, s, frame, t0, t1, t2):
                    break
                if act.kind in ('container', 'link', 'augment') or act.kind == 'buy' and 'wisp ' in act.why:
                    break                        # what it brings (a choice row, units, items) is read by the tick
        finally:
            self._in_burst = False
            if ran:
                self.log(ev='plan_burst', stage=stage0, steps=ran, seconds=round(time.time() - t_start, 2))

    # -- the model plans each round ----------------------------------------------
    PLAN_DEADLINE = 5        # seconds left when a missing plan lets local rules field units (moves only)

    def reconcile_item_roles(self, scr, rp):
        """Read one disputed primary recipient before compiling away its orders; no survey of healthy units."""
        pending = rp.p.get('_item_reconcile') or []
        tried = self.__dict__.setdefault('_item_role_reads', {})
        for uid in pending:
            u = self.roster.resolve(uid)
            if u is None or u.missing:
                continue
            key = (scr.stage, uid, u.loc, u.name, u.star)
            if tried.get(key, 0) >= 2:
                continue
            got = self.cap.read()
            if got is None:
                return
            tried[key] = tried.get(key, 0) + 1
            bars = B.green_bars(got[0])
            others = [self.unit_point(v.loc, bars) for v in self.roster.units
                      if v.name and v.name != u.name and not v.missing and not v.unverified][:1]
            name, stars, held = self.read_fresh(self.unit_point(u.loc, bars), others)
            self.log(ev='item_delivery', status='reconcile', stage=scr.stage, unit_id=uid,
                     loc=u.loc, read=name, stars=stars, equipped=held, requested=rp.p.get('items'))
            if name and name == u.name and held is not None and 'unknown' in held:
                # an icon our banks lack (Hand of Justice in game 78, on Murkwolf and Caitlyn) where the records hold
                # an item the read does not name: the records' item - they come from our own gives
                filled = fill_unknown(list(held), list(u.items or []))
                if ('unknown' not in filled and len(filled) == len(u.items or [])
                        and not (Counter(filled) - Counter(u.items or []))):
                    held = filled
            if name and held is not None and 'unknown' not in held:
                self.apply_partial_survey([[u.loc, name, stars, held]], scr, 'primary item recipient')
                here = self.roster.at(u.loc)
                if here is not None and here.name == name:
                    here.items_read, here.items_check, here.items_predicted = True, False, False
                    here.package_confirmed = (here.loc, here.name, here.star, tuple(here.items))
                self.prepare_comp(scr, rp)
            elif getattr(self, 'log_dir', None) is not None:
                self.snap(got[0], f'{scr.stage}-item-recipient-{uid}-unresolved', True, force=True)
            return                              # at most one read per pass; other actions may continue

    def planning_first_action(self, scr):
        """Owner e227: execute owned entries, then fill vacant slots from the comp.

        No model calls, item reads or spending. Recomputed after each verified move
        and level-up; never field a duplicate or override an explicit bench/sell.
        """
        repair = getattr(self, '_autofill_repair', None)
        if repair and repair[0] != scr.stage:
            self._autofill_repair = None
            self.need_survey = 'a purchase crossed combat auto-fill; read the home hex and bench slots'
            self.survey_places = (self.need_survey, repair[1])
            self._repair_pending = True
        if getattr(self, '_repair_pending', False):
            return None                         # decide reads the stale source before another deployment
        located = [u for u in self.roster.units if not u.missing]
        duplicates = [loc for loc, n in Counter(u.loc for u in located).items() if n > 1 and loc[0] == 'bench']
        repaired = self.__dict__.setdefault('_duplicate_reads', set())
        key = (scr.stage, tuple(sorted((u.id, u.loc) for u in located)))
        if duplicates and key not in repaired:
            repaired.add(key)
            self.need_survey = 'multiple owned records name one bench slot'
            self.survey_places = (self.need_survey, duplicates)
            self._repair_pending = True
            return None
        if getattr(self, 'next_plan', None) and self.plan_asked != scr.stage:
            self.start_next_plan(scr)
        cap = self.team_cap(scr)
        rp = self.round_plan
        if rp:
            if rp.carry:
                self.drop_carried_moves(scr)       # preserve e177: only explicit carried destinations run
            # the plan's bench sells first, the moment the bench bars are drawn (the owner, 2026-11-08: "just sell at
            # the start of the planning")
            if rp.sells_pending(self.roster) and self.bench_bars_drawn():
                act = rp.sell_step(self.roster, scr)
                if act:
                    return act
            act = rp.deployment_step(self.roster, cap, scr)
            if act:
                return act
            if rp.p.get('_item_reconcile'):
                Player.reconcile_item_roles(self, scr, rp)
        full = self.roster.slots_used() >= cap
        comp = getattr(self, 'forced', None)
        guide = getattr(comp, 'guide', {}) or {}
        names = list(((getattr(self, 'game_strategy', {}) or {}).get('boards') or {}).get(str(cap)) or [])
        early = (guide.get('early') or {}).get(str(cap)) or []
        if early:
            names += list(early[0].get('units') or [])
        names += list(getattr(comp, 'units', ()) or getattr(getattr(self.brain, 'plan', None), 'units', ()) or [])
        excluded = set((rp.p.get('sell') or []) if rp else [])
        excluded.update(e.get('id') for e in ((rp.p.get('bench') or []) if rp else []))
        fielded = {u.name for u in self.roster.board()}
        if full:
            # the owner (x18.10, game 66 3-6/4-1): a second copy of a fielded champion adds no trait - a comp unit on
            # the bench that is not fielded takes its hex (a second Cinderling stood where Mama Beak belonged: 6 of 7
            # Riftbeast). A copy the plan places by its id stays
            placed = {e.get('id') for e in ((rp.p.get('board') or []) if rp else [])}
            kept, dups = set(), []
            for u in sorted(self.roster.board(), key=lambda u: (-u.star, -len(u.items or []), u.id)):
                if u.name in kept and u.id not in placed and not u.missing and not u.unverified:
                    dups.append(u)
                kept.add(u.name)
            for d in dups:
                for name in dict.fromkeys(names):
                    if name in fielded:
                        continue
                    u = next((b for b in sorted(self.roster.bench(), key=lambda b: -b.star)
                              if b.name == name and not b.missing and not b.unverified and b.id not in excluded
                              and b.slots == d.slots and b.field_ready(scr.stage)), None)
                    if u is None:
                        continue
                    a = Action('move', (u, ('board', *d.loc[1:])),
                               f'comp deployment: {u} -> {tuple(d.loc[1:])} (swap with the second copy {d})')
                    if getattr(self.brain, 'failed', {}).get((a.kind, str(a.arg)), 0) < 2:
                        return a
            return None
        swap = Player.deployment_swap_target(self, scr)
        for name in dict.fromkeys(names):
            if name in fielded:
                continue
            if swap is not None:
                # the screen showed the board full though the records had room (game 81 5-1: two Bramblebacks and Mama
                # Beak benched, four drags to the empty corner refused "8/8 units on screen") - a swap onto the unit
                # that adds nothing (a second copy, else an unidentified one) instead of a drop on an empty hex
                u = next((b for b in sorted(self.roster.bench(), key=lambda b: -b.star) if b.name == name
                          and not b.missing and b.id not in excluded and b.slots == swap.slots
                          and b.field_ready(scr.stage)), None)
                if u is not None:
                    a = Action('move', (u, ('board', *swap.loc[1:])),
                               f'comp deployment: {u} -> {tuple(swap.loc[1:])} (swap with {swap}: the board is full on '
                               f'screen and it adds nothing)')
                    if getattr(self.brain, 'failed', {}).get((a.kind, str(a.arg)), 0) < 2:
                        return a
                continue
            candidates = [u for u in self.roster.bench() if u.name == name and not u.missing
                          and u.id not in excluded and self.roster.slots_used() + u.slots <= cap
                          and u.field_ready(scr.stage)]
            for u in sorted(candidates, key=lambda u: -u.star):
                hexes = list((guide.get('positions') or {}).get(name) or [])
                row = 0 if u.info and u.info.range > 1 else 3
                hexes += [(r, c) for r in (row, *[v for v in range(4) if v != row]) for c in range(7)]
                dst = next((('board', *h) for h in hexes if self.roster.at(('board', *h)) is None), None)
                if dst:
                    a = Action('move', (u, dst), f'comp deployment: {u} -> {dst[1:]}')
                    if getattr(self.brain, 'failed', {}).get((a.kind, str(a.arg)), 0) < 2:
                        return a
        return None

    def deployment_swap_target(self, scr):
        """After a drop this phase was refused because the screen showed the board full: the fielded unit a deployment
        swaps out - a second copy of a fielded champion (the lower star), else an unidentified unit - or None."""
        if getattr(self, '_board_full_at', None) != scr.stage:
            return None
        board = [u for u in self.roster.board() if not u.missing and not getattr(u, 'slotless', False)]
        named = sorted((u for u in board if u.name), key=lambda u: (-(u.star or 1), -len(u.items or ()), u.id))
        seen, dups = set(), []
        for u in named:
            (dups.append(u) if u.name in seen else seen.add(u.name))
        unknown = [u for u in board if not u.name]
        # else a fielded unit whose leaving loses no trait tier - never the carry, a holder of a completed item or the
        # Alpha - the cheapest first (game 81 5-1: the tracker had lost the second Scuttlecrab, so no copy was on record;
        # the carry's arrival rule swaps a 1-2 cost Riftbeast that turns on nothing else)
        from tft_player.planner import tier_steps
        from tft_player.comp_rules import rules
        carry = (rules().get('carry') or (None,))[0]
        marks = {k[1] for k in getattr(self.roster, 'marks', None) or [] if k[0] == 'Alpha Mark'}
        now = tier_steps(named)
        spare = [u for u in named if u.name != carry and u.id not in marks and u.slots == 1 and not any(
                 i and i != 'unknown' and i not in COMPONENTS for i in u.items or ())
                 and tier_steps([x for x in named if x is not u]) == now]
        spare.sort(key=lambda u: ((u.info.cost if u.info else 0) * 3 ** ((u.star or 1) - 1), u.id))
        return (dups + unknown + spare or [None])[0]

    def bench_bars_drawn(self):
        """A planning phase's bench health bars are up (at its start they come a moment after the bench units show):
        E on a bench unit before that sells nothing (bench_sell_blocked)."""
        frame = getattr(self, 'last_frame', None)
        if frame is None or REPLAY:
            return frame is not None
        if not any(u.loc[0] == 'bench' and not u.missing for u in self.roster.units):
            return True
        return any(B.bar_places(frame, exclude=self.watcher.tac[1])[0])

    def decide(self, scr, planning, frame):
        """The model's round plan when there is one; local rules only when it is
        late (or no model is connected). With a model, nothing spends gold or
        moves units except the plan: a local action while the plan is being
        made leaves the plan built on a stale state."""
        if self.pilot:
            return self.pilot_decide(scr, planning, frame)
        if DATA_MODE:
            return self.data_decide(scr, planning)
        if not self.planner_on():
            return self.brain.decide(scr, self.roster, planning)
        if not planning:
            return self.combat_decide(scr)
        if not Player.opening_inventory(self, scr, frame):
            return None
        Player.observe_wisp_shop(self, scr)
        if getattr(scr, 'shop_visible', False):
            Player.riftbeast_shop(self, scr)     # the round's first full shop: overrun with Riftbeasts? (x29.7)
        if self.plan_asked != scr.stage and self.next_plan:
            self.start_next_plan(scr)                # a queued next_round plan runs first: no model call to wait for
        if self.plan_asked != scr.stage and self.round_plan is None:
            act = self.outside_comp_sale(scr)        # owner e142: the initial obsolete bench before its question
            if act:
                return act
        # Dispatch is asynchronous: carried owned work must not hide the current
        # round's question until an item finally succeeds (game 64 4-3).
        if self.plan_asked != scr.stage:
            if not self.start_next_plan(scr):
                self.request_plan(scr, frame, 'plan')
        Player.mirror_sync(self, scr)
        act = Player.mirror_next(self, scr)      # a flip under way finishes first (game 79 4-6)
        if act:
            return act
        if getattr(self, '_repair_pending', False):
            self._repair_pending = False
            reason = self.need_survey
            if reason:
                self.targeted_check(scr, reason)
                if self.need_survey == reason and not getattr(self, 'survey_places', None):
                    self.need_survey = None
            return None                         # retry the move on the corrected records next tick
        after = getattr(self, '_ask_after_loot', None)
        if after and after[0] == scr.stage and not self.plan_pending and (scr.timer is None or scr.timer > 4):
            # a next_round written in the monster round runs as this round's plan, but the loot that came since was
            # not in it: the stage's first round gets its question at once, the next_round running meanwhile (run
            # 20261022-225248 4-1: 3-7's next_round - one free roll, Cassiopeia to a corner, no XP - ran; three Rods,
            # two Tears, a Cloak and a Morellonomicon came with the loot; the update went out only 10 s in, after an
            # event and the handover moves, and was answered after the phase: 4-1 fought at level 7 on 65 gold with
            # 6 components on the bench and Cassiopeia's items unbuilt, 87 -> 76)
            self._ask_after_loot = None
            self.request_plan(scr, frame, 'update', f'stage {scr.stage} opened after the monster round: your next_round '
                                                    f'plan from {after[1]} runs, but the loot that came since (items, units) '
                                                    f'was not in it - plan this round now (XP, items, board)',
                              interrupt=False)
        rp = self.round_plan
        # Observe/dispatch questions before owned work; this path spends nothing.
        # Owned placements precede accepted offers, which must land before item retries or refreshes.
        if getattr(scr, 'shop_visible', False) and getattr(scr, 'shop_raw', None):
            Player.wisp_event(self, scr, frame, observe_only=True)
        if (getattr(self, 'pending_events', None) and rp and not self.plan_pending
                and (scr.timer is None or scr.timer > 4) and not self.wake_waits(rp, scr)):
            ev = '; '.join(self.pending_events)
            self.log(ev='event_sent', stage=scr.stage, what=ev)
            self.request_plan(scr, frame, 'update', ev, interrupt=False)
            rp = self.round_plan
        # the comp's carry on the board with his items on a stand-in: the Item Remover and his gives come before the
        # phase's other steps, and a handover once started runs to its end (the owner, game 88 4-1, x32.4: "after
        # fielded the brambleback ... the holder of the items (mama beak) should be removed instantly and give to the
        # brambles. Make this to the harness level changes, once bramble is fielded, next action is to use remover")
        act = self.handover_action(scr) if self.handover_rp is not None else None
        if act is None and getattr(scr, 'shop_visible', False):
            act = self.carry_arrival_action(scr)
        if act == self.HANDOVER_WAIT:
            return None
        if act:
            return act
        rp = self.round_plan
        if rp:
            rp.hold_roll = getattr(self, 'wisp_hold', False)
            self.inject_gives(rp)
            act = rp.board_step(self.roster, self.team_cap(scr), scr, gives=False)
            if act:
                return act
        act = self.flag_wisp(scr, frame) or self.wisp_event(scr, frame)
        if act:
            return act
        if Player.wisp_must(self, scr):          # the round's Wisp is not forgotten (x30.14): its auto-buys, then the
            #                                     roll - which waits while the plan still buys from this shop
            act = self.auto_action(scr) or Player.shop_unlock_action(self, scr) or self.executor_roll(scr)
            if act:
                return act
        if rp:
            act = rp.board_step(self.roster, self.team_cap(scr), scr,
                                dependencies=not (getattr(self, 'plan_pending', False) and rp.p.get('_interrupted')))
            if act:
                return act
        act = self.outside_comp_sale(scr)
        if act:
            return act
        if getattr(self, 'trait_pending', None) and time.time() - self.round_started > 2.5:
            self.trait_check(frame, scr)             # a difference found when nothing could be read: first thing now
            return None
        act = self.stage1_buy(scr)
        if act:
            return act
        act = self.flag_wisp(scr, frame)         # a Wisp the player's flag decides: no question (request e184)
        if act:
            return act
        act = self.wisp_event(scr, frame)        # a Wisp: its own fast question; bought first on "buy"
        if act:
            return act
        rp = self.round_plan
        # ask on the first planning tick, with whatever this tick read: no waiting for unit reads,
        # gold or HP: unread facts are marked unknown. One screen read is ~0.35 s
        # and happens on every tick anyway. Run26 asked 3.5-19 s into the round.
        if self.plan_asked != scr.stage:
            if not self.start_next_plan(scr):
                self.request_plan(scr, frame, 'plan')
            rp = self.round_plan
        # A reply may have landed since the first board step. Owned ids and
        # gives precede sales, standing purchases and comp rewrites too (e187).
        if rp:
            self.inject_gives(rp)
            act = rp.board_step(self.roster, self.team_cap(scr), scr,
                                dependencies=not (getattr(self, 'plan_pending', False) and rp.p.get('_interrupted')))
            if act:
                return act
        act = self.roll_arrival_sale(scr)
        if act:
            return act
        act = self.carried_sell(scr)             # the last fight's after-fight sells it would not take
        if act:
            return act
        change = self.state_change(scr)          # keeps its baseline current in every case
        if change:
            # an update question for the model (WAKE) that never stops the running plan (at 3-5 a false
            # "new unit" that interrupted the plan stopped XP x8 and the roll-down)
            self.news_event('game_change', scr.stage, change)
        rp = self.round_plan
        act = self.auto_action(scr)              # the model's standing orders: auto-buy / auto-sell
        if act:
            return act
        act = Player.executor_level(self, scr)   # a level the gold pays keeping 50 (x32.3), or the early level 7 (x32.8)
        if act:
            return act
        act = self.handover_action(scr)          # ... and items to a final holder that arrived
        if act == self.HANDOVER_WAIT:
            return None                          # (the idle step names the items it waits for)
        if act:
            return act
        act = self.carry_arrival_action(scr)     # the comp's carry arrived: off-comp holders sold, his items (x28.1)
        if act:
            return act
        act = self.executor_roll(scr)            # the owner's rolls with no model decision (x28.5, x28.9)
        if act:
            return act
        act = self.mirror_back_action(scr)       # last fight's mirror positions: back to normal while the model thinks
        if act:
            return act
        act = Player.shop_unlock_action(self, scr) or Player.shop_lock_action(self, scr, frame)  # x30.6
        if act:
            return act
        rp = self.round_plan
        wr = getattr(self, 'wisp_refreshed', None)
        if wr and getattr(self, 'wisp_refreshed_stage', scr.stage) != scr.stage:
            self.wisp_refreshed = wr = None      # last round's shop: the round's own plan runs
        if wr and not self.plan_pending and (scr.timer is None or scr.timer > 4):
            # a Wisp the model chose to buy rerolled the shop: it decides on the new shop (spending waits,
            # placing units goes on). A fast roll-down's own Wisp buys never come here: no question there
            self.wisp_refreshed = None
            self.log(ev='wisp_refresh_call', stage=scr.stage, what=wr)
            self.request_plan(scr, frame, 'update', f'{wr} - decide on the new shop', interrupt=True)
            rp = self.round_plan
        if self.pending_events and rp and not self.plan_pending and (scr.timer is None or scr.timer > 4)                 and not self.wake_waits(rp, scr):
            # an event (a container appeared, a unit arrived or left, a record corrected) wakes the model;
            # the running plan keeps going meanwhile
            ev = '; '.join(self.pending_events)
            self.log(ev='event_sent', stage=scr.stage, what=ev)
            self.request_plan(scr, frame, 'update', ev, interrupt=False)
        if rp:
            rp.hold_roll = getattr(self, 'wisp_hold', False)
        if rp and not self.plan_pending and self.review_due(rp, scr, plan_done=False):
            return None
        if rp and self.plan_pending and rp.p.get('_interrupted'):
            # waiting for the model's answer to the change: nothing that spends, but keep
            # placing units as the old plan wanted (else the round can end with a hole in the board). Only the
            # moves are asked for: a step that spends, handed out and dropped here, counted as done (run 213247 3-5)
            return rp.board_step(self.roster, self.team_cap(scr), scr)
        if scr.level:
            self.last_level = scr.level
        if rp:
            self.inject_gives(rp)
            self.prepare_comp(scr, rp)
            act = rp.next_action(scr, self.roster, self.team_cap(scr))
            self.tell_plan_notes(rp, scr)
            if act:
                self.brain.status['plan'] = (f"{'CARRY-OVER ' if rp.carry else ''}"
                                             f"{'PRE-PLANNED ' if getattr(rp, 'pre', None) else ''}ACT {act}")
                return act
            act = self.comp_carry_action(scr, rp)
            if act:
                return act
            act = self.comp_seventh_action(scr, rp)
            if act:
                return act
            act = self.board_recheck_action(scr, rp)
            if act:
                return act
            rp.steps_done = True                 # nothing of the plan to do now (the Wisp roll's wait, executor_roll)
            act = self.mirror_action(scr)        # the mirror positions just before the fight (the owner, x29.10)
            if act:
                return act
            if rp.hold_roll and rp.roll_pending():
                self.brain.status['plan'] = 'roll waits for the Wisp answer'
                return None                      # not finished: the roll is still to come
            # plan finished: the model is asked again only if it asked for it (next_review)
            if not self.plan_pending and self.review_due(rp, scr, plan_done=True):
                return None
            self.brain.status['plan'] = 'plan done' if not rp.carry else 'carry-over done, waiting for the plan'
            if not rp.carry:
                self.plan_idle(rp, scr)
            return None
        # no fallback (the user, 2026-10-02): a late plan is a design problem to fix, never covered
        # by local rules - at 4-3 in run26 their swaps undid the model's board
        self.brain.status['plan'] = 'waiting for the model'
        return None

    def pilot_decide(self, scr, planning, frame):
        """Pilot mode: name new units first (a comp unit from a carousel or an orb only
        counts once named), open containers, then the pilot's action."""
        if planning and getattr(self, '_streak_read', None) != scr.stage:
            st = self.reader.streak(frame)          # the economy plan counts streak gold
            if st:
                self._streak_read = scr.stage
                self.pilot.streak = st[0] if st[1] == 'win' else -st[0] if st[1] == 'loss' else 0
        if planning and self.checked_stage != scr.stage and time.time() - self.round_started < 9:
            self.brain.status['identify'] = 'checking the board against the trait panel first'
            return None                          # check_roster runs once the units have settled
        if planning:
            unknown = [u for u in self.roster.units if u.name is None and self.tries.get(u.id, 0) < 2]
            if unknown and (scr.timer is None or scr.timer > 6):
                self.brain.status['identify'] = f'naming {len(unknown)} unit(s) first'
                return None                  # the idle step right-clicks them
            slot = next((i for i, o in self.bench_objs.items() if o['kind'] == 'interactable'), None)
            if slot is not None and not self.choice and (scr.timer or 0) > 8 and \
                    (slot, scr.stage) not in getattr(self, '_opened', set()):
                self._opened = getattr(self, '_opened', set()) | {(slot, scr.stage)}
                return Action('container', slot, f"pilot: open the {self.bench_objs[slot]['cls']}")
        if scr.shop_visible and scr.gold is not None:
            self.hud_seen = scr
        act = self.pilot.decide(scr, self.roster, planning)
        if planning and not self.wisps_bought.get(scr.stage, 0):
            buy = self.pilot_wisp(frame, scr)    # asks about a new Wisp at once (the model takes seconds)
            act = act or buy                     # a Wisp may reroll the shop: the pilot's buys go first
        return act

    def pilot_wisp(self, frame, scr):
        """Wisps in the shop: the model decides buy or skip, one question per Wisp and round.
        Returns the buy once it said yes (and the gold is there)."""
        from tft_player.coach import _cli
        if not _cli()[0]:
            return None                          # no model, no Wisps
        for slot, name, cost in self.wisp_offers(frame, scr):
            key = (scr.stage, name)
            ans = self.wisp_answers.get(key)
            if ans is None:
                c = self.pilot_call
                if c and c['answer'] is None and time.time() - c['t0'] < 60:
                    return None                  # one question at a time
                self.wisp_answers[key] = 'asked'
                prompt = (f"DECISION: Wisp, stage {scr.stage} (~{scr.timer or '?'}s left). Shop slot {slot}: "
                          f"{name}, costs {cost if cost is not None else '?'} gold. Effect: "
                          f"{self.wisp_effects().get(name, 'card text not read; not in the reference')} "
                          f"(N = the value on the card). One Wisp can be bought per round.\n\n"
                          f"STATE\n{self.pilot_state(scr)}\n\n"
                          'Reply {"buy": true} or {"buy": false}, then {"why": "..."}.')

                def done(decision, key=key):
                    self.wisp_answers[key] = (decision or {}).get('buy') is True
                self.pilot_ask('wisp', f'{name} ({cost if cost is not None else "?"} gold)', prompt,
                               self.screen_jpeg(), done)
                return None
            if ans is True and scr.gold is not None and (cost or 0) <= scr.gold:
                self.wisp_answers[key] = False   # bought once
                return Action('buy', slot, f'wisp {name} for {cost} gold (the model: buy)')
        return None

    def wisp_effects(self):
        """Wisp name -> effect text (data/reference/wisps.md). Its tables differ: the effect column is
        the one headed 'Effect' or 'Card text' (the second table has the cost seen in between)."""
        if not hasattr(self, '_wisp_fx'):
            self._wisp_fx = {}
            path = Path(__file__).resolve().parent.parent / 'data' / 'reference' / 'wisps.md'
            col = 1
            for row in path.read_text(encoding='utf-8').splitlines() if path.exists() else []:
                if not row.startswith('|'):
                    continue
                cells = [c.strip() for c in row.strip().strip('|').split('|')]
                if cells[0] == 'Wisp':
                    col = next((k for k, c in enumerate(cells) if c.startswith(('Effect', 'Card text'))), 1)
                elif len(cells) > col and cells[0] != '---' and 'on the card in the screenshot' not in cells[col]:
                    self._wisp_fx[cells[0]] = cells[col]
        return self._wisp_fx

    def wisp_specials(self, scr, frame):
        """{shop slot: (description, cost)} for the Wisps in the shop: the description and cost read off
        the card (~0.05-0.25 s once per card, then cached by its pixels) - the game's own text, with its
        numbers, valid after any patch. The Wisp table only when no text is found on the card."""
        from tft_player.pilot import wisp_names
        from tft_player.screen import match_name
        if not hasattr(self, '_wisp_names'):
            self._wisp_names = wisp_names()
        specials = {}
        for i in range(5):
            if not (scr.shop_raw[i] and not scr.shop[i]):
                continue
            desc, cost = self.reader.special_card(frame, i) if frame is not None else ('', None)
            if desc:
                specials[i] = (f'card text: "{desc}"', cost)
                continue
            raw = scr.shop_raw[i]
            name = match_name(raw, self._wisp_names, cutoff=0.75)
            if not name and len(raw) >= 4:
                # a long title is cut off in the name box ("Bunch-o" for Bunch-o'-Belts)
                pre = [n for n in self._wisp_names if n.lower().startswith(raw.lower())]
                name = pre[0] if len(pre) == 1 else None
            fx = self.wisp_effects().get(name) if name else None
            specials[i] = ((f'{name} (Wisp table): "{fx}"' + (' (N = a value not read from the card)'
                                                              if re.search(r'\bN\b', fx) else '')
                            if fx else 'card text not read'), cost)
        return specials

    WISP_WAIT = 15.0     # s: the longest the plan's roll waits for a Wisp answer (~8 s to the first word at 4-3)
    WISP_LATE = 12       # s left: a roll no longer stops or waits for a Wisp question. The answer takes 5-9 s, and
                         # the rolls left after it ran at timer 2 (run 20261030-101714 3-5: the level-7 roll stopped
                         # after 1 roll at timer ~10 for Stand Alone; Murkwolf and Scuttlecrab came too late to field)

    @classmethod
    def wisp_late(cls, timer, spent=0.0):
        """True when the planning phase has WISP_LATE s or less left (`timer` read `spent` s ago)."""
        return timer is not None and timer - spent <= cls.WISP_LATE

    SHOP_STILL = 2                       # frame pairs a rolled shop's name strips stay unchanged before it is read

    @staticmethod
    def wisp_name(raw):
        """The Wisp a shop card's title names (data/reference/wisps.md), or None; a long title is cut off in the name
        box ("Bunch-o" for Bunch-o'-Belts)."""
        from tft_player.pilot import wisp_names
        from tft_player.screen import match_name
        names = Player.__dict__.get('_WISP_NAMES') or wisp_names()
        Player._WISP_NAMES = names
        if not raw:
            return None
        name = match_name(raw, names, cutoff=0.75)
        if not name and len(raw) >= 4:
            pre = [n for n in names if n.lower().startswith(raw.lower())]
            name = pre[0] if len(pre) == 1 else None
        return name

    def flag_choice(self, raw):
        """'buy' | 'skip' by the player's Wisp flag for the card titled `raw`, or None (asked as before)."""
        from tft_player import wisp_flag as WF
        flag = getattr(self, 'wisp_flag', None)
        return WF.decide(flag, Player.wisp_name(raw)) if flag else None

    def default_wisp_flag(self, stage):
        """Owner e210/x14.8, stage-five combat Wisps are the comp's default standing order - from stage 4 (the owner,
        game 74b 4-3: the rolls are not stopped to ask, x26.15)."""
        if (stage and int(stage.split('-')[0]) >= 4 and not getattr(self, 'wisp_flag', None)
                and not getattr(self, '_wisp_flag_set', False)):
            from tft_player.coach import forced_comp
            if forced_comp() is not None:
                self.wisp_flag = {'combat': 'buy'}

    def set_wisp_flag(self, spec, stage):
        """strategy.wisp_flag (or a plan's or next_round's): kept until the player changes it, like fight_goal; a key
        the reply leaves out stays as it was (request e184)."""
        from tft_player import wisp_flag as WF
        if spec is None:
            return
        self._wisp_flag_set = True
        new, errs = WF.parse(spec, getattr(self, 'wisp_flag', None))
        for e in errs:
            self.news_event('wisp_flag', stage, e)
        if new != getattr(self, 'wisp_flag', None):
            self.wisp_flag = new
            self.log(ev='wisp_flag', stage=stage, flag=new)

    def wisps_flagged(self, scr, specials):
        """{shop slot: note} for the Wisps in the state that the flag decides: kept out of execution_plan.wisps."""
        out = {}
        for i in specials:
            d = Player.flag_choice(self, scr.shop_raw[i])
            if Player.explicit_wisp_choice(self, scr, i) is False:
                d = 'skip'
            if d:
                acquired = getattr(self, 'wisp_flagged', {}).get((scr.stage, i, scr.shop_raw[i])) == 'buy'
                out[i] = ('acquired by your Wisp flag' if acquired else
                          'authorised by your Wisp flag; purchase not yet confirmed' if d == 'buy'
                          else 'skipped by your Wisp flag')
        return out

    def flag_wisp(self, scr, frame):
        """A Wisp in the shop that the player's flag decides (request e184; game 59 5-2 and 6-2 rolled past Living
        Soil, Barrier, Hand Of Baron and Aftershock unasked under WISP_LATE): bought at once when the gold covers it,
        or left - with no question, at any time left. Told in the next state. -> the buy, or None."""
        if not getattr(scr, 'shop_visible', False):
            return None
        Player.default_wisp_flag(self, scr.stage)
        # stage 5: a Wisp is a must - any kind, whatever the flag or a plan says (the owner, game 78 x28.9: "wisp is
        # a must. Even no model call"; "5-3 wisp should be bought automatically, but ignored" - Greater Chaos was
        # skipped as an economy Wisp)
        must = Player.wisp_must(self, scr)
        if not getattr(self, 'wisp_flag', None) and not must:
            return None
        done = self.__dict__.setdefault('wisp_flagged', {})
        for i in range(5):
            raw = scr.shop_raw[i]
            if not raw or scr.shop[i] or done.get((scr.stage, i, raw)) in ('buy', 'skip', 'pending'):
                continue
            if getattr(self, 'wisp_q', {}).get((scr.stage, i, raw), {}).get('done'):
                continue
            explicit = Player.explicit_wisp_choice(self, scr, i)
            if explicit is False and not must:
                continue
            d = Player.wisp_must_choice(self, scr, raw, explicit) if must else Player.flag_choice(self, raw)
            if d is None:
                continue
            name = Player.wisp_name(raw)
            if d == 'buy':
                if scr.gold is None:
                    continue                     # no gold reading yet: the next tick
                cost = self.reader.shop_cost(frame, i) if frame is not None else None
                if cost is None or cost <= scr.gold:
                    done[(scr.stage, i, raw)] = 'pending'
                    self.log(ev='wisp_flag_buy', stage=scr.stage, slot=i, wisp=name, cost=cost)
                    self.news_event('wisp_flag', scr.stage, f'Wisp {name} authorised by your Wisp flag; purchase pending')
                    action = Action('buy', i, f'wisp {name} (your Wisp flag: buy)')
                    action.wisp_key = (scr.stage, i, raw)
                    action.wisp_offer = Player.wisp_offer_id(self, action.wisp_key)
                    return action
                done[(scr.stage, i, raw)] = 'short'
                self.news_event('wisp_flag', scr.stage, f'Wisp {name} not bought by your Wisp flag: it costs {cost}, '
                                                        f'gold {scr.gold}')
                continue
            done[(scr.stage, i, raw)] = 'skip'
            if must:
                skips = self.__dict__.setdefault('wisp_skips', {})
                skips[scr.stage] = skips.get(scr.stage, 0) + 1
            self.log(ev='wisp_flag_skip', stage=scr.stage, slot=i, wisp=name)
            self.news_event('wisp_flag', scr.stage, f'Wisp {name} skipped by your Wisp flag'
                            + (' - the executor rerolls for the next Wisp (up to 2 rolls)' if must else ''))
        return None

    def wisp_must(self, scr):
        """A Wisp every round with a fixed comp from its roll level (8) and in stage 5, while none is bought this round
        (the owner, game 78: x28.9 "wisp is a must ... it has to reroll for wisp in stage 5 and buy it"; x29.7 "after
        level up, every round, it should buy the wisp, if the model does not have time enough to answer yes, it should
        buy automatically")."""
        from tft_player.coach import forced_comp
        from tft_player.comp_rules import roll_level
        if forced_comp() is None or self.wisps_bought.get(scr.stage, 0):
            return False
        try:
            stage5 = int(str(scr.stage).split('-')[0]) >= 5
        except (TypeError, ValueError):
            return False
        level = getattr(scr, 'level', None) or getattr(self, 'last_level', None) or 0
        return stage5 or level >= (roll_level() or 8)

    WISP_DEADLINE = 6                   # seconds left from which a Wisp the model has not decided is bought (x29.7)
    WISP_ROLL_WAIT = 12                 # seconds left from which the Wisp roll no longer waits for the plan's buys

    def wisp_must_choice(self, scr, raw, explicit):
        """'buy' | 'skip' | None (not yet) for a Wisp of a round that must buy one (wisp_must; the owner, x29.7):
        the model decides - its plan's answer for the card, else its standing Wisp flag; a skip rerolls up to 2 times
        for the next Wisp (executor_roll), and the one showing after those rolls is bought. With no decision the buy
        waits for the model while it can: it is made once the round's plan is in without a word on it, or from
        WISP_DEADLINE seconds left."""
        d = 'buy' if explicit is True else 'skip' if explicit is False else Player.flag_choice(self, raw)
        tries = self.__dict__.setdefault('executor_rolls', {})
        if d == 'skip' and tries.get((scr.stage, 'after_skip'), 0) >= 2:
            return 'buy'
        if d is not None:
            return d
        rp = getattr(self, 'round_plan', None)
        answered = rp is not None and rp.stage == scr.stage and not getattr(self, 'plan_pending', False)
        late = scr.timer is not None and scr.timer <= Player.WISP_DEADLINE
        return 'buy' if answered or late else None

    def riftbeast_shop(self, scr):
        """Whether this round's shop opened overrun by the comp's trait (Riftbeast 5: "every 3 combats, your next shop
        is overrun with Riftbeasts"): every card a champion of the trait - noted the round's first full shop."""
        from tft_player.comp_rules import rules
        seen = self.__dict__.setdefault('_shop_kind', {})
        if scr.stage not in seen and all(scr.shop or [None]) and len(scr.shop or []) == 5:
            t = rules().get('trait')
            seen[scr.stage] = bool(t) and all(CATALOG.unit(n) and t in CATALOG.unit(n).traits for n in scr.shop)
        return bool(seen.get(scr.stage))

    WISP_ROLLS = 12                     # rolls a stage-5 round may spend looking for its Wisp (a safety cap)
    WISP_GOLD = 2                       # the roll's cost: a roll always shows a Wisp, and one can cost 0 (x30.6)

    def executor_roll(self, scr):
        """Rolls the executor makes with no model decision (the owner, game 78):
        - stage 5: while no Wisp is bought this round and none shows, a roll (x28.9: "It has to reroll for wisp in
          stage 5 and buy it"; 5-2 and 5-4 to 5-6 passed with no Wisp) - flag_wisp buys the one that shows;
        - from the comp's roll level (8): a shop with no Wisp and no champion the comp still needs is rolled once per
          planning phase (x28.5: "when there is no wisp and champion in the shop, reroll once automatically, does not
          need the model decision. The best decision should be reroll for bramble 2 star").
        Never with the round plan's own roll still to come, a Wisp question open or under 5 s left. -> the roll, or None."""
        from tft_player.comp_rules import roll_level
        from tft_player.coach import forced_comp
        if (forced_comp() is None or not getattr(scr, 'shop_visible', False) or scr.gold is None or scr.augments
                or scr.timer is None or scr.timer < 5 or REPLAY):
            return None
        rp = self.round_plan
        if rp is not None and rp.stage == scr.stage and rp.roll_pending() and not Player.wisp_must(self, scr):
            return None                          # the plan rolls itself (the round's Wisp roll goes first: x30.14)
        if any(isinstance(k, tuple) and k and k[0] == scr.stage and isinstance(q, dict) and not q.get('done')
               for k, q in (getattr(self, 'wisp_q', None) or {}).items()):
            return None
        skipped = getattr(self, 'wisp_flagged', {})
        if any(scr.shop_raw[i] and not scr.shop[i] and skipped.get((scr.stage, i, scr.shop_raw[i])) != 'skip'
               for i in range(5)):
            return None                          # a Wisp shows: flag_wisp buys it (a skipped one: the roll goes on)
        tries = self.__dict__.setdefault('executor_rolls', {})
        cap = Player.team_cap(self, scr)
        wanted = [n for n in scr.shop if n and CATALOG.unit(n) and CATALOG.unit(n).cost <= scr.gold
                  and self.auto_need(n) > 0
                  and Player.optional_buy_ok(self, n, scr.gold, cap, self.auto_buy.get(n, 1))]
        if Player.wisp_must(self, scr):
            # the owner, x29.7 - a round with no Wisp in its shop rolls once (a roll always shows one: the owner,
            # 2026-11-09); a shop overrun with Riftbeasts rolls up to 2 times; a Wisp the model skipped, up to 2 more.
            # The comp's champions the shop shows are bought first (auto-buy): "buy units needed during it"
            n = tries.get((scr.stage, 'wisp'), 0)
            if wanted:
                return None
            if scr.gold < Player.WISP_GOLD or n >= Player.WISP_ROLLS:
                return None
            if not getattr(self, 'wisp_skips', {}).get(scr.stage, 0):
                # a shop with a champion the comp can use (its units, MetaTFT's level-8 adds) is the plan's first: the
                # roll waits for the round's plan to be done with it ("buy units needed during it"; game 78 4-5; game 83
                # 4-1 and the meta review 83-87: the Wisp roll took a listed Brambleback away). Removed in x30 by a
                # misreading of "the Wisp first" - the owner, 2026-11-10: "I never said this".
                from tft_player.comp_rules import eighth_units
                comp = forced_comp()
                mine = set(getattr(comp, 'units', None) or ()) | set(eighth_units(getattr(comp, 'name', None)))
                useful = [x for x in scr.shop if x and (x in mine or self.auto_need(x) > 0)]
                done = (rp is not None and rp.stage == scr.stage and not getattr(self, 'plan_pending', False)
                        and getattr(rp, 'steps_done', False))
                if useful and not done and (scr.timer or 0) > Player.WISP_ROLL_WAIT:
                    return None
                # a card the comp's orders want that the gold cannot buy yet is never rolled away for a Wisp: the shop
                # is kept (shop_lock_action locks it in the last seconds) - game 83 4-1, Brambleback at 2 gold
                if any(CATALOG.unit(x) and self.auto_need(x) > 0 and CATALOG.unit(x).cost > scr.gold
                       for x in scr.shop if x):
                    return None
            if getattr(self, 'wisp_skips', {}).get(scr.stage, 0):
                k = tries.get((scr.stage, 'after_skip'), 0)
                if k >= 2:
                    return None
                tries[(scr.stage, 'wisp')], tries[(scr.stage, 'after_skip')] = n + 1, k + 1
                return Action('roll', None, f'roll for the next Wisp after a skip ({k + 1} of 2)')
            budget = 2 if Player.riftbeast_shop(self, scr) else 1
            if n >= budget:
                return None
            tries[(scr.stage, 'wisp')] = n + 1
            return Action('roll', None, f"roll for the round's Wisp ({n + 1} of {budget}"
                                        + (', a shop overrun with Riftbeasts)' if budget == 2 else ')'))
        lv = roll_level()
        if (not lv or (scr.level or 0) < lv or tries.get((scr.stage, 'once')) or tries.get((scr.stage, 'wisp'))
                or scr.gold < 2):
            return None                          # (one roll a phase: a Wisp roll made this phase is it)
        if wanted:
            return None
        tries[(scr.stage, 'once')] = 1
        return Action('roll', None, 'no Wisp and no champion the comp needs in the shop: one roll')

    MIRROR_MOVE_S = 0.9                 # one drag of the mirror flip (a swap moves two units): live board moves ~0.84 s

    def mirror_on(self, scr):
        """The mirror positions (the owner, game 78, x29.10: "it is something to change the position of the units to
        avoid the opponent to target us ... before rounds ends use the fastest time to switch positions ... even for
        the earlier rounds, when the comp is not finished"): on with a fixed comp, in the player rounds from stage 2
        (not x-7, the monster round); TFT_MIRROR=0 turns it off."""
        import os
        from tft_player.coach import forced_comp
        if os.environ.get('TFT_MIRROR', '1') == '0' or forced_comp() is None:
            return False
        try:
            a, b = (int(x) for x in str(scr.stage).split('-'))
        except (TypeError, ValueError):
            return False
        return a >= 2 and b != 7

    def mirror_order(self):
        """The board's units in the flip's order: the main tank, then the carry (the owner: "mainly switch main tank
        from 3rd to 5th, and then switch the brambleback with the small blue, and then bring rest to mirror position"),
        then the others right to left."""
        from tft_player import item_plan
        from tft_player.coach import forced_comp
        board = [u for u in self.roster.board() if u.id is not None and not u.missing and u.loc[0] == 'board']
        ip = item_plan.plan_for(getattr(forced_comp(), 'name', None))
        hs = item_plan.holders(board, ip) if ip else {}
        first = [u for u in (hs.get('tank'), hs.get('carry')) if u is not None]
        return first + sorted((u for u in board if u not in first), key=lambda u: (-u.loc[2], -u.loc[1]))

    @staticmethod
    def mirror_drags(moves):
        """The drags a list of (unit, destination) takes: a destination held by the unit whose own destination is the
        mover's hex is one swap."""
        at = {u.id: u.loc for u, _ in moves}
        n = 0
        for u, dst in moves:
            if at[u.id] == dst:
                continue
            other = next((v for v, _ in moves if at[v.id] == dst), None)
            if other is not None:
                at[other.id] = at[u.id]
            at[u.id] = dst
            n += 1
        return n

    def mirror_action(self, scr):
        """The flip to the mirror positions (column c -> 6 - c, the same row) as late in the planning phase as it can
        be, so the board the opponents see is the normal one: once the round's plan is done, when the time left is
        within 3 s of what the flip takes; with less time than it takes, no flip this round. -> the next drag or None"""
        if not Player.mirror_on(self, scr) or scr.timer is None:
            return None
        st = self.__dict__.setdefault('_mirror', {})
        if st.get('stage') != scr.stage:
            if getattr(self, 'plan_pending', False):
                return None                      # a model answer to come can add steps in the flip's seconds
            moves = [(u, ('board', u.loc[1], 6 - u.loc[2])) for u in Player.mirror_order(self) if u.loc[2] != 3]
            if not moves:
                return None
            need = Player.mirror_drags(moves) * Player.MIRROR_MOVE_S + 1.0
            if scr.timer > need + 3:
                return None
            st.clear()
            st.update(stage=scr.stage, moves=moves, orig={u.id: u.loc for u in self.roster.board()},
                      skipped=scr.timer < need)
            if st['skipped']:
                self.log(ev='mirror_skipped', stage=scr.stage, timer=scr.timer, need=round(need, 1))
                return None
            self.board_mirrored = True
            self.mirror_orig = st['orig']
            Player.mirror_sync(self, scr)
            self.log(ev='mirror', stage=scr.stage, timer=scr.timer, need=round(need, 1),
                     moves=[f'{u} -> {d[1:]}' for u, d in moves])
            self.news_event('mirror', scr.stage, 'the board took its mirror positions for the fight (column c -> 6 - c); '
                                                 'the executor puts it back to the normal positions at the start of the '
                                                 'next planning phase - your plans\' hexes are the normal positions')
        return Player.mirror_next(self, scr)

    def mirror_next(self, scr):
        """The next drag of this phase's flip once it started, or None. It goes before anything else in the phase (a
        late model update's sale and roll-down ran between its drags in game 79 4-6, and its last drag, at 2 s, missed
        the board's lock: the fight was half mirrored)."""
        st = getattr(self, '_mirror', {})
        if st.get('stage') != scr.stage or st.get('skipped') or st.get('done'):
            return None
        for u, dst in st['moves']:
            if u.missing or u.loc[0] != 'board' or u.loc == dst or u.id not in st['orig']:
                continue
            if self.roster.at(dst) is not None and self.roster.at(dst).id not in st['orig']:
                continue                         # a unit fielded since stands there: it stays
            a = Action('move', (u, dst), f'mirror position: {u.name} -> {dst[1:]}')
            a.mirror = True
            return a
        st['done'] = True
        return None

    def mirror_back_action(self, scr):
        """At a planning phase's start after a mirrored fight: each unit back to the hex it had before the flip, in the
        flip's order, while the model writes the round's plan. -> the next drag or None"""
        if not getattr(self, 'board_mirrored', False):
            return None
        orig = getattr(self, 'mirror_orig', None) or {}
        st = getattr(self, '_mirror', {})
        if st.get('stage') == scr.stage:
            return None                          # (the flip of this very phase)
        units = {u.id: u for u in self.roster.units}
        ids = list(dict.fromkeys([u.id for u, _ in st.get('moves') or []] + list(orig)))
        for u in (units[i] for i in ids if i in units):
            loc = orig.get(u.id)
            if (loc is None or u.missing or u.loc[0] != 'board' or u.loc == loc
                    or u.loc != ('board', loc[1], 6 - loc[2])):
                continue                         # back already (the middle column never moved: game 79 6-2 dragged
                                                 # Sentinel onto his own (3,3) ten times), or moved since by a plan
            occupant = self.roster.at(loc)
            if occupant is not None and occupant.id not in orig:
                continue
            a = Action('move', (u, loc), f'mirror position back: {u.name} -> {loc[1:]}')
            a.mirror = True
            return a
        self.board_mirrored = False
        self.log(ev='mirror_back', stage=scr.stage)
        return None

    def mirror_sync(self, scr):
        """A round plan of a phase whose board was flipped places its units on the mirror hexes (RoundPlan.mirrored)."""
        on = getattr(self, '_mirror', {}).get('stage') == scr.stage and not getattr(self, '_mirror', {}).get('skipped')
        for rp in (self.round_plan, getattr(self, 'handover_rp', None)):
            if rp is not None:
                rp.mirrored = bool(on and getattr(self, 'board_mirrored', False))

    _LOCK_REF = None

    def shop_locked(self, frame):
        """Whether the shop is locked: the padlock box (board.LOCK_BOX) against the open padlock's picture
        (data/hud/shop_unlocked.png) - True locked, False open, None unreadable."""
        if frame is None:
            return None
        if Player._LOCK_REF is None:
            ref = Path(__file__).resolve().parents[1] / 'data' / 'hud' / 'shop_unlocked-32x16.npy'
            if not ref.exists():
                return None
            Player._LOCK_REF = np.load(ref).astype(np.float32)     # the open padlock, 32x16 grey (no picture shipped)
        h, w = frame.shape[:2]
        x0, y0 = B.scale_point(B.LOCK_BOX[:2], w, h)
        x1, y1 = B.scale_point(B.LOCK_BOX[2:], w, h)
        c = cv2.resize(cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY), (32, 16))
        a = c.astype(np.float32) - c.mean()
        b = Player._LOCK_REF - Player._LOCK_REF.mean()
        corr = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
        return corr < 0.8

    def shop_lock_action(self, scr, frame):
        """The shop lock (the owner, game 79 4-6, x30.6: "after reroll, you could still buy 0 cost wisp, and then if the shop
        has brambleback for example, and could not afford it, it should lock the shop"): in a planning phase's last
        seconds (5 to 2 s left), a champion the comp's orders want (auto_buy) that costs more than the gold left is kept
        for the next round - the shop is locked. In a later phase, once no wanted card is left that the gold cannot
        buy (bought, or gone), a lock of ours is opened again, so the shop refreshes each round. -> the click or None."""
        st = self.__dict__.setdefault('_lock', {})
        if not getattr(scr, 'shop_visible', False) or scr.timer is None or scr.gold is None or frame is None:
            return None
        locked = Player.shop_locked(self, frame)
        if locked is None:
            return None
        dear = [n for n in scr.shop or () if n and CATALOG.unit(n) and self.auto_need(n) > 0
                and CATALOG.unit(n).cost > scr.gold]
        if locked or st.get('ours'):
            return None                          # locked already (ours is opened by shop_unlock_action)
        if 2 <= scr.timer <= 5 and dear and st.get('stage') != scr.stage:
            st.update(stage=scr.stage, ours=True)
            return Action('lock', None, f"lock the shop: {', '.join(dear)} costs more than the {scr.gold} gold left - "
                                        f"kept for the next round")
        return None

    def shop_unlock_action(self, scr):
        """Our lock of the last phase is opened once in this one, after its auto-buys took the kept card and before any
        roll - one click, by our own record (the padlock's look is not needed): the shop refreshes each round again."""
        st = getattr(self, '_lock', {})
        if not st.get('ours') or st.get('stage') == scr.stage or not getattr(scr, 'shop_visible', False):
            return None
        st['ours'] = False
        return Action('lock', None, 'unlock the shop: the round after the lock (its auto-buys first)')

    def explicit_wisp_choice(self, scr, slot):
        from tft_player.planner import same_wisp
        rp = getattr(self, 'round_plan', None)
        if rp is not None and rp.stage == scr.stage:
            for w in rp.p.get('wisps') or []:
                if (isinstance(w, dict) and w.get('slot') == slot
                        and (not w.get('name') or same_wisp(w['name'], scr.shop_raw[slot]))):
                    return bool(w.get('buy'))
        return None

    def wisp_offer_id(self, key):
        offer = getattr(self, '_wisp_live', {}).get(key[1])
        return offer['id'] if offer and offer['key'] == key else None

    def record_wisp(self, **event):
        if hasattr(self, 'log_dir') or 'log' in self.__dict__:
            self.log(**event)

    def observe_wisp_shop(self, scr, expire=False):
        """Retain offer identity across questions/answers; expire it on refresh or combat.

        This only uses this tick's shop text. It performs no input or extra read.
        """
        if not getattr(scr, 'shop_visible', False) or not getattr(scr, 'shop_raw', None):
            return
        signature = (scr.stage, tuple(scr.shop_raw), expire)
        previous = getattr(self, '_wisp_shop_signature', None)
        if previous == signature:
            return
        self._wisp_shop_signature = signature
        refresh = expire or previous is not None and (previous[0] != scr.stage or
                    sum(a != b for a, b in zip(previous[1], signature[1])) >= 2)
        live = self.__dict__.setdefault('_wisp_live', {})
        q = self.__dict__.setdefault('wisp_q', {})
        for slot, offer in list(live.items()):
            key = offer['key']
            if refresh or scr.shop[slot] or scr.shop_raw[slot] != key[2]:
                question = q.get(key)
                if question is not None and question.get('offer_id') in (None, offer['id']):
                    question['done'], question['expired'] = True, True
                Player.record_wisp(self, ev='wisp_offer_expired', offer_id=offer['id'], stage=key[0], slot=slot,
                         wisp=key[2], accepted=(question or {}).get('buy'), acquired=offer.get('acquired', False),
                         observed_stage=scr.stage, timer=getattr(scr, 'timer', None))
                live.pop(slot)
                # A new occurrence of the same title gets a fresh question object.
                # ask_wisp callbacks retain the old object, so cannot answer this one.
                q.pop(key, None)
                getattr(self, 'wisp_flagged', {}).pop(key, None)
        for slot, raw in enumerate([] if expire else scr.shop_raw):
            if slot in live or not raw or scr.shop[slot] or not Player.wisp_name(raw):
                continue
            serial = self.__dict__.get('_wisp_offer_serial', 0) + 1
            self._wisp_offer_serial = serial
            key = (scr.stage, slot, raw)
            live[slot] = {'id': serial, 'key': key}
            if key in q:
                q[key].setdefault('offer_id', serial)
            Player.record_wisp(self, ev='wisp_offer', offer_id=serial, stage=scr.stage, slot=slot, wisp=raw,
                     shop=list(scr.shop_raw), gold=getattr(scr, 'gold', None), timer=getattr(scr, 'timer', None))

    def wisp_event(self, scr, frame, observe_only=False):
        """A Wisp in the shop (planning): asked at once - a question in the game's conversation, ahead of any
        other waiting question (priority, like a choice screen) - with its card text and cost and the state,
        and bought before anything else on "buy". Sells, buys, XP, board and items go on meanwhile;
        only the roll waits (self.wisp_hold), at most WISP_WAIT s. A card counts as a Wisp once it read
        the same on two ticks in a row (a champion card mid-redraw reads as text that is no champion).
        The round's state call decides offers it owns; omitted or later natural
        offers get their own question while live. observe_only dispatches without
        consuming an accepted purchase. Returns the Wisp buy, or None."""
        q = self.__dict__.setdefault('wisp_q', {})
        Player.observe_wisp_shop(self, scr)
        Player.default_wisp_flag(self, scr.stage)
        if getattr(self, '_wisp_screen', None) is not scr:
            self._wisp_screen = scr
            self._wisp_previous_tick = getattr(self, '_wisp_prev', None)
            self._wisp_prev = (scr.stage, list(scr.shop_raw), list(scr.shop))
        prev = self._wisp_previous_tick
        if not scr.shop_visible:
            return None
        for i in range(5):
            raw = scr.shop_raw[i]
            if not raw or scr.shop[i] or not prev or prev[0] != scr.stage or prev[1][i] != raw \
                    or prev[2][i]:
                continue
            key = (scr.stage, i, raw)
            update_due = (getattr(self, 'plan_asked', scr.stage) != scr.stage
                          or (getattr(self, '_ask_after_loot', None) or (None,))[0] == scr.stage
                          or (getattr(self, 'pending_events', None) and not getattr(self, 'plan_pending', False)))
            if (not update_due and key not in q and key not in (getattr(self, 'wisp_flagged', None) or {})
                    and Player.flag_choice(self, raw) is None):
                q[key] = {'asked': time.time(), 'buy': None, 'done': False,
                          'offer_id': Player.wisp_offer_id(self, key)}
                self.ask_wisp(scr, frame, i, key)
        now, hold, act = time.time(), False, None
        for (stage, i, raw), w in list(q.items()):
            if stage != scr.stage or w['done']:
                continue
            here = scr.shop_raw[i] == raw and not scr.shop[i]
            if w['buy'] is None:
                hold = hold or (here and now - w['asked'] < min(self.WISP_WAIT, 8.0)
                                and not self.wisp_late(scr.timer))
            elif w['buy'] and here and act is None:
                hold = True                     # accepted purchases must land before any refresh can expire them
                if (w.get('cost') is not None and (getattr(scr, 'gold', None) is None
                                                  or scr.gold < w['cost'])):
                    continue                     # authorised funding sales can run; no money is invented
                if not observe_only:
                    act = Action('buy', i, f'wisp {raw} (the model: buy)')
                    act.wisp_key, act.wisp_offer = (stage, i, raw), w.get('offer_id')
            else:
                w['done'] = True                 # skipped, or the card is gone
        self.wisp_hold = hold
        return act

    def ask_wisp_now(self, scr, frame, i):
        """A Wisp in a new shop during a roll n times: asked at once; the rest of the rolls waits for the
        answer (wisp_event holds the roll while it is open)."""
        fresh = self.reader.read(frame)
        fresh.stage = fresh.stage or scr.stage
        if not (fresh.shop_raw and fresh.shop_raw[i]) or fresh.shop[i]:
            return                               # a full read names a champion there: no Wisp
        key = (scr.stage, i, fresh.shop_raw[i])
        Player.observe_wisp_shop(self, fresh)
        q = self.__dict__.setdefault('wisp_q', {})
        if key not in q and Player.flag_choice(self, fresh.shop_raw[i]) is None:
            q[key] = {'asked': time.time(), 'buy': None, 'done': False,
                      'offer_id': Player.wisp_offer_id(self, key)}
            self.ask_wisp(fresh, frame, i, key)

    @staticmethod
    def wisps_in_question(wq, stage, specials, shop_raw):
        """{shop slot: note} for the Wisps whose own question (ask_wisp) is still out: a state call keeps them out of
        its execution_plan.wisps. Replacing such an entry with a 'plan' one filed the answer under the plan, so
        wisp_event neither held the roll nor bought it (run 20261103-143833: 11 Wisps answered "buy" and rolled away,
        e.g. 4-3 Blood Money, 5-2 Hand Of Baron)."""
        return {k[1]: 'asked in its own Wisp question, its answer is carried out' for k, w in wq.items()
                if k[0] == stage and w['buy'] is None and w.get('via') != 'plan' and k[1] in specials
                and k[2] == shop_raw[k[1]]}

    def wisps_answered(self, plan, stage, seq=None):
        """A state call's reply decided the Wisps it was shown (execution_plan.wisps; the plan buys them)."""
        ws = {w.get('slot'): bool(w.get('buy')) for w in plan.get('wisps') or [] if isinstance(w, dict)}
        for (st, i, raw), w in list(getattr(self, 'wisp_q', {}).items()):
            if (st == stage and w.get('via') == 'plan' and w['buy'] is None
                    and (seq is None or w.get('plan_seq') == seq)):
                if i not in ws:
                    self.wisp_q.pop((st, i, raw), None)
                    Player.record_wisp(self, ev='wisp_omitted', stage=st, slot=i, wisp=raw, offer_id=w.get('offer_id'))
                    continue                       # omitted is undecided: route a live own question
                w['buy'], w['done'] = ws[i], not ws[i]
                for answer in plan.get('wisps') or []:
                    if isinstance(answer, dict) and answer.get('slot') == i and w.get('cost') is not None:
                        answer['cost'] = w['cost']
                Player.record_wisp(self, ev='wisp_answer', stage=st, slot=i, wisp=raw, buy=ws[i],
                         offer_id=w.get('offer_id'), via='plan')

    def stamp_wisp_offers(self, plan, offers):
        if not isinstance(plan, dict):
            return plan
        for w in plan.get('wisps') or []:
            if isinstance(w, dict) and w.get('slot') in offers:
                w['offer_id'], w['cost'] = offers[w['slot']]
        return plan

    def levelling_lines(self, scr):
        """TFT_ROUND_DATA=T5 (experiments/README.md section 9): our level against the levels MetaTFT's players of the comp
        reach by each stage (its page's levelling) - the fixed or committed comp, else the model's two most likely comps
        (a flex game that never commits never gets the comp's page: run 20261019-044828 was level 6 at 4-1, its comp's
        players 7). Objective data, no advice.
        The climb (TFT_CLIMB=1) always has it, with what the next level costs: level 8 came at 4-5 or later in 6 of
        13 games (one 2nd, the rest 4th-6th), and game 13 reckoned 7 -> 8 at ~40 gold (56 XP: 14 buys, 56 gold)."""
        climb = os.environ.get('TFT_CLIMB') == '1'
        if not (climb or os.environ.get('TFT_ROUND_DATA') in ('T5', 'T6')) or scr.level is None or not scr.stage:
            return []
        from tft_player.coach import climb_settings
        from tft_player.meta import META
        from tft_player.pilot import XP_TO_NEXT
        key = lambda st: tuple(int(x) for x in str(st).split('-'))
        comps = [self.forced] if self.forced else [META.by_name(n) for n, _ in
                                                    sorted(self.comp_p.items(), key=lambda kv: -kv[1])[:2]]
        out = []
        if climb:
            # the climb: no stage -> level schedule against ours in every planning state ("by 4-1: level 7; you: level
            # 6" in all 54 of game 35's) - no level is tied to a round (the owner, 2026-10-28). MetaTFT's levelling stays
            # in the comp brief; the costs below are the arithmetic.
            comps = []
        plan = climb_settings().get('levelling') if climb and self.forced else None
        if isinstance(plan, str) and plan.strip():
            # the comp plan's levels replace MetaTFT's (g15: "3-6: 7, 4-2: 8" every phase - its Fast 9 boards - while
            # the plan that wins this comp's Master+ boards rolls for Elise 3 and Cassiopeia 3 at levels 6-7 first)
            out.append(f'Levels of your comp plan (comp notes; MetaTFT\'s levelling is other boards\'): {plan.strip()} '
                       f'You: level {scr.level}.')
            comps = []
        for c in comps:
            lv = ((getattr(c, 'guide', None) or {}).get('levels') or []) if c else []
            if not lv:
                continue
            now = max((l for st, l in lv if key(st) <= key(scr.stage)), default=None)
            out.append(f"Levelling of {c.name} (MetaTFT, stage -> level): " + ', '.join(f'{st}: {l}' for st, l in lv)
                       + f" - by {scr.stage}: {f'level {now}' if now else f'before level {lv[0][1]}'}; you: level {scr.level}.")
        cur, need = getattr(scr, 'xp', None) or (0, XP_TO_NEXT.get(scr.level))
        if climb and need and scr.level < 10 and need > cur:
            buys = -(-(need - cur) // 4)
            out.append(f'Level {scr.level} -> {scr.level + 1}: {need - cur} XP more = {buys} XP buys ({4 * buys}g).')
            # two levels at once from 6 (demonstration 20261023-090521 4-1: 7 -> 9 in one phase, 25 buys for 94 gold, for
            # Birthday Reunion's level-9 2-star 5-cost; g1, g7 and g20 held a level-9 augment and never reached 9)
            nxt = XP_TO_NEXT.get(scr.level + 1)
            if scr.level >= 6 and nxt and scr.level + 1 < 10:
                two = need - cur + nxt
                buys2 = -(-two // 4)
                out.append(f'Level {scr.level} -> {scr.level + 2}: {two} XP more = {buys2} XP buys ({4 * buys2}g).')
        return out

    def ask_wisp(self, scr, frame, i, key):
        """The Wisp question (the game's conversation, priority): the card's text and cost and the state;
        {"buy": true|false} first."""
        from tft_player import planner
        from tft_player.coach import ask_model
        raw = scr.shop_raw[i]
        desc, cost = self.wisp_specials(scr, frame).get(i, ('card text not read', None))
        self.team_cap(scr)    # the team size the state shows: the level plus what the game's n/m added (Coronation)
        state = planner.state_text(scr, self.roster, self.brain.plan, self.bench_items, self.hp_log,
                                   {i: (desc, cost)}, scr.timer, goal=self.fight_goal)
        state = '\n'.join(l for l in state.splitlines() if not l.startswith('Wisp in shop:'))
        prompt = (f"EVENT: WISP IN THE SHOP, stage {scr.stage}, ~{scr.timer if scr.timer is not None else '?'} s "
                  f"of planning left.\n"
                  f"Shop slot {i}: Wisp '{raw}', costs {cost if cost is not None else '?'} gold. {desc}\n"
                  f"Wisps bought this round: {self.wisps_bought.get(scr.stage, 0)}. Your round plan runs on meanwhile (sells, buys, XP, board, items); its roll waits "
                  f"for this answer.\n\nSTATE\n{state}\n\n"
                  'Reply {"buy": true} or {"buy": false} first, then {"why": "<one sentence>"}.')
        image = self.screen_jpeg()      # review evidence, filtered at the model send gate
        q, t0, got = self.wisp_q, time.time(), {}
        entry = q[key]
        entry['cost'] = cost
        self.log(ev='wisp_asked', stage=scr.stage, slot=i, wisp=raw, cost=cost, desc=desc,
                 offer_id=entry.get('offer_id'))
        print(f'{scr.stage}: Wisp {raw!r} in shop slot {i} ({cost if cost is not None else "?"} gold) -> asking the model')

        def first(obj):
            if isinstance(obj, dict) and 'buy' in obj and entry['buy'] is None:
                entry['buy'] = obj['buy'] is True
                got['t'] = time.time() - t0
                self.log(ev='wisp_answer', stage=scr.stage, slot=i, wisp=raw, buy=entry['buy'],
                         offer_id=entry.get('offer_id'), expired=entry.get('expired', False), via='question')

        def run():
            try:
                text = ask_model(prompt, image, on_first=first, timeout=45, priority=True)
            except Exception as e:
                text = f'FAILED: {e}'
            if entry['buy'] is None:
                from tft_player.coach import json_objects
                for o in json_objects(text or ''):
                    first(o)
            if entry['buy'] is None:
                entry['buy'] = False            # no answer: not bought, and the roll goes on
            self.model_call('wisp', prompt, text, time.time() - t0, got.get('t'))
            print(f"[wisp {time.time() - t0:.1f}s] {'BUY' if entry['buy'] else 'skip'} {raw}: {str(text or '')[:140]}")
        threading.Thread(target=run, daemon=True).start()

    def wisp_offers(self, frame, scr):
        """[(slot, Wisp name, cost)] for the Wisps in the shop (a card whose text is no champion)."""
        from tft_player.pilot import wisp_names
        from tft_player.screen import match_name
        if not hasattr(self, '_wisp_names'):
            self._wisp_names = wisp_names()
        out = []
        for i in range(5):
            if scr.shop[i] is None and scr.shop_raw[i]:
                name = match_name(scr.shop_raw[i], self._wisp_names, cutoff=0.75)
                if name:
                    out.append((i, name, self.reader.shop_cost(frame, i)))
        return out

    def team_cap(self, scr=None):
        """Team slots: the level (the user, 2026-10-09), unless the game's own 'n/m' over the board shows another
        team size - a +team-size augment, a Tactician's Shield, Riftbeast 10 (the user, 2026-10-13). What it adds is
        kept for the frames that don't read it, and a level-up keeps it. The state shows the same number."""
        level = getattr(scr, 'level', None) if scr is not None else None
        previous = max(getattr(self, '_level_floor', 1), getattr(self, 'last_level', None) or 1)
        lvl = max(level or 1, getattr(self, '_level_floor', 1), getattr(self, 'last_level', None) or 1)
        size = getattr(scr, 'team_size', None) if scr is not None else None
        if lvl > previous:
            self._team_cap_stale = previous + max(0, getattr(self, 'team_extra', 0))
        if size and level:
            # The denominator can lag our XP press. It cannot subtract a slot
            # from the level (game 65 3-2: level 6, stale size 5).
            if size >= lvl and size != getattr(self, '_team_cap_stale', None):
                self.team_extra = size - lvl
                self._team_cap_stale = None
        self._level_floor = self.last_level = lvl
        cap = max(1, lvl + max(0, getattr(self, 'team_extra', 0)))
        if scr is not None:
            scr.level = lvl
            scr.team_size = cap
        return cap

    def team_full_notice(self, scr):
        """The game said "You can't exceed the maximum team size": a unit we fielded went back to the bench. The
        team size added over the level came from a misread "n/m" (run 20261030-064325 1-4: "3/3" with a unit over
        it read "375", team size 5 at level 3; Akali and Xayah were "fielded", stayed on the bench, and the 2-1
        fight went with 3 of 4 units): it is dropped, and every unit is read off the screen (a board check)."""
        if getattr(self, 'team_extra', 0) > 0:
            self.log(ev='team_extra_dropped', stage=scr.stage, extra=self.team_extra, level=scr.level)
            self.team_extra = 0
            self._team_cap_stale = None
        self.need_survey = self.need_survey or 'the game said the team is full: a unit we fielded is on the bench'

    def note_own_action(self, act, ok, scr):
        """What our own action changes: the shop (buy, roll), gold (sell) and the
        units. Anything else that changes is the game's doing."""
        self.expect = getattr(self, 'expect', {})
        if not ok:
            return
        now = time.time()
        if act.kind == 'buy':
            self.expect['shop'] = now
            self.buy_check = now + 1.5           # then: does the tracked bench match the screen?
            if scr.shop[act.arg] is None and scr.shop_raw[act.arg]:
                self.wisps_bought[scr.stage] = self.wisps_bought.get(scr.stage, 0) + 1
                # remember the rest of the shop and the gold: the Wisp matters to the plan only
                # if its effect changes them (Starting Town rerolls the shop; a Training Dummy doesn't)
                others = tuple(n for k, n in enumerate(scr.shop) if k != act.arg)
                self.pending_change = (f"Wisp '{scr.shop_raw[act.arg]}' bought", now, act.arg, others, scr.gold)
        elif act.kind == 'roll':
            self.expect['shop'] = now
            Player.observe_wisp_shop(self, scr, expire=True)
        elif act.kind == 'sell':
            self.expect['gold'] = now

    def state_change(self, scr):
        """A significant change the model hasn't seen: a Wisp we bought, the shop
        changing without our buy/roll, gold rising without a sell, a unit we didn't
        buy. Returns a description (and resets), or None."""
        if not scr.shop_visible or scr.gold is None:
            return None
        cur = (tuple(scr.shop), scr.stage, scr.gold)
        last, self.last_seen = getattr(self, 'last_seen', None), cur
        if last and last[1] != cur[1]:
            last = None                  # a new round: income and the new shop are expected
        # our own buy/roll/sell explains a change seen within 2s of it (OCR can lag a tick)
        expect = {k for k, t in getattr(self, 'expect', {}).items() if time.time() - t < 2.0}
        why = []
        pc = getattr(self, 'pending_change', None)
        if pc and time.time() - pc[1] < 1.5:
            return None                  # let the Wisp's effect show (new shop, gold) first
        if pc:
            self.pending_change = None
            label, _, slot, others, gold0 = pc
            now_others = tuple(n for k, n in enumerate(scr.shop) if k != slot)
            effects = []
            if now_others != others:
                effects.append('the shop changed')
            if scr.gold is not None and gold0 is not None and scr.gold > gold0:
                effects.append(f'gold rose to {scr.gold}')
            if effects:
                why.append(f"{label}: {' and '.join(effects)}")
            # a reroll redraws the shop: two read cards differ, not one flicker. And the question is this round's:
            # run 20261022-064841: Cutpurse (no reroll) was taken for one at 4-1 and the flag, never asked while
            # another answer was pending, froze the next round's item gives at 4-2
            if sum(1 for a_, b_ in zip(now_others, others) if a_ and b_ and a_ != b_) >= 2:
                self.wisp_refreshed = f'{label}: it rerolled the shop'
                self.wisp_refreshed_stage = scr.stage
            expect.add('shop')           # the Wisp's own slot showing its champion is not news
        if last and last[0] != cur[0] and 'shop' not in expect:
            why.append('the shop changed (not by our buy or roll)')
        if last and cur[2] > last[2] and 'gold' not in expect:
            why.append(f'gold went up {last[2]} -> {cur[2]} without a sell')
        known = set(getattr(self.round_plan, 'known', ()) or ())
        new = [u for u in self.roster.units
               if known and u.id not in known and not u.ours and u.id not in self.announced]
        if new:
            self.announced |= {u.id for u in new}
            why.append(f"new unit(s) we didn't buy: {', '.join(map(repr, new))}")
        return '; '.join(why) or None

    def screen_extras(self, frame, scr):
        """Everything else the screen shows that the model should know, as state lines."""
        out = []          # the trait-panel check's note: the controller's own work, logged only
        try:
            # the player list by recognition only (its detection OCR took 1-11 s before every plan, 2026-09-29): the
            # other players' HP and who is out, as the screen shows them (request e1, game 35). Covered by a unit's
            # panel: the last read, with its stage.
            ll = self.reader.lobby_line(frame, scr.hp) if frame is not None else None
            if ll:
                self._lobby_read = (scr.stage, ll)
                out.append(ll)
            elif getattr(self, '_lobby_read', None):
                out.append(f'{self._lobby_read[1]} (read at {self._lobby_read[0]}; the list is covered now)')
            st = self.reader.streak(frame) if scr.shop_visible else None
            out.append(f'Streak: {st[0]} {st[1]}' if st else 'Streak: none shown')
            odds = self.reader.shop_odds(frame) if scr.shop_visible else None
            if odds:
                out.append(f'Shop odds by cost 1-5: {odds}')
        except Exception as e:
            self.log(ev='extras_error', error=repr(e))
        gl = self.gift_line(scr.stage)
        if gl:
            out.append(gl)
        il = self.item_gift_line(scr.stage)
        if il:
            out.append(il)
        out += self.item_track.changed_lines(scr.stage)
        objs = getattr(self, 'bench_objs', {})
        if objs:
            out.append('Bench containers (not units): ' + '; '.join(
                f"slot {k}: {o['cls']}" if o['kind'] == 'interactable'
                else f"slot {k}: unrecognised object - not a unit, not a known container"
                for k, o in sorted(objs.items())))
        try:
            if frame is not None and self.phase.planning and time.time() - self.round_started > 4.0:
                pl = B.bar_places(frame, exclude=self.watcher.tac[1])[2]
                seen = {l[1:] for _, l in pl if l[0] == 'board'}
                tracked = {u.loc[1:] for u in self.roster.board() if not u.missing}
                if seen != tracked:          # the controller checks it; not the model's to act on
                    self.log(ev='bars_differ', stage=scr.stage, extra=sorted(seen - tracked),
                             absent=sorted(tracked - seen))
        except Exception as e:
            self.log(ev='extras_error', error=repr(e))
        if self.encounter:
            out.append(f"This game's encounter (1-1 banner): {self.encounter}")
        out.append('Augments taken: ' + (', '.join(self.augments_taken) or 'none'))
        sl = self.slammin_line()
        if sl:
            out.append(sl)
        out.append(f'Wisps bought this round: {self.wisps_bought.get(scr.stage, 0)}')
        out.append(self.auto_line())
        return out

    def slammin_line(self):
        """Slammin' pays 2 XP after a player combat only with no item on the item bench but consumables: which ones
        hold it back now (run 20261023-040039: a Bow or Gloves on the item bench at 3-5, 4-1, 4-2, 4-3, 4-5 - 10 XP)."""
        from tft_player.items import is_consumable, is_reusable
        if not any(str(a).startswith('Slammin') for a in self.augments_taken or ()):
            return None
        # the Alpha Mark stays on the item bench and does not hold it back (run 20261103-233832 6-1 -> 6-2: XP 26 -> 30
        # with only an Item Remover and the Alpha Mark there)
        held = [i for i in self.bench_items or () if i and i != 'unknown' and not is_consumable(i)
                and not is_reusable(i)]
        if not held:
            return None
        return (f"Slammin': no 2 XP after this fight while the item bench holds {', '.join(held)} - give "
                f"{'it' if len(held) == 1 else 'them'} to fielded units before the fight")

    def send_briefing(self, prep=None):
        """Rules, traits, items, augment tiers and every MetaTFT comp. prep: the preparation sent as we entered
        the queue - its answer is taken when it comes; without it (the game was already running) or when it
        failed, the briefing goes out now."""
        from tft_player.coach import ask_model, briefing_text
        if prep is not None:
            prep.join()
            r = prep.result or {}
            if r.get('answer') is not None:
                answer = r['answer'] + (f"\n[cut off: {r['cut']}]" if r.get('cut') else '')
                return self.take_briefing(r['prompt'], answer, r['seconds'], at=r['t'], meta=r.get('meta'))
            print(f"preparation failed ({r.get('error')}): the briefing goes now", flush=True)
        t0 = time.time()
        text, images = briefing_text(), None     # fixed comp: its MetaTFT page as text, no screenshots
        if self.forced:
            self.pages_sent.add(self.forced.name)
        try:
            answer = ask_model(text, images)
        except Exception as e:
            answer = f'FAILED: {e}'
        self.take_briefing(text, answer, time.time() - t0)

    def take_briefing(self, text, answer, latency, at=None, meta=None):
        """The briefing's answer: logged, and the trait choices it names are kept for the choice screen."""
        if self.forced:
            self.pages_sent.add(self.forced.name)
        self.model_call('briefing', text, answer, latency, at=at, meta=meta)
        print(f'[briefing {latency:.0f}s] {str(answer)[:80]}')
        from tft_player.coach import json_objects
        for obj in json_objects(answer if isinstance(answer, str) else ''):
            if isinstance(obj, dict) and isinstance(obj.get('trait_choices'), dict):
                self.trait_choices = obj['trait_choices']
                self.log(ev='trait_choices', choices=self.trait_choices)
            if isinstance(obj, dict) and isinstance(obj.get('strategy'), dict):
                self.update_comps(obj['strategy'], self.phase.stage)        # flex: P(comp) from the preparation

    KEY_STAGES = ()          # reminders to revise a game plan: off (no game plan is asked for)

    def update_comps(self, st, stage=None):
        """Flex: the model's P over comps (the overlay shows it until a commit) and its commit ("commit": a comp
        name) - from then on the game is a fixed-comp game on that comp (commit_comp). Before a commit no comp is
        the target: nothing is bought, sold or handed over for a comp the model has not chosen."""
        if self.forced or not isinstance(st, dict):
            return                       # fixed or committed comp: no P, nothing to choose
        self.take_comp_p(st, stage)
        cm = st.get('commit')
        if isinstance(cm, str) and cm.strip() and cm.strip().lower() not in ('null', 'none'):
            self.commit_comp(cm.strip(), stage)
        elif getattr(self, 'flex', None):
            # the climb's set: the comp whose P reached the set's commit_p is committed (the owner, 2026-11-06: "once
            # certain, commit"; GPT-6.1-Sol held P 0.93-0.99 with "commit": null for rounds - run 20261019-044828 was
            # level 6 at 4-1 without auto_buy or target hexes)
            from tft_player.coach import in_flex_set
            ps = {n: p for n, p in self.comp_p.items() if in_flex_set(n)}
            top = max(ps, key=ps.get) if ps else None
            if top and ps[top] >= self.flex['commit_p']:
                self.commit_comp(top, stage, by=f"P({top}) = {ps[top]:.2f} reached {self.flex['commit_p']:.2f}")

    def commit_due(self, stage):
        """Flex on the climb's set: at the set's commit_by round the comp with the highest P is committed when none
        is yet - the one the most of is held when no P was given (the owner's design, 2026-11-06)."""
        fs = getattr(self, 'flex', None)
        if not fs or self.forced:
            return
        key = lambda s: tuple(int(x) for x in re.findall(r'\d+', str(s or ''))[:2])     # noqa: E731
        if not key(stage) or key(stage) < key(fs['commit_by']):
            return
        from tft_player.coach import in_flex_set
        ps = {n: p for n, p in self.comp_p.items() if in_flex_set(n)}
        if ps:
            top = max(ps, key=ps.get)
            why = f"{fs['commit_by']} reached with no commit: P({top}) = {ps[top]:.2f} is the highest"
        else:
            from tft_player.flex import signals
            units = [(u.name, u.star or 1) for u in self.roster.units if u.name and u.info and not u.missing]
            held = [m.comp.name for m in signals(units, [], list(self.augments_taken or [])) if in_flex_set(m.comp.name)]
            top = held[0] if held else fs['comps'][0].name
            why = f"{fs['commit_by']} reached with no commit and no P given: the set comp most of which is held"
        self.commit_comp(top, stage, by=why)

    def take_comp_p(self, st, stage):
        """The reply's P over comps, normalised (comps + other = 1), for the overlay and the log."""
        if not isinstance(st.get('comps'), list):
            return
        ps = {}
        for c in st['comps']:
            if isinstance(c, dict) and c.get('name'):
                try:
                    ps[str(c['name'])] = max(0.0, float(c.get('p') or 0))
                except (TypeError, ValueError):
                    pass
        other = st.get('other')
        total = sum(ps.values()) + (float(other) if isinstance(other, (int, float)) else 0.0)
        if not ps or total <= 0:
            return
        p = {n: q / total for n, q in ps.items()}          # normalised: comps + other = 1
        hist = self.__dict__.setdefault('comp_p_hist', [])
        hist.append(p)
        del hist[:-3]
        if p != self.comp_p:
            self.comp_p = p
            self.log(ev='comp_p', stage=stage, p=p)

    def commit_comp(self, name, stage, by=None):
        """Flex: the model's commit - or the controller's on the climb's set (by: why, update_comps / commit_due). The
        comp becomes this game's fixed comp, as if the game had started with it (tools/play_comp.py --comp): target
        comp, auto_buy to its units, its brief ahead of the next question, the fixed-comp prompts and panels from then
        on. A later change of comp is a pivot."""
        from tft_player.coach import in_flex_set
        from tft_player.meta import META
        comp = META.by_name(name)
        if comp is None:
            self.news_event('commit_refused', stage, f'"commit": no MetaTFT comp is named {name!r} - nothing is '
                                                     f'committed; use the name exactly as in the briefing')
            return
        if not in_flex_set(comp.name):
            self.news_event('commit_refused', stage, f'"commit": {comp.name} is not in this game\'s comp set '
                                                     f'({", ".join(c.name for c in (getattr(self, "flex", None) or {}).get("comps", []))}) - nothing is '
                                                     f'committed')
            return
        self.forced, self.target, self.committed = comp, comp.name, stage
        os.environ['TFT_FORCE_COMP'] = comp.name         # forced_comp(): every prompt and panel follows it
        # a committed comp opens its anvils by itself (auto_open_anvil: a fixed comp does) - read once at the game's
        # start, a flex game never did (game 72, x24.8: 6-1's anvil waited on the plan's "open_container")
        from tft_player.coach import auto_open_anvil
        self.auto_open = auto_open_anvil()
        self.auto_buy = comp_auto_buy(comp.units, comp.name)
        self.set_comp(comp.id, 'commit')
        self.log(ev='commit', stage=stage, comp=comp.name, comp_id=comp.id, p=self.comp_p, by=by or 'the model')
        print(f'{stage}: COMMIT {comp.name} (P {self.comp_p})' + (f' by the controller: {by}' if by else ''),
              flush=True)
        self.news_event('commit', stage, (f'the controller committed to {comp.name} ({by})' if by else
                                          f'you committed to {comp.name}')
                        + ': it is the target comp from now on (auto_buy: its 1-4 cost units to 2-star; its MetaTFT '
                          'page, its notes and its own stage strategy come with the next question)')
        from tft_player import coach as _coach
        self._commit_block = _coach.commit_strategies(comp, stage)       # its own strategy from now on (x19.1)
        threading.Thread(target=self.load_commit_brief, args=(comp,), daemon=True).start()

    def load_commit_brief(self, comp):
        """The committed comp's brief - the fixed-comp briefing (its MetaTFT page as text, trait choices) - for the
        next question; our own text page of it when the site's page cannot be read."""
        from tft_player import coach
        t0 = time.time()
        try:
            text = coach.forced_briefing()
        except Exception as e:
            text = f'Target comp for this game: {comp.name} ({comp.id}).'
            self.log(ev='commit_brief_error', comp=comp.name, error=repr(e))
        if "MetaTFT page" not in text:
            f = coach.REFERENCE / 'metatft' / 'comps' / f'{comp.id}.md'
            if f.exists():
                text += '\n\nIts MetaTFT page as text:\n' + f.read_text(encoding='utf-8')
        # the climb's flex set (x19.1): the comp's own notes (its rules), its feedback and its stage strategy from now on
        for f in (coach.comp_notes_file(comp), coach.comp_feedback_file(comp)) if coach.flex_set() else ():
            body = f.read_text(encoding='utf-8').strip() if f.exists() else ''
            if body:
                text += f'\n\n=== {comp.name}: {f.parent.name}/{f.name} ===\n{body}'
        block = getattr(self, '_commit_block', '')
        if block:
            text += '\n\n' + block
        if self.forced is comp:
            coach.commit_brief(text)
            self.pages_sent.add(comp.name)
            self.log(ev='commit_brief', comp=comp.name, chars=len(text), secs=round(time.time() - t0, 1))

    def fetch_pages(self, name):
        """The target comp's MetaTFT pages (5 tabs), captured in the background."""
        from tft_player.metatft_shots import comp_pages
        t0 = time.time()
        try:
            comp_pages(name)
            from tft_player.metatft_shots import comp_brief
            text = comp_brief(name)          # the page as text; the screenshots repeat it
        except Exception as e:
            self.log(ev='pages_error', comp=name, error=repr(e))
            print(f'MetaTFT pages for {name} failed: {e}')
            # MetaTFT's list hides some low-play comps: send our text page of it instead
            from tft_player.coach import REFERENCE
            from tft_player.meta import META
            comp = META.by_name(name)
            f = REFERENCE / 'metatft' / 'comps' / f'{comp.id}.md' if comp else None
            if f and f.exists() and name == self.target and not self.forced:     # fixed comp: screenshots only
                self.pending_pages = (name, f.read_text(encoding='utf-8'))
            return
        if name == self.target:
            self.pending_pages = (name, text)
            self.log(ev='pages_ready', comp=name, secs=round(time.time() - t0, 1))

    def direction_lines(self):
        """Flex, before a commit: what we hold - units, items, emblems, augments - matched against every MetaTFT comp
        (tft_player/flex.py; experiments/README.md section 8: with it the model chose the comps the data supports
        and held back its commit when nothing pointed anywhere)."""
        from tft_player.flex import direction_block
        units = [(u.name, u.star or 1) for u in self.roster.units if u.name and u.info and not u.missing]
        items = ([i for i in (self.bench_items or []) if i and i != 'unknown'] +
                 [i for u in self.roster.units if not u.missing for i in (u.items or []) if i and i != 'unknown'])
        fs = getattr(self, 'flex', None)                 # the climb's flex set: only its comps
        try:
            return ['', direction_block(units, items, list(self.augments_taken or []),
                                        only=[c.name for c in fs['comps']] if fs else None)]
        except Exception as e:
            self.log(ev='direction_error', error=repr(e))
            return []

    def comps_line(self):
        fs = getattr(self, 'flex', None)
        rule = (f" The controller commits a comp when its P reaches {fs['commit_p']:.2f}, and at {fs['commit_by']} the "
                f"comp with the highest P; until then auto_buy collects every set comp's units." if fs else '')
        if fs:                                           # the Strategist's choice file, every state until a commit
            from tft_player.coach import flex_choice
            choice = flex_choice()
            rule += f'\n{choice}' if choice else ''
        if not self.comp_p:
            return 'Comp probabilities: none given yet.' + rule
        ps = ', '.join(f'P({n}) = {p:.2f}' for n, p in sorted(self.comp_p.items(), key=lambda kv: -kv[1]))
        line = f'Your comp probabilities (last reply): {ps}. Not committed yet ("commit" in strategy).'
        # your own P for the leading comp over the last replies, and what waits on the commit (GPT-6.1-Sol held
        # P(Riftbeast Pebbles) 0.93-0.99 for rounds with "commit": null - run 20261019-044828 played without auto_buy,
        # the comp's page or its target hexes)
        hist = getattr(self, 'comp_p_hist', [])
        top = max(self.comp_p, key=self.comp_p.get)
        if len(hist) >= 2:
            line += (f" P({top}) over your last {len(hist)} replies: " + ' -> '.join(f'{h.get(top, 0):.2f}' for h in hist)
                     + ('.' if fs else
                        '. Until a commit: no auto_buy for its units and no target hexes; a commit sends its MetaTFT page.'))
        return line + rule

    def inherit_carry(self, plan):
        """Last round's unfinished steps are never dropped by a newer plan (the user: they run first and are
        completed): a plan without a board or items keeps the carried ones; carried sells and keeps stay."""
        carry = self.round_plan.p if self.round_plan is not None and self.round_plan.carry else None
        if not carry:
            return
        if plan.get('board') is None and carry.get('board'):
            plan['board'] = carry['board']
            plan.setdefault('_requested_moves', {})['board'] = (carry.get('_requested_moves') or {}).get('board', [])
        if plan.get('bench') is None and carry.get('bench'):
            plan['bench'] = carry['bench']
            plan.setdefault('_requested_moves', {})['bench'] = (carry.get('_requested_moves') or {}).get('bench', [])
        if carry.get('items'):
            have = {(e.get('component'), e.get('id')) for e in plan.get('items') or []}
            plan['items'] = list(plan.get('items') or []) + [e for e in carry['items']
                                                             if (e.get('component'), e.get('id')) not in have]
        if carry.get('strip'):
            have = {(e.get('id'), e.get('tool')) for e in plan.get('strip') or []}
            plan['strip'] = list(plan.get('strip') or []) + [e for e in carry['strip']
                                                            if (e.get('id'), e.get('tool')) not in have]
        plan['sell'] = list(dict.fromkeys(list(plan.get('sell') or []) + list(carry.get('sell') or [])))
        plan['keep'] = list(dict.fromkeys(list(plan.get('keep') or []) + list(carry.get('keep') or [])))
        if carry.get('level_ups'):
            # XP held from the last round (the owner, x18.9) is bought at this round's start: a newer plan keeps it
            if int(plan.get('level_ups') or 0) <= int(carry['level_ups']) and carry.get('_level_target'):
                plan['_level_target'] = carry['_level_target']
            plan['level_ups'] = max(int(plan.get('level_ups') or 0), int(carry['level_ups']))

    def start_next_plan(self, scr):
        """The model's next_round plan (written last planning phase) runs as this round's plan at
        once, with no model call; the model is asked when it completes (or at its own next_review)."""
        from tft_player import planner
        if not self.next_plan or self.next_plan[0] == scr.stage:
            return False
        made, plan = self.next_plan
        self.next_plan = None
        target = Phase.next_stage(str(made))
        if target.endswith('-4') and not target.startswith('1-'):
            target = Phase.next_stage(target)            # the carousel has no planning phase
        if scr.stage != target:
            # written for the phase after `made`, which passed without it (the climb's game 26: 4-1's "level 8 and roll
            # at 4-2" - the augment and Coven screens took 4-2's planning phase - ran at 4-3 and bought 4 more XP)
            self.log(ev='next_round_stale', stage=scr.stage, made=made, target=target)
            self.news_event('next_round_stale', scr.stage, f'your next_round plan from {made} was for {target}, which '
                                                           f'passed without it: it did NOT run - plan this round')
            return False
        plan = dict(plan)
        plan.setdefault('next_review', [{'plan_complete': True}])
        self.cap_next_xp(plan, scr, made)
        # last round's unfinished steps go first and are never dropped (the user: 4-2's lineup and items
        # were still undone at 4-6)
        replacing = getattr(self, '_next_replaces', None)
        if replacing is self.round_plan and replacing is not None:
            # Its newer lineup supersedes the old moves. Unfinished gives still
            # follow their existing carry path, and completed spending gets credit.
            replacing.p['board'], replacing.p['bench'] = [], []
            replacing.carry = True
        self.inherit_carry(plan)
        self.plan_asked = scr.stage
        self.round_plan = planner.RoundPlan(plan, scr.stage, known=[u.id for u in self.roster.units],
                                            names=self.next_plan_names, gold=scr.gold)
        self.round_plan.pre = made
        self.round_plan.source_seq = getattr(self, 'next_plan_seq', None) or 0
        if replacing is not None:
            empty = {'levels': 0, 'bought': set(), 'sold': set(), 'itemised': set(),
                     'wisps': set(), 'rolled': 0, 'done': 0}
            self.round_plan.credit(replacing, empty)
            self._next_replaces = None
        if self.plan_pending:
            self.round_plan.pending_snapshot = self.round_plan.snapshot()
            self._pending_fallback = self.round_plan
        # written in the monster round: the loot that lands now was not in it - the round is asked about at once (decide)
        self._ask_after_loot = (scr.stage, made) if str(made).endswith('-7') else None
        self.apply_auto(plan, scr.stage)
        self.set_wisp_flag(plan.get('wisp_flag'), scr.stage)         # a next_round can set the Wisp flag (e184)
        self.plan_rejected(planner.validate(self.round_plan.p, self.roster, self.bench_items, self.round_plan.names), scr.stage)
        self.brain.protected = self.round_plan.protected_ids()
        self.news_event('next_round_started', scr.stage, f'your next_round plan from {made} ran from the start of this '
                                                         f'planning phase (gold {scr.gold}, level {scr.level})')
        self.log(ev='next_round_started', stage=scr.stage, made=made, plan=plan)
        print(f'{scr.stage}: running the next_round plan written at {made}')
        return True

    @staticmethod
    def gives_since(rp, snap):
        """The gives `rp` made after `snap` (what the model saw when asked): [(item, unit id)]; not the ones that did
        not land (carried to the next round) nor Removers."""
        if rp is None or not snap:
            return []
        return [(c, uid) for c, uid, k in sorted(rp.itemised - snap['itemised'], key=repr)
                if c != 'strip' and (c, uid, k) not in rp.item_later]

    def level_unread_check(self, scr, n, gold0):
        """The gold after n XP presses did not read: one full read decides, and the frame is kept. Its gold if read;
        else the level and XP unchanged from before the presses mean none landed (gold0 is returned); else None.
        Run 20261103-143833 6-1 (log 2358): level x14 at timer 29 on 60 gold, with an item anvil's choice opening over
        the shop; the gold went unread, all 14 were counted as landed, the step was never tried again and the level
        ran at timer 1 in combat - the free rerolls had gone at level 8."""
        got = self.cap.read()
        if got is None:
            return None
        fresh = self.reader.read(got[0])
        same = (fresh.level is not None and fresh.level == scr.level
                and getattr(fresh, 'xp', None) and fresh.xp == getattr(scr, 'xp', None))
        shot = f'{scr.stage}-level-unread'
        if shots_on() and getattr(self, 'log_dir', None) is not None:
            from PIL import Image
            Image.fromarray(got[0]).save(self.log_dir / f'{shot}.full.jpg', quality=92)
        self.log(ev='level_unread', stage=scr.stage, presses=n, gold0=gold0, gold=fresh.gold, level0=scr.level,
                 level=fresh.level, xp0=getattr(scr, 'xp', None), xp=getattr(fresh, 'xp', None),
                 shop_visible=getattr(fresh, 'shop_visible', None), shot=shot)
        if fresh.gold is not None:
            return fresh.gold
        return gold0 if same else None

    def confirm_level_buys(self, scr, landed):
        """Gold-confirmed XP opens team slots immediately, even before HUD animation."""
        from tft_player.pilot import XP_TO_NEXT
        level, xp = getattr(scr, 'level', None), getattr(scr, 'xp', None)
        if not isinstance(level, int) or not xp or not xp[1]:
            return
        self.team_cap(scr)                          # retain any extra slots at the pre-purchase level
        extra = max(0, getattr(self, 'team_extra', 0))
        points, need = xp[0] + landed * (getattr(self, 'xp_per_buy', 4) or 4), xp[1]
        before = level
        while need and points >= need:
            points, level = points - need, level + 1
            need = XP_TO_NEXT.get(level)
        scr.level, scr.xp = level, (points, need)
        scr.team_size = level + extra
        self._level_floor = max(level, getattr(self, '_level_floor', 1))
        scr.team_size = self.team_cap(scr)
        if level > before:
            self._team_cap_stale = before + extra
        self.log(ev='level_slots_confirmed', stage=scr.stage, before=before, level=level,
                 buys=landed, team_size=scr.team_size, xp=scr.xp)

    def note_xp_per_buy(self, scr, landed):
        """The XP one buy gives, from our own buys: the HUD's level and XP before and after `landed` buys. An augment
        can change it (the climb's checker, game 25: Level Up! gave 6 a buy while the calculator priced levels at 4,
        so level 7 looked half again dearer). Kept when two buys in a row agree; the calculator uses it."""
        from tft_player.pilot import XP_TO_NEXT
        lv0, xp0 = getattr(scr, 'level', None), getattr(scr, 'xp', None)
        if not landed or not isinstance(lv0, int) or not (isinstance(xp0, (tuple, list)) and xp0):
            return
        got = self.cap.read()
        after = self.reader.read(got[0]) if got else None
        lv1, xp1 = getattr(after, 'level', None), getattr(after, 'xp', None)
        if not isinstance(lv1, int) or not (isinstance(xp1, (tuple, list)) and xp1) or lv1 < lv0:
            return
        total = lambda lv, x: sum(XP_TO_NEXT.get(k, 0) for k in range(1, lv)) + x     # noqa: E731
        per = (total(lv1, xp1[0]) - total(lv0, xp0[0])) / landed
        if per != int(per) or not 2 <= per <= 12:
            return                               # a misread, or a level-up whose XP bar we did not read
        per = int(per)
        seen = getattr(self, '_xp_per_buy_seen', None)
        self._xp_per_buy_seen = per
        if seen == per and per != getattr(self, 'xp_per_buy', 4):
            self.xp_per_buy = per
            self.log(ev='xp_per_buy', stage=scr.stage, xp=per)

    def cap_next_xp(self, plan, scr, made):
        """A next_round's XP buys were counted before this phase's XP was known: buys past the last level they reach
        are dropped - the gold is the level's roll (run 20261023-045645 4-1: "xp 12" written at 3-7; 11 reached level 8
        from 12/56, the 12th took 4 of the 17 gold the roll-down had). Buys that reach no new level stay (gold above
        50 into XP)."""
        from tft_player.pilot import XP_TO_NEXT
        n, lvl, xp = plan.get('level_ups'), getattr(scr, 'level', None), getattr(scr, 'xp', None)
        if not (isinstance(n, int) and n > 1 and isinstance(lvl, int) and isinstance(xp, (tuple, list)) and len(xp) == 2
                and all(isinstance(v, int) for v in xp)):
            return
        pts, need, level, last = xp[0], xp[1], lvl, 0
        for b in range(1, n + 1):
            pts += getattr(self, 'xp_per_buy', None) or 4
            while need and pts >= need:
                pts, level, last = pts - need, level + 1, b
                need = XP_TO_NEXT.get(level)
        if 0 < last < n:
            plan['level_ups'] = last
            self.log(ev='next_round_xp_capped', stage=scr.stage, made=made, xp=n, kept=last, level=level)
            self.news_event('next_round_xp_capped', scr.stage, f'your next_round from {made} bought XP {last} times, not '
                                                               f'{n}: {last} reach level {level}, the rest is roll gold')

    def keep_next_plan(self, plan, stage, done=(), names=None, seq=None):
        """done: the gives the running plan made while the model wrote this reply - the same give in its next_round
        was written for an item that is no longer on the item bench (run 20261023-045645: 4-1's Tear and B.F. Sword went
        onto Cassiopeia in the fight, the next_round restated both; at 4-2 the Tear was rejected, the B.F. Sword - a
        second one, meant for Elise's Bow - went onto Cassiopeia's third slot, the Gunblade was refused there at 4-3 and
        a Deathblade, an AD item, was built on her at 5-1)."""
        nr = plan.get('next_round') if isinstance(plan, dict) else None
        if not isinstance(nr, dict):
            return
        from tft_player import planner
        nr = planner.normalize({'execution_plan': nr})
        if done and nr.get('items'):
            left, kept, dropped = list(done), [], []
            for e in nr['items']:
                key = (e.get('component'), e.get('id'))
                if 'from' not in e and key in left:
                    left.remove(key)
                    dropped.append(f'{key[0]} -> unit {key[1]}')
                else:
                    kept.append(e)
            nr['items'] = kept
            if dropped:
                self.log(ev='next_round_gives_done', stage=stage, dropped=dropped)
                self.news_event('next_round_gives_done', stage, f"your next_round's {', '.join(dropped)} was already "
                                                                f"given in this fight: dropped from it")
        # item-bench presence is checked when it starts, against that round's bench (run27 3-1)
        self.plan_rejected(planner.validate(nr, self.roster, None, self.unit_names()), stage, 'next_round ')
        self.tell_board_after(nr, stage, 'your next_round plan, from the board now: ')
        nr.pop('strategy', None)
        self.next_plan = (stage, nr)
        self.next_plan_seq = seq
        # each id as the model saw it when the question went (names), not when the answer came: a unit named while
        # the model thought keeps the plan's '?' view the loss checks compare with (request e193); ids new since, now
        self.next_plan_names = self.unit_names()
        self.next_plan_names.update(names or {})
        if hasattr(self.next_plan_names, 'items_view'):
            self.next_plan_names.items_view.update(getattr(names, 'items_view', {}))
        self.log(ev='next_round_planned', stage=stage, plan=nr)
        print(f'{stage}: next_round plan kept for the next planning phase: {json.dumps(nr)[:200]}')

    def late_next_round(self, plan, made, names=None, seq=None):
        """A reply whose round ended before it came is dropped - but its next_round was written for the round now
        open (run 20261022-074359 4-3: the fast_rr written in the 4-2 fight came 0.1 s into 4-3 and was lost with the
        rest; the model, told nothing, thought a Wisp had stopped it, and the roll ran 13 s late, into the fight, its
        buys benched: 34 -> 22). It runs while this round's plan is pending and replaces an older pre-plan until that
        script makes a move. Completed steps receive credit when it is replaced."""
        if not isinstance(plan.get('next_round'), dict):
            return
        now = self.phase.stage
        try:
            follows = bool(now) and Phase.next_stage(made) == now
        except (AttributeError, ValueError):
            follows = False
        if not follows:
            return
        rp = self.round_plan
        untouched = rp is None or rp.stage != now or ((getattr(rp, 'pre', None) or rp.carry)
                                                      and not any(d.startswith('move(') for d in rp.done))
        if seq is not None and rp is not None and getattr(rp, 'source_seq', 0) > seq:
            untouched = False
        other_pending = self.plan_pending and getattr(self, 'plan_seq', None) != seq
        if self.phase.planning and untouched:
            if rp is not None and rp.stage == now:
                rp.carry = True
                self._next_replaces = rp
                self._superseded_rp = rp
            self.keep_next_plan(plan, made, names=names, seq=seq)
            self.plan_asked = None               # decide(): start_next_plan installs it on the next tick
            self.log(ev='next_round_late', stage=now, made=made, ran=True, other_pending=other_pending)
            self.news_event('next_round_late', now, f'your next_round plan written at {made} came after {now} had '
                                                    f'opened: it runs now as this round\'s plan')
        else:
            self.log(ev='next_round_late', stage=now, made=made, ran=False)
            self.news_event('next_round_late', now, f'your next_round plan written at {made} came after {now} had '
                                                    f'opened and did NOT run: send what is still wanted in this '
                                                    f'round\'s plan')

    def correct_early_plan(self, rp, plan, stage):
        """The complete answer corrects the plan its first part installed early (run26: a streamed plan cut to 2 of 7
        board units ran the whole round): its lineup parts. That plan, not whatever runs now - an item handover may run
        in its place (run 20261023-010206 4-2: the correction wrote the update's board and items into the handover's
        plan; its Remover took Cassiopeia's 3 items off, its gives were replaced, she fought itemless, 77 -> 63)."""
        if rp is None:
            return
        rp.p['strategy'] = plan.get('strategy')
        for k in ('board', 'items', 'strip', 'keep', '_requested_moves'):
            if plan.get(k) is not None and plan.get(k) != rp.p.get(k):
                self.log(ev='plan_corrected', stage=stage, key=k)
                rp.p[k] = plan[k]

    def late_carousel_work(self, plan, made, names, seq):
        """A -3 answer reached the carousel: retain owned work for its next unlock.

        There is no new planning decision on -4. Game 65's explicit carry package
        (call 68) otherwise disappeared here before the 4-5 carry path even ran.
        No stale spending is resumed, and a newer accepted script wins.
        """
        from tft_player import planner
        now, rp = self.phase.stage, self.round_plan
        if (not now or not now.endswith('-4') or now.startswith('1-') or Phase.next_stage(made) != now
                or rp is not None and (not rp.carry or getattr(rp, 'source_seq', 0) > seq)):
            return False
        remaining, _ = planner.RoundPlan(plan, made, names=names).leftover(self.roster)
        if not remaining:
            return False
        requested = remaining['_requested_moves']
        remaining['board'], remaining['bench'], remaining['sell'] = requested['board'], requested['bench'], []
        if not any(remaining.get(k) for k in ('board', 'bench', 'items', 'strip')):
            return False
        self.inherit_carry(remaining)
        remaining['_source_seq'] = seq
        self.round_plan = planner.RoundPlan(remaining, now, carry=True, names=names)
        self.brain.protected = self.round_plan.protected_ids()
        self.log(ev='carousel_work_carried', stage=now, made=made, source_seq=seq,
                 board=remaining['board'], items=remaining['items'], strip=remaining['strip'])
        return True

    def drop_carried_moves(self, scr):
        """The round's plan is asked for: drop stale generated moves and carried sales.
        Explicit player destinations continue, like next_round's owned work, until
        the new reply replaces them. Game 65 lost Cinderling, Gromp and the carry
        conversion by dropping these too. Sales still stop (game 61 3-6, log 1143-1156: the 3-5 update's
        sells, to fund XP that never ran, sold the spare Cinderling, Pebbles and Scuttlecrab at timer 29-23 while the
        3-6 plan, written with them on the bench, kept them for star-ups). -> True when there were some."""
        rp = getattr(self, 'round_plan', None)
        if rp is None or not getattr(rp, 'carry', False) or not isinstance(rp.p, dict) \
                or not (rp.p.get('board') or rp.p.get('bench') or rp.p.get('sell')):
            return False
        requested = rp.p.get('_requested_moves') or {}
        board, bench = requested.get('board') or [], requested.get('bench') or []
        # a carried sale of a unit on none of the comp's boards goes on (the owner, game 74b 2-6: Caitlyn's 2-5 sale,
        # put off by an away fight, was dropped here and waited for the next answer - "it should be sold instantly");
        # the sales of comp units still stop (game 61, above)
        from tft_player import item_plan
        from tft_player.planner import _fixed_comp
        ip = item_plan.plan_for(getattr(_fixed_comp(), 'name', None)) or {}
        unit = lambda i: next((u for u in self.roster.units if u.id == i and u.name and not u.missing), None)
        keep_sells = [i for i in rp.p.get('sell') or [] if unit(i) is not None and _fixed_comp() is not None
                      and not item_plan.comp_unit(unit(i).name, ip)]
        dropped = rp.p.get('board', []) != board or rp.p.get('bench', []) != bench or             len(rp.p.get('sell') or []) != len(keep_sells)
        rp.p['board'], rp.p['bench'], rp.p['sell'] = list(board), list(bench), keep_sells
        self.log(ev='carry_moves_dropped' if dropped else 'carry_moves_continued', stage=scr.stage,
                 note=getattr(self, 'carry_note', ''), continuing={'board': board, 'bench': bench})
        return dropped

    def opening_inventory(self, scr, frame):
        """Read starting bodies before the first question, one targeted panel per pass."""
        if getattr(self, 'planned_once', False):
            return True
        from tft_player.state import Owned
        bars = B.green_bars(frame)
        _, _, places = B.bar_places(frame, bars=bars, exclude=self.watcher.tac[1])
        places = [(bar, loc) for bar, loc in places if loc[0] in ('bench', 'board')]
        if not places:
            return False                        # an empty transition frame is not known inventory
        attempts = self.__dict__.setdefault('_opening_reads', {})
        pending = []
        for bar, loc in places:
            u = self.roster.at(loc)
            if u is None:
                u = Owned(None, 1, loc)
                self.roster.units.append(u)
            if not u.name or u.unverified:
                pending.append((u, bar))
        if not pending:
            self.log(ev='opening_inventory', stage=scr.stage, status='identified',
                     units=[dict(id=u.id, name=u.name, stars=u.star, loc=u.loc) for u in self.roster.units])
            return True
        for u, bar in pending:
            key = (scr.stage, u.loc)
            if attempts.get(key, 0) >= 2:
                continue
            attempts[key] = attempts.get(key, 0) + 1
            others = [self.unit_point(v.loc, bars) for v in self.roster.units
                      if v.name and v.name != u.name and not v.missing][:1]
            name, star, held = self.read_fresh(self.unit_point(u.loc, bars), others)
            self.log(ev='opening_inventory', stage=scr.stage, status='read', loc=u.loc,
                     name=name, stars=star, items=held)
            if name:
                self.apply_partial_survey([[u.loc, name, star, held]], scr, 'starting inventory')
                here = self.roster.at(u.loc)
                if here is not None and here.name == name:
                    here.items_read, here.items_check, here.items_predicted = True, False, False
            elif getattr(self, 'log_dir', None) is not None:
                self.snap(frame, f'{scr.stage}-opening-inventory-unresolved', True, force=True)
            return False
        return False                            # retry an unread body next phase; never invent its identity

    def request_plan(self, scr, frame, kind='plan', change=None, interrupt=True):
        """kind 'plan': the round's first plan (with last round's leftovers);
        'update': a re-plan with the progress so far (and what changed)."""
        from tft_player import planner
        from tft_player.coach import ask_model
        if not getattr(self, 'planned_once', False) and kind != 'plan':
            return                              # the opening planning question owns the initial inventory read
        if kind == 'update' and Player.container_waiting(self, scr):
            # a container on the bench is opened before the model is asked again: its choice needs the model free, and
            # back-to-back update calls kept it shut all of 7-1 in game 79 (the owner, x30.16: "1. buy the wisp 2.
            # adjust position or items if needed 3. open the anvil and get a second item ... 4. give it")
            self.log(ev='update_after_container', stage=scr.stage, change=str(change)[:200])
            return
        if kind == 'plan' and not Player.opening_inventory(self, scr, frame):
            return
        self.plan_asked = scr.stage
        if not getattr(self, 'rushing', False):
            self.trait_check(frame, scr)     # the board's identities against the trait panel, fixed before asking
        self.plan_pending = True
        self.plan_calls += 1
        # the newest call's answer ends the wait, never an older one's: run 20261023-063338 4-3, the planning update's
        # full reply (+1234.0) cleared the flag while the combat-start call was out; 25 s later a third question went
        # with the state from before that call's roll, and the player read "gold 54" as "the roll at 7 didn't run"
        # (145 questions in 21 games went while one was out, 132 of them from paths meant to wait until none is)
        self.plan_seq = seq = getattr(self, 'plan_seq', 0) + 1
        self.pending_events = []         # the state below lists what they reported
        # no Wisps in combat: they leave the shop when it starts
        specials = self.wisp_specials(scr, frame) if self.phase.planning else {}
        wq = self.__dict__.setdefault('wisp_q', {})
        decided = {k[1]: ('buy' if w['buy'] else 'skip') for k, w in wq.items()
                   if k[0] == scr.stage and w['buy'] is not None and k[1] in specials and k[2] == scr.shop_raw[k[1]]}
        decided.update(self.wisps_in_question(wq, scr.stage, specials, scr.shop_raw))
        decided.update(self.wisps_flagged(scr, specials))
        for i in specials:
            if i not in decided:                 # this reply decides it (execution_plan.wisps): no own question
                key = (scr.stage, i, scr.shop_raw[i])
                wq[key] = {'asked': time.time(), 'buy': None, 'done': False, 'via': 'plan',
                           'plan_seq': seq, 'offer_id': Player.wisp_offer_id(self, key),
                           'cost': specials[i][1]}
        phase = 'planning' if self.phase.planning else 'combat'
        scr.hp_stale = self.phase.planning and getattr(self, '_hp_seen_stage', None) != self.phase.stage
        scr.xp_per_buy = getattr(self, 'xp_per_buy', None)      # the calculator's, when an augment changed it
        scr.item_ages = self.item_ages(scr.stage)               # how long each component has sat on the item bench
        # last round's carried board moves stop here: the model plans on the board its state shows (request e177, run
        # 20261103-233832 6-2: the 6-1 comp rule's "Brambleback -> (2,4)" ran 3 s after the 6-2 question went, benched
        # Taric, Elder Dragon's pair, and the plan written on the board with Taric was made on a board gone)
        moves_dropped = self.drop_carried_moves(scr) if kind == 'plan' else False
        self.team_cap(scr)    # the team size the state shows: the level plus what the game's n/m added (Coronation)
        state = planner.state_text(scr, self.roster, self.brain.plan, self.bench_items, self.hp_log,
                                   specials, scr.timer, self.screen_extras(frame, scr)
                                   + ([] if self.forced else [self.comps_line()])      # fixed comp: no P line
                                   + self.strategy_lines()
                                   + ([] if self.forced or phase != 'planning' else self.direction_lines())
                                   + (self.levelling_lines(scr) if phase == 'planning' else []),
                                   phase=phase, wisps_decided=decided, goal=self.fight_goal,
                                   wisp_flag=getattr(self, 'wisp_flag', None) or {})
        if kind == 'update':
            rp = self.round_plan
            progress = rp.progress() if rp else 'no plan yet'          # failures here; "Not done" without them
            if change:
                progress = f'CHANGED: {change}. Plan so far - {progress}'
                if rp and interrupt:
                    rp.p['_interrupted'] = True
            prompt = planner.update_prompt(state, progress, rp.leftover(self.roster, failed=False)[1] if rp else '')
            if phase == 'combat':
                prompt = prompt.replace('DECISION: same planning phase, update.',
                                        'DECISION: update during COMBAT (see the first STATE line).', 1)
                if str(change or '').startswith('COMBAT STARTED'):
                    # the combat-start call: the interest decision, as numbers, before the state
                    base = getattr(self, 'fight_base', None) or {}
                    at = (types.SimpleNamespace(gold=base['gold'], stage=scr.stage)       # gold at combat start: a
                          if base.get('stage') == scr.stage and base.get('gold') is not None else scr)   # fast win's +1
                    check = planner.interest_check(at, self.roster, self.augments_taken)
                    if check:          # it replaces the state's Economy line (the same numbers)
                        prompt = '\n'.join(l for l in prompt.split('\n') if not l.startswith('Economy: '))
                        prompt = prompt.replace('\n\nSTATE\n', f'\n\n{check}\n\nSTATE\n', 1)
                    # ... and the strategy rewritten every combat (the user, 2026-10-13: run 233635 kept its queue-time
                    # strategy all game, though from 3-7 it played otherwise)
                    # (not in the climb: its game plan changes only at checkpoints - the owner's structure)
                    if self.game_strategy.get('text') and os.environ.get('TFT_CLIMB') != '1':
                        prompt += ('\nSTRATEGY REWRITE (every combat): send {"game_strategy": {"text": ..., "why": ...}} '
                                   'in this reply - your whole strategy paragraph as it stands now, after what this game '
                                   'has shown; "why": what changed, or "unchanged".')
                        self.strategy_due = scr.stage
                    if os.environ.get('TFT_NEXT_ROUND'):
                        # (experiments/README.md section 9: plans arrive 20-28 s into a ~30 s planning phase with
                        # GPT-6.1-Sol; next-round plans went with its better places) the plan for the next planning
                        # phase is written now, during the fight, and runs the moment that phase opens
                        prompt += ('\nNEXT ROUND (required in this reply): after strategy, {"next_round": {...}} - the plan '
                                   'for the next planning phase (PLANNING AHEAD in your instructions). It runs the moment '
                                   'that phase opens; then you are asked again with what it did and the new shop.')
                    else:
                        prompt += stage_start_ask(scr.stage)
        else:
            prompt = planner.plan_prompt(state, first=not getattr(self, 'planned_once', False),
                                         carry=planner.carry_text(self.carry_note, self.brain.protected,
                                                                  moves_dropped=moves_dropped,
                                                                  continuing_moves=(self.round_plan.p.get('_requested_moves')
                                                                                    if self.round_plan and self.round_plan.carry else None)))
            self.planned_once = True
        news = self.take_news()
        if news:                             # what happened since the model last decided
            first, _, rest = prompt.partition('\n')
            prompt = f'{first}\nEVENTS SINCE YOUR LAST DECISION: {news}\n{rest}'
        stage, t0 = scr.stage, time.time()
        shop_at_plan = list(scr.shop)
        raw_at_plan = list(scr.shop_raw or [])          # the Wisps the state showed: a plan's Wisp is bought by name
        offers_at_plan = {i: (Player.wisp_offer_id(self, (scr.stage, i, scr.shop_raw[i])), cost)
                          for i, (_, cost) in specials.items()}
        known = [u.id for u in self.roster.units]
        names, gold0 = self.unit_names(), scr.gold       # what each id meant when the model saw the state
        lvl0 = (scr.level or self.last_level, scr.xp)    # a plan that levels up must field the new slot
        hp0 = scr.hp                                     # (the level-7 hold opens at low HP: comp_rules.low_hp_step)
        image = self.screen_jpeg()
        old_rp = self.round_plan if kind == 'update' else None
        snap = old_rp.snapshot() if old_rp is not None else None     # what the old plan had done when asked

        def credit(new):
            """The old plan kept running while the model wrote this update: its steps since the prompt are done."""
            if getattr(self, 'next_plan', None) and Phase.next_stage(self.next_plan[0]) == stage:
                self.next_plan, self._next_replaces = None, None
            if old_rp is not None and old_rp is not new:
                got = new.credit(old_rp, snap)
                if any(got.values()):
                    self.log(ev='update_credited', stage=stage, **got)
            fallback = getattr(self, '_pending_fallback', None)
            if fallback is not None and fallback.stage == stage and fallback is not old_rp:
                got = new.credit(fallback, fallback.pending_snapshot)
                if any(got.values()):
                    self.log(ev='update_credited', stage=stage, **got)
                self._pending_fallback = None

        early = {}

        def answered():
            if self.plan_seq == seq:
                self.plan_pending = False

        def install(obj):
            """The first answer (execution_plan) runs the moment it is written;
            the strategy is still being written."""
            plan = planner.normalize(obj) if isinstance(obj, dict) and 'execution_plan' in obj else None
            planner.stamp_wisps(plan, raw_at_plan)
            Player.stamp_wisp_offers(self, plan, offers_at_plan)
            if plan and stage in (self.phase.stage, getattr(self, 'stage_seen', None)):
                if getattr(self, '_executor_level_t', 0) > t0:
                    planner.aim_xp(plan, lvl0)   # the executor levelled while it was written (x32.3)
                errs = planner.validate(plan, self.roster, self.bench_items, names, level=lvl0,
                                       done=self.gives_since(old_rp, snap), queued=self.queued_gives(),
                                       gold=gold0, stage=stage if kind in ('plan', 'update') else None, hp=hp0)
                self.apply_plan_meta(plan, stage)
                self.inherit_carry(plan)
                self.round_plan = planner.RoundPlan(plan, stage, shop_at_plan, known=known, names=names, gold=gold0)
                self.round_plan.kind = kind
                self.round_plan.source_seq = seq
                credit(self.round_plan)
                self.plan_rejected(errs, stage)
                self.tell_board_after(plan, stage)
                self.brain.protected = self.round_plan.protected_ids()
                answered()
                self.wisps_answered(plan, stage, seq)
                early['t'], early['rp'] = time.time() - t0, self.round_plan
                print(f'[{kind} plan in {early["t"]:.1f}s: executing while the strategy is written]')
                # the panel shows the decision now, not when the whole reply is done (run23: it
                # looked frozen for the 10-20 s the reasoning took)
                self.live.model_call(kind, stage, prompt, json.dumps(obj), early['t'])

        def safe_install(obj):
            try:
                install(obj)
            except Exception as e:
                self.harness_error('plan install', e)

        def run():
            # whatever fails after the model answered, the call ends: plan_pending is cleared and the error logged (game
            # 78: a TypeError in the plan check ended this thread before answered() - plan_pending stayed set, no
            # model question went out at 4-5, 4-6, 5-2, 5-5 or 5-6 and no anvil was opened)
            try:
                handle()
            except Exception as e:
                self.harness_error('plan answer', e)
            finally:
                answered()

        def handle():
            try:
                text = ask_model(prompt, image, on_first=safe_install,
                                 still_wanted=lambda: stage in (self.phase.stage, getattr(self, 'stage_seen', None)))
            except Exception as e:
                text = f'FAILED: {e}'
            if kind == 'update' and str(change or '').startswith('COMBAT STARTED'):
                from tft_player.coach import release_stage_opening
                release_stage_opening(stage, answered=bool(text) and not text.startswith('FAILED:'))
            plan = planner.stamp_wisps(planner.parse(text), raw_at_plan)
            Player.stamp_wisp_offers(self, plan, offers_at_plan)
            full_errs = planner.validate(plan, self.roster, self.bench_items, names, level=lvl0,
                                         done=self.gives_since(old_rp, snap), queued=self.queued_gives(), gold=gold0,
                                         stage=stage if kind in ('plan', 'update') else None, hp=hp0) if plan else []
            # (the early install reported its own errors; the full answer only when it installs)
            from tft_player.coach import json_objects
            for o in json_objects(text):
                if isinstance(o, dict) and 'deliberation' in o:
                    d = o['deliberation']
                    self.last_ledger = (stage, json.dumps(d, ensure_ascii=False)[:1200] if not isinstance(d, str) else d[:1200])
            if plan and stage in (self.phase.stage, getattr(self, 'stage_seen', None)):
                if not early:            # no streamed first answer: install the whole reply now
                    if getattr(self, '_executor_level_t', 0) > t0:
                        planner.aim_xp(plan, lvl0)
                    self.inherit_carry(plan)
                    self.round_plan = planner.RoundPlan(plan, stage, shop_at_plan, known=known, names=names, gold=gold0)
                    self.round_plan.kind = kind
                    self.round_plan.source_seq = seq
                    credit(self.round_plan)
                    self.brain.protected = self.round_plan.protected_ids()
                    self.plan_rejected(full_errs, stage)
                    self.tell_board_after(plan, stage)
                else:
                    self.correct_early_plan(early.get('rp'), plan, stage)
                    self.tell_board_after(plan, stage)
                self.update_comps(plan.get('strategy'), stage)
                self.apply_plan_meta(plan, stage)
                self.keep_next_plan(plan, stage, done=self.gives_since(old_rp, snap), names=names, seq=seq)
                self.wisps_answered(plan, stage, seq)
            elif plan:
                self.late_carousel_work(plan, stage, names, seq)
                self.late_next_round(plan, stage, names=names, seq=seq)
            answered()
            self.model_call(kind, prompt, text, time.time() - t0, plan_s=early.get('t'))
            print(f'[{kind} {time.time() - t0:.0f}s] {(plan or {}).get("note") or (text or "")[:200]}')
            for w in (plan or {}).get('wisps') or []:
                if isinstance(w, dict):
                    print(f"  wisp slot {w.get('slot')}: {'BUY' if w.get('buy') else 'skip'} - {w.get('why')}")
        threading.Thread(target=run, daemon=True).start()

    def harness_error(self, where, e):
        """A fault of ours in a background step: printed (the game's output, the loop's monitor) and logged with its
        traceback; the game goes on."""
        import traceback
        tb = traceback.format_exc()
        print(f'HARNESS ERROR in {where}: {e!r}\n{tb}', flush=True)
        try:
            self.log(ev='harness_error', where=where, error=repr(e), trace=tb)
        except Exception:
            pass

    def note_goal(self, st, stage):
        """The result the player plays the fight for (the owner, 2026-10-29: a round's goal can be to lose - a loss
        streak into the monster round). It stands until changed; the review audits each round against it."""
        goal = str(st.get('fight_goal') or '').strip().lower()
        if goal not in ('win', 'lose'):
            return
        why = st.get('why') if isinstance(st.get('why'), dict) else {}
        if not self.fight_goal or self.fight_goal['goal'] != goal:
            self.log(ev='fight_goal', stage=stage, goal=goal, why=why.get('goal'))
            self.fight_goal = {'goal': goal, 'since': stage, 'why': why.get('goal')}
        elif why.get('goal'):
            self.fight_goal['why'] = why.get('goal')

    @staticmethod
    def named_units(st):
        """The champions a reply's "next" and notes name (game 57 4-3: the 4-2 reply said "At level 8, field Ashe (a
        5-cost) as the eighth unit", and the comp cleanup sold both Ashes as on no plan). -> {name}"""
        why = st.get('why') if isinstance(st.get('why'), dict) else {}
        text = f"{why.get('next') or ''} {st.get('notes') or ''}"
        return {n for n in CATALOG.units if re.search(rf"(?<![\w']){re.escape(n)}(?![\w'])", text)}

    def apply_plan_meta(self, plan, stage):
        """What the model's reply sets besides this round's actions: the components it wants
        from carousels and orbs (kept until it changes them), a pivot to another comp (with
        its reason), and a revised game plan (logged)."""
        from tft_player.items import COMPONENTS
        t = plan.get('item_targets')
        if isinstance(t, list):
            want = [i for i in t if i in COMPONENTS]
            if want != self.brain.plan.item_targets:
                self.brain.plan.item_targets = want
                self.log(ev='item_targets', stage=stage, targets=want, asked=t)
        self.apply_auto(plan, stage)
        st = plan.get('strategy')
        if not isinstance(st, dict):
            return
        self.note_goal(st, stage)
        self.set_wisp_flag(st.get('wisp_flag', plan.get('wisp_flag')), stage)
        self.named_next = Player.named_units(st)
        if st.get('game_plan'):
            self.log(ev='game_plan', stage=stage, text=st['game_plan'])
        pv = st.get('pivot')
        if not (self.forced and isinstance(pv, dict) and pv.get('to')):
            return
        if not self.committed:                                           # the chosen comp is played all game
            # (a flex game chose its comp itself: it may change it)
            if stage not in self.pivoted:
                self.pivoted.add(stage)
                self.news_event('pivot_refused', stage, f'comp changes are switched off in this harness mode: '
                                                               f'the target comp stays {self.forced.name}')
            return
        from tft_player.coach import in_flex_set
        from tft_player.meta import META
        comp = META.by_name(str(pv['to']))
        if not comp:
            self.news_event('pivot_refused', stage,
                            f"no MetaTFT comp named {pv.get('to')!r}: still playing {self.forced.name}")
        elif not in_flex_set(comp.name):              # the climb's flex set: a switch stays within it
            self.news_event('pivot_refused', stage,
                            f"{comp.name} is not in this game's comp set: still playing {self.forced.name}")
        elif comp.id != self.forced.id and stage not in self.pivoted:
            self.pivoted.add(stage)
            old, self.forced = self.forced, comp
            os.environ['TFT_FORCE_COMP'] = comp.name
            from tft_player.coach import auto_open_anvil
            self.auto_open = auto_open_anvil()
            self.target = comp.name
            from tft_player import coach as _coach
            self._commit_block = _coach.commit_strategies(comp, stage)   # the new comp's own strategy (x19.1)
            self.auto_buy = comp_auto_buy(comp.units, comp.name)
            self.set_comp(comp.id, 'pivot')
            self.log(ev='pivot', stage=stage, old=old.name, new=comp.name, why=pv.get('why'))
            print(f"{stage}: PIVOT {old.name} -> {comp.name}: {pv.get('why')}")
            self.news_event('pivot', stage, f"you changed comp from {old.name} to {comp.name} ({pv.get('why')}); "
                                            f"auto_buy: its 1-4 cost units to 2-star; its MetaTFT page comes with the next question")
            threading.Thread(target=self.load_commit_brief, args=(comp,), daemon=True).start()

    @staticmethod
    def planner_on():
        """The model plans rounds (and owns the target comp) when one is connected."""
        from tft_player.coach import _cli
        return bool(_cli()[0])

    def set_comp(self, cid, source):
        """The target comp is part of the game state; the model changes it."""
        from tft_player.meta import META
        comp = META.comps.get(str(cid)) if cid else None
        if comp and comp.id != self.brain.plan.comp:
            old = self.brain.plan.comp
            self.brain.plan.adopt(comp)
            print(f'target comp: {old} -> {comp.id} {comp.name} ({source})')
            self.log(ev='comp_change', old=old, new=comp.id, name=comp.name, source=source)

    def review_round(self, scr):
        """Combat started: show the model the board we fight with (runs during combat)."""
        from tft_player import planner
        from tft_player.coach import _cli, ask_model
        rp = self.round_plan
        if not rp or not _cli()[0]:
            return
        if self.forced:
            return       # fixed comp: no target to change, pages already in the briefing - nothing to review
        self.team_cap(scr)    # the team size the state shows: the level plus what the game's n/m added (Coronation)
        state = planner.state_text(scr, self.roster, self.brain.plan, self.bench_items, self.hp_log, {}, 0,
                                   goal=self.fight_goal)
        prompt = planner.review_prompt(state, rp.p, rp.done, rp.failed)
        image, t0 = self.screen_jpeg(), time.time()
        if self.target and self.target not in self.pages_sent and not self.pending_pages                 and time.time() - getattr(self, 'pages_tried', 0) > 60:
            self.pages_tried = time.time()          # an earlier capture failed: try again
            threading.Thread(target=self.fetch_pages, args=(self.target,), daemon=True).start()
        if self.pending_pages and isinstance(self.pending_pages[1], str):
            name, page = self.pending_pages          # no page on the site: our text page
            self.pending_pages = None
            self.pages_sent.add(name)
            prompt = f'MetaTFT data of the target comp, {name}:\n{page}\n\n' + prompt

        asked_at = scr.stage

        def run():
            try:
                text = ask_model(prompt, image, still_wanted=lambda: asked_at in (self.phase.stage,
                                                                                  getattr(self, 'stage_seen', None)))
            except Exception as e:
                text = f'FAILED: {e}'
            self.model_call('review', prompt, text, time.time() - t0)
            r = planner.parse(text) or {}
            self.update_comps(r.get('strategy'), self.phase.stage)
            self.apply_plan_meta(r, self.phase.stage)
            print(f"[review {time.time() - t0:.0f}s] {(r.get('strategy') or {}).get('now') or ''} problems: {r.get('problems') or 'none'}")
        threading.Thread(target=run, daemon=True).start()

    def handle_end(self, frame, scr):
        """'done' once the client says the game is over. When we are out, leave:
        click the elimination screen's exit button, or Settings -> EXIT TFT.
        Leaving is only ever done on evidence we are eliminated - exiting a game
        we are still in would forfeit it."""
        from tft_player.ending import EXIT_TFT_BUTTON
        E = self.ending
        if E.game_over:
            # placement from the game itself: the elimination screen's "Nth place" when we were
            # knocked out; a game that ended while we were still in it is a win (1st) - the client
            # closes the game window at once then, so there is no screen to read (run27: 1st place
            # recorded as null, since match history never answered)
            seen = ' '.join([E.eliminated_seen or ''] + list(getattr(E, 'words', []) or []))
            m = re.search(r'([1-8])\s*(ST|ND|RD|TH)\s*PLACE', seen.upper())     # "4THPLACE" (run29)
            if not m and not E.eliminated_seen and not getattr(self, 'exit_clicked', False) and not REPLAY:
                # knocked out in the last fight, the match ends at the same moment: the end screen is up but its
                # background read (3.5-4 s) has not come in yet (benchmark game 20261018-153420: logged 1st, then
                # "2NDPLACE" one event later) - it is read now; a win closes the window and shows nothing
                try:
                    E._scan()
                except Exception as e:
                    self.log(ev='end_scan_error', error=repr(e))
                seen = ' '.join([E.eliminated_seen or ''] + list(getattr(E, 'words', []) or []))
                m = re.search(r'([1-8])\s*(ST|ND|RD|TH)\s*PLACE', seen.upper())
            if m:
                self.placement_screen = (int(m.group(1)), f'elimination screen: "{m.group(0)}"')
            elif not getattr(self, 'exit_clicked', False) and not E.eliminated_seen:
                self.placement_screen = (1, 'the game ended while we were still in it')
            elif getattr(self, 'placement_screen', None) and self.placement_screen[0]:
                pass                                 # the elimination screen's place, read when it showed
            else:
                self.placement_screen = (None, f'eliminated, place not read ({E.eliminated_seen!r})')
            print(f'game over (client phase {E.phase}); placement from the screen: {self.placement_screen}')
            self.log(ev='game_over', phase=E.phase, stage=self.phase.stage, hp=getattr(self, '_hp', None),
                     placement_screen=self.placement_screen)
            return 'done'
        st = self.phase.stage and Phase._key(self.phase.stage)
        carousel = is_carousel(scr)    # its scan costs the chase CPU
        E.want_scan = bool(st and st >= (3, 1) and not scr.shop_visible and not scr.augments and not carousel)
        if time.time() - self.exit_tried < 5:
            return None
        if E.exit_button:
            x, y = E.exit_button
            ref = (960 + (x - frame.shape[1] / 2) * 1080 / frame.shape[0], y * 1080 / frame.shape[0])
            print(f'{self.phase.stage}: eliminated - clicking exit at {ref}')
            self.snap(frame, f'{self.phase.stage}-eliminated', full=True)
            self.log(ev='exit_click', how='button', xy=ref, text=E.eliminated_seen)   # not at=: log() reads at as the time
            self.exit_clicked = True
            self.inp.click(ref)
            self.exit_tried, E.exit_button = time.time(), None
            return 'exit'
        hp = getattr(self, '_hp', None)
        if E.eliminated_seen and hp is not None and hp <= 45:
            if not self.reader.settings_open(frame):
                self.log(ev='exit_click', how='menu', text=E.eliminated_seen, hp=hp)
                self.exit_clicked = True
                self.inp.key('escape')
            else:
                print(f'{self.phase.stage}: eliminated - Settings -> EXIT TFT')
                self.inp.click(EXIT_TFT_BUTTON)
            self.exit_tried = time.time() - 3.5
            return 'exit'
        return None

    def collect_augment_loot(self, frame, scr):
        """A picker just closed: scan the visible board and start walking before roster/plan work (e123).

        The PvP watcher's last scan can still be the covered augment frame. Game 52's 3-2 parked on that empty
        scan, then spent 8+12 seconds on other work with a gold orb visible. A fresh scan costs milliseconds;
        walking continues while the normal tick executes. Keep scanning briefly for the falling reward.
        """
        clicked = getattr(self, 'augment_clicked', 0)
        if (not clicked or not 0 <= time.time() - clicked < 20 or not scr.shop_visible or scr.augments
                or not getattr(self.phase, 'planning', False)):
            return False
        self.watcher.orbs = find_orbs(frame)
        self.watcher.orbs_at = time.time()
        tac, _ = find_tactician(frame)
        collected = self.collect_loot(frame, scr, tac)
        self.watcher.want_orbs = True             # keep watching for the falling reward while the next tick works
        return collected

    def collect_loot(self, frame, scr, tac):
        """Right-click toward orbs; the tactician walks while we keep playing."""
        if scr.stage is None and not getattr(self, '_stage_seen', False):
            # no stage read yet this game: the HUD is not where we look (run 20261010-002419: a stale window crop,
            # 25 minutes of clicks at 'orbs' and nothing else) - act on nothing until the screen reads
            return False
        self._stage_seen = True
        s, r = scr.round or (0, 0)
        pve = s == 1 or r == 7 or (s == 2 and r <= 1)
        # orbs are rare in PvP rounds: the watcher scans every frame in PvE, every ~2s otherwise - and every frame
        # while an orb is in sight. A scan is acted on while it is younger than a tick: with 0.5 s and ~1 s ticks,
        # PvP scans were nearly always stale when looked at (run 20261015-014311: the 5-7 orbs lay there all of
        # 6-1 and 6-2, no click)
        age = time.time() - self.watcher.orbs_at
        self.watcher.want_orbs = pve or bool(self.watcher.orbs) or age > 2.0
        if age > 2.5:
            return self.loot.target is not None
        orbs = self.watcher.orbs
        click = self.loot.next_click(orbs, tac, time.time())
        if self.loot.target is not None:
            self._orbs_seen_at = time.time()        # a live orb to walk to: no parking for a while
        if click:
            click = self.ground_point(click, tac)
        self.status_extra['loot'] = (f'{len(orbs)} orb(s) visible, walking to {tuple(round(v) for v in self.loot.target)}'
                                     if orbs and self.loot.target else
                                     f'{len(orbs)} orb(s) visible' if orbs else 'no orbs')
        if click:
            try:
                self.inp.click(click, 'right')
            except NotForeground:
                return False
            self._loot_at = time.time()
            self.log(ev='loot', stage=scr.stage, click=[round(click[0]), round(click[1])],
                     orbs=len(orbs), tactician=tac and [round(tac[0]), round(tac[1])],
                     speed=round(self.loot.speed))
        return bool(orbs)

    def item_sig(self, frame, i):
        """A slot's icon pixels, coarse: the same item keeps the same signature."""
        import cv2
        from tft_player.items import SLOT_HALF, slot_point
        s = frame.shape[0] / 1080
        x, y, _ = slot_point(i)
        c = frame[round((y - SLOT_HALF) * s):round((y + SLOT_HALF) * s), round((x - SLOT_HALF) * s):round((x + SLOT_HALF) * s)]
        if not c.size:
            return None
        small = cv2.resize(c, (8, 8), interpolation=cv2.INTER_AREA) // 32
        return small.tobytes()

    def identify_item(self):
        """Idle time: hover one item-bench slot whose item has no name read yet, and read
        the name from its tooltip (text, not the icon). Hover only - nothing is clicked."""
        from tft_player.items import slot_point
        sigs = getattr(self, 'item_sigs', None) or []
        # two tries per item and round (run 221257: two failed reads at 4-6 left a Radiant item
        # "unknown" for the rest of the game - it was never hovered again)
        stage = self.phase.stage
        track = self.item_track
        disputed = lambda i: bool(track.disputed[i] and not track.text[i])
        todo = [i for i, sig in enumerate(sigs) if sig and (sig not in self.item_names or disputed(i))
                and self.item_tries.get((sig, stage), 0) < 2]
        todo.sort(key=lambda i: not disputed(i))
        if not todo:
            return False
        if getattr(self, '_item_ocr_busy', False):
            return False                          # one tooltip read at a time
        i = todo[0]
        sig = sigs[i]
        self.item_tries[(sig, stage)] = self.item_tries.get((sig, stage), 0) + 1
        x, y, _ = slot_point(i)
        try:
            # approach from the screen's left edge: tooltips need motion, and the old start point (x 200) lay on the
            # trait panel - its trait tooltip was then read as the item's ("2 > 4 > 6 / Spellweaver", 15 times in the
            # games of 2026-10-10)
            self.inp.hover((4, y, 'L'))
            time.sleep(0.05)
            pre = self.cap.read()
            self.inp.hover((x, y, 'L'))
            time.sleep(0.4)                       # the tooltip fades in
            got = self.cap.read()
            self.inp.hover((960, 540))            # off the item bench again
        except NotForeground:
            return False
        if not got:
            return True
        frame = got[0]
        before = pre[0] if pre else None

        crop = (getattr(self, 'item_crops', None) or [None] * N_SLOTS)[i]

        def read():
            # off the loop: the title line is read in tens of ms, but when that fails the
            # detection over the whole tooltip takes 7-9 s
            from tft_player.items import learn_icon
            try:
                self.reader.tooltip_text = ''
                name = self.reader.item_tooltip(frame, (x, y), before=before)
                text = getattr(self.reader, 'tooltip_text', '')
                if name:
                    self.item_names[sig] = name
                    Player.record_item_name(self, i, sig, name, stage)
                    if crop is not None:
                        learn_icon(crop, re.sub(r'\s*\(.*$', '', name))   # known from the picture from now on
                elif text and trait_tooltip(text):
                    # a trait's tooltip, not the item's (the hover passed the trait panel): read again later. g15 2-1
                    # "Blackthorn", 4-2 "Coven" on the Item Remover's slot - the model named it "Coven Emblem", the
                    # state showed that emblem at 4-5 and 5-1 and two gives of it were refused
                    self.log(ev='item_read_trait', slot=i, text=text[:40])
                elif text:
                    # Preserve the unmatched tooltip text as evidence; identity remains unread.
                    self.news_event('item_unrecognised', stage,
                                    f'item bench slot {i}: its tooltip reads "{text[:80]}", which matches no '
                                    f'item name; the slot remains unknown')
                self.log(ev='item_read', slot=i, name=name, text=None if name else text)
                self.status_extra['identify'] = f'item slot {i}: {name or "tooltip not read"}'
            finally:
                self._item_ocr_busy = False
        self._item_ocr_busy = True
        threading.Thread(target=read, daemon=True).start()
        return True

    # events that change what the model sees: each wakes it with an update question when no question is
    # running and time allows (the user, 2026-10-09: no fixed number of calls a round; more when events need it)
    WAKE = {'game_change', 'unit_taken', 'unit_corrected', 'star_corrected', 'game_moved_unit', 'duplicated',
            'rolldown', 'after_fight', 'unit_missing', 'unit_found', 'board_after_lost',
            'already done'}            # request e39: a move ran while the reply was written - asked again to undo it

    # step_failed / plan_idle: in the plan report ("failed", "Not done"); roster_mismatch: tracking, fixed by the
    # controller - the state shows the result
    LOG_ONLY = {'step_failed', 'plan_idle', 'roster_mismatch'}

    def wake_waits(self, rp, scr):
        """A plan's own 'board after' alarm, late in the planning phase while that plan still has steps to take, waits
        for the fight's call: the reply cannot come back before the fight, building the question holds the executor,
        and the reply replaces the steps still running (request e41, game 44 3-3: the Riftbeast 7 plan came with
        10 s left, its alarm asked at once - 3 s to build the state, the reply with 2 s left was an empty plan written
        before the 8 XP buys ran; one of six moves was made)."""
        late = getattr(rp, 'LATE_S', 10)
        return (scr.timer is not None and scr.timer <= late and bool(self.pending_events)
                and all(e.startswith('board_after_lost:') for e in self.pending_events) and rp.pending(self.roster))

    def news_event(self, kind, stage, what):
        """Record something that happened; the next model call gets it. A WAKE kind also asks for that call."""
        self.log(ev=kind, stage=stage, what=what)
        if kind in self.LOG_ONLY:
            return
        line = f'{stage} {kind}: {what}'
        if line not in self.news:
            self.news.append(line)
            if kind in self.WAKE:
                self.pending_events.append(f'{kind}: {what}')

    def take_news(self):
        news, self.news = self.news[-12:], []
        return '; '.join(news)

    def review_due(self, rp, scr, plan_done):
        """The model's own next_review conditions for this round: when one holds, it is asked
        again (once per plan). Nothing else makes a mid-round model call."""
        conds = rp.p.get('next_review') if isinstance(rp.p, dict) else None
        if not conds or getattr(rp, 'review_fired', False):
            return False
        if not isinstance(conds, list):
            conds = [conds]
        met = None
        for c in conds:
            if not isinstance(c, dict):
                continue
            if c.get('plan_complete') and plan_done:
                met = 'the plan is complete'
            elif isinstance(c.get('seconds_left'), (int, float)) and scr.timer is not None \
                    and scr.timer <= c['seconds_left']:
                met = f"{scr.timer}s left"
            elif isinstance(c.get('gold_at_least'), (int, float)) and scr.gold is not None \
                    and scr.gold >= c['gold_at_least']:
                met = f'gold is {scr.gold}'
            elif isinstance(c.get('unit'), str) and isinstance(c.get('star'), int) and any(
                    u.name == c['unit'] and u.star >= c['star'] for u in self.roster.units):
                met = f"{c['unit']} is {c['star']}-star"
            if met:
                break
        if not met or (scr.timer is not None and scr.timer <= 2):
            return False
        rp.review_fired = True
        self.log(ev='review_requested', stage=scr.stage, condition=met)
        print(f'{scr.stage}: next_review met ({met}) -> asking the model')
        self.request_plan(scr, self.last_frame, 'update', f'your next_review condition is met: {met}', interrupt=False)
        return True

    def record_item_name(self, i, sig, name, stage):
        """Publish a tooltip read to execution and validation too, including in combat. An
        asynchronous read only names the slot if the same icon is still there in this round."""
        if getattr(getattr(self, 'phase', None), 'stage', None) != stage:
            return False
        got = self.cap.read()
        if got is None or self.item_sig(got[0], i) != sig:
            return False
        self.item_track.confirm(i, name)
        self.bench_items = self.item_track.names()
        return True

    def bench_names(self, frame, detail=False):
        """The item bench now: names (a tooltip-read name for this very icon beats the picture)."""
        d = read_item_bench(frame, detail=True)
        names = []
        said = getattr(self, 'item_names_model', {})
        for i, (n, _, _) in enumerate(d):
            sig = self.item_sig(frame, i) if n else None
            names.append(self.item_names.get(sig, said.get(sig, Player.picture_name(self, sig, n))) if sig else n)
        track = self.__dict__.get('_item_track')
        tracked = track.names() if track is not None else None
        if tracked and [n is not None for n in tracked] == [n is not None for n in names]:
            names = tracked                      # the same slots filled: the tracked names hold
        return (names, d) if detail else names

    @property
    def item_track(self):
        """The item bench slot by slot, kept from our own actions (items.ItemTracker)."""
        return self.__dict__.setdefault('_item_track', ItemTracker())

    def bench_shifted(self, i):
        """A give took slot i's item off: the items below moved up, so the icon signatures and crops of slots i on,
        read before the give, are another item's until the bench is read again - a tooltip read there must not be
        filed under them (run 20261103-233832 5-2: the Gloves went from slot 3 to Elder Dragon, the Radiant Giant
        Slayer moved up to slot 3 and its tooltip read was filed under the Gloves' slot-3 signature; at 5-7 the
        Gloves, back from the sold Scuttlecrab on slot 3, were named Radiant Giant Slayer as text, then 'unknown')."""
        for attr in ('item_sigs', 'item_crops'):
            cur = getattr(self, attr, None)
            if cur:
                setattr(self, attr, list(cur[:i]) + [None] * max(0, len(cur) - i))

    def confirm_slot(self, i):
        """The name on item-bench slot i, read as text from its tooltip's title line (the user, 2026-10-15: confirm
        the component's text before making the item - fast and accurate). ~0.3 s: a glide in from the screen's edge
        (tooltips need motion), the fade-in, one line of OCR. None when no title was read in time."""
        x, y, _ = slot_point(i)
        self.inp.hover((4, y, 'L'), steps=2, duration=0.03)
        pre = self.cap.read()                    # the screen with no tooltip: what the tooltip changes is read
        self.inp.hover((x, y, 'L'), steps=4, duration=0.06)
        end, name = time.time() + 0.8, None
        time.sleep(0.15)
        while time.time() < end and name is None:
            got = self.cap.read()
            if got is not None:
                name = self.reader.item_tooltip(got[0], (x, y), before=pre[0] if pre else None, title_only=True)
            if name is None:
                time.sleep(0.06)
        if name:
            self.item_track.confirm(i, name)
            self.bench_items = self.item_track.names()
            sig = (getattr(self, 'item_sigs', None) or [None] * N_SLOTS)[i]
            if sig:
                self.item_names[sig] = name
        self.log(ev='item_confirm', slot=i, name=name)
        return name

    def find_in_fight(self, unit, frame):
        """Where `unit` stands now in a fight (units move): right-click our units (green bars, nearest to where it
        was placed first) and read each one's panel until it is `unit` (name and star). -> the point to drop on (its
        body, under its bar, found again after the read), or None."""
        deadline = time.monotonic() + 3.0
        bars = self.fight_bars(frame)
        shown = self.reader.read_panel(frame) if bars and getattr(self, 'reader', None) else (None, None)
        if shown[0] == unit.name and (not shown[1] or shown[1] == unit.star):
            # the panel already shows the unit (left open by the last give's check, or by a give queued as crowded):
            # a click on it changes nothing, read_fresh takes that for a stale panel and needs a click on another
            # unit to land first - in a fight those miss (run 20261027-110551 3-3 Cassiopeia, 4-7 Elder Dragon: the
            # panel was open on the target before the search, "not found among our units in the fight"). Closed
            # first, by a right-click on the ground
            self.inp.click(PARK, 'right')
            for _ in range(8):
                time.sleep(0.06)
                got = self.cap.read()
                if got is None or self.reader.read_panel(got[0])[0] != unit.name:
                    break
            self.panel_before = (None, None)
            self.log(ev='fight_panel_closed', unit=repr(unit))
            got = self.cap.read()
            if got is not None:
                frame = got[0]
                bars = self.fight_bars(frame) or bars
        home = loc_point(unit.loc)
        # first the bars whose item row shows the items the unit holds, then the nearest to where it was placed: a
        # tank runs far from its hex (benchmark game 20261021-153057 4-3..5-1: Alistar** was not among the 6 bars
        # nearest (3, 3) - four gives "not found in the fight")
        from tft_player.items import read_unit_items
        held = {i for i in unit.items or [] if i and i != 'unknown'}
        shows = {b: len(held & {n for n, _, _ in read_unit_items(frame, b) or []}) if held else 0 for b in bars}
        bars.sort(key=lambda b: (-shows[b], (b[0] - home[0]) ** 2 + (b[1] + 45 - home[1]) ** 2))
        pts = [(b[0], b[1] + 45) for b in bars]
        reads = []                    # each panel read, logged: the next "not found" can be traced read by read

        def current_point(pt, dy):
            # Each panel read can take a second. Game 64 4-2 kept probing the bars
            # from the start of the search while Brambleback moved up the arena.
            got = self.cap.read()
            if got is None:
                return pt
            now = self.fight_bars(got[0])
            near = min(now, key=lambda b: (b[0] - pt[0]) ** 2 + (b[1] + dy - pt[1]) ** 2, default=None)
            if near is not None and abs(near[0] - pt[0]) < 120 and abs(near[1] + dy - pt[1]) < 120:
                return (near[0], near[1] + dy)
            return None

        def probe(pt, dy):
            pt = current_point(pt, dy)
            if pt is None or time.monotonic() >= deadline:
                return None, None
            name, stars, _ = self.read_fresh(pt, [])
            reads.append([round(pt[0]), round(pt[1]), name, stars])
            if name == unit.name and (not stars or stars == unit.star):
                self.log(ev='fight_search', unit=repr(unit), bars=len(bars), reads=reads)
                return name, current_point(pt, dy)
            return name, None

        retried, named = set(), set()
        for k, pt in enumerate(pts[:6]):
            if time.monotonic() >= deadline:
                break
            # A missed panel is not helped by probing two other moving units and this
            # point again. Game 49: that multiplied one lookup into a 31-second tick.
            name, target = probe(pt, 45)
            if target is not None:
                return target
            if name:
                named.add(k)
            if not name and len(retried) < 4:
                # Try the missed unit's body before spending the budget on other
                # units (game 64 4-2 exhausted it after six bar+45 reads).
                retried.add(k)
                _, target = probe((pt[0], pt[1] + FIGHT_BODY - 45), FIGHT_BODY)
                if target is not None:
                    return target
        # a bar whose read opened no panel, or one never read (6 of 8 are read first): read again at its body,
        # FIGHT_BODY under the bar - a big model's body (Brambleback) stands well below its bar, and an enemy over
        # the bar takes the click (run 20261030-101714 4-5: Alpha Mark for the carry Brambleback, 8 bars, two reads
        # null and two bars unread: "not found", the carry fought the round without it)
        again = [(b[0], b[1] + FIGHT_BODY) for k, (b, pt) in enumerate(zip(bars, pts))
                 if k not in retried and k not in named][:4 - len(retried)]
        for k, pt in enumerate(again):
            if time.monotonic() >= deadline:
                break
            _, target = probe(pt, FIGHT_BODY)
            if target is not None:
                return target
        self.log(ev='fight_search', unit=repr(unit), bars=len(bars), reads=reads)
        # no panel named it, but one bar's item row shows every item it holds (2 or more), and no other bar does: that
        # bar is the unit. Its own read named no champion of ours - an enemy stood over it in the crowd (run
        # 20261027-125046 5-1: Morgana's row read Crownguard + Morellonomicon, her point read Brambleback, the Rod
        # sat on the item bench through the fight; game 29 5-1 the same give, a null read at her point)
        if len(held) >= 2:
            full = [b for b in bars if shows[b] == len(held)]
            ours = {u.name for u in self.roster.units if u.name} if getattr(self, 'roster', None) else set()
            if len(full) == 1:
                pt = (full[0][0], full[0][1] + 45)
                said = next((r[2] for r in reads if (r[0], r[1]) == (round(pt[0]), round(pt[1]))), None)
                if said is None or said not in ours:
                    self.log(ev='fight_found_by_items', unit=repr(unit), point=[round(v) for v in pt], read=said)
                    return current_point(pt, 45)
        # The frame at the failed lookup, not just the frame before a long search:
        # retain the target, bars and roster alongside it for the next offline diagnosis.
        if getattr(self, 'log_dir', None) is not None:
            n = getattr(self, '_fight_search_shot', 0) + 1
            self._fight_search_shot = n
            shot = f'fight-search-{n:02d}'
            got = self.cap.read()
            self.snap(got[0] if got is not None else frame, shot, full=True)
            self.log(ev='fight_search_failed', unit=repr(unit), stage=getattr(self.phase, 'stage', None),
                     bars=[[round(v) for v in b] for b in bars], reads=reads,
                     budget_exhausted=time.monotonic() >= deadline, roster=repr(self.roster), shot=shot)
        return None

    def fight_bars(self, frame):
        """Our units' health bars in a fight, the tactician's own bar left out."""
        tac_box = self.watcher.tac[1] if getattr(self, 'watcher', None) else None
        inside = lambda b: tac_box is not None and tac_box[0] - 20 <= b[0] <= tac_box[2] + 20 and \
            tac_box[1] - 30 <= b[1] <= tac_box[3] + 10
        return [b for b in B.green_bars(frame, y_range=FIGHT_BARS_Y) if not inside(b)]

    def fight_row(self, frame, point, expect=None):
        """The item row of the unit standing at `point` in a fight (the bar nearest it), or None."""
        from tft_player.items import read_unit_items
        bars = B.green_bars(frame, y_range=FIGHT_BARS_Y)
        bar = min(bars, key=lambda b: (b[0] - point[0]) ** 2 + (b[1] + 45 - point[1]) ** 2, default=None)
        if bar is None or abs(bar[0] - point[0]) > 80 or abs(bar[1] + 45 - point[1]) > 80:
            return None
        return read_unit_items(frame, bar, expect)

    def unit_row(self, frame, unit, expect=None):
        """(bar, [(name, score, crop)]) for one tracked unit, or (None, None) when its bar is not
        found where it is tracked (or two units could own it)."""
        from tft_player.items import read_unit_items
        from tft_player.state import _hex_near
        _, _, places = B.bar_places(frame, exclude=self.watcher.tac[1])
        bar = next((b for b, loc in places if loc == unit.loc), None)
        if bar is None and unit.on_board:
            near = [b for b, loc in places if loc[0] == 'board' and _hex_near(loc[1:], unit.loc[1:])]
            bar = near[0] if len(near) == 1 else None
        return (bar, read_unit_items(frame, bar, expect)) if bar is not None else (None, None)

    def give_item(self, a, scr):
        """Drag an item onto a unit, by the item's IDENTITY (its slot is looked up on a fresh read
        just before the drag: run27 2-1 took the Gloves from a stored slot number and built
        Infinity Edge where the state logged Edge of Night). Success only when the screen shows it:
        the item left the bench and the unit's icon row shows it (or the item it completed); a
        consumable (Duplicator) must be gone, and a Duplicator's copy is added to the roster.
        The unit's items are what its row shows afterwards - never what was meant."""
        from tft_player.items import BY_PAIR, COMPONENTS, is_consumable, is_duplicator, is_reusable, learn_icon,             radiant_twin
        slot, item, unit = a.arg
        from tft_player.state import item_give_error
        blocked = item_give_error(unit, item)
        if blocked:
            self.fail_cause = blocked
            return False
        base = lambda n: re.sub(r'\s*\(.*$', '', n or '')
        if getattr(unit, 'row_ghosts', None):
            unit.row_ghosts = set()          # a give: what its row shows may be real now (read_roster_items)
        if is_reusable(item):
            alpha_started, alpha_plan = time.time(), getattr(self, 'round_plan', None)
            if (getattr(a, 'fight', False)
                    or not getattr(getattr(self, 'phase', None), 'planning', False)):
                self.deferred = True
                self.fail_cause = 'Alpha Mark held until planning'
                return False
            got = self.cap.read()
            if got is None:
                self.fail_cause = 'no frame to confirm the Alpha target'
                return False
            bars = B.green_bars(got[0])
            others = [self.unit_point(o.loc, bars) for o in self.roster.units
                      if o.name and o.name != unit.name and not o.missing and o.loc[0] in ('board', 'bench')][:2]
            name, stars, _ = self.read_fresh(self.unit_point(unit.loc, bars), others)
            self.log(ev='alpha_target_checked', stage=scr.stage, unit=repr(unit), read=name, stars=stars)
            if name != unit.name or stars and stars != unit.star:
                self.fail_cause = f'Alpha target panel reads {name or "no name"} {stars or "?"}-star'
                if name:
                    before_identity = (unit.name, unit.star)
                    unit.name, unit.star = name, stars or unit.star
                    Player.identity_corrected(self, unit, before_identity, scr)
                else:
                    self.reread = getattr(self, 'reread', set()) | {unit.id}
                    self.deferred = True
                return False
            unit.unverified, unit.sure = None, True
        if getattr(a, 'fight', False) and getattr(unit, 'riding', None):
            # a fight give finds its unit by its panel; the rider opens none (benchmark game 20261021-153057 3-5..3-7:
            # four gives to the rider Teemo "not found among our units in the fight")
            self.fail_cause = f'{unit} rides the BFF and opens no panel: it is not found in a fight - given when planning'
            return False
        if is_consumable(item) or 'Chest' in item:
            self.last_used = (base(item), unit, time.time())     # a choice screen may follow: its source
        elif (not is_reusable(item) and not getattr(a, 'fight', False)
                and getattr(getattr(self, 'phase', None), 'planning', False)):
            fast = Player.give_fast(self, a, scr, slot, item, unit)
            if fast is not None:
                return fast
        got = self.cap.read()
        if got is None:
            self.fail_cause = 'no frame to read the item bench from'
            return False
        names, detail = self.bench_names(got[0], detail=True)
        here = [k for k, n in enumerate(names) if n and base(n) == base(item)] or             [k for k, n in enumerate(names) if radiant_twin(base(n), base(item))]
        src = slot if slot in here else (here[0] if here else None)
        track = self.__dict__.get('_item_track') if isinstance(self, Player) else None
        if track is not None and not REPLAY:
            # confirmed by its tooltip text before the drag (a slot already read as text in this round, since it last
            # changed, is not hovered again: the game can change bench items at a round's start); a slot that turns out to hold another item is renamed and the next one tried
            cands = ([src] if src is not None else []) + [k for k in here if k != src]                 + [k for k, n in enumerate(names) if n == 'unknown']
            src, wrong = None, []
            neighbours = set()
            for probe in range(4):
                if probe >= len(cands):
                    break
                k = cands[probe]
                if k not in neighbours and track.text[k] and track.fresh[k] and base(track.slots[k]) == base(item):
                    src = k
                    break
                try:
                    said = self.confirm_slot(k)
                except NotForeground:
                    raise
                if base(said) == base(item) or (said is None and k not in neighbours):
                    src = k                      # matches, or no title read in time: act on the picture's name
                    break
                if radiant_twin(base(said), base(item)) and not any(
                        base(n) == base(item) for j, n in enumerate(names) if j != k):
                    src, item = k, base(said)    # the tooltip's name (Radiant or not) is the item's
                    self.log(ev='radiant_named', stage=scr.stage, slot=k, said=said, asked=a.arg[1])
                    break
                wrong.append(f'slot {k} holds {said or "an unread item"}')
                if said is not None:
                    # Game 53 3-5: the supposed Bloodthirster slot held Sword; Bloodthirster was one
                    # slot above it. A contradicted binding can mean shifted slots, including known names.
                    # Keep the four-probe budget; neighbours require their own matching tooltip.
                    for adjacent in (k - 1, k + 1):
                        if 0 <= adjacent < len(names) and names[adjacent] and adjacent not in cands:
                            cands.append(adjacent)
                            neighbours.add(adjacent)
            if src is None and wrong:
                self.fail_cause = f'{item} not found: its tooltip text says {"; ".join(wrong)}'
                return False
            if wrong:
                self.log(ev='item_misread', stage=scr.stage, item=item, wrong=wrong, used=src)
            got = self.cap.read() or got          # the bench as it is now, for the checks below
            names, detail = self.bench_names(got[0], detail=True)
        if src is None:
            if 'unknown' in names:
                self.identify_items_now = True       # read the unidentified slots now (any phase)
                self.fail_cause = (f'{item} is not among the identified items on the item bench; '
                                   f'unidentified slots {[k for k, n in enumerate(names) if n == "unknown"]} are being read')
            else:
                self.fail_cause = f'{item} is not on the item bench (it holds: {", ".join(n for n in names if n) or "nothing"})'
            return False
        fight = getattr(a, 'fight', False)
        target = None
        if fight:
            target = self.find_in_fight(unit, got[0])
            if target is None:
                self.fail_cause = f'{unit} was not found among our units in the fight (read off their panels)'
                # not searched for again in this fight, whatever plan comes next (fight_give)
                stage = getattr(scr, 'stage', None)
                unfound = getattr(self, 'fight_unfound', None)
                if not unfound or unfound[0] != stage:
                    unfound = self.fight_unfound = (stage, set())
                unfound[1].add(unit.id)
                return False
            now = self.cap.read()
            near = fight_crowded(self.fight_bars(now[0] if now is not None else got[0]), target)
            if near:
                # a drop among our bunched units lands on whichever is in front: not now - tried again in a moment
                # (units spread or fall), else given in the next planning phase (fight_crowded)
                self.fail_cause = (f'queued: {unit} stands among {len(near)} more of our units in the fight - a drop '
                                   f'there could land on one of them')
                self.deferred = True
                self.fight_give_wait = time.time() + 2.5
                self.log(ev='fight_give_crowded', stage=getattr(scr, 'stage', None), unit=repr(unit), item=item,
                         point=[round(v) for v in target], near=[[round(v) for v in b] for b in near])
                return False
            before = list(unit.items or [])
        else:
            bar, before = self.unit_row(got[0], unit)
            before = [n for n, _, _ in before] if before is not None else None
        # a component dropped on a unit holding a matching component completes an item. The row read before the
        # drop can miss an icon: the tracked items count too (run 20261015-003101 2-1: Negatron Cloak on Rek'Sai's
        # Chain Vest made Gargoyle Stoneplate, judged 'failed' and queued again)
        held = list(before or []) + [c for c in unit.items or [] if c not in (before or [])]
        done = next((BY_PAIR[tuple(sorted((c, item)))] for c in held
                     if item in COMPONENTS and c in COMPONENTS and tuple(sorted((c, item))) in BY_PAIR), None)
        expect = done or item
        # any item this component completes also shows it landed
        makes = {v for k, v in BY_PAIR.items() if item in k} if item in COMPONENTS else set()
        raw_name = lambda f, k, n: base(self.item_names.get(self.item_sig(f, k), n)) if n else None
        raw0 = [d[0] for d in detail] if detail else list(names)
        n_here = sum(1 for k, n in enumerate(raw0) if raw_name(got[0], k, n) == base(item))
        src_was = raw_name(got[0], src, raw0[src])
        inp = self.inp
        how, stage = 'fight', getattr(scr, 'stage', None)
        if target is None:
            target, how = Player.drop_point(self, unit, bar, stage)
            if how is None:
                return False
        self._last_drop = (unit.id, how, stage)
        if is_reusable(item) and (getattr(self, 'round_plan', None) is not alpha_plan
                or scr.timer is not None and scr.timer - (time.time() - alpha_started) <= 2):
            self.deferred = True
            self.fail_cause = 'Alpha target or planning window changed during its source check'
            return False
        inp.drag(slot_point(src), target or loc_point(unit.loc))
        if is_reusable(item):
            # it stays on the item bench (the Alpha Mark - the user, 2026-10-13): the drop is the use, and only the
            # game's own refusal fails it
            said = None
            for _ in range(8):
                time.sleep(0.12)
                got2 = self.cap.read()
                said = self.game_message(got2[0]) if got2 is not None and not REPLAY else None
                said = None if about_other(said, item) else said
                if said:
                    break
            if getattr(self, 'log_dir', None) is not None:
                n = self.__dict__.get('_alpha_shots', 0) + 1
                self._alpha_shots = n
                shot = f'{stage}-alpha-{unit.id}-{n:02d}'
                self.snap(got[0], shot + '-before', full=True, force=True)
                if got2 is not None:
                    self.snap(got2[0], shot + '-after', full=True, force=True)
                self.log(ev='alpha_drop_evidence', stage=stage, unit_id=unit.id, loc=unit.loc,
                         point=target or loc_point(unit.loc), slot=src, read=name, stars=stars,
                         said=said, shot=shot, after=got2 is not None)
            info = CATALOG.unit(unit.name) if unit.name else None
            if (said and 'not valid on target' in said.lower() and name == unit.name
                    and info is not None and 'Riftbeast' in info.traits):
                # the Riftbeast the panel confirmed just before the drop is already the Alpha: the game answers a
                # second Mark on it "Item not valid on target" (game 84 3-3: the Mark took on Scuttlecrab, the next
                # drop on it was refused; game 86 2-6..4-1: the Crab showed the Alpha's aura, 13 drops refused and
                # retried every phase). Recorded as its mark, so the Mark is not asked for it again.
                self.roster.mark_used(base(item), unit, scr.stage)
                self.log(ev='mark_already', stage=scr.stage, item=item, unit=repr(unit), said=said)
                return True
            if said:
                self.fail_cause = f'the game refused it: "{said}" ({item} on {unit})'
                self.log(ev='game_refused', stage=scr.stage, item=item, unit=repr(unit), said=said)
                self.items_from_panel_after_refusal(unit, item, said, scr)
                return False
            self.roster.mark_used(base(item), unit, scr.stage)
            self.log(ev='mark_used', stage=scr.stage, item=item, unit=repr(unit))
            return True
        # the new icon animates in: read at the first changed frame it was missing or matched another item, and
        # 9 of 26 'failed' hand-outs (runs 221257, 152104, 213127) showed the expected item on the unit a moment
        # later. So: done as soon as the row shows the expected item, else the row read last, 2 s after the drop
        left, after, frame2, said, said_at = False, None, None, None, 0.0
        left_at, panel_tried = None, False
        # the icon can take over 2 s to show: 'failed' hand-outs of run 20261010-121458 (fail08, 10, 16) showed the expected
        # item in the frame saved just after the 2 s window (Archangel's on Kog'Maw, Adaptive Helm on Amumu ...); 4 s was
        # still short in run 20261014-063145 (4-3 Bloodthirster on Master Yi, 4-6 Kraken's on Aphelios: both rows read
        # empty in the window and right in the frame saved after it). The wait ends as soon as the item shows.
        end = time.time() + 7.0
        dropped_at = time.time()
        while time.time() < end:
            time.sleep(0.08)
            got2 = self.cap.read()
            if got2 is None:
                continue
            frame2 = got2[0]
            now = read_item_bench(frame2)
            # by count, not by slot: the item bench closes up, and a second copy slides into the emptied slot (run
            # 20261015-003101 3-2: a Tear landed on Krug, the other Tear moved up, 'still on the bench' - the retry
            # built Blue Buff on Krug)
            if n_here:
                left = sum(1 for k, n in enumerate(now) if raw_name(frame2, k, n) == base(item)) < n_here
            else:                                # the picture never saw it as this item (named by text): the slot
                left = raw_name(frame2, src, now[src]) != src_was
            if not left and said is None and time.time() - said_at > 0.4:
                said, said_at = self.game_message(frame2), time.time()
                said = None if about_other(said, item) else said
                if said:
                    break                        # the game refused it, in its own words: no need to wait on
            if not left and time.time() - dropped_at > 1.5:
                # the item is still on the item bench 1.5 s after the drop: the drop missed (the bench closes up at
                # once) - no 7 s wait for an icon that cannot come (the owner, game 74b: "one failed should not block
                # the rest"; 3-1's missed B.F. Sword held every other step 12.3 s)
                break
            if left:
                if is_consumable(item):
                    break
                row = self.fight_row(frame2, target, expect) if fight else self.unit_row(frame2, unit, expect)[1]
                if row is not None:
                    after = row
                    names_ = [n for n, _, _ in row]
                    if expect in Player.radiant_row(names_, before, expect) or (not done and any(n in makes and n not in (before or []) for n in names_)):
                        break
                left_at = left_at or time.time()
                if not fight and not REPLAY and not panel_tried and time.time() - left_at >= 1.5:
                    # the item is off the bench and the row still lacks it: the info panel shows it at once. In 40 gives
                    # of games 2026-10-21..26 the row read empty or short for all 7 s and the panel then showed the
                    # item - ~8 s a give (game 25 4-1: Void Staff on Morgana took 8.7 s, so Jeweled Gauntlet and
                    # Stoneplate went in combat and the roll-down was cut)
                    panel_tried = True
                    others = [loc_point(o.loc) for o in self.roster.board() if o.name and o.name != unit.name][:1]
                    name, _, held = self.read_fresh(loc_point(unit.loc), others)
                    if name == unit.name and held and (expect in held or (not done and any(
                            n in makes and n not in (before or []) for n in held))):
                        self.log(ev='items_from_panel', stage=stage, unit=repr(unit), items=held,
                                 row=[n for n, _, _ in after or []], early=True)
                        after = [(n, 1.0, None) for n in held]
                        break
        if not left and said:
            self.log(ev='game_refused', stage=scr.stage, item=item, unit=repr(unit), said=said)
            self.refused_panel = None
            landed = self.items_from_panel_after_refusal(unit, item, said, scr, (expect, held))
            if landed is not None:
                left, after = True, [(n, 1.0, None) for n in landed]
            else:
                Player.misnamed_component(self, item, getattr(self, 'refused_panel', None), src, got[0], scr)
        if not left:
            from tft_player.items import item_desc
            rule = item_desc(base(item))
            if said:
                # run 20261010-113319 6-1/6-2: Guinsoo's dropped on a 3-item Azir four times; the game said why
                # each time and the log only guessed
                self.fail_cause = (f'the game refused it: "{said}" ({item} is still on the item bench; {unit} '
                                   f'holds {unit.items or "no items"})')
                return False
            self.fail_cause = ('the item is still on the item bench after the drop (missed the unit, 3 items '
                               'already, or the game refused it' + (f': "{rule}"' if is_consumable(item) and rule else '') + ')')
            last = getattr(self, '_last_drop', None)
            if not fight and last and last[0] == unit.id and last[1] in ('body', 'centre'):
                # the next drop for this unit goes to its other point (under its bar / its hex centre), and after both
                # this round, none until the next (benchmark game 20261021-122616: 15 drops under Teemo's bar, which
                # rode the Sprykin BFF at (1, 3), all stayed on the item bench, 3-6 to 5-1)
                missed = self.__dict__.setdefault('drop_missed', {})
                st, tags = missed.get(unit.id, (None, set()))
                missed[unit.id] = (last[2], (tags if st == last[2] else set()) | {last[1]})
            return False
        self.items_done.add((src, item))
        if track is not None:
            track.taken(src)                     # off the bench; the items below move up
            Player.bench_shifted(self, src)
        if is_consumable(item):
            if is_duplicator(item) and unit.name:
                # the game puts a 1-star copy on the bench (leftmost free slot) and merges 3 copies
                self.roster.last_merge = None
                copy = self.roster.bought(unit.name, ours=True)
                merged = (self.roster.last_merge or {}).get('unit')
                if merged is not None:
                    # run 20261022-010349 5-2: the copy made Sentinel 2 at once, the news spoke of "a 1-star copy (unit
                    # 37) on the bench", and the player sold that "spare" - the sale hit the 2-star
                    self.news_event('duplicated', scr.stage, f'{item} used on {unit}: its copy merged at once with '
                                    f'your other copies into {merged} (unit {merged.id}) - there is no spare copy')
                else:
                    self.news_event('duplicated', scr.stage, f'{item} used on {unit}: a 1-star {unit.name} copy '
                                                             f'(unit {copy.id}) was added on the bench')
                if self.roster.last_merge and not REPLAY:
                    self.confirm_merge(self.roster.last_merge, scr)
            return True
        if after is None and not fight and not REPLAY:
            # the row is hidden (our tactician's bar over it, run 20261015-132144 5-6): its info panel shows the items
            name, _, held_now = self.read_fresh(target or loc_point(unit.loc), [])
            if name == unit.name and held_now:
                after = [(n, 1.0, None) for n in held_now]
                self.log(ev='items_from_panel', stage=scr.stage, unit=repr(unit), items=held_now)
        if after is None and not fight and not REPLAY:
            found = self.locate_item(item, makes, unit)
            if found[0] is not None:              # another unit shows it: it went there
                return self.misplaced(item, makes, unit, scr, f"{unit}'s item row could not be read", found=found)
        if after is None:
            self.fail_cause = (f'{item} left the item bench, but {unit}\'s item row could not be read - where it '
                               f'went is unverified')
            unit.items_read = False
            self._unit_items_at = 0              # read every unit's items at once
            return False
        row = Player.radiant_row([n for n, _, _ in after], before, expect)
        # the icon that appeared is the item we gave: compared by what the row holds, not by place - the game can put
        # the new item first (run 20261016-052413 4-3: Flickerblades showed as ['unknown', 'Last Whisper', 'Blue
        # Buff'], every place differed, nothing was learned, and the give was reported missing and queued again)
        added = Counter(row) - Counter(before or [])
        k = row.index('unknown') if row.count('unknown') == 1 else None
        if (k is not None and dict(added) == {'unknown': 1} and after[k][2] is not None
                and len(row) == len(before or []) + (0 if done else 1)):
            learn_icon(after[k][2], expect)
            row[k] = expect
            self.log(ev='icon_learned', stage=scr.stage, name=expect, how='drag outcome')
        if expect not in row and not done:
            made = next((n for n in row if n in makes and n not in (before or [])), None)
            if made:
                expect = done = made         # it met a component the records did not have and completed it
        if expect not in row and not fight and not REPLAY:
            # the row lacks it: its info panel first - benchmark game 20261020-181325 3-3 and 3-6: Azir at (0,0) read
            # with no items for 7 s after each give (a Rod met his B.F. Sword; then a Gunblade); both were reported
            # missing, though he held them, and the model spent an Item Remover on him
            others = [loc_point(o.loc) for o in self.roster.board() if o.name and o.name != unit.name][:1]
            name, _, held = self.read_fresh(loc_point(unit.loc), others)
            if name == unit.name and held:
                self.log(ev='items_from_panel', stage=stage, unit=repr(unit), items=held, row=row)
                after, row = [(n, 1.0, None) for n in held], list(held)
                if expect not in row and Counter(row) - Counter(before or []) == Counter({'unknown': 1}):
                    # one new icon the panel does not know: the item given (3-3 of that game: Thief's Gloves on
                    # Xayah, her panel "Hextech Gunblade, Bloodthirster, unknown" - reported missing)
                    row[row.index('unknown')] = expect
                    after = [(n, 1.0, None) for n in row]
                made = next((n for n in row if n in makes and n not in (before or [])), None)
                if expect not in row and not done and made:
                    expect = done = made
        if expect not in row:
            had = list(before or []) + [c for c in unit.items or [] if c not in (before or [])]
            if len([n for n in row if n != 'unknown']) < len(had):
                # a row with fewer items than the unit held is a short read (items leave a unit only by a Remover or
                # a sell): the records stay and its info panel decides (run 20261021-231230 3-5: Cassiopeia's row read
                # empty after Gloves met her Rod; her records lost Shojin and Morellonomicon until 4-2, the model
                # planned with them missing and a Rod was refused on her full item slots)
                unit.items_check = True
                from tft_player.items import held_after_give
                unit.items_expect = held_after_give(had, base(item), done)    # for the unknown icons of that read
                self.reread = getattr(self, 'reread', set()) | {unit.id}
            else:
                unit.items, unit.items_read = row, True
            self._unit_items_at = 0
            if not fight and not REPLAY:
                return self.misplaced(item, makes, unit, scr, f'{unit} shows {row or "no items"} (expected {expect})')
            self.fail_cause = (f'{item} left the item bench, but {unit} shows {row or "no items"} '
                               f'(expected {expect}): it may be on another unit')
            return False
        # what it holds now follows from our own drop: what it held, plus the item (or the item it completed). A row
        # read with fewer icons than that missed some - items leave a unit only by a Remover, a sell or a temporary
        # item running out (run 20261013-220557 5-1/5-2: a Rod on 3-star Pebbles read as "Needlessly Large Rod"
        # alone, and Jeweled Gauntlet + Blue Buff were gone from the records - the model planned with them missing)
        from tft_player.items import held_after_give
        predicted = held_after_give(unit.items, item, done)
        if len([n for n in row if n != 'unknown']) < len(predicted) and all(n in predicted for n in row if n != 'unknown'):
            self.log(ev='items_kept', stage=scr.stage, unit=repr(unit), read=row, kept=predicted)
            lost = Player.fight_row_lacks(before, row) if fight else []
            if lost:
                # a fight row read off the bar nearest the drop that lacks a completed item the unit held is another
                # unit's row: the item may have landed on a neighbour (request e43, game 45 4-7: Edge of Night for
                # Brambleback at (3,5) read as [Edge of Night] without his Rageblade - it was on Scuttlecrab at
                # (3,4); the records kept it on Brambleback to 5-3 and the last fight was played without it). His
                # panel decides, first among the units to read
                unit.items_check = True
                unit.items_expect = predicted
                self.reread = getattr(self, 'reread', set()) | {unit.id}
                self.log(ev='fight_give_doubt', stage=scr.stage, unit=repr(unit), item=item, read=row, lacks=lost)
                self.news_event('item_unverified', scr.stage,
                                f"{item} given in the fight to {unit.name} (unit {unit.id}): the item row read there "
                                f"shows {', '.join(row) or 'nothing'} without his {', '.join(lost)} - it may have "
                                f"landed on a unit next to him; his info panel is read at the next chance")
            row = predicted
        unit.items, unit.items_read = row, True
        Player.note_final(self, unit, base(item), expect)
        self._gave_at = time.time()
        return True

    GIVE_LEFT_S = 0.6        # s after the drop for the item to leave the item bench; still there then, the drop missed

    def give_fast(self, a, scr, slot, item, unit):
        """A give in a planning phase at the speed of the drag (the owner, 2026-11-08, on game 74b's 4-6: "you do not
        need to check for every action, it is a waste of time ... same with the items, instantly, ms operation" - five
        gives took 16.1 s there: a tooltip hover before each drag, the unit's icon row read before and after it, waits
        of up to 7 s for the new icon, info-panel reads). The slot is the one the item tracker names (by its tooltip or
        its picture), the drop point the unit's own health bar in one frame; after the drag one thing is looked at: the
        item left the item bench (the bench closes up at once - the next give's slot depends on it, ~0.1 ms a frame).
        What the unit holds then follows from our own drop (held_after_give), as unit identities follow from our own
        actions; its icon row is read by the usual item reads, outside the action.
        -> True / False, or None when the tracker does not name the item there (an unread or disputed slot): the
        checked give then reads its tooltip."""
        from tft_player.items import BY_PAIR, COMPONENTS, bench_filled, held_after_give, radiant_twin
        base = lambda n: re.sub(r'\s*\(.*$', '', n or '')
        track = self.item_track
        names = track.names()
        here = [k for k, n in enumerate(names) if n and base(n) == base(item)] or \
            [k for k, n in enumerate(names) if n and radiant_twin(base(n), base(item))]
        if not here:
            return None
        src = slot if slot in here else here[0]
        got = self.cap.read()
        if got is None:
            return None
        t0, frame = time.perf_counter(), got[0]
        filled0 = bench_filled(frame)
        if not filled0[src]:
            return None                          # the screen shows no item there: the checked give looks
        # the slot's picture must not name another item (~27 ms a read): the tracker can be stale after a misjudged
        # action (game 77 4-1: the slot it called Bramble Vest held Titan's Resolve, and the give put it on Sentinel).
        # A picture that names another item sends the give to the checked path, which reads its tooltip
        seen = read_item_bench(frame)
        pic = seen[src] if src < len(seen) else None
        if pic and pic != 'unknown' and base(pic) != base(item) and not radiant_twin(base(pic), base(item)):
            self.log(ev='give_slot_differs', stage=getattr(scr, 'stage', None), slot=src, tracked=item, picture=pic)
            return None
        stage = getattr(scr, 'stage', None)
        target, how = Player.drop_point(self, unit, self.unit_bar(frame, unit.loc), stage)
        if how is None:
            return False
        self._last_drop = (unit.id, how, stage)
        self.inp.drag(slot_point(src), target or loc_point(unit.loc))
        t_drag = time.perf_counter()
        # gone on frames over 0.12 s: a missed drop's icon can be off its slot for a frame as it snaps back
        n0, left, since, got2 = sum(filled0), False, None, None
        end = time.time() + self.GIVE_LEFT_S
        while time.time() < end:
            got2 = self.cap.read() or got2
            if got2 is None:
                continue
            if sum(bench_filled(got2[0])) < n0:
                since = since or time.time()
                if time.time() - since >= 0.12:
                    left = True
                    break
            else:
                since = None
        ms = dict(drag=round((t_drag - t0) * 1000), check=round((time.perf_counter() - t_drag) * 1000))
        if not left:
            said = self.game_message(got2[0]) if got2 is not None and not REPLAY else None
            said = None if about_other(said, item) else said
            self.log(ev='give_fast', stage=stage, item=item, unit=repr(unit), slot=src, how=how, ok=False, said=said,
                     ms=ms)
            if said:
                self.log(ev='game_refused', stage=stage, item=item, unit=repr(unit), said=said)
                self.fail_cause = (f'the game refused it: "{said}" ({item} is still on the item bench; {unit} holds '
                                   f'{unit.items or "no items"})')
                self.items_from_panel_after_refusal(unit, item, said, scr)
                return False
            self.fail_cause = ('the item is still on the item bench after the drop (missed the unit, 3 items already, '
                               'or the game refused it)')
            if how in ('body', 'centre'):
                # the next drop for this unit goes to its other point, and after both none this round (give_item)
                missed = self.__dict__.setdefault('drop_missed', {})
                st, tags = missed.get(unit.id, (None, set()))
                missed[unit.id] = (stage, (tags if st == stage else set()) | {how})
            return False
        given = base(names[src])
        held = [i for i in unit.items or [] if i and i != 'unknown']
        pair = lambda c: tuple(sorted((c, given)))
        done = next((BY_PAIR[pair(c)] for c in held if given in COMPONENTS and c in COMPONENTS and pair(c) in BY_PAIR),
                    None)
        unit.items, unit.items_read = held_after_give(held, given, done), True
        unit.items_predicted = True
        track.taken(src)                         # off the bench; the items below move up
        Player.bench_shifted(self, src)
        self.bench_items = track.names()
        Player.note_final(self, unit, given, done or given)
        self._gave_at = time.time()
        self.log(ev='give_fast', stage=stage, item=item, unit=repr(unit), slot=src, how=how, ok=True, made=done,
                 ms=ms)
        if Player.is_comp_carry(self, unit) and not REPLAY:
            return Player.carry_give_landed(self, unit, given, done, held, scr)
        return True

    def is_comp_carry(self, unit):
        """The unit is the fixed comp's carry (its item plan's first holder: Brambleback)."""
        from tft_player import item_plan
        from tft_player.coach import forced_comp
        ip = item_plan.plan_for(getattr(forced_comp(), 'name', None))
        return bool(ip and unit is not None and unit.name == ip['carry'][0])

    CARRY_ROW_S = 1.0                        # s the carry's item row is looked at after a give to him

    def carry_give_landed(self, unit, given, made, before, scr):
        """A give to the comp's carry is looked at once on his item row: one that landed on a neighbour is found there
        (locate_item) and moved back onto him by the item-holder records - the Item Remover (misplaced, handover_action).
        The owner, game 88 4-3, x32.5: "items gives to the wolf, is this an error? why not two both on bramble? it should
        always field the main carry items first" - Titan's Resolve for Brambleback at (0,6) went onto Murkwolf at (1,5),
        both read only at 4-5. A row that cannot be read keeps the give (as every give at the drag's speed).
        -> True when it is on him (or unread), else False."""
        from tft_player.items import BY_PAIR, COMPONENTS
        expect, end, names = made or given, time.time() + self.CARRY_ROW_S, None
        while time.time() < end:
            got = self.cap.read()
            if got is None:
                continue
            _, row = self.unit_row(got[0], unit, expect)
            if row is None:
                continue
            names = [n for n, _, _ in row]
            if expect in names or len([n for n in names if n == 'unknown']) > len([n for n in before if n == 'unknown']):
                return True                      # on him (an icon not named yet that came now counts)
        if names is None:
            return True
        makes = {v for k, v in BY_PAIR.items() if given in k} if given in COMPONENTS else set()
        found = self.locate_item(given, makes, unit)
        self.log(ev='carry_give_check', stage=getattr(scr, 'stage', None), unit=repr(unit), item=given, row=names,
                 found=repr(found[0]) if found[0] is not None else None)
        if found[0] is None:
            return True                          # not on another unit either: his row may lag - the reads go on
        unit.items, unit.items_read = list(before), True
        return self.misplaced(given, makes, unit, scr, f"{unit}'s item row shows {names or 'nothing'}", found=found)

    @staticmethod
    def radiant_row(row, before, expect):
        """A unit's item row read after a give: one icon new since `before` read as the Radiant twin of the item given
        (or the plain one of a Radiant given) is that item - the icon read does not tell them apart (radiant_twin)."""
        from tft_player.items import radiant_twin
        row = list(row)
        if expect in row:
            return row
        new = Counter(row) - Counter(before or [])
        k = next((j for j, n in enumerate(row) if new.get(n) and radiant_twin(n, expect)), None)
        if k is not None:
            row[k] = expect
        return row

    @staticmethod
    def fight_row_lacks(before, row):
        """The completed items a unit held before a fight give that the row read after it lacks (request e43): a
        component can merge into the given item, a completed item cannot leave a unit in a fight."""
        from tft_player.items import COMPONENTS
        seen = [n for n in row or [] if n != 'unknown']
        if 'unknown' in (row or []):
            return []                            # an unread icon may be it
        return [n for n in before or [] if n and n != 'unknown' and n not in COMPONENTS
                and n != "Thief's Gloves" and n not in seen]

    def items_from_panel_after_refusal(self, unit, item, said, scr, landed_if=None):
        """The game refused a give: what the unit holds is read off its info panel (all its items) and the records and
        the model are told - its item row can read short (benchmark game 20261018-134910 5-2, 6-2: 3-star Veigar
        recorded with [Giant Slayer] only; the game refused four gives, "Item not valid on target", and the model went
        on planning them). landed_if = (expect, had): a panel that shows the expected item (the one given, or the one it
        completes with a component the unit held) newly on the unit means the message was a notice, not a refusal ->
        the panel's items (else None). Only the expected item counts: with short records any of a 3-item unit's items
        could pass for one a component "made" (4-2 of run 20261021-231230: a refused Rod on Cassiopeia, who held
        Jeweled Gauntlet, Shojin and Morellonomicon, her records empty). That run's notices: "pairing was auto
        assigned! ..." (4-1, a Rod on Veigar) and "Don't forget to use your Alpha Mark!" (6-3, Infinity Edge on Elder
        Dragon) were reported refused, the items on the units."""
        if REPLAY or not unit.on_board and unit.loc[0] != 'bench':
            return None
        try:
            got = self.cap.read()
            bars = B.green_bars(got[0]) if got else None
            others = [self.unit_point(o.loc, bars) for o in self.roster.units
                      if o.name and o.name != unit.name and not o.missing and o.loc[0] in ('board', 'bench')][:2]
            name, _, held = self.read_fresh(self.unit_point(unit.loc, bars), others)
        except NotForeground:
            return None
        self.log(ev='items_after_refusal', stage=scr.stage, unit=repr(unit), read=name, items=held)
        if name != unit.name or held is None:
            return None
        if landed_if:
            expect, had = landed_if
            if expect in Counter(held) - Counter(had or []):
                self.log(ev='notice_not_refusal', stage=scr.stage, item=item, unit=repr(unit), said=said, items=held)
                return list(held)
        if list(held) != list(unit.items or []):
            unit.items, unit.items_read = list(held), True
        self.refused_panel = list(held)
        self.news_event('item_refused', scr.stage, f'the game refused {item} on {unit} ("{said}"); its info panel shows '
                                                   f'{", ".join(held) if held else "no items"}')

    def picture_name(self, sig, n):
        """A picture match the game has already refused for this very icon (misnamed_component) is no name."""
        wrong = self.__dict__.get('item_wrong') or {}
        return 'unknown' if n and n != 'unknown' and sig in wrong and re.sub(r'\s*\(.*$', '', n) == wrong[sig] else n

    def misnamed_component(self, item, held, src, frame, scr):
        """The game refused a component on a unit whose info panel shows a free item slot: a component is valid on
        any such unit, so the item-bench slot holds something else. Its picture name is dropped (that icon is not
        read as it again), the slot becomes 'unknown' and its tooltip is read at once (run 20261028-032643: the
        Alpha Mark, read so by its tooltip at 5-2, became "Needlessly Large Rod" by its picture at 5-5; four gives
        to empty or 1-item units were refused, 5-5 to 6-2, and the model spent orders on it in 8 answers).
        -> True when the slot was renamed."""
        from tft_player.items import COMPONENTS
        base = lambda n: re.sub(r'\s*\(.*$', '', n or '')
        if base(item) not in COMPONENTS or held is None or len(held) >= 3 or src is None:
            return False
        sig = self.item_sig(frame, src) if frame is not None else None
        if sig is not None:
            self.__dict__.setdefault('item_wrong', {})[sig] = base(item)
            if base(self.item_names.get(sig) or '') == base(item):
                del self.item_names[sig]
        track = self.__dict__.get('_item_track')
        if track is not None and 0 <= src < track.n:
            track.slots[src], track.text[src] = 'unknown', False
        self.identify_items_now = True
        self.log(ev='item_misnamed', stage=scr.stage, slot=src, item=item, held=list(held))
        self.news_event('item_misnamed', scr.stage, f'item bench slot {src} is not {item}: the game refused it on a '
                                                    f'unit with a free item slot ({len(held)} items); its tooltip is '
                                                    f'read again')
        return True

    def hold_item_over(self, a, scr):
        """An item for a fielded unit in the last 2 s of planning waits: the round flips under the drop, the units fly
        off and the read finds no items (benchmark game 20261018-131328 3-5: Sparring Gloves to Pebbles at 1 s,
        'missing'). -> True when held over (the plan gives it again)."""
        unit = a.arg[2]
        # (a bench unit too: benchmark game 20261018-150046 3-6, a Giant's Belt for Alistar on the bench at 1 s)
        # A countdown rise read once (Phase.rise) is the fight starting more often than a misread: run 20261022-034805
        # 4-7 read 3 s, an 8 s give, then 28 - still "planning" until a second read - and a Negatron Cloak for Elise at
        # (3, 2) dropped into the fight landed on Morgana, whose Giant's Belt it made into Evenshroud
        flipping = scr.timer is not None and (scr.timer <= 2 or getattr(self.phase, 'rise', None) is not None)
        if (getattr(self.phase, 'planning', False) and flipping
                and not getattr(a, 'fight', False) and getattr(unit, 'loc', ('',))[0] in ('board', 'bench')):
            self.fail_cause = 'queued: the round flips in the last seconds'
            self.deferred = True
            return True
        return False

    def drop_point(self, unit, bar, stage):
        """Where an item for this unit is dropped: (point, 'body') just under its bar, (None, 'centre') for its hex
        centre, or (None, None) - not now, fail_cause says why. A point whose drop landed on another unit this round
        is not used again (release run 20261016-141957 4-6: a Recurve Bow for Xayah at (0,0) landed on Nidalee at
        (1,0), and the Negatron Cloak dropped on the same point made Kraken's Fury on Nidalee)."""
        stage_missed, tags = self.__dict__.setdefault('drop_missed', {}).get(unit.id, (None, set()))
        tags = tags if stage_missed == stage else set()
        if getattr(unit, 'riding', None) and unit.on_board:
            # the user, 2026-10-21: a BFF rider takes items dropped on the rider itself (the champion on top), not on
            # the BFF. Its bar stands ~93 px over its hex centre (others 50-76): 45 px under it is air over its head,
            # and the hex centre is the BFF's horns (benchmark game 20261021-153057 3-3..4-1: six drops for the rider
            # Teemo stayed on the item bench). Its body is ~85 px under its bar, ~15 px right of the bar's middle (three
            # recorded frames). Its hex is no safe guess (Teemo sat between (1, 3) and (1, 4) in game 20261021-134046,
            # and a point off that hex is on the next unit's bar): without its bar, or after a miss, none this round
            if bar is not None and 'body' not in tags:
                return (bar[0] + 15, bar[1] + 85), 'body'
            self.fail_cause = (f'{unit} rides the BFF: ' + ('a drop on it did not take' if 'body' in tags else
                               'its health bar is not seen at its hex') + ' - not given again until the next round')
            self.drop_missed[unit.id] = (stage, tags | {'body'})
            return None, None
        if bar is not None and 'body' not in tags:
            # onto the unit's own body, just under its bar - not its hex centre, where a neighbour's big model can
            # reach (run 20261015-132144 5-2: a Negatron Cloak for Aphelios at (0,2) went to Kog'Maw at (0,1)). With a
            # unit right behind it, not above its own row: there, under the bar is that unit's hex (4-6 above: the
            # drop was 31 px above Xayah's hex centre, 39 px from Nidalee's)
            point = (bar[0], bar[1] + 45)
            others = [o for o in (self.roster.board() if getattr(self, 'roster', None) else []) if o is not unit]
            if unit.on_board:
                hx, hy = B.hex_xy(*unit.loc[1:])
                # a tall model's bar stands far above its body: under the bar is air above its head (benchmark game
                # 20261018-131328 4-3, 5-1: 3-star Pebbles' bar ~120 px over its hex centre, two Sparring Gloves
                # stayed on the item bench) - the drop stays within 40 px of the hex centre
                point = (point[0], max(point[1], hy - 40))
                if any(0 < hy - B.hex_xy(*o.loc[1:])[1] < 100 and abs(B.hex_xy(*o.loc[1:])[0] - hx) < 90
                       for o in others):
                    point = (point[0], max(point[1], hy - 25))
            return point, 'body'
        if 'centre' in tags:
            self.fail_cause = (f'drops for {unit} did not take this round (dropped '
                               f'{" and ".join("under its bar" if t == "body" else "on its hex centre" for t in sorted(tags))}:'
                               f' they landed on another unit or stayed on the item bench) - not given again until the next'
                               f' round')
            return None, None
        return None, 'centre'

    def locate_item(self, item, makes, target):
        """Where a give that missed `target` went: the unit whose item row (or, when the row is covered, its info panel)
        now shows `item` - or an item it completed there (`makes`) - more often than its records do.
        -> (unit, item name) or (None, None). The units nearest the target first: a drop lands on a neighbour."""
        got = self.cap.read()
        if got is None:
            return None, None
        frame, wanted, hidden = got[0], {item} | set(makes or ()), []
        near = lambda u: (0 if u.loc[0] == target.loc[0] else 1,
                          sum(abs(a - b) for a, b in zip(u.loc[1:], target.loc[1:])))
        units = sorted((u for u in self.roster.units if u is not target and not u.missing
                        and u.loc[0] in ('board', 'bench')), key=near)
        gained = lambda names, u: [n for n in names if n in wanted and names.count(n) > (u.items or []).count(n)]
        for u in units:
            _, row = self.unit_row(frame, u)
            if row is None:
                hidden.append(u)
                continue
            names = [n for n, _, _ in row]
            new = gained(names, u)
            if new:
                u.items, u.items_read = names, True
                return u, new[0]
        for u in hidden[:3]:                     # rows covered (our tactician's bar, an effect): their panels
            name, _, held = self.read_fresh(loc_point(u.loc), [])
            new = gained(held, u) if name == u.name else []
            if new:
                u.items, u.items_read = list(held), True
                return u, new[0]
        return None, None

    def misplaced(self, item, makes, target, scr, seen, found=None):
        """A give left the item bench but its target does not show it: the item is found (locate_item) and gets its
        current holder (the unit it landed on) and its target holder (the unit the plan named) in the item-holder
        records (item_finals, the user's design 2026-10-14), so handover_action moves it there - an Item Remover, or a
        sell of a holder outside the comp. Run 20261015-132144 5-2: a Negatron Cloak for Aphelios went to Kog'Maw and
        stayed there all game. Not found anywhere: reported missing. -> False (the give itself did not land)."""
        stage = getattr(scr, 'stage', None)
        holder, got = found or self.locate_item(item, makes, target)
        if holder is not None:
            last = getattr(self, '_last_drop', None)
            if last and last[0] == target.id and last[1] in ('body', 'centre'):
                # the drop point that reached a neighbour is not used for this unit again this round (give_item)
                missed = self.__dict__.setdefault('drop_missed', {})
                st, tags = missed.get(target.id, (None, set()))
                missed[target.id] = (last[2], (tags if st == last[2] else set()) | {last[1]})
            self.set_final(holder.id, got, target.name, target.star)
            self.fail_cause = (f'{item} landed on {holder} (unit {holder.id}) as {got}, not on {target}: it is moved '
                               f'to {target.name} (its target holder)')
            self.news_event('item_misplaced', stage, f'{item} (given to {target}, unit {target.id}) landed on '
                            f'{holder} (unit {holder.id}) as {got}; it is being moved to {target.name}')
            self.log(ev='item_misplaced', stage=stage, item=item, got=got, holder=repr(holder), target=repr(target))
        else:
            self.fail_cause = (f'{item} left the item bench, but {seen} and no other unit shows it: missing '
                               f'(the item reads go on)')
            self.news_event('item_missing', stage, f'{item} (given to {target}, unit {target.id}) left the item '
                            f'bench but no unit shows it')
            self.log(ev='item_missing', stage=stage, item=item, target=repr(target))
        return False

    def game_message(self, frame):
        """The game's own one-line notice over the bench ("Item not valid on target. Check item description to
        equip appropriately."), or None. One OCR crop, ~0.1 s; only read after an action did not land."""
        h, w = frame.shape[:2]
        s = h / 1080
        box = (int(w / 2 - 480 * s), int(800 * s), int(w / 2 + 480 * s), int(852 * s))
        try:
            txt = (self.reader._texts(frame, [box])[0] or '').strip()
        except Exception:
            return None
        # a real notice is mostly whole words; a tooltip or the board behind reads as fragments (run
        # 20261010-235723 2-1: "them f r 1.5 scod age   mag am nd sg")
        words = txt.split()
        real = [w for w in words if re.fullmatch(r"[A-Za-z][a-z']{2,}[.,!?:]?", w)]
        ok = (re.fullmatch(r"[A-Za-z0-9 .,'!?:()-]{12,}", txt) and len(real) >= 4
              and len(real) >= 0.7 * len(words) and '  ' not in txt)
        return txt if ok else None

    def selected_here(self, frame, loc):
        """Is the selection ring (bright cyan, drawn at the feet of the unit whose panel is open) at
        loc? Measured on recorded reads: the ring covers >4% of a band around the unit's feet."""
        h, w = frame.shape[:2]
        s = h / 1080
        x, y = (B.BENCH[loc[1]][0], 775) if loc[0] == 'bench' else B.hex_xy(loc[1], loc[2])
        cx, cy = w / 2 + (x - 960) * s, y * s
        reg = frame[int(cy - 30 * s):int(cy + 30 * s), int(cx - 70 * s):int(cx + 70 * s)].astype(np.int16)
        if reg.size == 0:
            return True                          # off screen: no evidence against it
        r, g, b = reg[..., 0], reg[..., 1], reg[..., 2]
        return float(((b > 190) & (g > 170) & (b - r > 35)).mean()) > 0.04

    def unit_bar(self, frame, loc, places=None):
        """The health bar of the unit at loc in this frame (board: the one bar near the hex), or None."""
        from tft_player.state import _hex_near
        places = places if places is not None else B.bar_places(frame, exclude=self.watcher.tac[1])[2]
        bar = next((b for b, l in places if l == loc), None)
        if bar is None and loc[0] == 'board':
            near = [b for b, l in places if l[0] == 'board' and _hex_near(l[1:], loc[1:])]
            bar = near[0] if len(near) == 1 else None
        return bar

    def remember_look(self, u, frame, label=False):
        """Store u's appearance (champs.embed of its crop): later reads of its slot/hex are compared
        with it. label=True: its name was just read off its info panel, so the crop also goes into
        the champion library (data/champ-library) for naming by sight."""
        from tft_player import champs
        if frame is None or (REPLAY and not os.environ.get('TFT_LOOKS')):
            return
        bar = self.unit_bar(frame, u.loc)
        if bar is None and label:
            got = self.cap.read()                 # the frame before the click had no bar there: now
            frame = got[0] if got else frame
            bar = self.unit_bar(frame, u.loc)
        if bar is None and u.loc[0] == 'bench':
            # bench units show no health bar in combat (run32: 26 reads lost): the slot's fixed point,
            # if the slot is not empty
            from tft_player import bench as BN
            crop0 = BN.slot_crop(BN._bench_band(frame), u.loc[1])
            occ = BN.occupied_share(crop0, u.loc[1])
            if occ is None or occ >= BN.EMPTY:
                bar = (B.BENCH[u.loc[1]][0], self.BENCH_BAR_Y)
        crop = champs.unit_crop(frame, bar) if bar is not None else None
        if crop is None:
            self.log(ev='look_not_taken', unit=repr(u), why='no health bar at its place' if bar is None else 'crop off screen')
            u.look = None
            return
        try:
            u.look, u.look_loc, u.look_misses = champs.embed([crop])[0], u.loc[0], 0
            if label and u.name:
                champs.add_to_library(crop, u.name, u.star)
        except Exception as e:                   # model file missing / onnx error: say so, go on
            self.log(ev='look_error', error=repr(e))
            u.look = None

    # same unit, same place (tools/eval_champs.py, 365 same-unit / 2892 different-champion pairs,
    # 960x600 recordings): bench 0.75 -> 5.4% of same units flagged, 2.1% of other champions pass;
    # board 0.70 -> 8.4% flagged, 7.3% pass. Two low readings in a row are needed to act.
    LOOK_SAME = {'bench': 0.75, 'board': 0.70}
    LOOK_SAME_MOVED = 0.60    # after a move (bench <-> board view differs): looser, not measured

    GALLERY_EVERY = 20        # s between two gallery crops of one unit (different poses, rounds)

    def grow_gallery(self, u, crop, vec, scr):
        """The recognition library grows from normal games: a unit whose name is certain (its buy
        seen landing, or read off its info panel) and that still looks like itself gives a crop now
        and then, bench and board, until its champion has champs.NEED per place."""
        from tft_player import champs
        place = u.loc[0]
        if not (u.name and getattr(u, 'sure', False) and not u.unverified and not u.missing):
            return
        if time.time() - getattr(u, 'gallery_at', 0) < self.GALLERY_EVERY or not champs.wants(u.name, place):
            return
        u.gallery_at = time.time()
        champs.add_to_library(crop, u.name, u.star, vec=vec)
        champs.add_progress(u.name, place == 'bench', place == 'board')
        self.log(ev='gallery', stage=scr.stage, unit=repr(u), place=place)

    def check_looks(self, frame, places, scr):
        """Every unit still looks like itself: its crop now against the one stored when its identity
        was set (a wrong name never shows in occupancy - run27 tracked Kayle as Xayah all game).
        Runs in a background thread (~20 ms per unit); two low readings in a row mark the unit
        UNVERIFIED and it is read off its info panel once. A unit with no stored look gets one."""
        from tft_player import champs
        from tft_player.state import _hex_near
        if getattr(self, '_looks_busy', False) or (REPLAY and not os.environ.get('TFT_LOOKS')):
            return
        todo = []
        for bar, loc in places:
            u = self.roster.at(loc)
            if u is None and loc[0] == 'board':
                near = [x for x in self.roster.board() if _hex_near(x.loc[1:], loc[1:])]
                u = near[0] if len(near) == 1 else None
            if u is None or u.missing:
                continue
            crop = champs.unit_crop(frame, bar)
            if crop is not None:
                todo.append((u, crop))
        if not todo:
            return
        self._looks_busy = True

        def run():
            try:
                vecs = champs.embed([c for _, c in todo])
                for (u, crop), v in zip(todo, vecs):
                    if getattr(u, 'look', None) is None:
                        u.look, u.look_loc, u.look_misses = v, u.loc[0], 0
                        bought = getattr(u, 'bought_at', None)
                        if bought and time.time() - bought < 30 and u.loc == getattr(u, 'bought_loc', None) \
                                and u.name and not u.unverified and getattr(u, 'sure', False):
                            champs.add_to_library(crop, u.name, u.star)
                            u.labelled = True
                            self.log(ev='champ_labelled', how='buy landing', unit=repr(u))
                        continue
                    sim = max(float(np.dot(w, v)) for w in [u.look] + list(getattr(u, 'looks_ok', [])))
                    moved = getattr(u, 'look_loc', None) != u.loc[0]
                    if sim >= (self.LOOK_SAME_MOVED if moved else self.LOOK_SAME[u.loc[0]]):
                        u.look_misses = 0
                        if moved:
                            u.look, u.look_loc = v, u.loc[0]       # passed after a move: the new view
                        self.grow_gallery(u, crop, v, scr)
                        continue
                    if getattr(u, 'look_trusted', None) == u.star:
                        # read off its panel at this star level: a pose, an item glow or an effect, not
                        # another champion - the view is kept as one more reference, no re-read
                        u.looks_ok = (list(getattr(u, 'looks_ok', [])) + [v])[-4:]
                        u.look_misses = 0
                        continue
                    u.look_misses = getattr(u, 'look_misses', 0) + 1
                    if u.look_misses >= 2 and not u.unverified:
                        # checked silently, in idle time (identify_one): only a read that finds another
                        # champion is reported (unit_corrected) - the model is not sent false alarms
                        u.unverified = (f'looks different from when it was identified (similarity '
                                        f'{sim:.2f}): to be read')
                        self.reread = getattr(self, 'reread', set()) | {u.id}
                        self.tries[u.id] = 0
                        self.log(ev='look_changed', stage=scr.stage, unit=repr(u), sim=round(sim, 3))
                        u.look_before_read, u.name_before_read = u.look, u.name
                        u.look = None                      # taken again after the read
            except Exception as e:
                self.log(ev='look_error', error=repr(e))
            finally:
                self._looks_busy = False
        threading.Thread(target=run, daemon=True).start()

    def read_roster_items(self, frame, places, scr):
        """Every unit's items, read from the icon row over its health bar (items.read_unit_items).
        A changed reading must repeat once before it replaces the old one (a unit still walking,
        an icon half-drawn). 'unknown' = an item is there, not recognised."""
        from tft_player.items import read_unit_items
        from tft_player.state import _hex_near
        t0 = time.perf_counter()
        pend = getattr(self, '_items_pending', {})
        # a few units a step, in turn - a unit waiting for its confirming second read first: every unit's row each
        # 0.6 s took 4.3 s a time (run 20261015-161625: 325 s of the game, the loop blocked meanwhile)
        todo = []
        for bar, loc in places:
            u = self.roster.at(loc)
            if u is None and loc[0] == 'board':
                near = [x for x in self.roster.board() if _hex_near(x.loc[1:], loc[1:])]
                u = near[0] if len(near) == 1 else None      # two candidates: no guess
            if u is not None:
                todo.append((bar, loc, u))
        turn = self.__dict__.setdefault('_items_turn', {})
        todo.sort(key=lambda t: (t[2].id not in pend, t[2].items_read, turn.get(t[2].id, 0)))
        for bar, loc, u in todo[:self.ITEM_ROWS_PER_STEP]:
            turn[u.id] = time.time()
            got = [n for n, _, _ in read_unit_items(frame, bar)]
            if u.items_read:
                got = fill_unknown(got, u.items)
            # an icon its panel refuted (row_ghosts) is not read back onto the unit: run 20261029-070238, Murkwolf's
            # row read Hand Of Justice from 2-5 to 5-2, its panel showed no item four times, and each time the row put
            # the item back - the state showed a Hand Of Justice that was never there
            ghosts = getattr(u, 'row_ghosts', None)
            if ghosts:
                got = [n for n in got if n not in ghosts or n in (u.items or [])]
            if u.items_read and got == u.items:
                if getattr(u, 'items_predicted', False):
                    if pend.get(u.id) != got:
                        pend[u.id] = got
                        continue
                    u.items_predicted = False
                    self.log(ev='item_delivery', status='equipped', stage=scr.stage, unit_id=u.id,
                             loc=u.loc, equipped=list(got))
                pend.pop(u.id, None)
                continue
            if pend.get(u.id) != got:
                pend[u.id] = got                 # seen once: wait for a second identical read
                continue
            pend.pop(u.id, None)
            if u.items and len(got) < len(u.items) and not REPLAY:
                # fewer icons than the records hold: items leave a unit by our own Remover or sell (recorded at once)
                # or a temporary item running out - its info panel decides (benchmark game 20261020-200630: Azir's
                # row read short in 4 of 8 reads, his records lost two items, and three gives on him were refused: he
                # held three). Also for records not read yet - a merge's (the copies' items move to the merged unit):
                # run 20261021-231230 5-3, 2-star Morgana's row read empty and her Void Staff and two Morellonomicons
                # were gone from the state for the rest of the game
                if not getattr(u, 'items_check', False):
                    self.log(ev='items_short', stage=scr.stage, unit=repr(u), was=u.items, read=got)
                u.items_check = True
                self.reread = getattr(self, 'reread', set()) | {u.id}
                continue
            if got != u.items or not u.items_read:
                self.log(ev='unit_items', stage=scr.stage, unit=repr(u), was=u.items, read=got)
            u.items, u.items_read = got, True
            u.items_predicted = False
        self._items_pending = pend
        self.lat_items = time.perf_counter() - t0

    ITEM_ROWS_PER_STEP = 3                   # item rows read per loop step (read_roster_items)

    def open_container(self, slot, scr):
        """Open the container on a bench slot (E over it sells it, which opens its choice row)."""
        obj = self.bench_objs.get(slot)
        if not obj or obj['kind'] != 'interactable':
            self.log(ev='container_refused', slot=slot, seen=obj)
            print(f'refused: bench slot {slot} holds no recognised container ({obj})')
            return False
        Player.sell_key(self, loc_point(('bench', slot)))      # selling a container opens it
        self.container_opened = (obj['cls'], slot, time.time())
        self.auto_opened.add(slot)
        # a new choice screen comes: never the one answered before (run29 4-2: the second anvil's
        # screen was taken for the first, already answered one; the game picked on timeout)
        self.choice = None
        self.log(ev='container_open', stage=scr.stage, slot=slot, cls=obj['cls'], conf=obj['conf'])
        self.bench_objs.pop(slot, None)
        self.roster.bench_full_seen = False
        time.sleep(0.2)
        return True

    OPEN_LATE_S = 12                         # planning seconds left below which a container waits for the plan's steps
    PROBE_MIN = 0.55        # an unrecognised bench object scoring this against the containers is clicked (request e178)
    PROBE_CONTAINER = 0.80  # no unit panel opened and this score: a container (recorded units without a bar <= 0.77)

    def probe_bench_object(self, scr):
        """An unrecognised bench object that scores PROBE_MIN up to the container cutoff is clicked (right: the info
        panel, never E - that sells a unit) when the planning phase has time and the plan nothing to do, and what
        opens decides (request e178, the owner: "you could just click and confirm? there is plenty of time"): a unit's
        panel -> a unit (recorded, named); a choice row -> a container, answered as one; nothing, at a score over every
        recorded unit's -> a container (opened like one). Runs 20261103-233832 6-1/6-2 and 20261103-120119 6-1: item
        anvils on bench slot 5 scored 0.867 and 0.824 against 0.87, stayed 'unrecognised object' and were never opened.
        Over games 56-59 units without a detected bar score 0.55-0.77, anvils 0.82-0.99. -> True when it clicked."""
        from tft_player import bench as BN
        from tft_player.state import Owned
        if REPLAY or self.choice or not getattr(self.phase, 'planning', False) or scr.timer is None or scr.timer < 8:
            return False
        rp = getattr(self, 'round_plan', None)
        if rp is not None and rp.pending(self.roster) and scr.timer < 20:
            return False                         # the board is the plan's now
        if rp is not None and rp.all_in(scr.hp if scr.hp is not None else getattr(rp, 'last_hp', None)):
            return False                         # the last life's roll-down first (auto_open_container)
        tried = self.__dict__.setdefault('_probed', set())
        for slot, o in sorted(self.bench_objs.items()):
            if o['kind'] != 'unknown' or not self.PROBE_MIN <= (o.get('conf') or 0) < BN.CONTAINER \
                    or (slot, scr.stage) in tried or self.roster.at(('bench', slot)) is not None:
                continue
            tried.add((slot, scr.stage))
            got = self.cap.read()
            bars = B.green_bars(got[0]) if got else None
            others = [self.unit_point(u.loc, bars) for u in self.roster.units
                      if u.name and not u.missing and u.loc[0] == 'bench'][:2]
            name, stars, items = self.read_fresh(self.unit_point(('bench', slot), bars), others)
            if name:
                u = Owned(name, stars or 1, ('bench', slot))
                u.sure, u.items, u.items_read = True, list(items or []), True
                u.arrival_source = 'a bench object read as a unit'
                self.roster.units.append(u)
                self.bench_objs.pop(slot, None)
                self.roster.blocked = {k: v for k, v in self.roster.blocked.items() if k != slot}
                self.log(ev='bench_probe', stage=scr.stage, slot=slot, conf=o['conf'], found='unit', name=name,
                         stars=stars)
                self.news_event('game_change', scr.stage, f'bench slot {slot}: the object there is {u} (unit {u.id}, '
                                                          f'read off its panel)')
                return True
            got = self.cap.read()
            opts = self.read_choice(got[0]) if got is not None else []
            if not opts and o['conf'] < self.PROBE_CONTAINER:
                self.log(ev='bench_probe', stage=scr.stage, slot=slot, conf=o['conf'], found=None)
                return True
            cls = o.get('best') or 'container'
            self.__dict__.setdefault('_probed_containers', {})[slot] = cls
            self.bench_objs[slot] = {**o, 'kind': 'interactable', 'cls': cls}
            self.roster.blocked = {**self.roster.blocked, slot: f'a container ({cls})'}
            self.log(ev='bench_probe', stage=scr.stage, slot=slot, conf=o['conf'], cls=cls,
                     found='choice' if opts else 'container')
            if opts:
                # the click opened its choice row: answered as a container's
                self.container_opened = (cls, slot, time.time())
                self.auto_opened.add(slot)
                self.choice = None
                self.bench_objs.pop(slot, None)
            elif not self.auto_open:
                self.pending_events.append(f"EVENT CHOICE_CONTAINER_AVAILABLE: {cls} on bench slot {slot} "
                                           f"(execution_plan.open_container).")
            return True
        return False

    def container_waiting(self, scr):
        """A container on the bench that this planning phase has not opened yet, while there is time to open it (more than
        OPEN_LATE_S left) and the round plan has no steps left now - its moves and gives come first."""
        if not getattr(self.phase, 'planning', False) or not self.auto_open or scr.timer is None:
            return False
        if scr.timer <= self.OPEN_LATE_S:
            return False
        rp = getattr(self, 'round_plan', None)
        if rp is not None and rp.stage == scr.stage and not getattr(rp, 'steps_done', False):
            return False
        tried = getattr(self, '_auto_tried', set())
        return any(o.get('kind') == 'interactable' and (slot, scr.stage) not in tried
                   for slot, o in (getattr(self, 'bench_objs', None) or {}).items())

    def auto_open_container(self, scr):
        """A container is on the bench: if the model is free RIGHT NOW, hold it (coach.reserve_model) and open
        the container, so that the choice question goes out the moment its options show - in planning or in
        the fight. Never open and then wait for the model (the user, 2026-10-05). The held model is given
        back when the choice call ends, or when no choice screen came."""
        from tft_player import coach
        held = getattr(self, 'model_reserved', None)
        if held and not self.choice and time.time() - held > 8.0:          # opened, but no choice screen came
            self.model_reserved = None
            coach.release_model()
            self.log(ev='model_released', stage=scr.stage, why='no choice screen after opening')
        if not self.auto_open or self.choice or self.plan_pending or coach.STOPPED or held:
            return False
        if scr.timer is not None and scr.timer <= 1 and self.phase.planning:
            return False
        rp = getattr(self, 'round_plan', None)
        if (self.phase.planning and scr.timer is not None and scr.timer <= self.OPEN_LATE_S
                and rp is not None and rp.pending(self.roster)):
            # its choice screen holds every action until it is answered (~7-11 s): with the plan's steps still to
            # do, the container is opened in the fight instead (run 20261028-022436 6-1: an anvil opened with 5 s
            # left; Bloodthirster for Brambleback, the carry, was still to give and the fight that ended the game
            # was played without it)
            return False
        if self.phase.planning and rp is not None and rp.all_in(scr.hp if scr.hp is not None else getattr(rp, 'last_hp', None)):
            return False        # the last life's roll-down first (game 73 4-6: the anvil's choice screen took the rolls' time)
        for slot, o in sorted(self.bench_objs.items()):
            if o['kind'] != 'interactable' or (slot, scr.stage) in self._auto_tried:
                continue
            if not coach.reserve_model():
                return False                    # the model is busy: look again on the next read
            self._auto_tried.add((slot, scr.stage))
            print(f"{scr.stage}: {o['cls']} on bench slot {slot}: the model is free - opening it")
            self.model_reserved = time.time()
            try:
                ok = self.open_container(slot, scr)
            except NotForeground:
                ok = False
                self._auto_tried.discard((slot, scr.stage))
            if not ok:
                self.model_reserved = None
                coach.release_model()
            return ok
        return False

    def bench_perception(self, frame, bench_seen, scr, planning=True, containers_only=False):
        """Every bench slot's state (bench.classify: unit / empty / container / unknown / obscured).
        Non-unit objects must read the same twice before they count; a new container is an event
        for the model. Slots holding something that is not a unit are blocked for moves and buys
        (roster.blocked) - containers always; an unrecognised object only when seen in planning on
        our own board (in combat on an opponent's arena empty tiles look different)."""
        from tft_player import bench
        try:
            seen = bench.classify(frame, bench_seen)
        except Exception as e:
            self.log(ev='bench_error', error=repr(e))
            return
        probed = self.__dict__.setdefault('_probed_containers', {})      # slot -> class (probe_bench_object)
        for i in list(probed):
            if seen[i]['kind'] == 'unknown' and (seen[i].get('conf') or 0) >= self.PROBE_MIN:
                seen[i] = {**seen[i], 'kind': 'interactable', 'cls': probed[i]}
            elif seen[i]['kind'] in ('empty', 'unit'):
                del probed[i]                    # opened, or gone
        self.bench_kinds = seen
        # an 'unknown' slot where we track a unit is that unit (its bar was missed), not an object;
        # in combat, unknowns are the opponent arena's tiles: only containers are objects then
        # ... and only when this read shows our own bench (one slot matching our empty tile):
        # run29 3-6 reported 4 "objects" while the camera was on an opponent's arena
        own = any(o['kind'] == 'empty' for o in seen)
        now = {i: o for i, o in enumerate(seen)
               if o['kind'] == 'interactable'
               or (o['kind'] == 'unknown' and planning and own and not containers_only
                   and self.roster.at(('bench', i)) is None)}
        prev = self._bench_prev
        # a clear container match counts at once (so the round's first plan already has it);
        # anything weaker must read the same twice
        # an unrecognised object must read the same 3 times in a row (a unit passing, a hover)
        runs = getattr(self, '_bench_runs', {})
        runs = {i: (runs.get(i, 0) + 1 if prev.get(i, {}).get('kind') == o['kind'] and prev.get(i, {}).get('cls') == o['cls']
                    else 1) for i, o in now.items()}
        self._bench_runs = runs
        stable = {i: o for i, o in now.items()
                  if (o['kind'] == 'interactable' and (o['conf'] >= 0.9 or runs[i] >= 2))
                  or (o['kind'] == 'unknown' and runs[i] >= 3)}
        self._bench_prev = now
        for i, o in stable.items():
            old = self.bench_objs.get(i)
            if old and old['kind'] == o['kind'] and old['cls'] == o['cls']:
                continue
            self.log(ev='bench_object', stage=scr.stage, slot=i, kind=o['kind'], cls=o['cls'], conf=o['conf'])
            if o['kind'] == 'interactable':
                print(f"{scr.stage}: bench slot {i}: {o['cls']} ({o['conf']:.2f})"
                      + ('' if self.auto_open else ' -> telling the model'))
                if not self.auto_open:       # the model reports it and decides when to open
                    self.pending_events.append(f"EVENT CHOICE_CONTAINER_AVAILABLE: {o['cls']} on bench slot {i} "
                                               f"(execution_plan.open_container).")
            else:
                print(f"{scr.stage}: bench slot {i}: unrecognised object ({o['conf']:.2f})")
        keep_other = lambda o: containers_only and o['kind'] != 'interactable'   # a containers-only read keeps them
        for i in [i for i in self.bench_objs if i not in now and seen[i]['kind'] != 'obscured'
                  and (planning or self.bench_objs[i]['kind'] == 'interactable') and not keep_other(self.bench_objs[i])]:
            self.log(ev='bench_object_gone', stage=scr.stage, slot=i, was=self.bench_objs[i])
        # a covered slot is no evidence that its object left; in combat an unrecognised object
        # seen in planning is kept (the opponent's arena makes every free tile 'unknown')
        self.bench_objs = {i: o for i, o in self.bench_objs.items()
                           if i in now or seen[i]['kind'] == 'obscured' or keep_other(o)
                           or (not planning and o['kind'] == 'unknown' and seen[i]['kind'] == 'unknown')}
        self.bench_objs.update(stable)
        blocked = {}
        for i, o in self.bench_objs.items():
            if o['kind'] == 'interactable':
                blocked[i] = f"a container ({o['cls']})"
            elif o['kind'] == 'unknown' and self.roster.at(('bench', i)) is None and (planning or i in self.roster.blocked):
                # (a tracked unit there whose bar was missed is that unit, not an object)
                blocked[i] = 'something that is not a unit (unrecognised)'
        if blocked != self.roster.blocked:
            self.log(ev='bench_blocked', stage=scr.stage, slots=blocked)
        self.roster.blocked = blocked

    CHOICE_TITLE = (860, 812, 1060, 846)        # "Choose One" above a row of option cards
    CHOICE_TIMER = (1068, 835, 1105, 863)      # game 65 calls/0013-0.jpg: 26, separate from fight timer 2
    CHOICE_MARGIN = 4.0

    def read_choice(self, frame):
        """An open 'Choose One' row (an item anvil, a reward): [(point to click, option name)], or
        [] when none is up. The names sit at fixed places for 2-5 cards; otherwise the row is
        searched with text detection (slow)."""
        from tft_player.screen import match_name
        h, w = frame.shape[:2]
        s = h / 1080
        S = lambda b: (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        title, = self.fresh_texts(frame, [S(self.CHOICE_TITLE)])
        if 'choose' not in (title or '').lower():
            return []
        from tft_player.trait_text import choice_names
        names = list(self.reader.item_names()) + choice_names()     # items, and trait choices (Kha'Zix, Primal)
        # Kha'Zix's cards say "Evolve Ravager": the trait is the name ("evolveravager" was 0.70 from "ravager")
        name_of = lambda t: match_name(re.sub(r'^\W*evolve\s+', '', t or '', flags=re.I), names, 0.75)
        bottom = {}
        for n in (5, 4, 3, 2):
            centres = [962 + (k - (n - 1) / 2) * 265 for k in range(n)]
            txt = self.fresh_texts(frame, [S((c - 125, 955, c + 125, 995)) for c in centres])
            bottom.update(zip(centres, txt))
            got = [name_of(t) for t in txt]
            if all(got):
                return [((c, 915), g) for c, g in zip(centres, got)]
        # the purple Evolve panel (Kha'Zix) has taller cards: the names sit lower, at 995-1035 (run 20261029-062018
        # 3-3: the boxes above read nothing, the text detection cut the names at 1010 and found 2 of the 4 cards; the
        # model's "option 2, Evolve Ravager" was out of range and the screen ran out the planning phase)
        for n in (5, 4, 3, 2):
            centres = [962 + (k - (n - 1) / 2) * 265 for k in range(n)]
            got = [name_of(t) for t in self.fresh_texts(frame, [S((c - 125, 995, c + 125, 1035)) for c in centres])]
            if all(got):
                return [((c, 915), g) for c, g in zip(centres, got)]
        # a name on two lines (Coven's "Convert 40 Essence / to loot", "Continue gathering / Essence"): the line above
        # joined with the one read - one OCR batch (0.1 s) before the text detection below, which took 10-12 s a screen
        # in a game (run 20261022-025740 3-6 and 3-7: the planning phase's board moves never ran)
        cs = sorted(bottom)
        top = dict(zip(cs, self.fresh_texts(frame, [S((c - 125, 925, c + 125, 962)) for c in cs])))
        for n in (5, 4, 3, 2):
            centres = [962 + (k - (n - 1) / 2) * 265 for k in range(n)]
            got = [name_of(f'{top[c] or ""} {bottom[c] or ""}'.strip()) for c in centres]
            if all(got):
                return [((c, 915), g) for c, g in zip(centres, got)]
        # otherwise text detection over the cards: every card's lines, grouped by place, matched as a whole first
        # (then its biggest line). A card whose text matches no name is still a card, named None: the count must be
        # right (run 20261013-220557 5-5: Draven's 3 bounty cards were read as 2 - "10,000 damage." as Damage Amp,
        # "components" as 1 component - and the model's pick of the third card was refused as out of range)
        x0, y0 = round(w / 2 + (250 - 960) * s), round(900 * s)
        crop = frame[y0:round(1040 * s), x0:round(w / 2 + (1670 - 960) * s)]
        cards = []
        for cx, cy, ht, t in sorted(self.reader._detect(crop, 'choice'), key=lambda r: r[0]):
            if not t.strip():
                continue
            x = (x0 + cx - w / 2) / s + 960
            card = next((c for c in cards if abs(x - sum(c['x']) / len(c['x'])) < 100), None)
            if card is None:
                card = {'x': [], 'lines': []}
                cards.append(card)
            card['x'].append(x)
            card['lines'].append((cy, ht, t))
        out = []
        for c in cards:
            if not 330 <= sum(c['x']) / len(c['x']) <= 1595:
                continue                          # outside any card (5 cards: centres 432-1492): board text
            whole = ' '.join(t for _, _, t in sorted(c['lines']))
            biggest = max(c['lines'], key=lambda l: l[1])[2]
            out.append(((sum(c['x']) / len(c['x']), 915), name_of(whole) or name_of(biggest)))
        return sorted(out, key=lambda o: o[0][0])

    def choice_up(self, frame):
        """A 'Choose One' row is over the shop (its title read now, one OCR box)."""
        h, w = frame.shape[:2]
        s, b = h / 1080, self.CHOICE_TITLE
        box = (round(w / 2 + (b[0] - 960) * s), round(b[1] * s), round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
        try:
            title, = self.fresh_texts(frame, [box])
        except Exception:
            return False
        return 'choose' in (title or '').lower()

    def fresh_texts(self, frame, boxes):
        """OCR of these boxes now, never the reader's cache: its 'crop barely changed' test kept "Choose One" and
        the four Blessing names after the screen closed, when a dark shop tooltip took the dark banner's place (run
        20261015-132144 4-3: a phantom second Primal screen, the model asked twice, moves blocked for 6 s)."""
        for b in boxes:
            self.reader._cache.pop(b, None)
        return self.reader._texts(frame, boxes, None)

    def pilot_watch_choice(self, frame, scr):
        """Pilot mode: a 'Choose One' row is up - take the option the comp wants most (its item
        list, then components by what they complete). True while it is up."""
        c = self.choice
        if time.time() - getattr(self, '_pchoice_at', 0) < 0.4:
            return bool(c)
        self._pchoice_at = time.time()
        opts = self.read_choice(frame)
        if not opts:
            if c and time.time() - c['seen'] > 2.0:
                self.log(ev='choice_closed', stage=self.phase.stage, picked=c.get('clicked'))
                self.choice = None
            return False
        if c is None:
            c = self.choice = {'seen': time.time(), 'opts': [n for _, n in opts]}
            self.log(ev='choice_open', stage=self.phase.stage, source='pilot', options=c['opts'])
        c['seen'] = time.time()
        if not c.get('clicked'):
            vals = [self.pilot.choice_value(n, self.roster, self.bench_items) if n else -1e9 for _, n in opts]
            k = max(range(len(opts)), key=lambda i: vals[i])
            try:
                self.inp.click(opts[k][0])
            except NotForeground:
                return True
            c['clicked'] = opts[k][1]
            self.note_evolution(c['opts'], opts[k][1])
            self.log(ev='choice_pick', stage=self.phase.stage, option=k, name=opts[k][1], options=c['opts'])
            print(f'{self.phase.stage}: choice - {opts[k][1]} (of {", ".join(c["opts"])})')
            self.status_extra['choice'] = f'took {opts[k][1]} of {", ".join(c["opts"])}'
            time.sleep(0.5)
        return True

    def watch_choice(self, frame, scr):
        """EVENT CHOICE_SCREEN_OPEN: a row of option cards replaced the shop. The model sees the
        options and the state and answers with an option index; that card is clicked.
        True while a choice screen is up (nothing else is done then)."""
        if self.pilot:
            return self.pilot_watch_choice(frame, scr)
        from tft_player import bench
        if not self.choice and time.time() - getattr(self, '_choice_at', 0) < 0.3:
            return False
        self._choice_at = time.time()
        cards = bench.choice_cards(frame, fallback=not any(scr.shop or ()))
        # the "Choose One" title decides that a choice is up, and the option names come off the cards
        # (run 185805: shop art read as 2 cards five times - option 0 clicked on the shop; a 5-card
        # anvil read as 6 - "option 4 Giant Slayer" clicked Blue Buff; a component anvil's cards unseen)
        h, w = frame.shape[:2]
        s = h / 1080
        box = self.CHOICE_TITLE
        tbox = (round(w / 2 + (box[0] - 960) * s), round(box[1] * s), round(w / 2 + (box[2] - 960) * s), round(box[3] * s))
        # read fresh only while a choice is open (to see it close): a fresh OCR every 0.3 s for nothing cost 9 % of the
        # loop; an opening is seen by the cached read, which follows any real change of the crop
        title, = self.fresh_texts(frame, [tbox]) if self.choice else self.reader._texts(frame, [tbox], None)
        detected_at = time.time()
        remaining = None
        if 'choose' in (title or '').lower():
            b = Player.CHOICE_TIMER
            timer_box = (round(w / 2 + (b[0] - 960) * s), round(b[1] * s),
                         round(w / 2 + (b[2] - 960) * s), round(b[3] * s))
            raw, = self.fresh_texts(frame, [timer_box])
            if re.fullmatch(r'\d{1,2}', raw.strip()):
                remaining = int(raw.strip())
        named = []
        if 'choose' not in (title or '').lower():
            cards = []
        else:
            named = (self.choice or {}).get('named') or self.read_choice(frame)     # [(1080p point, option name)]
            if named and len(named) != len(cards):
                cards = [p for p, _ in named]            # the names read off the cards beat the colour scan
        c = self.choice
        if not cards:
            # still waiting for the model's answer: the screen stays "open" (run 152203 4-3: the cards were
            # not read for 2 s, the record was dropped, and the model's answer - Infinity Edge, shown in
            # the panel - found no choice to click; the game picked a Void Staff on timeout)
            waiting = c and c['pick'] is None and not c.get('clicked') and time.time() - c['asked'] < 25
            if c and time.time() - c['seen'] > 2.0 and not waiting:
                self.log(ev='choice_closed', stage=self.phase.stage, picked=c.get('clicked'))
                self.choice = None                 # the screen is gone
            return False
        if c is not None and c.get('clicked') and time.time() - c.get('clicked_at', time.time()) > 2.5:
            # cards still up 2.5 s after our click: a NEW choice screen (another container, a reward). Its cards are
            # read afresh: the old record's names and places are another row (release run 20261016-141957 3-6: the
            # second Primal screen has 3 cards, the first had 4 - Bear was clicked at the 4-card place, in the gap)
            self.log(ev='choice_closed', stage=self.phase.stage, picked=c.get('clicked'), how='replaced by a new one')
            c = self.choice = None
            named = self.read_choice(frame)
            cards = bench.choice_cards(frame, fallback=not any(scr.shop or ()))
            if named and len(named) != len(cards):
                cards = [p for p, _ in named]
            if not cards:
                return True
        lp = getattr(self, '_late_pick', None)
        if c is None and lp and time.time() - lp[0] < 20:
            # an answer that came after its record was dropped, and the cards are still up: use it
            self._late_pick = None
            c = self.choice = {'asked': lp[0], 'pick': (lp[1], lp[2]), 'cards': len(cards), 'seen': time.time()}
        if c is None:
            c = self.choice = {'asked': time.time(), 'pick': None, 'cards': len(cards), 'seen': time.time(),
                               'named': named, 'deadline': detected_at + (remaining if remaining is not None else 25)}
            names = [n for _, n in named] if len(named) == len(cards) and all(n for _, n in named) else None
            planned = self.planned_choice(names)
            if planned is not None:
                # the model planned this choice in its briefing (the Primal Blessing): taken at once, no model call -
                # it is told (the user, 2026-10-15: 9 s for a planned pick; it should take under 1 s)
                c['pick'], c['planned'] = planned, True
                self.log(ev='choice_planned', stage=self.phase.stage, option=planned[0], name=planned[1], of=names)
                how = ('first of the artifacts you listed with your augment' if getattr(self, '_wish_used', False)
                       else 'as you planned in your briefing')
                self._wish_used = False
                self.news_event('choice_taken', self.phase.stage, f'choice screen ({", ".join(names)}): took '
                                f'{planned[1]}, {how}')
            else:
                self.ask_choice(scr, len(cards), names)
        c['seen'] = time.time()
        if remaining is not None:
            c['deadline'] = min(c.get('deadline', detected_at + remaining), detected_at + remaining)
        if c['pick'] is None and not c.get('clicked') and c.get('deadline') and c['deadline'] - time.time() < 3.0:
            # the model's answer has not come with under 3 s left (the owner, game 72, x24.8: "send the model the
            # choices, if not enough time, pick the one needed"): the option the item plan wants first
            from tft_player import item_plan
            names = [n for _, n in c.get('named') or []]
            want = [w for w, _ in item_plan.wanted(self.roster, self.bench_items,
                                                     getattr(planner_fixed_comp(), 'name', None))]
            k = next((names.index(w) for w in want if w in names), None)
            if k is not None:
                c['pick'], c['planned'] = (k, names[k]), True
                self.log(ev='choice_item_plan_pick', stage=self.phase.stage, option=k, name=names[k], of=names)
                self.news_event('choice_taken', self.phase.stage, f'choice screen ({", ".join(names)}): no answer in '
                                f'time - took {names[k]}, what the item plan needs next')
        if c['pick'] is not None and not c.get('clicked'):
            i, name = c['pick']
            from tft_player.screen import match_name
            bare = re.sub(r'^\W*evolve\s+', '', str(name or ''), flags=re.I)      # "Evolve Ravager" -> Ravager
            by_name = [k for k, (_, n) in enumerate(c.get('named') or [])
                       if name and n and match_name(bare, [n], 0.75)] if len(c.get('named') or []) == len(cards) else []
            if len(by_name) == 1 and by_name[0] != i:
                self.log(ev='choice_by_name', option=i, name=name, card=by_name[0])
                i = by_name[0]                       # the card that carries the name the model chose
            if 0 <= i < len(cards):
                self.inp.click(cards[i])
                # off the shop at once: a cursor left on a shop card opens its tooltip there, over the cards' place
                self.inp.hover((960, 330), steps=2, duration=0.03)
                c['clicked'], c['clicked_at'] = name or i, time.time()
                card_names = [n for _, n in c.get('named') or []]
                if len(card_names) == len(cards):
                    self.note_evolution(card_names, card_names[i])
                self.log(ev='choice_pick', stage=self.phase.stage, option=i, name=name, of=len(cards))
                print(f'{self.phase.stage}: choice - option {i} of {len(cards)}: {name}')
                time.sleep(0.5)
            else:
                self.log(ev='choice_bad_index', option=i, of=len(cards))
                c['pick'] = None
                if not c.get('reasked'):
                    # the cards were miscounted (run 20261029-062018 3-3: 2 read, 4 on screen, "option 2" refused and
                    # nothing asked again - the screen held the shop past the fight). Read the row afresh and ask
                    # once more with that count; the game's own timer is the last resort
                    c['reasked'] = True
                    c['named'] = self.read_choice(frame)
                    n = len(c['named']) or len(cards)
                    names = [x for _, x in c['named']] if c['named'] and all(x for _, x in c['named']) else None
                    c['asked'], c['cards'] = time.time(), n
                    self.news_event('choice_reasked', self.phase.stage,
                                    f'choice screen: your option {i} is not among the {len(cards)} cards read; '
                                    f'the row was read again ({n} cards)')
                    self.ask_choice(scr, n, names)
        return True

    def add_pending_give(self, name, uid):
        """The model took an item on a choice screen "to" a unit: it is given to that unit as soon as it
        is on the item bench (run 221257 4-3: a Radiant item meant for Nidalee sat on the bench for rounds)."""
        from tft_player.screen import ScreenReader, match_name
        item = match_name(name, ScreenReader.item_names(), 0.8)
        if not item:
            self.news_event('give_not_planned', self.phase.stage,
                            f'"{name}" is not an item name in the game data: give it in a plan once it is on the item bench')
            return
        self.pending_gives = getattr(self, 'pending_gives', []) + [{'item': item, 'id': uid, 'plans': 0, 'at': time.time()}]
        self.log(ev='pending_give', stage=self.phase.stage, item=item, unit=uid)

    def queued_gives(self):
        """The choice screens' gives still waiting for their item ({'component', 'id'}), for the plan checks (e38)."""
        return [{'component': g['item'], 'id': g['id']} for g in getattr(self, 'pending_gives', None) or []
                if g.get('plans', 0) < 3 and time.time() - g.get('at', 0) <= 240]

    def inject_gives(self, rp):
        """Each pending give goes into the running round plan's items once (into at most 3 plans, within
        4 minutes); it is dropped when the unit holds the item."""
        base = lambda n: re.sub(r'\s*\(.*$', '', n or '')
        keep = []
        for g in getattr(self, 'pending_gives', []):
            u = self.roster.resolve(g['id'])
            if u is None or any(base(i) == g['item'] for i in u.items) or g['plans'] >= 3 \
                    or time.time() - g['at'] > 240:
                continue
            keep.append(g)
            done = getattr(rp, 'gives_injected', set())
            if (g['item'], g['id']) not in done:
                rp.gives_injected = done | {(g['item'], g['id'])}
                g['plans'] += 1
                items = rp.p.setdefault('items', [])
                k = max([e.get('k', -1) for e in items] + [-1]) + 1
                items.append({'component': g['item'], 'id': u.id, 'k': k})
                self.log(ev='give_injected', stage=self.phase.stage, item=g['item'], unit=u.id)
        self.pending_gives = keep

    def planned_choice(self, names):
        """(option index, name) of the first planned trait choice (briefing trait_choices, e.g. Primal: Tiger, then
        Bear) on this screen that was not taken yet, or None."""
        if not names:
            return None
        wish = getattr(self, 'artifact_wish', None)
        if wish:
            # the artifacts the model listed with its artifact augment (the owner, x16.1): the best one offered
            from tft_player.screen import match_name
            for want in wish:
                hit = match_name(want, names, 0.8)
                if hit:
                    self.artifact_wish, self._wish_used = None, True
                    return names.index(hit), hit
        seen = self.__dict__.setdefault('choice_sets', [])
        if sorted(names) in seen:
            # the same options again: a decision that recurs (Coven's convert / continue), whose answer depends on
            # the state then - the model is asked each time (the climb's checker, games 22-26: the briefing's order
            # converted at 85 against a 250 plan, continued at 48 HP on a loss streak)
            return None
        seen.append(sorted(names))
        taken = self.__dict__.setdefault('trait_taken', [])
        for trait, want in (getattr(self, 'trait_choices', None) or {}).items():
            for name in (want if isinstance(want, list) else [want]):
                if name in names and (trait, name) not in taken:
                    taken.append((trait, name))
                    return names.index(name), name
        return None

    def trait_choice_text(self):
        """The trait choices (e.g. Primal Blessings) the model planned in its briefing, as text."""
        tc = getattr(self, 'trait_choices', None) or {}
        return '; '.join(f"{t}: {', then '.join(map(str, v)) if isinstance(v, list) else v}" for t, v in tc.items())

    def trait_option_text(self, names):
        """Trait options on a choice screen (Kha'Zix's evolution): each trait's effect, from the game data."""
        try:
            from tft_player.trait_text import trait_line
            lines = [trait_line(x, 1) for x in names or [] if x in CATALOG.traits]
            return ('Trait effects (game data): ' + ' | '.join(l for l in lines if l) + '\n') if any(lines) else ''
        except Exception:
            return ''

    def choice_price_text(self, frame, named):
        """Curio Cart uses the choice row but charges gold: read prices, never infer them from item names.

        Run 20261103-120119 call 0044: Tear/Evenshroud/Vanguard Emblem/Reforger cost 6/12/15/0;
        the question named all four without their prices. Patch 18.6 raises emblems to 17.
        Ordinary anvils have no price digits here, so they add no price line.
        """
        if frame is None or not named:
            return ''
        h, w = frame.shape[:2]
        s = h / 1080
        def box(x0, y0, x1, y1):
            return (round(w / 2 + (x0 - 960) * s), round(y0 * s),
                    round(w / 2 + (x1 - 960) * s), round(y1 * s))
        boxes = [box(p[0] + 85, 1015, p[0] + 123, 1050) for p, _ in named]
        # The balance is on the choice panel, not the hidden ordinary shop HUD.
        texts = self.fresh_texts(frame, boxes + [box(1500, 812, 1578, 842)])
        number = lambda text: int(text.strip()) if re.fullmatch(r'\s*\d{1,3}\s*', text or '') else None
        costs, gold = [number(t) for t in texts[:-1]], number(texts[-1])
        if not any(c is not None for c in costs):
            return ''
        prices = ', '.join(f'{i} {name or "unread item"}: {str(cost) + "g" if cost is not None else "price unread"}'
                           for i, ((_, name), cost) in enumerate(zip(named, costs)))
        balance = f'{gold}g' if gold is not None else 'unread'
        self.log(ev='choice_prices', stage=self.phase.stage, names=[n for _, n in named],
                 costs=costs, gold=gold)
        return f'Card prices (read on screen): {prices}. Current balance: {balance}.\n'

    def ask_choice(self, scr, n, names=None):
        from tft_player import planner
        from tft_player.coach import ask_model
        question = self.choice                     # bind callbacks to this exact screen
        co = self.container_opened
        lu = getattr(self, 'last_used', None)
        if co and time.time() - co[2] < 20:
            source = f'{co[0]} (you opened it on bench slot {co[1]})'
        elif lu and time.time() - lu[2] < 30:
            source = f'{lu[0]} (you used it on {lu[1].name or "?"}, unit {lu[1].id})'
        else:
            source = "not a container we opened (for example an augment's reward)"
        planned = self.trait_choice_text()
        # A short question of its own, carrying every read option and the current board/items.
        board = ', '.join(f"{u.name or '?'}{'*' * u.star}" + (f"[{', '.join(u.items)}]" if u.items else '')
                          for u in self.roster.board()) or 'none'
        held = ', '.join(i for i in (self.bench_items or []) if i and i != 'unknown') or 'none'
        prompt = (f'CHOICE SCREEN, stage {self.phase.stage}. Source: {source}.\n'
                  f'There are {n} options in a row where the shop was, numbered 0 (left) to {n - 1} '
                  f'(right). The game has a timer on it.\n'
                  + (f"The options, as read off the cards: {', '.join(f'{k} {x}' for k, x in enumerate(names))}.\n"
                     if names else 'Option names and effects have not been read.\n')
                  + self.choice_price_text(self.last_frame, (self.choice or {}).get('named'))
                  + self.trait_option_text(names)
                  + f'Board: {board}. Item bench: {held}.\n'
                  + (f'Your planned trait choices (from your briefing): {planned}.\n' if planned else '')
                  + 'Reply with two JSON objects and nothing else: first {"option": <index>, "name": "<supplied option '
                  'name, or unknown>"} - for an item, add "to": <unit id> and the controller gives it to that unit as soon '
                  'as it is on the item bench - then {"why": "<1-3 sentences>"}.')
        image, t0 = self.screen_jpeg(), time.time()
        self.log(ev='choice_open', stage=self.phase.stage, source=source, options=n)
        print(f'{self.phase.stage}: choice screen ({n} options, source: {source}) -> asking the model')

        def pick(obj):
            if self.choice is not question or question.get('clicked') or question.get('fallback'):
                self.log(ev='choice_answer_expired', stage=scr.stage)
                return
            if isinstance(obj, dict) and isinstance(obj.get('option'), int):
                to = obj.get('to')
                if isinstance(to, int) and not isinstance(to, bool) and isinstance(obj.get('name'), str):
                    self.add_pending_give(obj['name'], to)
                if self.choice is not None:
                    self.choice['pick'] = (obj['option'], obj.get('name'))
                else:           # the record was dropped meanwhile: keep the answer for the screen's next read
                    self._late_pick = (time.time(), obj['option'], obj.get('name'))
                    self.log(ev='choice_late_pick', stage=self.phase.stage, option=obj['option'], name=obj.get('name'))

        def run():
            try:
                text = ask_model(prompt, image, on_first=pick, timeout=45, priority=True)
            except Exception as e:
                text = f'FAILED: {e}'
            finally:
                if getattr(self, 'model_reserved', None):          # the model held for this question
                    from tft_player import coach
                    self.model_reserved = None
                    coach.release_model()
            self.model_call('choice', prompt, text, time.time() - t0)
            print(f"[choice {time.time() - t0:.0f}s] {str(text)[:160]}")
        threading.Thread(target=run, daemon=True).start()

    # -- the roster against the screen (pilot mode) ---------------------------------------------
    SURVEYS_PER_ROUND = 2

    def check_roster(self, frame, scr):
        """Is the tracked board what the game shows? The trait panel counts the fielded units'
        traits; a count that differs twice in a row (units still moving read once wrong), a
        bench-slot mismatch after a buy, or the first planning phase at level 8 (the roll-down
        needs exact copies) -> survey: every unit's name read off its info panel. True when
        the roster was rebuilt (the caller skips this tick)."""
        if REPLAY or (scr.timer is not None and scr.timer < 8):
            self.checked_stage = scr.stage
            return False                        # a recording does not answer clicks
        if getattr(getattr(self, 'phase', None), 'rise', None) is not None:
            return False                        # the countdown just rose: maybe the fight began (see idle_survey)
        done = self.surveys.get(scr.stage, 0)
        if done >= self.SURVEYS_PER_ROUND:
            self.checked_stage = scr.stage
            return False
        reason = None
        eight_at = self.pilot.eight_at if self.pilot else (4, 2)
        unknown = [u for u in self.roster.units if u.name is None and self.tries.get(u.id, 0) >= 2]
        if (scr.level or 0) >= 8 and not self.surveyed_at8 and (scr.round or (0, 0)) >= eight_at:
            self.surveyed_at8 = True
            reason = 'level-8 roll-down: exact copies'
        elif self.need_survey:
            reason = self.need_survey
        elif unknown:
            reason = f'{len(unknown)} unit(s) still unnamed after two reads'      # never act on a "?"
        elif time.time() - getattr(self, '_trait_at', 0) > (0.5 if self.checked_stage != scr.stage else 1.5):
            self._trait_at = time.time()
            seen = self.reader.trait_panel(frame)
            self.learn_evolution(seen)
            mine = self.tracked_traits()
            bad = self.trait_diff(seen, mine)
            self._trait_bad = (self._trait_bad + 1) if bad and seen else 0
            if not bad or not seen:
                self.checked_stage = scr.stage      # the board matches: act on it
            if bad and tuple(bad) == self._trait_left:
                self._trait_bad = 0             # what a survey could not explain (an emblem's trait): not again
                self.checked_stage = scr.stage
            if self._trait_bad >= 2:
                reason = 'trait panel != tracked board: ' + ', '.join(
                    f'{t} screen {seen.get(t, 0)} tracked {mine.get(t, 0)}' for t in bad[:4])
        if not reason:
            return False
        self.need_survey = None
        self.surveys[scr.stage] = done + 1
        if not self.targeted_check(scr, reason):
            self.survey(scr, reason)
        self.checked_stage = scr.stage
        # a trait difference that survives the survey is not a tracking error: remember it
        got = self.cap.read()
        self._trait_left = tuple(self.trait_diff(self.reader.trait_panel(got[0]), self.tracked_traits())) if got else ()
        return True

    def board_check(self, frame, scr, places):
        """Model player: our health bars against where the roster says our units stand. Not every round -
        only when something is wrong: a step failed in a way that points at a wrong record (input_failed
        sets need_survey), or the hexes on screen and in the records differ for a lasting moment (a second
        Varus the game fielded, a unit never moved where we thought). Then every unit is read
        off its info panel and the roster takes the places the screen shows (survey) - a few seconds
        that save the game (run 152203: Nidalee stood at (0,2) while the roster said (1,0), eight item
        drops hit empty ground and two fights were played without the carry)."""
        if REPLAY or scr.timer is None or scr.stage is None:
            return False
        seen = {l[1:] for _, l in places if l[0] == 'board'}
        tracked = {u.loc[1:] for u in self.roster.board() if not u.missing}
        # a hex where a board check found our unit with no health bar seen: its bar is hidden there, the record is
        # right - no new check for it while that unit stands there (game 50, run 20261030-184045: Murkwolf at (3, 4),
        # the same check at 6-1 (10.9 s) and 6-2 (9.3 s), Murkwolf found both times, a Wisp's buy waiting behind it)
        blind = getattr(self, 'bar_blind', {})
        hidden = {u.loc[1:] for u in self.roster.board() if blind.get(u.loc[1:]) == u.id}
        hidden |= self.hexes_under_tactician(frame, tracked - seen)
        seen, tracked = seen - hidden, tracked - hidden
        if seen == tracked:
            self._board_bad_since = None
        elif time.time() - self.round_started > 4.0:         # units have walked back to their hexes
            since = getattr(self, '_board_bad_since', None) or time.time()
            self._board_bad_since = since
            if time.time() - since > 1.5 and not self.need_survey:        # a lasting difference, not one bad read
                self.need_survey = (f'the board differs from the records: on screen only {sorted(seen - tracked)}, '
                                    f'tracked only {sorted(tracked - seen)}')
                self.survey_places = (self.need_survey, [('board', *h) for h in sorted(seen ^ tracked)])
            if time.time() - since > 1.5 and self.phase.planning:
                self.find_rider(scr, tracked - seen, seen - tracked)
            if time.time() - since > 3.0 and self.phase.planning:
                # an unverified or unnamed record on a hex with no health bar for 3 s is no unit: set it aside now,
                # not at an idle survey that a busy round never reaches (run 185805: a phantom "?" at (0,0) from 4-2
                # to 5-3 - three moves failed on it and it held Aphelios's corner)
                # (a BFF is gone whenever Sprykin drops below 3: it comes back by a read of its panel)
                # a named unit too when the trait panel agrees it is not fielded (trait_gone): two witnesses
                gone = getattr(self, 'trait_gone', set())
                for u in [u for u in self.roster.board() if u.loc[1:] in tracked - seen
                          and (u.unverified or not u.name or u.slotless or u.id in gone)]:
                    u.missing = 'no unit is seen at its hex (no health bar for 3 s)'
                    self.news_event('unit_missing', scr.stage, f'{u.name or "?"} (unit {u.id}) at hex {u.loc[1:]}: no unit '
                                                               f'is seen there - set aside (kept, not deleted)')
                    Player.record_roster_gap(self, frame, scr, f'missing:{u.id}',
                                             [any(l == ('bench', i) for _, l in places) for i in range(9)],
                                             {l[1:] for _, l in places if l[0] == 'board'})
        return False                                          # the check itself waits for idle time (idle_survey)

    def record_roster_gap(self, frame, scr, cause, bench_seen, board_seen):
        """Retain the exact missing-unit observation, once per cause/round, without another read.

        Game 52's Gnar was marked missing between the start/fight screenshots; neither
        frame can explain the correction that then displaced Sentinel. Keep the pixels,
        screen count and ledger together so the next recurrence can be checked offline.
        """
        if frame is None or getattr(self, 'log_dir', None) is None:
            return
        if getattr(self, '_roster_gap_stage', None) != scr.stage:
            self._roster_gap_stage, self._roster_gap_seen = scr.stage, set()
        if cause in self._roster_gap_seen:
            return
        self._roster_gap_seen.add(cause)
        shot = f'{scr.stage}-roster-gap-{len(self._roster_gap_seen):02d}'
        # snap(full=True) prefers cap.raw, which may already be a newer frame than
        # this observation. Save the supplied frame so the pixels and ledger agree.
        if shots_on():
            from PIL import Image
            Image.fromarray(frame).save(self.log_dir / f'{shot}.full.jpg', quality=92)
        self.log(ev='roster_gap', stage=scr.stage, timer=scr.timer, cause=cause, shot=shot,
                 board_count=getattr(scr, 'board_count', None), level=getattr(scr, 'level', None),
                 bench_seen=[i for i, b in enumerate(bench_seen) if b],
                 board_seen=[list(h) for h in sorted(board_seen)],
                 units=[dict(id=u.id, name=u.name, star=u.star, loc=list(u.loc),
                             missing=u.missing, unverified=u.unverified, items=list(u.items or []))
                        for u in self.roster.units],
                 blocked=dict(self.roster.blocked), slots_used=self.roster.slots_used())

    def drop_stale_board_check(self):
        """A new phase: a board check asked for by a board difference of the last phase is dropped - the units have
        moved since, and board_check asks again if the difference is still there (game 50 6-2: 6-1's request, set
        while our tactician stood over Murkwolf, ran 3 s into 6-2 with the tactician gone - 9.3 s, Murkwolf where he
        was tracked)."""
        if str(getattr(self, 'need_survey', None) or '').startswith('the board differs'):
            self.need_survey = None
            self.survey_places = None
            self._board_bad_since = None

    def hexes_under_tactician(self, frame, hexes):
        """Those of `hexes` whose unit's health bar would be behind our tactician's name and level bar: bars there are
        left out (they can be the tactician's own green), so no bar there is no sign of no unit (game 50 6-1: the
        tactician stood by a golden orb over Murkwolf at (3, 4); the board check ran 10.9 s and found him there)."""
        tac = getattr(getattr(self, 'watcher', None), 'tac', (None, None))[1]
        if not tac or frame is None or not hexes:
            return set()
        h, w = frame.shape[:2]
        s = h / 1080
        out = set()
        for rc in hexes:
            x, y = B.scale_point(B.hex_xy(*rc), w, h)
            # a bar stands 40-150 px (1080p) over its hex centre and is ~80 px wide
            if tac[0] - 45 * s <= x <= tac[2] + 45 * s and tac[1] - 15 * s <= y - 40 * s and y - 150 * s <= tac[3] + 15 * s:
                out.add(rc)
        return out

    def find_rider(self, scr, empty_hexes, new_hexes):
        """The game's own rider pick at combat start ("... Rider was auto assigned!"): the Sprykin it picked leaves
        its hex and rides the BFF on the BFF's hex (run 20261012-071645 2-5; run 20261010-235723 4-5). A fielded
        Sprykin missing from its hex while the BFF's old hex shows a unit no record holds is the rider: its place
        becomes that hex."""
        # also when the BFF was only seen with its rider on it (run 20261012-194821 3-2: Sprykin 3 came during 3-1, the
        # game seated Veigar at combat start, and the unit at (1, 4) that opens no panel was recorded as the BFF while
        # Veigar's own hex (0, 3) stood empty): a recorded BFF plus one Sprykin gone from its hex = that Sprykin rides it
        bff = next((u for u in self.roster.units if u.name == BFF and not u.missing), None)
        hexes = bff.loc if bff is not None else getattr(self, 'bff_hex', None)
        if hexes is None and not getattr(self, 'bff_assigned', False) and len(new_hexes) == 1 \
                and self.sprykin_fielded() >= 3:
            # ... and when the BFF itself was never seen: the game seats a Sprykin on it in the planning phase that
            # Sprykin 3 comes (run 20261012-233635 2-5: Rammus made Sprykin 3, Veigar went onto the BFF between Azir
            # and its own hex, and the bot knew only at 2-6) - one Sprykin gone, one new hex = that Sprykin on the BFF
            hexes = ('board',) + tuple(next(iter(new_hexes)))
        if hexes is None or any(getattr(u, 'riding', None) for u in self.roster.units):
            return
        if bff is None and hexes[1:] not in new_hexes:
            return
        gone = [u for u in self.roster.fielded() if u.loc[1:] in empty_hexes and u.info
                and 'Sprykin' in u.info.traits and not u.unverified]
        if len(gone) == 1:
            old = gone[0].loc[1:]
            if bff is not None:
                self.roster.units.remove(bff)
            self.bff_hex, self.bff_assigned = hexes, True
            gone[0].loc, gone[0].riding = hexes, True
            self.news_event('bff_rider', scr.stage, f'{gone[0]} rides the BFF (the game chose it: it left its hex '
                                                    f'{old} for the BFF at {hexes[1:]})')

    def idle_survey(self, scr):
        """A requested board check, only in idle time (the user, 2026-10-07: only when something is wrong
        AND nothing else is to do; it stops the moment the model's plan comes): no plan steps waiting, a
        planning phase with time left. -> True when it ran (even if a plan cut it short)."""
        if not self.need_survey or scr.timer is None or scr.timer < 9 or REPLAY:
            return False
        if getattr(getattr(self, 'phase', None), 'rise', None) is not None:
            # a countdown rise read once counts as the fight on its second read: until then the "time left" may be the
            # fight's clock (benchmark game 20261018-174944 3-5: a 7 s roll-down ended in combat, the next tick read 30
            # s, a board check ran in the fight, saw no bench bars and set six bench units missing - dropped at 3-7)
            return False
        if time.time() - self.round_started < 2.5:
            return False                                      # units still walking back after the fight
        if self.plan_asked != scr.stage:
            # this round's plan (a queued next_round, or the question) starts first: at 5-3 of run 185805 a
            # 7.5 s survey ran before the queued Amumu-for-Rakan swap, which then started with 3 s left
            return False
        rp = self.round_plan                                  # (called only when the plan had no step to take)
        named = getattr(self, 'survey_places', None)
        continuing = named and named[0] == self.need_survey
        if self.surveys.get(scr.stage, 0) >= 1 and not continuing:
            self.need_survey = None
            return False
        reason = self.need_survey
        self.need_survey = None
        self.surveys[scr.stage] = self.surveys.get(scr.stage, 0) + 1
        plan0 = rp

        def interrupted():
            # a new plan with steps to do arrived: the check gives way
            cur = self.round_plan
            return cur is not None and cur is not plan0 and cur.pending(self.roster)
        if not self.targeted_check(scr, reason):
            self.survey(scr, reason, stop=interrupted)
        self._board_bad_since = None
        return True

    def tracked_traits(self):
        """The traits the game's panel should show for the board we track: each fielded champion counted once,
        plus the trait of an emblem a unit holds (if it lacks that trait), the trait a unit evolved into
        (Kha'Zix) and the extra count the game data gives (Elder Dragon: +2 Riftbeast)."""
        from tft_player.state import board_traits
        return board_traits(self.roster.board())

    def learn_evolution(self, seen):
        """Kha'Zix evolves into one trait of his choice for good (Rival, game data). The choice we click is recorded
        (note_evolution); one the game made without us (its timer) shows on the panel: exactly one of his options
        one higher than the tracked board, everything else equal, with a fielded Kha'Zix that has not evolved.
        Then he evolved into that one. -> the trait learned, or None."""
        from tft_player.trait_text import evolutions
        if not seen:
            return None
        mine = self.tracked_traits()
        diff = {t: seen.get(t, 0) - mine.get(t, 0) for t in set(seen) | set(mine) if seen.get(t, 0) != mine.get(t, 0)}
        if len(seen) >= 9:                                    # 9+ rows: the rest hide behind '4+'
            diff = {t: d for t, d in diff.items() if t in seen}
        if len(diff) != 1:
            return None
        (t, d), = diff.items()
        for name, opts in evolutions().items():
            if d == 1 and t in opts and any(u.name == name and not u.evolved for u in self.roster.board()):
                self.roster.evolve(name, t)
                self.log(ev='evolved', stage=self.phase.stage, unit=name, trait=t, how='trait panel')
                self.news_event('evolved', self.phase.stage, f'{name} evolved into {t} (the trait panel shows it)')
                return t
        return None

    def note_evolution(self, options, chosen):
        """A choice screen whose options are an evolving champion's traits (Kha'Zix: Executioner, Rapidfire,
        Ravager, Spellweaver) and the one clicked: that champion now has it for good."""
        from tft_player.trait_text import evolutions
        opts = {o for o in options or [] if o}
        for name, choices in evolutions().items():
            if chosen in choices and len(opts) >= 2 and opts <= set(choices) \
                    and any(u.name == name for u in self.roster.units):
                self.roster.evolve(name, chosen)
                self.log(ev='evolved', stage=self.phase.stage, unit=name, trait=chosen, how='choice clicked')
                return True
        return False

    def trait_equation(self, seen):
        """The board as an equation, traits as the variables: the game's trait panel (seen) against the traits of
        the board we track. -> (bad traits, hypotheses). A hypothesis is ONE record wrong in a way that balances
        every compared trait: ('replace', unit, champion) - that unit is really that champion; ('add', None,
        champion) - an untracked unit of that champion is fielded; ('remove', unit, None) - that unit is not
        fielded. Hypotheses only decide what is read first; the read decides."""
        mine = self.tracked_traits()
        full = len(seen) < 9                                  # 9+ rows: the rest hide behind '4+'
        keys = set(seen) | ({t for t, c in mine.items() if c} if full else set())
        bad = sorted(t for t in keys if seen.get(t, 0) != mine.get(t, 0))
        if not bad:
            return bad, []
        board = [u for u in self.roster.board() if u.name]
        names = Counter(u.name for u in board)
        from tft_player.trait_text import trait_bonus

        def tr(n, unit=None):
            """A champion's count on the panel: its traits once (and what this unit evolved into), plus the
            game data's extra (Elder Dragon: +2 Riftbeast)."""
            own = set(CATALOG.unit(n).traits) if n and CATALOG.unit(n) else set()
            own |= set(unit.evolved) if unit is not None else set()
            out = Counter(own)
            out.update({t: k for t, k in trait_bonus(n).items() if t in own})
            return out

        def balanced(m):
            return all(seen.get(t, 0) == m.get(t, 0) for t in (set(seen) | ({t for t, c in m.items() if c} if full else set())))
        hyps = []
        for u in board:
            less = Counter(mine)
            if names[u.name] == 1:
                less.subtract(tr(u.name, u))                  # its traits leave only if it is its champion's only copy
            if balanced(less):
                hyps.append(('remove', u, None))
            for c in CATALOG.units:
                if c == u.name:
                    continue
                more = Counter(less)
                if c not in names:
                    more.update(tr(c))
                if balanced(more):
                    hyps.append(('replace', u, c))
        for c in CATALOG.units:
            if c not in names:
                more = Counter(mine)
                more.update(tr(c))
                if balanced(more):
                    hyps.append(('add', None, c))
        return bad, hyps

    def trait_check(self, frame, scr):
        """Confirmation before a model call that uses the board's champions (round plans and updates, a choice that
        gives an item to a unit - the user, 2026-10-08): the game's trait panel against the board we track. Equal:
        the board's identities are confirmed. Different: the units the difference points at are read now (at most
        3; first the ones a single-error hypothesis names) and their records fixed before the question goes out.
        With too little time left (< 6 s, or units still walking back), the model is told and the check runs
        first thing in the next planning phase. A difference that survives the reads (an augment's extra trait,
        a hidden '4+' row) is remembered, not read again. Never in combat (the user: no checks once the
        champions move)."""
        if REPLAY or frame is None or not self.phase.planning:
            return
        seen = self.reader.trait_panel(frame)
        if not seen:
            return
        # the panel must hold still: one read under an opening choice screen gave 2 of 5 traits (run 010416 2-1)
        time.sleep(0.15)
        got = self.cap.read()
        if not got or self.reader.trait_panel(got[0]) != seen:
            return
        Player.log_lineup_diff(self, scr)
        t0 = time.time()
        self.learn_evolution(seen)
        bad, hyps = self.trait_equation(seen)
        # records the panel says are not fielded: with no health bar at their hex either, they are set aside
        # (board_check) - a named unit too (run 20261013-220557 4-5: the sold Murkwolf's record stayed on (2, 3))
        self.trait_gone = {u.id for k, u, _ in hyps if k == 'remove'}
        if not bad:
            self.trait_note, self.trait_pending = None, None
            return
        if (scr.stage, tuple(bad)) == getattr(self, '_trait_left', ()):
            return                                            # read this round and still unexplained: not again this round
        hyp_text = '; '.join(f"{'unit ' + str(u.id) + ' at ' + str(tuple(u.loc[1:])) + ' is ' + c if k == 'replace' else 'an untracked ' + c + ' is fielded' if k == 'add' else 'unit ' + str(u.id) + ' is not fielded'}"
                             for k, u, c in hyps[:3])
        mine = self.tracked_traits()
        diff = ', '.join(f'{t} screen {seen.get(t, 0)} / tracked {mine.get(t, 0)}' for t in bad)
        # the owner (x18.10): the difference is shown on the overlay as soon as it is seen
        self.alert('TRAITS', f'trait panel differs from the records: {diff}'[:90], 'State')
        settled = self.phase.planning and time.time() - self.round_started > 2.5
        if not settled or scr.timer is None or scr.timer < 6:
            # nothing can be read now: the model is told, and the check comes first in the next planning phase
            self.trait_note = (f'Trait panel check: the board differs from the records ({diff})'
                               + (f'; most likely {hyp_text}' if hyp_text else '') + '. Being checked.')
            self.trait_pending = scr.stage
            self.log(ev='trait_check', stage=scr.stage, bad=bad, hyps=hyp_text, read=[], deferred=True)
            return
        # what to read: the units the hypotheses name, then the rest the difference points at
        over = {t for t in bad if mine.get(t, 0) > seen.get(t, 0)}
        under = {t for t in bad if seen.get(t, 0) > mine.get(t, 0)}
        traits = lambda u: set(CATALOG.unit(u.name).traits) if u.name and CATALOG.unit(u.name) else set()
        board = self.roster.board()
        named = [u for k, u, c in hyps if u is not None]
        rest = sorted([u for u in board if not u.name or u.unverified or traits(u) & over or (under and not traits(u) & under)],
                      key=lambda u: (not (u.unverified or not u.name), not (traits(u) & over)))
        suspects, ids = [], set()
        for u in named + rest:                              # in order, each unit once
            if u.id not in ids:
                ids.add(u.id)
                suspects.append(u)
        bars = B.green_bars(frame)
        _, _, places = B.bar_places(frame, exclude=self.watcher.tac[1], bars=bars)
        tracked = {u.loc for u in board}
        untracked = [loc for _, loc in places if loc[0] == 'board' and loc not in tracked]
        if any(k == 'add' for k, _, _ in hyps):
            todo = [('hex', l) for l in untracked[:2]] + [('unit', u) for u in suspects]
        else:
            todo = [('unit', u) for u in suspects] + [('hex', l) for l in untracked]
        todo = todo[:3]
        fixed, reads = [], []
        try:
            k = 0
            while k < len(todo) and k < 6:
                kind, x = todo[k]
                k += 1
                loc = x.loc if kind == 'unit' else x
                name, stars = self.trusted_read(loc, bars)
                reads.append(f'{loc[1:]}: {name or "no read"}{"*" * (stars or 0)}')
                if not name:
                    continue
                u = self.roster.at(loc)
                if u is None:
                    from tft_player.state import Owned
                    back = next((m for m in self.roster.missing() if m.name == name), None)
                    u = back or Owned(name, stars or 1, loc, ours=True)
                    if back is None:
                        self.roster.units.append(u)
                    u.missing, u.loc = None, loc
                    fixed.append(f'{name} found at {loc[1:]}')
                elif u.name != name or (stars and stars != u.star):
                    # a confirmed unit is renamed only by two agreeing witnesses: the trait panel's own hypothesis for
                    # this unit is one (a second read of the same panel almost never counted: 29 of 31 came back
                    # "no read" in 32 games, 10-16..10-19 - run 20261019-224321 1-4 kept the fielded LeBlanc as Ornn)
                    told = any(k == 'replace' and h is u and c == name for k, h, c in hyps)
                    if u.name and not u.unverified and u.name != name and not told:
                        again = self.trusted_read(loc, bars)[0]
                        reads.append(f'{loc[1:]} again: {again or "no read"}')
                        if again != name:
                            continue                          # a confirmed unit is renamed only by two agreeing reads
                    fixed.append(f'unit {u.id} at {loc[1:]} is {name}{"*" * (stars or u.star)}, not '
                                 f'{u.name or "?"}{"*" * u.star}')
                    # the champion read here was recorded on another unit: that record is wrong too - read it next
                    # (run 185805 4-3: Aphelios read where Vi was recorded; the recorded "Aphelios" was Malphite)
                    for o in board:
                        if o is not u and o.name == name and o.loc != loc and o.id not in ids:
                            ids.add(o.id)
                            todo.insert(k, ('unit', o))
                    # a bench record of it on a slot that shows no unit is this one, fielded by the game at combat start
                    # (run 20261019-224321: LeBlanc's record stayed on bench slot 0 and its sell failed 5 times)
                    ghosts = [o for o in self.roster.bench() if o is not u and o.name == name]
                    lights = self.bench_lights(frame) if ghosts else None
                    for o in ghosts:
                        if lights is not None and lights[o.loc[1]] is False:
                            o.missing = f'its champion was read at {loc[1:]}; bench slot {o.loc[1]} shows no unit'
                            fixed.append(f'{name} record on the empty bench slot {o.loc[1]} set aside')
                u.name, u.star = name, stars or u.star
                u.unverified, u.sure = None, True
                if not self.trait_equation(seen)[0]:
                    break                                     # balanced: the board is confirmed
        except NotForeground:
            return
        bad2, _ = self.trait_equation(seen)
        self._trait_left = (scr.stage, tuple(bad2))
        self.trait_pending = None
        self.trait_note = (f'Trait panel check: still differs after reading ({", ".join(bad2)}) - maybe an '
                           f'augment or emblem trait, or a hidden row.') if bad2 else None
        self.log(ev='trait_check', stage=scr.stage, screen=dict(seen), tracked=dict(mine), bad=bad, hyps=hyp_text,
                 read=[repr(x) if kind == 'unit' else list(x) for kind, x in todo], reads=reads, fixed=fixed, left=bad2,
                 seconds=round(time.time() - t0, 2))
        if fixed:
            self.news_event('unit_corrected', scr.stage, 'trait panel check: ' + '; '.join(fixed))
            self.alert('RECORD', 'trait check fixed ' + '; '.join(fixed)[:80], 'State')
        print(f'{scr.stage}: trait check {bad} -> read {len(todo)}' + (f', fixed: {fixed}' if fixed else '')
              + (f', left {bad2}' if bad2 else ''))

    def trusted_read(self, loc, bars):
        """A unit's panel read that counts only when it can be this unit's: the click on its own bar's body; a
        name that a CONFIRMED unit on a neighbouring hex has (not the one recorded here) is a neighbour's
        panel (run 152104 5-5) - tried once more higher on the body, then not trusted. -> (name, stars)."""
        from tft_player.state import _hex_near
        others = [self.unit_point(u.loc, bars) for u in self.roster.units if u.name and u.loc != loc and not u.missing][:2]
        name, stars, _ = self.read_fresh(self.unit_point(loc, bars), others)
        if not name or loc[0] != 'board':
            return name, stars
        here = self.roster.at(loc)
        neigh = {u.name for u in self.roster.board() if u.loc != loc and _hex_near(u.loc[1:], loc[1:])
                 and u.name and not u.unverified}
        if name in neigh and (here is None or here.name != name):
            px, py = self.unit_point(loc, bars)
            name2, stars2, fresh = self.read_unit((px, py - 25))
            if fresh and name2 and name2 not in neigh:
                return name2, stars2
            self.log(ev='read_untrusted', loc=list(loc), read=name, neighbours=sorted(neigh))
            return None, None
        return name, stars

    @staticmethod
    def trait_diff(seen, mine):
        """Traits whose count on the game's panel differs from the tracked board's. With 9+ rows
        shown more may be hidden ('4+'), so then only the rows shown count."""
        if not seen:
            return []
        keys = set(seen) | {t for t, c in mine.items() if c >= min(CATALOG.traits.get(t) or (1,))}
        return sorted(t for t in keys if seen.get(t, 0) != mine.get(t, 0) and (t in seen or len(seen) < 9))

    def log_lineup_diff(self, scr):
        """Compare the current records with the newest requested placement separately from perception."""
        requested = getattr(self, 'newest_lineup', None)
        if not requested or requested[0] != scr.stage:
            return
        different = []
        for entry in requested[1]:
            loc = ('board', *entry['hex'])
            unit = self.roster.resolve(entry['id']) if 'id' in entry else None
            here = self.roster.at(loc)
            if (unit is not None and (unit.missing or unit.loc != loc)) or (
                    unit is None and (here is None or here.name != entry.get('name'))):
                different.append(entry)
        signature = (scr.stage, repr(different))
        if different and signature != getattr(self, '_lineup_diff', None):
            self.log(ev='lineup_diff', stage=scr.stage, requested=different, roster=repr(self.roster))
        self._lineup_diff = signature

    def confirm_before_field(self, unit, src, scr):
        """The user, 2026-10-09: a bench unit is clicked and read before it is dragged onto the board. -> True
        when it reads as another champion: its record is fixed (and the other records of that champion are
        read too - a swap), this unit's remaining plan steps (its hex, its items) are dropped, the model is
        told (an update question), and nothing is dragged. A read that gives no name changes nothing.
        (run 185805 4-3: "Aphelios" from bench slot 4 was Malphite; it was fielded and got Guinsoo's.)"""
        others = [self.unit_point(o.loc) for o in self.roster.units
                  if o.name and o.name != unit.name and o.loc[0] == 'bench' and not o.missing][:2]
        try:
            name, stars, _ = self.read_fresh(self.unit_point(src), others)
        except NotForeground:
            raise
        self.log(ev='confirm_before_field', stage=scr.stage, unit=repr(unit), read=name, stars=stars)
        if not name:
            # the owner (x18.10, game 66 3-6): a click that reads no name never leads to a drag - "Krug" at bench slot 5
            # read nothing and was dragged anyway; it was a Cinderling, and Mama Beak then found the board full. The
            # unit is read again (identify_one) and its step waits for that read
            misses = self.__dict__.setdefault('_field_no_name', {})
            key = (scr.stage, unit.id)
            misses[key] = misses.get(key, 0) + 1
            if misses[key] >= 2:
                unit.field_read_hold = (scr.stage, src)
            if unit.unverified or not getattr(unit, 'sure', False) or misses[key] >= 2:
                # (a unit read before gets one more click first: one panel that did not open is no doubt yet)
                if not unit.unverified:
                    unit.unverified = 'no name read when clicked before fielding'
                unit.sure = False
                self.reread = getattr(self, 'reread', set()) | {unit.id}
                self.tries[unit.id] = 0
            self.deferred = True
            self.fail_cause = f'{unit} gave no name when clicked: read again before it is moved onto the board'
            return True
        if name == unit.name:
            getattr(self, '_field_no_name', {}).pop((scr.stage, unit.id), None)
            unit.field_read_hold = None
            unit.unverified, unit.sure = None, True
            if stars:
                unit.star = stars
            Player.record_carousel_panel(self, unit, name, stars, scr)
            return False
        if getattr(unit, 'sure', False):
            got = self.cap.read()
            if not got or not self.selected_here(got[0], src):
                self.log(ev='read_not_selected', stage=scr.stage, unit=repr(unit), read=name)
                self.deferred = True
                return True
        old_identity = (unit.name, unit.star)
        old = f'{unit.name}{"*" * unit.star}'
        unit.name, unit.unverified, unit.sure = name, None, True
        if stars:
            unit.star = stars
        Player.identity_corrected(self, unit, old_identity, scr)
        Player.record_carousel_panel(self, unit, name, stars, scr)
        self.reread = getattr(self, 'reread', set()) | {o.id for o in self.roster.units if o is not unit and not o.missing
                                                         and (o.name == name or o.unverified)}
        self.news_event('unit_corrected', scr.stage, f'unit {unit.id} on bench slot {src[1]} is {name}{"*" * unit.star}, '
                                                      f'not {old} (read before it was moved onto the board): not moved; '
                                                      f'its hex and items in your plan are dropped')
        self.fail_cause = f'unit {unit.id} reads {name}, not {old}: not moved onto the board'
        return True

    def identity_corrected(self, unit, before, scr):
        """Publish a positive identity fact and cancel stale intent in every owning plan."""
        if before[0] == unit.name:
            return
        unit.identity_corrected = True
        seen = set()
        for attr in ('round_plan', 'handover_rp', '_handover_saved', '_pending_fallback',
                     '_next_replaces', '_superseded_rp'):
            rp = getattr(self, attr, None)
            if rp is None or id(rp) in seen:
                continue
            seen.add(id(rp))
            if isinstance(getattr(rp, 'p', None), dict):
                Player.drop_stale_identity(self, rp.p, getattr(rp, 'names', {}), unit)
        nxt = getattr(self, 'next_plan', None)
        if nxt:
            Player.drop_stale_identity(self, nxt[1], getattr(self, 'next_plan_names', {}), unit)
        self.news_event('unit_corrected', scr.stage,
                        f'unit {unit.id} at {unit.loc} is {unit.name} {unit.star}-star, '
                        f'not {before[0] or "unidentified"} {before[1]}-star; stale steps cancelled')

    def drop_stale_identity(self, plan, names, unit):
        meant = names.get(unit.id)
        if meant and meant[0] == unit.name:
            return                              # a newer decision already used this correction
        for key in ('board', 'bench', 'items', 'strip'):
            if isinstance(plan.get(key), list):
                plan[key] = [e for e in plan[key] if not isinstance(e, dict)
                             or (e.get('id') != unit.id and e.get('from') != unit.id)]
        if isinstance(plan.get('sell'), list):
            plan['sell'] = [i for i in plan['sell'] if i != unit.id]

    def confirm_incoming(self, scr):
        """Before a fielded unit is benched, the bench units the plan fields whose identity is an inference
        (unverified) are read first, each once: a fielded unit benched for a unit that turns out to be another
        champion leaves its team slot empty for the fight (run 20261028-012516 6-2: Scuttlecrab and Krug** were
        benched and Mama Beak swapped out for an "Elder Dragon" that read Kayle only when it was dragged - the
        fight was 7 units of 9, lost 32 -> 14, out at 6-3). -> True when one read as another champion
        (confirm_before_field drops its entries; the planner then keeps the others on the board)."""
        rp = getattr(self, 'round_plan', None)
        if rp is None:
            return False
        done = self.__dict__.setdefault('_incoming_read', set())
        for u, _ in rp._lineup(self.roster):
            if u.on_board or u.loc[0] != 'bench' or not u.name or not u.unverified or u.id in done:
                continue
            done.add(u.id)
            if self.confirm_before_field(u, u.loc, scr):
                return True
        return False

    def bench_sell_blocked(self, frame, unit):
        """Before E on a bench unit, its slot is looked at: in 14 of 19 'failed' sells of four games on 2026-10-18 the
        slot had no health bar before the E. -> 'wait' when no bench bar is drawn at all yet (a planning phase opening:
        4-7 at 29 s showed a full bench with no bars), 'stale' when other bars are drawn but none in its slot and its
        tile does not read as a unit (the unit had gone or moved: the same frame showed a Brambleback where Cinderling
        was tracked), else None. Only in a planning phase: in combat no bench bar is ever drawn, and 130 of 163 bench
        sells in combat (2026-10-15..18) went through."""
        from tft_player import bench as BN
        if not getattr(getattr(self, 'phase', None), 'planning', False):
            return None
        bars = B.bar_places(frame, exclude=self.watcher.tac[1] if getattr(self, 'watcher', None) else None)[0]
        if not any(bars):
            return 'wait' if any(u.loc[0] == 'bench' and u is not unit for u in self.roster.units) else None
        if not bars[unit.loc[1]] and BN.classify(frame, bars)[unit.loc[1]]['kind'] != 'unit':
            return 'stale'
        return None

    def stale_bench_slot(self, frame, unit):
        """A bench unit to sell is not at its tracked slot: the one slot whose health bar no tracked bench unit
        explains, when there is exactly one (else None). Once a unit a phase: a second miss is not chased."""
        stage = getattr(getattr(self, 'phase', None), 'stage', None) or getattr(self.roster, 'stage', None)
        moved = self.__dict__.setdefault('_sell_relocated', set())
        if (unit.id, stage) in moved:
            return None
        bars = B.bar_places(frame, exclude=self.watcher.tac[1] if getattr(self, 'watcher', None) else None)[0]
        tracked = {u.loc[1] for u in self.roster.bench() if not u.missing}
        free = [i for i, b in enumerate(bars) if b and i not in tracked and not self.on_tactician(i)]
        if len(free) != 1:
            return None
        moved.add((unit.id, stage))
        return free[0]

    def occupant_differs(self, occupant, dst, scr):
        """Before a bench unit is swapped onto an occupied hex, the unit standing there is read: the swap sends it to the
        bench. -> True when it is another champion than the records say: nothing is dragged, the board is read again
        (a survey) and the model is told. A read that gives no name changes nothing. (Benchmark game 20261018-115603
        4-1: a swap meant to bench Scuttlecrab benched Pebbles, the 3-item carry, for the fight - the records had
        her at another hex since a 3-7 correction.)"""
        try:
            got = self.cap.read()
            name, stars, _ = self.read_fresh(self.unit_point(dst, B.green_bars(got[0]) if got else None), [])
        except NotForeground:
            return False
        self.log(ev='confirm_occupant', stage=scr.stage, unit=repr(occupant), read=name, stars=stars)
        if not name or name == occupant.name:
            if name:
                occupant.unverified, occupant.sure = None, True
            return False
        self.need_survey = f'hex {dst[1:]} holds {name}, not {occupant.name} (read before a swap)'
        self.survey_places = (self.need_survey, [dst])
        self.fail_cause = (f'hex {dst[1:]} holds {name}, not {occupant}: not swapped (it would have gone to the bench); '
                           f'the board is read again')
        self.news_event('unit_corrected', scr.stage, f'hex {dst[1:]} holds {name}{"*" * (stars or 1)}, not {occupant} '
                                                      f'(read before a swap onto it): not swapped; the board is read again')
        return True

    def unit_point(self, loc, bars=None):
        """Where to click, sell or drag a unit: its body just under its health bar - on the bench too (the bench slot's
        point is at its feet, below a small model: E there sold nothing and panel reads read nothing - run
        20261015-161625: LeBlanc at 6-3, Kayle at 2-6). Without a bar: body height over the slot / hex."""
        if loc[0] == 'bench':
            own = [b for b, place in B.bar_places(None, bars=bars or [])[2] if place == loc]
            if own:
                return (own[0][0], own[0][1] + 45)
            return (B.BENCH[loc[1]][0], self.BENCH_BAR_Y + 45)
        hx, hy = B.hex_xy(loc[1], loc[2])
        # only the bar the occupancy read gives this hex: the nearest bar by x can be a neighbour's
        # (run 152104 5-5: (1,0)'s point took 2-star Aphelios's bar at (0,0), 28 px off vs Nidalee's
        # own 30 px, and three board checks read Nidalee as "Aphelios")
        own = [b for b, place in B.bar_places(None, bars=bars or [])[2] if place == loc]
        if own:
            bx, by = own[0]
            # a tall model's bar is far over its hex: 45 px under it is between Kayle's wings (161404: read nothing)
            return (bx, by + (45 if hy - by <= 110 else round((hy - by) * 0.55)))
        return (hx, hy - 15)

    def tall_hint(self):
        """For board.bar_places, in a planning phase: {hex: bar height} of our fielded units whose champion and star
        level carry a bar too high for nearest-hex matching (board.BAR_DY). Fights move units: nothing then."""
        if not getattr(getattr(self, 'phase', None), 'planning', False) or not B.BAR_DY:
            return {}
        return {u.loc[1:]: B.BAR_DY[(u.name, u.star)] for u in self.roster.board()
                if not u.missing and (u.name, u.star) in B.BAR_DY}

    def learn_tall_bars(self, found, bars):
        """A board check's reads: a unit read at its hex centre (no bar of its own) under a bar 95-175 px up that is
        unplaced or gave an empty read at the hex it was matched to is a tall model - that bar is its own. Its height
        is learned for the champion and star level (board.BAR_DY) and the empty read is dropped: no '?' record.
        (Benchmark game 20261018-161404: Kayle 2-star at (0,0), bar 142 px up; four board checks 3-6..4-6, two
        phantom '?' units.) -> the reads to keep."""
        places = B.bar_places(None, bars=bars or [])[2]
        own = {loc for _, loc in places}
        empty = {tuple(f[0]) for f in found if f[0][0] == 'board' and not f[1]}
        placed = {tuple(b) for b, _ in places}
        drop = set()
        for f in found:
            loc, name = tuple(f[0]), f[1]
            if loc[0] != 'board' or not name or loc in own:
                continue
            rec = self.roster.at(loc)
            stars = f[2] or (rec.star if rec is not None and rec.name == name else 1)    # (a badge can go unread)
            hx, hy = B.hex_xy(*loc[1:])
            above = [(b, pl) for b, pl in places if pl in empty and pl not in drop] + \
                    [(b, None) for b in bars or [] if tuple(b) not in placed]
            cands = [(b, pl) for b, pl in above if abs(b[0] - hx) < 45 and 95 <= hy - b[1] <= 175]
            if len(cands) != 1:
                continue
            b, pl = cands[0]
            B.BAR_DY[(name, stars)] = round(hy - b[1])
            if pl is not None:
                drop.add(pl)
            self.log(ev='tall_bar', unit=name, stars=stars, hex=list(loc[1:]), height=round(hy - b[1]),
                     read_as=list(pl[1:]) if pl else None)
        return [f for f in found if tuple(f[0]) not in drop]

    def tactician_off(self, point):
        """Our tactician stands over `point` (1080p; a unit we are about to click, read, drag or sell): it is walked to
        PARK first - a click there hit the tactician, an E-sell missed (run 20261015-161625 2-6: four panel reads of
        bench slot 1 read nothing, two sells did not land). Waits at most 1.2 s for it to move off."""
        if REPLAY:
            return
        box = self.watcher.tac[1] if getattr(self, 'watcher', None) else None
        covers = lambda b: b is not None and b[0] - 50 <= point[0] <= b[2] + 50 and b[1] - 20 <= point[1] <= b[3] + 170
        if not covers(box):
            return
        try:
            self.inp.click(PARK, 'right')
        except NotForeground:
            return
        self._parked = self.phase.stage
        self.log(ev='tactician_moved_off', point=[round(point[0]), round(point[1])])
        end = time.time() + 1.2
        while time.time() < end and covers(self.watcher.tac[1]):
            time.sleep(0.08)

    def read_unit(self, point):
        """Right-click a unit and read its info panel -> (name, stars, fresh). fresh: the panel
        changed after the click; unchanged means it already showed this champion (a duplicate)
        or the click missed. The panel's items are kept in self.panel_items."""
        self.tactician_off(point)
        # A loot right-click can open a panel on a unit. Its delayed panel change
        # belongs to that click, even if a bench read has since been requested.
        settle = 0.8 - (time.time() - getattr(self, '_loot_at', 0))
        if settle > 0:
            time.sleep(settle)
        got = self.cap.read()
        before = self.reader.panel_sig(got[0]) if got else None
        # the champion the panel showed before the click: a panel that "changed" but still shows it is the old one
        # redrawn (its health, its mana) or a twin - the new one may still come (benchmark game 20261020-181325 4-1:
        # Kobuko's panel, open from the check before a swap, passed as the read of the hex Zyra had just taken; the
        # swap was judged failed and the records went wrong for two rounds)
        self.panel_before = self.reader.read_panel(got[0]) if got else (None, None)
        self.inp.click(point, 'right')
        for k in range(10):
            time.sleep(0.07)
            got = self.cap.read()
            if not got:
                continue
            sig = self.reader.panel_sig(got[0])
            changed = before is None or sig.shape != before.shape or \
                (np.abs(sig - before).max(axis=2) > 40).mean() >= 0.05
            if changed or k >= 6:
                name, stars = self.reader.read_panel(got[0])
                if changed and name and (name, stars) == self.panel_before and k < 6:
                    continue
                if name or k >= 6:
                    self.panel_items = read_panel_items(got[0]) if name else []
                    return name, stars, changed
        return None, None, False

    def read_fresh(self, point, others):
        """read_unit that never takes a stale panel: when the panel did not change (the same
        champion as before, or a missed click), show another unit first (`others`: points of
        units with a different name) and read again. A read of the champion the panel showed before the click is no
        fresh read either (read_unit: the old panel redrawn). -> (name, stars, items) or (None, None, [])."""
        def new(name, stars, fresh):
            if fresh and (name, stars) != getattr(self, 'panel_before', None):
                return bool(name)
            # An already open panel is not empty. For a loss check, a selection ring at the
            # requested place is an independent witness that it belongs to this unit.
            loc = getattr(self, '_loss_loc', None)
            got = self.cap.read() if name and loc is not None else None
            return bool(got and self.selected_here(got[0], loc))
        name, stars, fresh = self.read_unit(point)
        if new(name, stars, fresh):
            return name, stars, list(self.panel_items)
        if getattr(self, '_loss_loc', None) is not None:
            # Exactly one retry after an unreadable loss check. Moving the hover off and
            # back lets the game update the target before the second click.
            self.inp.move(PARK)
            self.inp.move(point)
            time.sleep(SELL_HOVER)
            name, stars, fresh = self.read_unit(point)
            if new(name, stars, fresh):
                return name, stars, list(self.panel_items)
            return None, None, []
        for other in others[:2]:
            self.read_unit(other)
            name, stars, fresh = self.read_unit(point)
            if new(name, stars, fresh):
                return name, stars, list(self.panel_items)
        return None, None, []

    def targeted_check(self, scr, reason):
        """A board check of the places its trigger names (self.survey_places), not of every unit (the owner,
        2026-10-30: "why not just check the specific units?" - game 50's four checks read every unit, 37 s, where each
        trigger named one hex or slot). Each place is read off its info panel (~0.6 s) and its record corrected; a
        record read gone from its place is set aside, and the places whose health bar no record explains are read too
        (where the unit went). -> True when the records then explain every bar on screen; False -> the full check.
        A trigger that names no place gets the full check."""
        named = getattr(self, 'survey_places', None)
        self.survey_places = None
        places = named[1] if named and named[0] == reason else None    # the places this reason named, not another's
        if not places or REPLAY:
            return False
        t0 = time.time()
        tac_box = self.watcher.tac[1]
        got = self.cap.read()
        if not got:
            return False
        bars = B.green_bars(got[0])
        bench_seen, board_seen = B.occupancy(got[0], exclude=tac_box)
        on_screen = {('bench', i) for i, b in enumerate(bench_seen) if b} | {('board', *h) for h in board_seen}
        bounded = (getattr(scr, 'timer', None) is not None
                   or getattr(getattr(self, 'phase', None), 'planning', False))
        context = getattr(self, '_survey_recovery', None)
        if context is None or context['stage'] != scr.stage or context['reason'] != reason:
            context = {'stage': scr.stage, 'reason': reason, 'read': set()}
        found, read = [], set(context['read'])
        # A broad neighbour check took 9.9 seconds in game 64. Read a small
        # slice, apply positive facts immediately, and leave the rest for idle time.
        remaining = []

        def read_places(locs):
            for loc in locs:
                loc = tuple(loc)
                if loc in read:
                    continue
                if bounded and (len(found) >= 2 or time.time() - t0 >= 1.5):
                    remaining.append(loc)
                    continue
                others = [self.unit_point(f[0], bars) for f in found if f[1]][-2:]
                name, stars, items = self.read_fresh(self.unit_point(loc, bars), others)
                read.add(loc)
                found.append([loc, name, stars, items])
                here = self.roster.at(loc)
                if not name and here is not None and loc not in on_screen:
                    here.missing = 'a board check found no unit at its place'
        try:
            read_places(places)
            self.apply_partial_survey([f for f in found if f[1]], scr, reason)
            explained = {u.loc for u in self.roster.units if not u.missing}
            extra = sorted(on_screen - explained - read, key=lambda l: (l[0] != 'board', l[1:]))
            read_places(extra if bounded else extra[:4])
        except NotForeground:
            self.survey_places = (reason, [tuple(l) for l in places if tuple(l) not in read] or places)
            self.need_survey = reason
            return True                                   # not sent: the records stay; asked again if still wrong
        self.apply_partial_survey([f for f in found if f[1]], scr, reason)
        tracked = {u.loc for u in self.roster.units if not u.missing}
        blind = {loc for loc, name, _, _ in found if name and loc[0] == 'board' and loc not in on_screen}
        ok = on_screen <= tracked and not {l for l in tracked - on_screen if l[0] == 'bench'}             and not ({l for l in tracked - on_screen if l[0] == 'board'} - blind)
        if ok:
            self.bar_blind = {loc[1:]: self.roster.at(loc).id for loc in blind if self.roster.at(loc) is not None}
        took = time.time() - t0
        remaining = list(dict.fromkeys(remaining))
        if remaining:
            self.survey_places, self.need_survey = (reason, remaining), reason
            context['read'] = read
            self._survey_recovery = context
        else:
            self._survey_recovery = None
            if bounded and not ok:
                self.log(ev='recovery_unresolved', stage=scr.stage, reason=reason,
                         on_screen=sorted(on_screen), roster=repr(self.roster))
                if hasattr(self, 'log_dir'):
                    self.snap(got[0], f'{scr.stage}-recovery-unresolved', True, force=True)
        print(f"{scr.stage}: targeted check ({reason}) in {took:.1f}s: {len(found)} place(s) read"
              + (' - remaining places queued' if remaining else '' if ok else ' - board still differs'))
        self.log(ev='targeted_check', stage=scr.stage, reason=reason, seconds=round(took, 1), ok=ok,
                 pending=remaining, found=[(list(f[0]), f[1], f[2]) for f in found], roster=repr(self.roster))
        return bounded or ok

    def apply_partial_survey(self, found, scr, reason):
        """A board check stopped by the model's plan: each place read takes the unit read there (a tracked
        unit of that name and star level moves to it), unread places keep their records."""
        from tft_player.state import Owned
        changed = []
        reserved = {u.id for loc, name, stars, _ in found
                    if (u := self.roster.at(loc)) is not None and u.name == name
                    and (stars is None or stars == u.star)}
        used = set()
        for loc, name, stars, items in found:
            if not name:
                continue
            here = self.roster.at(loc)
            if here is not None and here.name == name and (stars is None or here.star == stars):
                here.unverified = None                   # confirmed: otherwise unchanged
                here.items = [i for i in items if i != 'unknown']
                here.sure, here.star_assumed = True, False
                used.add(here.id)
                continue
            if self.roster.same_unit_read(here, name, stars, items):
                self.survey_star(here, stars, scr.stage)  # the same unit, another star level: it keeps its id
                here.unverified = None
                here.items = [i for i in items if i != 'unknown']
                changed.append(f'{name} at {loc[1:]} is {stars}-star')
                continue
            cands = [u for u in self.roster.units if u.name == name and (stars is None or u.star == stars)
                     and u is not here and u.id not in reserved | used]
            u = min(cands, key=lambda x: (x.loc[0] != loc[0], x.loc)) if cands else None
            if u is None and here is not None:
                before_identity = (here.name, here.star)
                here.name, here.star = name, stars or 1
                here.items = [i for i in items if i != 'unknown']
                Player.identity_corrected(self, here, before_identity, scr)
                u = here
            if here is not None and here is not u:
                # the record that stood here is somewhere else: kept, as not located (found again later)
                here.missing = f'a board check read {name} at its place'
            if u is None:
                u = Owned(name, stars or 1, loc, ours=True)
                self.roster.units.append(u)
            u.loc, u.unverified, u.missing = loc, None, None
            u.sure, u.star_assumed = True, False
            u.items = [i for i in items if i != 'unknown']
            used.add(u.id)
            changed.append(f'{name} at {loc[1:]}')
        for loc, name, stars, held in found:
            here = self.roster.at(loc)
            if not name or here is None or here.name != name:
                continue
            for other in self.roster.units:
                if other is not here and other.loc == loc and not other.missing:
                    other.missing = f'one physical unit at {loc} was positively read as {name}'
                    changed.append(f'unit {other.id} is not a second body at {loc}')
        print(f'{scr.stage}: board check ({reason}) stopped for the plan after {len(found)} reads'
              + (f'; corrected: {", ".join(changed)}' if changed else ''))
        self.log(ev='survey_partial', stage=scr.stage, reason=reason, read=len(found), changed=changed)
        if changed and ('news_event' in self.__dict__ or hasattr(self, 'news')):
            self.news_event('unit_corrected', scr.stage, 'board check: ' + '; '.join(changed))

    def survey_star(self, u, stars, stage):
        """A board check read u's own place as u's champion at another star level: u keeps its id (the plan's
        references to it stay valid) and takes the star level read. A rise is a merge we did not see: copies not
        seen anywhere merge into it (Roster.set_star). A fall is an assumed merge that did not happen."""
        visible = {x.loc for x in self.roster.units if not x.missing}
        old, gone = self.roster.set_star(u, stars, visible)
        if stars > old:
            u.star_badge_at = time.time()       # the copies it used, once seen missing, merge into it
        self.log(ev='star_corrected', stage=stage, unit=repr(u), was=old, now=stars, used=[repr(x) for x in gone],
                 source='board check')
        what = (f'{u.name} (unit {u.id}) is {stars}-star on its panel (we had {old}-star)'
                + (f'; its copies {", ".join(str(x.id) for x in gone)} are gone (merged)' if gone else
                   '; a merge we assumed did not happen' if stars < old else ''))
        if hasattr(self, 'news_event'):
            self.news_event('star_corrected', stage, what)

    def check_borrowed_panels(self, found, bars):
        """A click can open a neighbour's panel (a bigger model in front of the hex). Two places that
        read the same champion, stars and items, where the records had a different confirmed unit on
        one of them: that place is read again at its hex centre. If it still shows the neighbour,
        its record stays (marked unverified) - a doubtful read never renames a confirmed unit."""
        def same(a, b):
            return a[1] and a[1] == b[1] and a[2] == b[2] and sorted(a[3]) == sorted(b[3]) \
                and (a[3] or (a[2] or 1) >= 2)       # equal items, or two 2-stars: one is borrowed
        for f in found:
            twin = next((g for g in found if g is not f and same(f, g)), None)
            rec = self.roster.at(f[0])
            if twin is None or rec is None or not rec.name or rec.name == f[1] or rec.unverified or rec.missing:
                continue
            hx, hy = B.hex_xy(*f[0][1:]) if f[0][0] == 'board' else B.BENCH[f[0][1]]
            px, py = self.unit_point(f[0], bars)
            # higher on its own body (away from a model in front), then the hex centre
            for point in ((px, py - 25), (hx, hy - 15)):
                self.read_unit(self.unit_point(twin[0], bars))          # show the twin, so a change is seen
                name, stars, fresh = self.read_unit(point)
                if fresh and name and (name, stars) != (twin[1], twin[2]):
                    f[1], f[2], f[3] = name, stars, list(self.panel_items)
                    break
            else:
                note = (f'a board check read {twin[1]}{"*" * (twin[2] or 1)} here, the same panel as at '
                        f'{tuple(twin[0][1:])}: kept as {rec.name}{"*" * rec.star}')
                f[1], f[2], f[3] = rec.name, rec.star, list(rec.items)
                f.append(note)
            self.log(ev='borrowed_panel', loc=list(f[0]), twin=list(twin[0]), record=rec.name, now=f[1],
                     kept=len(f) > 4)
            where = tuple(f[0][1:]) if f[0][0] == 'board' else f'bench {f[0][1]}'
            self.state_note = (time.time(), f'{rec.name} {where}: ' + ('read as its neighbour - kept' if len(f) > 4
                               else f're-read as {f[1]}'), 'muted' if len(f) > 4 else 'amber')

    def survey_records(self, found):
        """A finished board check's reads -> (new records, records dropped). Only the wrong records change (the
        user, 2026-10-09): a place read as the champion recorded there, or read without a name, keeps its record
        as it is; a place read as another champion takes that champion's record from elsewhere (or a new one);
        a record whose place showed nothing is kept as missing."""
        from tft_player.state import Owned
        old = list(self.roster.units)
        new, rest = [], []
        for f in found:                                   # 1. records the reads confirm: unchanged
            loc, name, stars, items, *kept = f
            here = next((x for x in old if x.loc == loc and not x.missing), None)
            # the BFF rider opens no panel: a name read at its hex is another unit's panel (benchmark game
            # 20261021-141731 4-3: read as Stonebark Tree, the rider Teemo's record was replaced, dropped at 4-5, and
            # the 3-star Teemo on the BFF was moved like any unit at 5-2)
            rider = here is not None and getattr(here, 'riding', None) and here.on_board
            if here is not None and (not name or kept or rider or (here.name == name and (stars is None or here.star == stars))
                                     or self.roster.same_unit_read(here, name, stars, items)):
                if kept:
                    here.unverified = kept[0]             # its read was another unit's panel: the record stays
                elif name and not rider:
                    here.unverified = None
                    if stars and stars != here.star:
                        # the champion recorded here at another star level: the same unit, its star was wrong
                        self.survey_star(here, stars, getattr(self.phase, 'stage', None))
                        here.items = [i for i in items if i != 'unknown']
                    if stars:
                        here.star_assumed = False         # its star level read off its panel
                old.remove(here)
                new.append(here)
            else:
                rest.append(f)
        for loc, name, stars, items, *kept in rest:       # 2. the wrong or missing ones
            cands = [u for u in old if name and u.name == name and (stars is None or u.star == stars)]
            u = min(cands, key=lambda x: (x.loc[0] != loc[0], len(x.items) == 0)) if cands else None
            if u is None:
                u = next((x for x in old if x.loc == loc and not x.name), None)    # a '?' there takes the name
            if u is None:
                u = Owned(name, stars or 1, loc, ours=True)
            else:
                old.remove(u)
            u.loc, u.missing, u.name = loc, None, name or u.name
            if kept:
                u.unverified = kept[0]
            elif name:
                u.items = [i for i in items if i != 'unknown']      # what the panel shows it holds
                u.unverified = None
                if stars:
                    u.star, u.star_assumed = stars, False
            new.append(u)
        # 3. a unit is removed only by a sell, a merge or a game effect we saw: one this check did not find is
        # kept, as not located - a bar and a click can both miss (a monster or the tactician in front); a later
        # read names it again. Only nameless '?' records are dropped.
        for u in list(old):
            if u.name:
                u.missing = 'not seen in a board check (no health bar, no panel at its place)'
                new.append(u)
                old.remove(u)
                self.state_note = (time.time(), f'{u.name} {tuple(u.loc[1:]) if u.on_board else "bench " + str(u.loc[1])}: '
                                   'unseen - kept as missing', 'amber')
        return new, old

    def survey(self, scr, reason, stop=None):
        """Read every fielded and benched unit (health bars say where) off its info panel and
        rebuild the roster from what the game shows; tracked units keep ids and items when
        they match by name and stars."""
        t0 = time.time()
        tac_box = self.watcher.tac[1]
        # health bars flicker (a monster or effect in front, an animation): one frame can miss a
        # fielded unit, and a missed unit used to leave the records (run 152104 4-7: 2-star Sentinel
        # at (3,4) dropped, then Diana was moved onto its hex). Occupancy is the union of a few frames.
        seen = Counter()                                          # place -> frames that showed a bar there
        bars = None
        for k in range(3):
            got = self.cap.read()
            if not got:
                continue
            b_seen, h_seen = B.occupancy(got[0], exclude=tac_box)
            seen.update([('bench', i) for i, b in enumerate(b_seen) if b] + [('board', r, c) for r, c in h_seen])
            fb = [b for b in B.green_bars(got[0])
                  if not (tac_box and tac_box[0] <= b[0] <= tac_box[2] + 40 and tac_box[1] <= b[1] <= tac_box[3])]
            if bars is None or len(fb) > len(bars):
                bars = fb                                     # click points from the fullest frame
            if k < 2:
                time.sleep(0.08)
        if bars is None:
            return
        # a place seen in 2 of 3 frames holds a unit; one seen once (a unit still walking back after the
        # fight - run 213127 2-7 made two phantom '?' units of that) and a confirmed unit's place that no
        # bar showed are only probed: ONE click, and no panel there means no unit (a probe costs ~0.6 s;
        # read_fresh's retries made empty probes ~3 s each - the 4-3 check took 16.9 s)
        firm = sorted((l for l, n in seen.items() if n >= 2), key=lambda l: (l[0] != 'board', l[1:]))
        board_seen = {l[1:] for l in firm if l[0] == 'board'}
        probes = [l for l, n in seen.items() if n == 1]
        probes += [u.loc for u in self.roster.units if u.on_board and u.name and not u.missing and not u.unverified
                   and u.loc not in firm and u.loc not in probes]
        probes = sorted(set(probes), key=lambda l: (l[0] != 'board', l[1:]))[:4]
        found, partial = [], False
        try:
            for loc in firm:
                if stop is not None and stop():
                    partial = True                               # the model's plan came: it goes first
                    break
                shown = found[-1][1] if found else None          # the champion the panel shows now
                others = [self.unit_point(f[0], bars) for f in found if f[1] and f[1] != shown]
                name, stars, items = self.read_fresh(self.unit_point(loc, bars), others)
                found.append([loc, name, stars, items])
            for loc in ([] if partial else probes):
                if stop is not None and stop():
                    partial = True
                    break
                name, stars, fresh = self.read_unit(self.unit_point(loc, bars))
                if fresh and name:                               # the panel changed to a unit standing there
                    found.append([loc, name, stars, list(self.panel_items)])
                    if loc[0] == 'board':
                        board_seen.add(loc[1:])
            if not partial:
                self.check_borrowed_panels(found, bars)
                found = self.learn_tall_bars(found, bars)
        except NotForeground:
            return
        if partial:
            # cut short: only the places read are corrected; nothing is removed or marked missing
            self.apply_partial_survey(found, scr, reason)
            return
        if not any(l[0] == 'bench' for l in seen):
            # no bench health bar in any frame (a fight began, or the bars are not drawn yet): the bench was not read,
            # so its records stay as they are - not missing (174944 3-5: six bench units dropped a round later)
            read = {tuple(f[0]) for f in found}
            found += [[u.loc, u.name, u.star, list(u.items), 'not read: no bench health bar during a board check']
                      for u in self.roster.units if u.loc[0] == 'bench' and not u.missing and u.name
                      and tuple(u.loc) not in read]
        before = repr(self.roster)
        new, old = self.survey_records(found)
        renamed = [(f[0], f[1]) for f in found if f[1] and (r := self.roster.at(f[0])) is not None and r.name
                   and r.name != f[1] and not r.unverified and len(f) <= 4 and not getattr(r, 'riding', None)]
        if renamed:
            loc, name = renamed[0]
            self.alert('RECORD', f'board check: {self.roster.at(loc).name} at {loc[1:]} now read as {name}', 'State')
        kept = [u for u in new if u.missing and u.missing.startswith('not seen in a board check')]
        if kept:
            self.alert('RECORD', 'board check could not find ' + ', '.join(f'{u.name}{"*" * u.star}' for u in kept[:3])
                       + ' - kept as not located', 'State')
        old_locs = {u.id: (u.loc, u) for u in self.roster.units}
        self.roster.units = new
        self.roster.board_seen = len(board_seen)
        # board units found by a probe (no health bar in any frame): their hexes hide the bar (see board_check)
        self.bar_blind = {tuple(f[0][1:]): u.id for f in found if f[0][0] == 'board' and tuple(f[0]) not in
                          {tuple(l) for l in firm} and f[1] and (u := self.roster.at(tuple(f[0]))) is not None
                          and u.name == f[1]}
        moved = [f'{u.name or "?"} (unit {u.id}) is at {tuple(u.loc[1:]) if u.on_board else "bench " + str(u.loc[1])}, '
                 f'not at {tuple(old_locs[u.id][0][1:]) if old_locs[u.id][0][0] == "board" else "bench " + str(old_locs[u.id][0][1])}'
                 for u in new if u.id in old_locs and old_locs[u.id][0] != u.loc]
        if moved:
            self.news_event('survey', scr.stage, 'board check: places corrected: ' + '; '.join(moved))
        elif 'model asked' in str(reason):           # the answer to the model's own request
            self.news_event('survey', scr.stage, f'the board check you asked for: all {len(new)} places match the records')
        took = time.time() - t0
        print(f'{scr.stage}: survey ({reason}) in {took:.1f}s -> {self.roster}')
        self.log(ev='survey', stage=scr.stage, reason=reason, seconds=round(took, 1), before=before,
                 after=repr(self.roster), found=[(list(f[0]), f[1], f[2], f[3]) for f in found],
                 lost=[repr(u) for u in old])
        self.status_extra['identify'] = f'survey ({reason}): {len(found)} units read'

    def identify_one(self, scr, combat=False, arrivals=False, only_ids=None):
        """Read ONE unit off its info panel: a unit that arrived without our buy ('?'), or one an
        event may have changed (self.reread: a Wisp/augment that transforms units, a sell or move
        that did not confirm). Known units are never read again. arrivals: only '?' units on the bench,
        read at once (local time, the user 2026-10-09); the rest only in otherwise idle time - while the
        model thinks, or in combat (bench units only: fighting units move)."""
        reread = getattr(self, 'reread', set())
        # an unnamed unit gets its 4 tries again in each new round: it is never left '?' for the rest of the game
        # (request e9, game 37: Kennen from Luxury Subscription failed its 4 reads at 3-2 and stayed '?' to 3-5; the
        # player placed it on a guess, the move was refused and 3-5 was fought 6 of 7)
        tried_at = self.__dict__.setdefault('_tries_stage', {})
        for u in self.roster.units:
            if u.name is None and self.tries.get(u.id, 0) >= 4 and tried_at.get(u.id) != scr.stage:
                self.tries[u.id] = 0
        unknown = [u for u in self.roster.units
                   if (u.name is None or (u.id in reread and not arrivals)) and self.tries.get(u.id, 0) < 4
                   and (only_ids is None or u.id in only_ids)
                   and not u.missing and not (combat and u.on_board) and not (arrivals and u.on_board)]
        if not unknown or (not combat and scr.timer is not None and scr.timer < (1 if arrivals else 4)):
            return False
        # least-tried first: consecutive clicks go to different units, so the panel
        # switches between them (two copies of one champion look identical)
        # a unit whose items are in doubt first (request e43, game 45: Brambleback's records kept an Edge of Night
        # from 4-7 to 5-3, flagged at 5-1, read only at 5-3 after the bench units)
        # ... and a unit whose star badge contradicts its record before the rest (game 80 5-5: the 2-star carry's panel
        # waited behind the bench units until 5-6)
        u = min(unknown, key=lambda x: (self.tries.get(x.id, 0), not getattr(x, 'badge_star', None),
                                        not getattr(x, 'items_check', False), x.on_board, x.loc))
        got = self.cap.read()
        if got and u.on_board and not REPLAY:
            # a fielded unit is read only where its health bar shows (run 221257 5-2: Aphelios on hex
            # (0,0) was clicked as the planning phase ended, no bar was there, the panel showed a
            # Leona 2-star, and the model then benched and stripped that "Leona" and sold it at 5-3)
            # (the bar must be assigned to THIS hex: the old 75 px / 140 px window took the bar of the unit on the
            # neighbouring hex - run 20261010-235723 4-3: a stale Azir record at (0,0) was "read" off the real Azir***
            # at (1,0), which was then recorded a second time)
            if u.loc not in {loc for _, loc in B.bar_places(got[0], exclude=self.watcher.tac[1])[2]}:
                return False
        retry = u.id in tried_at                 # read before (this round or an earlier one) and not named
        self.tries[u.id] = self.tries.get(u.id, 0) + 1
        tried_at[u.id] = scr.stage
        try:
            pre = got[0] if got else None         # the unit before the click: no selection ring yet
            # a fielded unit is clicked on its body under its health bar (the hex centre missed
            # the 1-1 carousel unit twice in the first pilot game)
            point = self.unit_point(u.loc, B.green_bars(got[0]) if got else None)
            self.tactician_off(point)
            others = [o for o in self.roster.units if retry and o.name and o.loc[0] == 'bench' and not o.missing
                      and o is not u]
            other = next((o for o in others if o.name != u.name), None) or next(iter(others), None)
            if other is not None:
                # a read of this unit failed before: another unit's panel first, so that this one's panel is a change
                # (an unchanged panel names nobody - when the failed try had opened this unit's panel, every later
                # try saw no change: request e9, Kennen's four reads at 3-2; read_fresh does the same)
                self.inp.click(self.unit_point(other.loc), 'right')
                time.sleep(0.15)
                got = self.cap.read() or got
                # that unit's panel is open now: a read of its name is its panel still, unless the selection ring
                # stands at u (run 20261030-120756: the click on Scuttlecrab** at (3,2) opened no panel, the Pebbles**
                # panel of bench slot 1 was read - "unit 37 is Pebbles 2-star" at 5-1, 5-2, 5-7 and 6-1, and the 6-1
                # plan that benched Elder Dragon was refused as "Riftbeast falls 7 -> 6")
                self._last_panel_name = other.name
            before = self.reader.panel_sig(got[0]) if got else None
            self.inp.click(point, 'right')
            name, stars = None, None
            # The panel may still show the unit clicked before (or the click missed):
            # read it only once it has visibly changed after our click. An unchanged
            # panel names nobody - the unit stays '?' and is tried again, never guessed.
            for _ in range(10):
                time.sleep(0.08)
                got = self.cap.read()
                if not got:
                    continue
                sig = self.reader.panel_sig(got[0])
                if before is not None and sig.shape == before.shape and \
                        (np.abs(sig - before).max(axis=2) > 40).mean() < 0.05:
                    continue                     # unchanged: not (yet) this unit's panel
                name, stars = self.reader.read_panel(got[0])
                # the same champion as the panel showed before: maybe still that unit's panel (it can
                # change by itself, e.g. a merge to 2-star) - then the selection ring must be here.
                # (Only then: on recordings the ring test alone missed 62% of good reads.)
                if name and name == getattr(self, '_last_panel_name', None) and not self.selected_here(got[0], u.loc):
                    # the panel changed but shows another unit (run29 4-2: bench slot 3 read as the
                    # board Kog'Maw, whose panel had just changed to 2-star)
                    self.log(ev='read_not_selected', stage=scr.stage, unit=repr(u), read=name)
                    name, stars = None, None
                    continue
                if name:
                    self._last_panel_name = name
                    # the panel shows its items too: a carousel unit arrives holding one, and this read is what
                    # confirms the pick (25 of 25 carousels of 2026-10-10 were "could not be told": the names
                    # were read here, the items never)
                    self.panel_items = read_panel_items(got[0])
                    break
            # the panel is left open: it closes on our next click. (Escape is never
            # used - if the panel already closed, Escape opens the game menu.)
        except NotForeground:
            return False
        if name:
            g = getattr(u, 'gift', None)
            if g:
                u.gift = None
                if name != g['name']:
                    # not this gift: gifts landing together were named in slot order - the unit given this one's
                    # name takes the other name (read again too); else the gift is still to come
                    o = next((o for o in self.roster.units if o is not u and getattr(o, 'gift', None)
                              and o.gift['name'] == name), None)
                    if o is not None:
                        o.gift, o.name = g, g['name']
                        self.reread = reread | {o.id}
                        self.tries[o.id] = 0
                    else:
                        self.gifts = getattr(self, 'gifts', []) + [g]
            elif u.name is None:
                g = next((g for g in getattr(self, 'gifts', []) if g['name'] == name), None)
                if g:
                    self.gifts.remove(g)                                # a gift that arrived unnamed
            if u.id in reread:
                reread.discard(u.id)
                if u.name and (u.name != name or (stars and stars != u.star)):
                    self.news_event('unit_corrected', scr.stage, f'unit {u.id} is {name} {stars or u.star}-star, '
                                                                  f'not {u.name} {u.star}-star')
                    if u.name != name:
                        # a name a correction gave is checked again before the unit is sold (run 20261030-050404 3-3:
                        # two in-combat reads swapped Diana* and Cinderling** on the bench, and "sell Diana*@b5" sold
                        # the Cinderling 2-star at 3-5)
                        u.corrected = scr.stage
                        # a swap: the other records of this champion, and the unverified ones, are read too
                        # (run 185805 4-2: "Aphelios" was Malphite, and the real Aphelios was recorded as Vi)
                        self.reread = reread | {o.id for o in self.roster.units if o is not u and not o.missing
                                                and (o.name == name or o.unverified)}
            before_identity = (u.name, u.star)
            u.name = name
            u.unverified = None                  # read off its own info panel now
            u.sure = True
            if not u.on_board or not u.items_read or getattr(u, 'items_check', False):
                held = list(self.panel_items)
                if getattr(u, 'items_check', False):
                    held = fill_from_records(held, u.items, getattr(u, 'items_expect', None))
                    self.log(ev='items_from_panel', stage=scr.stage, unit=repr(u), items=held, was=u.items)
                    # an item its row read that its panel does not show is a ghost of that row: the row does not put it
                    # back (read_roster_items) until a give to the unit
                    u.row_ghosts = set(getattr(u, 'row_ghosts', None) or ()) | {
                        n for n in u.items or [] if n != 'unknown' and n not in held}
                u.items, u.items_read, u.items_check, u.items_expect = held, True, False, None
                u.items_predicted = False
            getattr(self, 'reread_asked', set()).discard(u.id)
            if stars:
                if stars > u.star:
                    u.star_badge_at = time.time()    # a merge: star_check takes its missing copies off the records
                u.star = stars
                u.star_assumed = False
                u.badge_star = None
            old_look = getattr(u, 'look_before_read', None)
            confirmed_same = old_look is not None and u.name == getattr(u, 'name_before_read', None)
            self.remember_look(u, pre, label=True)   # its appearance now that its name is confirmed
            u.labelled = True
            if confirmed_same:
                # the read said it IS the same unit: the earlier look stays a valid reference too
                # (run29: Kog'Maw on hex (0,1) was flagged 5 times and read as itself each time)
                u.looks_ok = (list(getattr(u, 'looks_ok', [])) + [old_look])[-2:]
            # read off its own panel: its looks are not questioned again at this star level (run 033507:
            # 61 of 91 re-reads named the same unit again - Kog'Maw 8 flags, Shen 4)
            u.look_trusted = u.star
            u.look_before_read = None
            # the game merges any 3 same-star copies, so 3 in our roster means a
            # read was wrong: look at the others again
            dup = [x for x in self.roster.copies(name, u.star) if x is not u]
            # in combat a fielded copy waits to merge (Roster._merge), so three real 1-stars can stand then; and a
            # confirmed fielded copy holding items keeps its name (request e193 E: two sure bench copies left only
            # the fielded 3-item carry to clear - game 59's 6-1 starting point)
            waits = getattr(self.roster, 'in_combat', False) and any(x.on_board for x in dup + [u])
            if len(dup) >= 2 and not waits:
                dup = [x for x in dup if not (x.on_board and getattr(x, 'sure', False) and x.items
                                              and not x.unverified)]
                # the doubtful records first: a copy read off its own panel keeps its name while an unverified or
                # unread one can take the blame (run 20261103-233832 6-1: the bought copy read at bench slot 8 cleared
                # the fielded Brambleback (unit 55, read at 5-5, three items) with the unverified bench record; the
                # board unit cannot be read in combat, the state showed "? (unidentified)", Riftbeast 6, and at 6-2
                # the player sold him with Rageblade, Quicksilver and Edge of Night)
                doubtful = [x for x in dup if x.unverified or not getattr(x, 'sure', False)]
                for x in (doubtful or dup):
                    if self.tries.get(x.id, 0) < 2:     # re-check once more at most - no loops
                        x.name = None
            print(f'{scr.stage} identified {u}')
            u.panel_read = {'stage': scr.stage, 'name': name, 'stars': stars}
            Player.identity_corrected(self, u, before_identity, scr)
            if before_identity[0] is None:
                self.news_event('unit_identified', scr.stage,
                                f'unit {u.id} at {u.loc} identified as {u.name} {u.star}-star')
            Player.record_carousel_panel(self, u, name, stars, scr, self.panel_items)
        elif u.name is None and u.loc[0] == 'bench' and not combat and pre is not None and not REPLAY \
                and getattr(getattr(self, 'phase', None), 'planning', False):
            self.ghost_check(u, pre, scr)
        self.log(ev='identify', stage=scr.stage, unit=repr(u), name=name, stars=stars)
        self.status_extra['identify'] = (f'read {u} from its info panel' if name
                                         else f'could not read the panel for {u}')
        self.live.event(scr.stage, 'identify', self.status_extra['identify'])
        return True

    def ghost_check(self, u, pre, scr):
        """A '?' bench record whose click opened no panel of it, while the frame before the click drew bench health
        bars but none at its slot: twice, and the record goes (an unnamed record holds nothing to keep). Run
        20261103-233832: the fielded Brambleback's model at (0,6) spilled over bench slot 8 at 5-6 and a '?' was
        recorded there. 5-6 and 5-7 showed the slot empty (no bar; eight clicks opened no panel of it) and the tile
        was never learned as empty while a record held it, so no read was evidence. It stayed: at 6-1 it hid where
        the bought Brambleback landed, and the buy was recorded on Kog'Maw's slot 1."""
        try:
            bench_bars = B.bar_places(pre, exclude=self.watcher.tac[1])[0]
        except Exception:
            return
        if not bench_bars or bench_bars[u.loc[1]] or not any(bench_bars):
            return
        u.empty_reads = getattr(u, 'empty_reads', 0) + 1
        if u.empty_reads < 2 or u not in self.roster.units:
            return
        self.roster.units.remove(u)
        self.__dict__.setdefault('_ghost_slots', {})[u.loc[1]] = scr.stage
        self.log(ev='ghost_dropped', stage=scr.stage, unit=u.id, slot=u.loc[1])
        self.news_event('unit_missing', scr.stage, f'the unidentified unit {u.id} on bench slot {u.loc[1]} is not '
                                                   f'there: no health bar and no panel at its slot - removed from '
                                                   f'the records')

    def check_carousel_unit(self, scr):
        """What the carousel gave: the ONE unit that arrived since it ended (not bought) holding an item,
        named and with its item row read - loot drops arrive meanwhile but hold nothing. Several such
        units: the one holding what we went for. Reported once; nothing found by the end of the next
        planning phase is reported as unknown, never guessed."""
        cc = getattr(self, 'carousel_confirm', None)
        if not cc:
            return
        if not cc.get('return_recorded') and hasattr(self, 'log_dir'):
            got_frame = self.cap.read()
            if got_frame:
                self.snap(got_frame[0], f'{cc["stage"]}-carousel-return', True, force=True)
                cc['return_recorded'] = True
        new = [u for u in self.roster.units if u.id not in cc['before'] and not u.ours and not u.missing]
        arrivals = list(new)
        # A carousel copy can merge into an existing unit while unrelated loot arrives.
        # The one new loot unit is then NOT evidence of which champion the carousel gave.
        from tft_player.carousel_returns import purchase_merges, unconfirmed_merges
        explained = purchase_merges(self.roster, cc, scr.stage)
        merged = [u.id for u in self.roster.units if u.id in cc.get('before_stars', {})
                  and u.id not in explained
                  and u.star > cc['before_stars'][u.id]]
        merged = sorted(set(merged) | unconfirmed_merges(self.roster, cc))
        cc['possible_merges'] = merged
        if merged:
            new = []
        # Some arrivals already have a name (a gift/roster inference) and thus
        # never enter identify_one's unnamed-arrival queue. A carousel label
        # still needs that unit's own panel, even when its item arrived separately.
        requested = cc.setdefault('panel_requested', set())
        panel_candidates = arrivals + [u for u in self.roster.units if u.id in merged and not u.missing]
        unread = {u.id for u in panel_candidates
                  if not getattr(u, 'panel_read', None)
                  or (u.id in merged and u.panel_read.get('stage') != scr.stage)} - requested
        if unread:
            self.reread = getattr(self, 'reread', set()) | unread
            tries = self.__dict__.setdefault('tries', {})
            for uid in unread:
                tries[uid] = 0
            requested.update(unread)
        carried = cc.get('carried')
        got = cc.get('bench_item') or self.carousel_bench_item(cc)
        if got and not (carried and carried[1]):
            # the carousel's item came to the item bench, not on the unit (request e12, game 38: all five carousels
            # ended "could not be told" - each time one item of that carousel's offer arrived on the item bench 1-3 s
            # after it ended, and the unit it gave held none)
            cc['bench_item'] = got
            pick = new[0] if len(new) == 1 else None
            unit = carried[0] if carried and carried[0] else (
                pick.name if pick and getattr(pick, 'panel_read', {}).get('name') == pick.name else None)
            if unit is None and time.time() - cc['at'] <= 75:
                return                         # item arrives first; retain the task until the unit's panel is read
            if unit is None:
                self.log(ev='carousel_label_unresolved', stage=cc['stage'], item=got,
                         candidates=[{'id': u.id, 'name': u.name, 'panel': getattr(u, 'panel_read', None)}
                                     for u in panel_candidates], possible_merges=cc.get('possible_merges', []),
                         reason='no unique returned unit with a panel read')
                Player.record_carousel_inventory(self, cc, scr)
            if pick and unit and not (carried and carried[0]):
                Player.label_carousel_return(self, cc, pick, got, scr)
            self.carousel_confirm = None
            what = (f"carousel: we got {got} (it arrived on the item bench)"
                    + (f" with {unit}" if unit else ''))
            print(f"{scr.stage} {what} (went for: {cc['went'] or 'not sure'})")
            self.log(ev='carousel_confirmed', stage=cc['stage'], unit=unit, items=[got], chaser_got=cc['got'],
                     went_for=cc['went'], match=Player.carousel_match(cc, got),
                     chaser_match=(got == cc['got']) if cc['got'] else None, how='item bench',
                     agreement=Player.carousel_agreement(cc, got, unit))
            self.news_event('carousel', cc['stage'], what)
            Player.carousel_mismatch(self, cc, got, 'item bench')
            return
        if carried and carried[0]:
            # read off its panel while we carried it: the unit that arrives with that name is it - no second click
            u = next((u for u in new if u.name == carried[0]), None)
            if u is None and time.time() - cc['at'] < 75:
                return
            if u is not None and carried[1]:
                u.items, u.items_read = [carried[1]], True
            self.carousel_confirm = None
            self.log(ev='carousel_confirmed', stage=cc['stage'], unit=carried[0], items=[carried[1]] if carried[1] else [],
                     chaser_got=cc['got'], went_for=cc['went'], match=Player.carousel_match(cc, carried[1]),
                     chaser_match=(carried[1] == cc['got']) if carried[1] and cc['got'] else None,
                     how='carried panel', on_bench=u is not None,
                     agreement=Player.carousel_agreement(cc, carried[1], carried[0]))
            Player.carousel_mismatch(self, cc, carried[1], 'carried panel')
            return
        # backup when the carried unit could not be read: look, don't wait: each new unit is clicked once and the item in its panel read - matched against what this
        # carousel offered, else named by its tooltip (the user, 2026-10-12; in run 20261010-235723 all 5 carousels
        # ended "could not be told": the unit came back, its panel item was never read, and Kayle with our Sparring
        # Gloves was sold as a plain unit)
        checked = cc.setdefault('checked', set())
        todo = sorted((u for u in new if u.name and not u.slotless and u.id not in checked and not u.items_read),
                      key=lambda u: (u.on_board, u.loc))
        if todo and not REPLAY and self.phase.planning and (scr.timer or 0) >= 4 and not cc.get('reading'):
            u = todo[0]
            checked.add(u.id)
            self.read_carousel_item(u, cc, scr)
        held = [u for u in new if u.name and u.items_read and len(u.items) == 1
                and getattr(u, 'panel_read', {}).get('name') == u.name]
        pick = None
        if len(held) == 1:
            pick = held[0]
        elif held:
            matching = [u for u in held if u.items[0] == (cc['got'] or cc['went'])]
            pick = matching[0] if len(matching) == 1 else None
        if pick is not None:
            item = pick.items[0] if pick.items[0] != 'unknown' else 'an item not recognised'
            ok = Player.carousel_match(cc, item)
            what = f"carousel: we got {pick.name} holding {item}"
        elif time.time() - cc['at'] > 75:                    # a planning phase and a fight have passed
            pick, item, ok = None, None, None
            what = "carousel: which unit it gave could not be told (no new unit holding an item was seen)"
            Player.record_carousel_inventory(self, cc, scr)
        else:
            return
        self.carousel_confirm = None
        if pick:
            Player.label_carousel_return(self, cc, pick, item, scr)
        print(f"{scr.stage} {what} (went for: {cc['went'] or 'not sure'})")
        self.log(ev='carousel_confirmed', stage=cc['stage'], unit=pick.name if pick else None, items=[item] if item else [],
                 chaser_got=cc['got'], went_for=cc['went'], match=ok,
                 chaser_match=(item == cc['got']) if item and cc['got'] else None,
                 agreement=Player.carousel_agreement(cc, item, pick.name if pick else None))
        self.news_event('carousel', cc['stage'], what)
        Player.carousel_mismatch(self, cc, item, 'unit panel')

    @staticmethod
    def carousel_target(cc):
        """The item the chase last went for: its last target, else the release choice (5-4 of game 83: the
        target was cleared as each pickup went unverified, went_for None)."""
        return cc.get('went') or (cc.get('release_choice') or {}).get('item')

    @staticmethod
    def carousel_match(cc, item):
        """What came back against what we went for (e260). Not against the chaser's own guess of what it got:
        game 83 4-4 went for Sparring Gloves, the chaser called Chain Vest ours, Chain Vest came back and the
        confirm logged match true (also 84 4-4, Belt -> Gloves). Unknown when either side is unknown."""
        went = Player.carousel_target(cc)
        if not item or not went or item == 'an item not recognised':
            return None
        return item == went

    def carousel_mismatch(self, cc, item, how):
        """A return that is not what the chase went for is logged as such, with the route that led to it (e260:
        83 4-4, 84 4-4, 85 3-4, 86 4-4 came back with another component than the one targeted)."""
        if Player.carousel_match(cc, item) is not False:
            return
        went = Player.carousel_target(cc)
        self.log(ev='carousel_mismatch', stage=cc['stage'], went_for=went, got=item, chaser_got=cc.get('got'),
                 how=how, release_choice=(cc.get('release_choice') or {}).get('item'),
                 first_contact=(cc.get('first_contact') or {}).get('item'))
        self.news_event('carousel', cc['stage'], f'carousel: went for {went}, came back with {item}')

    @staticmethod
    def carousel_agreement(cc, item, champion):
        """Compare an independent return to route/contact; missing evidence stays unknown."""
        out = {}
        for label in ('release_choice', 'first_contact'):
            seen = cc.get(label) or {}
            out[label] = dict(seen=seen or None,
                item_match=(seen['item'] == item if seen.get('item') and item else None),
                champion_match=(seen['champion'] == champion if seen.get('champion') and champion else None))
        return out

    def record_carousel_panel(self, unit, name, stars, scr, items=None):
        """Keep a fresh panel even when the return occupied a mis-tracked baseline slot.

        Game 64's Ahri panel occupied an own-bought Sentinel record from before the
        carousel. Membership/ownership flags must not hide that fresh panel.
        This records a witness, not a claim that the unit came from the carousel.
        """
        cc = getattr(self, 'carousel_confirm', None)
        if not cc or not name:
            return
        unit.panel_read = {'stage': scr.stage, 'name': name, 'stars': stars}
        if not hasattr(self, 'log_dir'):
            return
        key = unit.id, name, stars
        saved = cc.setdefault('saved_panels', set())
        if key in saved:
            return
        got = self.cap.read()
        if not got:
            return
        saved.add(key)
        tag = f'{cc["stage"]}-carousel-return-unit-{unit.id}-{name.replace(" ", "_")}-{stars}'
        self.snap(got[0], tag, True, force=True)
        self.log(ev='carousel_return_panel', stage=cc['stage'], unit_id=unit.id,
                 unit=name, stars=stars, items=list(items if items is not None else getattr(self, 'panel_items', [])),
                 frame=tag, candidate_type=('merge' if unit.id in cc.get('possible_merges', []) else
                                            'baseline' if unit.id in cc['before'] else 'arrival'))

    def record_carousel_inventory(self, cc, scr):
        """An unresolved return needs the complete inventory, not only filtered candidates."""
        frame = None
        if hasattr(self, 'log_dir'):
            got = self.cap.read()
            if got:
                frame = f'{cc["stage"]}-carousel-unresolved'
                self.snap(got[0], frame, True, force=True)
        self.log(ev='carousel_return_inventory', stage=cc['stage'], panel_stage=scr.stage, frame=frame,
                 before=sorted(cc['before']), before_stars=cc.get('before_stars', {}),
                 before_copies=cc.get('before_copies', []),
                 merges=self.roster.merge_history[cc.get('merge_index', 0):],
                 units=[{'id': u.id, 'name': u.name, 'stars': u.star, 'loc': u.loc, 'ours': u.ours,
                         'missing': u.missing, 'unverified': u.unverified,
                         'panel': getattr(u, 'panel_read', None)} for u in self.roster.units])

    def label_carousel_return(self, cc, unit, item, scr):
        """A returned unit's own panel supplies a carousel label even when the chase never
        confirmed a pickup. Do not turn an ambiguous loot arrival into a champion label."""
        panel = getattr(unit, 'panel_read', {})
        if panel.get('name') != unit.name:
            return
        done = self.__dict__.setdefault('_carried_read', {})
        if done.get(cc['stage']) and done[cc['stage']][0]:
            return
        done[cc['stage']] = (unit.name, item)
        self.log(ev='carousel_carried', stage=cc['stage'], unit=unit.name, stars=panel.get('stars'),
                 item=item, unit_id=unit.id, loc=unit.loc, how='returned unit panel',
                 panel_stage=panel['stage'], chaser_got=cc.get('got'))

    def carousel_bench_item(self, cc):
        """The one item that arrived on the item bench within 20 s after the carousel ended and was on its offer (any
        item when the offer was not read), or None."""
        track = self.__dict__.get('_item_track')
        arr = getattr(track, 'arrived', None) if track is not None else None
        if not arr or not (cc['at'] <= arr[0] <= cc['at'] + 20):
            return None
        offer = set(cc.get('offer') or ())
        names = [n for n in arr[1] if n and n != 'unknown' and (not offer or n in offer)]
        return names[0] if len(names) == 1 else None

    def read_carousel_item(self, u, cc, scr):
        """Click u, read the item in its panel: by its icon among the carousel's offer when that is clear, else by the
        tooltip the game shows while the item is hovered (read off the loop). Sets u.items when named; a unit whose
        panel shows no item is no carousel unit (a loot drop)."""
        from tft_player.items import PANEL_ITEMS, identify_among, learn_icon, panel_item_crops
        try:
            got = self.cap.read()
            bars = B.green_bars(got[0]) if got and u.on_board else None
            point = self.unit_point(u.loc, bars)
            name, stars, fresh = self.read_unit(point)
            got = self.cap.read()
        except NotForeground:
            return
        if not got or not name:
            cc['checked'].discard(u.id)              # no panel: tried again on a later tick
            return
        if not fresh or name != u.name or (stars is not None and stars != u.star):
            return                                 # an old/other panel cannot label this return
        Player.record_carousel_panel(self, u, name, stars, scr)
        crops = panel_item_crops(got[0])
        if crops is None:
            u.items, u.items_read = [], True        # holds nothing: not what the carousel gave
            return
        item, score, lead = identify_among(crops, cc.get('offer') or [])
        self.log(ev='carousel_item_icon', stage=scr.stage, unit=repr(u), item=item, score=round(score, 2),
                 lead=round(lead, 2), offer=cc.get('offer'))
        if item and score >= 0.75 and lead >= 0.2:
            u.items, u.items_read = [item], True
            return
        # not clear from the icon: hover the item and read the game's own tooltip
        px, py = PANEL_ITEMS[0]
        try:
            self.inp.hover((px - 200, py, 'R'))
            self.inp.hover((px, py, 'R'))
            time.sleep(0.4)                          # the tooltip fades in
            tip = self.cap.read()
            self.inp.hover((960, 540))
        except NotForeground:
            return
        if not tip:
            return
        frame, crop = tip[0], crops[len(crops) // 2]

        def read():
            try:
                from tft_player.screen import match_name
                h, w = frame.shape[:2]
                s = h / 1080
                # the tooltip opens beside the item, inside the screen: the right part of the screen around the panel
                x0, y0, x1, y1 = round(w - 820 * s), round(300 * s), w, round(1000 * s)
                texts = self.reader._detect(frame[y0:y1, x0:x1], 'panel_item_tip')
                found = [(ht, n) for _, _, ht, t in texts
                         if (n := match_name(re.sub(r'\s*\(.*$', '', t), self.reader.item_names(), 0.82))]
                name = max(found)[1] if found else None   # the title is the biggest item name in the tooltip
                self.log(ev='carousel_item_tooltip', stage=scr.stage, unit=repr(u), name=name,
                         texts=[t for *_, t in texts][:8])
                if name:
                    u.items, u.items_read = [name], True
                    learn_icon(crop, name)           # known by its picture from now on
            finally:
                cc['reading'] = False
        cc['reading'] = True
        threading.Thread(target=read, daemon=True).start()

    def read_encounter(self, frame, scr):
        """1-1: this game's encounter, as its banner says it - read off the loop, retried on later frames."""
        if scr.stage != '1-1' or self.encounter or self._enc_busy:
            return
        self._enc_busy = True

        def read(f=frame):                   # 2-5 s of OCR
            try:
                if not hasattr(self, '_enc_reader'):
                    self._enc_reader = ScreenReader()     # its own: OCR runs in this thread
                self.encounter = self._enc_reader.encounter(f)
                if self.encounter:
                    self.log(ev='encounter', text=self.encounter)
                    print(f'encounter: {self.encounter}')
            except Exception as e:
                self.log(ev='encounter_error', error=repr(e))
            finally:
                self._enc_busy = False
        threading.Thread(target=read, daemon=True).start()

    def paused_tick(self, frame, scr, paused):
        """F7 pause: the game is read and the board tracked up to here (stage, HP, merges, round changes), nothing
        is clicked and the model is not asked. On continue the plan from before the pause is dropped and the model
        gets the state as it is now: the round's first question again, told about the pause. -> True while paused."""
        was = getattr(self, '_paused_at', None)
        if paused:
            if was is None:
                self._paused_at = (time.time(), scr.stage)
                self.log(ev='paused', stage=scr.stage)
                self.live.event(scr.stage, 'paused', 'F7: paused - no input, no model questions')
            time.sleep(0.1)
            return True
        if was is not None:
            self._paused_at = None
            secs = round(time.time() - was[0])
            self.log(ev='resumed', stage=scr.stage, paused_at=was[1], seconds=secs)
            self.live.event(scr.stage, 'resumed', f'F7: continuing after {secs} s')
            self.round_plan, self.next_plan, self.plan_asked = None, None, None
            self.news_event('pause_over', scr.stage, f'the user paused the bot at {was[1]} for {secs} s (no input, no '
                                                 f'questions); the plan from before is dropped - this state is as '
                                                 f'the game is now')
            if not self.phase.planning:
                self.request_plan(scr, frame, 'plan')    # in a fight: ask now (planning asks on its first tick)
        return False

    def bench_located(self, scr, sell=None):
        """A bench action (E sell, a drag from or to the bench, an item onto a bench unit) only where the bench is: at
        home. In planning we are home; in a fight we may be on the opponent's arena, where the camera shows the bench
        elsewhere and its units have no bars to find them by (run 20261014-063145: six sells and two item drops landed
        on empty tiles). A sell (sell: the unit) is done there anyway: the unit is found by its info panel
        (away_bench_point) and E is pressed where it was read (the owner, 2026-11-08: "a lot of the sell actions are
        received during combat, sell during combat as well"). The other steps wait for the next planning phase.
        -> True when we are home, or the unit to sell was found."""
        ph = getattr(self, 'phase', None)
        if ph is None or ph.planning or REPLAY:
            return True
        # the fight's first seconds (units teleport, the camera may move) and its last one (they go home, a choice may
        # open): 2026-10-18 combat sells at +2.2 s (171815 4-5) and at timer 0 (171815 4-1, 161404 3-1) pressed E on
        # an empty bench; one 18 s later in the same fight sold
        if time.time() - getattr(self, 'round_started', 0.0) < 3.0 or (scr.timer is not None and scr.timer <= 1):
            self.fail_cause = 'the fight is starting or ending: the bench is used a moment later'
            self.deferred = self.retry_soon = True
            return False
        got = self.cap.read()
        sim = B.home_similarity(got[0]) if got else None
        if sim is None or sim >= B.HOME_MIN:
            if got is None or not self.bench_not_ours(got[0]):
                return True
            sim = None                           # an arena like ours, but our bench units are not on its bench
        if getattr(self, '_away_logged', None) != scr.stage:
            self._away_logged = scr.stage
            self.log(ev='away_fight', stage=scr.stage, similarity=None if sim is None else round(sim, 2))
        if sell is not None:
            pt = Player.away_bench_point(self, sell, scr)
            if pt is not None:
                self._away_spot = (sell.id, pt)
                return True
        self.fail_cause = 'an away fight: the bench is not where it is at home - done in the next planning phase'
        self.deferred = True
        rp = getattr(self, 'round_plan', None)
        if rp is not None and sell is not None:
            rp.away = scr.stage                  # its bench sells wait, the buys, XP and rolls behind them go on
        return False

    AWAY_DX = (0, 90, -90)   # our bench's x offset on other arenas (1080p): 0 (game 74b 2-5), +90 (run 20261014-063145 2-5)

    def away_bench_point(self, u, scr):
        """Where our bench unit `u` stands in a fight on another player's arena: its info panel is read at its slot's
        point shifted by each arena offset seen (AWAY_DX; the offset found first in this fight first) until the panel
        names it at its star. E is pressed there, so only where the unit was read (a wrong offset can point at a
        neighbour). -> the point, or None."""
        if REPLAY or u.loc[0] != 'bench' or not u.name:
            return None
        known = getattr(self, '_away_dx', None)
        dxs = [known[1]] if known and known[0] == scr.stage else []
        dxs += [d for d in self.AWAY_DX if d not in dxs]
        x, y = B.BENCH[u.loc[1]][0], self.BENCH_BAR_Y + 45
        reads = []
        for dx in dxs:
            pt = (x + dx, y)
            name, stars, _ = self.read_fresh(pt, [])
            reads.append([dx, name, stars])
            if name == u.name and (not stars or stars == u.star):
                self._away_dx = (scr.stage, dx)
                self.log(ev='away_bench_found', stage=scr.stage, unit=repr(u), dx=dx, reads=reads)
                return pt
        self.log(ev='away_bench_not_found', stage=scr.stage, unit=repr(u), reads=reads)
        return None

    def bench_not_ours(self, frame):
        """In a fight: every bench tile our tracked bench units stand on (two or more, each with its empty picture)
        looks empty - an opponent's arena with the same scenery as ours, or our units still teleporting. A bench
        action there would press on nothing."""
        from tft_player import bench as BN
        slots = {u.loc[1] for u in self.roster.bench() if not u.missing}
        kinds = BN.classify(frame, [False] * 9)
        seen = [kinds[i]['kind'] for i in slots if not self.on_tactician(i)
                and not (kinds[i]['kind'] == 'unknown' and kinds[i]['occ'] is None)]     # (no empty picture yet)
        return len(seen) >= 2 and all(k == 'empty' for k in seen)

    def bench_reference(self, frame, scr):
        """Our home arena's scenery, once, from a stage-1 planning phase (stage 1 is played at home)."""
        if B._HOME_REF[0] is None and not REPLAY and scr.stage and scr.stage.startswith('1-') and self.phase.planning \
                and scr.shop_visible and time.time() - getattr(self, 'round_started', time.time()) > 3.0:
            B._HOME_REF[0] = B.arena_sig(frame)
            self.log(ev='home_reference', stage=scr.stage)

    def standings_shown(self, frame, scr):
        """The player list, not the game's Damage Dealt panel, on the right: the carousel's release order, the HP
        standings in every prompt and players alive are read from it. Checked every 2 s in planning and carousels;
        an open panel is closed with the people button under its sword (a few tries per round)."""
        now = time.time()
        if REPLAY or now - getattr(self, '_standings_at', 0) < 2.0 or not (scr.shop_visible or is_carousel(scr)):
            return
        self._standings_at = now
        if not self.reader.damage_panel(frame):
            return
        tries = self._standings_tries = (self._standings_tries if getattr(self, '_standings_stage', None) == scr.stage
                                         else 0) + 1
        self._standings_stage = scr.stage
        self.log(ev='damage_panel', stage=scr.stage, try_=tries)
        if tries <= 3:
            self.inp.click(ScreenReader.STANDINGS_BUTTON)

    def handle_carousel(self, frame, scr):
        """Chase the unit carrying the item we want most."""
        if not getattr(self, '_carousel_pending', None):
            from tft_player.carousel_returns import baseline
            self._carousel_pending = {'stage': scr.stage, 'before': {u.id for u in self.roster.units},
                                      'before_stars': {u.id: u.star for u in self.roster.units},
                                      **baseline(self.roster, time.time())}
        self.carousel_stage = scr.stage
        now = time.time()
        t_cap = getattr(self, '_t_cap', None) or now
        self.chaser.choice_disabled = ('Hard Bargain' if 'Hard Bargain' in getattr(self, 'augments_taken', [])
                                       else None)
        # two reads: the offer for the model's ranking, then the route's read just before our gate opens -
        # in between nothing is read (the chase waits; the CPU stays free)
        rk = getattr(self, '_rank', None)
        waiting = (not self.chaser.champion_of and self.chaser.phase(now) == 'wait'
                   and rk is not None and rk.get('stage') == scr.stage
                   and rk.get('asked'))               # the first read went to the model
        # (no player bars, no 'carried' test: a carried unit leaves the ring, and the chase sees that - and the chase
        # ticked every 1.0-1.4 s in the games of 2026-10-10, walking to where its target had been)
        markers = [] if waiting else find_carousel_items(frame)
        markers = self.read_carousel_offer(frame, markers, scr)
        held = self._held = []
        # players' bars (~60 ms) only while the target is unseen: they tell a hidden target from a taken one
        bars = self._bars = find_bars(frame) if self.chaser.lost_since is not None else None
        # The carousel HUD counts down individual release waves (66-70).
        # Its current number establishes neither our release nor the end.
        self.chaser.ends_at = None
        # Re-read HP order each tick. The shrinking red portrait arcs still
        # identify low-HP living players, and the blue circle overrides timing.
        rt = release_timer(frame) if not self.reader.damage_panel(frame) else None
        if rt and scr.timer and 0 < scr.timer <= 36:
            if rt != getattr(self, '_release_rows', None):
                self._release_rows = rt
                self.log(ev='carousel_release', stage=scr.stage, timer=scr.timer,
                         release_timer=rt[0], row=rt[1], alive=rt[2], timer_scope='release wave')
        g = gate_ring(frame)
        self._carousel_gate_pixels = g
        old_release = self.chaser.release_at
        if self.chaser.observe_gate(g, t_cap):
            self.log(ev='carousel_gate_open', stage=scr.stage, timer=scr.timer, ring=g,
                     peak=self.chaser.gate_peak,
                     schedule_in=None if old_release is None else round(old_release - now, 1))
        if not self.chaser.choice_disabled:
            self.carousel_rank(markers, scr, now)
        tac = self.own.find(frame, now, hp=getattr(self, '_hp', None))   # our HP bar: how far we moved
        # everything above was seen in the frame captured at t_cap: the chase predicts from then, and its click lands
        # ~0.06 s after now
        click = self.chaser.next_click(markers, self.bench_items, self.roster, self.brain.plan,
                                       pos=tac, now=t_cap, held=held, bars=bars, lag=time.time() + 0.06 - t_cap)
        if click:
            if self.chaser.item and self.chaser.target_angle is not None and not getattr(self, '_announced', None) == scr.stage:
                print(f'{scr.stage} carousel: going for {self.chaser.item}')
                self.live.event(scr.stage, 'carousel', f'going for {self.chaser.item} '
                                f'(seen: {[m[2] for m in markers]})')
                self.log(ev='carousel', stage=scr.stage, want=self.chaser.item,
                         seen=[m[2] for m in markers])
                self._announced = scr.stage
                self.news_event('carousel', scr.stage,
                                f'the controller went for {self.chaser.item} (on offer: {[m[2] for m in markers]})')
            try:
                self.inp.click(click, 'right')
            except NotForeground:
                pass
        self.show_carousel(scr, markers, tac, click)
        self.record_carousel(frame, scr, markers, tac, click, now)
        self.read_carried(scr)

    def read_carousel_offer(self, frame, markers, scr):
        """Only the first offer's views have a carousel-held-out calibration."""
        if sum(bool(m[2]) for m in markers) < 5 or getattr(self, '_carousel_read_stage', None) == scr.stage:
            return markers
        from tft_player.carousel_champions import read_offer
        started = time.perf_counter()
        offer = read_offer(frame, markers)
        self._carousel_read_stage = scr.stage
        self.log(ev='carousel_champions', stage=scr.stage, reads=offer.champions, markers=list(offer),
                 seconds=round(time.perf_counter() - started, 3), scope='first_offer')
        return offer

    def read_carried(self, scr):
        """Once we hold a unit: click the unit we carry and read its panel - the champion and the item it brings (the
        user, 2026-10-12: "just click on the dragged component and you could see what it is"). Once per carousel."""
        from tft_player.carousel import UNIT_DY
        ch = self.chaser
        done = self.__dict__.setdefault('_carried_read', {})
        if not ch.got or scr.stage in done or REPLAY:
            return
        mine = sorted((math.dist(q, ch.pos), q) for q, it in ch.ring.off if it == ch.got and ch.pos is not None)
        if not mine or mine[0][0] > 120:
            return                                     # the carried unit's icon is not seen beside us yet
        q = ch.ring.screen(mine[0][1])
        done[scr.stage] = None
        # the eye icon over the floating item icon opens the unit's panel (the user, 2026-10-12: "once grab it,
        # click on the eye icon on top to confirm it"): 40 px above the item icon (run 20261012-071645 2-4); the
        # unit's body under the icon (right, then left button) if the eye opens none
        tries = [((q[0], q[1] - UNIT_DY), 'left'), (q, 'right'), (q, 'left')]
        from tft_player.items import identify_among, panel_item_crops
        try:
            got = self.cap.read()
            before = self.reader.panel_sig(got[0]) if got else None
            name = item = None
            for point, button in tries:
                self.inp.click(point, button)
                for _ in range(8):
                    time.sleep(0.08)
                    got = self.cap.read()
                    if not got:
                        continue
                    sig = self.reader.panel_sig(got[0])
                    if before is not None and sig.shape == before.shape and                             (np.abs(sig - before).max(axis=2) > 40).mean() < 0.05:
                        continue                       # no new panel yet
                    name, _ = self.reader.read_panel(got[0])
                    if name:
                        break
                if name:
                    crops = panel_item_crops(got[0])
                    offer = sorted(getattr(self, '_car_offer', {}).get(scr.stage, ()))
                    it, score, lead = identify_among(crops, offer) if crops else (None, -1.0, 0.0)
                    item = it if it and score >= 0.75 and lead >= 0.2 else None
                    self.log(ev='carousel_carried', stage=scr.stage, button=button, xy=point, unit=name,
                             item=item, icon=it,
                             score=round(score, 2), lead=round(lead, 2), chaser_got=ch.got)
                    self.save_panel_shot(f'{scr.stage}_carousel', f'{name}-{item or "unread"}')
                    break
            done[scr.stage] = (name, item)
            if not name:
                self.log(ev='carousel_carried', stage=scr.stage, unit=None, note='no panel opened on the carried unit')
        except NotForeground:
            done.pop(scr.stage, None)

    def carousel_champions_first(self):
        """The comp's champions for a same-item carousel (CarouselChaser.champ_first): the carry first, then the comp's
        final-board units the costliest first (the owner, game 79 5-4, x30.9: "1. bramble, 2. elder dragon")."""
        from tft_player.planner import _fixed_comp
        from tft_player import item_plan
        comp = _fixed_comp()
        ip = item_plan.plan_for(getattr(comp, 'name', None))
        carry = ip['carry'][0] if ip else None
        units = [u for u in (getattr(comp, 'units', None) or ()) if CATALOG.unit(u) and u != carry]
        return ([carry] if carry else []) + sorted(units, key=lambda u: -CATALOG.unit(u).cost)

    def carousel_rank(self, markers, scr, now):
        """The model ranks what this carousel offers (components early, completed items later),
        in the background: the chase starts on the targets it set before and switches to its
        ranking the moment the answer's first JSON object is written."""
        from tft_player.coach import _cli, ask_model
        from tft_player import carousel as CR
        st = getattr(self, '_rank', None)
        if st is None or st['stage'] != scr.stage:
            st = self._rank = {'stage': scr.stage, 'asked': False, 'ranked': None, 'applied': False, 'source': None,
                               'version': None, 'v1': False}
        # v1: the list the model prepared in the fight before - the WHOLE list, in its order, from the first
        # tick (the first view shows only part of the ring: run 221257 2-4 / 5-4 lost items cut from it)
        prep = getattr(self, 'carousel_prep', None)
        if not st['v1'] and prep and prep['stage'] == scr.stage:
            st['v1'] = True
            if not st['ranked']:
                st['ranked'], st['source'], st['version'] = list(prep['ranked']), 'prepared in the fight before', 'v1'
        # the comp's item plan (item_plan.wanted): what its main carry or main tank needs next is ranked first
        from tft_player import item_plan
        from tft_player.planner import _fixed_comp
        want = item_plan.wanted(getattr(self, 'roster', None), getattr(self, 'bench_items', None),
                                getattr(_fixed_comp(), 'name', None))
        if want and not st['ranked']:
            st['ranked'], st['source'], st['version'] = [], 'item plan', 'plan'
        if want and st['ranked'] is not None and not st['applied']:
            comp = _fixed_comp()
            units = set(getattr(comp, 'units', None) or ()) | set(
                (item_plan.plan_for(getattr(comp, 'name', None)) or {}).get('extra') or ())
            st['model_ranked'] = list(st['ranked'])
            st['item_ranked'] = [n for n, _ in want]
            st['ranked'] = item_plan.first(want, st['ranked'], units, live=st['version'] == 'v2')
        if st['ranked'] and not st['applied']:
            st['applied'] = True
            late = self.chaser.set_priority(st['ranked'], st['source']) is False
            print(f"{scr.stage} carousel: priority {st['version']} {st['ranked']} ({st['source']})")
            self.live.event(scr.stage, 'carousel', f"priority {st['version']}: {', '.join(st['ranked'])}")
            self.log(ev='carousel_ranking', stage=scr.stage, ranked=st['ranked'], source=st['source'],
                     model_ranked=st.get('model_ranked'), item_plan_ranked=st.get('item_ranked'),
                     version=st['version'], late=late, adopted=not late,
                     active_ranked=list(self.chaser.priority), active_source=getattr(self.chaser, 'source', None),
                     on_offer=[m[2] for m in markers])
        # v2: once the offer is seen, the model ranks EVERY item on it, seeing which champion carries
        # it - asked even when v1 covers the offer (run 152104 5-4: v1 had only Deathblade on offer;
        # once it was taken nothing ranked was left)
        if st['asked'] or not _cli()[0] or sum(1 for m in markers if m[2]) < 5:
            return
        st['asked'] = True
        offer = markers
        targets = list(self.chaser.priority or self.brain.plan.item_targets or [])
        state = (f"HP {getattr(self, '_hp', None)}, {self.roster!r}, item bench: "
                 f"{[i for i in (self.bench_items or []) if i] or 'empty'}")
        prompt = CR.priority_prompt(offer, state, targets)
        image = self.carousel_jpeg(markers)

        def run():
            t0 = time.time()

            def first(obj):
                r = CR.parse_priority(obj, offer)
                if r and st['version'] != 'v2':
                    st['latency'], st['ranked'], st['version'] = time.time() - t0, r, 'v2'
                    st['source'], st['applied'] = f'ranked on the offer in {time.time() - t0:.1f}s', False
            try:
                ans = ask_model(prompt, image, on_first=first, timeout=40)
                if st['version'] != 'v2':
                    from tft_player.coach import json_objects
                    for obj in json_objects(ans or ''):
                        first(obj)
                self.model_call('carousel', prompt, ans, time.time() - t0, st.get('latency'))
            except Exception as e:
                self.log(ev='carousel_rank_error', error=repr(e))
        threading.Thread(target=run, daemon=True).start()

    def record_carousel(self, frame, scr, markers, tac, click, now):
        """Every carousel tick: the frame and what the chaser saw and decided, for checking the
        real game against tools/sim_carousel.py (pickup range, release, speed, rotation)."""
        # the items this carousel offered: what the unit we bring back can hold (check_carousel_unit)
        self.__dict__.setdefault('_car_offer', {}).setdefault(scr.stage, set()).update(m[2] for m in markers if m[2])
        try:
            ch = self.chaser
            d = self.log_dir / 'carousel' / scr.stage
            d.mkdir(parents=True, exist_ok=True)
            n = getattr(self, '_car_n', 0) + 1
            self._car_n = n
            name = f'{n:05d}_{now:.2f}.jpg'
            if not hasattr(self, '_car_q'):                  # saving takes ~50ms: off the carousel loop
                import queue
                self._car_q = queue.Queue(maxsize=30)

                def saver():
                    while True:
                        path, img = self._car_q.get()
                        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                threading.Thread(target=saver, daemon=True).start()
            # One full-size frame per chase tick, written off the action thread. Contact can last under a
            # second: a 1 Hz sample could miss the exact failed pickup (owner e124).
            # Owner e124: the normal game had no frames at all, so neither the missed contact nor the
            # Sentinel's identity could be checked. Keep these even when test/video recording is off.
            if not self._car_q.full():
                self._car_saved = now
                self._car_q.put((str(d / name), frame))
            else:
                name = None
            self._car_live = time.time()
            # (not while the game video records: game.mp4 already holds the carousel, and this recorder takes frames
            # from the same capture stream as the chase - in video mode each read waits for a NEW frame, so the two
            # took turns. Run 20261010-235723: chase ticks 0.6-1.1 s apart, the recorder thread busy 90% meanwhile)
            if not getattr(self, '_car_rec_on', False) and hasattr(self.cap, 'cam') and not getattr(self, 'recorder', None) and test_mode():
                # the whole carousel at a steady 8 fps (the tick runs ~1.6/s): what happened between two ticks,
                # for replaying the chase (the user, 2026-10-09: record carousel rounds to fix them). Started again
                # after a pause in the ticks (run 010345 2-4: one 2.3 s gap ended it after 6 s, once per stage)
                self._car_rec_on = True
                raw_dir = d / 'raw'
                raw_dir.mkdir(exist_ok=True)

                def recorder():
                    while time.time() - getattr(self, '_car_live', 0) < 2.0:
                        t = time.time()
                        try:
                            f = self.cap.read_raw()     # (never cam.get_latest_frame: it takes frames from the loop)
                            if f is not None:
                                f = np.asarray(f)
                                h, w = f.shape[:2]
                                small = cv2.resize(f, (round(w * 540 / h), 540), interpolation=cv2.INTER_AREA)
                                cv2.imwrite(str(raw_dir / f'{t:.3f}.jpg'), cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                        except Exception:
                            pass
                        time.sleep(max(0.0, 0.125 - (time.time() - t)))
                    self._car_rec_on = False
                threading.Thread(target=recorder, daemon=True).start()
            r = lambda p: None if p is None else [round(float(p[0]), 1), round(float(p[1]), 1)]
            track = ch.track_of.get(ch.target_id)
            self.log(ev='carousel_tick', stage=scr.stage, frame=name, t=round(now, 3),
                     captured_at=getattr(self, '_t_cap', None), lag=round(ch.lag, 3),
                     target_id=ch.target_id, target_angle=ch.target_angle,
                     target_radius=getattr(ch, 'target_r', None), contact=ch.contact,
                     target_item_seen_at=ch.item_seen.get(ch.target_id),
                     first_contact=ch.first_contact, release_choice=ch.release_choice,
                     target_seen_at=track[2] if track else None,
                     markers=[[round(x, 1), round(y, 1), it] for x, y, it in markers], held=getattr(self, '_held', None),
                     tac=r(tac), color=self.own.color, click=r(click), item=ch.item,
                     target=r(ch.view()['target']), stepped=ch.stepped, contact_in=ch.contact_in,
                     ring=[round(float(v), 1) for v in (ch.ring.cx, ch.ring.cy, ch.ring.R)],
                     omega=None if ch.omega is None else round(ch.omega, 3), speed=round(ch.speed),
                     free=len(ch.units), taken=list(ch.taken), taken_why=getattr(ch, 'taken_why', []), got=ch.got,
                     unresolved=list(ch.unresolved),
                     pos=r(ch.pos_s), released=ch.released, feet_off=r(ch.off), lost=ch.lost_since is not None,
                     gate_closed=ch.gate_closed, gate_pixels=getattr(self, '_carousel_gate_pixels', None),
                     gate_peak=ch.gate_peak, gate_open_at=ch.gate_open_at, release_at=ch.release_at,
                     release_delay=ch.delay, priority=list(ch.priority), rank_source=ch.source,
                     choice_disabled=ch.choice_disabled, gate_confirmed=ch.gate_confirmed,
                     path=[r(p) for p in ch.path])
        except Exception as e:
            self.log(ev='carousel_record_error', error=repr(e))

    def show_carousel(self, scr, markers, tac, click):
        """For people watching: the decision panel shows the model's priority list as the chase works
        through it (taken, target, grabbed - frozen once grabbed); the comp and execution panels are hidden
        (the comp panel sits in the carousel's icon search); the target layer rings the unit we go for."""
        if not self.overlay:
            return
        ch = self.chaser
        view = ch.view()
        st = getattr(self, '_rank', None) or {}
        version = st.get('version') if st.get('stage') == scr.stage else None
        listed = list(ch.priority or [])
        rank = {it: stt for it, stt in view['ranking']}
        items = [(it, rank.get(it, 'free')) for it in listed if it in rank]
        if ch.item and ch.item not in listed and not ch.got:
            items.append((ch.item, 'target'))             # a unit the model did not rank: shown as such
        frozen = getattr(self, '_car_frozen', None)
        if ch.got and (frozen is None or frozen[0] != scr.stage):
            self._car_frozen = (scr.stage, items)
        elif frozen and frozen[0] == scr.stage and ch.got:
            items = frozen[1]
        note = ('grabbed ' + ch.got) if ch.got else ('not in the model\'s list' if ch.item and ch.item not in listed else '')
        phase = ('picked up' if ch.got else 'waiting for our gate' if ch.delay > 0.5 else
                 'stepping in' if ch.stepped else 'walking to it' if ch.item else 'reading the carousel')
        from tft_player.coach import forced_comp
        fc = forced_comp()
        # built directly (no parsing of the model's answers): this runs every chase tick
        top = {'stage': scr.stage, 'title': fc.name if fc else None, 'alerts': self.alert_lines(),
               'thinking': self.thinking_now(), 'took': self.last_took(),
               'carousel': {'version': version, 'list': items, 'note': note or phase}}
        view['markers'] = [(m[0], m[1]) for m in markers]
        if tac is None:
            view['pos'] = None
        self.overlay.show(top_left=top, top_right=None, bottom_left=None, path=view)

    def carousel_jpeg(self, markers):
        """The carousel for the model, each unit's item written beside its icon, so it sees which
        champion carries which item."""
        import io
        from PIL import Image, ImageDraw
        raw = getattr(self.cap, 'raw', None)
        if raw is None:
            return self.screen_jpeg()
        im = Image.fromarray(raw).convert('RGB')
        s = im.height / 1080
        d = ImageDraw.Draw(im)
        for x, y, it in markers:                     # markers are 1080p reference points (x centred on 960)
            if not it:
                continue
            tx, ty = im.width / 2 + (x - 960 + 22) * s, (y - 12) * s
            box = d.textbbox((tx, ty), it)
            d.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0))
            d.text((tx, ty), it, fill=(255, 230, 120))
        im.thumbnail((1568, 1568))
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=85)
        return buf.getvalue()

    def screen_jpeg(self):
        """The live screen for the model: native frame, 1568px on the long side."""
        import io
        from PIL import Image
        raw = getattr(self.cap, 'raw', None)
        if raw is None:
            got = self.cap.read()
            raw = got[0] if got else None
        if raw is None:
            return None
        im = Image.fromarray(raw)
        im.thumbnail((1568, 1568))
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=85)
        return buf.getvalue()

    def ground_point(self, orb, pos=None):
        """A right-click ON a unit opens its info panel instead of walking. Click
        the ground beside the orb (still inside pickup range), as far from our
        units as possible; with none free there, a point past the orb on the walk from the tactician (pos), so the
        walk runs over it (game 74b 4-1: the click went 48 px above an orb under Krug's bar, out of pickup range, and
        selected him)."""
        import math
        bodies = [(x - 65, y - 130, x + 65, y + 20)
                  for u in self.roster.units if not u.missing for x, y in [loc_point(u.loc)]]
        frame = getattr(self, 'last_frame', None)
        bodies += [(x - 65, y + 10, x + 65, y + 150) for x, y in B.green_bars(frame)] if frame is not None else []
        free = lambda p: all(not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1) for x0, y0, x1, y1 in bodies)
        for radius in (0, 28, 40):
            for k in range(8):
                angle = k * math.pi / 4
                point = (orb[0] + radius * math.cos(angle), orb[1] + radius * math.sin(angle) * 0.8)
                if free(point):
                    return point
        if pos is not None and math.dist(pos, orb) > 1:
            dx, dy = (orb[0] - pos[0]) / math.dist(pos, orb), (orb[1] - pos[1]) / math.dist(pos, orb)
            for beyond in (50, 80, 110, 140, 170):
                point = (orb[0] + dx * beyond, orb[1] + dy * beyond)
                if free(point):
                    return point
        for radius in (60, 90, 120):
            for k in range(8):
                angle = k * math.pi / 4
                point = (orb[0] + radius * math.cos(angle), orb[1] + radius * math.sin(angle) * 0.8)
                if all(not (x0 <= point[0] <= x1 and y0 <= point[1] <= y1) for x0, y0, x1, y1 in bodies):
                    return point
        return None

    # -- pilot mode: the model's judgement calls (augments, Wisps) ----------------------------------
    def pilot_state(self, scr):
        """The game as the model needs it for a judgement call."""
        u = lambda x: f"{x.name or '?'} {x.star}-star" + (f" [{', '.join(x.items)}]" if x.items else '')
        h = scr if scr.gold is not None else (self.hud_seen or scr)
        seen = '' if h is scr else f' (read at {h.stage}: the picker hides the HUD)'
        plan = self.pilot.last_plan
        econ = (f"level-8 roll-down planned for {plan['spike'][0]}-{plan['spike'][1]}; projected gold left for "
                f"rolling then: {plan['roll_gold']}; this comp's players roll ~{plan['roll_need']} gold at "
                f"level 8 (MetaTFT average)" if plan else 'no projection yet')
        st = self.pilot.streak
        items = [i for i in (self.bench_items or []) if i]
        return (f"Stage {scr.stage or self.phase.stage}. Gold {h.gold if h.gold is not None else '?'}, "
                f"level {h.level or '?'}" + (f" (xp {h.xp[0]}/{h.xp[1]})" if h.xp else '') + seen +
                f". HP {scr.hp if scr.hp is not None else '?'}. "
                f"Streak: {f'{st} wins' if st > 0 else f'{-st} losses' if st < 0 else 'none'}.\n"
                f"Economy: {econ}.\n"
                f"Board: {', '.join(u(x) for x in self.roster.board()) or 'empty'}\n"
                f"Bench: {', '.join(u(x) for x in self.roster.bench()) or 'empty'}\n"
                f"Item bench: {', '.join(items) or 'empty'}\n"
                f"Comp units not owned yet: {', '.join(self.pilot.missing(self.roster)) or 'none'}\n"
                + (f"This game's encounter (1-1 banner): {self.encounter}\n" if self.encounter else '') +
                f"Augments taken: {', '.join(self.augments_taken) or 'none'}")

    def pilot_ask(self, kind, about, prompt, image, done, **info):
        """One judgement call to the model in the background. done(decision) as soon as the
        decision is written (its "why" follows); done(None) when the call fails. The question,
        then the answer, show top-left."""
        from tft_player.coach import ask_model, json_objects
        call = dict(info, kind=kind, about=about, stage=self.phase.stage, t0=time.time(), answer=None, why=None)
        self.pilot_call = call

        def decided(obj):
            if isinstance(obj, dict) and call['answer'] is None and set(obj) - {'why'}:
                call['answer'], call['t1'] = obj, time.time()
                done(obj)

        def run():
            try:
                text = ask_model(prompt, image, on_first=decided) or ''
            except Exception as e:
                text = f'FAILED: {e}'
            objs = [o for o in json_objects(text) if isinstance(o, dict)]
            call['why'] = next((o['why'] for o in objs if o.get('why')), None) or (text[:300] if not objs else '')
            if call['answer'] is None:
                dec = next((o for o in objs if set(o) - {'why'}), None)
                call['answer'], call['t1'] = dec or {}, time.time()
                done(dec)
            self.model_call(kind, prompt, text, time.time() - call['t0'], call['t1'] - call['t0'])
            print(f"[model {kind} {call['t1'] - call['t0']:.0f}s] {call['answer']} - {str(call['why'])[:160]}")
        threading.Thread(target=run, daemon=True).start()

    def pilot_handle_augment(self, scr):
        """The augment picker in pilot mode: the cards are read off the screen, and the model,
        given their text, MetaTFT's tiers (a guide) and the game, takes one or rerolls. Only when
        it has no answer in time does the comp's tier ranking decide."""
        from tft_player.coach import _cli
        from tft_player.meta import META
        stage = scr.stage or self.phase.stage
        if self.reroll_stage != stage:
            self.reroll_stage, self.rerolled = stage, set()
        if time.time() - self.augment_clicked < 1.5:
            time.sleep(0.1)                      # picker closing / a rerolled card still flipping
            return
        rerollable = [i for i in range(3) if i not in self.rerolled]
        hand = (stage, tuple(sorted(self.rerolled)))
        if self.augment_seen != hand:
            self.augment_seen, self.aug_answer = hand, None
            self.augment_default, why = self.pilot_augment(rerollable)
            names = list(self.augment_read)
            print(f'{stage} augments {names} - tier-list fallback {self.augment_default}')
            self.log(ev='augment_options', stage=stage, options=names, default=self.augment_default, why=why)
            if _cli()[0]:
                from tft_player.coach import augment_cards
                lines = []
                for i, n in enumerate(names):
                    a = META.augments.get(n or '') or {}
                    lines.append(f"{i}: [MetaTFT tier for this comp: {self.pilot.augment_tiers.get(n, 'not rated')}; "
                                 f"over all comps: {a.get('tier') or 'not rated'}]")
                prompt = (f"DECISION: augment, stage {stage} (~{scr.timer or '?'}s left). "
                          + augment_cards(names, self.augment_raw) + '\n' + '\n'.join(lines) +
                          f"\nCards you can still reroll (each once): {rerollable or 'none'}.\n\n"
                          f"STATE\n{self.pilot_state(scr)}\n\n"
                          'Reply {"pick": <card>} or {"reroll": [<cards>]}, then {"why": "..."}.')

                def done(decision, hand=hand):
                    if self.augment_seen == hand:
                        self.aug_answer = decision or {}
                self.pilot_ask('augment', ' / '.join(n or '?' for n in names), prompt, self.screen_jpeg(), done,
                               names=names)
        ans = self.aug_answer
        c = self.pilot_call
        waited = time.time() - c['t0'] if c and c['kind'] == 'augment' else 99
        if ans is None and _cli()[0] and (scr.timer is None or scr.timer > 6) and waited < 45:
            time.sleep(0.15)                     # the model is deciding
            return
        ans = ans or {}
        reroll = ans.get('reroll')
        reroll = [reroll] if isinstance(reroll, int) else reroll if isinstance(reroll, list) else []
        reroll = [i for i in reroll if i in rerollable]
        if reroll and (scr.timer is None or scr.timer > 10):
            return self.reroll_augment(stage, self.augment_read, reroll, 'model')
        pick, src = ans.get('pick'), 'model'
        if not isinstance(pick, int) or not 0 <= pick <= 2:
            kind, i = self.augment_default
            pick = i if kind == 'pick' else \
                self.pilot.augment_decision(self.augment_read, [], self.pilot.gold_first(None, self.roster))[0][1]
            src = 'tier list (no model answer in time)' if _cli()[0] else 'tier list (no model)'
        from tft_player.coach import owner_banned_augments
        if self.augment_read[pick] in owner_banned_augments():
            self.log(ev='augment_banned', stage=stage, pick=pick, name=self.augment_read[pick], timer=scr.timer)
            self.augment_seen, self.aug_answer = None, None
            return                              # ask again with the owner mark; never substitute a card
        taken = self.augment_read[pick] or f'card {pick}'
        if src != 'model' and c and c['kind'] == 'augment' and c['answer'] is None:
            c['late'] = taken                    # an answer arriving now is shown as too late
        print(f'{stage} t={scr.timer}: augment {pick} {taken} ({src})')
        self.status_extra['augment'] = f'took {taken} ({src})'
        self.live.event(stage, 'augment', self.status_extra['augment'])
        self.log(ev='augment', stage=stage, options=self.augment_read, pick=pick, name=taken, source=src)
        self.augments_taken.append(taken)
        self.execute(Action('augment', pick, src), scr)
        self.augment_clicked = time.time()
        time.sleep(0.5)

    AUGMENT_GONE_S = 2.0     # s the augment screen stays closed before a pick that never came is reported
    AUGMENT_WAIT_S = 8.0     # s an augment round waits for its cards (15 games: they came 1.1-4.0 s into the round)
    AUG_FORCE_S = 3          # s left on the augment timer when the backup takes a card (the owner, x16.2)

    def augment_due(self, scr):
        """An augment round (2-1, 3-2, 4-2) whose augment is not picked yet, within AUGMENT_WAIT_S of its start: its
        cards are still to come up over the board."""
        from tft_player.coach import AUGMENT_ROUNDS
        stage = scr.stage or self.phase.stage
        if stage not in AUGMENT_ROUNDS or getattr(self, 'augment_done_stage', None) == stage:
            return False
        return time.time() - getattr(self, 'round_started', 0) < self.AUGMENT_WAIT_S

    def augment_missed(self, scr):
        """The augment screen closed without our pick (it came after the timer: benchmark game 20261018-123359 3-2
        and 4-2 answered with 11-12 s left in 10.5-10.7 s) - the game took a card on its own and nothing said which.
        The model is told, and its augment list says so, instead of carrying on without it. Which card the game took is
        not read; the state explicitly records that uncertainty."""
        if not getattr(self, 'augment_open', None):
            return
        now = time.time()
        if getattr(self, 'augment_gone', None) is None:
            self.augment_gone = now
            return
        if now - self.augment_gone < self.AUGMENT_GONE_S:
            return
        o, self.augment_open, self.augment_gone = self.augment_open, None, None
        self.augment_done_stage = o['stage']
        names = [n or '?' for n in o['names']]
        self.augments_taken.append(f"one of {' / '.join(names)} (the game's own pick)")
        self.log(ev='augment_missed', stage=o['stage'], options=names)
        self.news_event('augment_missed', o['stage'], f"your augment pick at {o['stage']} did not reach the game in "
                                                      f"time: the game took one of the cards on its own ({', '.join(names)}) "
                                                      f"- which one is not read")

    AUG_MARGIN = 4.0         # s around one augment answer: a rerolled card's flip, the screenshot, the click

    def answer_time(self):
        """Seconds this model takes to decide an augment: the slowest of this game's last three answers, else the
        pre-game speed check's (coach.SPEED), else None (not known: every step is offered)."""
        from tft_player import coach as C
        got = [t for t, _ in (getattr(getattr(self, 'coach', None), 'answers', None) or [])[-3:] if t]
        return max(got) if got else (C.SPEED.get('augment') or None)

    def aug_steps(self, timer):
        """Augment answers the choice's time left still fits; None when the time or the answer time is not known."""
        t = self.answer_time()
        if not t or timer is None:
            return None
        return int(max(0.0, timer - self.AUG_MARGIN) // t)

    def handle_augment(self, scr):
        """Local default first (meta tiers, ms): reroll weak cards at once, then ask
        the coach about the settled hand and take its pick if it arrives in time."""
        if self.pilot:
            return self.pilot_handle_augment(scr)
        names, p = scr.augments, self.brain.plan
        stage = scr.stage or self.phase.stage          # the HUD is hidden behind the picker
        if self.reroll_stage != stage:
            self.reroll_stage, self.rerolled = stage, set()
        if time.time() - self.augment_clicked < 1.5:
            time.sleep(0.1)             # picker closing / rerolled card still animating
            return
        if getattr(self, 'augment_done_stage', None) == stage:
            # this round's pick was clicked and the cards still show (game 66 2-1, 2026-11-06: the model's Investment
            # Strategy II was clicked twice, the cards still read 4 s later, and the backup took Upward Mobility): the
            # same card again - never another card, never the backup, never a second record
            again = getattr(self, 'augment_taken_idx', None)
            if again is not None and getattr(self, '_augment_again', 0) < 3:
                self._augment_again = getattr(self, '_augment_again', 0) + 1
                self.log(ev='augment_click_again', stage=stage, pick=again, timer=scr.timer)
                self.execute(Action('augment', again, 'the same card again'), scr)
                self.augment_clicked = time.time()
            p.augment_pick = p.augment_reroll = None
            time.sleep(0.1)
            return
        rerollable = [i for i in range(len(names)) if i not in self.rerolled]
        hand = (stage, tuple(sorted(self.rerolled)))   # a hand = the stage and the cards rerolled so far
        if self.augment_seen == hand:
            names = list(self.augment_read)
        if self.augment_seen != hand:
            if self.planner_on() and self.coach.busy:
                # still writing about the last hand (its "why"): ask about this one once it is free
                if (scr.timer or 0) > 5:
                    time.sleep(0.2)
                    return
            names = self.read_augment_hand(stage)
            self.augment_seen = hand
            self._augment_ban_retry = None
            self.augment_open = {'stage': stage, 'names': list(names)}
            p.augment_pick = p.augment_reroll = None
            if self.pilot:
                self.augment_default, why = self.pilot_augment(rerollable)
            else:
                self.augment_default, why = local_augment(names, self.roster, stage, rerollable, p)
            print(f'{stage} augments {names} - local {self.augment_default} ({why})')
            self.log(ev='augment_options', stage=stage, options=names, default=self.augment_default, why=why)
            # without a model, local rules may reroll; with one, the model decides
            if not self.planner_on() and self.augment_default[0] == 'reroll' and (scr.timer is None or scr.timer > 8):
                return self.reroll_augment(stage, names, [self.augment_default[1]], 'local')
            self.aug_wait, self.aug_reasks = self.coach.finished, 0
            # the owner (x16.2, game 65 3-2: "why not try to reroll and ask the model, and then use the force selection
            # as the backup?"): every reroll left is offered and every answer asked for - no "take a card now"; when no
            # pick has come with AUG_FORCE_S left, the backup takes one (below)
            self.aug_final = False
            self.coach.ask(scr, self.roster, augments=names, rerollable=rerollable,
                           image=self.screen_jpeg(), default=self.augment_default, note=self.augment_note,
                           held=list(self.augments_taken), raw=self.augment_raw)
            self.augment_note = None
        want = [i for i in (p.augment_reroll or []) if i in rerollable]
        bad = [i for i in (p.augment_reroll or []) if i not in rerollable]
        if bad:
            # each card rerolls once: a reroll asked for a card that used it is told to the model, never dropped silently
            self.augment_note = (f'Your last reply asked to reroll card {", ".join(map(str, bad))}, which has no reroll '
                                 f'left (each card rerolls once).')
            self.log(ev='augment_reroll_invalid', stage=stage, asked=p.augment_reroll, rerollable=rerollable)
            p.augment_reroll = want or None
        if want:
            # the owner (x16.2): the reroll is made and the model asked about the new cards; the backup pick covers an
            # answer that does not come in time
            return self.reroll_augment(stage, names, want, 'coach')
        if self.reject_banned_augment(scr, names, p.augment_pick, rerollable):
            return
        pick = p.augment_pick
        # the owner's backup (x16.2): with AUG_FORCE_S left and no pick from the model, a card is taken here
        force = scr.timer is not None and scr.timer <= self.AUG_FORCE_S
        if pick is None:
            # the model is still answering about this hand: wait for it
            if self.coach.busy and self.coach.pending_augment and not force:
                time.sleep(0.25)
                return
            if self.planner_on() and not self.pilot and not force:
                if not want and self.coach.finished > self.aug_wait and self.aug_reasks < 2 and \
                        (scr.timer is None or scr.timer > 8):
                    # its answer about this hand came and holds no pick or reroll that can be done: say why, ask again
                    note = self.augment_note or 'Your last reply held no "pick" or "reroll" that could be carried out.'
                    self.augment_note, p.augment_reroll = None, None
                    self.aug_wait, self.aug_reasks = self.coach.finished, self.aug_reasks + 1
                    self.log(ev='augment_reask', stage=stage, why=note)
                    final = getattr(self, 'aug_final', False)
                    self.coach.ask(scr, self.roster, augments=names, rerollable=[] if final else rerollable,
                                   image=self.screen_jpeg(), default=self.augment_default, note=note,
                                   timing=(scr.timer, self.answer_time()) if final else None,
                                   held=list(self.augments_taken), raw=self.augment_raw)
                    return
                # no fallback (the user): the model decides; if it never answers, the game's own
                # timer picks a card, as it would for a player who didn't click
                self.status_extra['augment'] = 'waiting for the model'
                time.sleep(0.25)
                return
            kind, i = self.augment_default
            if kind == 'pick':
                pick = i
            elif self.pilot:
                pick = self.pilot.augment_decision(self.augment_read, [], self.pilot.gold_first(None, self.roster))[0][1]
            else:
                pick = local_augment(names, self.roster, stage, [], p)[0][1]
            src = 'backup' if force and self.planner_on() else 'local'
            if src == 'backup':
                self.log(ev='augment_backup', stage=stage, timer=scr.timer, pick=pick, options=names)
        else:
            src = 'coach'
            # a reroll left on another card is free information: use it before taking the pick, and ask again with
            # the new card (the user, 2026-10-11: "the last reroll is always used if it is not picked")
            others = [i for i in rerollable if i != pick]
            # (only with time for the answer about the new card: else the pick is taken now - the user, 2026-10-20)
            fits = self.aug_steps(scr.timer)
            if others and not force and (scr.timer is None or scr.timer > 12) and (fits is None or fits >= 1):
                p.augment_pick = None
                self.log(ev='augment_reroll_rest', stage=stage, pick=pick, reroll=others)
                return self.reroll_augment(stage, names, others, 'unused rerolls before the pick')
        if self.reject_banned_augment(scr, names, pick, rerollable):
            return
        taken = (getattr(p, 'augment_name', None) if src == 'coach' else None) or \
            (self.augment_read[pick] if self.pilot and self.augment_read[pick] else names[pick])
        p.augment_name = None
        print(f'{stage} t={scr.timer}: augment {pick} {taken} ({src})')
        self.status_extra['augment'] = f'picked {taken} ({names[pick]}, {src})'
        self.live.event(stage, 'augment', self.status_extra['augment'])
        self.log(ev='augment', stage=stage, options=names, pick=pick, name=taken, source=src)
        self.unit_change(taken, stage)
        self.expect_gifts(taken, stage)
        self.augments_taken.append(taken)
        self.augment_open = None
        self.augment_done_stage, self.augment_taken_idx, self._augment_again = stage, pick, 0
        # an artifact augment's pick with its wanted artifacts (the owner, x16.1): its choice is taken from this list
        arts = getattr(p, 'augment_artifacts', None) if src == 'coach' else None
        p.augment_artifacts = None
        if arts:
            self.artifact_wish = list(arts)
            self.log(ev='artifact_wish', stage=stage, augment=taken, artifacts=arts)
        self.execute(Action('augment', pick, src), scr)
        self.augment_clicked = time.time()
        p.augment_pick = None
        self.aug_wait = self.coach.finished          # this answer is carried out: not one to ask again about
        if self.overlay and getattr(self, '_cards_shown', False):
            # the verdict boxes go with the pick, not at the next overlay refresh: what follows can block the loop
            # (run 20261016-052413 4-2: they stayed over Portable Forge's artifact row, "TAKING" over the first card)
            self.overlay.show(path=None)
            self._cards_shown = False
        time.sleep(0.5)

    def reject_banned_augment(self, scr, names, pick, rerollable):
        """Reject the owner's banned card and re-ask the same hand, including its rerolls."""
        from tft_player.coach import owner_banned_augments
        p = self.brain.plan
        if isinstance(pick, int) and 0 <= pick < len(names) and names[pick] in owner_banned_augments():
            self._augment_ban_retry = (f'Your pick of card {pick}, {names[pick]}, was rejected: '
                                       '[never picked in this comp: do not pick it; you may reroll this card]')
            self.log(ev='augment_banned', stage=scr.stage, pick=pick, name=names[pick], timer=scr.timer)
            p.augment_pick = p.augment_name = p.augment_reroll = None
        note = getattr(self, '_augment_ban_retry', None)
        if not note:
            return False
        if not self.coach.busy:
            asked = self.coach.ask(scr, self.roster, augments=names, rerollable=rerollable,
                                   image=self.screen_jpeg(), default=self.augment_default, note=note,
                                   held=list(self.augments_taken), raw=self.augment_raw)
            if asked:
                self._augment_ban_retry = None
                self.aug_wait = self.coach.finished
                self.aug_final = False
        return True

    def read_augment_hand(self, stage):
        """After the flip, require two consecutive matching OCR names per card.
        Keep a previously confirmed card only if its position has not been rerolled.
        Capital Gains I -> II once changed a fallback pick; a single read is not evidence.
        """
        if getattr(self, '_aug_known_stage', None) != stage:
            self._aug_known_stage, self._aug_known = stage, {}
            self._aug_seen_rerolled = set()
        for i in self.rerolled:
            if i in self._aug_known and i not in getattr(self, '_aug_seen_rerolled', set()):
                del self._aug_known[i]          # a new card sits there now
        self._aug_seen_rerolled = set(self.rerolled)
        names, previous, raw = [None] * 3, [None] * 3, [''] * 3
        for _ in range(4):
            time.sleep(0.4)              # a rerolled card flips for a moment
            got = self.cap.read()
            if got is None:
                continue
            current, raw = self.reader.read_augments(got[0])
            names = [n if n and n == previous[i] else None for i, n in enumerate(current)]
            previous = current
            if all(n or i in self._aug_known for i, n in enumerate(names)):
                break
        names = [n or self._aug_known.get(i) for i, n in enumerate(names)]
        self._aug_known.update({i: n for i, n in enumerate(names) if n})
        self.augment_read, self.augment_raw = names, raw
        return names

    def pilot_augment(self, rerollable):
        """Pilot mode: rank the same confirmed card names sent to the model."""
        names = self.read_augment_hand(self.reroll_stage)
        raw = self.augment_raw
        gold_first = self.pilot.gold_first(None, self.roster)
        decision, vals = self.pilot.augment_decision(names, rerollable, gold_first)
        return decision, (f'pilot tiers {list(zip(names, vals))}{" (behind on gold, no streak: gold augments first)" if gold_first else ""}'
                          f' (read {raw})')

    def reroll_augment(self, stage, names, cards, src):
        """Reroll one or several cards at once (the model can keep the best and
        reroll the rest in one go); it is asked again about the new hand."""
        print(f'{stage}: reroll augment cards {cards} ({[names[i] for i in cards]}) - {src}')
        self.log(ev='augment_reroll', stage=stage, options=names, cards=cards, source=src)
        self.brain.plan.augment_reroll = None
        for i in cards:
            self.rerolled.add(i)             # each card rerolls once, whether or not the click took
            try:
                self.inp.click(AUGMENT_REROLL[i])
            except NotForeground:
                return
            time.sleep(0.25)
        self.augment_clicked = time.time()

    def roll_arrival_sale(self, scr, spec=None):
        """Sell a confirmed, unwanted loot arrival while executing the player's search."""
        rp = self.round_plan
        if spec is None:
            if rp is None or rp.stage != scr.stage or not rp.roll_pending():
                return None
            spec = rp.p.get('roll_n') or rp.p.get('fast_rr') or rp.p.get('roll') or {}
        arrivals = [u for u in self.roster.bench() if getattr(u, 'arrival_source', None) == 'loot']
        if not arrivals:
            return None
        want = set(spec.get('buy') or ()) | set(self.auto_buy)
        keep = set(spec.get('keep') or ())
        if rp is not None:
            keep |= rp.protected_ids()
            want |= {e.get('name') for e in rp.p.get('board') or []}
            want |= {rp.unit(self.roster, uid).name for uid in keep if rp.unit(self.roster, uid)}
        free_items = sum(not it for it in (self.bench_items or []))
        for u in arrivals:
            if (u.name and CATALOG.unit(u.name)
                    and u.id not in keep and u.name not in want and not u.unverified and not u.missing
                    and u.star < 2 and len(u.items) <= free_items):
                return Action('sell', u, f'plan: unwanted loot arrival during roll-down ({u})')
        return None

    def run_rolldown(self, spec, scr):
        """A whole roll-down as one script (the user: the target is clear, no check per action):
            read the shop -> buy every wanted card at once (logged, no waiting) -> press D ->
            wait only until the 5 card strips have redrawn (pixels, ~0.1 ms per look) -> read the 5
            names (+ gold, one small OCR batch) -> repeat; stop at the stop condition or the floor.
        Gold is counted as we spend and re-synced from each shop read. ONE check at the end: the
        next planning read reconciles the bench with what we logged. Run26: 1.9 s per roll.
        spec['times']: reroll that many times at most (roll n times). spec['wisps']: 'auto' (fast roll-down):
        a Wisp is bought on the spot and its shop read again before the next roll; 'ask' (roll n times): a
        Wisp in a new shop stops the script and the model is asked (spec['stopped'] = 'wisp',
        spec['done'] = the rolls made); a Wisp click the game refuses means its Wisp limit is reached."""
        from tft_player import planner
        Player.default_wisp_flag(self, scr.stage)
        want, floor = set(spec.get('buy') or []), int(spec.get('floor') or 0)
        # copy targets: a listed champion is bought until board + bench hold its star level's copies
        # (given star, else its auto_buy star, else 2) - never every copy that shows up (run 152104)
        targets = {n: int((spec.get('targets') or {}).get(n) or self.auto_buy.get(n) or 2) for n in want}
        from tft_player.comp_rules import rules as comp_rules_of, upgrade_keep, trait_count
        comp_guard = bool(comp_rules_of().get('entry_hexes'))
        if comp_guard:
            targets = {n: min(star, comp_rules_of()['upgrade_stars'].get(n, star)) for n, star in targets.items()}
        from tft_player.comp_rules import eighth_units, optional_room
        comp_name = getattr(planner._fixed_comp(), 'name', None)
        cap = getattr(scr, 'team_size', None) or getattr(scr, 'level', None)
        if not optional_room(self.roster, comp_name, cap):
            optional = {n for n in want if targets[n] == 1 and n in eighth_units(comp_name, cap or 8)}
            want -= optional
            targets = {n: star for n, star in targets.items() if n not in optional}
        comp_final = set(getattr(planner._fixed_comp(), 'units', None) or ()) if comp_guard else set()
        refused = self.__dict__.setdefault('wisp_refused', {})
        stopped, skipped = None, []
        keep, stop = set(spec.get('keep') or ()), {'stop_when': spec.get('stop_when') or []}
        keep |= upgrade_keep(self.roster, targets)
        t_start = time.time()
        action_plan = getattr(self, 'round_plan', None)
        planning = getattr(getattr(self, 'phase', None), 'planning', not getattr(action_plan, 'combat', False))
        phase_end = t_start + max(0, scr.timer - 1) if planning and scr.timer is not None else float('inf')
        deadline = min(t_start + Player.BURST_SECONDS, phase_end)

        def yield_reason(frame=None):
            if action_plan is not None and getattr(self, 'round_plan', None) is not action_plan:
                return 'superseded'
            if time.time() >= deadline:
                return 'phase_end' if phase_end <= deadline else 'time_slice'
            clock = getattr(getattr(self, 'reader', None), 'read_clock', None)
            if planning and frame is not None and callable(clock):
                stage, timer = clock(frame)
                if stage and stage != scr.stage or timer is not None and (
                        timer <= 1 or scr.timer is not None and timer > scr.timer + 1):
                    return 'phase_end'
            return None

        rolls, bought, sold, wisps, missed = 0, [], [], 0, []
        flag_left = {}                           # (stage, card) -> what the Wisp flag did with a Wisp it did not buy
        shops = []                               # every shop seen: names as read, '-' = unread (a missed target shows here)
        gold = scr.gold if scr.gold is not None else getattr(self.round_plan, 'last_gold', None)

        # 0. sell first, when the model's plan says so (sell_others: fast_rr's default, a plain roll's not):
        # bench units outside the buy list, auto_buy, keep and the plan's board make room and gold before the
        # first roll (run 152104 4-1: a full bench skipped Sentinels while rolling on). Bench units only, so
        # this works in combat too.
        free_items = sum(1 for i in self.bench_items if not i) if self.bench_items is not None else 0
        for u in (list(self.roster.bench()) if spec.get('sell_others') or comp_guard and floor < 20 else []):
            if yield_reason():
                break
            if (u.id in keep or not u.name or u.unverified or u.missing or u.name in want
                    or self.auto_buy.get(u.name) or len(u.items) > free_items):
                continue
            if comp_guard and (u.items or u.name in comp_final
                               or (u.info and comp_rules_of()['trait'] in u.info.traits)):
                continue                         # only confirmed obsolete bodies; never strip a tank for roll gold
            value = Player.roll_sale(self, u)
            if value is None:
                continue
            sold.append(u.name)
            free_items -= len(u.items)
            if gold is not None:
                gold += value
            time.sleep(0.05)
        if sold:
            time.sleep(0.15)                     # let the gold and the shop settle before reading them

        def shop_now():
            got = self.cap.read()
            if got is None:
                return None, [None] * 5, [''] * 5, None
            g, names, raw = self.reader.read_shop(got[0])
            return g, names, raw, got[0]

        def stable_shop(timeout=1.0):
            """The shop after a roll, read from a frame whose five name strips stayed pixel-still over the
            SHOP_STILL frames before it: a card read mid-redraw is another card (the owner, 2026-10-30, game 50 6-2:
            "it rerolled too fast ... make it a bit slower so it could read more carefully" - 12 rolls in 11.5 s, and
            a unit bought as Elder Dragon was Nidalee). It was two text reads SHOP_SETTLE s apart that had to agree;
            unchanged pixels are the stricter test, and need no fixed wait and one text read (the owner, 2026-11-08:
            "make latency fast, as fast as possible")."""
            prev, still, end, got = None, 0, time.time() + timeout, None
            while time.time() < end:
                g_ = self.cap.read()
                if g_ is None:
                    continue
                got = g_
                sig = self.reader.shop_sig(got[0])
                still = still + 1 if prev is not None and all(
                    (np.abs(a_ - b_) > 40).mean() < 0.02 for a_, b_ in zip(sig, prev)) else 0
                prev = sig
                if still >= Player.SHOP_STILL:
                    break
            if got is None:
                return None, [None] * 5, [''] * 5, None
            g, names, raw = self.reader.read_shop(got[0])
            return g, names, raw, got[0]

        def strip_changed(a_, b_):
            return (np.abs(a_ - b_) > 40).mean() > 0.1

        def card_gone(i, sig0):
            return self.wait_card(i, sig0, 0.45, strip_changed)

        def redrawn(sig0, need, timeout=1.0):
            """Wait until at least `need` card strips differ from sig0 and the shop stands still."""
            prev, t0 = None, time.time()
            while time.time() - t0 < timeout:
                got = self.cap.read()            # (the next frame: its wait is the pace, no sleep)
                if got is None or sig0 is None:
                    continue
                sig = self.reader.shop_sig(got[0])
                changed = sum(1 for a_, b_ in zip(sig, sig0) if (np.abs(a_ - b_) > 40).mean() > 0.1)
                if changed >= need and prev is not None and all(
                        (np.abs(a_ - b_) > 40).mean() < 0.02 for a_, b_ in zip(sig, prev)):
                    return True
                prev = sig
            return False

        def show(now=''):
            # the roll-down runs as one script (no ticks): it pushes its own progress to the overlay
            if not self.overlay:
                return
            total = spec.get('times')
            steps = ([('done', 'SELL', 'Sold ' + ', '.join(sold), 'room first')] if sold else []) + [
                ('done' if now.startswith('done') else 'run', 'ROLL', f"Roll {rolls}{'/' + str(total) if total else ''} · gold {gold}", '')] +                 ([('done', 'BUY', 'Bought ' + ', '.join(bought[-6:]), '')] if bought else []) +                 ([('guard', 'BUY', ', '.join(skipped) + ' skipped', 'bench full - rolling stopped')] if skipped else [])
            # its rows join the round's list (one panel, no separate roll-down page - the user, 2026-10-10)
            self.roll_live = steps
            self.overlay.show(bottom_left=self.exec_panel(scr, True))

        g, names, raw, frame = shop_now()
        gold = g if g is not None else gold
        if gold is None:
            self.fail_cause = 'roll-down: gold could not be read'
            return False
        show('reading the shop')
        free = scr.free_rolls or 0               # free rerolls go first: they cost no gold, so no gold floor stops them
        # the comp's roll level (comp_rules.roll_level; the owner, game 74 x25.8: "set it as hard rule, no reroll at
        # level 7" for a level-8 comp): below it no reroll runs, free ones kept (they do not expire) - the last life's
        # roll-down (x24.7) is the one exception; the cards already in the shop are still bought
        from tft_player.comp_rules import roll_level
        level_now = getattr(scr, 'level', None) or getattr(self, 'last_level', None)
        low = roll_level()
        low = low if low and level_now and level_now < low and not spec.get('all_in') and not spec.get('one_roll')             else None                            # (the owner's one reroll for a Wisp - planner._wisp_reroll - is let through)
        from tft_player.comp_rules import low_hp_step
        e = low_hp_step(getattr(scr, 'hp', None) or getattr(self, 'last_hp', None), self.roster.board())
        if low and e and level_now >= e['level']:
            low = None                           # HP low without Riftbeast 7: the plan's rolls at level 7 (owner-hp-paths)
        # the plan lists its buys best first: a card further down leaves the gold the first listed champion still short
        # of its copies costs, unless it completes its own star level (run 20261022-064841 4-1: 13 gold went on
        # Cassiopeia x2 and a 1-star Elder Dragon, then Morgana - her 2-star - showed twice at 2 gold)
        order = [n for n in spec.get('buy') or [] if n in want]
        comp = planner._fixed_comp()
        carry = (comp_rules_of().get('carry') or (None,))[0] or next(iter(getattr(comp, 'carries', ()) or ()), None)
        if carry in order:
            order.remove(carry)
            order.insert(0, carry)               # owner e144: the comp's searched-for carry buys first

        def need(n):
            return self.auto_need(n, targets[n]) if n in targets else self.auto_need(n)

        def reserve(name):
            for n in order:
                if n == name:
                    break
                if need(n) > 0 and CATALOG.unit(n):
                    return CATALOG.unit(n).cost, n
            return 0, None
        rank = lambda n: order.index(n) if n in order else len(order)
        # the comp's final-board 5-costs none of which is owned, bought one copy each at level 8+ after the listed
        # champions (request e32; "skip_fives": true skips it) - game 42 4-3 rolled a Gnar away twice at 20 gold
        from tft_player.comp_rules import roll_fives
        fc = None if spec.get('skip_fives') else planner._fixed_comp()
        fives = roll_fives(getattr(scr, 'level', None) or getattr(self, 'last_level', None),
                           getattr(fc, 'units', None), self.roster.units) if fc else []
        # the comp's rule (comp_rules 'wisp_after_roll', request e50 - game 45 3-5: the level-7 roll stopped after 1 roll
        # for Stand Alone, the answer took 5 s and the rolls left ran at timer 2): a roll that asks about Wisps does not
        # stop for one in the middle; a Wisp it rolls past is told, and one in its last shop is asked when it ends
        after_roll = spec.get('wisps') == 'ask' and bool(comp_rules_of().get('wisp_after_roll')
                                                     or spec.get('wisps_after'))
        passed = []
        arrivals_read = set()
        split, t_top = [], time.time()
        while True:                              # stops only below: the model's conditions, or the game's
            stopped = yield_reason(frame)
            if stopped:
                break
            if frame is not None and callable(getattr(self, 'bench_lights', None)):
                lights = self.bench_lights(frame)
                if lights is not None:
                    self.adopt_unexplained(lights, scr, bars=B.bar_places(frame, exclude=self.watcher.tac[1])[0])
                # Read each new loot unit once in this script. Existing unknown objects cannot hold the search.
                arrivals = {u.id for u in self.roster.bench() if getattr(u, 'arrival_source', None) == 'loot'
                            and not u.name and not u.missing} - arrivals_read
                for uid in sorted(arrivals):
                    self.identify_one(scr, combat=getattr(self.round_plan, 'combat', False),
                                      arrivals=True, only_ids={uid})
                    arrivals_read.add(uid)
            while True:
                sale = Player.roll_arrival_sale(self, scr, spec)
                if sale is None:
                    break
                value = Player.roll_sale(self, sale.arg)
                if value is None:
                    break
                sold.append(sale.arg.name)
                gold += value
                self.log(ev='rolldown_arrival_sold', stage=scr.stage, unit=repr(sale.arg), value=value)
            shops.append([n or ('?' if r else '-') for n, r in zip(names, raw)])
            Player.observe_wisp_shop(self, types.SimpleNamespace(
                stage=scr.stage, shop_visible=True, shop=names, shop_raw=raw, gold=gold,
                timer=max(0, scr.timer - (time.time() - t_start)) if scr.timer is not None else None))
            wisp_first = False
            if planning and not refused.get(scr.stage):
                for i, (name, title) in enumerate(zip(names, raw)):
                    if not name and title and Player.flag_choice(self, title) == 'buy':
                        cost = self.reader.shop_cost(frame, i) if frame is not None else None
                        wisp_first = cost is None or cost <= gold
                        if wisp_first:
                            break
            # 1. buy the cards still short of their copy target in this shop - clicks only, logged as bought; the
            #    plan's first-listed champion first
            champions = [] if wisp_first else sorted(enumerate(names), key=lambda x: rank(x[1]))
            for i, name in champions:
                stopped = yield_reason()
                if stopped:
                    break
                info = CATALOG.unit(name) if name else None
                if info is None:
                    continue
                listed = name in want and self.auto_need(name, targets[name]) > 0
                if listed and spec.get('distinct'):  # the roll for the 7th: no more of the trait once it is owned
                    listed = trait_count(self.roster.units, spec['distinct'][0]) < spec['distinct'][1]
                # auto-buy's champions too, while short of their star target (no floor: no gold reserve)
                first_core = comp_guard and name in comp_rules_of()['entry_hexes'] and not any(
                    u.name == name and not u.missing for u in self.roster.units)
                auto = name not in want and (self.auto_need(name) > 0 or first_core) and info.cost <= gold
                five = not (listed or auto) and name in fives and self.auto_need(name, 1) > 0 and info.cost <= gold
                if not (listed or auto or five):
                    continue
                if not Player.optional_buy_ok(self, name, gold, Player.team_cap(self, scr),
                                              targets.get(name, self.auto_buy.get(name, 1))):
                    continue
                if info.cost > gold and listed and (need(name) == 1 or comp_guard):
                    gold += self.sell_for_last_copy(name, info.cost - gold, want, keep, sold, scr.stage)
                if info.cost > gold:
                    missed.append(f'{name} (costs {info.cost}, gold {gold})')
                    continue
                kept, top = reserve(name)
                if kept and gold - info.cost < kept and need(name) > 1:
                    missed.append(f'{name} (gold {gold} kept for {top}, listed first)')
                    continue
                # the floor stops the rolling, not the buying of a card rolled into (run 185805 6-3: Sentinel,
                # wanted for its 2-star, was passed at 32 gold with a floor of 30 and no word to the model)
                room_trait = spec.get('make_room') or (comp_rules_of().get('trait') if comp_guard else None)
                if (listed or five or first_core) and not self.roster.free_bench() and len(self.roster.copies(name, 1)) < 2:
                    g_ = Player.room_for_roll_target(self, name, want, keep, sold, scr.stage)
                    if g_ is not None:
                        gold += g_
                        got = self.cap.read()     # the sold slot must be absent from the buy's before-picture
                        frame = got[0] if got else None
                elif (room_trait and first_core and not self.roster.free_bench()
                        and len(self.roster.copies(name, 1)) < 2):
                    g_ = Player.make_room(self, room_trait, want, keep | upgrade_keep(self.roster, targets), sold)
                    if g_ is not None:
                        gold += g_
                if not self.roster.free_bench() and len(self.roster.copies(name, 1)) < 2:
                    skipped.append(name)         # bench full and no merge: the card cannot be bought
                    continue
                # confirmed before it is recorded and before any roll (the user: slower is fine): a click the
                # game took late was rolled away by the next D and still logged as bought (run 213127: 11
                # 'bench slots tracked != on screen after a buy'). A card still there gets one more click -
                # an empty slot ignores a click, so a late first click cannot buy twice
                sig_now = self.reader.shop_sig(frame) if frame is not None else None
                # the bench before the click: the slot that lights up is where this unit landed (a Wisp's or an
                # augment's unit that came meanwhile is already lit here, so it is not taken for ours)
                lights0 = self.bench_lights(frame) if callable(getattr(self, 'bench_lights', None)) else None
                self._landing_before = frame.copy() if frame is not None else None
                self._landing_guess = None       # legacy name, now set only by an observed landing
                if lights0 is not None:
                    # a unit that came since the last action (the Wisp just bought, an augment's delivery) is
                    # recorded now - unnamed, read off its panel - before our buy, as a normal buy does
                    self.adopt_unexplained(lights0, scr, bars=B.bar_places(frame, exclude=self.watcher.tac[1])[0]
                                           if frame is not None else None)
                took = False
                for attempt in range(2):
                    self.inp.click(B.SHOP_CLICK[i])
                    took = card_gone(i, sig_now)
                    if took:
                        break
                if not took:
                    missed.append(name)
                    continue
                if lights0 is not None:
                    slot = self.landing_slot(name, lights0, scr)
                    if slot is None:
                        slot = getattr(self, '_landing_guess', None)
                    self.note_landing(self.roster.bought(name, slot=slot, merge=not self._no_merge))
                    self.copies_doubted(name, scr)
                else:
                    self.roster.bought(name)
                bought.append(name)
                gold -= info.cost
                rp = self.round_plan
                if (rp is not None and not getattr(rp, 'combat', False)
                        and rp.arrival_pending(self.roster)):
                    stopped = 'field'
                    break                       # let the normal verified move executor field the hit now
                if five:
                    self.news_event('comp rule', scr.stage, f'{name} bought by the roll (final-board 5-cost) - one '
                                    f'copy while none is owned, at level 8+ ("skip_fives": true in execution_plan '
                                    f'turns it off for a plan)')
                got = self.cap.read()
                frame = got[0] if got else frame         # the next card's 'before' picture
            if stopped:
                break
            if skipped:
                stopped = 'bench_full'           # rolling on would only skip more wanted cards
                self.news_event('rolldown', scr.stage, f'roll-down stopped after {rolls} rolls: the bench is full '
                                f'and {", ".join(skipped)} could not be bought (no copy to merge with). Sell or '
                                f'field bench units, then roll again if you want to.')
                break
            # Wisps (a Wisp: text that is not a champion)
            mode = spec.get('wisps')
            owner_search = str(scr.stage).split('-')[0] in ('4', '5')
            if owner_search and mode in (False, None):
                mode = 'ask'
            wslots = [] if refused.get(scr.stage) else [i for i, (n, r) in enumerate(zip(names, raw)) if r and not n]
            # the player's Wisp flag (request e184): a Wisp it says buy or skip is bought or left here, in any roll and
            # at any time left, with no question (game 59 5-2 / 6-2: the roll-downs began at timer ~9, under WISP_LATE,
            # and rolled past Living Soil, Barrier, Hand Of Baron and Aftershock unasked)
            flag_buy = None
            for i in list(wslots):
                d = Player.flag_choice(self, raw[i])
                if d:
                    wslots.remove(i)
                if d == 'buy' and flag_buy is None:
                    flag_buy = i
                elif d == 'skip' and (scr.stage, raw[i]) not in flag_left:
                    flag_left[(scr.stage, raw[i])] = d
            unasked = [i for i in wslots if (scr.stage, i, raw[i]) not in getattr(self, 'wisp_q', {})]
            if flag_buy is not None or (wslots and (mode == 'auto' or (mode == 'first' and rolls > 0 and not wisps))):
                i = flag_buy if flag_buy is not None else wslots[0]
                cost = self.reader.shop_cost(frame, i) if frame is not None else None
                if flag_buy is not None and cost is not None and cost > gold:
                    flag_left[(scr.stage, raw[i])] = f'not bought: it costs {cost}, gold {gold}'
                elif cost is None or gold - cost >= min(floor, gold) or flag_buy is not None:
                    sig0 = self.reader.shop_sig(frame) if frame is not None else None
                    offer_id = Player.wisp_offer_id(self, (scr.stage, i, raw[i]))
                    gold_before_wisp = gold
                    Player.record_wisp(self, ev='wisp_roll_decision', stage=scr.stage, slot=i, wisp=raw[i],
                                       offer_id=offer_id, buy=True, cost=cost, gold=gold,
                                       policy='flag' if flag_buy is not None else mode)
                    self.inp.click(B.SHOP_CLICK[i])
                    if redrawn(sig0, 1):
                        wisps += 1
                        live = getattr(self, '_wisp_live', {}).get(i)
                        if live:
                            live['acquired'] = True
                        card = raw[i]
                        # the model hears which Wisp the roll bought and what it does (a Wisp can add champions:
                        # run 185805 4-2, a Malphite) - a unit it adds is recorded unnamed and read before the next buy
                        try:
                            from tft_player.screen import match_name
                            fx = self.wisp_effects()
                            wname = match_name(raw[i], list(fx), 0.75) if raw[i] else None
                        except Exception:
                            fx, wname = {}, None
                        self.news_event('rolldown_wisp', scr.stage, f"the roll-down bought the Wisp '{wname or raw[i] or '?'}'"
                                        f"{f' ({cost}g)' if cost is not None else ''}"
                                        f"{': ' + fx[wname] if wname in fx else ''}")
                        if flag_buy is not None:
                            self.news_event('wisp_flag', scr.stage, f"Wisp {Player.wisp_name(raw[i]) or raw[i]} bought "
                                                                    f"by your Wisp flag (in the roll)")
                        self.wisps_bought[scr.stage] = self.wisps_bought.get(scr.stage, 0) + 1
                        gold -= cost or 0
                        # its effect may have rerolled the shop; the champion card under the Wisp shows a moment after
                        # it goes - read the settled shop, and that slot until a card shows (the owner, game 79 4-5,
                        # x30.5: "the brambleback hides behind the wisp card, after it bought it, it rerolled, it should
                        # first check that position before reroll" - one frame read it empty)
                        g, names, raw, frame = stable_shop()
                        slot_end = time.time() + 0.8
                        while not (names[i] or raw[i]) and time.time() < slot_end:
                            g, names, raw, frame = stable_shop(0.4)
                        if g is not None:
                            gold = g
                        self.log(ev='rolldown_wisp', stage=scr.stage, slot=i, card=card, cost=cost,
                                 offer_id=offer_id, acquired=True, gold_before=gold_before_wisp, gold_after=gold,
                                 timer=max(0, scr.timer - (time.time() - t_start)) if scr.timer is not None else None)
                        continue                 # buy from this shop before rolling again
                    refused[scr.stage] = True    # the game took no more Wisps this round
                    if wisp_first:
                        continue                # buy champions from this shop after the refusal
            elif wslots and mode == 'ask' and after_roll and not owner_search:
                pass                             # the comp's rule: asked after the roll, if its last shop holds it
            elif unasked and mode == 'ask' and rolls > 0 and getattr(self, 'ask_in_roll', False) and (
                    owner_search or not self.wisp_late(scr.timer, time.time() - t_start)):
                stopped = 'wisp'                 # a Wisp in a new shop: the model decides, then the rest
                self.ask_wisp_now(scr, frame, unasked[0])
                break
            # (no question in a roll - the owner, game 74b 4-3: "when level up and reroll, there is no time to ask the
            # model every time what wisp to get ... the model could preset if combat or eco wisp acceptable": the Wisp
            # flag decides, a Wisp it does not cover is passed and told - ask_in_roll is off)
            if comp_guard and want and all(self.roster.proven_copies(n) >= 3 ** (targets[n] - 1) for n in want):
                stopped = 'targets_met'
                break
            # 2. stop? Only the model's own conditions (the user, 2026-10-09: "until 2-star X and 2-star Y, or until
            #    20 gold" is the only rule): its stop_when, its gold floor, its number of rolls
            if stop['stop_when'] and planner.RoundPlan.fast_rr_met(stop, self.roster):
                stopped = 'stop_when_met'
                break
            if spec.get('distinct'):                 # the comp's roll for the 7th (request e55): 7 distinct owned
                if trait_count(self.roster.units, spec['distinct'][0]) >= spec['distinct'][1]:
                    stopped = 'distinct_reached'
                    break
            if not free and gold - 2 < floor:
                stopped = 'gold_floor'
                break
            if comp_guard:
                remaining = [n for n in (want or set(self.auto_buy)) if CATALOG.unit(n) and need(n) > 0]
                if remaining:
                    price = max(CATALOG.unit(n).cost for n in remaining)
                    # The owner's explicit spending floor wins over the comp reserve.
                    if not free and gold - 2 < price and not spec.get('floor_set'):
                        stopped = 'kept_for_buy'
                        self.news_event('rolldown', scr.stage, f'roll stopped at {gold} gold: the next paid refresh '
                                        f'needs 2 + {price} purchase gold; that reserve stays in the bank')
                        break
                    if any(n in names and CATALOG.unit(n).cost > gold for n in remaining):
                        stopped = 'visible_target_unaffordable'
                        break                   # leave the target visible for a funded purchase
                    if not self.roster.free_bench() and not any(
                            len(self.roster.copies(n, 1)) >= 2 for n in remaining):
                        g_ = Player.make_room(self, comp_rules_of()['trait'], want,
                                              keep | upgrade_keep(self.roster, targets), sold)
                        if g_ is None:
                            stopped = 'bench_full'
                            break
                        gold += g_
            # a paid roll keeps the price of the first-listed champion still short: a shop it shows in must be one we
            # can buy it from (run 20261023-040039 4-5: rolls to 3 and to 0 gold for Morgana, listed first; she showed
            # in both last shops, one copy from her 2-star, and was not bought - the carry stayed 1-star to the end)
            # A floor the plan gave itself is the player's decision and wins (the climb's game 36 4-5: at 5 HP, "roll 3,
            # gold_floor 0" rolled 0 times, 5 gold kept for Brambleback's price - request e6)
            first = next((n for n in order if need(n) > 0 and CATALOG.unit(n)), None)
            if not free and first and gold - 2 < CATALOG.unit(first).cost and not spec.get('floor_set'):
                stopped = 'kept_for_buy'
                self.news_event('rolldown', scr.stage, f'roll-down stopped after {rolls} rolls at {gold} gold: one more '
                                f'roll would leave less than the {CATALOG.unit(first).cost} gold {first} (listed first) '
                                f'costs; the gold is kept to buy {first} when a shop shows it')
                break
            # a next_round roll-down (its floor written a round earlier, before this round's gold was known): no paid
            # roll that leaves less gold than every champion still wanted costs, when the bench cannot sell for the
            # difference either - a shop it rolls into can buy none of them (request e27, game 41 6-1: 8 gold, the
            # Wisp 3, rolls to 3 and to 1 with floor 0; Elder Dragon showed twice at 5, Brambleback's last copy 4)
            if not free and spec.get('ahead'):
                short_ = [n for n in order if need(n) > 0 and CATALOG.unit(n)]

                def reachable(n):
                    c = CATALOG.unit(n).cost
                    if gold - 2 >= c:
                        return True
                    return need(n) == 1 and gold - 2 + sum(planner.sell_value(u) for u in Player.last_copy_sellable(
                        self, n, want, keep)) >= c
                if short_ and not any(reachable(n) for n in short_):
                    stopped = 'no_target_affordable'
                    cheapest = min(short_, key=lambda n: CATALOG.unit(n).cost)
                    self.news_event('rolldown', scr.stage, f'roll-down stopped after {rolls} rolls at {gold} gold: one '
                                    f'more roll would leave {gold - 2}, less than any champion it buys still costs '
                                    f'({cheapest} {CATALOG.unit(cheapest).cost}g the cheapest), and the bench does not '
                                    f'sell for the difference (your next_round floor {floor} was set before this '
                                    f"round's gold was known)")
                    break
            # a free reroll that could find no card to buy (every champion still short costs more than the gold
            # left, or its gold is kept for the first-listed one) is kept: free rerolls do not expire - run
            # 20261022-064841 4-1: Epic Rolldown's free rerolls ran on at 2 gold, Morgana, Sentinel and two Dianas
            # seen and not bought
            def buyable(n):
                info_ = CATALOG.unit(n)
                if not info_ or need(n) <= 0 or info_.cost > gold:
                    return False
                return need(n) == 1 or gold - info_.cost >= reserve(n)[0]
            short_ = [n for n in want | set(self.auto_buy) if CATALOG.unit(n) and need(n) > 0]
            if free and short_ and not any(buyable(n) for n in short_):
                stopped = 'free_rolls_kept'
                self.news_event('rolldown', scr.stage, f'roll-down stopped after {rolls} rolls at {gold} gold: no '
                                f'champion still wanted is affordable; {free} free rerolls kept for later (they do not '
                                f'expire)')
                break
            if spec.get('times') is not None and rolls >= int(spec['times']):
                break
            # a shop that read no champion under an open "Choose One" row is not rolled: the row covers the cards, and
            # a roll throws them away unseen (run 20261104-102852 6-1, log 2251-2259: the anvil opened on bench slot 5,
            # then 4 rolls with every shop read '-', and the first one rolled away the Elder Dragon in slot 4).
            # The rolls left run once the choice is answered
            if not any(names) and frame is not None and Player.choice_up(self, frame):
                stopped = 'choice'
                self.log(ev='rolldown_choice_open', stage=scr.stage, rolls=rolls)
                break
            if low:
                stopped = 'roll_level'
                self.news_event('rolldown', scr.stage, f'roll-down stopped after {rolls} rolls: no reroll below level '
                                f'{low} for this comp, free ones included (level {level_now} now; the comp rule) - the '
                                f'cards in the shop were bought')
                break
            # 3. roll, and wait just until the new cards are drawn
            stopped = yield_reason(frame)
            if stopped:
                break
            sig0 = self.reader.shop_sig(frame) if frame is not None else None
            raw0 = list(raw)
            t_roll = time.time()
            if mode == 'ask' and rolls > 0 and not refused.get(scr.stage):     # a roll loses the shop's Wisps:
                passed += [raw[i] for i, (n, r) in enumerate(zip(names, raw))      # told and logged (the flag's
                           if r and not n and not Player.flag_choice(self, r)]     # are told below)
            self.inp.key('d')
            rolls += 1
            cost = 0 if free else 2
            free = max(0, free - 1)
            gold -= cost
            show('rolling')
            gold_before = gold + cost
            changed = redrawn(sig0, 3)
            t_drawn = time.time()
            g, names, raw, frame = stable_shop()
            # where each roll's time goes (s): the shop's buys and checks before it, the redraw, the still shop's read
            split.append((round(t_roll - t_top, 3), round(t_drawn - t_roll, 3), round(time.time() - t_drawn, 3)))
            t_top = time.time()
            if not changed:
                # a free reroll (an encounter, Invested++) spends no gold, so only the cards tell: the redraw can take
                # over 1 s on a busy machine (run 20261010-235723 2-3 / 3-2: rolls that changed the shop were taken
                # for refused ones). Wait a little more, then compare the card texts themselves
                changed = redrawn(sig0, 3, timeout=1.2)
                g, names, raw, frame = stable_shop()
                changed = changed or sum(1 for a_, b_ in zip(raw, raw0) if a_ and b_ and a_ != b_) >= 2
            if not changed and g is not None and g >= gold_before:
                # the game did not take the roll: D is pressed once more before the roll-down stops (game 73 4-5: one
                # roll of 49 gold's roll-down, then "the game did not take a roll" at 47 gold; 4-6 died with 57)
                time.sleep(0.3)
                self.inp.key('d')
                changed = redrawn(sig0, 3, timeout=1.2)
                g, names, raw, frame = stable_shop()
                changed = changed or sum(1 for a_, b_ in zip(raw, raw0) if a_ and b_ and a_ != b_) >= 2
                self.log(ev='roll_retried', stage=scr.stage, rolls=rolls, taken=bool(changed or g is not None and g < gold_before))
            if g is not None:
                gold = g                         # re-sync (part of the same small OCR batch)
            # the game did not take the roll (no new cards, no gold spent): it cannot roll now - stop and say so
            if not changed and g is not None and g >= gold_before:
                rolls -= 1
                stopped = 'roll_refused'
                self.news_event('rolldown', scr.stage, f'roll-down stopped after {rolls} rolls: the game did not '
                                f'take a roll (the shop did not change, gold stayed {g}).')
                break
            Player.observe_wisp_shop(self, types.SimpleNamespace(
                stage=scr.stage, shop_visible=True, shop=[None] * 5, shop_raw=raw0,
                gold=gold_before, timer=scr.timer), expire=True)
        took = time.time() - t_start
        for (_, card), d in flag_left.items():
            name = Player.wisp_name(card) or card
            self.news_event('wisp_flag', scr.stage, f'Wisp {name} skipped by your Wisp flag (in the roll)' if d == 'skip'
                            else f'Wisp {name} {d} (your Wisp flag: buy; in the roll)')
        if passed and not after_roll:                    # rolled past late in the phase (WISP_LATE): lost unanswered
            self.news_event('rolldown', scr.stage, f"the roll rolled past the Wisp(s) {', '.join(passed)} unanswered "
                            f"(too little time left to ask); a rolled shop loses its Wisps")
        if after_roll and stopped != 'wisp':
            last = [] if refused.get(scr.stage) or not rolls else [i for i, (n, r) in enumerate(zip(names, raw)) if r and not n]
            if passed:
                self.news_event('comp rule', scr.stage, f"the roll rolled past the Wisp(s) {', '.join(passed)} without "
                                f"stopping (a Wisp is answered after the roll, never in the middle of it)"
                                + (' - the Wisp in its last shop is asked now' if last else ''))
            if last and frame is not None:
                self.ask_wisp_now(scr, frame, last[0])
        spec['done'], spec['stopped'] = rolls, stopped
        if spec.get('seventh'):
            sv = spec['seventh']
            found = [n for n in bought if n in sv['buy']]
            now = trait_count(self.roster.units, sv['trait'])
            self.news_event('comp rule', scr.stage, f"rolled {rolls} times for the {sv['want']}th {sv['trait']} "
                            f"({', '.join(found) + ' found' if found else 'not found'}; {sv['trait']} champions owned "
                            f"{sv['have']} -> {now}), gold {sv.get('gold')} -> {gold}"
                            + (f", sold {', '.join(sold)} for room" if sold else '')
                            + ' - the plan had no roll at level 7+ with the 7th missing ("hold_gold": true in '
                              'execution_plan skips it for a plan)')
        show(f'done: {rolls} rolls' + (f', stopped ({stopped.replace("_", " ")})' if stopped else ''))
        self.roll_live = None                             # the finished roll shows as its round_acts row
        # the gold the last roll needed and the bench outside the comp, next to what was missed (request e14)
        money = Player.rolldown_money(self, gold)
        spec['summary'] = (f'{rolls} rolls, bought {", ".join(bought[-6:]) or "nothing"}'
                           + (f', stopped: {stopped.replace("_", " ")}' if stopped else '') + money)
        if skipped:
            self.alert('SKIP', f'{", ".join(skipped)} not bought: bench full', 'Execution')
        if missed:                                       # a wanted card was in a shop and not bought: the model hears it
            self.news_event('rolldown', scr.stage, f'roll-down: wanted cards seen but not bought: {", ".join(missed)}'
                            + money)
        self.log(ev='rolldown', stage=scr.stage, rolls=rolls, bought=bought, sold=sold, wisps=wisps, shops=shops,
                 want=sorted(want), targets=targets, skipped=skipped, missed=missed, floor=floor, gold_end=gold,
                 seconds=round(took, 1), per_roll=round(took / max(1, rolls), 2), stopped=stopped,
                 wisps_passed=passed, split=split)
        print(f'{scr.stage}: roll-down - {rolls} rolls in {took:.1f}s, bought {bought or "nothing"}'
              + (f', sold {sold}' if sold else '') + (f', stopped: {stopped}' if stopped else '')
              + (f' (skipped {skipped})' if skipped else ''))
        self.buy_check = time.time() + 1.5      # the ONE check: bench slots vs the log, next planning read
        return True

    def room_for_roll_target(self, name, want, keep, sold, stage):
        """Owner e144: a visible search hit makes room before another roll, even in combat.
        Prefer an itemless bench unit on none of the comp's boards; otherwise the cheapest eligible
        bench unit outside the plan's board. Identity, explicit keeps and item capacity still apply.
        """
        from tft_player import planner
        comp = planner._fixed_comp()
        comp_units = set(getattr(comp, 'units', ()) or ())
        guide = getattr(comp, 'guide', {}) or {}
        for section in ('early', 'final'):
            for boards in (guide.get(section) or {}).values():
                for board in boards:
                    comp_units.update(board.get('units') or ())
        rp = self.round_plan
        board = (rp.p.get('board') or []) if rp else []
        protected = set(keep) | {e['id'] for e in board if isinstance(e.get('id'), int)}
        board_names = {e.get('name') for e in board if e.get('name')}
        free_items = sum(not it for it in (self.bench_items or ()))
        eligible = [u for u in self.roster.bench() if u.info and not u.missing and not u.unverified
                    and u.id not in protected and u.name not in board_names and u.name not in want
                    and len(u.items) <= free_items]
        ambiguous = [u for u in eligible if sum(v.loc == u.loc for v in self.roster.bench()) > 1]
        eligible = [u for u in eligible if u not in ambiguous]
        preferred = [u for u in eligible if u.name not in comp_units and not any(u.items)]
        if not (preferred or eligible):
            if ambiguous:
                self.need_survey = 'multiple owned records name the same sale source slot'
                self.survey_places = (self.need_survey, list(dict.fromkeys(u.loc for u in ambiguous)))
                self._repair_pending = True
            return None
        unit = min(preferred or eligible, key=lambda u: (planner.sell_value(u), u.loc))
        value = Player.roll_sale(self, unit, verify=True)
        if value is None:
            return None
        sold.append(unit.name)
        self.log(ev='rolldown_room_for_target', stage=stage, want=name, sold=repr(unit), value=value,
                 preference='itemless outside comp boards' if preferred else 'lowest sale value outside plan board')
        return value

    def make_room(self, trait, want, keep, sold):
        """A full bench in a roll that may sell for room (spec 'make_room': the comp's roll for the 7th, request e55):
        the cheapest benched unit with no item, outside `trait`, not wanted by the roll, kept by the plan or in doubt
        is sold. -> the gold it brought, or None (no such unit; trait None: the roll may not sell)."""
        if not trait:
            return None
        from tft_player import planner
        final = set(getattr(planner._fixed_comp(), 'units', None) or ())
        cands = [u for u in self.roster.bench() if u.name and not u.missing and not u.unverified and not any(u.items)
                 and u.id not in keep and u.name not in want and not self.auto_buy.get(u.name)
                 and u.name not in final and u.info and trait not in u.info.traits]
        if not cands:
            return None
        u = min(cands, key=lambda x: (x.cost * 3 ** ((x.star or 1) - 1), x.loc))
        value = Player.roll_sale(self, u)
        if value is None:
            return None
        sold.append(u.name)
        time.sleep(0.05)
        return value

    def roll_sale(self, unit, verify=False):
        """A comp rolldown spends sale proceeds only after the gold increase is observed (e57)."""
        from tft_player.comp_rules import rules
        from tft_player.planner import sell_value
        if not Player.confirm_before_loss(self, unit, types.SimpleNamespace(
                stage=getattr(self, 'stage_seen', None), timer=None), 'sell'):
            return None
        if verify or rules().get('entry_hexes'):
            got = self.cap.read()
            gold = self.reader.read_shop(got[0])[0] if got else None
            if gold is None:
                return None
            Player.sell_key(self, loc_point(unit.loc))
            after = Player.sell_seen(self, gold, timeout=0.6)
            if after is False:
                self.log(ev='rolldown_sell_failed', unit=repr(unit), gold=gold, roster=repr(self.roster))
                if got and hasattr(self, 'snap'):
                    self.snap(got[0], f'rolldown-sell-failed-{unit.id}', True)
                return None
            value = after - gold
        else:
            Player.sell_key(self, loc_point(unit.loc))
            value = sell_value(unit)
        self.roster.sold(unit)
        return value

    def sell_key(self, point):
        """E over `point`, after the cursor has rested there SELL_HOVER s: the game sells the unit it has under the
        cursor, and picks it a frame or two after the cursor arrives (execute's sell does the same: run 20261013-220557
        4-5). Run 20261104-102852: the roll-down pressed E on the bench 0.03 s after moving there, right after an item
        drag onto a board unit, and the game sold that unit - Gnar at 5-5 (log 2011-2022, E on Vi at bench slot 1),
        Taric at 6-5 (log 2631-2641, E on Ezreal at slot 2)."""
        move = getattr(self.inp, 'move', None)           # (Input has it; a recorded-input stand-in may not)
        if not REPLAY and move is not None:
            move(point)
            time.sleep(SELL_HOVER)
        self.inp.key('e', at=point)

    def sell_for_last_copy(self, name, short, want, keep, sold, stage=None):
        """The card that completes a listed champion's star level shows and the gold is `short`: bench units sell for
        it, the fewest gold over what is short first (request e20; game 40 4-6 and 5-1: the third Brambleback was
        rolled past at 2, 0 and 3 gold while a benched Cinderling 2* sold for 2 and a Mama Beak 2* for 8).
        -> the gold the sales add (0 when they cannot cover it: nothing is sold)."""
        from tft_player import planner
        ok = Player.last_copy_sellable(self, name, want, keep)
        # units auto_buy does not want back first: a sold one it wants would be bought again
        pick = next(([u] for u in ok if planner.sell_value(u) >= short and not self.auto_buy.get(u.name)),
                    None) or next(([u] for u in ok if planner.sell_value(u) >= short), None)
        if pick is None:                         # no one unit covers it: the cheapest ones, while they add up
            pick, got = [], 0
            for u in ok:
                pick.append(u)
                got += planner.sell_value(u)
                if got >= short:
                    break
            if got < short:
                return 0
        return Player._sell_picked(self, pick, sold, stage, name, short)

    def last_copy_sellable(self, name, want, keep):
        """The bench units sell_for_last_copy may sell for `name`'s last copy, cheapest first: never a unit the plan
        fields or keeps, one of the roll's buy list, one with items the item bench cannot take, nor - while the board is
        below the comp's step (comp_rules: Riftbeast 7) - the only copy of a champion of its trait."""
        from tft_player import planner
        from tft_player.comp_rules import rules
        from tft_player.state import board_traits
        r = rules()
        t, step = r.get('trait'), r.get('keep')
        short_of_step = bool(t and step and board_traits(self.roster.board())[t] < step)
        rp = self.round_plan
        guard = set(keep) | (rp.protected_ids() if rp else set())
        from tft_player.comp_rules import plan_targets, upgrade_keep
        guard |= upgrade_keep(self.roster, plan_targets(getattr(rp, 'p', None)))
        final = set(getattr(planner._fixed_comp(), 'units', ()) or ())
        free_items = sum(1 for i in self.bench_items if not i) if self.bench_items is not None else 0
        ok = []
        for u in self.roster.bench():
            if (u.id in guard or not u.name or u.unverified or u.missing or u.name in want or u.name == name
                    or u.name in final
                    or len(u.items) > free_items):
                continue
            if short_of_step and t in (u.info.traits if u.info else ()) and \
                    not any(x.name == u.name for x in self.roster.units if x is not u):
                continue
            ok.append(u)
        return sorted(ok, key=planner.sell_value)

    def _sell_picked(self, pick, sold, stage, name, short):
        from tft_player import planner
        got, confirmed = 0, []
        for u in pick:
            value = Player.roll_sale(self, u)
            if value is None:
                break
            sold.append(u.name)
            confirmed.append(u)
            got += value
            time.sleep(0.05)
        if not confirmed:
            return 0
        what = ', '.join(u.name + '*' * u.star for u in confirmed)
        self.news_event('rolldown', stage, f'roll-down: visible target {name}, gold was {short} '
                                           f'short: sold {what} ({got}g) to buy it')
        self.log(ev='rolldown_sell_for_copy', stage=stage, want=name, short=short, sold=[u.name for u in confirmed], gold=got)
        return got

    def rolldown_money(self, gold):
        """For a roll-down's summary: the gold left, each target unit's gold for its last copy after a roll (cost + 2)
        and what the bench units outside the target comp sell for (planner.roll_gold_lines' numbers)."""
        from tft_player import planner
        try:
            need = [(n, c, h) for n, c, h in planner.missing_units(self.roster, self.brain.plan) if h > 0]
            worth, units = planner.outside_bench(self.roster, self.brain.plan)
        except Exception:                                # a summary never fails on its arithmetic
            return ''
        out = f'; gold left {gold}' if gold is not None else ''
        if need:
            out += '; a last copy after a roll needs ' + ', '.join(
                f'{n} {c + 2}g ({h} of 3 copies held)' for n, c, h in need)
        if units:
            out += (f'; bench units outside the target comp sell for {worth}g ('
                    + ', '.join(f'[id {u.id}] {u.name}{"*" * u.star} {planner.sell_value(u)}g' for u in units) + ')')
        return out

    def wait_for(self, cond, timeout=2.0):
        """Poll fresh frames until cond(screen) holds; the HUD lags an input by a few frames and one
        read takes ~0.8s, so the default leaves time for two reads."""
        end = time.time() + timeout
        while time.time() < end:
            _, after = self.read()               # (the next frame: its wait is the pace, no sleep)
            if after is not None and cond(after):
                return True
        return False

    # -- bench lights: one per slot, on while a unit's health bar is there ------------------
    def bench_lights(self, frame=None):
        """Per bench slot: True = something is there (the tile differs from its empty image),
        False = empty, None = no evidence (covered by a banner or panel, or a container)."""
        from tft_player import bench as BN
        if frame is None:
            got = self.cap.read()
            if got is None:
                return None
            frame = got[0]
        refs = BN.empty_refs()
        # a slot with our unit's health bar is lit without the container match (~10 ms a slot in a game: the bench
        # lights are looked at on every frame of a buy's landing and of each roll)
        bars = B.bar_places(frame, exclude=self.watcher.tac[1])[0] if not REPLAY else [False] * 9
        # a slot whose bare tile is not learned yet reads 'unknown' whatever is on it: no evidence (an arrival
        # recorded there is a '?' that blocks the slot's own learning)
        lights = [None if i not in refs else True if o['kind'] in ('unit', 'unknown') else False if o['kind'] == 'empty'
                  else None for i, o in enumerate(BN.classify(frame, bars))]
        # our tactician standing on a slot fills its tile too: no evidence there (run 20261005-022645
        # 3-3: it stood on slot 0 and was read as an arriving unit 8 times)
        return [None if self.on_tactician(i) else x for i, x in enumerate(lights)]

    def on_tactician(self, i):
        tac_box = self.watcher.tac[1]
        return tac_box is not None and tac_box[3] > 600 and tac_box[0] - 60 <= B.BENCH[i][0] <= tac_box[2] + 60

    def learn_bench_baseline(self, frame, scr):
        """Each bench slot's 'off' light for this game's arena (no pictures are stored: the arena changes
        the tiles). Stage 1-2 planning, before the first shop: every bench slot without a bar is empty. Later - the player started
        mid-game, or a slot was busy at 1-2 - a slot is taken once it is visibly free. A slot counts when
        it has no health bar, nothing tracked, no tactician on it, and looks the same as the read before
        (nothing passing through; after 1-2: for 2 s), and holds no container and no tooltip or panel."""
        from tft_player import bench as BN
        got = self.__dict__.setdefault('_base_got', set())
        # not 1-1: the game opens on the lobby-wide intro view (all eight tacticians on one map), and
        # its ground taken as 'empty' lit 5 bench slots as units for the whole of both games of 2026-10-10
        if len(got) == 9 or not scr.stage or scr.stage == '1-1':
            return
        later = scr.stage != '1-2'
        if not later and not self.phase.planning:
            return                       # bench units show no health bar in combat: a unit there passes for the bare
                                         # tile (run 20261026-002447: the opening units on slots 0-4 were learned as
                                         # 'empty' in the 1-2 fight; those slots read 'unknown' all game, three '?'
                                         # units no read could name filled them and every buy was tracked one slot off)
        if later:
            if not self.phase.planning or not scr.shop_visible or time.time() - getattr(self, '_base_at', 0) < 1.0:
                return                   # after 1-2: in planning with our own HUD up (not a carousel), once a second
            self._base_at = time.time()
        tac_box = self.watcher.tac[1]
        bars = B.bar_places(frame, exclude=tac_box)[0]
        fr = BN._bench_band(frame)
        crops = [BN.slot_crop(fr, i).astype(np.int16) for i in range(9)]
        prev, self._base_prev = getattr(self, '_base_prev', None), crops
        if prev is None:
            return
        ok = [i for i in range(9) if i not in got and not bars[i] and not self.on_tactician(i)
              and self.roster.at(('bench', i)) is None and np.abs(crops[i] - prev[i]).mean() < 4]
        # no anvil and no tooltip or panel over it: those are things on the slot, not its bare tile (also at 1-2:
        # run 20261016-052413 had the opening encounter's anvil on slot 1, learned as 'empty' - the anvil then
        # read as a free slot)
        ok = [i for i in ok if BN.container_match(BN.slot_crop(fr, i))[1] < BN.CONTAINER
              and float((BN.slot_crop(fr, i)[40:].max(axis=2) < 40).mean()) < BN.OBSCURED]
        if later:
            # after 1-2 a slot's picture is taken only once it has looked the same for 2 s: a glow on a free slot
            # (hover, sell flash) outlasts one read (run 20261016-052413 2-6: a glow on slot 1 became its 'empty'
            # picture, and the bare slot then read as taken)
            cand, now, steady = self.__dict__.setdefault('_base_cand', {}), time.time(), []
            for i in range(9):
                if i not in ok:
                    cand.pop(i, None)
                elif i in cand and np.abs(crops[i] - cand[i][0]).mean() < 4:
                    if now - cand[i][1] >= 2.0:
                        steady.append(i)
                        cand.pop(i, None)
                else:
                    cand[i] = (crops[i], now)
            ok = steady
        if ok:
            BN.learn_empty(frame, ok)
            got.update(ok)
            self.log(ev='bench_baseline', stage=scr.stage, slots=ok, have=sorted(got))

    LIT_FOR = 1.5                             # s a slot must stay lit before an unexplained unit counts

    def watch_lights(self, frame, scr):
        """Planning, every read: a slot lit without a break for LIT_FOR s that no tracked unit and no buy of the
        last seconds explains is a unit that came without our action - recorded and read. Two lit reads in a row
        were not enough (run 20261014-063145: the 4-4 carousel's champion walking over slots 4 and 5 to the bench,
        and a unit fading out after its sell at 4-6, became '?' units no read could name - and Aphelios, bought into
        slot 4, then shared it with one)."""
        now = time.time()
        lights = self.bench_lights(frame)
        since = self.__dict__.setdefault('_lit_since', {})
        for i, t in getattr(self.roster, 'vacated', {}).items():
            if now - t < 3.0 and since.get(i, now) < t:
                since[i] = t                     # lit since our sell/move emptied it, not before: the fade-out
        for i, lit in enumerate(lights or []):
            if lit:
                since.setdefault(i, now)
            else:
                since.pop(i, None)
        recent = [b for b in getattr(self.roster, 'recent_buys', []) if now - b[2] < 3]
        pending = [b for b in self.roster.pending_buys if now - b[1] < 15]
        if lights is None or recent or pending:
            return
        bars = B.bar_places(frame, exclude=self.watcher.tac[1])[0] if frame is not None else None
        self.adopt_unexplained([x if not x else now - since.get(i, now) >= Player.LIT_FOR for i, x in enumerate(lights)],
                               scr, bars=bars)

    def expect_gifts(self, augment, stage):
        """The champions an augment's (or Wisp's) own text says it gives ('... Gain a Camille, an Elise, and a
        Caitlyn.'): recorded as expected, told to the model at once, and the units that then arrive without a buy take
        these names instead of being unknown (the user, 2026-10-16). A gift at a later round or level stays expected
        until it lands. From the card's text in the game data - no augment is named in the code."""
        from tft_player.coach import augment_text
        text = re.sub(r'<[^>]+>', ' ', augment_text(augment, full=True) or self.effect_text(augment) or '')
        names = sorted(CATALOG.units, key=len, reverse=True)
        gifts = []
        timing = re.compile(r'\b(level|stage|rounds?|after|every|when|each|combats?)\b', re.I)
        for sentence in re.split(r'(?<=[.!])\s+', text):
            # 'Gain ...' is what we get; '<unit> gains ...' is an effect on a unit, not a gift
            m = re.search(r'\b[Gg]ain\b', sentence)
            if not m:
                continue
            before, sentence = sentence[:m.start()], sentence[m.end():]
            found = []
            for n in names:
                for f in re.finditer(r'\b' + re.escape(n) + r'\b', sentence):
                    if not any(a <= f.start() < b for a, b, _ in found):
                        found.append((f.start(), f.end(), n))
            found.sort()
            for k, (_, end, n) in enumerate(found):
                # when: said before 'Gain' ('At level 7, gain ...') for all, or right after the name ('a Vi now
                # and a Nidalee after 3 rounds')
                after = sentence[end:found[k + 1][0] if k + 1 < len(found) else len(sentence)]
                after = re.split(r'\b(and|with)\b', after)[0] if not timing.search(before) else after
                when = before if timing.search(before) else after if timing.search(after) else ''
                when = re.sub(r'\s+', ' ', when).strip(' ,.;')
                gifts.append({'name': n, 'from': augment, 'later': bool(when), 'when': when, 'stage': stage})
        self.expect_item_gifts(augment, text, stage)
        if gifts:
            self.gifts = getattr(self, 'gifts', []) + gifts
            self.log(ev='gifts_expected', stage=stage, augment=augment, names=[g['name'] for g in gifts])
            self.news_event('gifts_expected', stage, f"{augment} gives "
                            + ', '.join(g['name'] + (f" ({g['when']})" if g['when'] else '') for g in gifts)
                            + ': place them by name in "board", they are fielded the moment they land')

    def expect_item_gifts(self, augment, text, stage):
        """The items an augment's own text says it gives ('Gain a Spirit Visage.'): listed in the state until one is
        seen on the item bench or on a unit (game 37, run 20261029-042013: Spirit Of Redemption's Spirit Visage was
        never seen, and nothing told the player it was still owed). From the card's text in the game data."""
        from tft_player.items import COMPONENTS, ICONS, RECIPES
        names = sorted(set(RECIPES) | set(COMPONENTS) | set(ICONS), key=len, reverse=True)
        got = []
        for sentence in re.split(r'(?<=[.!])\s+', text or ''):
            m = re.search(r'\b[Gg]ain\b', sentence)
            if not m:
                continue
            rest = sentence[m.end():]
            for n in names:
                f = re.search(r'\b(?:an?|1|one)\s+' + re.escape(n) + r"(?![\w'])", rest, re.I)
                if f and not any(n in g for g in got):
                    got.append(n)
        if got:
            self.item_gifts = getattr(self, 'item_gifts', []) + [{'name': n, 'from': augment, 'stage': stage} for n in got]
            self.log(ev='item_gifts_expected', stage=stage, augment=augment, names=got)

    def item_gift_line(self, stage=None):
        """The state's line for items an augment gives that are not seen yet; one seen on the item bench or on any of
        our units has arrived."""
        gifts = getattr(self, 'item_gifts', [])
        if not gifts:
            return None
        base = lambda n: re.sub(r'\s*\(.*$', '', n or '').strip().lower()
        seen = {base(n) for n in (self.bench_items or []) if n} | {
            base(n) for u in self.roster.units for n in (u.items or []) if n}
        for g in [g for g in gifts if base(g['name']) in seen]:
            gifts.remove(g)
            self.log(ev='item_gift_arrived', stage=stage, name=g['name'], augment=g['from'])
        if not gifts:
            return None
        return ('Items your augments give, not seen on the item bench or on a unit yet: '
                + ', '.join(f"{g['name']} ({g['from']}, taken at {g['stage']})" for g in gifts) + '.')

    def gift_line(self, stage=None):
        """The state's line for champions an augment gives that have not landed yet. A gift due now that has not
        landed two rounds later is dropped (it went elsewhere or the text was misread); later gifts stay."""
        from tft_player.coach import stage_key
        now = stage_key(stage) if stage else None
        if now:
            def due_passed(g):
                k = stage_key(g['stage'])
                return not g['later'] and k and (now[0] * 10 + now[1]) - (k[0] * 10 + k[1]) >= 2 + 3 * (now[0] > k[0])
            for g in [g for g in getattr(self, 'gifts', []) if due_passed(g)]:
                self.gifts.remove(g)
                self.log(ev='gift_dropped', stage=stage, name=g['name'], augment=g['from'])
        gifts = getattr(self, 'gifts', [])
        if not gifts:
            return None
        return ('Champions your augments give, not on the bench yet: '
                + ', '.join(f"{g['name']} ({g['from']}{', ' + g['when'] if g['when'] else ''})" for g in gifts)
                + '. Place them by name in "board"; they are fielded the moment they land.')

    def bench_full_of_ours(self):
        """Every bench slot shows one of our (green) health bars: a full bench of ours, not another arena - both read
        with no empty tile (benchmark game 20261020-193454 3-7: two loot units filled slots 7 and 8 and were never
        recorded; three Rammus buys failed on the full bench and records named the two Rammus)."""
        got = self.cap.read() if getattr(self, 'cap', None) is not None and not REPLAY else None
        if got is None:
            return False
        tac = getattr(getattr(self, 'watcher', None), 'tac', (None, None))[1]
        return all(B.bar_places(got[0], exclude=tac)[0])

    def adopt_unexplained(self, lights, scr, bars=None):
        """A lit slot no tracked unit explains got its unit without a buy of ours (loot orb,
        carousel, augment): record it unnamed - identify_one reads it off its info panel.
        Only while no tracked bench unit's slot reads empty (else it may be a tracked unit we
        mis-placed: reconcile settles that). bars: the bench health bars of the same frame - a lit slot with
        no bar of its own next to a slot whose unit shows one is that neighbour's model spilling over the tile,
        not an arrival (run 20261028-012516 5-6: the Sentinel on slot 2 lit slot 1; the '?' recorded there was
        never read, held the slot, and the Elder Dragon bought into it at 5-7 was recorded on slot 3, where a
        Kayle from a loot orb then stood: 6-2 benched three units for that "Elder Dragon" and fought 7 of 9)."""
        from tft_player import bench as BN
        from tft_player.state import Owned
        tracked = {u.loc[1] for u in self.roster.bench()}
        refs = BN.empty_refs()
        covered = [i for i, x in enumerate(lights) if x is None and not self.on_tactician(i) and i in refs]
        if False not in lights and not self.bench_full_of_ours():
            return                  # not one slot looks like our empty tile: another arena is on screen
                                    # (away fights: run 20261005-022645 4-2 read 6 'arrivals' there)
        if covered or any(lights[i] is False for i in tracked):
            return                  # part of the bench covered (banner, panel): its neighbours misread too
        # a slot our own sell or move just emptied stays lit while the unit fades out: no arrival there for 3 s (run
        # 20261016-011010 4-2: a "unit we did not buy" in bench slot 3, 0.4 s after Yunara was sold from it - eight
        # panel reads of nothing followed). Nothing is adopted in the last 2 s of planning either (units teleport).
        now = time.time()
        vacated = getattr(self.roster, 'vacated', {})
        timer = getattr(scr, 'timer', None)
        late = getattr(getattr(self, 'phase', None), 'planning', False) and timer is not None and timer <= 2
        if late and not REPLAY:
            return
        for i, lit in enumerate(lights):
            spill = bars is not None and not bars[i] and any(bars[j] for j in (i - 1, i + 1) if 0 <= j < len(bars))
            # a slot whose '?' was found empty this round (ghost_check): its look is no arrival while it shows no bar
            ghost = getattr(self, '_ghost_slots', {}).get(i) == getattr(scr, 'stage', None) and not (bars and bars[i])
            if lit and i not in tracked and i not in self.roster.blocked and not self.on_tactician(i) \
                    and now - vacated.get(i, 0) > 3.0 and not spill and not ghost:
                u = Owned(None, 1, ('bench', i))
                gifts = getattr(self, 'gifts', [])
                # an expected gift: its name from the augment's text (several at once: in slot order, read again in
                # idle time - a read that differs corrects it and the gift stays expected). A gift due later is not
                # given to a unit that may have come from a loot orb.
                g = next((g for g in gifts if not g['later']), None) or                     (gifts[0] if gifts and time.time() - getattr(self, '_loot_at', 0) > 30 else None)
                if g:
                    gifts.remove(g)
                    u.gift = g
                    u.name = g['name']
                    u.unverified = f"named from {g['from']}'s text: to be read"
                    self.reread = getattr(self, 'reread', set()) | {u.id}
                    self.tries[u.id] = 0
                    self.log(ev='gift_arrived', stage=scr.stage, slot=i, name=u.name, augment=g['from'])
                    print(f"{scr.stage}: bench slot {i}: {u.name} from {g['from']}")
                self.roster.units.append(u)
                src = 'loot' if time.time() - getattr(self, '_loot_at', 0) < 30 else 'no action of ours'
                u.arrival_source = src
                self.log(ev='unit_arrived', stage=scr.stage, slot=i, source=src)
                print(f'{scr.stage}: bench slot {i} lit without a buy ({src}): reading it')

    def save_board_gallery(self, frame, scr):
        """At the start of each fight: a crop of every unit on screen (board and bench: whatever has a health
        bar) with the labels the harness holds - champion, star level (tracked and read off the badge), items,
        place, flags - in runs/<ts>/units/<stage>/. Units the screen shows that we do not track, and tracked
        units no bar shows, are listed too: that is where the harness is wrong (tools/units_report.py makes the
        contact sheet). The disk writes run off the game loop. Only with --record."""
        if REPLAY or not test_mode():
            return
        try:
            from tft_player import stars
            raw = getattr(self.cap, 'raw', None)
            src = raw if raw is not None else frame
            h, w = src.shape[:2]
            s = h / 1080
            d = self.log_dir / 'units' / scr.stage
            places = B.bar_places(frame, exclude=self.watcher.tac[1])[2]
            last = self.__dict__.setdefault('_gal_last', {})        # unit key -> (labels, file) last saved
            rows, jobs = [], []
            for (bx, by), loc in sorted(places, key=lambda p: p[1]):
                u = self.roster.at(loc)
                badge = stars.read_star(frame, (bx, by))
                tracked = None if u is None else {
                    'id': u.id, 'name': u.name, 'star': u.star, 'items': list(u.items),
                    'unverified': u.unverified or None, 'auto_placed': getattr(u, 'auto_placed', False)}
                # a new crop only when what the harness holds for the unit changed (champion, star level,
                # badge, items, flags); otherwise the row points at the crop already saved
                key = ('unit', u.id) if u is not None else ('screen', loc)
                sig = json.dumps([tracked and {k: v for k, v in tracked.items() if k != 'id'}, badge], default=str)
                prev = last.get(key)
                if prev and prev[0] == sig:
                    fn, new = prev[1], False
                else:
                    fn, new = f"{scr.stage}/{loc[0]}-{'-'.join(map(str, loc[1:]))}.jpg", True
                    X, Y = round(w / 2 + (bx - 960) * s), round(by * s)
                    crop = src[max(0, Y - round(22 * s)):Y + round(160 * s),
                               max(0, X - round(75 * s)):X + round(75 * s)].copy()
                    jobs.append((self.log_dir / 'units' / fn, crop))
                    last[key] = (sig, fn)
                rows.append({'file': fn, 'new_crop': new, 'where': list(loc), 'badge_star': badge, 'tracked': tracked})
            seen = {tuple(r['where']) for r in rows}
            for u in self.roster.units:
                if u.loc not in seen:
                    rows.append({'file': None, 'where': list(u.loc), 'badge_star': None,
                                 'tracked': {'id': u.id, 'name': u.name, 'star': u.star, 'items': list(u.items),
                                             'missing': u.missing or None, 'unverified': u.unverified or None}})
            meta = {'stage': scr.stage, 'level': scr.level, 'gold': scr.gold, 'hp': getattr(self, '_hp', None),
                    'units': rows}

            def write():
                d.mkdir(parents=True, exist_ok=True)
                for path, crop in jobs:
                    cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
                (d / 'labels.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
            threading.Thread(target=write, daemon=True).start()
        except Exception as e:
            self.log(ev='gallery_error', error=repr(e))

    def star_check(self, frame, scr, places):
        """Planning, every ~0.5 s: the star badge on each unit's health bar against the tracked star level. A
        different badge read twice in a row corrects the roster - a merge the rules did not see (three copies
        given at once, a copy bought in the fight that merged when it ended) or one they assumed that did not
        happen (run 152203 3-6: a 2-star Vi that was 1-star for a round). The model is told."""
        if REPLAY or time.time() - getattr(self, '_star_at', 0) < 0.5:
            return
        self._star_at = time.time()
        from tft_player import stars
        pend = self.__dict__.setdefault('_star_pend', {})
        read = stars.read_stars(frame, places)
        visible = set(read)
        for loc, star in read.items():
            u = self.roster.at(loc)
            if u is None or star is None or u.star == star or (star == 3 and u.star != 2):
                # (a 3 is taken only for a unit we track as 2-star: the gold badge is new to us)
                if u is not None:
                    pend.pop(u.id, None)
                    if star == u.star:
                        u.badge_star = None
                continue
            prev = pend.get(u.id)
            if prev is None or prev[0] != star:
                pend[u.id] = (star, time.time())
                continue
            if time.time() - prev[1] < 0.4:
                continue
            pend.pop(u.id, None)
            if star > u.star:
                # a rise is a merge: it needs two more copies at the old star level (tracked, missing, or
                # units not named yet). Without them it is a misread badge (run 033507: a lone Vi read as
                # 2-star 3 times, a Kog'Maw with one copy twice - each "corrected" back a round later)
                # ... and the copies a merge used are gone from the screen: one still standing there was not used
                # (run 20261015-014311 5-1: two 1-star Aphelios side by side on the bench, one badge read as 2-star,
                # the copies counted were the other Aphelios and an unnamed board unit - both still there; the
                # roll-down then passed an Aphelios as "2-star reached" and the model sold the real copy)
                copies = [x for x in self.roster.units if x is not u and (x.name is None or
                                                                         (x.name == u.name and x.star == u.star))
                          and (x.missing or x.loc not in visible)]
                if len(copies) < 2:
                    self.log(ev='star_rise_ignored', stage=scr.stage, unit=repr(u), read=star, copies=len(copies))
                    # the state says what the screen shows until the panel settles it (game 80 5-5: Brambleback's
                    # 2-star badge read 3 times, the panel read only at 5-6; the state said 1-star, and a fourth copy
                    # was bought)
                    u.badge_star = (star, scr.stage)
                    # its panel tells which, once a stage: a copy can arrive unseen and merge at once (g15 4-6: a
                    # LeBlanc copy made Elise 2-star - its badge was read 16 times to 5-1, the state said 1-star)
                    asked = self.__dict__.setdefault('_rise_reread', set())
                    if (u.id, scr.stage) not in asked:
                        asked.add((u.id, scr.stage))
                        self.reread = getattr(self, 'reread', set()) | {u.id}
                        self.tries[u.id] = 0
                    continue
            elif not getattr(u, 'star_assumed', False):
                # a drop is only an assumed merge that did not happen: a star level we SAW (badge or panel)
                # does not fall on a badge read alone (item glows, effects) - its panel is read once instead
                if u.id not in getattr(self, 'reread', set()):
                    self.reread = getattr(self, 'reread', set()) | {u.id}
                    self.tries[u.id] = 0
                    self.log(ev='star_drop_check', stage=scr.stage, unit=repr(u), read=star)
                continue
            old, gone = self.roster.set_star(u, star, visible)
            u.badge_star = None
            self.log(ev='star_corrected', stage=scr.stage, unit=repr(u), was=old, now=star,
                     used=[repr(x) for x in gone])
            what = (f'{u.name or "a unit"} (unit {u.id}) is {star}-star on the screen (we had {old}-star)'
                    + (f'; its copies {", ".join(str(x.id) for x in gone)} are gone (merged)' if gone else
                       '; a merge we assumed did not happen, the copies it used are still there' if star < old else ''))
            print(f'{scr.stage}: {what}')
            self.news_event('star_corrected', scr.stage, what)
            if star < old:
                self.reread = getattr(self, 'reread', set()) | {u.id}
            else:
                u.star_badge_at = time.time()
        # a unit that arrived already upgraded (three copies given at once) has no name yet when its badge is
        # read: the copies it used turn up as missing units once it is named - they are merged away
        for u in list(self.roster.units):
            if u.name and u.star >= 2 and time.time() - getattr(u, 'star_badge_at', 0) < 120:
                gone = [m for m in self.roster.units if m is not u and m.name == u.name and m.star == u.star - 1
                        and m.missing][:2]
                for m in gone:
                    u.items = (u.items + m.items)[:3]
                    u.items_read = False
                    self.roster.alias[m.id] = u.id
                    self.roster.units.remove(m)
                if gone:
                    self.log(ev='merged_copies_removed', stage=scr.stage, unit=repr(u), copies=[repr(m) for m in gone])

    def sprykin_fielded(self):
        """Distinct Sprykin champions on the board (the trait counts each champion once)."""
        return len({u.name for u in self.roster.fielded() if u.info and 'Sprykin' in u.info.traits})

    def save_panel_shot(self, tag, text):
        """Data for what a panel says when a unit came without our action (the BFF, a carousel or augment unit):
        the full frame with the panel open, in runs/<ts>/panels/<tag>.jpg, written off the loop."""
        got = self.cap.read() if shots_on() else None
        if got is None or REPLAY:
            return
        frame = getattr(self.cap, 'raw', None)
        frame = frame if frame is not None else got[0]
        path = self.log_dir / 'panels' / f'{tag}_{re.sub(r"[^A-Za-z0-9]+", "-", text or "none")[:30]}.jpg'

        def save(f=frame.copy()):
            try:
                from PIL import Image
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(f).save(path, quality=90)
            except Exception as e:
                self.log(ev='panel_shot_error', error=repr(e))
        threading.Thread(target=save, daemon=True).start()

    def read_new_unit(self, frame, scr):
        """A health bar on a board hex where nothing is recorded (the 1-1 carousel champion, a unit the game
        gives us): click it and read its panel at once (the user, 2026-10-09 - run 010416 knew of the 1-1 unit
        only at 1-3). Planning only (units stand still); a place seen on two ticks; each place once a round. A
        click that opens no champion panel (our tactician, an orb) records nothing. -> True when it clicked."""
        if REPLAY or frame is None or not self.phase.planning or time.time() - self.round_started < 1.0:
            return False
        if scr.timer is not None and scr.timer < 1:
            return False
        bars = B.green_bars(frame)
        _, _, places = B.bar_places(frame, bars=bars)
        tracked = {u.loc for u in self.roster.units if not u.missing}
        new = {loc for _, loc in places if loc[0] == 'board' and loc not in tracked}
        prev, self._untracked_prev = getattr(self, '_untracked_prev', set()), new
        tried = self.__dict__.setdefault('_untracked_tried', set())
        todo = sorted(loc for loc in new & prev if (self.phase.stage, loc) not in tried)
        if not todo:
            return False
        loc = todo[0]
        tried.add((self.phase.stage, loc))
        try:
            name, stars, fresh = self.read_unit(self.unit_point(loc, bars))
        except NotForeground:
            return True
        raw = getattr(self.reader, 'last_panel_text', None)
        self.log(ev='new_unit_read', stage=scr.stage, loc=list(loc), name=name, stars=stars, fresh=fresh, panel_text=raw)
        self.save_panel_shot(f'{scr.stage}_new_{loc[1]}{loc[2]}', raw)
        if not name and loc[0] == 'board' and self.roster.at(loc) is None and self.sprykin_fielded() >= 3 \
                and getattr(self, 'bff_assigned', False):
            # the BFF has had its rider: a unit there that opens no panel is the rider with the BFF under it - a
            # champion (a team slot), named when a read or the model names it (run 20261012-072956 3-1..4-6: the rider
            # Teemo at (1, 4) was recorded as an unnamed unit)
            from tft_player.state import Owned
            # the one fielded Sprykin whose own hex shows no unit now left it for the BFF (benchmark game 20261021-141731
            # 3-2: Teemo** stayed recorded at (0, 0) beside a nameless rider, and a fight give looked for it there);
            # else a missing Sprykin
            occupied = {l for _, l in places}
            left = [m for m in self.roster.fielded() if m.info and 'Sprykin' in m.info.traits and m.loc not in occupied
                    and not getattr(m, 'riding', None)]
            u = left[0] if len(left) == 1 else                 next((m for m in self.roster.missing() if m.info and 'Sprykin' in m.info.traits), None)
            if u is None:
                u = Owned(None, 1, loc, ours=True)
                self.roster.units.append(u)
            u.loc, u.missing, u.unverified, u.riding = loc, None, None, True
            self.news_event('bff_rider', scr.stage, f'a unit at hex {loc[1:]} opens no panel: the BFF rider '
                                                    f'({u.name or "not named yet"}, unit {u.id})')
            return True
        if not name and loc[0] == 'board' and self.roster.at(loc) is None and self.sprykin_fielded() >= 3:
            # before its first rider, a unit on the board that opens no panel while Sprykin is active is the BFF
            # (run 20261010-235723 4-5..4-7: three clicks on it opened nothing). Recorded without a team slot
            from tft_player.state import BFF, Owned
            u = next((m for m in self.roster.units if m.name == BFF), None)
            if u is None:
                u = Owned(BFF, 1, loc)
                self.roster.units.append(u)
            u.loc, u.missing, u.unverified = loc, None, None
            self.news_event('bff', scr.stage, f'the BFF (Sprykin) stands at hex {loc[1:]} (unit {u.id})')
            return True
        if not (fresh and name) or self.roster.at(loc) is not None:
            return True
        from tft_player.state import Owned
        # a missing record of that champion is this unit (a read names it); so is a fielded record of the same
        # champion and star level whose own hex shows no unit (a stale place: run 20261010-235723 4-3 recorded the
        # one Azir*** twice, and the board looked full); else it is a new one
        # so is a bench record of that champion whose slot is empty now: the game fielded it into an empty board slot
        # at combat start, unseen (release run 20261016-141957 3-3: Sejuani, fielded at 3-2, became a second record
        # while the first stayed "on the bench" - the model's sell order for her lost its unit, Vi met a full board)
        occupied = {l for _, l in places}
        same = lambda m: m.name == name and (stars is None or m.star == stars)
        lights = self.bench_lights(frame) if any(same(m) for m in self.roster.bench()) else None
        from_bench = next((m for m in self.roster.bench() if same(m) and lights and lights[m.loc[1]] is False), None)
        u = next((m for m in self.roster.missing() if same(m)), None) or from_bench \
            or next((m for m in self.roster.board() if same(m) and m.loc not in occupied
                     and not getattr(m, 'riding', None)), None)
        # a unit we did not put there: from the bench (or a new record) - the game's own placement, not the model's
        by_game = u is None or u.loc[0] == 'bench'
        was = u.loc if u is not None else None
        if u is None:
            u = Owned(name, stars or 1, loc, ours=True)
            self.roster.units.append(u)
        u.loc, u.missing, u.unverified = loc, None, None
        if by_game:
            u.auto_placed = True
        u.items = [i for i in self.panel_items if i != 'unknown']
        print(f'{scr.stage}: unit at {loc[1:]} where none was recorded: read as {name}{"*" * (stars or 1)}'
              + (f' (unit {u.id}, from bench slot {was[1]})' if was and was[0] == 'bench' else ''))
        self.news_event('unit_found', scr.stage, f'{u.name}{"*" * u.star} (unit {u.id}) stands at hex {loc[1:]}: '
                        + (f'the game fielded it from bench slot {was[1]} (an empty board slot at combat start)'
                           if was and was[0] == 'bench' else 'read off its panel (it was not in the records)'))
        return True

    def check_autofill(self, frame, scr):
        """The first seconds of a fight: a bench slot where a unit is tracked goes dark with no operation of
        ours (a sell, a move and a merge update the roster at once) - the game fielded that unit into an empty
        board slot at combat start (run 152203: a second Varus fielded this way, the records never knew). The
        game's order (the user, 2026-10-09; run 010416 1-3 and 1-4 agree): bench units left to right go to the
        free hexes from the bottom left, to the right - (0,0), (0,1) ... (0,6), then the next row. So their
        hexes are recorded at once, no read needed."""
        t0 = getattr(self, 'combat_t0', None)
        if t0 is None or self._autofill_stage == scr.stage or REPLAY:
            return
        age = time.time() - t0
        # the first seconds of the fight - longer while two bench reads have not been taken (run 20261020-110532 4-1:
        # ticks of 4-7 s gave one read in 5 s; Yorick, fielded from slot 0, was never recorded, and Sentinel was
        # refused "board is full" for two rounds). A later dark slot is no operation of ours: those update the roster
        reads = getattr(self, '_autofill_reads', 0)
        if age > 15.0 or age > 5.0 and reads >= 2:
            self._autofill_stage = scr.stage
            return
        if age < 0.6:
            return                              # the fill happens as combat starts; let the picture settle
        lights = self.bench_lights(frame)
        if lights is None:
            return                              # unreadable, or another arena is on screen
        prev, self._autofill_prev = self._autofill_prev, lights
        self._autofill_reads = reads + 1
        if prev is None:
            return
        bench = [u for u in self.roster.bench() if not u.missing]           # by slot, left to right
        dark = lambda u: lights[u.loc[1]] is False and prev[u.loc[1]] is False
        unread = lambda u: lights[u.loc[1]] is None and prev[u.loc[1]] is None
        gone = [u for u in bench if dark(u)]
        free = [('board', r, c) for r in range(4) for c in range(7) if self.roster.at(('board', r, c)) is None]
        # the game fields only up to the team size, and each fielded unit shows a health bar on its hex: dark bench
        # slots beyond that are not fielded units (run 20261010-010345 2-2/2-3: 4 recorded fielded at a full board)
        cap = self.team_cap(scr) or 0
        room = max(0, cap - self.roster.slots_used())
        try:
            barred = {l for _, l in B.bar_places(frame, exclude=self.watcher.tac[1])[2] if l[0] == 'board'}
        except Exception:
            barred = None
        # a unit whose slot cannot be read (our tactician stands on it) is no evidence either way - but the game
        # fields from the left: with more new bars on free hexes than dark slots, the unread units from the left up
        # to the first one still lit are the ones it fielded (run 20261020-092031 3-1: Scuttlecrab on slot 0 under
        # the tactician was fielded unseen, and five sells on it failed through the fight)
        new = [loc for loc in free if barred is not None and loc in barred]
        if room and len(new) > len(gone):
            lead = []
            for u in bench:
                if not (dark(u) or unread(u)):
                    break                       # a unit still on its slot: none right of it went first
                lead.append(u)
            if len(lead) > len(gone):
                gone = lead
        if not gone:
            return
        self._autofill_stage = scr.stage
        gone.sort(key=lambda u: u.loc[1])
        crossed_buy = any(getattr(u, 'bought_at', 0) >= t0 - 3 for u in gone)
        if room and crossed_buy:
            # Game 71 1-3: Krug left b0 for auto-fill while the late Varus buy
            # refilled b0. Varus's predicted b2 was dark. Neither the dark slot
            # nor a bar moving through h20 identifies the fielded champion.
            # Read the canonical home hex and the affected bench on return.
            places = free[:room] + [u.loc for u in bench[:room]] + [u.loc for u in gone]
            self._autofill_repair = (scr.stage, list(dict.fromkeys(places)))
            self.log(ev='auto_fill_uncertain', stage=scr.stage, places=self._autofill_repair[1],
                     roster=repr(self.roster))
            if getattr(self, 'log_dir', None) is not None:
                self.snap(frame, f'{scr.stage}-auto-fill-uncertain', True, force=True)
            return
        # Combat bars have already left their home hexes. The game's placement
        # order gives their home positions; bars only corroborate the count.
        count = room if barred is None else min(room, len(new))
        pairs = list(zip(gone, free))[:count]
        if not pairs:
            self.log(ev='auto_fill_refused', stage=scr.stage, dark=[u.loc[1] for u in gone], room=room)
            return
        moves = []
        for u, loc in pairs:
            moves.append(f"{u.name or '?'} (unit {u.id}) bench slot {u.loc[1]} -> hex {loc[1:]}"
                         + ('' if dark(u) else ' (its slot was not readable: by the game\'s order)'))
            if not dark(u):
                u.unverified = f'fielded by the game at combat start from bench slot {u.loc[1]}, by its order: to be read'
                self.reread = getattr(self, 'reread', set()) | {u.id}
            u.loc, u.auto_placed = loc, True
            if u.name is None:
                self.reread = getattr(self, 'reread', set()) | {u.id}      # a '?' is still to be named
        self.log(ev='auto_fill', stage=scr.stage, units=moves)
        print(f'{scr.stage}: the game fielded {"; ".join(moves)} (its order: bottom left, to the right)')
        self.news_event('game_moved_unit', scr.stage, f'the game fielded units from the bench at combat start (empty '
                        f'board slots): {"; ".join(moves)}.')

    def landing_slot(self, name, lights0, scr):
        """Where a bought unit landed: the slot that lit up after the click. None when a merge is
        expected (two copies owned: no new light) or nothing lit in time (then predicted)."""
        self._landing_note, self._no_merge, self._landing_guess = None, False, None
        if lights0 is None or REPLAY:
            return None
        merge = len(self.roster.copies(name, 1)) >= 2
        # a merge is expected with two copies held: no slot lights then. One that lights all the same means the game
        # did NOT merge - a "copy" is another champion (run 20261015-161625 3-3: an "Ornn" that was a Veigar made a
        # merge we assumed, the new Ornn was then taken for a unit we did not buy, and the stars went wrong for rounds)
        # (copies on the bench: the merge shows the moment one of their slots goes dark, below; copies on the board
        # only: no light within 0.4 s of the card leaving the shop is the merge - it was 0.7 s)
        held_bench = [u for u in self.roster.copies(name, 1) if u.loc[0] == 'bench'] if merge else []
        end = time.time() + ((0.7 if held_bench else 0.4) if merge else 1.5)
        lit = None                               # (slot, when): a new light while a merge is expected
        # the bench slots of the copies we hold: a merge empties at least one of them (the merged unit stays on one
        # copy's place). One lit before and dark now means the game merged, and the newly lit slot is something else
        # that arrived at the same moment (run 20261026-011825 4-7: an anvil from "Convert Essence to loot" landed in
        # slot 1 as Morgana's third copy was bought; the merge was judged not to have happened and 22 gold went into
        # rolls for a Morgana 2 we had; 4-2 the same with Diana)
        held = [u.loc[1] for u in self.roster.copies(name, 1) if u.loc[0] == 'bench'] if merge else []
        emptied = lambda lights: [s for s in held if lights0[s] and lights[s] is False]
        while time.time() < end:
            lights = self.bench_lights()         # (the next frame: its wait is the pace, no sleep)
            if merge and lit is None and lights and emptied(lights):
                return None                      # a copy we hold left its slot: the game merged them
            new = [i for i in range(9) if lights and lights[i] and lights0[i] is False]
            # A one-frame empty read of a known slot is not a new landing.
            # Game 63 4-2 mistook Brambleback's slot for the Murkwolf purchase.
            for i in list(new):
                here = self.roster.at(('bench', i))
                if here is not None and here.name and getattr(here, 'sure', False) and not here.unverified:
                    name_here, star_here, _ = self.read_fresh(self.unit_point(here.loc), [])
                    if name_here != name or star_here != 1:
                        new.remove(i)
                        lights0 = list(lights0)
                        lights0[i] = True
                        self.log(ev='buy_landing_occupied', stage=scr.stage, slot=i,
                                 unit=name, tracked=repr(here), read=name_here)
            if merge and lit is not None:
                if lights and lights[lit[0]] is False:
                    # the bought copy showed in a slot for a moment and flew into the merge (benchmark game
                    # 20261018-134910: all 9 'did not happen' verdicts were merges - each left ghost copies that were
                    # dropped a round later, and Veigar's 3-star read was ignored for want of copies)
                    self.log(ev='merge_after_flash', stage=scr.stage, unit=name, slot=lit[0],
                             secs=round(time.time() - lit[1], 2))
                    return None
                if lights and emptied(lights):
                    self.log(ev='merge_with_arrival', stage=scr.stage, unit=name, slot=lit[0],
                             emptied=emptied(lights))
                    return None
                if time.time() - lit[1] >= getattr(self, 'MERGE_SETTLE', MERGE_SETTLE):
                    self._no_merge = True
                    self.log(ev='merge_did_not_happen', stage=scr.stage, unit=name, slot=lit[0])
                    return lit[0]
                continue
            if new and merge:
                lit = (new[0], time.time())     # still lit after MERGE_SETTLE: no merge
                end = max(end, time.time() + getattr(self, 'MERGE_SETTLE', MERGE_SETTLE) + 0.2)
                continue
            if new:
                if len(new) > 1:
                    # another unit arrived with ours (a Wisp's unit, an augment's delivery): which is ours is
                    # read, not assumed (run 185805 4-2: Malphite took the slot the bought Aphelios was recorded in)
                    self._landing_note = f'landed while another unit arrived (bench slots {new} lit up together)'
                    self.log(ev='buy_landing_unclear', stage=scr.stage, unit=name, slots=new)
                return new[0]
        if merge and lit is not None:
            self._no_merge = True                # lit until the end of the wait: no merge
            self.log(ev='merge_did_not_happen', stage=scr.stage, unit=name, slot=lit[0])
            return lit[0]
        if merge:
            return None                          # nothing lit: the merge happened
        before = getattr(self, '_landing_before', None)
        got = self.cap.read() if before is not None else None
        if got:
            from tft_player import bench as BN
            candidates = BN.landing_candidates(before, got[0])
            # Tile evidence locates the arrival; its panel must also name the bought
            # champion. This excludes the unrelated changes that misplaced Karma and
            # Elder Dragon in game 60 (e196).
            if len(candidates) == 1:
                i = candidates[0]
                loc = ('bench', i)
                others = [self.unit_point(u.loc) for u in self.roster.units if u.name and u.name != name]
                read, stars, _ = self.read_fresh(self.unit_point(loc, B.green_bars(got[0])), others)
                if read == name and stars == 1:
                    self._landing_guess = i      # compatibility with recorded landing readers; never a guess
                    self.log(ev='buy_landing_observed', stage=scr.stage, unit=name, slot=i,
                             how='empty tile before, occupied after, panel name and stars')
                    return i
        self._landing_note = 'its landing slot was not seen (slot predicted)'
        # no guess from the screen (change review of game 60 removed e175's tile-change guess: run 20261104-102852
        # 1-3 put Karma on Varus's place and 5-7 put Elder Dragon on slot 5, where an item anvil stood)
        self.log(ev='buy_landing_unseen', stage=scr.stage, unit=name)
        return None

    def copies_doubted(self, name, scr):
        """After a buy the game did not merge although we held two copies: those records are read off their panels
        again - one of them is another champion."""
        if not getattr(self, '_no_merge', False):
            return
        self._no_merge = False
        doubted = [u for u in self.roster.units if u.name == name and u.star == 1 and not getattr(u, 'sure', False)]
        for u in doubted:
            u.unverified = f'the game did not merge a {name} bought next to it: read again'
            self.reread = getattr(self, 'reread', set()) | {u.id}
            self.tries[u.id] = 0
        self.news_event('unit_corrected', scr.stage, f'a {name} bought next to two recorded copies did not merge: '
                        f'those records ({", ".join(str(u.id) for u in doubted)}) are read again')

    def buy_refusal(self, scr):
        """Keep the game's capacity fact until a confirmed action frees a slot.

        Game 73 4-1: an anvil occupied the predicted free slot; three more
        purchases retried the full bench, and each was filed as a pending unit.
        """
        got = self.cap.read()
        said = None
        if got is not None:
            frame = got[0]
            h, w = frame.shape[:2]
            s = h / 1080
            box = (int(w / 2 - 480 * s), int(810 * s), int(w / 2 + 480 * s), int(843 * s))
            # The wider game-message crop includes the ability tooltip's line
            # above this notice in game 73 and rejects the whole OCR result.
            said = (self.reader._texts(frame, [box])[0] or '').strip()
        full = bool(re.search(r'be(?:n|on)ch\s+is?\s+full?\b', said or '', re.I))
        if not full:
            said = self.game_message(got[0]) if got is not None else None
        if full:
            self.roster.bench_full_seen = True
            self.news_event('buy_blocked', scr.stage, f'the game refused the purchase: {said}')
        self.fail_cause = (f'the game refused the purchase: {said}' if said else
                           'neither gold nor the shop slot changed after the click '
                           '(not enough gold, bench full, or the shop moved on)')
        return full

    def execute_buy_burst(self, a, scr, name, more, lights0):
        """Several wanted champions bought at once (auto_burst): the slots are clicked back to back, one read confirms
        which emptied, and each is recorded where the game puts it - the leftmost free bench slot, then merges
        (Roster.bought). A landing the bench lights do not show where predicted is read before the unit is fielded
        (unverified: confirm_before_field reads it). -> True when the first slot was bought."""
        from tft_player.planner import RoundPlan
        predicted = copy.deepcopy(self.roster)
        picks, seen = [], set()
        gold = scr.gold or 0
        for j, n in [(a.arg, name)] + list(more):
            info = CATALOG.unit(n)
            if (j in seen or not 0 <= j < 5 or scr.shop[j] != n or info is None or info.cost > gold
                    or not predicted.buy_has_room(n)):
                continue
            seen.add(j)
            picks.append((j, n))
            gold -= info.cost
            predicted.bought(n)
        if not picks or picks[0][0] != a.arg:
            self.fail_cause = 'burst first card is no longer affordable or has no landing room'
            return False
        plans = [getattr(self, 'round_plan', None)]
        t0 = time.time()
        got0 = None if REPLAY else self.cap.read()
        sig0 = self.reader.shop_sig(got0[0]) if got0 else None          # the cards' name strips before the clicks
        sent = []
        for k, (j, _) in enumerate(picks):
            if k:
                time.sleep(0.06)                 # two game frames between the clicks
            try:
                self.inp.click(B.SHOP_CLICK[j])
            except NotForeground:
                if not sent:
                    raise
                break
            sent.append(picks[k])
        done = set()

        def emptied(s):
            if s.stage != scr.stage or not getattr(s, 'shop_visible', True) or s.gold is None:
                return False
            empty = [(j, n) for j, n in sent if j in done or s.shop[j] is None]
            if sum(CATALOG.unit(n).cost for _, n in empty) <= scr.gold - s.gold:
                done.update(j for j, _ in empty)
            return len(done) == len(sent)
        if sig0 is not None:
            # each card bought leaves its slot: its name strip changes on two frames in a row (pixels, no HUD read -
            # the gold-sum wait read the whole HUD per look: 1.6-1.7 s for game 74b's 5-1 and 6-1 bursts)
            streak, end = {j: 0 for j, _ in sent}, time.time() + 1.0
            while time.time() < end and len(done) < len(sent):
                got = self.cap.read()
                if got is None:
                    continue
                sig = self.reader.shop_sig(got[0])
                for j, _ in sent:
                    streak[j] = streak[j] + 1 if (np.abs(sig[j] - sig0[j]) > 40).mean() > 0.1 else 0
                    if streak[j] >= 2:
                        done.add(j)
        else:
            self.wait_for(emptied, timeout=2.5)
        plans.append(getattr(self, 'round_plan', None))
        new = []
        full = Player.buy_refusal(self, scr) if len(done) < len(sent) else False
        for j, n in sent:
            if j in done:
                self.roster.last_merge = None
                u = self.roster.bought(n)
                new.append(u)
                if a.why.startswith('auto-'):
                    for rp in {id(p): p for p in plans if p is not None}.values():
                        k = next((k for k, want in enumerate(rp.to_buy) if want == n and k not in rp.bought), None)
                        if k is not None:
                            rp.bought.add(k)
                if self.roster.last_merge and not REPLAY:
                    self.confirm_merge(self.roster.last_merge, scr)
            elif not full:
                self.roster.buy_unconfirmed(n)
        lights = self.bench_lights() if lights0 is not None and not REPLAY else None
        doubt = []
        for u in new:
            if u in self.roster.units and u.loc[0] == 'bench' and lights is not None and lights[u.loc[1]] is not True:
                u.unverified = f'{RoundPlan.LANDING_DOUBT} (bought in a burst; its predicted slot shows no unit)'
                self.reread = getattr(self, 'reread', set()) | {u.id}
                doubt.append(u.id)
        self.log(ev='buy_burst', stage=scr.stage, slots=[j for j, _ in picks], names=[n for _, n in picks],
                 bought=sorted(done), doubt=doubt, seconds=round(time.time() - t0, 2))
        return a.arg in done

    def note_landing(self, unit):
        """A bought unit whose landing was unclear: unverified, and read off its panel in idle time."""
        note = getattr(self, '_landing_note', None)
        if note and unit is not None and unit in self.roster.units:
            unit.unverified = note
            self.reread = getattr(self, 'reread', set()) | {unit.id}

    # -- execution + verification ----------------------------------------------------
    def execute_wisp_buy(self, action, scr):
        """Deliver one authorised live offer; gold alone never proves its acquisition."""
        slot = action.arg
        key = getattr(action, 'wisp_key', (scr.stage, slot, scr.shop_raw[slot]))
        entry = self.__dict__.setdefault('wisp_q', {}).setdefault(
            key, {'asked': time.time(), 'buy': True, 'done': False})
        self.__dict__.setdefault('wisp_flagged', {}).pop(key, None)
        if entry.get('expired') or entry.get('done'):
            self.fail_cause = 'Wisp offer already expired or completed'
            return False
        if time.time() < entry.get('retry_at', 0):
            self.deferred = True
            return False
        got = self.cap.read()
        if not got:
            self.deferred = True
            return False
        gold, names, raw = self.reader.read_shop(got[0])
        current = types.SimpleNamespace(stage=scr.stage, shop_visible=True, shop=names, shop_raw=raw,
                                        gold=gold, timer=scr.timer)
        Player.observe_wisp_shop(self, current)
        offer = Player.wisp_offer_id(self, key)
        expected = getattr(action, 'wisp_offer', None)
        if (names[slot] or raw[slot] != key[2] or key[0] != scr.stage
                or entry.get('expired') or expected is not None and offer != expected):
            entry['done'], entry['expired'] = True, True
            self.fail_cause = 'the accepted Wisp belongs to an expired shop'
            self.news_event('game_change', scr.stage,
                            f'accepted Wisp {key[2]} not bought: current slot {slot} is {names[slot] or raw[slot] or "empty"}')
            Player.record_wisp(self, ev='wisp_delivery', stage=scr.stage, slot=slot, wisp=key[2],
                               offer_id=expected, acquired=False, reason=self.fail_cause)
            return False
        cost = self.reader.shop_cost(got[0], slot)
        entry['cost'] = cost
        if gold is None or cost is not None and gold < cost:
            self.fail_cause = f'Wisp not affordable now: cost {cost}, observed gold {gold}'
            self.deferred = True
            Player.record_wisp(self, ev='wisp_delivery', stage=scr.stage, slot=slot, wisp=key[2],
                               offer_id=offer, acquired=False, cost=cost, gold_before=gold,
                               timer=scr.timer, reason=self.fail_cause)
            return False
        entry['attempts'] = entry.get('attempts', 0) + 1
        try:
            self.inp.click(B.SHOP_CLICK[slot])
        except NotForeground:
            entry['attempts'] -= 1
            raise
        observed = {}

        def acquired(state):
            observed['gold'], observed['raw'] = state.gold, state.shop_raw[slot]
            return state.shop_raw[slot] != key[2] and (state.shop[slot] is not None or not state.shop_raw[slot])

        ok = self.wait_for(acquired, timeout=1.2)
        entry['done'] = ok or entry['attempts'] >= 2
        entry['acquired'] = ok
        entry['retry_at'] = time.time() + .5
        if not ok:
            self.fail_cause = 'Wisp acquisition not observed after the click'
            self.deferred = not entry['done']
        else:
            live = getattr(self, '_wisp_live', {}).get(slot)
            if live and live['key'] == key:
                live['acquired'] = True
            self.__dict__.setdefault('wisp_flagged', {})[key] = 'buy'
            self.news_event('wisp_bought', scr.stage, f'Wisp {key[2]} acquired; gold {gold} -> {observed.get("gold")}')
        Player.record_wisp(self, ev='wisp_delivery', stage=scr.stage, slot=slot, wisp=key[2],
                           offer_id=offer, acquired=ok, cost=cost, gold_before=gold,
                           gold_after=observed.get('gold'), timer=scr.timer, attempt=entry['attempts'])
        return ok

    def execute(self, a: Action, scr) -> bool:
        action_plan = getattr(self, 'round_plan', None)
        sale_read_required = False
        if a.kind == 'sell' and not a.arg.on_board:
            from tft_player.comp_rules import plan_targets, rules, upgrade_keep
            rp = getattr(self, 'round_plan', None)
            if rules().get('entry_hexes') and a.arg.id in upgrade_keep(self.roster, plan_targets(getattr(rp, 'p', None))):
                if not getattr(rp, 'explicit_spare_sale', lambda u: False)(a.arg):
                    self.fail_cause = f'comp rule: {a.arg} is retained for its star upgrade'
                    self.news_event('comp rule', scr.stage, self.fail_cause)
                    return False                 # includes after-fight/carried sells, before any input
                sale_read_required = True
        inp = self.inp
        if (a.kind == 'buy' and (getattr(a, 'wisp_key', None)
                or scr.shop[a.arg] is None and scr.shop_raw[a.arg] and Player.wisp_name(scr.shop_raw[a.arg]))):
            return Player.execute_wisp_buy(self, a, scr)
        if a.kind in ('sell', 'move', 'item', 'strip'):
            # a unit under our tactician cannot be clicked, dragged or sold: walk it off first
            unit = a.arg if a.kind == 'sell' else a.arg[0] if a.kind == 'move' else a.arg[2] if a.kind in ('item', 'strip') and len(a.arg) > 2 else None
            locs = [getattr(unit, 'loc', None)] + ([a.arg[1]] if a.kind == 'move' else [])
            for loc in locs:
                if isinstance(loc, tuple) and loc and loc[0] in ('bench', 'board'):
                    self.tactician_off(loc_point(loc)[:2])
        if a.kind == 'buy':
            name = scr.shop[a.arg]            # None for a wisp: nothing lands on our bench by name
            gold0 = scr.gold
            if DATA_MODE:
                got0 = self.cap.read()
                self._bench_before = self.bench_pixels(got0[0]) if got0 else None
                from tft_player import bench as BN
                BN._memo.clear()
                self._bench_before_kinds = [o['kind'] for o in BN.classify(got0[0], B.occupancy(got0[0])[0])] if got0 else None
            got0 = self.cap.read() if name else None             # one frame: the bench and the shop before the click
            lights0 = self.bench_lights(got0[0]) if got0 else None
            self._landing_before = got0[0].copy() if got0 else None
            self._landing_guess = None
            if lights0 is not None:
                self.adopt_unexplained(lights0, scr)
            if name and not self.roster.buy_has_room(name):
                self.deferred = True
                self.fail_cause = 'purchase queued: the bench is full; a sell, fielding or merge must free room'
                return False
            more = getattr(a, 'more', None) if name else None
            if more:
                return self.execute_buy_burst(a, scr, name, more, lights0)
            sig0 = self.reader.shop_sig(got0[0]) if got0 and not REPLAY else None
            inp.click(B.SHOP_CLICK[a.arg])
            raw0 = scr.shop_raw[a.arg]
            if name and sig0 is not None:
                # a champion bought leaves its shop slot: the card's name strip changes (pixels, ~0.03 ms a look) - no
                # HUD read per look (the owner, 2026-11-08: an auto-buy took 2.6 s; its wait read the whole HUD, ~0.8 s
                # a read, until the gold dropped). A refused click (gold, a full bench) leaves the card where it is
                ok = self.wait_card(a.arg, sig0, 1.0, lambda a_, b_: (np.abs(a_ - b_) > 40).mean() > 0.1)
            else:
                # a Wisp can cost 0 and leaves the champion behind it in the slot: bought
                # when that slot's text changes
                ok = self.wait_for(lambda s: s.gold is not None and s.gold < gold0 or
                                   (s.shop[a.arg] is None if name else s.shop_raw[a.arg] != raw0), timeout=2.5)
            full = Player.buy_refusal(self, scr) if not ok else False
            if ok and name:
                self.roster.last_merge = None
                slot = self.landing_slot(name, lights0, scr)
                if slot is None:
                    slot = getattr(self, '_landing_guess', None)
                new = self.roster.bought(name, slot=slot, merge=not getattr(self, '_no_merge', False))
                self.copies_doubted(name, scr)
                self.note_landing(new)
                if self.roster.last_merge and not REPLAY:
                    self.confirm_merge(self.roster.last_merge, scr)
                elif DATA_MODE and new in self.roster.units:
                    self.photograph_buy(new)
            elif name and not full:
                self.roster.buy_unconfirmed(name)   # the unit that shows up on the bench is this one
            return ok
        if a.kind in ('sell', 'move', 'item', 'container'):
            on_bench = (a.kind == 'sell' and a.arg.loc[0] == 'bench') or (a.kind == 'container') or \
                (a.kind == 'move' and 'bench' in (a.arg[0].loc[0], a.arg[1][0])) or \
                (a.kind == 'item' and getattr(a.arg[2], 'loc', ('',))[0] == 'bench')
            self._away_spot = None
            if on_bench and not self.bench_located(scr, a.arg if a.kind == 'sell' else None):
                return False
        if a.kind == 'sell':
            away = getattr(self, '_away_spot', None)
            away = away[1] if away and away[0] == a.arg.id else None    # read off its panel in this away fight
            self._away_spot = None
            got = None if REPLAY else self.cap.read()
            if a.arg.loc[0] == 'bench' and got is not None:
                why = self.bench_sell_blocked(got[0], a.arg)
                if why == 'wait':
                    self.fail_cause = 'no bench health bar is drawn yet: tried again in a moment'
                    self.deferred = self.retry_soon = True
                    return False
                if why == 'stale':
                    old = a.arg.loc[1]
                    new = Player.stale_bench_slot(self, got[0], a.arg)
                    if new is not None:
                        # the one bench slot with a unit no record holds: the same unit, mis-tracked (game 82 5-5:
                        # Sett tracked at slot 5, which was empty, while slot 4 held a unit no record had). Its panel
                        # is read before the sale (an unverified unit is read first), so a wrong guess sells nothing
                        a.arg.loc = ('bench', new)
                        a.arg.unverified = f'tracked at bench slot {old}, empty on screen; found at slot {new}: to be read'
                        self.log(ev='sell_relocated', stage=scr.stage, unit=a.arg.id, was=old, now=new)
                        self.fail_cause = (f'bench slot {old} is empty on screen, where {a.arg.name} is tracked - '
                                           f'found at slot {new}, the one slot with an untracked unit: read, then sold')
                        self.deferred = self.retry_soon = True
                        return False
                    a.arg.unverified = f'not seen at bench slot {a.arg.loc[1]} where it is tracked: to be read'
                    self.reread = getattr(self, 'reread', set()) | {a.arg.id}
                    self.fail_cause = (f'bench slot {a.arg.loc[1]} is empty on screen, where {a.arg} is tracked - '
                                       f'nothing was sold; it is looked for')
                    return False
            if away is None and not Player.confirm_before_loss(self, a.arg, scr, 'sell', sale_read_required):
                return False
            if (a.arg.on_board and getattr(getattr(self, 'phase', None), 'planning', False)
                    and scr.timer is not None and scr.timer <= 2):
                self.fail_cause = 'queued: the round flips in the last seconds (board sale)'
                self.deferred = True
                return False
            got = None if REPLAY else self.cap.read()
            gold0 = (self.reader.read_shop(got[0])[0] if got else None) or scr.gold   # the gold right before E
            if not REPLAY:
                # the game sells the unit it has under the cursor: give its hover time to move here first (run
                # 20261013-220557 4-5: E on Amumu at bench slot 1 sold the fielded Murkwolf 2-star)
                spot = away or self.unit_point(a.arg.loc, B.green_bars(got[0]) if got else None)
                inp.move(spot)
                # the game picks the hovered unit a frame or two after the cursor arrives; at 0.08 s under load E
                # found nothing under it (flex game 20261018-112700 2-2: Rek'Sai hovered in the frame after, not sold)
                time.sleep(SELL_HOVER)
            else:
                spot = loc_point(a.arg.loc)
            inp.key('e', at=spot)
            # checked, like every other action: run26 counted 3 sells that the game never did ("Your
            # bench is full!"), so the roster had room the game didn't and every buy failed. The fast
            # evidence is the unit's place going empty (bars, ~0.15 s); the gold text lags and costs OCR,
            # so it is only the fallback
            took = REPLAY or self.sell_seen(gold0)
            if took and took is not True and a.arg.name and a.arg.info:
                # the gold it brought tells which unit went: another amount means another unit was sold
                from tft_player.planner import sell_value
                want = sell_value(a.arg)
                if took - gold0 != want:
                    self.log(ev='sell_value_mismatch', stage=scr.stage, unit=repr(a.arg), expected=want,
                             got=took - gold0)
                    self.news_event('sell_mismatch', scr.stage,
                                    f'selling {a.arg} brought {took - gold0} gold, not its {want}: maybe another unit '
                                    f'was sold - the board and bench are read again')
                    self.need_survey = f'sell of {a.arg} brought {took - gold0}g, not {want}g'
            if not took:
                self.fail_cause = f'nothing was sold: gold stayed {gold0} and {a.arg.loc} still holds a unit'
                if self.pilot:
                    self.need_survey = f'a sell of {a.arg} did not take'
                else:
                    self.reread = getattr(self, 'reread', set()) | {a.arg.id}   # read that one unit, once
                # its place may be stale: the retry reads its panel first (benchmark game 20261018-102854 3-5: Caitlyn
                # sold twice at (1,1) while she stood at (0,2))
                a.arg.unverified = a.arg.unverified or 'a sell at this place did not take'
                return False
            # its items come back to the item bench: named there when the picture does not know them (game 36 4-2)
            self.item_track.expect(list(a.arg.items or []), time.time())
            self.roster.sold(a.arg)
            if a.why.startswith('comp cleanup:'):
                self.news_event('comp cleanup', scr.stage, f'sold {a.arg.name} (unit {a.arg.id}) from the bench: '
                                'on none of the comp boards or planned units'
                                + (f'; items returned to the item bench: {", ".join(a.arg.items)}'
                                   if a.arg.items else ''))
            return True
        if a.kind == 'container':
            if a.arg in self.auto_opened and not self.bench_objs.get(a.arg):
                return True                      # already opened by the harness: nothing left to do
            ok = self.open_container(a.arg, scr)
            if not ok:
                self.fail_cause = f'bench slot {a.arg} holds no recognised container'
            return ok
        if a.kind == 'move':
            unit, dst = a.arg
            src = unit.loc
            if getattr(unit, 'riding', False) and src[0] == 'board':
                # a unit riding the Sprykin BFF is not dragged off it: dragging the rider moves the BFF onto the Sprykin
                # it is dropped on (the user, 2026-10-12); dropped on a hex it did not move (benchmark game
                # 20261021-125639: 9 moves of the rider Teemo from (1, 0) failed, 3-1 to 4-3)
                self.fail_cause = (f'{unit} rides the BFF: a rider is not dragged off it - a "link" from it to another '
                                   f'Sprykin makes that one the rider')
                return False
            if self.phase.planning and scr.timer is not None and scr.timer <= 1 and 'board' in (src[0], dst[0]):
                # the board locks as combat starts: a drag in the last second does not land (the user: 1 s, not
                # earlier - every second of planning counts). It is kept for the next round.
                self.fail_cause = 'queued for the next planning phase (the board locks at combat start)'
                self.deferred = True
                return False
            # one frame for the looks before the drag (each fresh read waits for the next frame, ~33 ms); taken again
            # only after a click of ours (a panel read)
            look = None if REPLAY else self.cap.read()
            if not REPLAY:
                got = look
                if got is not None and src[0] == 'board' and self.unit_bar(got[0], src) is None:
                    # run29 3-2: "Rammus at (0,0)" was dragged twice from an empty hex
                    unit.unverified = f'not seen at hex ({src[1]},{src[2]}) where it is tracked: to be read'
                    self.reread = getattr(self, 'reread', set()) | {unit.id}
                    # Game 71 1-4: combat auto-fill had attached Varus to a moving
                    # Krug's bar. Repeating the drag cannot repair that record.
                    self.need_survey = f'no unit is seen at {src}, where {unit} is tracked'
                    self.survey_places = (self.need_survey, [src])
                    self._repair_pending, self.deferred = True, True
                    self.fail_cause = self.need_survey + ' - its place is being read before another drag'
                    return False
            cap = self.team_cap(scr)
            if src[0] == 'bench' and dst[0] == 'board' and self.roster.at(dst) is None and cap:
                got = look or self.cap.read()
                bars = B.occupancy(got[0], exclude=self.watcher.tac[1])[1] if got else set()
                # bars count units, the cap counts team slots: an Elder Dragon shows one bar and uses 2, and needs 2;
                # a unit that uses none (Elderwood plants, the BFF, a Training Dummy) shows a bar and uses 0 (run
                # 20261015-115044 3-5: 5 champions + a Stonebark Tree read "6/6 full", Sentinel was never fielded)
                n_board = len(bars) - sum(1 for u in self.roster.board() if u.slotless and u.loc[1:] in bars)
                n_board += sum(u.slots - 1 for u in self.roster.board() if u.slots > 1)
                # the game's own count (the n of its "n/m") leaves plants out even when our records lost them (run
                # 20261015-161625 5-6..6-1: "9/9 full" with two untracked plants); it can lag a move by a second, the
                # bars cannot - the lower of the two, and the drag is checked on screen anyway
                if getattr(scr, 'board_count', None) is not None:
                    n_board = min(n_board, scr.board_count)
                if n_board + unit.slots > cap:
                    # the board is full (run 152203: two untracked Varus held slots): a drop on an empty
                    # hex is refused, so say so instead of dragging
                    if n_board > self.roster.slots_used():
                        # more on screen than in the records: a unit they lost stands on the board (run 20261020-110532
                        # 4-2: Yorick, fielded by the game unrecorded, had Sentinel refused three times) - a board check
                        self.need_survey = self.need_survey or (f'{n_board} units on the board, '
                                                                 f'{self.roster.slots_used()} in the records')
                    self._board_full_at = scr.stage    # the deployment swaps from now on (deployment_swap_target)
                    self.fail_cause = (f'the board is full ({n_board}/{cap} units on screen): put a unit on the '
                                       f'bench or sell one first, or drag onto an occupied hex to swap')
                    return False
            if dst[0] == 'bench' and not self.confirm_before_loss(unit, scr, 'unfield'):
                return False
            if src[0] == 'board' and dst[0] == 'bench' and not REPLAY and self.confirm_incoming(scr):
                return False                     # the unit it made room for is another champion: the plan is redone
            if dst[0] == 'bench' and not REPLAY and self.roster.at(dst) is None:
                # the slot is looked at before the drop: a container or an object there takes no unit (run 20261016-011010
                # 2-2: Rakan dragged onto a component anvil the records thought was a free slot)
                got = self.cap.read()            # (after any panel read above: a fresh frame)
                if got is not None:
                    dst = self.bench_drop_slot(unit, src, dst, got[0], scr.stage)
                    if dst is True or dst is None:
                        return dst is True
            if src[0] == 'board' and dst[0] == 'bench' and self.w_to_bench(unit, src):
                return True                      # W with the cursor on the unit: faster and exact
            # a unit is read off its panel before it is fielded only while its identity is in doubt (never read, or
            # unverified): one read off its panel, or seen landing from our own buy, is tracked by our own actions
            # (the owner, 2026-11-08: "you do not need to check for every action, it is a waste of time")
            read_first = not getattr(unit, 'sure', False) or bool(unit.unverified)
            if (src[0] == 'bench' and dst[0] == 'board' and unit.name and not REPLAY and read_first
                    and self.confirm_before_field(unit, src, scr)):
                return False                     # it is another champion: records fixed, the model is asked again
            occupant = self.roster.at(dst) if dst[0] == 'board' else None
            if src[0] == 'bench' and occupant is not None and occupant is not unit and not REPLAY:
                if not self.confirm_before_loss(occupant, scr, 'unfield'):
                    return False
            # the board's bars, the unit's point and the picture before the drag: one fresh frame
            got = self.cap.read() if not REPLAY else None
            pre = (set(B.occupancy(got[0], exclude=self.watcher.tac[1])[1]) if got else None) if not REPLAY else self.board_now()
            start = loc_point(src)
            if src[0] == 'bench' and not REPLAY:          # a small bench unit is not under its slot's point
                start = self.unit_point(src, B.green_bars(got[0]) if got else None)
            if a.why.startswith('plan') and action_plan is not None and (
                    self.round_plan is not action_plan or getattr(self, '_superseded_rp', None) is action_plan):
                self.deferred = True
                self.fail_cause = 'a newer lineup arrived before the drag: the new plan runs'
                return False
            self._move_before = got[0].copy() if got else None
            inp.drag(start, loc_point(dst))
            took = self.move_took(src, dst, pre=pre)
            if took == 'elsewhere':
                took = self.fix_landing(unit, dst)
            elif took == 'swap' and not REPLAY and unit.name and not getattr(a, 'mirror', False):
                # a swap: the bars look the same before and after, so read who stands there now (a click)
                swap = self.swap_left_another(unit, src, dst)
                if swap is True:
                    return False
                if swap == 'unconfirmed':
                    self.need_survey = f'no panel confirmed the drag from {src} to {dst}'
                    self.survey_places = (self.need_survey, self.move_changes(None, None, src, dst)[1])
                    self.deferred, self._repair_pending = True, True
                    self.fail_cause = 'the swap is unconfirmed: its places are being read before retrying'
                    return False
            if not took:
                got = self.cap.read()
                said = self.game_message(got[0]) if got is not None and not REPLAY else None
                self.fail_cause = getattr(self, 'landing_note', None) or \
                    (f'the unit did not move; the game said: "{said}"' if said else
                     'the unit did not move: the drop missed or the target was refused')
                self.landing_note = None
                return False                     # roster unchanged: it still matches the screen
            self.roster.moved(unit, dst)
            return True
        if a.kind == 'lock':
            self.inp.click(B.SHOP_LOCK)          # the shop's padlock (shop_lock_action)
            time.sleep(0.15)
            self.log(ev='shop_lock', stage=scr.stage, why=a.why)
            return True
        if a.kind == 'item':
            if self.hold_item_over(a, scr):
                return False
            return self.give_item(a, scr)
        if a.kind == 'link':
            loc, target = a.arg
            if self.phase.planning and scr.timer is not None and scr.timer <= 2:
                self.fail_cause = 'too late in planning for a link (it takes ~1.5 s): kept for the next round'
                self.deferred = True
                return False
            # data for checking a link on screen (the progress bar, what changes once linked): frames from just
            # before the press to 1.5 s after the release, in runs/<ts>/links/<n>_<t>.jpg
            shots, stop = [], threading.Event()

            def film():
                t0 = time.time()
                while not stop.is_set() and time.time() - t0 < 4.0:
                    raw = getattr(self.cap, 'raw', None)
                    if raw is not None:
                        shots.append((time.time() - t0, raw.copy()))
                    time.sleep(0.2)
            filmer = threading.Thread(target=film, daemon=True)
            filmer.start()
            inp.drag_hover(loc_point(loc), loc_point(target.loc), hover=1.0)
            time.sleep(1.5)
            stop.set()
            filmer.join(timeout=1)
            n = self.__dict__.setdefault('_links_n', [0])
            n[0] += 1
            folder, k = self.log_dir / 'links', n[0]

            def save(shots=shots):
                from PIL import Image
                folder.mkdir(exist_ok=True)
                for t, f in shots:
                    Image.fromarray(f).resize((f.shape[1] * 1080 // f.shape[0], 1080)).save(
                        folder / f'{k:02d}_{scr.stage}_{t:4.1f}s.jpg', quality=85)
            if test_mode():                      # data for learning the link's look: test games only
                threading.Thread(target=save, daemon=True).start()
            got = self.cap.read()
            said = self.game_message(got[0]) if got is not None and not REPLAY else None
            # no on-screen check of the link yet (its look is to be learned from a recorded game): the game's own
            # notice, if it shows one, is passed on
            self.log(ev='link', stage=scr.stage, src=list(loc), to=repr(target), said=said)
            # the BFF goes to the target (the user, 2026-10-12: drag the rider, hover the other Sprykin, "and then the
            # BFF went to the new champion"); from the BFF itself (before its first rider) it leaves its own hex
            src = self.roster.at(loc)
            from tft_player.trait_text import pair_hosts
            hosts = pair_hosts()
            if target.name in hosts or (src is not None and src.name in hosts):
                # an ally dragged onto Taric: the two are paired (a line between them) and stay paired through moves
                # and rounds (the user, 2026-10-13) - this is no BFF rider
                host, ally = (target, src) if target.name in hosts else (src, target)
                if ally is not None:
                    for r in self.roster.units:
                        if getattr(r, 'paired', None) in (host.id, ally.id):
                            r.paired = None
                    host.paired, ally.paired = ally.id, host.id
                    self.log(ev='paired', stage=scr.stage, host=repr(host), ally=repr(ally))
                if said:
                    self.news_event('link', scr.stage, f'link {loc[1:]} -> {target}: the game said "{said}"')
                return True
            if src is not None and src.name == BFF:
                self.bff_hex, self.bff_assigned = src.loc, True
                self.roster.units.remove(src)
            for r in self.roster.units:
                if getattr(r, 'riding', None) and r is not target:
                    r.riding = None
            target.riding = True
            if said:
                self.news_event('link', scr.stage, f'link {loc[1:]} -> {target}: the game said "{said}"')
            return True
        if a.kind == 'strip':
            slot, tool, unit = a.arg
            # no panel read first unless the unit's identity is in doubt (the owner, 2026-11-08: "remover should be
            # <1s, instantly" - game 74b 4-6's Remover took 3.5 s, most of it a panel read of Murkwolf)
            # (or after a Remover that brought nothing back from it: its row may show icons that are no items - game 84
            # 3-3: Murkwolf's row read a Hand of Justice, the panel none, five drops)
            checked = not REPLAY and bool(unit.unverified or getattr(unit, 'strip_missed', False))
            self.loss_read = None
            if checked and not Player.confirm_before_loss(self, unit, scr, 'strip', force=True):
                return False
            from tft_player.items import bench_filled, bench_slot_crop
            got = self.cap.read()
            if got is None:
                self.fail_cause = 'no frame to read the Remover source'
                return False
            picture = read_item_bench(got[0])
            source = picture[slot] if slot < len(picture) else None
            from tft_player.items import is_reforger
            matches = is_remover if is_remover(tool) else is_reforger
            if source is None or source != 'unknown' and not matches(source):
                # A successful last-use strip can have closed up the bench.
                # Never retry by dragging the item that inherited the tool's slot.
                self.item_track.confirm(slot, source)
                self.bench_items = self.item_track.names()
                self.deferred = True
                self.fail_cause = f'strip source changed: slot {slot} shows {source}, not {tool}'
                self.log(ev='item_delivery', status='source_changed', stage=scr.stage,
                         unit_id=unit.id, slot=slot, requested=tool, seen=source)
                return False
            _, row = self.unit_row(got[0], unit)
            seen = [n for n, _, _ in row] if row is not None else None
            if not checked and (seen is None or Counter(seen) != Counter(unit.items or ())):
                if not Player.confirm_before_loss(self, unit, scr, 'strip', force=True):
                    return False                 # only a discrepant row pays for a panel read
            shown = getattr(self, 'loss_read', None)
            if (shown and shown[0] == unit.id and shown[1] == unit.name and isinstance(shown[3], list)
                    and not shown[3] and unit.items):
                # its panel shows no items: the record's were never on it (game 84 3-3..3-5: a Hand of Justice read
                # off Murkwolf's row, the panel empty; the Remover dropped on it five times, "the items did not come
                # back", and the Hand's give to Mama Beak waited on it). The record is corrected; nothing to take off
                self.log(ev='strip_nothing', stage=scr.stage, unit=repr(unit), was=list(unit.items))
                self.news_event('strip', scr.stage, f'{tool} not used on {unit}: its panel shows no items '
                                                    f'(the record had {", ".join(unit.items)}); the record is corrected')
                unit.row_ghosts = set(getattr(unit, 'row_ghosts', None) or ()) | set(unit.items)
                unit.items.clear()
                self.deferred = True
                self.fail_cause = f'{unit} holds no items (its panel): nothing for the {tool} to take off'
                return False
            before = sum(bench_filled(got[0]))
            tool0 = bench_slot_crop(got[0], slot).astype(np.int16)      # the tool's own slot before the drag
            # onto the unit's body, as item gives are (2026-10-16): the hex centre or the bench slot's floor point
            # missed (benchmark game 20261018-150046: Ornn on bench slot 5, Shen at (3,4), Rek'Sai at (3,2) - "the
            # items did not come back", three times)
            bars = B.green_bars(got[0]) if got is not None and not REPLAY else None
            if unit.on_board:
                own = next((b for b, place in B.bar_places(None, bars=bars or [])[2] if place == unit.loc), None)
                point = Player.drop_point(self, unit, own, scr.stage)[0] if own is not None else None
            else:
                point = self.unit_point(unit.loc, bars) if bars is not None else None
            inp.drag(slot_point(slot), point or loc_point(unit.loc))
            # a tool on its last use leaves the item bench and the items after it move up a slot (game 77 3-3 and 4-1:
            # the Remover on Mama Beak worked - Titan's Resolve back, the Remover gone, one slot more filled, not two -
            # and was taken for a miss five times; each retry dragged the item now in its slot, Titan's back onto Mama
            # Beak, then a "Bramble Vest" that was the Titan's onto Sentinel). A used-up tool: one filled slot fewer and
            # its own slot's picture changed (the next item moved in, or empty); a missed drag leaves the tool there
            last_use = (uses_left(tool) or 1) <= 1
            took, end, streak, moved = False, time.time() + 2.0, 0, False
            while time.time() < end:             # the items come back to the bench a moment later (up to ~2 s)
                got = self.cap.read() or got     # (each read is the next frame: no sleep)
                n_now = sum(bench_filled(got[0]))
                moved = last_use and float(np.abs(bench_slot_crop(got[0], slot).astype(np.int16) - tool0).mean()) > 25
                ok_now = n_now >= before + len(unit.items) or (moved and n_now >= before + len(unit.items) - 1)
                streak = streak + 1 if ok_now else 0
                if streak >= 2:
                    took = True
                    break
            now = read_item_bench(got[0]) if took else None      # their names, once (~27 ms)
            if not took:
                unit.strip_missed = True
                self.fail_cause = 'the items did not come back to the item bench (missed the unit, or the tool could not be used)'
                return False
            self.brain.stripped.add(unit.id)
            self.news_event('strip', scr.stage, f'{tool} used on {unit}: {", ".join(unit.items)} back on the item bench')
            # the fight reads no item bench: the tracked one stands in, and the transfer's gives look for them there
            consumed = bool(last_use and moved)
            self.item_track.returned(list(unit.items) if is_remover(tool) else ['unknown'] * len(unit.items),
                                     now, slot, tool_consumed=consumed)
            if consumed:
                Player.bench_shifted(self, slot)
            self.bench_items = self.item_track.names()
            self.log(ev='item_delivery', status='stripped', stage=scr.stage, unit_id=unit.id,
                     stripped=list(unit.items), tool=tool, tool_consumed=consumed, bench=self.bench_items)
            unit.items.clear()
            sig = (getattr(self, 'item_sigs', None) or [None] * N_SLOTS)[slot]
            n = uses_left(self.item_names.get(sig, tool))
            if sig in self.item_names and n:
                self.item_names[sig] = re.sub(r'\(\d+ uses', f'({n - 1} uses', self.item_names[sig])
            return True
        if a.kind == 'level':
            n = a.arg if isinstance(a.arg, int) and a.arg > 0 else 1
            gold0 = scr.gold if scr.gold is not None else getattr(self.round_plan, 'last_gold', None)
            fight = getattr(getattr(self, 'phase', None), 'planning', True) is False
            # in a fight XP runs when the gold after it stays at 50 or more: it loses no interest (the owner, game 74
            # x25.4 - the planner hands such a step out, and this hold refused it: game 75 3-5, 3 XP buys at 65 gold)
            rich = fight and gold0 is not None and gold0 - 4 * n >= 50
            if (fight and not rich) or (not fight and scr.timer is not None and scr.timer <= 5):
                a.landed = 0
                self.deferred = True
                self.fail_cause = 'XP held until the next planning phase'
                return False
            # n presses in a row, ONE check at the end (the user: like a player pressing F n times)
            for _ in range(n):
                inp.key('f')
                time.sleep(0.06)                 # released over two game frames too, or two presses read as one
            # the HUD shows the result a few frames later - sometimes later than 0.25 s (run 20261015-003101 3-1:
            # 27 -> 27 read, 23 a moment later, and the 'missed' press was bought again): read until the gold shows
            # all n presses, or stops changing, up to 1.2 s
            gold1, end, prev = None, time.time() + 1.2, None
            while time.time() < end:
                got = self.cap.read()            # (the next frame: its wait is the pace, no sleep)
                g = self.reader.read_shop(got[0])[0] if got else None
                if g is not None:
                    gold1 = g
                    if gold0 is None or gold0 - g >= 4 * n or (g == prev and gold0 - g > 0):
                        break
                    prev = g
            if gold1 is None and gold0 is not None:
                gold1 = self.level_unread_check(scr, n, gold0)
            if gold0 is not None and gold1 is not None:
                a.landed = max(0, min(n, round((gold0 - gold1) / 4)))
            else:
                a.landed = n                     # unread: the next state shows the level anyway
            if a.landed < n:
                self.fail_cause = f'{n - a.landed} of {n} XP buys did not land (gold {gold0} -> {gold1})'
            self.note_xp_per_buy(scr, a.landed)
            if gold0 is not None and gold1 is not None and a.landed:
                self.confirm_level_buys(scr, a.landed)
            return a.landed > 0
        if a.kind == 'rolldown':
            return self.run_rolldown(a.arg, scr)
        if a.kind == 'roll':
            raw0 = list(scr.shop_raw)
            inp.key('d')
            self.brain.rolled_this_round += 1
            # the next decision must see the NEW shop: wait until the cards changed and read the
            # same twice in a row (a card caught mid-flip reads as garbage, and the shop would be
            # rolled past without its champions being looked at)
            seen = []

            def settled(s):
                seen.append(list(s.shop_raw))
                return (s.shop_visible and sum(1 for x, y in zip(s.shop_raw, raw0) if x != y) >= 3
                        and len(seen) >= 2 and seen[-1] == seen[-2])
            if not self.wait_for(settled, timeout=1.5):
                self.log(ev='roll_unsettled', stage=scr.stage, before=raw0, reads=seen[-3:])
            return True
        if a.kind == 'augment':
            from tft_player.coach import owner_banned_augments
            names = getattr(self, 'augment_read', None) or scr.augments
            if names[a.arg] in owner_banned_augments():
                self.fail_cause = f'{names[a.arg]} [never picked in this comp: do not pick it; you may reroll this card]'
                return False
            inp.click(AUGMENT_CARDS[a.arg])
            time.sleep(0.4)
            return True
        return False

    def wait_card(self, i, sig0, timeout, changed):
        """Shop card i was bought: its strip differs from sig0 (the card left the slot), on two frames in a
        row. sig0 None (no picture before): accepted after a short wait."""
        if sig0 is None or REPLAY:
            time.sleep(0.1)
            return True
        end, streak = time.time() + timeout, 0
        while time.time() < end:
            got = self.cap.read()                # (the next frame: its wait is the pace, no sleep)
            if got is None:
                continue
            sig = self.reader.shop_sig(got[0])
            streak = streak + 1 if changed(sig[i], sig0[i]) else 0
            if streak >= 2:
                return True
        return False

    def park_tactician(self, scr):
        """Once a planning phase, when no loot is left to collect and the units have walked back: walk our
        tactician to PARK (one right-click on the ground), so it hides no unit or bar. Usually it already stands
        there (the game puts it there after the trip home): no click then."""
        if REPLAY or getattr(self, '_parked', None) == scr.stage or time.time() - self.round_started < 1.5:
            return
        if scr.timer is not None and scr.timer < 4:
            return
        if self.loot.target is not None or time.time() - getattr(self, '_orbs_seen_at', 0) < 4.5:
            # an orb is still out (orbs blink out of detection, and PvP rounds scan every ~2 s): parking now walks the
            # tactician away from it (run 20261010-064944 2-3: parked twice on the way, the orb took 8 s)
            return
        self._parked = scr.stage
        feet = self.watcher.tac[0] if getattr(self, 'watcher', None) is not None else None
        if feet is not None and math.dist(feet, PARK_FEET) < 60:
            return                               # already there
        try:
            self.inp.click(PARK, 'right')
        except NotForeground:
            return
        self._panel_close_at = time.time()       # the same click closes an open unit panel (close_unit_panel)
        self.log(ev='parked', stage=scr.stage)

    def close_unit_panel(self, frame, scr, combat=False):
        """The unit info panel our own reads leave open covers the player list, and with it our HP box: HP goes unread.
        Run 20261022-225248: the panel was open at 20 of 33 planning-phase screenshots, every one from 6-3 on - the
        state said HP 54 in the final duel while the screen said 39, 24, 8. It closes on our next click on the ground
        (4-6 and 6-7 there: open at the phase's start, closed after park_tactician's right-click). So while this phase's
        HP is unread and the panel is up: one right-click at PARK (where the tactician waits anyway), at most 2 a
        phase, 3 s apart; never on loot, in a phase's first 2 s or its last 5.
        combat: a player fight's damage is read at its end off the same HP box (hp_filter's _hp_after). A panel opened
        by the last planning seconds' reads (here never closed in the last 5) stayed up through the fight: run
        20261027-133823 lost 3-1, 3-3, 3-6 and 4-3 with a panel open at the fight's start, none of the four losses was
        read, and the next rounds' first questions went with 70, 60, 51 and 34 for 60, 51, 44 and 24 - every lost fight
        with no panel was read. So in a player fight the panel is closed too: from 2 s in, at most 3 a fight, 3 s apart,
        until its damage is read."""
        if REPLAY or not scr.shop_visible:
            return False
        now = time.time()
        if combat:
            from tft_player.planner import is_pvp
            after = getattr(self, '_hp_after', None)
            if not is_pvp(scr.stage) or (after is not None and after[1] == scr.stage):
                return False                     # no player fight, or its damage is read
            seen = getattr(self, '_combat_seen', None)
            if not seen or seen[0] != scr.stage:
                self._combat_seen = seen = (scr.stage, now)
            if now - seen[1] < 2.0 or now - getattr(self, '_panel_close_at', 0) < 3.0:
                return False
            key, most = (scr.stage, 'combat'), 3
        else:
            if scr.timer is None or scr.timer < 5:
                return False
            if getattr(self, '_hp_phase', None) == scr.stage:
                return False                     # this phase's HP is read
            if now - getattr(self, 'round_started', now) < 2.0 or now - getattr(self, '_panel_close_at', 0) < 3.0:
                return False
            key, most = scr.stage, 2
        if self.loot.target is not None or now - getattr(self, '_orbs_seen_at', 0) < 4.5:
            return False                         # a click on the ground would walk the tactician off an orb
        if not self.reader.unit_panel_open(frame):
            return False
        if getattr(self, '_panel_close_stage', None) != key:
            self._panel_close_stage, self._panel_close_tries = key, 0
        if self._panel_close_tries >= most:
            return False
        self._panel_close_tries += 1
        self._panel_close_at = now
        try:
            self.inp.click(PARK, 'right')
        except NotForeground:
            return False
        self.log(ev='panel_closed', stage=scr.stage, try_=self._panel_close_tries, combat=combat)
        return True

    def sell_seen(self, gold0, timeout=1.5):
        """-> the gold after the sell, or False. A sell happened only when the gold rose (the user, 2026-10-12: "sell should only be confirmed by the change of
        the gold, not by whether it pressed E"). The unit's bar going is no proof: at combat start every unit flies to the
        other board and every bar goes at once (run 20261012-194821 2-5: E on Rek'Sai at t=1 hit an empty slot, the
        empty slot was taken for a sell, and Rek'Sai stayed on the bench). -> True / False (no rise in time: not sold,
        tried again)."""
        if gold0 is None:
            return False
        end = time.time() + timeout
        while time.time() < end:
            got = self.cap.read()                # (the next frame: its wait is the pace, no sleep)
            if got is None:
                continue
            g = self.reader.read_shop(got[0])[0]
            if g is not None and g > gold0:
                return g                          # the gold after: what it brought tells which unit went
        return False

    def board_now(self):
        """The board hexes our health bars show right now (a fresh frame)."""
        got = self.cap.read()
        return set(B.occupancy(got[0], exclude=self.watcher.tac[1])[1]) if got else None

    @staticmethod
    def sell_read_first(u):
        """A sell can't be undone: a unit whose identity is in doubt - unverified, or named by a correction (a
        misread there can swap two records) - is read off its panel before it is sold."""
        return bool(u.unverified or getattr(u, 'corrected', None))

    @staticmethod
    def completed_items(items):
        from tft_player.items import RECIPES
        return sorted(i for i in items or [] if i in RECIPES or str(i).startswith('Radiant '))

    @staticmethod
    def holds_completed(u):
        return bool(Player.completed_items(u.items))

    def plan_view(self, u):
        return getattr(getattr(self, 'round_plan', None), 'names', {}).get(u.id)

    def confirm_before_loss(self, u, scr, what, force=False):
        """e207: positive mismatches refuse; unread completed items hold only this tick. A sell (it cannot be undone)
        reads a fielded or item-holding unit's panel first; a unit taken off the board (a drag back) only while its
        identity is in doubt (the owner, 2026-11-08: "you do not need to check for every action").
        """
        meant = Player.plan_view(self, u)
        completed = Player.holds_completed(u)
        renamed = meant is not None and meant[0] != u.name
        uncertain = Player.sell_read_first(u) or getattr(u, 'star_assumed', False)
        doubted = uncertain or not getattr(u, 'sure', False)
        rp = getattr(self, 'round_plan', None)
        valuable = (u.on_board or completed) if what == 'sell' else (completed and doubted)
        if REPLAY or not (valuable or renamed or uncertain or force):
            return True
        expected_name, expected_star = u.name, u.star
        expected_items = getattr(rp, 'items_view', {}).get(u.id, tuple(u.items))
        got = self.cap.read()
        bars = B.green_bars(got[0]) if got else None
        pt = self.unit_point(u.loc, bars)
        others = [self.unit_point(o.loc, bars) for o in self.roster.units
                  if o is not u and o.name and o.name != u.name and not o.missing]
        self._loss_loc = u.loc
        try:
            name, stars, items = self.read_fresh(pt, others)
        finally:
            self._loss_loc = None
        self.loss_read = (u.id, name, stars, items)      # what the panel showed (a strip reads its items)
        corrected = getattr(u, 'loss_corrected', None) == (u.name, u.star, tuple(u.items))
        if corrected:
            expected_items = tuple(u.items)
        mismatch = bool(name and (
            (renamed and (meant[0] is not None or completed) and not corrected)
            or ((expected_name is not None or completed) and name != expected_name)
            or (stars and expected_star and stars != expected_star)
            or (isinstance(items, list) and Player.completed_items(items) and 'unknown' not in items
                and Player.completed_items(items) != Player.completed_items(expected_items))))
        # the panel's name and stars are the unit's and every item it names is one of the record's: an icon read as
        # "unknown" is an icon our banks lack, not another unit (game 78: Caitlyn's sale held at every try from 3-6 to
        # 5-3 - her Hand of Justice always read "unknown" - two completed items idle on the bench for nine rounds)
        known = [i for i in items if i and i != 'unknown'] if isinstance(items, list) else []
        confirmed = (bool(name) and name == expected_name and (not stars or not expected_star or stars == expected_star)
                     and isinstance(items, list) and len(items) == len(expected_items)
                     and not (Counter(known) - Counter(expected_items)))
        unread = name is None or (completed and isinstance(items, list) and 'unknown' in items and not confirmed)
        if unread or mismatch:
            after = self.cap.read()
            if hasattr(self, 'log_dir') and after:
                self.snap(after[0], f'{scr.stage}-loss-{what}-{u.id}', True, force=True)
                if got:
                    self.snap(got[0], f'{scr.stage}-loss-{what}-{u.id}-before', True, force=True)
                self.log(ev='loss_panel_read', stage=scr.stage, unit_id=u.id, loc=u.loc, point=pt,
                         read=name, stars=stars, items=items, expected_items=expected_items,
                         raw=getattr(self.reader, 'last_panel_text', None),
                         panel_before=getattr(self, 'panel_before', None),
                         positive_mismatch=mismatch, unread=unread)
        if not mismatch and not (unread and (completed or force)):
            if name:
                u.name, u.star, u.sure = name, stars or u.star, True
                u.unverified = u.corrected = None
            return True
        why = (f'panel reads {name} {stars or "?"}-star, completed items '
               f'{Player.completed_items(items)}; plan saw {expected_name or "?"} '
               f'{expected_star}-star, completed items {Player.completed_items(expected_items)}'
               if mismatch else 'panel still unreadable after retry; completed-item unit held this tick')
        self.fail_cause = why
        self.log(ev=f'{what}_refused', stage=scr.stage,
                 unit=repr(u), read=name, stars=stars, items=items, planned_as=expected_name,
                 positive_mismatch=mismatch, held_tick=not mismatch, point=pt)
        self.news_event('unit_corrected' if mismatch else 'unit_unreadable', scr.stage,
                        f'{what} of unit {u.id} not done: {why}')
        if not mismatch:
            self.deferred = True
            return False
        if name:
            before_identity = (u.name, u.star)
            u.name, u.star, u.sure = name, stars or u.star, True
            if isinstance(items, list) and 'unknown' not in items:
                u.row_ghosts = set(getattr(u, 'row_ghosts', ()) or ()) | (set(u.items) - set(items))
                u.items, u.items_read = list(items), True
            u.loss_corrected = (u.name, u.star, tuple(u.items))
            Player.identity_corrected(self, u, before_identity, scr)
        self.need_survey = f'{what} refused at {u.loc}: panel differs from the records'
        self.survey_places = (self.need_survey, [u.loc])
        self._repair_pending = what == 'unfield'
        # A refused loss must not return via inherit_carry with fresh identities (e192 F).
        if isinstance(getattr(rp, 'p', None), dict):
            if what == 'sell':
                rp.p['sell'] = [i for i in rp.p.get('sell') or [] if i != u.id]
                getattr(rp, 'sold', set()).add(u.id)
            elif what == 'unfield':
                # Hold this tick. After the targeted check, the intended swap
                # can run against the corrected occupant and its actual items.
                self.deferred = True
        return False

    def bench_drop_slot(self, unit, src, dst, frame, stage):
        """The bench slot a drop goes to, looked at first: a container or an object there takes no unit. -> the slot
        to drop on (another free one when it is taken), True when W benched the unit already, None when no slot is
        free (self.fail_cause says why)."""
        from tft_player import bench as BN
        bench_seen = B.bar_places(frame, exclude=self.watcher.tac[1])[0]
        kinds = BN.classify(frame, bench_seen)
        if kinds[dst[1]]['kind'] == 'empty':
            return dst
        if kinds[dst[1]]['kind'] == 'interactable':
            self.roster.blocked = {**self.roster.blocked, dst[1]: f"a container ({kinds[dst[1]]['cls']})"}
        free = [i for i, k in enumerate(kinds) if k['kind'] == 'empty' and self.roster.at(('bench', i)) is None]
        if (not free and src[0] == 'board' and kinds[dst[1]]['kind'] in ('unknown', 'obscured')
                and self.w_to_bench(unit, src)):
            # the slots are covered, not taken (run 20261030-050404 5-6: the "Taric pairing was auto assigned!" banner
            # lay over the bench, every free slot read unknown, Kog'Maw was never benched and Elder Dragon not
            # fielded): W lets the game pick a free slot, checked by the bars
            return True
        if not free:
            self.fail_cause = (f"bench slot {dst[1]} holds {kinds[dst[1]]['cls'] or 'something'} and no "
                               f"bench slot is free")
            return None
        self.log(ev='bench_slot_taken', stage=stage, slot=dst[1], kind=kinds[dst[1]]['kind'],
                 cls=kinds[dst[1]]['cls'], instead=free[0])
        return ('bench', free[0])

    def w_to_bench(self, unit, src):
        """W with the cursor on a fielded unit sends it to the bench (the game's own hotkey: no drag, no
        hex to miss). True when a new bench bar appeared and a board bar went; the unit's slot is then
        the one the screen shows. False -> the caller drags instead."""
        got = self.cap.read()
        if got is None or REPLAY:
            return False
        tac = self.watcher.tac[1]
        bench0, board0 = B.occupancy(got[0], exclude=tac)
        bars = B.green_bars(got[0])
        try:
            self.inp.key('w', at=self.unit_point(src, bars))
        except NotForeground:
            return False
        streak, end = 0, time.time() + 0.8
        while time.time() < end:
            time.sleep(0.07)
            got = self.cap.read()
            if got is None:
                continue
            bench, board = B.occupancy(got[0], exclude=tac)
            new = [i for i in range(9) if bench[i] and not bench0[i]]
            streak = streak + 1 if new and len(board) < len(board0) else 0
            if streak >= 2:
                gone = set(board0) - set(board)
                if len(gone) == 1 and tuple(src[1:]) not in gone and tuple(src[1:]) in board:
                    # W took the unit next to it (run 20261029-051106 6-3: "Maokai@h22 -> bench" benched the
                    # Scuttlecrab** at (3,2), and the records called it Maokai until 6-5): who went is recorded,
                    # it is put back, and the caller drags the unit meant
                    return self.w_took_another(src, gone.pop(), new[0])
                self.roster.moved(unit, ('bench', new[0]))
                self.log(ev='w_to_bench', unit=repr(unit), slot=new[0])
                return True
        self.log(ev='w_failed', unit=repr(unit), src=src)
        return False

    def w_took_another(self, src, hex_gone, slot):
        """W meant for the unit at src sent the one at hex_gone to bench slot `slot`. The records follow the screen,
        the other unit is dragged back to its hex (checked by its bar). -> False: the caller drags src's unit."""
        other = self.roster.at(('board', *hex_gone))
        self.log(ev='w_wrong_unit', src=list(src[1:]), went=list(hex_gone), slot=slot,
                 unit=repr(other) if other is not None else None)
        if other is None:
            return False
        self.roster.moved(other, ('bench', slot))
        try:
            self.inp.drag(loc_point(('bench', slot)), loc_point(('board', *hex_gone)))
        except NotForeground:
            return False
        end = time.time() + 0.8
        while time.time() < end:
            time.sleep(0.07)
            got = self.cap.read()
            if got is None:
                continue
            bench, board = B.occupancy(got[0], exclude=self.watcher.tac[1])
            if tuple(hex_gone) in board and not bench[slot]:
                self.roster.moved(other, ('board', *hex_gone))
                return False
        self.log(ev='w_put_back_failed', unit=repr(other), slot=slot)
        return False

    def fix_landing(self, unit, dst):
        """The drop landed on another hex than dst (move_took saw where). Read the unit there off its info
        panel (a click, ~0.4 s) to be sure it is this one, record the true place, and drag it to dst.
        -> True when it now stands on dst."""
        landed = self.landed_at
        got = self.cap.read()
        if landed is None or got is None:
            return False
        bars = B.green_bars(got[0])
        name, _, _ = self.read_unit(self.unit_point(('board', *landed), bars))
        if unit.name and name and name != unit.name:
            self.need_survey = f'a drop to {dst[1:]} left {unit.name} unseen; {name} stands at {landed}'
            self.survey_places = (self.need_survey, [dst, ('board', *landed)])
            self.landing_note = f'the unit is not where the drop went; {name} stands at {landed}'
            return False
        unit.loc = ('board', *landed)
        self.log(ev='landed_elsewhere', unit=repr(unit), aimed=dst, landed=landed)
        pre = self.board_now()
        self.inp.drag(loc_point(unit.loc), loc_point(dst))
        again = self.move_took(unit.loc, dst, pre=pre)
        if again == 'elsewhere':
            unit.loc = ('board', *self.landed_at)
            self.landing_note = f'the unit stands at {self.landed_at}, not {dst[1:]} (dragged twice)'
            return False
        return bool(again)

    def move_took(self, src, dst, timeout=0.7, pre=None):
        """Did a drag src -> dst happen? Checked on fresh frames from our units'
        health bars. A drag onto an occupied spot swaps two units, which bars
        can't tell apart from no change and requires swap_left_another's check."""
        from tft_player.state import _hex_near
        if self.roster.at(dst) is not None:
            time.sleep(0.08)
            return 'swap'
        end = time.time() + timeout
        streak = 0                       # a bar can blink out for one frame: need two in a row
        others_benched = any(u.loc[0] == 'bench' and u.loc != src for u in self.roster.units if not u.missing)
        while time.time() < end:
            got = self.cap.read()                # (the next frame: its wait is the pace, no sleep)
            if got is None:
                continue
            bench, board = B.occupancy(got[0], exclude=self.watcher.tac[1])
            lights = None
            if dst[0] == 'bench' and others_benched and not any(bench):
                # no bench bar at all while other units stand there: the round flipped to combat (bench units show
                # no bars in a fight) - the tiles tell (benchmark game 20261018-161404 1-4: Akali went from slot 2 to
                # 3 at 1 s, "did not move" kept her at 2, and five combat sells pressed E on the empty slot)
                lights = self.bench_lights(got[0])
            took = (src[0] == 'bench' and dst[0] == 'board' and not bench[src[1]]) or \
                   (dst[0] == 'bench' and lights is None and bench[dst[1]] and (src[0] != 'bench' or not bench[src[1]])) or \
                   (dst[0] == 'bench' and lights is not None and lights[dst[1]] is True
                    and (src[0] != 'bench' or lights[src[1]] is False)) or \
                   (src[0] == 'board' and dst[0] == 'board' and any(_hex_near(dst[1:], h) for h in board)
                    and not any(h == src[1:] for h in board)) or \
                   (src[0] == 'board' and dst[0] == 'board' and pre is not None and dst[1:] in board
                    and dst[1:] not in pre)
            # (the last: a bar new on exactly dst is the unit arriving, even when src still shows one - a tall
            # neighbour's bar maps onto src: run 20261018-161404 5-2, Vi stood on (3,3) and "did not move" twice,
            # Kayle's bar at (0,0) read as (1,0))
            streak = streak + 1 if took else 0
            if streak >= 2:
                # it moved - but WHERE? a bar on exactly dst, or a new bar somewhere else (the drop went to
                # another hex: the roster must not be told it is on dst)
                self.landed_at = None
                if dst[0] == 'board' and pre is not None and dst[1:] not in board:
                    new = sorted(board - pre)
                    if len(new) == 1:
                        self.landed_at = new[0]
                        return 'elsewhere'
                return True
        self.log(ev='move_failed', src=src, dst=dst)
        return False

    @staticmethod
    def move_changes(before, after, src, dst):
        """Changed bodies at the source, target and adjacent source positions."""
        import cv2
        from tft_player.state import _hex_near
        places = [src, dst]
        if src[0] == 'bench':
            places += [('bench', i) for i in (src[1] - 1, src[1] + 1) if 0 <= i < 9]
        else:
            places += [('board', r, c) for r in range(4) for c in range(7)
                       if (r, c) != src[1:] and _hex_near((r, c), src[1:])]
        if before is None or after is None:
            return [], places

        def bodies(frame):
            h, w = frame.shape[:2]
            scaled = cv2.resize(frame, (round(w * 1080 / h), 1080))
            crops = []
            for loc in places:
                x, y = loc_point(loc)
                x = round(scaled.shape[1] / 2 + x - 960)
                crops.append(scaled[round(y) - 90:round(y) + 10, x - 40:x + 40].astype(np.float32))
            return crops

        changed = [loc for loc, a, b in zip(places, bodies(before), bodies(after))
                   if a.shape == b.shape and a.size and float(np.abs(a - b).mean()) > 14]
        return changed, places

    def swap_left_another(self, unit, src, dst):
        """After a swap onto an occupied hex (bars look the same before and after) the unit on dst is read off its
        panel. -> True when it is another champion (the records are checked by a survey) or when no panel is there and
        the moved unit still stands where it came from (the swap did not happen); 'unconfirmed' when neither place
        reads; else False. Only a panel that changed
        for this click counts: a click that missed leaves the panel of the unit selected before (benchmark game
        20261018-161404 1-4: Kayle did land on (0,0), the read gave the benched Leona selected earlier, "after the
        swap Leona stands at (0,0)" failed the step and its retry swapped Kayle back)."""
        time.sleep(0.15)
        got = self.cap.read()
        if got is None:
            return 'unconfirmed'
        changed, places = self.move_changes(getattr(self, '_move_before', None), got[0], src, dst)
        unexpected = [loc for loc in changed if loc not in (src, dst)]
        if unexpected or (dst in changed and src not in changed):
            occupant = self.roster.at(dst)
            # Animation in a neighbour's crop is not proof of a wrong drag.
            # A fresh identity at BOTH ends can prove the intended swap instead.
            bars = B.green_bars(got[0])
            if occupant is not None and occupant.name and occupant.name != unit.name:
                at_dst = self.read_fresh(self.unit_point(dst, bars), [self.unit_point(src, bars)])
                at_src = self.read_fresh(self.unit_point(src, bars), [self.unit_point(dst, bars)])
                if (at_dst[0] == unit.name and at_src[0] == occupant.name
                        and at_dst[1] in (None, unit.star) and at_src[1] in (None, occupant.star)):
                    self.log(ev='swap_confirmed', src=src, dst=dst, unit=unit.name, occupant=occupant.name,
                             crop_changes=changed)
                    return False
            self.need_survey = f'drag from {src} to {dst} changed other places {unexpected or [dst]}'
            self.survey_places = (self.need_survey, list(dict.fromkeys([dst, src] + unexpected + places)))
            self.fail_cause = 'the drag moved another unit: its source, target and neighbours are being read'
            self.deferred = True
            self._repair_pending = True
            self.log(ev='drag_wrong_unit', src=src, dst=dst, changed=changed, check=places)
            return True
        occupant = next((u for u in self.roster.units if u.loc == dst and u is not unit), None)
        sure = lambda u: u is None or (getattr(u, 'sure', False) and not u.unverified)
        if src in changed and dst in changed and sure(unit) and sure(occupant):
            # the pictures show the swap - both places changed, no other one did - and both units are known: no panel
            # read (the owner, 2026-11-08: "you do not need to check for every action"; game 74b 6-1's Gnar swap took
            # 3.0 s, most of it two panel reads)
            self.log(ev='swap_seen', src=src, dst=dst, unit=repr(unit), occupant=repr(occupant), changed=changed)
            return False
        bars = B.green_bars(got[0])
        # the swapped-out unit now stands on src: shown first when the panel did not change (another champion)
        others = [self.unit_point(src, bars)] if occupant is not None and occupant.name != unit.name else []
        name = self.read_fresh(self.unit_point(dst, bars), others)[0]
        if name and name != unit.name:
            self.need_survey = f'a swap onto {dst[1:]} left {name} there, not {unit.name}'
            self.survey_places = (self.need_survey, [dst, src])
            self.fail_cause = f'after the swap {name} stands at {dst[1:]}, not {unit.name}: the records are being checked'
            self.deferred, self._repair_pending = True, True
            return True
        if name:
            return False
        # no panel on the hex: the place it came from tells - the unit still there means the swap did not happen (run
        # 20261020-094910 4-6: Vi onto Sejuani's hex with 4 s left, the drop came in the fight and was refused; the
        # records swapped them anyway, and a bench move, two sells and a board move failed on them)
        back = self.read_fresh(self.unit_point(src, bars), [self.unit_point(dst, bars)])[0]
        if back and back == unit.name and (occupant is None or occupant.name != unit.name):
            where = f'bench slot {src[1]}' if src[0] == 'bench' else f'{src[1:]}'
            self.need_survey = f'{unit.name} remains at {where} after a drag to {dst}'
            self.survey_places = (self.need_survey, places)
            self.fail_cause = (f'the intended swap did not happen: {unit.name} still stands at {where}, '
                               'the target and neighbours are being checked')
            self.deferred = True
            self._repair_pending = True
            return True
        if back and occupant is not None and back == occupant.name:
            return False                            # the occupant took its place: done
        return 'unconfirmed'                        # no read either way: done, both to be read

    def stop_recording(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder = None

    def latency_report(self):
        if not self.lat:
            return ''
        a = np.array(self.lat) * 1000
        return ' | '.join(f'{k} p50 {np.percentile(a[:, i], 50):.0f} p95 {np.percentile(a[:, i], 95):.0f}ms'
                          for i, k in enumerate(('read', 'decide', 'act')))


def bench_choice_open(frame):
    from tft_player import bench
    return bool(bench.choice_cards(frame))


