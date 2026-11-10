"""Standalone TFT lobby: log state and foreground UI input; no local API.

LeagueClient remains an alias for callers predating patch 18.6.
"""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import re
import threading
import time

LOG_DIR = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'TFT/Saved/Logs'
PUUID_IN_LOG = re.compile(r'puuid=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})&accountId=')
_OUR_PUUID = {}


def our_puuid(log_path=None):
    """The signed-in account's PUUID, from the client's own requests in its log ("puuid=<id>&accountId="), so any
    account plays with no setting (it was one account's id, written here). TFT_PUUID overrides. None while the
    log has none; a found id is kept for the log it came from."""
    if os.environ.get('TFT_PUUID'):
        return os.environ['TFT_PUUID']
    path = Path(log_path or LOG_DIR / 'TFT.log')
    try:
        key = (str(path), path.stat().st_ctime_ns)
    except OSError:
        return None
    if key not in _OUR_PUUID:
        try:
            ids = PUUID_IN_LOG.findall(path.read_text(encoding='utf-8', errors='replace'))
        except OSError:
            return None
        if not ids:
            return None
        _OUR_PUUID[key] = max(set(ids), key=ids.count)
    return _OUR_PUUID[key]
TFT_QUEUES = {1090: 'TFT Normal', 1100: 'TFT Ranked'}
QUEUED = ('Matchmaking', 'ReadyCheck', 'GameStart', 'InProgress')
PLAYING = ('GameStart', 'InProgress', 'Reconnect')
POINTS = {'play': (.89, .915), 1090: (.415, .35), 1100: (.583, .35),
          'start': (.885, .925), 'cancel': (.773, .915),
          'accept': (.5, .636), 'back': (.028, .026), 'ok': (.5, .53)}


class LobbyError(RuntimeError):
    pass


class LobbyLog:
    """Incremental current-session reader, safe for polling from EndWatcher.

    Rolls/truncation reset state. Backups are test evidence, never a fallback
    for a missing current log. Partial lines wait for their newline.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.identity = None
        self.reset()

    def reset(self):
        self.offset = 0
        self.pending = b''
        self.phase = None
        self.queue = None
        self.error = None
        self.seq = 0
        self.events = deque(maxlen=128)

    def feed(self, line):
        self.seq += 1
        # Do not retain RMS payloads/tokens in diagnostics.
        if len(line) < 700:
            self.events.append((self.seq, line))
        m = re.search(r'Entering LobbyState\[ETFTLobbyState::(\w+)\]', line)
        if m:
            self.phase = {'Frontend': 'None', 'SoloMode': 'Lobby',
                          'Matchmaking': 'Matchmaking', 'InGame': 'InProgress',
                          'EndOfGame': 'EndOfGame'}.get(m[1], self.phase)
        m = re.search(r'Entering state \[ETFTFrontendState::(\w+)\]', line)
        if m and m[1] == 'Home':
            self.phase, self.queue = 'None', None
        if 'Successfully set GameMode[' in line:
            m = re.search(r'QueueId\[(\d+)\]', line)
            if m:
                self.queue = int(m[1])
        if 'Player started matchmaking' in line or 'Matchmaking Update[MATCHMAKING]' in line:
            self.phase, self.error = 'Matchmaking', None
        if 'Matchmaking Update[AFK_CHECK]' in line or 'Match found, showing AFK Check Modal' in line:
            self.phase = 'ReadyCheck'
        if 'Removed from matchmaking Reason[' in line:
            self.phase = 'Lobby'
        m = re.search(r'Matchmaking Error\[([^]]+)\]', line)
        if m:
            self.error, self.phase = m[1], 'Lobby'
        if 'Joining game server at Host[' in line:
            self.phase = 'GameStart'
        if ('GAMEFLOW_EVENT.GAME_COMPLETED' in line
                or 'updated phase from Gameplay to Idle' in line):
            self.phase = 'EndOfGame'
        if 'Entering state [ETFTFrontendState::EndOfGame]' in line:
            self.phase = 'EndOfGame'

    def read(self):
        with self.lock:
            try:
                with self.path.open('rb') as f:
                    stat = os.fstat(f.fileno())
                    identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
                    if identity != self.identity or stat.st_size < self.offset:
                        self.reset()
                        self.identity = identity
                    f.seek(self.offset)
                    data = self.pending + f.read()
                    self.offset = f.tell()
                lines = data.split(b'\n')
                self.pending = lines.pop()
                for line in lines:
                    self.feed(line.decode('utf-8', errors='replace'))
            except FileNotFoundError:
                self.reset()
                self.identity = None
            return self

    def confirmed(self, after, predicate):
        with self.lock:
            self.read()
            return any(n > after and predicate(line) for n, line in self.events)


def screen_state(hits, width, height):
    """Objective UI facts from OCR centres in client-area pixels."""
    words = [(t.upper().strip(), (x / width, y / height)) for t, (x, y) in hits]

    def has(text, box):
        x0, y0, x1, y1 = box
        return any(text in t and x0 < x < x1 and y0 < y < y1 for t, (x, y) in words)

    cards = {q for q, text, box in (
        (1090, 'NORMAL', (.33, .42, .49, .49)),
        (1100, 'RANKED', (.5, .42, .67, .49))) if has(text, box)}
    mode = next((q for q, text in ((1090, 'NORMAL'), (1100, 'RANKED'))
                 if has(text, (0, 0, .2, .08))), None)
    rank = None
    for t, (x, y) in words:
        if x < .25 and y < .09:
            m = re.search(r'\b(IRON|BRONZE|SILVER|GOLD|PLATINUM|EMERALD|DIAMOND|MASTER|GRANDMASTER|CHALLENGER)(?:\s+(IV|III|II|I))?\s*$', t)
            if m:
                rank = {'tier': m[1], 'division': m[2], 'lp': None}
    return {'cards': cards, 'mode': mode, 'rank': rank,
            'home': has('PLAY', (.8, .87, 1, 1)),
            'start': has('START', (.8, .87, 1, 1)),
            'accept': has('ACCEPT', (.4, .58, .6, .69)),
            'declined': has('DECLINED', (.36, .43, .64, .5)),
            'hits': words}


class _Desktop:
    """Lazy native adapter; fake adapters exercise the complete flow offline."""
    def window(self):
        """The client's window; a minimized one is restored first (11-06: the owner minimized it while the loop
        waited, and find_client skips minimized windows, so the live check read 'TFT client is not running')."""
        from tft_player.game import find_client
        c = find_client()
        if c is None:
            restore_minimized()
            c = find_client()
        return c

    def screen(self):
        import numpy as np
        from PIL import ImageGrab
        from tft_player.game import bring_to_front, win32gui
        from tft_player.screen import ScreenReader
        c = self.window()
        if c is None:
            raise LobbyError('TFT client is not running')
        if win32gui.GetForegroundWindow() != c.hwnd and not bring_to_front(c.hwnd):
            raise LobbyError('TFT client is not visible in front; screen unreadable')
        frame = np.asarray(ImageGrab.grab(bbox=(c.left, c.top, c.left+c.width, c.top+c.height)))
        self.frame = frame
        if not hasattr(self, 'reader'):
            self.reader = ScreenReader()
        return screen_state(self.reader.find_text(frame, width=c.width), c.width, c.height)

    def click(self, point):
        from tft_player.game import Input, bring_to_front, win32gui
        c = self.window()
        if c is None:
            raise LobbyError('TFT client is not running')
        if not bring_to_front(c.hwnd) or win32gui.GetForegroundWindow() != c.hwnd:
            raise LobbyError('TFT client could not be brought to the foreground; no click sent')
        # Input.click supplies SetCursorPos + relative 1 px nudge, F8, foreground
        # and covered-target guards. Lobby fractions scale independently on x/y.
        point = c.from_screen((c.left + point[0]*c.width, c.top + point[1]*c.height))
        Input(c).click(point)


def restore_minimized(wait=1.5):
    """Restore the TFT client's window when it is minimized. -> whether one was restored."""
    import win32con
    import win32gui
    import win32process
    import psutil
    from tft_player.game import TFT_EXES
    found = []

    def visit(h, _):
        try:
            if win32gui.IsIconic(h) and psutil.Process(win32process.GetWindowThreadProcessId(h)[1]).name().lower() in TFT_EXES:
                found.append(h)
        except Exception:
            pass
    win32gui.EnumWindows(visit, None)
    for h in found:
        win32gui.ShowWindow(h, win32con.SW_RESTORE)
    if found:
        time.sleep(wait)
    return bool(found)


class StandaloneClient:
    def __init__(self, log_path=None, stats_path=None, ui=None, now=time.monotonic, sleep=time.sleep):
        self.log = LobbyLog(log_path or LOG_DIR / 'TFT.log')
        self.stats_path = Path(stats_path or LOG_DIR / 'TFTEoGStats.json')
        self.ui = ui if ui is not None else _Desktop()
        self.now, self.sleep = now, sleep
        self._baseline_ms = time.time()*1000
        self._baseline_id = self._stats().get('gameId')

    def phase(self):
        state = self.log.read().phase
        if not self.ui.window() and state != 'EndOfGame':
            return None
        return state

    def queue_id(self):
        return self.log.read().queue

    def rank(self):
        if self.phase() != 'None':
            return None
        try:
            return self.ui.screen()['rank']
        except LobbyError:
            return None

    def queue_closed(self, queue_id):
        """Read-only preflight. Hidden mode cards cannot be checked from home."""
        if not self.ui.window():
            return 'TFT client is not running'
        if queue_id not in TFT_QUEUES:
            return f'queue {queue_id}: no supported mode card'
        phase = self.phase()
        if phase is None:
            return 'TFT client is not signed in or its lobby state is not available'
        if phase not in ('None', 'Lobby'):
            return f'TFT client phase {phase}'
        try:
            s = self.ui.screen()
        except LobbyError as e:
            return str(e)
        if s['cards'] and queue_id not in s['cards']:
            return f'queue {queue_id}: mode card missing'
        if s['home'] or queue_id in s['cards'] or s['start']:
            return None
        return f'queue {queue_id}: mode card or lobby controls missing (sign-in or modal may be open)'

    def _click_wait(self, point, predicate, seconds=2.0):
        after = self.log.read().seq
        self.ui.click(point)
        end = self.now() + seconds
        while self.now() < end:
            if self.log.confirmed(after, predicate):
                return True
            self.sleep(.1)
        # No log acknowledgement: re-observe, never mutate state from a click.
        self._record_screen('click_unconfirmed')
        return False

    def _record_screen(self, reason):
        s = self.ui.screen()
        directory = getattr(self, 'record_dir', None)
        frame = getattr(self.ui, 'frame', None)
        if directory is not None and frame is not None:
            from PIL import Image
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            name = f'{time.time_ns()}-{reason}'
            Image.fromarray(frame).save(directory / f'{name}.png')
            facts = {**s, 'cards': sorted(s['cards']), 'phase': self.phase(), 'queue_id': self.queue_id()}
            (directory / f'{name}.json').write_text(json.dumps(facts), encoding='utf-8')
        return s

    def create_lobby(self, queue_id):
        if queue_id not in TFT_QUEUES:
            raise LobbyError(f'queue {queue_id}: no supported mode card')
        for _ in range(6):
            phase = self.phase()
            if phase in PLAYING:
                raise LobbyError('Cannot change mode during a game')
            if phase in ('Matchmaking', 'ReadyCheck'):
                if not self.cancel_search():
                    raise LobbyError('Search cancellation not confirmed')
                continue
            s = self.ui.screen()
            if s['declined']:
                self._dismiss_error()
            elif s['start'] and s['mode'] == queue_id and self.queue_id() == queue_id and phase == 'Lobby':
                return True
            elif s['cards']:
                if queue_id not in s['cards']:
                    raise LobbyError(f'queue {queue_id}: mode card missing')
                self._click_wait(POINTS[queue_id], lambda l: 'Successfully set GameMode[' in l
                                 and f'QueueId[{queue_id}]' in l)
            elif s['start']:
                self._click_wait(POINTS['back'], lambda l: 'Entering state [ETFTFrontendState::Home]' in l)
            elif s['home']:
                # The card overlay has no dedicated marker in the recording.
                # Re-read it after navigation; only selection + SoloMode confirms
                # create_lobby, never the PLAY click itself.
                self._click_wait(POINTS['play'], lambda l: 'Successfully set GameMode[' in l)
            else:
                raise LobbyError('No TFT home, mode cards or lobby controls visible')
        raise LobbyError(f'queue {queue_id}: lobby selection not confirmed by log and screen')

    def _dismiss_error(self):
        return self._click_wait(POINTS['ok'], lambda l: 'Applying input config for leaf-most node [WB_Lobby' in l)

    def start_search(self):
        s = self.ui.screen()
        if s['declined']:
            self._dismiss_error()
            s = self.ui.screen()
            if s['declined']:
                raise LobbyError('Ready-check decline dialog dismissal not confirmed')
        if not s['start'] or s['mode'] != self.queue_id():
            raise LobbyError('START and the confirmed queue are not visible')
        if not self._click_wait(POINTS['start'], lambda l: 'Player started matchmaking' in l):
            raise LobbyError('START not confirmed by log')
        return True

    def cancel_search(self):
        if self.phase() not in ('Matchmaking', 'ReadyCheck'):
            return True
        # The AFK modal's X declines a match; do not send cancel through it.
        if self.phase() == 'ReadyCheck':
            return False
        return self._click_wait(POINTS['cancel'], lambda l: 'Removed from matchmaking Reason[' in l
                                or 'Entering LobbyState[ETFTLobbyState::SoloMode]' in l)

    def accept_ready_check(self):
        if self.phase() != 'ReadyCheck':
            return False
        if not self.ui.screen()['accept']:
            return False
        return self._click_wait(POINTS['accept'], lambda l: '"afkReady":true' in l
                                or 'Joining game server at Host[' in l
                                or 'Entering LobbyState[ETFTLobbyState::InGame]' in l)

    def dismiss_end_of_game(self):
        if self.phase() in ('None', 'Lobby'):
            return True
        if self.phase() != 'EndOfGame':
            return False
        s = self.ui.screen()
        # No standalone EOG screenshot yet: locate a labelled control, never
        # guess the old League stats-screen coordinate.
        for text, point in s['hits']:
            if text in ('CONTINUE', 'PLAY AGAIN', 'BACK TO LOBBY', 'EXIT NOW', 'EXIT GAME'):
                return self._click_wait(point, lambda l: 'Entering LobbyState[ETFTLobbyState::Frontend]' in l
                                        or 'Entering LobbyState[ETFTLobbyState::SoloMode]' in l
                                        or 'Entering state [ETFTFrontendState::Home]' in l)
        self._record_screen('end_controls_missing')
        return False

    def _stats(self):
        try:
            data = json.loads(self.stats_path.read_text(encoding='utf-8-sig'))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def begin_game(self, since_ms):
        """Snapshot the previous result before queueing each game in a batch."""
        self._baseline_ms = since_ms
        self._baseline_id = self._stats().get('gameId')

    def fetch_placement(self, since_ms):
        try:
            with self.stats_path.open(encoding='utf-8-sig') as f:
                data = json.load(f)
                written_ms = os.fstat(f.fileno()).st_mtime_ns / 1_000_000
            game_id = data.get('gameId')
            if not game_id or written_ms < since_ms:
                return None
            if since_ms >= self._baseline_ms and game_id == self._baseline_id:
                return None
            players = (data.get('statsBlock') or {}).get('players') or []
            ours = our_puuid(self.log.path)
            me = next((p for p in players if ours and p.get('PUUID') == ours), None)
            if not me or type(me.get('ffaStanding')) is not int or not 1 <= me['ffaStanding'] <= 8:
                return None
            units = lambda p: [(u.get('championName'), u.get('starLevel'), u.get('itemNames'))
                               for u in p.get('boardPieces', [])]
            return {'placement': me['ffaStanding'], 'placement_source': 'TFTEoGStats.json',
                    'game_id': game_id, 'queue_id': data.get('queueId'), 'is_ranked': data.get('isRanked'),
                    'units': units(me), 'lobby': [{'place': p.get('ffaStanding'), 'us': p is me,
                                                 'units': units(p)} for p in players]}
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def close(self):
        """No connection or capture to close; never closes the TFT process."""

    def enter_match(self, **kwargs):
        return enter_match(self, **kwargs)


LeagueClient = StandaloneClient


def enter_match(client, emit=print, timeout=1800, poll=1.0, require_tft=True,
                now=time.monotonic, queue_id=None, sleep=time.sleep):
    """Queue the requested mode, accept AFK checks, return the existing contract."""
    queue = queue_id if queue_id is not None else client.queue_id()
    if queue not in TFT_QUEUES:
        raise LobbyError(f'queue {queue}: no supported TFT mode card')
    deadline, seen, accepted, searches = now()+timeout, None, 0, 0
    beat = now()
    while now() < deadline:
        try:
            phase = client.phase()
            if phase != seen:
                emit({'type': 'gameflow_phase', 'phase': phase, 'accepted': accepted})
                seen = phase
            elif now() - beat > 30:
                emit({'type': 'still_waiting', 'phase': phase, 'accepted': accepted})
                beat = now()
            if phase == 'InProgress':
                return {'entered': True, 'phase': phase, 'accepted': accepted, 'searches': searches}
            if phase == 'ReadyCheck':
                try:
                    if client.queue_id() != queue:
                        raise LobbyError('Ready check is for a different or unknown queue')
                    if client.accept_ready_check():
                        accepted += 1
                except Exception as e:
                    emit({'type': 'ready_check_retry', 'error': repr(e)})
            elif phase == 'EndOfGame':
                if client.dismiss_end_of_game():
                    emit({'type': 'end_of_game_dismissed', 'phase': client.phase()})
            elif phase in ('None', 'Lobby'):
                client.create_lobby(queue)
                emit({'type': 'lobby_ready', 'queue_id': queue, 'queue': TFT_QUEUES[queue], 'can_start': True})
                client.start_search()
                searches += 1
                emit({'type': 'matchmaking_started', 'searches': searches})
            elif phase == 'Matchmaking' and client.queue_id() != queue:
                if not client.cancel_search():
                    raise LobbyError('Wrong-queue search cancellation not confirmed')
            elif phase is None:
                raise LobbyError('TFT client not running, not signed in, or log state unavailable')
        except Exception as e:
            emit({'type': 'client_error', 'error': repr(e)})
        sleep(poll)
    try:
        if client.cancel_search():
            emit({'type': 'search_cancelled', 'why': 'no match in time'})
        else:
            emit({'type': 'cancel_failed', 'error': 'cancellation not confirmed; ready check may still be active'})
    except Exception as e:
        emit({'type': 'cancel_failed', 'error': repr(e)})
    return {'entered': False, 'phase': seen, 'accepted': accepted, 'searches': searches, 'reason': 'timeout'}
