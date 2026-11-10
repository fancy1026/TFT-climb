"""The execution grade of a game, from its log.

before_game (a Normal game to 3-2 before a ranked one) was the coordinator's own design, built from the owner's
2026-11-05 words asking for "1-2 complete rounds or some tests to make sure all the changes are working before an
actual game" - the owner never asked for a live game: "I never said live check before ranked, never do it" (11-06).
agent/climb.py no longer calls it; changes are proven offline. Every ranked game is graded from its log afterwards
(after_ranked): a fault found there is recorded for the batch review.

Healthy games 56-63 (the limits below): 0 inputs not sent with the game in front; every buy, roll and level-up landed;
item gives 64-92% (55-90% by 3-2), moves 80-100%, sales 83-100%; at most 0.7 refused inputs a round; plan steps left
at combat start in at most 69% of the calls; loot orbs left at combat start in at most 4 rounds; ticks at most 25 s;
both augments chosen by 3-2. Game 64: 226 inputs not sent, item gives 0 of 46, 2.2 refused inputs a round, steps left
in 16 of 17 calls."""
import json
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
STATE = AGENT / 'smoke.json'
HARNESS = ('tft_player', 'tools/play_comp.py', 'tools/play_match.py')
UNTIL = '3-2'                     # the game allows a surrender from 3-2
QUEUE = 1090                      # TFT Normal: no LP
FIXES = 2                         # Editor sessions on a failed check before the loop stops
# kind: (lowest share that landed, fewest tries to judge it)
RATES = {'buy': (0.9, 3), 'item': (0.5, 4), 'move': (0.75, 4), 'sell': (0.75, 4), 'rolldown': (0.9, 3)}


def stage_key(s):
    try:
        a, b = str(s).split('-')
        return int(a) * 10 + int(b)
    except (ValueError, AttributeError):
        return None


def events(run_dir):
    f = Path(run_dir) / 'log.jsonl'
    out = []
    if f.exists():
        for line in f.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def grade(run_dir, until=None, check=False, output=None):
    """A game's execution against healthy games, up to `until` (a stage) or whole. `check`: the live check's own
    conditions too (buys made, augments chosen, the carousel read, 3-2 reached, no crash in `output`).
    -> {'ok', 'faults', 'numbers'}"""
    lim = stage_key(until) if until else None
    acts = defaultdict(lambda: [0, 0])
    bad = defaultdict(list)                  # a few failed examples per kind, for the Editor
    not_sent, rounds, augments = [], set(), set()
    refused = calls = left = orbs = carousels = 0
    slow = 0.0
    over = False
    for e in events(run_dir):
        ev, st = e.get('ev'), e.get('stage')
        k = stage_key(st)
        if ev == 'game_result' or (ev == 'round' and str(e.get('screen', '')).startswith('ELIMINATED')):
            over = True
        if lim and k and k > lim:
            continue
        if ev == 'round' and k:
            rounds.add(st)
        elif ev == 'action':
            kind = str(e.get('act', '')).split('(')[0]
            acts[kind][1] += 1
            acts[kind][0] += bool(e.get('ok'))
            if not e.get('ok') and len(bad[kind]) < 3:
                bad[kind].append(f"{st}: {str(e.get('act'))[:140]}")
        elif ev == 'input_not_sent' and (e.get('diag') or {}).get('game_foreground'):
            not_sent.append(e)
        elif ev == 'input_failed':
            refused += 1
        elif ev == 'combat_start_call':
            calls += 1
            left += bool(e.get('left'))
            orbs += bool(e.get('orbs'))
        elif ev == 'slow_tick':
            slow = max(slow, float(e.get('seconds') or 0))
        elif ev == 'augment' and st:
            augments.add(st)
        elif ev == 'carousel_result':
            carousels += 1
    n = max(len(rounds), 1)
    faults = []
    if len(not_sent) > 2:
        first = not_sent[0]
        faults.append(f"{len(not_sent)} inputs not sent with the game in front (healthy games: 0); the first at "
                      f"{first.get('stage')}: {str(first.get('act'))[:160]}, cursor {(first.get('diag') or {}).get('cursor')}")
    for kind, (share, least) in RATES.items():
        ok, tries = acts[kind]
        if kind == 'item':
            tries += sum(1 for e in not_sent if str(e.get('act', '')).startswith('item'))
        if tries >= least and ok / tries < share:
            faults.append(f"{kind}: {ok} of {tries} landed (healthy: at least {share:.0%})"
                          + (f"; failed: {'; '.join(bad[kind])}" if bad[kind] else ''))
    if acts['level'][1] - acts['level'][0] >= 2:
        faults.append(f"level: {acts['level'][1] - acts['level'][0]} XP buys failed (healthy: none)")
    if refused / n > 1.0:
        faults.append(f"{refused} inputs refused or unconfirmed in {n} rounds (healthy: at most 0.7 a round)")
    if calls >= 6 and left / calls > 0.75:
        faults.append(f"plan steps still left at combat start in {left} of {calls} calls (healthy: at most 69%)")
    if orbs > 4:
        faults.append(f"loot orbs left at combat start in {orbs} rounds (healthy: at most 4)")
    if slow > 40:
        faults.append(f"a tick took {slow:.0f} s (healthy: at most 25 s)")
    reached = max((stage_key(r) for r in rounds), default=0)
    if check:
        if acts['buy'][0] < 3:
            faults.append(f"only {acts['buy'][0]} buys landed by {until} (healthy: 10 or more)")
        for a in ('2-1', '3-2'):
            if stage_key(a) <= reached and a not in augments:
                faults.append(f"no augment chosen at {a}")
        if reached >= stage_key('2-5') and not carousels:
            faults.append("no carousel result recorded at 2-4")
        if lim and reached < lim and not over:
            faults.append(f"the game stopped at {max(rounds, key=stage_key) if rounds else 'its start'}, before {until}")
        if crash(output):
            faults.append(crash(output))
    numbers = {'rounds': len(rounds), 'reached': max(rounds, key=stage_key) if rounds else None,
               'actions': {k: v for k, v in acts.items()}, 'not_sent': len(not_sent), 'refused': refused,
               'left': [left, calls], 'orbs': orbs, 'slowest': round(slow, 1), 'augments': sorted(augments),
               'carousels': carousels}
    return {'ok': not faults, 'faults': faults, 'numbers': numbers}


def load():
    try:
        return json.loads(STATE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {'good': None, 'checks': []}


def save(state):
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def changed(good):
    """The harness differs from the commit that last played clean (or none has)."""
    if not good:
        return True
    r = subprocess.run(['git', 'diff', '--quiet', good, 'HEAD', '--', *HARNESS], cwd=ROOT)
    return r.returncode != 0


def play_check(cfg, n):
    """A Normal game on the current code until 3-2, then a surrender, on the model game n will play.
    -> (record or None, output file)."""
    import climb
    import backends
    if cfg.get('compare_model') and n % 2:              # as climb.play: the odd games play the compared model
        cfg = {**cfg, 'model': cfg['compare_model'], 'effort': cfg.get('compare_effort', cfg['effort'])}
    model, effort = backends.choose(cfg['model'], cfg['effort'], cfg)
    out = ROOT / 'runs' / f'climb-check-{datetime.now():%Y%m%d-%H%M%S}.out'
    cmd = [climb.PY, str(ROOT / 'tools' / 'play_comp.py'), '--comp', cfg['comp'], '--model', model,
           '--effort', effort, '--until', UNTIL]
    env = {**climb.os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8', 'TFT_CLIMB': '1',
           'TFT_NO_CHAT_WINDOW': '1', 'TFT_OVERLAY_NO_CAPTURE': '1'}
    climb.say(f'live check before game {n}: a Normal game until {UNTIL} on {model} {effort} (output {out.name})')
    rec = None
    with open(out, 'w', encoding='utf-8') as f:
        p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace', env=env)
        for line in p.stdout:
            f.write(line)
            f.flush()
            if line.startswith('{') and '"result"' in line and ('"run_dir"' in line or '"stopped' in line):
                try:
                    rec = json.loads(line)
                except ValueError:
                    pass
            elif any(k in line for k in ('Traceback', 'F8', 'surrender:')):
                climb.say('live check:', line.strip()[:200])
        p.wait()
    return rec, out


def lobby_open(c, queue_id=QUEUE):
    """None when the client opens a lobby for queue_id (it is left again), else why not. The queue list and the
    eligibility check can say open while the lobby is refused (11-05 09:30, every TFT queue on OC1 but Normal turned
    off: Normal "Available" and eligible, its lobby INVALID_REQUEST_FOR_PRODUCT), and each refused check costs its
    speed check and preparation."""
    from tft_player.lobby import LobbyError
    why = c.queue_closed(queue_id)                       # the standalone client (18.6): no API, its screen and log
    if why:
        return why
    try:
        c.create_lobby(queue_id)                         # selects the mode, no search; the lobby stays open
        return None
    except LobbyError as e:
        return f"queue {queue_id}: lobby refused ({e})"


def wait_ready(poll=120):
    """Until no other game runs, the client is idle and opens a Normal lobby. -> False at agent/STOP."""
    import climb
    said = None
    while not (climb.AGENT / 'STOP').exists():
        why = climb.other_game_running()
        if not why:
            try:
                c = climb.client()
                phase = c.phase()
                why = f'client phase {phase}' if phase not in climb.IDLE else climb.queue_closed(c, QUEUE) or lobby_open(c)
                c.close()
            except Exception as e:
                why = f'client: {e}'
        if not why:
            if said:
                climb.say('live check: the Normal queue opens a lobby')
            return True
        if why != said:
            climb.say('live check waits:', why)
            said = why
        time.sleep(poll)
    return False


def refused(rec, out):
    """The game never started on Riot's side (no match found, or the client refused the lobby): not the harness."""
    return 'could not enter a match' in str((rec or {}).get('result')) or 'LobbyError:' in (crash(out) or '')


def crash(output):
    """'the player crashed: <its last frame> <the error>' from a game's output, or None."""
    text = Path(output).read_text(encoding='utf-8', errors='replace') if output and Path(output).exists() else ''
    head = 'Traceback (most recent call last)'
    if head not in text:
        return None
    lines = text[text.index(head):].splitlines()[1:]
    k = next((i for i, l in enumerate(lines) if l and not l[0].isspace()), len(lines))
    frames = [l.strip() for l in lines[:k] if l.strip().startswith('File ')]
    return f"the player crashed: {frames[-1] if frames else ''} {lines[k].strip() if k < len(lines) else ''}"[:400]


def request_text(rec, g, n):
    run = (f"run runs/{Path(rec['run_dir']).name}" if rec.get('run_dir')
           else 'no run folder (the player ended before its game)')
    return (f"[live check before game {n}, P0] The live check - a Normal game played on the current code until "
            f"{UNTIL}, {run}, its output runs/{Path(rec.get('_out', '')).name} - failed, so no ranked game "
            f"starts until it passes: " + ' / '.join(g['faults']) + f". Numbers: {json.dumps(g['numbers'])}. Healthy "
            f"games 56-63 by 3-2: 0 inputs not sent, every buy landed, item gives 55-90%, moves at least 77%, sales at "
            f"least 86%, both augments chosen. Find the cause in this run's log and frames (and in the commits since "
            f"the code that last played clean), fix it with a regression test from this run, and keep every fix the "
            f"owner asked for.")


def before_game(cfg, n):
    """True once the harness that will play has played clean, live or in a ranked game; False: stop the loop. It runs
    as soon as the Normal queue opens a lobby, while Ranked may still be closed, and waits while Riot's side refuses
    the game; a crash before the game is a failed check like any other."""
    import climb
    state = load()
    if not changed(state.get('good')):
        return True
    failed = 0
    while True:
        if not climb.wait_codex() or not wait_ready():  # never a check on the fallback model
            return False
        head = climb.git('rev-parse', '--short', 'HEAD')
        rec, out = play_check(cfg, n)
        if not (rec or {}).get('run_dir') and refused(rec, out):
            climb.say(f"live check: no game on Riot's side ({(rec or {}).get('result') or 'the lobby was refused'}): "
                      f"again in 2 min")
            time.sleep(120)
            continue
        if str((rec or {}).get('result', '')).startswith('stopped'):   # F8, or no game as climb.main stops on
            climb.say(f"live check: {rec.get('result')}: stopping")
            return False
        rec = dict(rec or {}, _out=str(out))
        run = rec.get('run_dir')
        g = grade(run, UNTIL, check=True, output=out) if run else \
            {'ok': False, 'faults': [crash(out) or f"the player ended without a game ({rec.get('result') or 'no record'})"],
             'numbers': {}}
        row = {'when': datetime.now().strftime('%Y-%m-%d %H:%M'), 'code': head, 'run': Path(run).name if run else None,
               'result': rec.get('result'), **g}
        if run:
            (Path(run) / 'live-check.json').write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding='utf-8')
        state['checks'] = (state.get('checks') or [])[-40:] + [row]
        if g['ok']:
            state['good'] = head
            save(state)
            climb.commit_paths(f'Live check passed on {head} before game {n}', STATE.relative_to(ROOT))
            climb.say(f"live check: clean on {head} ({g['numbers']['reached']} reached): the ranked game starts")
            return True
        save(state)
        climb.say(f"live check: failed on {head}: {' / '.join(g['faults'])[:600]}")
        failed += 1
        if failed > FIXES:
            break
        made = climb.add_requests('editor', [request_text(rec, g, n)], n, f'Live check before game {n}')
        climb.commit_paths(f'Live check failed on {head} before game {n}', STATE.relative_to(ROOT),
                           climb.REQUESTS.relative_to(ROOT))
        climb.say(f"live check: the Editor on {made[0]['id']} (attempt {failed} of {FIXES})")
        climb.editor_on_requests(made, cfg)
    climb.commit_paths(f'Live check still failing before game {n}', STATE.relative_to(ROOT))
    climb.say(f'live check: still failing after {FIXES} Editor sessions: stopping, no ranked game on this code')
    return False


def after_ranked(row, rec):
    """A ranked game graded like the check: clean, its code is good; a fault is recorded and goes to the Editor at the
    batch review."""
    import climb
    if not rec or not rec.get('run_dir'):
        return None
    g = grade(rec['run_dir'])
    state = load()
    code = row.get('code')
    state['ranked'] = (state.get('ranked') or [])[-40:] + [{'game': row.get('n'), 'run': row.get('run'), 'code': code,
                                                            **g}]
    if g['ok'] and code:
        if state.get('good') != code and not changed(code):
            state['good'] = code
    else:
        # the owner, 2026-11-06: inside a batch only the owner's feedback changes the code - the fault waits for the
        # batch review, and the same code plays on (no live check without a change)
        text = (f"[game {row.get('n')}] The game's execution, graded against healthy games, failed: "
                + ' / '.join(g['faults']) + f". Numbers: {json.dumps(g['numbers'])}. Run runs/{row.get('run')}. "
                f"Find the cause in its log and frames and fix it with a regression test from this run.")
        made = climb.add_requests('editor', [text], row.get('n'), f"Execution grade of game {row.get('n')}")
        climb.say(f"execution grade: game {row.get('n')} failed ({' / '.join(g['faults'])[:300]}): {made[0]['id']}")
    save(state)
    climb.commit_paths(f"Execution grade of game {row.get('n')}: {'clean' if g['ok'] else 'faults'}",
                       STATE.relative_to(ROOT), climb.REQUESTS.relative_to(ROOT))
    return g
