"""The lead (the owner's structure v3): plays the agent's games and runs every step of the game loop and the
strategy loop in turn; it never edits knowledge itself. Code triggers every step.

    python agent/climb.py [--games N] [--until HH:MM] [--no-coach]
    python agent/climb.py --after runs/<id>     record a game played by hand, run the loop's steps on it, then stop
    python agent/climb.py --roles runs/<id>     the steps after a game, on a recorded game

Each cycle:
 1. wait until no other game runs on this PC and the League client is idle; the rank before (client API);
 2. one game: tools/play_comp.py with agent/config.json's comp, model and effort, TFT_CLIMB=1, on the knowledge as
    committed (its memory hash and commit are recorded with the game);
 3. the rank after; the game's record and process numbers -> agent/games.jsonl;
 4. metrics (agent/metrics.py -> agent/metrics/<run>.json) and the packet (agent/digest.py);
 5. the Reviewer (agent/roles/reviewer.md), per stage: the stage report - every decision of every round graded
    against the best one the state allowed, never by the fight result, HP or the place, with the whole game in
    numbers and the luck measured (agent/stage_review.py); code scores each part out of 10 - then the diff report of
    each strategy stage against the comp's earlier games and the strategy version each played; the summary (what
    decided the place, harness faults, information gaps); the report page for the owner; the change review;
 6. the Editor (agent/roles/harness_fixer.md), given the review's lists and the open requests: code fixes with
    regression tests and a watch; the gate (agent/check.py) takes back a code change that fails it;
 7. the Strategist (agent/roles/strategist.md), one session per stage the diff reports mark revise or revert: a new
    stage strategy with its watch, a revert, or keep; the Judge (agent/judge.py stage_gate) asks the stage's
    recorded strong players' decisions with the current and the new text, three times each, and the new one ships
    only if it scores no lower (agent/stages.py keeps every version);
 8. the Coordinator's report: three lines in the channel (agent/channel.jsonl);
 10. live evaluation (agent/live_eval.py) and the dashboard (agent/dashboard.py).
After every role a guard takes back any change outside the files that role may write. The next game waits until
every step has finished.
Stops after --games, at --until (no game starts after it), when a game was stopped (F8), or when agent/STOP exists.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
GAMES = AGENT / 'games.jsonl'
REVIEWS = AGENT / 'reviews'
PY = sys.executable
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
COACH_TIMEOUT = 60 * 60
IDLE = ('Lobby', 'None', None, 'EndOfGame', 'PreEndOfGame', 'WaitingForStats')
STAGES = ('2-1', '2-5', '3-1', '3-2', '3-5', '4-1', '4-2', '4-5', '5-1', '5-5', '6-1')


def say(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def config():
    return json.loads((AGENT / 'config.json').read_text(encoding='utf-8'))


def records():
    if not GAMES.exists():
        return []
    return [json.loads(l) for l in GAMES.read_text(encoding='utf-8').splitlines() if l.strip()]


def append(rec):
    with open(GAMES, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def update(run, **fields):
    recs = records()
    for r in recs:
        if r.get('run') == run:
            r.update(fields)
    GAMES.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in recs), encoding='utf-8')


def git(*a):
    r = subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return r.stdout.strip()


def other_game_running():
    """Another game process on this PC (a benchmark, a test game): only one game client."""
    import psutil
    me = os.getpid()
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(p.info['cmdline'] or [])
        except Exception:
            continue
        if p.info['pid'] != me and any(s in cmd for s in ('play_comp.py', 'play_match.py', 'benchmark.py')):
            return cmd[:160]
    return None


def client():
    from tft_player.lobby import LeagueClient
    return LeagueClient()


def rank():
    """{'tier', 'division', 'lp', 'wins', 'losses'} of TFT Ranked, or None."""
    try:                                                 # the standalone client (18.6) shows tier and division, no LP
        c = client()
        try:
            return c.rank()
        finally:
            c.close()
    except Exception as e:
        say('rank not read:', e)
        return None


TIERS = ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND', 'MASTER', 'GRANDMASTER', 'CHALLENGER']
DIVS = {'IV': 0, 'III': 1, 'II': 2, 'I': 3}


def ladder(r):
    """A rank as one number of LP from Iron IV 0 (100 LP a division), for the curve."""
    if not r or r.get('tier') not in TIERS or r.get('lp') is None:   # the standalone client shows no LP: unknown,
        return None                                                  # never a measured 0
    t = TIERS.index(r['tier'])
    if t >= 7:
        return 2800 + (r.get('lp') or 0)
    return t * 400 + DIVS.get(r.get('division'), 0) * 100 + (r.get('lp') or 0)


def queue_closed(c, queue_id):
    """Why the client will not open a lobby for queue_id, or None. 2026-11-04 after Riot's OC1 login incident: Ranked TFT
    was "QueueDisabled" for an hour, the lobby POST answered INVALID_REQUEST, and each try crashed the game process
    after its speed check and preparation (model calls) and stopped the loop."""
    if c.phase() == 'EndOfGame':                         # a check's surrender leaves the results screen up (11-06)
        c.dismiss_end_of_game()
    return c.queue_closed(queue_id)                      # the standalone client (18.6): its screen and log


def wait_codex(poll=600):
    """With "wait_for_codex" in agent/config.json, nothing runs while Codex's usage is used up: no game and no role on
    the fallback (the owner, 2026-11-04: wait for the weekly reset, then use Codex fully; the owner applies a manual
    reset when it runs out). Polls the probe (agent/backends.py, every PROBE_EVERY s). -> False at agent/STOP."""
    import backends
    cfg = config()
    if not cfg.get('wait_for_codex') or not str(cfg.get('model', '')).startswith('codex:'):
        return True
    said = False
    while backends.codex_limited():
        if (AGENT / 'STOP').exists():
            return False
        if not said:
            say('waiting: Codex usage is used up - no game or role on the fallback until Codex answers '
                '(agent/config.json wait_for_codex)')
            said = True
        time.sleep(poll)
    if said:
        say('Codex answers again: the framework goes on')
    return True


def wait_idle(poll=20):
    said = None
    while True:
        busy = other_game_running()
        phase = closed = None
        if not busy:
            try:
                c = client()
                phase = c.phase()
                if phase in IDLE:
                    closed = queue_closed(c, 1100 if config().get('queue', 'ranked') == 'ranked' else 1090)
                c.close()
            except Exception as e:
                phase = f'client: {e}'
        if not busy and phase in IDLE and not closed:
            return
        why = busy or closed or f'client phase {phase}'
        if why != said:
            say('waiting:', why)
            said = why
        time.sleep(poll)


def mouse_ready(deadline):
    """The owner (2026-11-08, game 77 sat on its results screen with every click refused): when Windows refuses our
    mouse input, Mouse Without Borders is closed automatically and the input tried again. Checked before each game
    while the TFT client is open (the refusal comes with it: with the client closed the mouse was taken). Still refused:
    tried again every 3 minutes, closing it again if it came back. -> True when the mouse is taken; False at the
    deadline or agent/STOP."""
    import psutil
    from tft_player.setup import close_mwb, mouse_share

    def client_open():
        return any((p.info['name'] or '').lower() in ('tftclient-win64-shipping.exe', 'tftclient.exe')
                   for p in psutil.process_iter(['name']))

    # the mouse counts as taken when nine in ten moves over 3 s are (a single refusal also comes with it working); on
    # 11-08 it came back ~6 min after the client was started again with Mouse Without Borders closed
    said = False
    while client_open() and mouse_share() < 0.9:
        left = close_mwb(wait=90)
        time.sleep(2)
        if mouse_share() >= 0.9:
            say('framework: Windows refused our mouse input - Mouse Without Borders closed, the mouse is taken again')
            return True
        if not said:
            say('framework: Windows refuses our mouse input with the TFT client open, with Mouse Without Borders '
                + (f'still running ({", ".join(left)})' if left else 'closed') + ': trying again every 3 minutes')
            said = True
        end = time.time() + 180
        while time.time() < end:
            if (AGENT / 'STOP').exists() or (deadline and datetime.now() >= deadline):
                return False
            time.sleep(5)
    if said:
        say('framework: the mouse is taken again')
    return True


def play(cfg, n):
    """One game; returns the play_match record (dict) or None. With "compare_model" in agent/config.json the odd
    games play it instead of "model" (the owner, 2026-10-30: the player on GPT-6.1 Terra, compared against Sol)."""
    if cfg.get('compare_model') and n % 2:
        cfg = {**cfg, 'model': cfg['compare_model'], 'effort': cfg.get('compare_effort', cfg['effort'])}
    import backends                                      # Codex used up: Opus 5.5 medium (the owner, 2026-11-03)
    model, effort = backends.choose(cfg['model'], cfg['effort'], cfg)
    if model != cfg['model']:
        say(f"game {n}: Codex's usage is used up ({backends.MARK.name}): the player plays on {model} {effort}")
        cfg = {**cfg, 'model': model, 'effort': effort}
    import expert                                        # the owner's feedback per comp, as the player reads it (x19)
    expert.render_feedback()
    out = ROOT / 'runs' / f'climb-game-{n:02d}.out'
    out.parent.mkdir(exist_ok=True)
    cmd = [PY, str(ROOT / 'tools' / 'play_comp.py'), '--comp', cfg['comp'], '--model', cfg['model'],
           '--effort', cfg['effort']] + (['--ranked'] if cfg.get('queue', 'ranked') == 'ranked' else [])         + (['--video-hq'] if cfg.get('video') == 'hq' else ['--video'] if cfg.get('video') else [])
    # (the owner, 2026-11-09: "record the game from now on, high quality" - agent/config.json "video": "hq":
    # runs/<run>/game.mp4 at 30 fps, native size, quality 20)
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8', 'TFT_CLIMB': '1',
           'TFT_NO_CHAT_WINDOW': '1', 'TFT_OVERLAY_NO_CAPTURE': '1'}
    say(f'game {n}: {cfg["comp"]}, {cfg["model"]} {cfg["effort"]} (output {out.name})')
    rec = None
    with open(out, 'w', encoding='utf-8') as f:
        p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace', env=env)
        for line in p.stdout:
            f.write(line)
            f.flush()
            if line.startswith('{') and '"result"' in line and ('"run_dir"' in line or '"stopped' in line):
                try:
                    rec = json.loads(line)          # a game's record, or why no game was played
                except ValueError:
                    pass
            elif any(k in line for k in ('TOO SLOW', 'Traceback', 'F8', 'stopped')):
                say('game:', line.strip()[:200])
        p.wait()
    if rec and rec.get('run_dir') and str(cfg['model']).startswith('codex:'):
        hits = backends.limit_in_run(rec['run_dir'])
        if hits:
            backends.mark(hits[0], f"game {n}")
            say(f'game {n}: Codex hit its usage limit in the game ({hits[0][:120]}): later parts use the fallback')
    if rec is not None:
        rec.update(model=cfg['model'], effort=cfg['effort'])     # what played, for the record (fallback, compare)
    return rec


def end_screen_place(run_dir):
    log = Path(run_dir) / 'log.jsonl'
    if not log.exists():
        return None
    for line in reversed(log.read_text(encoding='utf-8', errors='replace').splitlines()):
        if '"eliminated_stop"' in line or '"end_screen"' in line:
            m = re.search(r'([1-8])\s*(?:ST|ND|RD|TH)', json.loads(line).get('text') or '')
            if m:
                return int(m.group(1))
    return None


def process_numbers(run_dir):
    """The numbers a game is judged by besides its place (tools/game_metrics.py)."""
    try:
        from game_metrics import metrics
        m = metrics(str(run_dir))
    except Exception as e:
        return {'metrics_error': str(e)}
    R = {r['stage']: r for r in m.get('rounds', [])}
    pick = lambda key: {s: R[s].get(key) for s in STAGES if s in R}
    items_on_board = {}
    for e in _events(run_dir):
        if e.get('ev') == 'fight' and e.get('stage') in STAGES:
            board = (e.get('roster') or '').split(' bench=[')[0]
            items_on_board[e['stage']] = sum(len(x.split(',')) for x in re.findall(r'@h\d\d\[([^\]]*)\]', board))
    keys = ('placement', 'last_stage', 'minutes', 'comp_final', 'comp_best', 'carry', 'carry_stars_final',
            'level_final', 'late', 'failed_steps', 'rejected', 'sells_failed', 'cost_usd', 'git')
    out = {k: m.get(k) for k in keys if k in m}
    out.update(hp=pick('hp'), level=pick('level'), gold=pick('gold'), items_on_board=items_on_board)
    return out


def _events(run_dir):
    f = Path(run_dir) / 'log.jsonl'
    out = []
    if f.exists():
        for line in f.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def game_code(run_dir):
    """The code version the game ran (its run_config: git hash, '+dirty' with uncommitted files)."""
    cfg = next((e for e in _events(run_dir) if e.get('ev') == 'run_config'), {})
    return (cfg.get('git') or git('rev-parse', '--short', 'HEAD')) + ('+dirty' if cfg.get('dirty_files') else '')


def memory_hash():
    import hashlib
    h = hashlib.sha1()
    for f in sorted((AGENT / 'memory').rglob('*.md')):
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


def record_game(rec, cfg, n, before):
    run_dir = Path(rec['run_dir'])
    run = run_dir.name
    place = rec.get('placement') or rec.get('placement_screen') or end_screen_place(run_dir)
    after = None
    for _ in range(12):                      # LP shows up a little after the game
        after = rank()
        if after and before and (after.get('wins'), after.get('losses'), after.get('lp')) != \
                (before.get('wins'), before.get('losses'), before.get('lp')):
            break
        time.sleep(10)
    row = {'n': n, 'run': run, 'when': datetime.now().strftime('%Y-%m-%d %H:%M'), 'comp': cfg['comp'],
           'comp_played': rec.get('comp_name'),
           'model': rec.get('model') or cfg['model'], 'effort': rec.get('effort') or cfg['effort'],
           'result': rec.get('result'), 'placement': place,
           'placement_source': rec.get('placement_source') or ('screen' if place else None),
           'rank_before': before, 'rank_after': after, 'ladder_before': ladder(before), 'ladder_after': ladder(after),
           'lp_change': (ladder(after) - ladder(before)) if ladder(after) is not None and ladder(before) is not None else None,
           'code': game_code(run_dir), 'memory': memory_hash(),
           'numbers': process_numbers(run_dir)}
    append(row)
    say(f'game {n} recorded: place {place}, LP change {row["lp_change"]}, rank {after}')
    return row


# what each role may change; the guard takes back anything else it changed (agent/FRAMEWORK.md, "Roles")
MAY_WRITE = {
    'harness_fixer': ('tft_player/', 'tools/', 'tests/', 'agent/reviews/', 'data/'),
    'reviewer': ('agent/model/reviews/', 'agent/reviews/'),
    'strategist': ('agent/comps/', 'agent/model/strategist/', 'agent/reviews/'),
    'comp_coach': ('agent/comps/', 'agent/reviews/'),
}
PROTECTED = ('agent/memory/', 'agent/decisions/', 'agent/judge.py', 'agent/cases.py', 'agent/schemas/',
             'agent/check.py', 'agent/climb.py', 'agent/FRAMEWORK.md', 'agent/roles/', 'agent/verdicts/',
             # the measurement a review is checked against, the records, and the Judge with its cache and inputs
             # (the structure review, 29 Oct: a role could change the HP its tags are checked against, pre-seed the
             # held-out answers, or set its own schedule)
             'agent/metrics.py', 'agent/metrics/', 'agent/games.jsonl', 'agent/model/cache/', 'agent/judge-cache/',
             'agent/model/judge_change.py', 'agent/model/replay.py', 'agent/model/decide.py',
             'agent/model/modelfile.py', 'agent/model/build_memory.py', 'agent/model/game-model.md',
             'agent/model/mechanics.json', 'agent/model/calc.py', 'agent/model/replays/',
             'agent/model/requests.jsonl', 'agent/model/strategist/state.json', 'tools/replay_decisions.py',
             'agent/config.json', 'agent/study_queue.py', 'agent/comp_coach.py', 'agent/rounds.py',
             'agent/transcribe.py', 'agent/comp_select.py',
             # the owner's feedback: the Coordinator's, as changes to the framework from outside it (agent/expert.py)
             'agent/expert/', 'agent/expert.py',
             # the change review's record: verdicts are the Reviewer's, the record is code's (agent/changes.py)
             'agent/changes.jsonl', 'agent/changes.py',
             # the stage strategies and their review code: versions are written by code after the Judge, never by a role
             'agent/stages.py', 'agent/stage_review.py',
             # the live check a harness change must pass before a ranked game, its record and its test (agent/smoke.py)
             'agent/smoke.py', 'agent/smoke.json', 'tests/test_live_check.py', 'agent/model/strategist/owner_rules.json',
             'agent/patch_watch.py', 'agent/patch_watch.json')
# files no role writes wherever they are: a comp's stage strategies (code writes them, after the Judge) and the games
# kept for the Judge
PROTECTED_NAMES = ('/stages.json', '/heldout.json')


def changed_since(head):
    """Paths changed since that commit, committed or not. (git status -z: NUL-separated and unquoted - git() strips its
    output, and a stripped first line " M x" lost the path's first letter.)"""
    out = set(git('diff', '--name-only', head).splitlines())
    raw = subprocess.run(['git', 'status', '--porcelain', '-z', '--untracked-files=all'], cwd=ROOT, capture_output=True,
                         text=True, encoding='utf-8', errors='replace').stdout
    entries = raw.split('\0')
    i = 0
    while i < len(entries):
        e = entries[i]
        if len(e) > 3:
            out.add(e[3:])
            if e[0] in 'RC':
                i += 1                                    # a rename's source path follows; the new path counts
        i += 1
    return {p.replace('\\', '/') for p in out if p}


def file_hash(path):
    import hashlib
    try:
        return hashlib.sha1((ROOT / path).read_bytes()).hexdigest()
    except OSError:
        return None


def snapshot():
    """The tree before a role runs: HEAD, and the content hash of every file that differs from it (dirty tracked files
    and untracked ones - the games' learned pictures, caches, logs). The guard judges a role only by what changed
    after this, so files that were already there are never touched (2026-10-24: a guard without it deleted the
    untracked gallery pictures and learned icons of 21 games and reset two learned-data files)."""
    import shutil
    head = git('rev-parse', 'HEAD')
    files = {p: file_hash(p) for p in changed_since(head)}
    tracked = set(git('ls-tree', '-r', '--name-only', head).splitlines())
    keep = ROOT / 'runs' / 'guard-snapshots' / time.strftime('%Y%m%d-%H%M%S')
    backup = {}
    for p in files:                                      # dirty tracked files: their exact content, to restore
        if p in tracked and (ROOT / p).exists():
            (keep / p).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / p, keep / p)
            backup[p] = (keep / p).relative_to(ROOT).as_posix()
    runs = sorted(os.listdir(ROOT / 'runs')) if (ROOT / 'runs').exists() else []
    return {'head': head, 'files': files, 'backup': backup, 'runs': runs}


def role_changes(snap):
    """Paths the role changed: in its commits since the snapshot, or now different from the snapshot's content."""
    out = set(git('diff', '--name-only', snap['head'], 'HEAD').splitlines())
    for p in changed_since(snap['head']):
        if p not in snap['files'] or file_hash(p) != snap['files'][p]:
            out.add(p)
    return {p.replace('\\', '/') for p in out if p}


LOOP_FILES = ('runs/', 'agent/climb.log', 'agent/climb-', 'agent/dashboard/', 'agent/channel.jsonl',
              'agent/queue.json', 'agent/console/')            # agent/climb-<date>.log: the loop's own output
# written by run_role's own code around a session, never by the role: the Codex usage mark (agent/backends.py) and a
# role's disagreements with the owner (agent/expert.py). Game 55's reviews: the guard took both back after every
# fallback session, so each role tried Codex again and the comp coach's disagreement was lost.
CODE_WRITES = ('agent/model/codex-limit.json', 'agent/expert/disagreements.jsonl')


def forbidden(role, paths):
    """The paths this role may not have changed: protected ones, and any outside its MAY_WRITE (the loop's own files
    aside)."""
    return sorted(p for p in paths if p not in CODE_WRITES and (
                  p.startswith(PROTECTED) or p.endswith(PROTECTED_NAMES)
                  or not p.startswith(MAY_WRITE[role] + LOOP_FILES)
                  # the Strategist writes only its candidates in a comp's folder (agent/comps/<comp>/candidates/)
                  or (role == 'strategist' and p.startswith('agent/comps/') and '/candidates/' not in p)))


def runs_lost(snap):
    """The run folders there before the role and gone after it (runs/ is gitignored: the only copy of every recorded
    game). 2026-11-03, after game 55: the Editor linked runs/ into a scratch worktree with a junction, and
    `git worktree remove --force` deleted all of it through the link. -> the names."""
    now = set(os.listdir(ROOT / 'runs')) if (ROOT / 'runs').exists() else set()
    return sorted(set(snap.get('runs') or ()) - now - {'guard-snapshots', 'guard-quarantine'})


def stop_for_lost_runs(role, lost):
    """Stop the loop before the next game (a game's frames would overwrite what a recovery tool could still find)."""
    why = (f'{time.strftime("%Y-%m-%d %H:%M")}: the {role} session deleted {len(lost)} run folder(s) '
           f'({", ".join(lost[:8])}{" ..." if len(lost) > 8 else ""}); the loop stopped before the next game')
    (AGENT / 'STOP').write_text(why + '\n', encoding='utf-8')
    say(f'guard: {why} - agent/STOP written')


def guard(role, snap):
    """Take back what this role changed outside its files (MAY_WRITE), and anything protected - only what it changed
    (role_changes), and never by deleting: the role's version of each file goes to runs/guard-quarantine/<time>/,
    then the file gets back the content it had before the role (a file that was new is moved out). Run folders the
    role deleted stop the loop (runs_lost). -> the paths."""
    import shutil
    lost = runs_lost(snap)
    if lost:
        stop_for_lost_runs(role, lost)
    bad = forbidden(role, role_changes(snap))
    if not bad:
        return bad
    head = snap['head']
    tracked = set(git('ls-tree', '-r', '--name-only', head).splitlines())
    quarantine = ROOT / 'runs' / 'guard-quarantine' / f'{time.strftime("%Y%m%d-%H%M%S")}-{role}'
    kept = []
    for p in bad:
        src = ROOT / p
        if src.exists():
            (quarantine / p).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, quarantine / p)
        if p in snap.get('backup', {}):
            (ROOT / p).write_bytes((ROOT / snap['backup'][p]).read_bytes())   # dirty before: its content then
        elif p in tracked and p not in snap['files']:
            git('checkout', head, '--', p)               # clean before: as committed
        elif p not in snap['files']:
            if src.exists():
                src.unlink()                             # new from the role (its copy is in the quarantine)
        else:
            kept.append(p)                               # an untracked file that existed: left as the role made it
    if kept:
        say(f'guard: {role} changed untracked files that existed before ({", ".join(kept)[:200]}): left, copies in '
            f'{quarantine.relative_to(ROOT)}')
    for i in range(0, len(bad), 50):                     # short command lines (WinError 206 at ~4,000 paths)
        git('add', '-A', '--', *bad[i:i + 50])
    git('commit', '-q', '-m', f'Guard: take back what the {role} changed outside its files: ' + ', '.join(bad)[:400])
    say(f'guard: {role} changed files it may not: {", ".join(bad)[:300]} - taken back (copies in {quarantine.relative_to(ROOT)})')
    return bad


# One Editor conversation per batch (the owner, 2026-11-06, on a fresh Editor session per request: "should it be a bit
# persistent?"): about half of a fresh session went on finding its way in the code again (requests-20261106-0212:
# 16 of 33 commands before its first edit, redoing what the session before it had understood). Within a batch each
# Editor session resumes the last one; a new batch, EDITOR_RESUMES sessions, another model or a failed resume starts
# fresh. A change the gate took back is told to the next session, with the check's output.
EDITOR_THREAD = REVIEWS / 'editor-thread.json'
EDITOR_RESUMES = 6


def editor_batch():
    """The batch the Editor's conversation belongs to: the last game whose batch review ran."""
    return max([r['n'] for r in full_rows()] + [batch_start() - 1])


def editor_resume(model):
    """The Editor conversation to resume, or None."""
    t = load_json(EDITOR_THREAD) or {}
    ok = (t.get('thread') and t.get('batch') == editor_batch() and t.get('model') == model
          and int(t.get('sessions') or 0) < EDITOR_RESUMES)
    return t if ok else None


def thread_of(log):
    """The Codex session id in a session's JSON log (its thread.started line), or None."""
    try:
        with open(log, encoding='utf-8', errors='replace') as f:
            for line in f:
                if '"thread.started"' in line:
                    return json.loads(line).get('thread_id')
    except (OSError, ValueError):
        pass
    return None


def run_role(role, prompt, tag, cfg, _retry=True, _fresh=False):
    """One role's session: headless Claude Code in this repo, following agent/roles/<role>.md. -> its final JSON.
    While Codex's usage is used up the session runs on the fallback model (agent/backends.py); a Codex session that
    stops on a usage limit marks it and runs again there once. The Editor resumes its batch's conversation
    (EDITOR_THREAD)."""
    from tft_player.coach import claude_cli, cli_env
    import backends
    REVIEWS.mkdir(parents=True, exist_ok=True)
    brief = prompt
    prompt = f'You are the {role.replace("_", " ")}. Follow agent/roles/{role}.md exactly (read it and agent/FRAMEWORK.md first). ' + prompt
    log = REVIEWS / f'{tag}-{role}.jsonl'
    model, effort = backends.choose(cfg.get('coach_model', 'codex:gpt-6.1-sol'), cfg.get('coach_effort', 'high'), cfg)
    codex = model.startswith('codex:')                   # "codex:<model>": Codex on the ChatGPT login, full access here
    editor = role == 'harness_fixer'
    prev = editor_resume(model) if editor and codex and not _fresh else None
    taken = (load_json(EDITOR_THREAD) or {}).get('gate_failed') if editor else None
    if taken:
        prompt = (f"The gate took back your last change: agent/check.py failed (its output: {taken['file']}; its end: "
                  f"{taken['tail']}). " + prompt)
    if prev:
        prompt = (f"A new request in this same Editor conversation. Other commits may have landed since your last turn "
                  f"(`git log --oneline -8`): read a file again before you change it. " + prompt)
    log.with_suffix('.brief.txt').write_text(prompt, encoding='utf-8')     # the control room shows it (agent/console.py)
    if codex:                                            # (game 47's stage 4-5 audits, same packets: GPT-6.1-Sol high
        from tft_player.coach import codex_cli, codex_sandbox, codex_service_tier   # found 3 of the owner's 3 hidden
        cmd = [codex_cli(), 'exec', '--json', '--skip-git-repo-check', '-s', 'danger-full-access',  # findings, Opus
               '-c', 'approval_policy="never"', '-m', model.split(':', 1)[1],                       # 5.5 medium 1)
               '-c', f'model_reasoning_effort="{effort}"']
        if codex_sandbox():
            cmd += ['-c', f'windows.sandbox="{codex_sandbox()}"']
        if codex_service_tier():
            cmd += ['-c', f'service_tier="{codex_service_tier()}"']
        cmd += ['resume', prev['thread'], '--', '-'] if prev else ['-']
    else:
        cmd = [claude_cli(), '-p', prompt, '--model', backends.cli_model(model), '--effort', effort, '--output-format',
               'stream-json', '--verbose', '--permission-mode', 'bypassPermissions']
    env = cli_env()
    env.pop('CLAUDE_CODE_PROMPT_CACHE_TTL', None)        # the roles' turns are minutes apart at times
    env.pop('ANTHROPIC_API_KEY', None)                   # on the Claude login, as the player
    env.pop('ANTHROPIC_AUTH_TOKEN', None)
    say(f'{role}: {tag} ({model} {effort}) ...')
    t0, final = time.time(), None
    with open(log, 'w', encoding='utf-8') as f:
        p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE if codex else subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                             errors='replace', env=env)
        if codex:
            p.stdin.write(prompt)
            p.stdin.close()
        try:
            for line in p.stdout:
                f.write(line)
                if '"type":"result"' in line:
                    try:
                        final = json.loads(line)
                    except ValueError:
                        pass
                elif codex and '"agent_message"' in line:     # Codex: its last message is the answer
                    try:
                        item = json.loads(line).get('item') or {}
                        if item.get('type') == 'agent_message':
                            final = {'result': item.get('text') or ''}
                    except ValueError:
                        pass
                if time.time() - t0 > COACH_TIMEOUT:
                    say(f'{role}: out of time, stopped')
                    break
        finally:
            if p.poll() is None:
                subprocess.run(['taskkill', '/T', '/F', '/PID', str(p.pid)], capture_output=True)
    text = (final or {}).get('result') or ''
    summary = None
    try:
        from tft_player.coach import json_objects
        objs = [o for o in json_objects(text) if isinstance(o, dict)]
        summary = objs[-1] if objs else None
    except Exception:
        summary = None
    if prev and not final:                               # the resume itself failed (no answer at all): a fresh one
        say(f'{role}: its conversation {prev["thread"]} could not be resumed: a fresh session')
        return run_role(role, brief, f'{tag}-fresh', cfg, _retry, _fresh=True)
    if editor and codex:                                 # the conversation the next Editor session resumes
        tid = thread_of(log) or (prev or {}).get('thread')
        if tid:
            same = bool(prev) and tid == prev['thread']
            EDITOR_THREAD.write_text(json.dumps({
                'thread': tid, 'batch': editor_batch(), 'model': model, 'when': time.strftime('%Y-%m-%d %H:%M'),
                'sessions': int(prev.get('sessions') or 0) + 1 if same else 1}) + '\n', encoding='utf-8')
    if codex and not summary and _retry:
        hits = backends.limit_errors(log.read_text(encoding='utf-8', errors='replace'))
        if hits:                                         # Codex used up mid-session: the fallback runs it again
            backends.mark(hits[0], f'{role} {tag}')
            say(f'{role}: Codex hit its usage limit ({hits[0][:120]}): again on the fallback model')
            return run_role(role, brief, f'{tag}-fallback', cfg, _retry=False)
    # a role that disagrees with the owner's feedback says why; it goes to the Coordinator, never into the feedback
    import expert
    expert.note_disagreements(role, tag, summary)
    told = json.dumps(summary, ensure_ascii=False)[:400] if summary else (text[-300:] or 'no answer')
    say(f'{role}: done in {(time.time() - t0) / 60:.1f} min, cost ${(final or {}).get("total_cost_usd", 0):.2f}: {told}')
    # the role's own words to the team channel (its final JSON's "post"); the full session stays in its log
    sys.path.insert(0, str(AGENT))
    from channel import post as channel_post
    words = summary.get('post') if isinstance(summary, dict) else None
    channel_post(role, words or f'I finished ({(time.time() - t0) / 60:.0f} min) without a post for the channel; my answer: '
                 f'{told[:300]}', refs=[str(log.relative_to(ROOT))])
    return {f'{role}': summary, f'{role}_minutes': round((time.time() - t0) / 60, 1),
            f'{role}_cost_usd': (final or {}).get('total_cost_usd')}


CODE = ('tft_player', 'tools', 'tests')


def take_back(before_head, before_files, message, output, name='check-failed'):
    """The harness fixer's code change goes back to before it - files it added included: a `git checkout <head> --
    dirs` kept a new regression test that fails without its fix, and every later check would have failed with it (the
    structure review, 29 Oct). Untracked files that were there before (before_files: the snapshot's) stay; new ones go
    to runs/guard-quarantine/. The next Editor session is told why, with `output`."""
    import shutil
    failed = REVIEWS / f'{name}-{time.strftime("%Y%m%d-%H%M%S")}.txt'
    failed.write_text(output, encoding='utf-8')
    t = load_json(EDITOR_THREAD) or {}                    # the next Editor session is told, with the output
    t['gate_failed'] = {'file': failed.relative_to(ROOT).as_posix(), 'why': message, 'tail': output[-1500:]}
    EDITOR_THREAD.write_text(json.dumps(t) + '\n', encoding='utf-8')
    code = [d for d in CODE if (ROOT / d).exists()]      # (a pathspec that matches nothing fails the whole restore)
    git('restore', f'--source={before_head}', '--staged', '--worktree', '--', *code)
    quarantine = ROOT / 'runs' / 'guard-quarantine' / f'{time.strftime("%Y%m%d-%H%M%S")}-gate'
    raw = subprocess.run(['git', 'status', '--porcelain', '-z', '--untracked-files=all', '--', *code], cwd=ROOT,
                         capture_output=True, text=True, encoding='utf-8', errors='replace').stdout
    for e in raw.split('\0'):
        path = e[3:].replace('\\', '/') if e.startswith('?? ') else None
        if path and path not in (before_files or {}) and (ROOT / path).exists():
            (quarantine / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ROOT / path), str(quarantine / path))
    git('add', '-A', *code)
    git('commit', '-q', '-m', message)


LAST_GATE = {}                       # the last gate's round cases: {'flipped': [ids], 'broken': [ids]}


def cases_now():
    """The round cases on the code as it is now (agent/round_cases.py: harness cases with the game's answers, decision
    cases with their saved answers - no model call) -> {id: result}, the baseline an Editor session's change is judged
    against; None when it could not run."""
    sys.path.insert(0, str(AGENT))
    import round_cases
    try:
        return round_cases.run_sub()
    except Exception as e:
        say(f'round cases: not run ({e.__class__.__name__}: {e})')
        return None


def code_changed(before_head):
    return bool(git('diff', '--name-only', before_head, '--', *[d for d in CODE if (ROOT / d).exists()])
                or git('status', '--porcelain', '--', *[d for d in CODE if (ROOT / d).exists()]))


def gate(before_head, before_files=None, cases=None):
    """agent/check.py, then the round cases (agent/round_cases.py) against `cases` - the same cases on the code
    before the Editor's session: a case that passed before and fails now takes the change back, like a failed check
    (the owner, 2026-11-10: a fix lands only when its case flips, and a case that passed before must still pass).
    LAST_GATE says which cases flipped to pass."""
    r = subprocess.run([PY, str(AGENT / 'check.py')], cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
    ok = r.returncode == 0
    say('check:', 'passed' if ok else 'FAILED - the code changes are taken back')
    LAST_GATE.clear()
    if not ok:
        take_back(before_head, before_files, 'Take back the harness fixer\'s code change: agent/check.py failed',
                  r.stdout + r.stderr)
    elif cases is not None and code_changed(before_head):
        import round_cases
        after = round_cases.run_sub()
        broken, flipped = round_cases.compare(cases, after)
        LAST_GATE.update(flipped=flipped, broken=broken)
        if broken:
            text = ('Round cases that passed before the change and fail after it (agent/round_cases.py; run '
                    '`agent/round_cases.py run <id>` to see one):\n' + '\n'.join(
                        f"[{k}] {(after.get(k) or {}).get('detail')}" for k in broken))
            say(f"round cases: {', '.join(broken)} passed before the change and fail after it - the change is taken "
                f"back")
            take_back(before_head, before_files, f"Take back the harness fixer's code change: round case(s) "
                      f"{', '.join(broken)} passed before it and fail after it", text, name='cases-broken')
            round_cases.run_sub()                         # the results of the code as it is again
            ok = False
        elif flipped:
            say(f"round cases: {', '.join(flipped)} pass now (they failed before the change)")
    if git('status', '--porcelain', 'tft_player', 'tools', 'tests', 'agent/reviews', 'data', 'agent/round_cases'):
        git('add', '-A', 'tft_player', 'tools', 'tests', 'agent/reviews', 'data', 'agent/round_cases')
        git('commit', '-q', '-m', 'Harness fixer changes after the last game (committed by agent/climb.py)')
    return ok


def close_done(ids, how):
    """The Editor's completed requests closed - one with round cases only when they all pass on the new code (the
    owner, 2026-11-10: a fix lands only when its case flips) - and every open request whose cases all pass now. A
    request the Editor listed whose case still fails stays open, with the failing cases named. -> closed ids."""
    import round_cases
    res = round_cases.results()
    close, kept = set(), {}
    for r in requests(kind='editor'):
        cs, all_pass = round_cases.request_state(r['id'], res)
        if cs and all_pass:
            close.add(r['id'])
        elif r['id'] in set(ids or ()):
            if cs:
                kept[r['id']] = [c for c in cs if (res.get(c) or {}).get('status') != 'pass']
            else:
                close.add(r['id'])
    close_requests(close, how)
    if kept:
        rows = requests(None)
        for r in rows:
            if r['id'] in kept:
                r['case_failing'] = kept[r['id']]
        REQUESTS.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        say('requests kept open, their round cases still fail: '
            + '; '.join(f"{k} ({', '.join(v)})" for k, v in kept.items()))
    return sorted(close)


def cases_brief():
    """The round cases as the Editor reads them: the rule, and the failing harness cases, costliest first."""
    sys.path.insert(0, str(AGENT))
    import round_cases
    cs = round_cases.load()
    if not cs:
        return ''
    rows = round_cases.failing_text(cases=[c for c in cs if c['mode'] == 'harness'])
    return (f"Round cases (agent/round_cases.py, {len(cs)} open: real rounds of our games, each with a check of what "
            f"should come out): a request with cases is done only when its cases pass, and every case that passes now "
            f"must still pass after your change - the climb runs them all after your session and takes the change "
            f"back if one breaks. Run `.venv/Scripts/python.exe agent/round_cases.py run <id>` "
            f"before and after your change. " + (('Failing: ' + ' '.join(rows) + ' ') if rows else ''))


def case_requests():
    """Every failing harness case on the Editor's list: a case with no open request gets one (the case is its text),
    so it stays there until it passes (the owner, 2026-11-10). -> the requests made."""
    sys.path.insert(0, str(AGENT))
    import round_cases
    rows, res = round_cases.load(None), round_cases.results()
    open_ids = {r['id'] for r in requests(kind='editor')}
    made = []
    for c in rows:
        r = res.get(c['id']) or {}
        if c.get('status', 'open') != 'open' or c['mode'] != 'harness' or r.get('status') in (None, 'pass') \
                or c.get('request') in open_ids:
            continue
        frm = 'Owner (round case)' if str(c.get('source', '')).lower().startswith('owner') else 'Round cases'
        new = add_requests('editor', [f"Round case {c['id']} fails (game {c.get('game')} {c['round']}, run {c['run']} "
                                      f"call {c['call']}): expect {c['expect']}; check `{c['check']}`; now: "
                                      f"{r.get('detail')}. Fix its cause; the request closes when the case passes."],
                           c.get('game'), frm)
        c['request'] = new[0]['id']
        made += new
    if made:
        round_cases._write(rows)
        commit_paths('Failing round cases on the Editor\'s list (committed by agent/climb.py)', 'agent/round_cases',
                     REQUESTS.relative_to(ROOT))
        say(f"round cases: {len(made)} failing case(s) given to the Editor: {', '.join(r['id'] for r in made)}")
    return made


def cases_from(specs, run, source, request=None, cost=2):
    """Cases a Reviewer wrote with its findings ({round, call, mode, expect, check[, setup, run]}) added: each must
    fail on the round as played. -> [(id, added, why)]."""
    import round_cases
    out = []
    todo = []
    for s in specs or []:
        if not isinstance(s, dict):
            continue
        c = {k: s.get(k) for k in ('round', 'call', 'mode', 'expect', 'check', 'setup', 'id') if s.get(k) is not None}
        c.update(run=s.get('run') or run, source=source, cost=int(s.get('cost') or cost))
        if request:
            c['request'] = request
        todo.append(c)
    for r in round_cases.add_sub(todo):
        out.append((r.get('id'), r.get('added'), r.get('why')))
    if out:
        say('round cases from ' + source + ': ' + '; '.join(f"{i}: {why}" for i, _, why in out))
        commit_paths(f'Round cases from {source} (committed by agent/climb.py)', 'agent/round_cases')
    return out


def editor_from_commits(head):
    """The Editor's work read off its commits since `head` (code paths only): {"fixed": ["<hash> <subject>"],
    "requests_done": [ids named in them, e.g. "request e2"]}."""
    log = git('log', '--format=%h%x09%s%x09%b%x1e', f'{head}..HEAD', '--', 'tft_player', 'tools', 'tests')
    fixed, ids = [], set()
    for rec in log.split('\x1e'):
        parts = rec.strip().split('\t')
        if len(parts) >= 2 and parts[0]:
            fixed.append(f'{parts[0]} {parts[1]}')
            ids |= set(re.findall(r'\b(e\d+)\b', ' '.join(parts[1:])))
    return {'fixed': fixed, 'requests_done': sorted(ids), 'from_commits': True} if fixed else {}


def commit_paths(msg, *paths):
    """git add and commit those of `paths` that exist (one missing pathspec makes `git add` stage nothing - the
    structure review, 29 Oct); -> whether anything was committed."""
    have = [str(x) for x in paths if (ROOT / str(x)).exists()]
    if not have:
        return False
    git('add', '-A', '--', *have)
    if not git('diff', '--cached', '--name-only', '--', *have):
        return False
    git('commit', '-q', '-m', msg)
    return True


def notes_current(comp):
    """The notes the player reads (agent/memory/comps/<comp>.md and agent/memory/general.md: the owner's FIRST RULES)
    rebuilt from the owner's rules files before each game, and committed when that changed them. Before, a rule written
    to agent/expert/rules reached the player only when the comp coach next ran: the x30 rules of 2026-11-09 07:33 were
    first read in game 83, not 81."""
    try:
        import comp_coach
        import expert
        paths = [comp_coach.render(c) for c in (comp, expert.GENERAL)
                 if c and (expert.RULES / f'{expert.slug(c)}.json').exists()]
        if commit_paths('The comp notes rebuilt from the owner rules before the game',
                        *[p.relative_to(ROOT) for p in paths]):
            say('framework: the comp notes rebuilt from the owner rules')
    except Exception as e:                               # never stops a game: the notes stay as they were
        say(f'framework: the notes rebuild failed ({e.__class__.__name__}: {e})')


REQUESTS = AGENT / 'model' / 'requests.jsonl'
STRATEGY_STATE = AGENT / 'model' / 'strategist' / 'state.json'
STRATEGY_EVERY = 5                  # games between Strategist runs when no new pattern starts one
OWNER_CHECK = AGENT / 'model' / 'strategist' / 'owner_rules.json'   # the owner's rules the stage texts were checked on


def owner_rule_keys(comp):
    """{rule id: id plus a hash of its text} of the owner's rules for comp: a changed text is a new rule to check."""
    import hashlib
    import expert
    return {r.get('id') or r['rule'][:30]: f"{r.get('id')}:{hashlib.sha1(r['rule'].encode('utf-8')).hexdigest()[:10]}"
            for r in expert.rules(comp)['rules']}


def owner_rules_unchecked(comp):
    """The owner's rules for comp, added or changed since the current stage texts were last checked against them
    (the meta reviews of games 55-59 and 60-64: stage 4 v3 played on against x11.1, which came after it)."""
    try:
        done = set(json.loads(OWNER_CHECK.read_text(encoding='utf-8')).get(comp) or [])
    except (OSError, ValueError):
        done = set()
    return [i for i, k in owner_rule_keys(comp).items() if k not in done]


def mark_owner_rules_checked(comp):
    try:
        d = json.loads(OWNER_CHECK.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        d = {}
    d[comp] = sorted(owner_rule_keys(comp).values())
    OWNER_CHECK.parent.mkdir(parents=True, exist_ok=True)
    OWNER_CHECK.write_text(json.dumps(d, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def requests(status='open', kind=None):
    rows = [json.loads(l) for l in REQUESTS.read_text(encoding='utf-8').splitlines() if l.strip()] \
        if REQUESTS.exists() else []
    return [r for r in rows if (status is None or r.get('status') == status) and (kind is None or r.get('kind') == kind)]


def left_before(run, games=4, chars=1800):
    """The faults earlier Editors left ("## Left" of the last games' faults reports, before `run`): only this game's
    review reached the Editor, so a fault left once was not seen again unless the Reviewer named it anew (games 40-42:
    the 2-1 plan arriving after the augment questions left the board unfielded three games running)."""
    out = []
    for f in sorted(REVIEWS.glob('2026*-faults.md'), reverse=True):
        if f.name >= f'{run}-faults.md':
            continue
        text = f.read_text(encoding='utf-8', errors='replace')
        part = re.split(r'\n## Left[^\n]*\n', text, maxsplit=1)
        if len(part) == 2:
            left = re.split(r'\n## ', part[1], maxsplit=1)[0].strip()
            out.append(f"{f.name}:\n{left[:chars]}")
        if len(out) >= games:
            break
    return out


def add_requests(kind, texts, game, frm):
    """Requests to the Editor ('editor') or the owner ('owner') -> agent/model/requests.jsonl, with ids."""
    rows = requests(None)
    out = []
    if isinstance(texts, str):                           # one request sent as a string, not a list
        texts = [texts]
    for t in texts or []:
        if not str(t).strip():
            continue
        r = {'id': f'{kind[0]}{len(rows) + len(out) + 1}', 'kind': kind, 'from': frm, 'game': game,
             'when': datetime.now().strftime('%Y-%m-%d %H:%M'), 'text': str(t).strip(), 'status': 'open'}
        out.append(r)
    if out:
        REQUESTS.parent.mkdir(parents=True, exist_ok=True)
        with open(REQUESTS, 'a', encoding='utf-8') as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return out


def close_requests(ids, how):
    if not ids or not REQUESTS.exists():
        return
    rows = requests(None)
    for r in rows:
        if r['id'] in ids and r.get('status') == 'open':
            r.update(status='done', done=how)
    REQUESTS.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def strategy_due(n, pats):
    """(due, why): every STRATEGY_EVERY games, or when a pattern is newly flagged (3+ of the latest 5 games, and not
    flagged at the last run) - the structure's cadence: "every 5 games, or when metrics show a pattern in 3+ games".
    A pattern that stays flagged waits for the 5-game cycle (the structure review, 29 Oct: re-running on the same
    pattern made it every game)."""
    try:
        state = json.loads(STRATEGY_STATE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        state = {}
    new = sorted(set(pats.get('flagged', [])) - set(state.get('flagged') or []))
    if new:
        return True, 'pattern in 3+ of the latest 5 games: ' + ', '.join(new)
    if n - int(state.get('last_game') or 0) >= STRATEGY_EVERY:
        return True, f'{STRATEGY_EVERY} games since the last run'
    return False, 'no new pattern, and fewer than 5 games since the last run'


REFLECT_FROM = 5                    # a 5th-8th place: the reflection before the next game
STUDY_ROUNDS = 2                    # Strategist passes after the first, each on newly read strong players' games


def strategist_pass(row, cfg, about, told, tag):
    """One Strategist run: its requests to the Editor and structural requests (decided by the framework - the Director
    reads them; the owner, 2026-10-29: "I do not make the choice, the framework makes the decision"), its study
    requests, committed. -> (info, the proposal files it wrote)."""
    run, n = row['run'], row['n']
    snap = snapshot()
    info = run_role('strategist', about + told + 'Finish with the JSON line agent/roles/strategist.md asks for.',
                    tag, cfg)
    guard('strategist', snap)
    st = info.get('strategist') if isinstance(info.get('strategist'), dict) else {}
    info['requests'] = add_requests('editor', st.get('editor'), n, 'Strategist') + \
        add_requests('coach', st.get('comp_coach'), n, 'Strategist') + \
        add_requests('structural', st.get('structural') or st.get('owner'), n, 'Strategist')
    study = [x for x in st.get('study') or [] if isinstance(x, dict)]
    if study:
        q = AGENT / 'sources' / 'requests.jsonl'
        with open(q, 'a', encoding='utf-8') as f:
            for x in study:
                f.write(json.dumps({**x, 'game': n, 'status': 'open'}, ensure_ascii=False) + '\n')
        info['study'] = study
    commit_paths(f'Strategist after game {n} ({run}, {tag}) (committed by agent/climb.py)', 'agent/model/proposals',
                 'agent/model/evidence.md', 'agent/model/strategist', 'agent/reviews', REQUESTS.relative_to(ROOT),
                 'agent/sources/requests.jsonl')
    props = sorted(x for x in role_changes(snap) if x.startswith('agent/model/proposals/') and x.endswith('.json'))
    return info, props


def judge_all(n, props):
    """The Judge on each proposal file; an accepted principle is in the playbook, an accepted state change becomes a
    request for the Editor; a crash is a fault, not a rejection. -> [verdicts]."""
    verdicts = []
    before_judge = git('rev-parse', 'HEAD')
    for p in props:
        try:
            prop = json.loads((ROOT / p).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            prop = {}
        if prop.get('type') == 'comp_rule':
            # a hard rule for one comp: kept on its evidence, no replay (the owner, 2026-10-29: hard-coding a decision
            # the review finds the model getting wrong is fine); the executor enforces it - an Editor request
            import comp_coach
            rule, why = comp_coach.add_rule(str(prop.get('comp') or config().get('comp')), prop)
            say(f"comp rule {Path(p).stem}: " + (f"kept - {rule['rule'][:160]}" if rule else f'not kept - {why}'))
            if rule:
                add_requests('editor', [f"Comp {prop.get('comp') or config().get('comp')}: enforce this rule in the "
                                        f"executor, even when a plan breaks it: {rule['rule']} (why: {rule['why']}). "
                                        f"Rule file: agent/comps/{comp_coach.slug(str(prop.get('comp') or config().get('comp')))}/rules.json."],
                             n, 'Strategist')
            verdicts.append({'proposal': p, 'accept': bool(rule), 'reasons': [why] if why else ['comp rule kept'],
                             'comp_rule': True})
            continue
        say(f'judge: {p} ...')
        r = subprocess.run([PY, str(AGENT / 'model' / 'judge_change.py'), str(ROOT / p)], cwd=ROOT,
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        out = ROOT / 'agent' / 'model' / 'replays' / f'{Path(p).stem}.json'
        try:
            v = json.loads(out.read_text(encoding='utf-8')) if r.returncode == 0 and out.exists() else None
        except ValueError:
            v = None
        if v is None:                                    # a crash is not a rejection: it is a fault to fix
            err = (r.stderr.strip() or r.stdout.strip())[-400:]
            say('judge: FAILED -', err)
            verdicts.append({'proposal': p, 'accept': False, 'failed': err})
            continue
        say('judge:', 'accepted' if v.get('accept') else 'rejected - ' + '; '.join(v.get('reasons') or [])[:300])
        if v.get('accept') and v.get('build'):           # an accepted state change: the Editor builds it
            add_requests('editor', [f"Build the state change the Judge accepted ({Path(p).stem}): {v['build']}"],
                         n, 'Judge')
        verdicts.append({'proposal': p, 'accept': bool(v.get('accept')), 'reasons': v.get('reasons'),
                         'state': v.get('kind') == 'state'})
    commit_paths(f'Judge verdicts after game {n} (committed by agent/climb.py)', 'agent/model/replays',
                 'agent/model/proposals', REQUESTS.relative_to(ROOT), 'agent/comps', 'agent/memory/comps')
    if any(v['accept'] and not v.get('state') for v in verdicts) and not gate_ok():
        # the playbook the accepted changes made breaks the check: all of this step's changes go back, as rejections
        git('restore', f'--source={before_judge}', '--staged', '--worktree', '--', 'agent/model/game-model.md',
            'agent/memory/playbook.md')
        commit_paths(f'Take back the changes accepted after game {n}: agent/check.py failed with them',
                     'agent/model/game-model.md', 'agent/memory/playbook.md')
        for v in verdicts:
            if v['accept'] and not v.get('state'):
                v.update(accept=False, reasons=['accepted, then taken back: agent/check.py failed with it'])
    return verdicts


def comp_videos(comp, n=12):
    """Strong players' games of this comp on YouTube (yt-dlp search, no login), for the reflection; [] if none."""
    try:
        import study_queue
        return study_queue.search_youtube(f'TFT {comp}', n)
    except Exception as e:
        say(f'comp videos: {e.__class__.__name__}: {e}')
        return []


def strategy_step(row, cfg, about, editor_notes=None):
    """The strategy loop. After a 5th-8th place it always runs, as a reflection: the game's verdict (luck, strategy,
    execution, information - from the Reviewer's grades), strong players' games of the same comp found and studied,
    the Strategist's proposals from them tested by the Judge, and one revision round if every proposal was rejected
    while strategy lost the game. Otherwise when a pattern is newly flagged or every 5 games. -> info for the record."""
    import metrics
    run, n = row['run'], row['n']
    try:
        pats = metrics.patterns()
    except Exception as e:                               # the loop goes on: no strategy loop this game, and why
        say(f'strategy loop: no patterns ({e.__class__.__name__}: {e})')
        return {'strategy_due': f'patterns failed: {e.__class__.__name__}'}
    place = row.get('placement')
    reflect = isinstance(place, int) and place >= REFLECT_FROM
    due, why = (True, f'place {place}: the reflection') if reflect else strategy_due(n, pats)
    asked = requests(kind='strategist')                  # removals from the change review (agent/changes.py)
    if asked and not due:
        due, why = True, f'{len(asked)} removal(s) from the change review'

    say(f'strategy loop: {"due" if due else "not due"} ({why})')
    if not due:
        return {'strategy_due': why}
    notes = [x for x in (editor_notes or []) if str(x).strip()]
    base = (f"This game's round-by-round review is agent/model/reviews/{run}.md, every decision's grade in "
            f"agent/model/reviews/{run}-audit/ (the better decision, the player's reason, whether it holds, the cause): "
            f"examine every decision graded bad and every reason that does not hold. agent/metrics/patterns.json lists "
            f"the patterns across games; open requests are in agent/model/requests.jsonl. "
            + ('The Editor\'s strategy notes from this game: ' + ' '.join(f'- {x}' for x in notes[:6]) + ' '
               if notes else '')
            + ('The change review asks you to remove these principles (a remove proposal each, for the Judge): '
               + ' '.join(f"[{r['id']}] {r['text']}" for r in asked[:6]) + ' ' if asked else ''))
    close_requests({r['id'] for r in asked[:6]}, f'handed to the Strategist after game {n}')
    verdict, reasons = {}, []
    try:
        au = json.loads((metrics.OUT / f'{run}.json').read_text(encoding='utf-8')).get('audit') or {}
        verdict, reasons = au.get('verdict') or {}, au.get('reasons') or []
    except (OSError, ValueError, AttributeError):         # no metrics file (or a stand-in module): no verdict
        pass
    if not reflect:
        info, props = strategist_pass(row, cfg, about, f'The strategy loop is due: {why}. ' + base, run)
        info.update(strategy_due=why, verdicts=judge_all(n, props))
    else:
        videos = comp_videos(row.get('comp') or cfg.get('comp'))
        vlist = '; '.join(f"{v['url']} ({v['duration'] // 60} min, {v['channel']}: {v['title'][:70]})"
                          for v in videos if v.get('duration')) or 'none found'
        told = (f"REFLECTION: this game placed {place}. Before the next game, find out why and change what made it. The "
                f"Reviewer's ranked reasons for the place: {json.dumps(reasons, ensure_ascii=False)[:900]}. Its bad "
                f"decisions by cause (very bad counts 2): {json.dumps(verdict)} (strategy = the player's reasoning; "
                f"information = the state lacked what the decision needed; execution = the harness did not do what was "
                f"decided; luck = no bad decision decided it). Decide "
                f"first whether it was an unlucky game, a strategy fault, an execution fault or missing information, "
                f"with the evidence, and say so in \"verdict\". If strategy: study how strong players play this comp - "
                f"pick 1-2 full games from these YouTube videos of it (study requests with \"url\"; code downloads "
                f"them, reads their decisions and plays along with them before your next pass): {vlist}. " + base)
        info, props = strategist_pass(row, cfg, about, told, f'{run}-reflect')
        if info.get('study'):
            try:
                import study_queue
                info['study_queue'] = study_queue.work()
            except Exception as e:
                say(f'study queue: {e.__class__.__name__}: {e}')
        done = list((info.get('study_queue') or {}).get('done') or [])
        studied, round_ = list(done), 2
        # study rounds: while the Strategist has read new games, has no proposal yet and asks for more, the games are
        # read and it gets another pass (game 40's reflection stopped at "watching" after one game, asking for two
        # more) - at most STUDY_ROUNDS passes after the first
        while done and round_ <= 1 + STUDY_ROUNDS:
            more, props2 = strategist_pass(row, cfg, about, (
                f"REFLECTION, pass {round_} (game placed {place}): the strong players' games you asked for are read: "
                f"{', '.join(done)} (all studied so far: {', '.join(studied)}). Their play-alongs are in "
                f"agent/model/playalong/ (our player against theirs, round by round) and their own words in "
                f"runs/vod-*/game*.vtt captions. Compare how they played this comp with this game, and propose what to "
                f"change - a principle, or a state proposal when the player lacked a number. If the evidence is still "
                f"short, ask for the games that would settle it." + (' This is the last study pass: propose from what you '
                                                                     'have.' if round_ == 1 + STUDY_ROUNDS else '')
                + ' ' + base), f'{run}-reflect{round_}')
            props += props2
            info[f'strategist_{round_}'] = more.get('strategist')
            info['requests'] = (info.get('requests') or []) + (more.get('requests') or [])
            done = []
            if more.get('study') and not props2 and round_ < 1 + STUDY_ROUNDS:
                try:
                    import study_queue
                    q = study_queue.work()
                    done = list(q.get('done') or [])
                    studied += done
                except Exception as e:
                    say(f'study queue: {e.__class__.__name__}: {e}')
            round_ += 1
        info['studied'] = studied
        verdicts = judge_all(n, props)
        main = (verdict or {}).get('main')
        if main == 'strategy' and verdicts and not any(v['accept'] for v in verdicts):
            names = ', '.join(Path(v['proposal']).stem for v in verdicts)
            more, props3 = strategist_pass(row, cfg, about, (
                f"REFLECTION, revision (game placed {place}, lost to strategy): the Judge rejected {names} - read "
                f"agent/model/replays/<name>.md for each (what the current playbook already decides in those rounds, "
                f"and what the candidate changed). Revise: a change that moves the decisions the game got wrong, or a "
                f"state change if the player lacked a number it needed. " + base), f'{run}-reflect3')
            verdicts += judge_all(n, props3)
            info['strategist_3'] = more.get('strategist')
            info['requests'] = (info.get('requests') or []) + (more.get('requests') or [])
        info.update(strategy_due=why, verdicts=verdicts, reflection={'place': place, 'verdict': verdict,
                                                                     'videos': len(videos),
                                                                     'studied': info.get('studied') or []})
    STRATEGY_STATE.parent.mkdir(parents=True, exist_ok=True)
    STRATEGY_STATE.write_text(json.dumps({'last_game': n, 'flagged': pats.get('flagged', []),
                                          'seen': {k: [g['run'] for g in v] for k, v in pats['signals'].items()}},
                                         indent=1), encoding='utf-8')
    commit_paths(f'Strategy loop state after game {n}', STRATEGY_STATE.relative_to(ROOT))
    return info


def report(row, info):
    """The Coordinator's three lines for the owner, in the channel: the game, what the Editor fixed, what the strategy
    loop changed or asks."""
    sys.path.insert(0, str(AGENT))
    from channel import post as channel_post
    au = info.get('audit') or {}
    # the place and the decisions, never HP (the owner, 2026-10-29): the Reviewer's ranked reasons, then its grades
    reasons = '; '.join(str(r.get('reason'))[:90] for r in (au.get('reasons') or [])[:2])
    areas = sorted((au.get('by_area') or {}).items(), key=lambda kv: -kv[1])[:4]
    by = ((f'Decided by: {reasons}. ' if reasons else '')
          + (f"Bad decisions: {au.get('bad')} of {au.get('graded')} graded"
             + (f" ({', '.join(f'{k} {v}' for k, v in areas)})" if areas else '') if au.get('graded') else
             'Not reviewed round by round'))
    fx = info.get('harness_fixer') if isinstance(info.get('harness_fixer'), dict) else {}
    fixed = '; '.join(str(x) for x in (fx.get('fixed') or [])[:3]) or 'nothing'
    if info.get('check_passed') is False and fx.get('fixed'):
        fixed += ' - taken back: agent/check.py failed with it'      # (game 35's report named a fix the gate undid)
    name = lambda v: Path(v['proposal']).stem
    acc = [name(v) for v in info.get('verdicts') or [] if v.get('accept')]
    rej = [name(v) for v in info.get('verdicts') or [] if not v.get('accept') and not v.get('failed')]
    bad = [name(v) for v in info.get('verdicts') or [] if v.get('failed')]
    owner = [r['id'] for r in info.get('requests') or [] if r['kind'] in ('owner', 'structural')]
    refl = info.get('reflection') or {}
    dl = (info.get('study_queue') or {}).get('needs_download') or []
    rank = row.get('rank_after') or {}
    lines = [f"Game {row['n']} ({row.get('comp')}): place {row.get('placement')}, LP {row.get('lp_change'):+d} -> "
             f"{rank.get('tier', '?')} {rank.get('division', '')} {rank.get('lp', '')}. {by}."
             if isinstance(row.get('lp_change'), int) else
             f"Game {row['n']} ({row.get('comp')}): place {row.get('placement')}. {by}.",
             f"Editor: {fixed}.",
             ('Diff reports: ' + ', '.join(f"stage {s} {(d or {}).get('verdict')}" for s, d in info['diffs'].items())
              + '. Stage strategies: ' + ('; '.join(f"stage {s}: {r['done']}" for s, r in
                                                    (info.get('stage_strategies') or {}).items()) or 'none changed') + '.')
             if info.get('diffs') else
             ('Strategy: ' + (f"in the playbook now: {', '.join(acc)}; " if acc else '')
              + (f"rejected by the Judge: {', '.join(rej)}; " if rej else '')
              + (f"the Judge failed on: {', '.join(bad)} (a fault to fix); " if bad else '')
              + (f"reflection: verdict {(refl.get('verdict') or {}).get('main') or '?'}, studied "
                 f"{', '.join(refl.get('studied') or []) or 'no video'}; " if refl else '')
              + (f"structural requests for the Director: {', '.join(owner)}; " if owner else '')
              + (f"videos to download for the Strategist: {', '.join(dl)}." if dl else '')
              if 'strategist' in info else f"Strategy loop not run ({info.get('strategy_due', 'not due')}).")]
    cr = info.get('change_review') or {}
    if cr.get('changes'):                                # the change review (agent/changes.py), when it ran
        lines.append(f"Changes since the last game: {cr['changes']} examined - {cr.get('keep', 0)} keep, "
                     f"{cr.get('remove', 0)} remove" + (f" ({'; '.join(cr['removed'][:4])})" if cr.get('removed') else '')
                     + f", {cr.get('monitor', 0)} watched.")
    text = ' | '.join(l.strip() for l in lines if l.strip())
    channel_post('Coordinator', text, to='Owner')
    say('report:', text)
    return lines


def gate_ok():
    r = subprocess.run([PY, str(AGENT / 'check.py')], cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
    return r.returncode == 0


def approve(rid, note=''):
    """The owner's yes to a request (python agent/climb.py --approve o3 [--note ...]): the Editor builds it next."""
    rows = requests(None)
    r = next((x for x in rows if x['id'] == rid), None)
    if not r or r.get('kind') != 'owner':
        sys.exit(f'{rid} is not an owner request')
    close_requests({rid}, 'approved by the owner' + (f': {note}' if note else ''))
    new = add_requests('editor', [f"Build the change the owner approved ({rid}): {r['text']}"
                                  + (f" The owner's note: {note}" if note else '')], r.get('game'), 'Owner')
    commit_paths(f'The owner approved {rid}: the Editor builds it next', REQUESTS.relative_to(ROOT))
    return new


def dashboard():
    subprocess.run([PY, str(AGENT / 'dashboard.py')], cwd=ROOT)
    subprocess.run([PY, str(AGENT / 'live_eval.py')], cwd=ROOT)


def roles_on(run_name, cfg):
    """The steps after a game (4-9) on a game already recorded in agent/games.jsonl (no new record)."""
    row = next((r for r in records() if r['run'] == run_name), None)
    if not row:
        sys.exit(f'{run_name} is not in agent/games.jsonl')
    run_after(row, cfg)


AUDITORS = 3                        # stage audits run side by side


def game_comp(run):
    """The comp a recorded game played (its run_config), or the climb's comp."""
    sys.path.insert(0, str(AGENT))
    import metrics
    try:
        ev = metrics.events(ROOT / 'runs' / run)
        comp = next((e.get('comp') for e in ev if e.get('ev') == 'run_config' and e.get('comp')), None)
        # a flex game: the comp it committed to (its last switch), reviewed like a fixed-comp game (the owner, 11-07:
        # "it is exactly the same framework as the brambleback thing")
        comp = comp or next((e.get('new') or e.get('comp') for e in reversed(ev)
                             if e.get('ev') in ('commit', 'pivot') and (e.get('new') or e.get('comp_id'))), None)
    except Exception:
        comp = None
    return comp or config().get('comp')


def diff_reports(run, about, cfg, audited):
    """Part 2 of each stage's review (the owner, 2026-10-31: "compare the stage report with the prior games. this
    audits the stage strategy, a diff report"): code builds each strategy stage's packet - this game's stage against
    the same stage of the comp's earlier games, the strategy version each played, the watches due - and the Reviewer
    writes diff-<s>.json (Job 4): what the strategy changed, what the play changed, the watches, and keep / revise /
    revert / not enough evidence. -> ({stage: diff}, cost)."""
    import stage_review
    from concurrent.futures import ThreadPoolExecutor
    comp = game_comp(run)
    keys = sorted({stage_review.stage_of(f'{s}-1') for s in audited} - {''})
    folder = AGENT / 'model' / 'reviews' / f'{run}-audit'

    def one(s):
        path, _ = stage_review.diff_packet(run, s, comp, stage_review.due_watches(comp, s, run))
        brief = (about + f"Your job: Job 4, the diff report of strategy stage {s} ({comp}). The packet is "
                         f"{path.relative_to(ROOT).as_posix()}; this game's audit of the stage is in "
                         f"agent/model/reviews/{run}-audit/. Write agent/model/reviews/{run}-audit/diff-{s}.json. ")
        r = run_role('reviewer', brief + 'Finish with the JSON line Job 4 asks for.', f'{run}-diff{s}', cfg)
        miss = stage_review.diff_gaps(stage_review.diff_of(run, s))
        if miss:
            say(f'reviewer: diff report of stage {s} leaves {len(miss)} gap(s): {"; ".join(miss[:4])}')
            r2 = run_role('reviewer', brief + 'Your diff report is not complete - complete the same file (keep what is '
                          'there): ' + '; '.join(miss) + '. Finish with the JSON line Job 4 asks for.',
                          f'{run}-diff{s}-fix', cfg)
            r['reviewer_cost_usd'] = float(r.get('reviewer_cost_usd') or 0) + float(r2.get('reviewer_cost_usd') or 0)
        return float(r.get('reviewer_cost_usd') or 0)

    with ThreadPoolExecutor(max_workers=AUDITORS) as pool:
        cost = sum(pool.map(one, keys))
    out = {s: stage_review.diff_of(run, s) for s in keys}
    say('diff reports: ' + ', '.join(f"stage {s} {(d or {}).get('verdict', 'missing')}" for s, d in out.items()))
    return {s: {k: (d or {}).get(k) for k in ('verdict', 'why')} for s, d in out.items()}, cost


def stage_strategy_step(row, cfg, about, diffs):
    """The Strategist (the owner, 2026-10-31: "it should just read the review and then change the strategy"): one
    session per stage the diff reports mark revise or revert; it decides what to change (or to keep it) and writes the
    new text with its watch; code checks a rewrite with the Judge on the stage's recorded decisions (agent/judge.py
    stage_gate) and ships it only if it scores no lower; a revert ships at once. -> info for the record."""
    import stages
    import judge
    from concurrent.futures import ThreadPoolExecutor
    import meta_review
    run, n = row['run'], row['n']
    comp = game_comp(run)
    due = {s: d for s, d in (diffs or {}).items() if (d or {}).get('verdict') in ('revise', 'revert')}
    # the meta review's stages (agent/meta_review.py): read by the first Strategist run after it, with this game's
    # diff report (a stage the diff marked keep is due when the batch says otherwise)
    meta, meta_stages = meta_review.strategist_note()
    for s in sorted(meta_stages & set(stages.NAMES) - set(due)):
        due[s] = {'verdict': f"{((diffs or {}).get(s) or {}).get('verdict') or 'not reviewed'}, and the meta review "
                             f"asks for a change"}
    # an owner rule added or changed since the stage texts were checked against them: every stage is due, rewritten
    # only where it disagrees with the FIRST RULES
    unchecked = owner_rules_unchecked(comp)
    if unchecked:
        for s in sorted(set(stages.NAMES) - set(due)):
            due[s] = {'verdict': f"not marked for a change, but the owner's rule(s) {', '.join(unchecked[:8])} came "
                                 f"after this text was last checked against the FIRST RULES: check it against them and "
                                 f"rewrite it only where it disagrees, otherwise keep it"}
    # a stage whose decision cases fail with its current text (agent/round_cases.py; the owner, 2026-11-10): due,
    # and the Judge accepts a rewrite that makes them pass and breaks none
    import round_cases
    case_fail = {}
    for c in round_cases.failing():
        if c['mode'] == 'decision' and c.get('comp') == comp:
            case_fail.setdefault(round_cases.stage_of(c['round']), []).append(c)
    for s, cs in sorted(case_fail.items()):
        if s in stages.NAMES and s not in due:
            due[s] = {'verdict': f"not marked for a change, but its round case(s) {', '.join(c['id'] for c in cs)} "
                                 f"fail with the current text"}
    if not due:
        say('strategist: no stage marked revise or revert')
        return {'strategy_due': 'no stage marked revise or revert'}
    cand = AGENT / 'comps' / stages.slug(comp) / 'candidates'
    cand.mkdir(parents=True, exist_ok=True)
    snap = snapshot()

    def one(s):
        out = cand / f'{run}-stage-{s}.json'
        brief = (about + f"Your job: the stage strategy of {stages.NAMES[s]} for {comp}. The diff report marks it "
                         f"{due[s]['verdict']}: agent/model/reviews/{run}-audit/diff-{s}.json; the stage reports are "
                         f"agent/model/reviews/{run}-audit/stage-*.json; the diff packet is agent/reviews/{run}-diff/"
                         f"stage-{s}.md; the versions so far are in agent/comps/{stages.slug(comp)}/stages.json "
                         f"(`python agent/stages.py history \"{comp}\" {s}`). Write {out.relative_to(ROOT).as_posix()}. "
                 + meta
                 + (("Round cases of this stage that fail with the current text (agent/round_cases.py: a real round of "
                     "our games, the question asked again of the game model; the Judge accepts a rewrite that makes "
                     "them pass and breaks no other case of the stage): "
                     + ' '.join(round_cases.failing_text(cases=case_fail[s])) + ' ') if case_fail.get(s) else ''))
        return s, run_role('strategist', brief + 'Finish with the JSON line agent/roles/strategist.md asks for.',
                           f'{run}-strategy{s}', cfg)

    with ThreadPoolExecutor(max_workers=AUDITORS) as pool:
        sessions = dict(pool.map(one, sorted(due)))
    guard('strategist', snap)
    if meta:
        meta_review.mark_strategist_had(n)
    if unchecked:
        mark_owner_rules_checked(comp)
        commit_paths('Stage texts checked against the owner rules', OWNER_CHECK.relative_to(ROOT))
    results = {}
    for s in sorted(due):
        f = cand / f'{run}-stage-{s}.json'
        try:
            c = json.loads(f.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            results[s] = {'done': 'no candidate written'}
            continue
        act = str(c.get('action') or '').lower()
        if act == 'keep':
            results[s] = {'done': 'kept', 'why': c.get('why')}
        elif act == 'revert':
            v, why = stages.revert(comp, s, str(c.get('why') or ''), by=f'Strategist after game {n}')
            results[s] = {'done': f"reverted to version {v['v']}" if v else f'not reverted: {why}'}
        elif act == 'rewrite':
            cur = stages.current(comp).get(s)
            ok, why = stages.check(c.get('text'), c.get('sources'))
            if not ok:
                results[s] = {'done': f'not tested: {why}'}
                continue
            say(f'judge: stage {s} strategy of {comp} ...')
            try:
                v = judge.stage_gate(comp, s, cur['text'] if cur else '', c['text'])
            except Exception as e:                       # a Judge crash is a fault, not a rejection
                results[s] = {'done': f'judge failed: {e.__class__.__name__}: {e}'}
                continue
            if v['accept']:
                ver, why = stages.add_version(comp, s, c['text'], c.get('why'), c.get('sources') or [],
                                              by=f'Strategist after game {n}', watch=c.get('watch'), judge=v)
                results[s] = {'done': f"version {ver['v']} in use ({v['reason']})" if ver else f'not written: {why}'}
            else:
                stages.reject(comp, s, c['text'], c.get('why'), by=f'Strategist after game {n}', judge=v)
                results[s] = {'done': f"rejected by the Judge ({v['reason']})"}
        else:
            results[s] = {'done': f'no action ({act!r})'}
        say(f"strategist: stage {s}: {results[s]['done']}")
    commit_paths(f'Stage strategies of {comp} after game {n} (committed by agent/climb.py)', 'agent/comps',
                 'agent/model/reviews', 'agent/reviews', 'agent/sources/requests.jsonl')
    study = [x for r in sessions.values() for x in ((r.get('strategist') or {}).get('study') or [])
             if isinstance(r.get('strategist'), dict) and isinstance(x, dict)]
    if study:                                            # its study requests go to Learn
        with open(AGENT / 'sources' / 'requests.jsonl', 'a', encoding='utf-8') as fh:
            for x in study:
                fh.write(json.dumps({**x, 'game': n, 'status': 'open'}, ensure_ascii=False) + '\n')
    return {'strategist': {s: (r.get('strategist') if isinstance(r.get('strategist'), dict) else None)
                           for s, r in sessions.items()},
            'strategist_cost_usd': sum(float(r.get('strategist_cost_usd') or 0) for r in sessions.values()),
            'stage_strategies': results, 'study': study}


def review_game(run, about, cfg):
    """The Reviewer, round by round (the owner, 2026-10-29: "review every single round for every single action ...
    whether the decision made was the best decision"; "HP should not be used at all"): code builds each stage's
    packet (agent/rounds.py); one audit session per stage grades every decision of every round, side by side; code
    checks each audit is complete and sends a stage back once with what it left out; the summary session ranks what
    decided the place and the worst decisions; code adds the round-by-round grades below it and attaches the tally to
    the game's metrics. -> info for the record."""
    import metrics
    import rounds as rounds_mod
    from concurrent.futures import ThreadPoolExecutor
    rounds, stages = rounds_mod.write_packets(ROOT / 'runs' / run)
    snap = snapshot()                                    # the sessions' writes are guarded; code's (metrics) after it
    folder = AGENT / 'model' / 'reviews' / f'{run}-audit'
    folder.mkdir(parents=True, exist_ok=True)
    out = AGENT / 'model' / 'reviews' / f'{run}.md'

    def audit_of(s):
        f = folder / f'stage-{s}.json'
        try:
            return json.loads(f.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None

    def audit(s, missing=None):
        brief = (about + f"Your job: Job 1, audit stage {s}. The packet is agent/reviews/{run}-rounds/stage-{s}.md; write "
                         f"agent/model/reviews/{run}-audit/stage-{s}.json. ")
        if missing:
            brief += ("Your audit of this stage is not complete - complete the same file (keep what is there): "
                      + '; '.join(missing[:40]) + '. ')
        return run_role('reviewer', brief + 'Finish with the JSON line Job 1 asks for.',
                        f'{run}-stage{s}' + ('-fix' if missing else ''), cfg)

    cost = 0.0
    with ThreadPoolExecutor(max_workers=AUDITORS) as pool:
        for r in pool.map(audit, stages):
            cost += float(r.get('reviewer_cost_usd') or 0)
    for s in stages:                                     # one more pass for a stage with gaps
        missing = rounds_mod.gaps(rounds, audit_of(s) or {}, s)
        if missing:
            say(f'reviewer: stage {s} audit leaves {len(missing)} gap(s): {"; ".join(missing[:4])}')
            cost += float(audit(s, missing).get('reviewer_cost_usd') or 0)
    audits = [a for a in (audit_of(s) for s in stages) if a]
    left = [g for s in stages for g in rounds_mod.gaps(rounds, audit_of(s) or {}, s)]
    grades = rounds_mod.render(rounds, audits)
    (AGENT / 'model' / 'reviews' / f'{run}-rounds.md').write_text(grades, encoding='utf-8')
    diffs, dcost = diff_reports(run, about, cfg, stages)
    cost += dcost
    info = run_role('reviewer', about + (
        f"Your job: Job 2, the game summary. The stage audits and the diff reports are agent/model/reviews/{run}-audit/ "
        f"(stage-<n>.json, diff-<n>.json); the round-by-round report is agent/model/reviews/{run}-rounds.md; the "
        f"measured numbers are agent/metrics/{run}.json. Finish with the JSON line Job 2 asks for."), run, cfg)
    guard('reviewer', snap)
    import stage_review
    try:
        info['report_page'] = str(stage_review.report_page(run, folder=AGENT / 'model' / 'reviews').relative_to(ROOT))
    except Exception as e:
        say(f'report page: {run} failed ({e.__class__.__name__}: {e})')
    info['diffs'] = diffs
    info['reviewer_cost_usd'] = cost + float(info.get('reviewer_cost_usd') or 0)
    text = out.read_text(encoding='utf-8') if out.exists() else f'# Review: {run}\n\n(no summary was written)\n'
    if '## Round by round' not in text:
        out.write_text(text.rstrip() + '\n\n' + grades, encoding='utf-8')
    rv = info.get('reviewer') if isinstance(info.get('reviewer'), dict) else {}
    tally = rounds_mod.tally(rounds, audits)
    tally['gaps'] = left
    try:
        info['audit'] = metrics.attach_audit(run, tally, rv.get('placement_reasons'))
    except Exception as e:
        say(f'metrics: the audit of {run} not attached ({e.__class__.__name__}: {e})')
        info['audit'] = tally
    say(f"reviewer: {tally['graded']} decisions graded, {tally['bad']} bad ({tally['very_bad']} very bad)"
        + (f", {len(left)} gap(s) left" if left else ''))
    return info


def change_review(run, about, cfg, since=None):
    """The change review (the owner, 2026-10-31: every change to the harness and the game is examined in the next game
    or more, and kept, removed or watched; a change found not useful is reverted): code lists the changes since the
    previous game and those still watched (agent/changes.py); the Reviewer gives each a verdict (Job 3); code records
    them and sends each removal to the role that owns that kind of change. -> info for the record."""
    sys.path.insert(0, str(AGENT))
    import changes
    path, listed = changes.packet(run, since)
    if not listed:
        return {'change_review': {'changes': 0}}
    snap = snapshot()
    brief = (about + f"Your job: Job 3, the change review. The packet is {path.relative_to(ROOT).as_posix()} "
                     f"({len(listed)} change(s)); write agent/model/reviews/{run}-changes.json. ")
    info = run_role('reviewer', brief + 'Finish with the JSON line Job 3 asks for.', f'{run}-changes', cfg)
    missing = changes.gaps(run, listed)
    if missing:
        say(f'change review: {len(missing)} gap(s): {"; ".join(missing[:4])}')
        info = run_role('reviewer', brief + 'Your verdict file is not complete - complete the same file (keep what is '
                        'there): ' + '; '.join(missing[:30]) + '. Finish with the JSON line Job 3 asks for.',
                        f'{run}-changes-fix', cfg)
    guard('reviewer', snap)
    rows = changes.record(run, listed)
    open_texts = [r['text'] for r in requests()]
    made = []
    for kind, text in changes.removal_requests(rows, open_texts):
        made += add_requests(kind, [text], changes.game_of(run), 'Reviewer (change review)')
    out = changes.summary(rows) | {'changes': len(listed), 'requests': [r['id'] for r in made],
                                   'left': changes.gaps(run, listed)}
    say(f"change review: {len(listed)} change(s): {out['keep']} keep, {out['remove']} remove, {out['monitor']} monitor"
        + (f" - requests {', '.join(out['requests'])}" if made else ''))
    commit_paths(f'Change review of {run}: {out["keep"]} keep, {out["remove"]} remove, {out["monitor"]} monitor '
                 f'(committed by agent/climb.py)', 'agent/changes.jsonl', 'agent/model/reviews', 'agent/reviews',
                 REQUESTS.relative_to(ROOT))
    return {'change_review': out, 'reviewer_changes_cost_usd': info.get('reviewer_cost_usd')}


def removed_note(kinds):
    """The parts the change review removed, for the roles that would build them again."""
    sys.path.insert(0, str(AGENT))
    import changes
    rows = [r for r in changes.removed(kinds) if not r.get('part')][-10:] or changes.removed(kinds)[-10:]
    return ('Removed by the change review - do not build them again: ' + ' '.join(
        f"[{r['change']}] {r['part'] or r['subject']}: {r['evidence'][:160]}" for r in rows) + ' ') if rows else ''


def run_after(row, cfg):
    run = row['run']
    digest = REVIEWS / f'{run}-digest.md'
    REVIEWS.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, str(AGENT / 'digest.py'), str(ROOT / 'runs' / run), '--out', str(digest)], cwd=ROOT)
    about = (f"Game {row['n']}: run {run}, {cfg['comp']}, place {row['placement']}, LP change {row['lp_change']}. "
             f"The packet is {digest.relative_to(ROOT)}; the run folder is runs/{run}/. ")
    info = {}
    review = AGENT / 'model' / 'reviews' / f'{run}.md'
    sys.path.insert(0, str(AGENT))
    import metrics
    try:                                                 # the measured game, for the Reviewer and the Strategist
        if (ROOT / 'runs' / run).exists():
            metrics.write(ROOT / 'runs' / run)
    except Exception as e:
        say(f'metrics: {run} failed ({e.__class__.__name__}: {e})')
    if cfg.get('strategy') == 'model':
        # the owner's structure (2026-10-27, step 4): the Reviewer reads the game first; its harness list goes to the
        # Editor (harness fixer) with the packet - before, the fixer ran first and the review's harness faults reached
        # no one (game 32: H1, H3-H6 left)
        info.update(review_game(run, about, cfg))         # (its sessions are guarded inside)
        git('add', '-A', 'agent/model/reviews', 'agent/reviews', 'agent/metrics')
        git('commit', '-q', '-m', f'Round-by-round review of {run} (committed by agent/climb.py)')
        try:                                             # before the Editor: its removals are the Editor's requests
            info.update(change_review(run, about, cfg))
        except Exception as e:
            say(f'change review: {run} failed ({e.__class__.__name__}: {e})')
    snap = snapshot()
    told = (f"The Reviewer's review of this game is {review.relative_to(ROOT)}: its harness and information-gap "
            f"sections list faults for you, with evidence. " if review.exists() else '')
    open_reqs = requests(kind='editor')
    if open_reqs:
        told += ('Open requests for you (agent/model/requests.jsonl; list the ids you complete in "requests_done"): '
                 + ' '.join(f"[{r['id']}] {r['text']}" for r in open_reqs[:8]) + ' ')
    told += removed_note(('harness', 'comp rules'))
    told += cases_brief()
    left = left_before(run)
    if left:
        told += ("Faults the Editors of earlier games left (a fault that shows again in this game has cost a round in "
                 "every game since - rank it by that total):\n" + '\n\n'.join(left) + '\n')
    base = cases_now()
    info.update(run_role('harness_fixer', about + told + 'Finish with the JSON line your role file asks for.', run, cfg))
    guard('harness_fixer', snap)
    fx = info.get('harness_fixer') if isinstance(info.get('harness_fixer'), dict) else {}
    if not fx:
        # the Editor ended without its final JSON line (game 36: two fixes and requests e2, e3 built, and the report
        # said "Editor: nothing" while e2 and e3 stayed open): its commits stand in, and the request ids they name
        fx = editor_from_commits(snap['head'])
        info['harness_fixer'] = fx
    ok = gate(snap['head'], snap.get('files'), base)
    if ok:
        close_done(fx.get('requests_done'), f"Editor after game {row['n']}")
        commit_paths(f"Requests the Editor completed after game {row['n']}", REQUESTS.relative_to(ROOT))
    if cfg.get('strategy') == 'model':
        # the stage strategies (the owner's master plan, 2026-11-01): the Strategist on the stages the diff reports
        # mark revise or revert, each rewrite checked by the Judge; the old principle loop (playbook) is retired
        info.update(stage_strategy_step(row, cfg, about, info.get('diffs')))
        try:                                             # the strong players' games it asked for, for its next run
            import study_queue
            if 'study_queue' not in info:
                info['study_queue'] = study_queue.work()
            git('add', '-A', 'agent/decisions', 'agent/model/playalong', 'agent/sources')
            if git('status', '--porcelain', '--', 'agent/decisions', 'agent/model/playalong', 'agent/sources'):
                git('commit', '-q', '-m', f"Strong players' games read for the Strategist after game {row['n']}")
        except Exception as e:
            say(f'study queue: {e.__class__.__name__}: {e}')
    info['check_passed'] = ok
    info['report'] = report(row, info)
    update(run, **{k: v for k, v in info.items() if k not in ('requests', 'check_passed')}, check_passed=ok,
           code_after=git('rev-parse', '--short', 'HEAD'), memory_after=memory_hash())


# the owner, 2026-11-05 (after the full steps took 75 min after game 63 and 2.5 h after game 64): "2 hours would play
# 1 game; instead it should gradually make less and less changes, and ideally play 5-10 games before a full review ...
# during the 5-10 games maybe just focus on the key problems or large errors". The full steps (stage reports, change
# review, Strategist, comp coach, meta review) run on the last game of a batch; the games before it get the quick
# check: the execution grade (code, agent/smoke.py), one Reviewer session listing only large errors, and the Editor
# only on P0 requests. Stage strategies and comp notes stay as they are through a batch, so the batch measures them.
# the owner, 2026-11-06: "Between each game, do not review. Only review after a batch of games, like 5 games, not 10
# ... During those 5 games, you only pause and edit when I give you feedback. If I do not give you any feedback after a
# game, you continue the wrong way. Play 5 games, then do the review. When doing the review, the reviewer could compare
# rounds or stages side by side across them instead of just looking at a single one." Inside a batch a game gets only
# code's packets and measures (no model session); the Editor works only the owner's requests and the live check that
# proves them; everything else (the execution grade's faults, the patch check, the comp coach, strong players' games)
# waits for the batch review.
BATCH = 5                           # games per batch review


def full_rows(rows=None):
    """The games that got the full steps (rows from before batches count as full)."""
    return [r for r in (rows if rows is not None else records())
            if 'reviewer' in r and r.get('review_mode') not in ('light', 'none')]


def batch_size(rows=None):
    return BATCH


def batch_start(cfg=None):
    """The first game of the batch rule (config "batch_start"): games before it never end a batch."""
    return int((config() if cfg is None else cfg).get('batch_start') or 1)


def full_review_due(row, rows=None):
    """Whether the review runs after game `row`: after every game that is not in the top 4, over the games since the
    last review (the owner, 2026-11-11). A top-4 game, or one whose place was not read, waits for the next review."""
    p = row.get('placement')
    return isinstance(p, int) and p > 4


def batch_rows(row, rows=None):
    """The games of the batch that `row` ends (or is in), in order."""
    rows = rows if rows is not None else records()
    last = max([r['n'] for r in full_rows(rows) if r['n'] < row['n']] + [batch_start() - 1])
    return [r for r in rows if last < r['n'] <= row['n'] and r.get('run')]


def p0(r):
    """A request worked inside a batch: the owner's feedback, and the live check that proves its fix."""
    return str(r.get('from', '')).startswith(('Owner', 'Live check'))


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def quick_after(row, cfg):
    """The steps after a game inside a batch: the packet and measures (code), one Reviewer session on large errors
    only, and the Editor only when a P0 request is open (the execution grade's, this check's, the owner's)."""
    run, n = row['run'], row['n']
    digest = REVIEWS / f'{run}-digest.md'
    REVIEWS.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, str(AGENT / 'digest.py'), str(ROOT / 'runs' / run), '--out', str(digest)], cwd=ROOT)
    sys.path.insert(0, str(AGENT))
    import metrics
    try:
        if (ROOT / 'runs' / run).exists():
            metrics.write(ROOT / 'runs' / run)
    except Exception as e:
        say(f'metrics: {run} failed ({e.__class__.__name__}: {e})')
    out = AGENT / 'model' / 'reviews' / f'{run}-quick.json'
    rel = out.relative_to(ROOT).as_posix()
    grade = next((g for g in (load_json(AGENT / 'smoke.json') or {}).get('ranked') or [] if g.get('run') == run), {})
    brief = (about_of(row, cfg)
             + f"Your job: the quick check of this game, inside a batch of {batch_size()} games whose full review comes "
               f"at the batch's end (the owner, 2026-11-05: between full reviews, only the key problems and large "
               f"errors). Code's execution grade of every action: "
               f"{json.dumps({k: grade.get(k) for k in ('faults', 'numbers')})}. List only large errors: a decision "
               f"or action that cost a fight or more, or a stage's strength - a planned or owned unit not fielded, "
               f"items not given or given to the wrong unit, a wrong sell, a level, roll or Wisp buy the plan made that "
               f"did not happen, a stall or crash, the carousel taking the wrong item, a FIRST RULE broken. No strategy "
               f"opinions and no small mistakes: those wait for the full review. For each: the rounds, what happened, "
               f"the cause (harness = the code did not do what was decided; information = the player was not shown "
               f"what it needed; strategy = the decision itself) and the evidence (log lines, call numbers, "
               f"screenshots). Write {rel}: "
             + '{"large_errors": [{"rounds": "4-1, 4-2", "what": "...", "cause": "harness|information|strategy", '
               '"evidence": "...", "case": {"round": "4-1", "call": 59, "mode": "harness|decision", "expect": "...", '
               '"check": "..."}}]}. The "case" (agent/round_cases.py, roles/reviewer.md "Round cases"): the round '
               'whose question made the decision, kept so the fix is proven on this game - mode harness when the code '
               'did not do what the answer decided, decision when the answer itself was wrong; its check must be False '
               'on the round as played (`agent/round_cases.py try --run ' + run + ' --call <n> --round <r> --check '
               '"..."`). Leave it out when no question shows the error (a misread screen, a failed drag). '
               'Finish with one JSON line: {"quick": "' + rel
             + '", "large_errors": <n>, "post": "<1-2 plain sentences in the first person>"}.')
    snap = snapshot()
    info = run_role('reviewer', brief, f'{run}-quick', cfg)
    guard('reviewer', snap)
    errors = (load_json(out) or {}).get('large_errors') or []
    fix = [e for e in errors if e.get('cause') in ('harness', 'information')]
    if fix:
        made = add_requests('editor', [f"[game {n}, P0] Large errors the quick check found (harness/information): "
                                       + ' / '.join(f"{e.get('rounds')}: {e.get('what')} ({e.get('cause')}; evidence: "
                                                    f"{str(e.get('evidence'))[:300]})" for e in fix[:6])
                                       + f". Run runs/{run}, the check {rel}. Fix the cause; its round cases (if any) "
                                         f"must pass."], n, f'Quick check of game {n}')
        cases_from([e.get('case') for e in fix if isinstance(e.get('case'), dict)], run,
                   f'quick check of game {n}', request=made[0]['id'] if made else None)
    strat = [e.get('case') for e in errors if e.get('cause') == 'strategy' and isinstance(e.get('case'), dict)]
    if strat:                                            # the Strategist's and the Judge's
        cases_from(strat, run, f'quick check of game {n}')
    git('add', '-A', 'agent/model/reviews', 'agent/reviews', 'agent/metrics', str(REQUESTS.relative_to(ROOT)))
    git('commit', '-q', '-m', f'Quick check of {run}: {len(errors)} large error(s) (committed by agent/climb.py)')
    todo = [r for r in requests(kind='editor') if p0(r) and int(r.get('tries') or 0) < EDITOR_TRIES]
    fx = editor_on_requests(todo, cfg) if todo else {}
    nxt = max([r['n'] for r in full_rows()] or [0]) + batch_size()
    say(f"quick check of game {n}: {len(errors)} large error(s), {len(fix)} to the Editor; "
        + (f"the Editor on {', '.join(r['id'] for r in todo[:6])}" if todo else 'no P0 request')
        + f"; the full steps after game {nxt}")
    update(run, reviewer=info.get('reviewer') or {'quick': rel}, review_mode='light', quick_errors=errors,
           harness_fixer=fx, code_after=git('rev-parse', '--short', 'HEAD'), memory_after=memory_hash())


def game_packets(row):
    """Code only, no model: a game's packet, measures and stage packets, for the batch review."""
    run = row['run']
    if not (ROOT / 'runs' / run).exists():
        return
    REVIEWS.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, str(AGENT / 'digest.py'), str(ROOT / 'runs' / run), '--out',
                    str(REVIEWS / f'{run}-digest.md')], cwd=ROOT)
    sys.path.insert(0, str(AGENT))
    import metrics
    import rounds as rounds_mod
    for what, f in (('metrics', lambda: metrics.write(ROOT / 'runs' / run)),
                    ('stage packets', lambda: rounds_mod.write_packets(ROOT / 'runs' / run))):
        try:
            f()
        except Exception as e:
            say(f'{what}: {run} failed ({e.__class__.__name__}: {e})')


def in_batch(row, cfg):
    """A game inside a batch: its packets (code), nothing else - the owner's feedback is what changes the play."""
    game_packets(row)
    git('add', '-A', 'agent/reviews', 'agent/metrics')
    git('commit', '-q', '-m', f"Packets of {row['run']} for the next review (committed by agent/climb.py)")
    update(row['run'], review_mode='none')
    say(f"game {row['n']}: place {row.get('placement')}, no review; the next one runs after a game that is not in "
        f"the top 4")


def batch_after(row, cfg):
    """The batch review, on the game that ends a batch (the owner, 2026-11-06: "the reviewer could compare rounds or
    stages side by side across them instead of just looking at a single one"): code builds each game's stage packets;
    one Reviewer session per stage (Job 6) reads that stage of every game of the batch side by side and grades each
    game's decisions in Job 1's format, with what the side-by-side view shows (the same round played differently, the
    same fault again); code checks each audit is complete and sends a stage back once; then the meta review (Job 5)
    of the batch starts from those comparisons, its actions go to the roles that own them, the Editor works the open
    requests (guard, gate), and the Strategist the stages the meta review names."""
    import metrics
    import rounds as rounds_mod
    from concurrent.futures import ThreadPoolExecutor
    rows = batch_rows(row)
    for r in rows:
        if r is rows[-1] or not (REVIEWS / f"{r['run']}-rounds" / 'rounds.json').exists():
            game_packets(r)
    rounds = {}
    for r in rows:
        try:
            rounds[r['run']] = json.loads((REVIEWS / f"{r['run']}-rounds" / 'rounds.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            say(f"batch review: game {r['n']} has no stage packets: left out")
    rows = [r for r in rows if r['run'] in rounds]
    if not rows:
        say('batch review: no game with stage packets')
        return
    a = f"{rows[0]['n']}-{rows[-1]['n']}"
    folder = AGENT / 'model' / 'reviews' / 'batch' / a
    folder.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (AGENT / 'model' / 'reviews' / f"{r['run']}-audit").mkdir(parents=True, exist_ok=True)
    stage_of = lambda x: x['stage'].split('-')[0]                       # noqa: E731
    stages = sorted({int(stage_of(x)) for v in rounds.values() for x in v})
    games = '; '.join(f"game {r['n']}: run {r['run']}, {r.get('comp_played') or r.get('comp')}, place "
                      f"{r.get('placement')}, player {r.get('model')} {r.get('effort', '')}" for r in rows)

    def audit_of(run, s):
        try:
            return json.loads((AGENT / 'model' / 'reviews' / f'{run}-audit' / f'stage-{s}.json').read_text(
                encoding='utf-8'))
        except (OSError, ValueError):
            return None

    def missing(s):
        out = [f"game {r['n']} ({r['run']}): {g}" for r in rows
               for g in rounds_mod.gaps(rounds[r['run']], audit_of(r['run'], s) or {}, s)
               if any(stage_of(x) == str(s) for x in rounds[r['run']])]
        if load_json(folder / f'stage-{s}.json') is None:
            out.append(f'agent/model/reviews/batch/{a}/stage-{s}.json not written')
        return out

    def job(s, gaps=None):
        have = [r for r in rows if any(stage_of(x) == str(s) for x in rounds[r['run']])]
        brief = (f"The batch: games {a} ({len(rows)} games): {games}. Your job: Job 6, stage {s} across the batch, "
                 f"side by side. The packets: "
                 + ', '.join(f"agent/reviews/{r['run']}-rounds/stage-{s}.md (game {r['n']})" for r in have)
                 + f". Write each game's audit agent/model/reviews/<run>-audit/stage-{s}.json (Job 1's format) and the "
                   f"comparison agent/model/reviews/batch/{a}/stage-{s}.json. ")
        if gaps:
            brief += ('Your audits are not complete - complete the same files (keep what is there): '
                      + '; '.join(gaps[:60]) + '. ')
        return run_role('reviewer', brief + 'Finish with the JSON line Job 6 asks for.',
                        f'batch-{a}-stage{s}' + ('-fix' if gaps else ''), cfg)

    snap = snapshot()
    with ThreadPoolExecutor(max_workers=AUDITORS) as pool:
        list(pool.map(job, stages))
    for s in stages:
        gaps = missing(s)
        if gaps:
            say(f'batch review: stage {s} leaves {len(gaps)} gap(s): {"; ".join(gaps[:4])}')
            job(s, gaps)
    guard('reviewer', snap)
    for r in rows:
        run = r['run']
        audits = [x for x in (audit_of(run, s) for s in stages) if x]
        (AGENT / 'model' / 'reviews' / f'{run}-rounds.md').write_text(rounds_mod.render(rounds[run], audits),
                                                                      encoding='utf-8')
        tally = rounds_mod.tally(rounds[run], audits)
        tally['gaps'] = [g for s in stages for g in rounds_mod.gaps(rounds[run], audit_of(run, s) or {}, s)]
        try:
            audit = metrics.attach_audit(run, tally, None)
        except Exception as e:
            say(f'metrics: the audit of {run} not attached ({e.__class__.__name__}: {e})')
            audit = tally
        update(run, reviewer={'batch': a, 'comparisons': f'agent/model/reviews/batch/{a}'}, review_mode='batch',
               audit=audit)
    commit_paths(f'Batch review of games {a}: stages side by side (committed by agent/climb.py)',
                 'agent/model/reviews', 'agent/reviews', 'agent/metrics')
    meta = meta_step(cfg, rows, extra=(f"Start from the stage comparisons across the batch, agent/model/reviews/batch/"
                                       f"{a}/stage-*.json (Job 6), and the games' audits agent/model/reviews/<run>-audit/"
                                       f"stage-*.json; the games had no review of their own. "))
    try:
        case_requests()
    except Exception as e:
        say(f'round cases: {e.__class__.__name__}: {e}')
    todo = [r for r in requests(kind='editor') if int(r.get('tries') or 0) < EDITOR_TRIES]
    fx = editor_on_requests(todo, cfg) if todo else {}
    for _ in range(EDITOR_PASSES - 1):                   # another session while the last one made cases pass
        todo = [r for r in requests(kind='editor') if int(r.get('tries') or 0) < EDITOR_TRIES]
        if not (todo and (fx.get('cases') or {}).get('flipped')):
            break
        fx = editor_on_requests(todo, cfg)
    import round_cases
    dec = [c['id'] for c in round_cases.load() if c['mode'] == 'decision']
    if dec:                                              # the decision cases asked again on the notes as they are
        round_cases.run_sub(dec, ask=True)
        commit_paths('Decision cases asked again (committed by agent/climb.py)', 'agent/round_cases')
    strat = {}
    if cfg.get('strategy') == 'model':
        strat = stage_strategy_step(row, cfg, about_of(row, cfg), None)
    fail = round_cases.failing()
    post_owner(f"Round cases after games {a}: {len(round_cases.load()) - len(fail)} of {len(round_cases.load())} "
               f"pass" + (f"; failing: {', '.join(c['id'] for c in fail[:10])}" if fail else '')
               + ' (agent/round_cases/results.json).')
    flex = [r for r in rows if r.get('comp') == 'flex']
    if flex and cfg.get('comp') == 'flex':              # the choice file, from what the batch's choices led to
        strat['choice'] = choice_plan(cfg, about=(
            "Revise it from the batch's flex games: " + '; '.join(
                f"game {r['n']} (run {r['run']}, place {r.get('placement')}, committed to {r.get('comp_played')})"
                for r in flex) + f" - their commits in runs/<run>/log.jsonl (\"commit\" and \"comp_p\" events), "
            f"their audits agent/model/reviews/<run>-audit/stage-*.json and the batch's comparisons "
            f"agent/model/reviews/batch/{a}/. Keep it, revise it or write it anew; say why."))
    say(f"batch review of games {a}: {len(stages)} stage(s) compared; meta review "
        f"{'failed' if meta.get('failed') else str(meta.get('actions')) + ' action(s)'}; "
        f"Editor {'on ' + ', '.join(r['id'] for r in todo[:4]) if todo else 'no request'}")
    update(row['run'], harness_fixer=fx, batch_meta=meta, **strat, code_after=git('rev-parse', '--short', 'HEAD'),
           memory_after=memory_hash())


def after_game(rec, cfg, n, before, use_coach=True):
    row = record_game(rec, cfg, n, before)
    try:                                                 # the execution grade: recorded, its faults wait for the batch
        import smoke
        sys.modules.setdefault('climb', sys.modules[__name__])
        smoke.after_ranked(row, rec)
    except Exception as e:
        say(f'execution grade: game {n} failed ({e.__class__.__name__}: {e})')
    if use_coach and not str(rec.get('result', '')).startswith('stopped'):
        if full_review_due(row):
            say(f"game {n}: place {row.get('placement')}, not in the top 4: the review of the games since the last one")
            batch_after(row, cfg)
        else:
            in_batch(row, cfg)
    dashboard()
    return row


PENDING_GAMES = 3                   # the latest games checked for steps that did not run
EDITOR_TRIES = 2                    # a request the Editor did not complete in this many sessions waits for a game
EDITOR_PASSES = 3                   # Editor sessions after a batch review, while each one makes round cases pass


def about_of(row, cfg):
    run = row['run']
    return (f"Game {row['n']}: run {run}, {row.get('comp') or cfg['comp']}, place {row.get('placement')}, LP change "
            f"{row.get('lp_change')}. The packet is agent/reviews/{run}-digest.md; the run folder is runs/{run}/. ")


def pending(cfg):
    """The learning work waiting before the next game, in order: a game whose steps did not run (it was never
    reviewed), a reflection due after a 5th-8th place, open requests for the Editor, strong players' games asked for
    and not yet read. -> [(what, function)]."""
    tasks = []
    rows = [r for r in records()[-PENDING_GAMES:] if r.get('run') and (ROOT / 'runs' / r['run']).exists()]
    for r in rows:
        if r.get('placement') and 'reviewer' not in r and not r.get('review_mode') and r['n'] >= batch_start(cfg):
            step = batch_after if full_review_due(r) else in_batch
            tasks.append((f"the steps after game {r['n']} (they did not run)", lambda r=r, f=step: f(r, cfg)))
    last = records()
    # inside a batch: the owner's requests only (BATCH); a game before batch_start counts as inside too
    inside = bool(last) and (last[-1].get('review_mode') in ('light', 'none') or last[-1]['n'] < batch_start(cfg))
    try:
        import meta_review
        if not tasks and not inside and meta_review.due():
            tasks.append((f"the meta review of games {meta_review.name(meta_review.batch())}",
                          lambda: meta_step(cfg)))
    except Exception as e:
        say(f'meta review: {e.__class__.__name__}: {e}')
    try:                                                 # a new patch or mid-patch update (agent/patch_watch.py)
        import patch_watch
        if not inside and patch_watch.due():
            tasks.append(("the patch check (Riot's TFT patch notes)", lambda: patch_watch.step(cfg)))
    except Exception as e:
        say(f'patch check: {e.__class__.__name__}: {e}')
    try:                                                 # a failing harness case is a request until it passes
        case_requests()
    except Exception as e:
        say(f'round cases: {e.__class__.__name__}: {e}')
    todo = [r for r in requests(kind='editor') if int(r.get('tries') or 0) < EDITOR_TRIES and (p0(r) or not inside)]
    if todo:
        tasks.append((f"the Editor on {len(todo)} open request(s): {', '.join(r['id'] for r in todo[:6])}",
                      lambda todo=todo: editor_on_requests(todo, cfg)))
    try:
        import study_queue
        if not inside and any(r.get('status') == 'open' for r in study_queue.rows()):
            tasks.append(("strong players' games asked for", lambda: study_queue.work()))
    except Exception:
        pass
    # the comps: only a comp learned from strong players' games is played (the owner, 2026-10-29); the comp to learn
    # next and the one played get their games found and their notes written when new games of them are read
    try:
        import comp_coach
        import comp_select
        ch = comp_select.choose(cfg.get('comp'))
        # flex: both comps of the set are learned like Brambleback was (strong players' games, the comp coach)
        fs = (cfg.get('flex') or {}).get('comps') if cfg.get('comp') == 'flex' else None
        learn = list(fs) if fs else [ch['focus']] if ch['focus'] else []
        for comp in learn if not inside else []:                           # one comp at a time: the one played
            d = comp_coach.due(comp)
            if d == 'find':
                tasks.append((f"strong players' games of {comp} to find", lambda comp=comp: comp_coach.find(comp)))
            elif d == 'words':
                tasks.append((f"the strong players' words in {comp}'s games (transcribed)",
                               lambda comp=comp: comp_coach.words_for(comp)))
            elif d == 'coach' and not inside:
                tasks.append((f"the comp coach on {comp}", lambda comp=comp: comp_coach_step(comp, cfg)))
    except Exception as e:
        say(f'comp tasks: {e.__class__.__name__}: {e}')
    return tasks


META_TO = {'editor': 'editor', 'comp_coach': 'coach', 'owner': 'owner'}   # the strategist's: stage_strategy_step


def post_owner(text):
    from channel import post as channel_post
    channel_post('Coordinator', text, to='Owner')


def meta_step(cfg, rows=None, extra=''):
    """The meta review (agent/meta_review.py; the owner, 2026-11-03: a review after every game, and "a meta review
    every 5-10 games, additionally"): code builds the batch's packet; one Reviewer session (Job 5) reads the games
    together, sent back once with what it left out; its actions go to the Editor, the comp coach and the owner as
    requests, the Strategist's to the next Strategist run. A session that wrote nothing is recorded as failed, so the
    loop does not retry it before the next batch. `rows`: the batch (default: the reviewed games since the last meta
    review); `extra`: more of the brief. -> info."""
    import meta_review
    rows = rows or meta_review.batch()
    a, n = meta_review.name(rows), rows[-1]['n']
    path = meta_review.packet(rows)
    brief = (f"Your job: Job 5, the meta review of games {a} ({len(rows)} games of {rows[-1].get('comp')}). The packet "
             f"is {path.relative_to(ROOT).as_posix()}. Write agent/model/reviews/meta/{a}.md and "
             f"agent/model/reviews/meta/{a}.json. " + extra)
    snap = snapshot()
    info = run_role('reviewer', brief + 'Finish with the JSON line Job 5 asks for.', f'meta-{a}', cfg)
    cost = float(info.get('reviewer_cost_usd') or 0)
    miss = meta_review.gaps(meta_review.result(rows))
    if miss and meta_review.result(rows) is not None:
        say(f'meta review: {len(miss)} gap(s): {"; ".join(miss[:4])}')
        r2 = run_role('reviewer', brief + 'Your meta review is not complete - complete the same files (keep what is '
                      'there): ' + '; '.join(miss) + '. Finish with the JSON line Job 5 asks for.', f'meta-{a}-fix', cfg)
        cost += float(r2.get('reviewer_cost_usd') or 0)
    guard('reviewer', snap)
    m = meta_review.result(rows)
    f = meta_review.META / f'{a}.json'
    if not isinstance(m, dict):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({'games': [r['n'] for r in rows], 'failed': 'no meta review written'}) + '\n',
                     encoding='utf-8')
        say(f'meta review: games {a}: no meta review written (recorded as failed)')
        commit_paths(f'Meta review of games {a} failed (committed by agent/climb.py)', 'agent/model/reviews/meta',
                     'agent/reviews')
        return {'failed': True, 'cost_usd': cost}
    m['games'] = [r['n'] for r in rows]
    f.write_text(json.dumps(m, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    made = []
    for x in m.get('actions') or []:
        kind = META_TO.get(x.get('to')) if isinstance(x, dict) else None
        new = []
        if kind:
            new = add_requests(kind, [f"{x.get('text')} (evidence: {x.get('evidence') or '-'})"], n,
                               f'Reviewer (meta review of games {a})')
            made += new
        if isinstance(x, dict) and isinstance(x.get('case'), dict) and x['case'].get('run'):
            try:
                cases_from([x['case']], x['case']['run'], f'meta review of games {a}',
                           request=new[0]['id'] if new and kind == 'editor' else None)
            except Exception as e:
                say(f'round cases: {e.__class__.__name__}: {e}')
    acts = [x for x in m.get('actions') or [] if isinstance(x, dict)]
    text = (f"Meta review of games {a}: {len(m.get('recurring') or [])} recurring problem(s), "
            f"{len(acts)} action(s)" + (f" - requests {', '.join(r['id'] for r in made)}" if made else '')
            + (f"; {sum(x.get('to') == 'strategist' for x in acts)} for the next Strategist run"
               if any(x.get('to') == 'strategist' for x in acts) else '')
            + f". agent/model/reviews/meta/{a}.md")
    post_owner(text)
    say(text)
    commit_paths(f'Meta review of games {a} (committed by agent/climb.py)', 'agent/model/reviews/meta', 'agent/reviews',
                 REQUESTS.relative_to(ROOT))
    return {'actions': len(acts), 'requests': [r['id'] for r in made], 'cost_usd': cost}


def comp_coach_step(comp, cfg):
    """The comp coach on one comp: its material built, the session (agent/roles/comp_coach.md), the guard, its proposal
    checked and written as the player's comp notes; the notes' rules go to the Editor, to make the harness follow them
    (the owner, 2026-10-29: "then modify the harness or the player or the prompts to make it follow"). -> what apply
    kept and dropped."""
    import comp_coach
    path, used = comp_coach.material(comp)
    ours = [r['run'] for r in records() if (r.get('comp_played') or r.get('comp')) == comp and r.get('run')][-8:]
    reviews = [f'agent/model/reviews/{r}.md' for r in ours if (AGENT / 'model' / 'reviews' / f'{r}.md').exists()]
    brief = (f"The comp: {comp}. The material: {path.relative_to(ROOT).as_posix()} ({len(used)} strong players' "
             f"games: {', '.join(used) or 'none'}). Our games of it, reviewed round by round: "
             f"{', '.join(reviews) or 'none yet'}. Its current notes: agent/memory/comps/{comp_coach.slug(comp)}.md. "
             f"Write agent/comps/{comp_coach.slug(comp)}/proposed.json and finish with the JSON line your role file "
             f"asks for.")
    # requests to the comp coach (the Strategist's "comp_coach"): a change to the comp notes went to the Editor, who
    # may not write them, and was left five games running (e31, levels tied to rounds)
    brief += ' ' + removed_note(('comp notes', 'comp rules'))
    asked = [r for r in requests(kind='coach')]
    if asked:
        brief += (' Open requests for you (list the ids you complete in "requests_done"): '
                  + ' '.join(f"[{r['id']}] {r['text']}" for r in asked[:8]))
    snap = snapshot()
    info = run_role('comp_coach', brief, f'comp-{comp_coach.slug(comp)}-{datetime.now():%Y%m%d-%H%M}', cfg)
    guard('comp_coach', snap)
    done = (info.get('comp_coach') or {}).get('requests_done') if isinstance(info.get('comp_coach'), dict) else None
    if done:
        close_requests(set(done), f'Comp coach on {comp}')
    res = comp_coach.apply(comp)
    know = json.loads((comp_coach.COMPS / comp_coach.slug(comp) / 'knowledge.json').read_text(encoding='utf-8'))
    rules = [p['text'] for a in know.get('areas') or [] for p in a.get('points') or [] if p.get('kind') == 'rule']
    if rules:
        # new code rules are new behaviour: the owner's yes first (`climb.py --approve <id>` hands it to the Editor;
        # the owner's master plan, 2026-11-01) - game 52's pre-game Editor was about to build e81 without it
        head = f"Comp {comp}: the comp coach's rules"
        old = {r['id'] for r in requests(kind='owner') if str(r.get('text', '')).startswith(head)}
        new = add_requests('owner', [f"{head} (agent/memory/comps/{comp_coach.slug(comp)}.md). "
                                     "Make the harness let the player follow each, and make the executor keep the "
                                     "mechanical ones (who is fielded, where items go, when rolls run) when a plan "
                                     "breaks them; list the ones that need nothing: " + ' | '.join(rules[:10])],
                           None, 'Comp coach')
        if new:                      # one open request per comp: the latest rules replace the ones still waiting
            close_requests(old, f"replaced by {new[0]['id']} (the comp coach's later rules)")
    say(f"comp coach: {comp}: {res['kept']} point(s) kept, {len(res['dropped'])} dropped")
    commit_paths(f"Comp notes for {comp} from strong players' games (the comp coach)", 'agent/comps',
                 'agent/memory/comps', REQUESTS.relative_to(ROOT))
    return res


FLEX_COMMIT_P = 0.7                  # the P at which the controller commits a set comp (agreed with the owner, 11-06)
CHOICE = AGENT / 'comps' / 'flex' / 'choice.json'   # the Strategist's choice file: every version, the current last
CHOICE_MAX = 2500                    # its characters: it goes with every planning state until the commit


def choice_current():
    """The choice file's version in use ({'v', 'text', 'why', 'by', 'when'}), or None."""
    try:
        rows = json.loads(CHOICE.read_text(encoding='utf-8')).get('versions') or []
    except (OSError, ValueError):
        rows = []
    return next((v for v in reversed(rows) if v.get('status') == 'current'), None)


def choice_plan(cfg, about=''):
    """The Strategist's choice file (the owner, 2026-11-07: "it should also get another file written by the strategist
    on how to choose and when to choose which comp"): how and when the player picks between the flex set's comps - what
    in the shop, the items, the augments and the 2-stars points to each, when to commit, when a switch is worth it. The
    player reads it in its preparation and in every planning state until a commit (coach.flex_choice); written once
    before the first flex games, then revised after the batch reviews (about: the batch's games). -> info for the
    record."""
    fs = cfg.get('flex') if isinstance(cfg.get('flex'), dict) else {}
    comps = list(fs.get('comps') or [])
    if cfg.get('comp') != 'flex' or len(comps) < 2:
        return {'choice': 'no flex set'}
    import stages
    from tft_player.meta import META
    stamp = f'{datetime.now():%Y%m%d-%H%M}'
    out = AGENT / 'comps' / 'flex' / 'candidates' / f'{stamp}-choice.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    cur = choice_current()
    pages = ', '.join(f"data/reference/metatft/comps/{META.by_name(c).id}.md" for c in comps if META.by_name(c))
    brief = (
        "This session is not one stage of one comp: it writes the choice file of the climb's flex games (the owner, "
        "2026-11-07: \"another file written by the strategist on how to choose and when to choose which comp\"). The "
        f"player now plays flex between {' and '.join(comps)}: during the game it figures out which to play. Before "
        "a commit it has both comps' MetaTFT pages, both comps' notes with their rules and the owner's feedback on each, "
        "the rules for every comp, and this file; in every reply it gives its probability (P) that each comp is this "
        f"game's final board. The controller commits the comp whose P reaches {fs.get('commit_p')}, and at "
        f"{fs.get('commit_by')} the comp with the highest P if none is committed by then; the player may commit earlier "
        "itself and may switch to the other comp after the commit. From the commit on it plays that comp's own stage "
        "strategies, notes and rules - the framework of a fixed-comp game. Until the commit auto_buy collects both "
        "comps' 1-4 cost units.\n\n"
        + (f"The current version (v{cur['v']}) is in {CHOICE.relative_to(ROOT).as_posix()}. {about}\n\n" if cur else
           f"There is no version yet: write the first. {about}\n\n") +
        f"Write how and when to choose, at most {CHOICE_MAX} characters, in plain words the player can act on in a "
        "30-second planning phase: what in the shop, the items, the augments, the 2-stars and the emblems points to "
        "each comp, how to play while the choice is open (only what both comps share - economy, levelling, what to "
        "keep or sell; no comp's own play: that comes with its strategies at the commit), when the choice is clear "
        "enough to commit, and when a switch after the commit is worth it. The owner's rules come first "
        "(agent/memory/general.md, the FIRST RULES in agent/memory/comps/*.md, the owner's feedback in "
        "agent/memory/feedback/): the file never says otherwise.\n\nRead: both comps' MetaTFT pages (" + pages + ", and "
        "data/metatft_shots/<comp>/brief.txt where there is one), their notes (agent/memory/comps/), their stage "
        "strategies (agent/comps/<comp>/stages.json where there are any), strong players' games "
        "(agent/comps/<comp>/material.md where there is one), our games' reviews (agent/model/reviews/) and "
        "agent/games.jsonl. Never read the games kept for the Judge (agent/comps/<comp>/heldout.json).\n\nWrite "
        f"{out.relative_to(ROOT).as_posix()}: {{\"text\": \"<the choice file>\", \"why\": \"<what it rests on, with the "
        "evidence>\", \"sources\": [{\"game\": \"<run or vod game>\", \"round\": \"3-5\"}]}. A source must be a game we "
        "hold and a round it had.\n\nRules: never edit the stage strategies, the comp notes, the memory, code, roles or "
        "the config; never run the Judge; never start a game; never push. Final answer, one JSON line: {\"file\": \"<the "
        "file>\", \"post\": \"<to the team channel, 1-3 short plain sentences in the first person>\"}.")
    snap = snapshot()
    info = run_role('strategist', brief, f'flex-choice-{stamp}', cfg)
    guard('strategist', snap)
    try:
        cand = json.loads(out.read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        say(f'flex: the Strategist wrote no choice file ({e.__class__.__name__})')
        return {'choice': 'nothing written', 'strategist': info.get('strategist')}
    text = str(cand.get('text') or '').strip()
    from comp_coach import check_source
    bad = [s for s in cand.get('sources') or [] if not (isinstance(s, dict) and check_source(s))]
    if not text or len(text) > CHOICE_MAX or bad:
        why = 'no text' if not text else f'{len(text)} characters' if len(text) > CHOICE_MAX else f'sources {bad[:2]}'
        say(f'flex: the choice file is not used ({why})')
        return {'choice': f'not used: {why}'}
    try:
        data = json.loads(CHOICE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        data = {'versions': []}
    for v in data['versions']:
        if v.get('status') == 'current':
            v['status'] = 'replaced'
    ver = {'v': len(data['versions']) + 1, 'text': text, 'why': cand.get('why'), 'sources': cand.get('sources') or [],
           'by': f'Strategist {stamp}', 'when': datetime.now().strftime('%Y-%m-%d %H:%M'), 'status': 'current'}
    data['versions'].append(ver)
    CHOICE.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    commit_paths(f"The flex choice file, version {ver['v']}, written by the Strategist", 'agent/comps/flex')
    say(f"flex: choice file version {ver['v']} in use ({len(text)} characters)")
    return {'choice': ver['v'], 'strategist': info.get('strategist')}


def choose_comp(did_work):
    """The comp the next game plays: the focus (agent/comp_select.py - the comp played stays it while MetaTFT rates it
    S or A), written to agent/config.json when it changes, and played only once learned. -> the comp, or None (no
    game: its learning is not done). "flex" (the owner, 2026-11-06/07): the game flexes between the comps of
    agent/config.json "flex" and figures out in the game which to play."""
    import comp_select
    say(f'MetaTFT: {comp_select.refresh()}')
    cfg = config()
    if cfg.get('comp') == 'flex':                    # the set (agent/config.json "flex"): chosen in the game
        return 'flex' if len((cfg.get('flex') or {}).get('comps') or []) >= 2 else None
    ch = comp_select.choose(cfg.get('comp'))
    if ch['focus'] and ch['focus'] != cfg.get('comp'):
        say(f"framework: the comp changes from {cfg.get('comp')} to {ch['focus']} (no longer S or A, or no strong "
            f"players' games of it to learn from)")
        cfg['comp'] = ch['focus']
        (AGENT / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        commit_paths(f"The climb's comp is {ch['focus']}", 'agent/config.json')
    if not ch['play']:
        say(f"framework: {ch['focus'] or 'no comp'} is not learned from strong players' games yet: no game")
        return None
    return ch['play']


def editor_on_requests(todo, cfg):
    """The Editor between games on the open requests, the costliest first: its guard, the gate, the requests it
    completed closed; each one tried is counted (a request it cannot complete stops being retried)."""
    sys.path.insert(0, str(AGENT))
    import round_cases
    res, cost = round_cases.results(), {}
    for c in round_cases.load():                         # a request whose round case fails, costliest first
        if c.get('request') and (res.get(c['id']) or {}).get('status') != 'pass':
            cost[c['request']] = max(cost.get(c['request'], 0), int(c.get('cost') or 1))
    todo = sorted(todo, key=lambda r: -cost.get(r['id'], 0))
    ids = [r['id'] for r in todo[:4]]
    brief = ("No game was just played: the framework is working the open requests, the costliest first. "
             + ' '.join(f"[{r['id']}] {r['text']}" for r in todo[:4])
             + " Each is objective state data or a harness fault, no advice. Read the code the request touches and its "
               "callers first. The check must pass. List the ids you complete in \"requests_done\". Finish with the "
               "JSON line your role file asks for. "
             + cases_brief() + removed_note(('harness', 'comp rules')))
    rows = requests(None)
    for r in rows:
        if r['id'] in ids:
            r['tries'] = int(r.get('tries') or 0) + 1
    REQUESTS.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    snap = snapshot()
    base = cases_now()
    info = run_role('harness_fixer', brief, f"requests-{datetime.now():%Y%m%d-%H%M}", cfg)
    guard('harness_fixer', snap)
    fx = info.get('harness_fixer') if isinstance(info.get('harness_fixer'), dict) else {}
    if not fx:
        fx = editor_from_commits(snap['head'])
    ok = gate(snap['head'], snap.get('files'), base)
    if ok:
        fx['closed'] = close_done(fx.get('requests_done'), f"the Editor between games ({datetime.now():%m-%d %H:%M})")
    fx['cases'] = dict(LAST_GATE)
    commit_paths('Requests the Editor worked between games', REQUESTS.relative_to(ROOT))
    return fx


def work_pending(cfg):
    """Every learning task waiting, one at a time, stopping at agent/STOP. -> how many ran."""
    done = 0
    for _ in range(12):                                  # a task that keeps coming back is not run forever
        if (AGENT / 'STOP').exists():
            break
        tasks = pending(cfg)
        if not tasks:
            break
        what, fn = tasks[0]
        say(f'framework: {what}')
        try:
            fn()
        except Exception as e:                           # one task failing stops neither the others nor the games
            say(f'framework: {what} failed ({e.__class__.__name__}: {e})')
            if what.startswith('the reflection'):
                row = records()[-1]
                update(row['run'], reflection={'place': row.get('placement'), 'failed': str(e)[:300]})
        done += 1
        dashboard()
    return done


# the loop's own code: committed changes to it load between games (a fresh loop runs it; this one waits for it and
# stops) - edits not yet committed never load
SELF = ('agent/climb.py', 'agent/smoke.py', 'agent/patch_watch.py', 'agent/backends.py', 'agent/meta_review.py',
        'agent/rounds.py', 'agent/expert.py')
SELF_AT_START = None


def self_version():
    return git('log', '-1', '--format=%H', '--', *SELF)


def self_changed():
    """The loop's own files have a newer commit than when it started, and none of them has an uncommitted edit."""
    return (SELF_AT_START is not None and self_version() != SELF_AT_START
            and not git('status', '--porcelain', '--', *SELF))


def main():
    global SELF_AT_START
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=99)
    ap.add_argument('--until', help='HH:MM: no game starts after this time')
    ap.add_argument('--no-coach', action='store_true')
    ap.add_argument('--after', help='a run folder played by hand: record, review and gate it, then stop')
    ap.add_argument('--rank-before', help='JSON of the rank before that game (with --after)')
    ap.add_argument('--roles', help='a recorded game (runs/<id>): the steps after a game, on it')
    ap.add_argument('--approve', help="the owner's yes to a request, e.g. o3: the Editor builds it after the next game")
    ap.add_argument('--note', default='', help='with --approve: the owner\'s words')
    ap.add_argument('--changes', help='a run (runs/<id>): its change review only (agent/changes.py)')
    ap.add_argument('--since', help='with --changes: the run or commit to list changes from (default: the game before)')
    a = ap.parse_args()
    if a.approve:
        print(approve(a.approve, a.note))
        return
    if a.roles:
        roles_on(Path(a.roles).name, config())
        return
    if a.changes:
        run = Path(a.changes).name
        row = next((r for r in records() if r.get('run') == run), {})
        about = (f"Game {row.get('n', '?')}: run {run}, {row.get('comp') or config()['comp']}, place "
                 f"{row.get('placement', '?')}. The run folder is runs/{run}/. ")
        print(json.dumps(change_review(run, about, config(), a.since), ensure_ascii=False))
        return
    if a.after:
        run_dir = (ROOT / a.after) if not Path(a.after).is_absolute() else Path(a.after)
        res = next((e for e in reversed(_events(run_dir)) if e.get('ev') == 'game_result'), {})
        rec = {**res, 'run_dir': str(run_dir)}
        after_game(rec, config(), len(records()) + 1, json.loads(a.rank_before) if a.rank_before else None,
                   not a.no_coach)
        return
    deadline = None
    if a.until:
        from datetime import timedelta
        hh, mm = (int(x) for x in a.until.split(':'))
        deadline = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline <= datetime.now():
            deadline += timedelta(days=1)            # 09:10 asked at 22:00 means tomorrow
        say(f'no game starts after {deadline:%Y-%m-%d %H:%M}')
    played, crashes = 0, 0
    say('framework: running')
    SELF_AT_START = self_version()
    while True:
        if (AGENT / 'STOP').exists():
            say('agent/STOP exists: stopping')
            break
        if self_changed():
            say('framework: its own code has a new commit: the loop restarts on it')
            sys.exit(subprocess.run([sys.executable, *sys.argv], cwd=ROOT).returncode)
        if not wait_codex():
            say('agent/STOP exists: stopping')
            break
        # the learning that waits comes first, a game after it (the owner, 2026-10-29: the whole framework, not only
        # the games)
        did = work_pending(config()) if not a.no_coach else 0
        if (AGENT / 'STOP').exists():
            say('agent/STOP exists: stopping')
            break
        # only a comp learned from strong players' games is played (the owner, 2026-10-29)
        try:
            comp = choose_comp(did) if not a.no_coach else config().get('comp')
        except Exception as e:
            say(f'framework: the comp choice failed ({e.__class__.__name__}: {e}): no game')
            comp = None
        if comp is None:
            if did:
                continue                                 # the learning moved on: the next cycle
            say('framework: nothing more to learn now and no comp learned: stopping')
            break
        if played >= a.games:
            say(f'{played} game(s) played: stopping')
            break
        if deadline and datetime.now() >= deadline:
            say(f'past {a.until}: no new game')
            break
        # no Normal test game before a ranked one (the owner, 2026-11-06: "why do you start a normal game to test? you
        # should never do this" - a day went to checks, not games): changes are proven offline by the gate, and each
        # ranked game is graded from its log after it (smoke.after_ranked)
        import smoke
        sys.modules.setdefault('climb', sys.modules[__name__])
        if deadline and datetime.now() >= deadline:
            say(f'past {a.until}: no new game')
            break
        wait_idle()
        if deadline and datetime.now() >= deadline:
            say(f'past {a.until}: no new game')
            break
        if not mouse_ready(deadline):
            say('the mouse is still refused: no new game')
            break
        cfg = config()
        notes_current(comp)                              # the owner's rules as the player reads them, this game
        n = len(records()) + 1
        before = rank()
        t_start = time.time()
        rec = play(cfg, n)
        if rec is None and smoke.refused(rec, ROOT / 'runs' / f'climb-game-{n:02d}.out'):
            # the client refused the lobby though its eligibility said open (11-05: INVALID_REQUEST_FOR_PRODUCT):
            # Riot's side, not a crash - queue again
            say('the client refused the lobby: queueing again in 2 min')
            time.sleep(120)
            continue
        if rec and not rec.get('run_dir') and 'could not enter a match' in str(rec.get('result')):
            # no match found in the queue's time (06:00-06:30 on 10-23: 30 min in Bronze III): not a crash - queue again
            say('no match found in the queue: queueing again in 2 min')
            time.sleep(120)
            continue
        if rec and not rec.get('run_dir'):
            say('no game was played:', rec.get('result'))
            break
        played += 1
        if rec is None:
            # the game process ended without its record (a crash, or no match): review the run folder it left,
            # if any - the coach may find and fix the cause; two in a row stop the loop
            crashes += 1
            runs = sorted((d for d in (ROOT / 'runs').iterdir() if d.is_dir() and d.stat().st_mtime > t_start
                           and (d / 'log.jsonl').exists()), key=lambda d: d.stat().st_mtime)
            say(f'no game record (crash {crashes}): ' + (f'reviewing {runs[-1].name}' if runs else 'no run folder'))
            if crashes >= 2 or not runs:
                say('stopping: the game did not start or crashed twice')
                break
            rec = {'run_dir': str(runs[-1]), 'result': 'crashed (no record)'}
        else:
            crashes = 0
        wait_codex()                                     # a game that used Codex up is reviewed on Codex too
        after_game(rec, cfg, n, before, not a.no_coach)
        if str(rec.get('result', '')).startswith('stopped'):
            say('the game was stopped:', rec.get('result'))
            break


if __name__ == '__main__':
    main()
