"""Ask the player of a recorded game about a stage (the owner's structure v3: the Reviewer "may question the Player's
stage conversations ... but treats the answers as hypotheses: the log is the evidence").

    python agent/ask_player.py runs/<id> 3-5 "Why did you hold level 6 here with 45 gold?"

The question goes to a copy of the conversation that played that round (the last call naming the round, else the
stage's last call), resumed with the game's own system prompt, model, effort and reference-file tools; the game's transcript is
not changed and the copy is deleted after. One answer, printed.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import replay_decisions as rd  # noqa: E402


def stage_call(game, rnd):
    """The stage's last call that names the round (e.g. '3-5'), else the stage's last call - only that stage's own
    conversation: a later stage's messages name earlier rounds too (game 35: '3-5' found a stage-4 call first)."""
    calls = sorted(int(p.name[:4]) for p in game.calls.glob('*-result.json') if p.name[:4].isdigit())
    st = rnd.split('-')[0]
    same = [n for n in calls if (game.calls / f'{n:04d}-prompt.txt').exists() and game.stage(n) == st]
    named = [n for n in same if re.search(rf'\b{re.escape(rnd)}\b', game.prompt(n))]
    return (named or same or [None])[-1]


class Calls:
    """A game's saved calls (runs/<id>/calls) without its transcript: what stage_call needs for any backend."""
    def __init__(self, run):
        self.calls = Path(run) / 'calls'
        self._stage = {}

    prompt = rd.Game.prompt
    stage = rd.Game.stage

    def session(self, n):
        f = self.calls / f'{n:04d}-result.json'
        return json.loads(f.read_text(encoding='utf-8')).get('session_id') if f.exists() else None


def codex_rollout(thread_id):
    """Codex's own record of a thread (sessions/YYYY/MM/DD/rollout-*-<thread id>.jsonl), or None."""
    home = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex') / 'sessions'
    found = sorted(home.rglob(f'rollout-*-{thread_id}.jsonl'), key=lambda p: p.stat().st_mtime) if thread_id else []
    return found[-1] if found else None


def ask_codex(game, n, sid, text, timeout=300):
    """A Codex game (games 63 and 64 asked a Claude transcript that does not exist and got no answer): the question
    goes to a copy of call n's thread cut after that call's own turn, so the player answers from what it had then,
    resumed with the game's instructions, model, effort and folder; the copy is deleted after."""
    from tft_player.coach import codex_cli, codex_sandbox, codex_service_tier
    src = codex_rollout(sid)
    if src is None:
        return f'the Codex session of call {n} ({sid}) is not on this PC'
    rows = [json.loads(l) for l in src.read_text(encoding='utf-8', errors='replace').splitlines() if l.strip()]
    turn = sum(1 for m in sorted(int(p.name[:4]) for p in game.calls.glob('*-result.json') if p.name[:4].isdigit())
               if m <= n and game.session(m) == sid)          # call n is this thread's turn-th turn
    done = 0
    for i, r in enumerate(rows):
        if r.get('type') == 'event_msg' and (r.get('payload') or {}).get('type') == 'task_complete':
            done += 1
            if done == turn:
                rows = rows[:i + 1]
                break
    meta = rows[0].get('payload') or {}
    ctx = next((r['payload'] for r in reversed(rows) if r.get('type') == 'turn_context'), {})
    new = str(uuid.uuid4())
    rows[0] = dict(rows[0], payload=dict(meta, id=new, session_id=new))
    day = datetime.now()
    copy = src.parent.parent.parent.parent / f'{day:%Y}' / f'{day:%m}' / f'{day:%d}' / \
        f'rollout-{day:%Y-%m-%dT%H-%M-%S}-{new}.jsonl'
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    instr = Path(tempfile.gettempdir()) / f'tft-ask-player-{new}.md'
    instr.write_text((meta.get('base_instructions') or {}).get('text') or '', encoding='utf-8')
    cmd = [codex_cli(), 'exec', '--json', '--ignore-user-config', '--ignore-rules', '--skip-git-repo-check',
           '-s', 'read-only', '-c', 'approval_policy="never"', '-m', ctx.get('model') or 'gpt-6.1-sol',
           '-c', f'model_reasoning_effort="{ctx.get("effort") or "medium"}"',
           '-c', f'model_instructions_file="{instr.as_posix()}"']
    if codex_sandbox():
        cmd += ['-c', f'windows.sandbox="{codex_sandbox()}"']
    if codex_service_tier():
        cmd += ['-c', f'service_tier="{codex_service_tier()}"']
    cmd += ['resume', new, '--', '-']
    said, err = [], None
    try:
        p = subprocess.run(cmd, input=text.encode('utf-8'), capture_output=True, timeout=timeout,
                           cwd=meta.get('cwd') or None, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for line in p.stdout.decode('utf-8', 'replace').splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            item = ev.get('item') or {}
            if ev.get('type') == 'item.completed' and item.get('type') == 'agent_message':
                said.append(item.get('text') or '')
            elif ev.get('type') in ('turn.failed', 'error'):
                err = (ev.get('error') or {}).get('message') or ev.get('message') or err
    except subprocess.TimeoutExpired:
        err = f'no answer in {timeout} s'
    finally:
        for f in (copy, instr):
            f.unlink(missing_ok=True)
    answer = '\n'.join(s for s in said if s.strip()).strip()
    return answer if answer else f'no answer ({(err or "Codex gave none")[-200:]})'


def ask(run, rnd, question, timeout=300):
    calls = Calls(run)
    n = stage_call(calls, rnd)
    if n is None:
        return f'no call of round {rnd} in {run}'
    r = json.loads((calls.calls / f'{n:04d}-result.json').read_text(encoding='utf-8'))
    text = (f'A question from the review of this game, after it ended (round {rnd}). Answer from what you saw and '
            f'decided then; say where you are unsure. No plan is needed.\n\n{question}')
    if r.get('backend') == 'codex':
        return ask_codex(calls, n, r.get('session_id'), text, timeout)
    if r.get('backend') not in (None, 'claude'):
        return f"call {n} played on {r.get('backend')}, which this tool cannot ask again"
    game = rd.Game(run)
    sid = r.get('session_id')
    src = game.proj / f'{sid}.jsonl'
    if not src.exists():
        return f'the transcript of call {n} (session {sid}) is not on this PC'
    entries = [json.loads(l) for l in src.read_text(encoding='utf-8').splitlines() if l.strip()]
    entries = [e for e in entries if e.get('type') in rd.KEEP]
    result, tools, secs, err = game.ask(entries, [{'type': 'text', 'text': text}], timeout=timeout)
    if not result:
        return f'no answer ({err.strip()[-200:]})'
    return result.get('result') or ''


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('round')
    ap.add_argument('question')
    a = ap.parse_args()
    print(ask(a.run, a.round, a.question))


if __name__ == '__main__':
    main()
