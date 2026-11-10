"""Which backend the climb's model parts run on: Codex on the ChatGPT login, or - once its usage is gone - the
fallback, Claude Opus 5.5 medium on the Claude login (the owner, 2026-11-03: "there is 11% usage left on the chatgpt,
if it is gone, could it be switched to the opus 5.5 medium?").

A Codex run that fails with a usage or rate limit writes agent/model/codex-limit.json. While it is there, every new
game, role session and Judge run takes agent/config.json "fallback_model" / "fallback_effort" instead of a "codex:"
model; a short Codex probe (at most every PROBE_EVERY seconds) removes the mark once Codex answers again, so the
climb goes back to the models the owner chose.
"""
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

AGENT = Path(__file__).resolve().parent
ROOT = AGENT.parent
MARK = AGENT / 'model' / 'codex-limit.json'
FALLBACK = ('claude:claude-opus-5-5', 'medium')
PROBE_EVERY = 600                      # s between probes while the mark is there
LIMIT_RX = re.compile(r'usage limit|rate limit|quota|limit (?:has been |was )?reached|too many requests|\b429\b|'
                      r'exceeded your|out of (?:credits|usage)|try again (?:in|at|later)', re.I)


def limit_errors(text):
    """The usage/rate-limit messages among a Codex --json run's error events (never the model's own words)."""
    out = []
    for line in (text or '').splitlines():
        if not line.startswith('{'):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get('type') in ('error', 'turn.failed'):
            msg = ev.get('message') or (ev.get('error') or {}).get('message') or json.dumps(ev)[:300]
            if LIMIT_RX.search(str(msg)):
                out.append(str(msg)[:300])
    return out


def mark(message, where):
    MARK.parent.mkdir(parents=True, exist_ok=True)
    MARK.write_text(json.dumps({'when': datetime.now().isoformat(timespec='seconds'), 'message': message,
                                'where': where, 'checked': time.time()}, ensure_ascii=False), encoding='utf-8')


def probe():
    """A one-line Codex question -> True (it answered), False (a usage/rate limit), None (anything else)."""
    from tft_player.coach import codex_cli, codex_sandbox, codex_service_tier
    cmd = [codex_cli(), 'exec', '--json', '--skip-git-repo-check', '-s', 'read-only', '-c', 'approval_policy="never"',
           '-m', 'gpt-6.1-sol', '-c', 'model_reasoning_effort="low"']
    if codex_sandbox():
        cmd += ['-c', f'windows.sandbox="{codex_sandbox()}"']
    if codex_service_tier():
        cmd += ['-c', f'service_tier="{codex_service_tier()}"']
    try:
        p = subprocess.run(cmd + ['-'], input='Reply with OK.', capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=120, cwd=ROOT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if limit_errors(p.stdout + '\n' + p.stderr):
        return False
    return True if '"agent_message"' in p.stdout else None


def codex_limited():
    """True while the mark is there and Codex has not answered a probe since."""
    if not MARK.exists():
        return False
    try:
        m = json.loads(MARK.read_text(encoding='utf-8'))
    except ValueError:
        m = {}
    if time.time() - float(m.get('checked') or 0) < PROBE_EVERY:
        return True
    ok = probe()
    if ok:
        MARK.unlink(missing_ok=True)
        return False
    m['checked'] = time.time()
    MARK.write_text(json.dumps(m, ensure_ascii=False), encoding='utf-8')
    return True


def choose(model, effort, cfg=None):
    """(model, effort) for a model part: the fallback in place of a "codex:" model while Codex is limited."""
    if not str(model).startswith('codex:') or not codex_limited():
        return model, effort
    cfg = cfg or {}
    return cfg.get('fallback_model') or FALLBACK[0], cfg.get('fallback_effort') or FALLBACK[1]


def limit_in_run(run_dir):
    """A game's model calls that failed on a usage/rate limit (runs/<run>/log.jsonl error fields) -> [message]."""
    out = []
    log = Path(run_dir) / 'log.jsonl'
    if not log.exists():
        return out
    for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
        if '"error"' not in line and 'error' not in line.lower():
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        texts = [ev.get('error')] + ([ev.get('msg')] if ev.get('ev') == 'coach' and 'error' in str(ev.get('msg')).lower()
                                     else [])
        out += [str(t)[:300] for t in texts if t and LIMIT_RX.search(str(t))]
    return out


def cli_model(model):
    """The Claude CLI's --model value for a 'claude:<id>' name."""
    return str(model).split(':', 1)[1] if str(model).startswith('claude:') else model
