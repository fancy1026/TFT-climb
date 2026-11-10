"""The coach: an LLM that sets strategic direction, never in the action path.

Runs the Cursor (or Codex) CLI in a background thread. A call takes ~20s
(CLI start + model), so it is asked early - when the augment picker opens, or
during combat - and its answer only updates `brain.plan`. If it is late, the
local default (meta.py) has already acted.

Local code builds the menu - candidate comps and augment options with MetaTFT
evidence - and a default choice; the coach compares the menu against the game.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

from tft_player.catalog import CATALOG
from tft_player.meta import META

CATALOG_SET = 18

def _cli():
    """Backend for the model (TFT_COACH_BACKEND, set by use_model). 'claude': the Claude Code CLI, which keeps
    two conversations per game and signs in with the user's Claude login or ANTHROPIC_API_KEY. 'openrouter':
    any OpenRouter model, also two conversations per game. 'codex': the Codex CLI on the user's ChatGPT login, two
    conversations per game too. 'claude-api': single Anthropic API calls with no conversation (not for full games).
    The Cursor CLI only answers the augment coach, not round plans."""
    want = os.environ.get('TFT_COACH_BACKEND', 'claude').lower()
    if want == 'claude':
        cli = claude_cli()
        return ('claude-cli', cli) if cli else (None, None)
    if want == 'claude-api' and (os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_AUTH_TOKEN')):
        return 'claude', 'anthropic-sdk'
    if want == 'cursor':
        exe = cursor_cli()
        return ('cursor', exe) if exe else (None, None)
    if want == 'codex':
        exe = codex_cli()
        return ('codex', exe) if exe else (None, None)
    if want == 'openrouter' and api_key():
        return 'openrouter', OR_MODEL
    if want == 'grok':
        exe = grok_cli()
        return ('grok', exe) if exe else (None, None)
    if want == 'agy':
        exe = agy_cli()
        return ('agy', exe) if exe else (None, None)
    return None, None


OR_MODEL = os.environ.get('TFT_OR_MODEL', 'openai/gpt-6-sol')
OR_EFFORT = os.environ.get('TFT_OR_EFFORT', 'low')
MODEL_SPECS = ('claude[:<model id>] (Claude login), claude-key[:<model id>] (Anthropic API key), codex[:<model id>] '
               '(ChatGPT login, Codex CLI), openai:<model id> (OpenAI API key) or openrouter:<model id> (OpenRouter '
               'API key)')
# the chat-completions loop (_ORSession) talks to OpenRouter or to OpenAI directly: same messages, tools and stream
API_HOSTS = {'openrouter': ('https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY', 'OpenRouter'),
             'openai': ('https://api.openai.com/v1', 'OPENAI_API_KEY', 'OpenAI')}


def api_host():
    return os.environ.get('TFT_API_HOST', 'openrouter')


def api_key():
    return user_env(API_HOSTS[api_host()][1])


def use_model(spec, effort=None):
    """The game's model, for this process and the processes it starts. spec: 'claude' or 'claude:<model id>'
    (Claude Code CLI) or 'openrouter:<model id>' (OPENROUTER_API_KEY; ids as on openrouter.ai/models).
    effort: low / medium / high. Returns why the model can't play, or None."""
    global CLAUDE_MODEL, CLAUDE_EFFORT, OR_MODEL, OR_EFFORT
    backend, _, mid = str(spec).partition(':')
    backend, mid = backend.strip().lower(), mid.strip()
    if backend in ('claude', 'claude-key'):
        if not claude_cli():
            return 'Claude Code was not found: install it from claude.com/claude-code'
        if backend == 'claude-key':
            key = anthropic_key()
            if not key:
                return 'ANTHROPIC_API_KEY is not set'
            why = check_anthropic_key(key)
            if why:
                return why
            os.environ['ANTHROPIC_API_KEY'] = key
        # 'claude' plays on the Claude login: a key left in the environment would bill the API instead (cli_env)
        os.environ['TFT_CLAUDE_AUTH'] = 'key' if backend == 'claude-key' else 'login'
        os.environ['TFT_COACH_BACKEND'] = 'claude'
        os.environ['TFT_CLAUDE_MODEL'] = CLAUDE_MODEL = mid or CLAUDE_MODEL
        if effort:
            os.environ['TFT_CLAUDE_EFFORT'] = CLAUDE_EFFORT = effort
        return None
    if backend in ('codex', 'chatgpt'):
        return _use_codex(mid, effort)
    if backend == 'grok':
        return _use_grok(mid, effort)
    if backend in ('agy', 'gemini'):
        return _use_agy(mid, effort)
    if backend == 'cursor':
        return _use_cursor(mid, effort)
    if backend == 'openai':
        return _use_openai(mid, effort)
    if backend != 'openrouter':
        return f'unknown model {spec!r}: use {MODEL_SPECS}'
    if not mid:
        return 'name the model: openrouter:<model id>, as listed on openrouter.ai/models'
    if not openrouter_key():
        return 'OPENROUTER_API_KEY is not set'
    try:
        import urllib.request
        models = json.loads(urllib.request.urlopen('https://openrouter.ai/api/v1/models', timeout=15).read())['data']
    except (OSError, ValueError, KeyError) as e:
        return f'could not reach OpenRouter to check the model: {e}'
    m = next((x for x in models if x.get('id') == mid), None)
    if m is None:
        return f'OpenRouter has no model {mid!r} (ids as on openrouter.ai/models, e.g. openai/gpt-6-sol)'
    if 'tools' not in (m.get('supported_parameters') or []):
        return f'{mid} does not support tool calls: the model reads its reference files through tools'
    try:
        req = urllib.request.Request('https://openrouter.ai/api/v1/key',
                                     headers={'Authorization': 'Bearer ' + openrouter_key()})
        left = json.loads(urllib.request.urlopen(req, timeout=15).read())['data'].get('limit_remaining')
    except (OSError, ValueError, KeyError) as e:
        return f'OpenRouter did not accept the key: {e}'
    if left is not None and left < 1:
        return f'the OpenRouter key has ${left:.2f} of credit left - add credit on openrouter.ai/settings/credits'
    os.environ['TFT_COACH_BACKEND'], os.environ['TFT_API_HOST'] = 'openrouter', 'openrouter'
    os.environ['TFT_OR_MODEL'] = OR_MODEL = mid
    if effort:
        os.environ['TFT_OR_EFFORT'] = OR_EFFORT = effort
    return None


def _use_openai(mid, effort):
    """An OpenAI model through the chat-completions loop. OpenAI's model list says nothing about images or
    tools: a model without them fails at its first question."""
    global OR_MODEL, OR_EFFORT
    import urllib.error
    import urllib.request
    if not mid:
        return 'name the model: openai:<model id>, as listed on platform.openai.com/docs/models'
    key = user_env('OPENAI_API_KEY')
    if not key:
        return 'OPENAI_API_KEY is not set'
    try:
        req = urllib.request.Request(API_HOSTS['openai'][0] + '/models', headers={'Authorization': 'Bearer ' + key})
        ids = {m.get('id') for m in json.loads(urllib.request.urlopen(req, timeout=15).read())['data']}
    except urllib.error.HTTPError as e:
        return f'the OpenAI API did not accept the key (HTTP {e.code})'
    except (OSError, ValueError, KeyError) as e:
        return f'could not reach the OpenAI API to check the model: {e}'
    if mid not in ids:
        return f'OpenAI has no model {mid!r} for this key (ids as on platform.openai.com/docs/models)'
    os.environ['TFT_COACH_BACKEND'], os.environ['TFT_API_HOST'] = 'openrouter', 'openai'
    os.environ['TFT_OR_MODEL'] = OR_MODEL = mid
    if effort:
        os.environ['TFT_OR_EFFORT'] = OR_EFFORT = effort
    return None


def _use_codex(mid, effort):
    """A ChatGPT model through the Codex CLI on the user's ChatGPT login (the Codex app ships the CLI)."""
    global CODEX_MODEL, CODEX_EFFORT
    exe = codex_cli()
    if not exe:
        return 'Codex was not found: install the Codex app (or the Codex CLI) and log in with your ChatGPT account'
    try:
        out = subprocess.run([exe, 'login', 'status'], capture_output=True, timeout=30,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        said = (out.stdout + out.stderr).decode('utf-8', 'replace').lower()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f'Codex did not start: {e}'
    if 'logged in' not in said or 'not logged in' in said:
        return 'Codex is not logged in: run "codex login" (or log in in the Codex app)'
    os.environ['TFT_COACH_BACKEND'] = 'codex'
    os.environ['TFT_CODEX_MODEL'] = CODEX_MODEL = mid or CODEX_MODEL
    if effort:
        os.environ['TFT_CODEX_EFFORT'] = CODEX_EFFORT = effort
    return None


def model_name():
    """The game's model as a --model spec ('claude:claude-opus-5-5', 'openrouter:openai/gpt-6-sol'), or None."""
    kind = _cli()[0]
    cli = 'claude-key' if os.environ.get('TFT_CLAUDE_AUTH') == 'key' else 'claude'
    return {'claude-cli': f'{cli}:{CLAUDE_MODEL}', 'claude': f'claude-api:{CLAUDE_MODEL}',
            'openrouter': f'{api_host()}:{OR_MODEL}', 'codex': f'codex:{CODEX_MODEL}',
            'grok': f'grok:{GROK_MODEL}', 'agy': f'agy:{AGY_MODEL}', 'cursor': f'cursor:{CURSOR_MODEL}'}.get(kind, kind)


def model_settings(kind=None):
    """(model, effort) the game's backend runs with - for the run log (a Codex game was logged as Claude's model
    and effort: release run 20261016-141957)."""
    kind = kind or _cli()[0]
    return {'openrouter': (OR_MODEL, OR_EFFORT), 'codex': (CODEX_MODEL, CODEX_EFFORT),
            'grok': (GROK_MODEL, GROK_EFFORT), 'agy': (AGY_MODEL, AGY_EFFORT),
            'cursor': (CURSOR_MODEL, CURSOR_EFFORT)}.get(kind, (CLAUDE_MODEL, CLAUDE_EFFORT))


def user_env(name):
    """A variable from the environment, or the Windows user environment (set since this terminal opened)."""
    k = os.environ.get(name)
    if not k and os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as h:
                k = winreg.QueryValueEx(h, name)[0]
        except OSError:
            k = None
    return k


def save_user_env(name, value):
    """Keep a key for later runs: the Windows user environment (as setx does), and this process."""
    os.environ[name] = value
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment', 0, winreg.KEY_SET_VALUE) as h:
            winreg.SetValueEx(h, name, 0, winreg.REG_SZ, value)


def openrouter_key():
    return user_env('OPENROUTER_API_KEY')


def anthropic_key():
    return user_env('ANTHROPIC_API_KEY')


def check_anthropic_key(key):
    """Does the Anthropic API accept the key? -> why not, or None."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request('https://api.anthropic.com/v1/models',
                                 headers={'x-api-key': key, 'anthropic-version': '2023-06-01'})
    try:
        urllib.request.urlopen(req, timeout=15).read()
    except urllib.error.HTTPError as e:
        return f'the Anthropic API did not accept the key (HTTP {e.code})'
    except OSError as e:
        return f'could not reach the Anthropic API to check the key: {e}'
    return None


TERMINAL_FLAGS = (getattr(__import__('subprocess'), 'CREATE_NEW_CONSOLE', 0) | 0x01000000      # CREATE_BREAKAWAY_FROM_JOB
                  | getattr(__import__('subprocess'), 'CREATE_NEW_PROCESS_GROUP', 0))


def cli_env():
    """The environment for the model's CLI: without the markers a parent Claude Code session
    leaves (games started from a Claude Code session made the model a "child session", whose
    transcript is not saved - the user's post-game discussion was lost, 2026-10-02)."""
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith('CLAUDE_CODE_') or k in ('CLAUDECODE', 'CLAUDE_AGENT_SDK_VERSION'))}
    env['CLAUDE_CODE_FORCE_SESSION_PERSISTENCE'] = '1'
    # questions come every 5-30 s: the 5-minute cache is enough, and its writes cost 5/8 of the 1-hour cache's,
    # which the CLI picks on a Claude login (experiments/token-cost: -23% of a game's cost)
    env['CLAUDE_CODE_PROMPT_CACHE_TTL'] = '5m'
    if os.environ.get('TFT_CLAUDE_AUTH') == 'login':
        env.pop('ANTHROPIC_API_KEY', None)
        env.pop('ANTHROPIC_AUTH_TOKEN', None)
    return env


_CLAUDE_EXE = []


def claude_cli():
    """Claude Code CLI bundled with the desktop app (uses the user's Claude login). Found once: the player asks
    every tick (planner_on), and a PATH search plus a glob there showed up in slow ticks up to 20 s."""
    if not _CLAUDE_EXE:
        found = _find_claude_cli()
        if not found:
            return None
        _CLAUDE_EXE.append(found)
    return _CLAUDE_EXE[0]


def _find_claude_cli():
    found = shutil.which('claude')
    if found:
        return found
    native = Path.home() / '.local' / 'bin' / 'claude.exe'     # `claude update` installs here (2.1.295 knows Haiku 5.5)
    if native.exists():
        return str(native)
    # <version>/claude.exe, or <version>/<build hash>/claude.exe since the app's 2.1.28x updates
    # (the old one-level glob found nothing and run21 played on fallback rules without a model)
    base = Path(os.environ.get('APPDATA', '')) / 'Claude' / 'claude-code'
    version = lambda p: next(([int(x) if x.isdigit() else 0 for x in d.name.split('.')]
                              for d in [p.parent, *p.parents] if d.parent == base), [0])
    exes = sorted(base.glob('**/claude.exe'), key=lambda p: (version(p), p.stat().st_mtime))
    return str(exes[-1]) if exes else None


_CODEX_EXE = []


def codex_cli():
    """The Codex CLI: on PATH (npm install -g @openai/codex), or the one the Codex app ships
    (%LOCALAPPDATA%/OpenAI/Codex/bin/<build>/codex.exe, newest build). Found once."""
    if not _CODEX_EXE:
        found = shutil.which('codex')
        if not found:
            base = Path(os.environ.get('LOCALAPPDATA', '')) / 'OpenAI' / 'Codex' / 'bin'
            exes = sorted(base.glob('*/codex.exe'), key=lambda p: p.stat().st_mtime)
            found = str(exes[-1]) if exes else None
        if not found:
            return None
        _CODEX_EXE.append(found)
    return _CODEX_EXE[0]


def codex_sandbox():
    """The Windows sandbox mode the user's Codex is set up with (config.toml [windows] sandbox): the game's
    calls skip the user's config (no plugins or MCP servers in a game), but without this setting the read-only
    sandbox refuses every shell command and the model cannot read its reference files."""
    return (_codex_config().get('windows') or {}).get('sandbox')


def codex_rollout_answer(thread_id, since):
    """The last answer Codex recorded in its own session file for this thread (sessions/YYYY/MM/DD/rollout-*-<thread
    id>.jsonl) - its turn's final agent message - if that turn completed after `since` (time.time()); else None."""
    if not thread_id:
        return None
    from datetime import datetime
    home = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex') / 'sessions'
    try:
        f = max(home.rglob(f'rollout-*-{thread_id}.jsonl'), key=lambda x: x.stat().st_mtime)
        lines = f.read_text(encoding='utf-8', errors='replace').splitlines()
    except (ValueError, OSError):
        return None
    for line in reversed(lines):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        p = r.get('payload') or {}
        if p.get('type') == 'task_complete':
            try:
                at = datetime.fromisoformat(str(r.get('timestamp')).replace('Z', '+00:00')).timestamp()
            except ValueError:
                return None
            return p.get('last_agent_message') if at >= since - 1 else None
    return None


def codex_service_tier():
    """The service tier the user's Codex is set up with ("priority" = Fast: about 2x speed for more of the plan's
    usage); the game's calls skip the user's config, so it is passed back in (2026-10-18: flex questions answered
    in 7-25 s with it)."""
    return _codex_config().get('service_tier')


def _codex_config():
    try:
        import tomllib
        home = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex')
        return tomllib.loads((home / 'config.toml').read_text(encoding='utf-8'))
    except Exception:
        return {}


REFERENCE = Path(__file__).resolve().parent.parent / 'data' / 'reference'

# The self-improving agent (agent/README.md): TFT_CLIMB=1 (agent/climb.py) gives the player its memory -
# agent/memory, written by the coach after each game - in the preparation and in every stage's first message.
# The player's CLI then runs in PLAYER_DIR, outside any git repo: inside the repo it loaded the repo's CLAUDE.md
# (the developer's and the coach's notes) into every game conversation.
CLIMB = os.environ.get('TFT_CLIMB') == '1'
AGENT = Path(__file__).resolve().parent.parent / 'agent'
MEMORY = AGENT / 'memory'
PLAYER_DIR = Path(__file__).resolve().parent.parent.parent / (Path(__file__).resolve().parent.parent.name + '-player')
# the general playbook left the player's prompt with the stage strategies (the owner's master plan, 2026-11-01):
# what it taught now lives in each comp's stage strategies (agent/stages.py)
MEMORY_FILES = (('general.md', "FIRST RULES FOR EVERY COMP - the owner's rules, whatever the comp (a comp's own rules "
                               "are in its notes)"),
                ('comp', "COMP NOTES - what strong players do with this comp and why, learned from their games and "
                         "yours by the comp coach"),
                ('feedback', "OWNER'S FEEDBACK - the mistakes the owner found in our earlier games, by comp, game and "
                             "round"),
                ('harness.md', 'HARNESS NOTES - what the controller can and cannot do, from your reviews'))


def memory_root():
    """agent/memory, or TFT_MEMORY_DIR: another memory folder (the Judge's reproduction test asks a game's questions
    again with a candidate playbook)."""
    return Path(os.environ['TFT_MEMORY_DIR']) if os.environ.get('TFT_MEMORY_DIR') else MEMORY


def comp_notes_file(comp):
    return memory_root() / 'comps' / (re.sub(r'[^A-Za-z0-9]+', '_', comp.name).strip('_') + '.md')


def comp_feedback_file(comp):
    """The owner's feedback on our games with the comp (agent/expert.py render_feedback)."""
    return memory_root() / 'feedback' / (re.sub(r'[^A-Za-z0-9]+', '_', comp.name).strip('_') + '.md')


def current_harness_text(text):
    """Omit the obsolete state-line reference from player copies; source memory belongs to the coach."""
    return text.replace(
        'The current playbook with the comp notes and the strongest-board line now fields him on the replayed '
        '4-5 and 4-6 questions, so this rule is the floor under that judgement, not the only fix.', '')


def memory_block():
    """CLIMB: the agent's memory files as one text block for the player ('' otherwise, or when they are empty)."""
    if not CLIMB:
        return ''
    comp = forced_comp()
    fs = flex_set()
    parts = []
    for name, title in MEMORY_FILES:
        # flex (no comp chosen yet): the notes of the set's comps (agent/config.json "flex"), else of every comp
        # played so far
        notes = ([comp_notes_file(comp)] if comp else [comp_notes_file(c) for c in fs['comps']] if fs
                 else sorted((memory_root() / 'comps').glob('*.md')))
        # the owner's feedback (agent/expert.py render_feedback): every comp's, then the comps' in play (x19.6)
        said = ([memory_root() / 'feedback' / 'Every_comp.md'] + [comp_feedback_file(c) for c in
                ([comp] if comp else fs['comps'] if fs else [])])
        files = notes if name == 'comp' else said if name == 'feedback' else [memory_root() / name]
        for f in files:
            text = current_harness_text(f.read_text(encoding='utf-8')).strip() if f.exists() else ''
            if text:
                # flex before a commit: each comp's notes and feedback apply if it is played
                tag = (' - for the comp it names, if you play it' if fs and not comp and name in ('comp', 'feedback')
                       and f.name != 'Every_comp.md' else '')
                parts.append(f'=== {title}{tag} ({f.name}) ===\n{text}')
    if not parts:
        return ''
    return ('YOUR MEMORY - written by you (the same model, as your own coach) after reviewing your earlier games on '
            'this account; also in your working directory under memory/.\n\n' + '\n\n'.join(parts))


_STRATS = {}                         # the stage strategies one game plays, read once (stage_strategies)


def stage_strategies():
    """CLIMB with a fixed comp: {stage: {'v', 'text'}} - the comp's stage strategies as they stood when this game
    started (agent/stages.py), or the file TFT_STAGE_STRATEGIES names (the Judge's test of a candidate). Read once per
    game (new_game clears it) and written to the run folder, so the review knows which versions played."""
    if 'data' in _STRATS:
        return _STRATS['data']
    comp, data = forced_comp(), {}
    # a flex game on the climb's set (agent/config.json "flex"): no stage strategy until a commit - then the comp's own
    # (commit_strategies); until then the Strategist's choice file goes with every planning state (flex_choice)
    name = comp.name if comp is not None else FLEX_PLAN if flex_set() else None
    if CLIMB and name and name != FLEX_PLAN:
        try:
            if os.environ.get('TFT_STAGE_STRATEGIES'):
                data = json.loads(Path(os.environ['TFT_STAGE_STRATEGIES']).read_text(encoding='utf-8'))
            else:
                import importlib.util
                spec = importlib.util.spec_from_file_location('climb_stages', AGENT / 'stages.py')
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                data = mod.frozen(name)
        except Exception as e:                           # a broken file is a fault to see, not a silent default
            print(f'stage strategies: not read ({e.__class__.__name__}: {e})', flush=True)
            data = {}
    _STRATS['data'], _STRATS['name'] = data, name
    _save_strategies()
    return data


def commit_strategies(comp, stage):
    """Flex on the climb's set, at a commit or a later switch (the owner, 2026-11-07: "once the comp is committed, comp
    specific strategy should be enabled"; "once certain, the specific comp strategy will be invoked and used"): the
    comp's own stage strategies (agent/comps/<comp>/stages.json) replace the set's plan for the rest of the game; the
    run's record keeps the set's plan and each commit. -> the block for the stage it happened in ('' when the comp has
    no stage strategies: the set's plan stays)."""
    if not CLIMB or not (_STRATS.get('name') == FLEX_PLAN or _STRATS.get('commits')) \
            or os.environ.get('TFT_STAGE_STRATEGIES'):
        return ''
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('climb_stages', AGENT / 'stages.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        data = mod.frozen(comp.name)
    except Exception as e:                               # a broken file is a fault to see, not a silent default
        print(f'stage strategies of {comp.name}: not read ({e.__class__.__name__}: {e})', flush=True)
        return ''
    if not data:
        print(f'{comp.name} has no stage strategies yet: it plays without one', flush=True)
        return ''
    if not _STRATS.get('commits'):
        _STRATS['set'] = _STRATS.get('data')
    _STRATS.setdefault('commits', []).append({'comp': comp.name, 'stage': stage, 'stages': data})
    _STRATS['data'], _STRATS['name'], _STRATS['saved'] = data, comp.name, False
    _save_strategies()
    block = stage_strategy_text(stage)
    if block:                                            # it goes with the commit's brief (player.load_commit_brief)
        _STRATS.setdefault('sent', set()).add((_STAGE['conv'] or 'first', block))
    return block


def _save_strategies():
    """runs/<run>/stage_strategies.json, once the run folder is there: the queue-time preparation reads the
    strategies before any call has made it (game 52 had no record, so its diff reports could not tell which versions
    played), so the first saved call writes it. A flex game's record is the set's plan, with each commit's own
    strategies under "commits"."""
    data = _STRATS.get('data')
    if CALL_DIR is None or not data or _STRATS.get('saved') or not Path(CALL_DIR).parent.is_dir():
        return
    record = ({'comp': FLEX_PLAN, 'stages': _STRATS.get('set'), 'commits': _STRATS['commits']}
              if _STRATS.get('commits') else
              {'comp': _STRATS.get('name') or getattr(forced_comp(), 'name', None), 'stages': data})
    try:
        (Path(CALL_DIR).parent / 'stage_strategies.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding='utf-8')
        _STRATS['saved'] = True
    except OSError:
        pass


def stage_strategy_text(stage):
    """The strategy for the stage's first message per conversation ('' with none). Stages 5-7 share the text."""
    m = re.match(r'\s*(\d+)', str(stage or ''))
    if not m:
        return ''
    s = str(min(int(m.group(1)), 5))
    ver = stage_strategies().get(s)
    comp = forced_comp()
    name = _STRATS.get('name') or (comp.name if comp is not None else None)
    if not ver or not name:
        return ''
    names = {'1': 'stage 1 (1-1 to 1-4)', '5': 'stages 5 to 7'}
    return (f"STAGE {s} STRATEGY for {name} ({names.get(s, 'stage ' + s)}, version {ver['v']}) - learned from "
            f"strong players' games and our earlier games with this comp; it applies to every decision of this "
            f"stage, and where it disagrees with the comp's FIRST RULES, the FIRST RULES hold:\n{ver['text'].strip()}")


def climb_settings():
    """CLIMB: the coach's settings (agent/config.json), read when asked - {} otherwise. Two shape the game: "levelling",
    the comp plan's levels stated in every planning state in place of MetaTFT's line, and "auto_buy_stars", the star
    level auto_buy starts at for named comp units (g15: MetaTFT's "Fast 9" line and auto_buy at 2 stars every
    phase, while the comp's Master+ boards win with Elise 3 + Cassiopeia 3 - our 15 games: Elise 3 once)."""
    if os.environ.get('TFT_CLIMB') != '1':
        return {}
    try:
        cfg = json.loads((AGENT / 'config.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def auto_buy_stars():
    """{champion: 1-3} from climb_settings' "auto_buy_stars" (valid entries only)."""
    stars = climb_settings().get('auto_buy_stars')
    return {n: s for n, s in (stars.items() if isinstance(stars, dict) else ())
            if isinstance(n, str) and isinstance(s, int) and 1 <= s <= 3}


FLEX_PLAN = 'flex'                   # the climb's flex set: the name its plan has in agent/comps (agent/stages.py)


def flex_set():
    """CLIMB flex on a set (agent/config.json "flex"; the owner, 2026-11-06: 2-3 S-tier comps the Strategist plans
    on, the player's probability of each direction every reply, a commit once certain): {'comps': [Comp],
    'commit_p': the P that commits a comp, 'commit_by': the round the comp with the highest P is committed at} - or
    None (a plain flex game: every MetaTFT comp)."""
    f = climb_settings().get('flex')
    if not isinstance(f, dict):
        return None
    comps = [c for c in (META.by_name(str(n)) for n in f.get('comps') or []) if c]
    if not comps:
        return None
    try:
        p = float(f.get('commit_p', 0.7))
    except (TypeError, ValueError):
        p = 0.7
    by = str(f.get('commit_by') or '')
    return {'comps': comps, 'commit_p': p, 'commit_by': by if re.fullmatch(r'\d-\d', by) else '4-1'}


def flex_choice():
    """The Strategist's choice file of the climb's flex set (agent/comps/flex/choice.json; the owner, 2026-11-07: "a
    file written by the strategist on how to choose and when to choose which comp"): the block the preparation and
    every planning state carry until a commit ('' with none)."""
    if not flex_set():
        return ''
    try:
        rows = json.loads((AGENT / 'comps' / 'flex' / 'choice.json').read_text(encoding='utf-8')).get('versions') or []
    except (OSError, ValueError):
        rows = []
    ver = next((v for v in reversed(rows) if v.get('status') == 'current'), None)
    if not ver or not str(ver.get('text') or '').strip():
        return ''
    return (f"HOW TO CHOOSE THE COMP (the Strategist's choice file, version {ver['v']}; where it disagrees with the "
            f"FIRST RULES, the FIRST RULES hold):\n{ver['text'].strip()}")


def in_flex_set(name):
    """Flex on a set: whether a comp name is one of the set's (True when there is no set)."""
    fs = flex_set()
    comp = META.by_name(str(name)) if name else None
    return fs is None or (comp is not None and any(c.id == comp.id for c in fs['comps']))


def sync_player_dir():
    """CLIMB: the player's working directory before a game - the reference files and the memory, copied fresh."""
    if not CLIMB:
        return
    try:
        PLAYER_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(REFERENCE, PLAYER_DIR, dirs_exist_ok=True)
        if (PLAYER_DIR / 'memory').exists():
            shutil.rmtree(PLAYER_DIR / 'memory')
        if MEMORY.exists():
            shutil.copytree(MEMORY, PLAYER_DIR / 'memory')
            for f in (PLAYER_DIR / 'memory').rglob('*.md'):
                text = f.read_text(encoding='utf-8')
                cleaned = current_harness_text(text)
                if cleaned != text:
                    f.write_text(cleaned, encoding='utf-8')
    except OSError as e:
        print('player folder not synced:', e)


def player_cwd():
    """The model CLI's working directory: PLAYER_DIR in climb games, else the reference files."""
    if CLIMB and PLAYER_DIR.exists():
        return str(PLAYER_DIR)
    return str(REFERENCE) if REFERENCE.exists() else tempfile.mkdtemp(prefix='tft-claude-')


OPENING = """You are playing a live Teamfight Tactics game (Set <SET>, patch <PATCH>).
Your goal is the best expected final placement. You make every strategic decision in this game.

HOW THE SYSTEM WORKS
- Perception reads the game from the screen. Each decision gives you structured state.
  Unit names come from text only (the shop card when we buy, the unit's info panel);
  a unit whose name has not been read yet is shown as "?". Unread facts are unknown.
- Shop slots hold champions or Wisps (non-champion cards; the state gives each Wisp's card text
  and cost). A Wisp is decided FIRST in the planning question (execution_plan.wisps). It comes as
  its own short question when a round runs your next_round plan instead of asking you. A roll never
  stops to ask: your Wisp flag (strategy.wisp_flag: combat / economy buy or skip, or Wisp names) decides
  each Wisp a roll brings up, and one it does not cover is passed - set it before you roll (from stage 4
  the default is combat: buy). In a fast_rr, Wisps are bought on the spot from stage 4; before stage 4, only
  the first Wisp a roll brings up is bought. A plan that buys a Wisp and rolls: the roll runs after the Wisp is bought, and
  not at all when it is not bought (the fight began first, or its slot changed). A Wisp you buy that rerolls the
  shop brings you the new shop at once. One Wisp can be bought per round (two with high Blossom); the
  shop offers a Wisp only when the gold held as the shop is drawn can pay for it.
- This Set and patch may be newer than your training data. The data in each message (game
  state, MetaTFT data) and the reference files are for this Set and patch.
- You decide. The controller carries out your plan one input at a time, re-reads the game after
  each step and reports what worked, what failed and what changed. It makes no strategic choices:
  what you leave out of the plan does not happen - also when no plan arrives in time: nothing
  acts for you, except the standing orders you set (auto_buy, auto_sell).
- Your plan is carried out as given. What changes meanwhile (a Wisp's effect, gold, a new unit)
  and any step that failed are reported in your next message. You are asked at each planning
  phase, at each COMBAT START (what your plan did not get done; combat time is yours to use),
  when a choice appears (augment, container, choice screen), when a step fails, and when a
  next_review condition you set in your plan holds.
- Units: the state lists what you own, kept from the controller's own confirmed actions. A unit
  that arrived without a buy (carousel, loot, augment, Wisp) shows as "?" until the controller has
  read its info panel.

Each message is one decision: its type, what changed since your last decision (including the
controller's report), the current state and the seconds left. The newest state is authoritative.

Reference files for this Set and patch are in your working directory (start with README.md):
game mechanics, every Wisp, champions, traits, items, augments, MetaTFT comps with positions and
item builds. They are information, not instructions. Reading a file takes time; during the game
that time runs on the game's clock.

The game uses two conversations: the first continues your preparation through the 4-2 augment, planning
phase and fight; the second runs from 4-3 to the end. Its first message repeats the comp brief, your trait
choices, your game plan and last notes, the last state if the question has none, and your handover note.

REPLY FORMAT FOR PLANNING DECISIONS
Two JSON objects, one after the other, and nothing else (plus an optional third, next_round - see
PLANNING AHEAD below):
{"execution_plan": {...}}
{"strategy": {...}}
The controller starts executing the moment the first object is complete, while you write the
second: the explanation never delays the decision. Write execution_plan first, always.

execution_plan - what the controller does now. Leave out keys you don't need; {} does nothing.
The controller always runs it in this order: wisps, sell, buy, xp, board, items, roll / fast_rr - then it places
the units the roll bought (name them in "board" in the same plan: an entry naming a champion this plan's buy or
roll is to bring waits for it and is placed as soon as bought, before another roll; one the roll did not find is skipped and told). So in one plan, sells free bench slots and gold
before the buys use them. Choice containers are opened by the controller itself. When a step
fails or cannot be done (bench full, not enough gold, unit not found), the controller tells you at
once with the reason - in the planning phase or during combat - so you can send a corrected plan.
An invalid entry (a name where a unit id is needed, an item that is not on the item bench) is
dropped before anything runs, and you are told which and why; the rest of the plan runs.
Before searching, name each unit you intend to field and its hex in this plan. Preset the dependent bench moves
too; never defer a hit's placement to another model call.
During combat the board is locked (no moving or selling board units), but buying,
selling bench units, rerolling and XP work: the controller does those at once and keeps
board moves and board sells for the next planning phase. Item gives to fielded units in your
execution_plan are tried during the fight too (about 6 in 10 land): in a combat reply keep every
give in execution_plan, not in next_round - one that does not land in the fight carries into the
next planning phase and runs first there. Wisps leave the shop when combat
starts (the champion card under each one shows): no Wisp can be bought in combat.
{
  "wisps":   [{"slot": <shop slot>, "buy": true|false, "why": "<one sentence>"}],
             // FIRST: one entry for every Wisp the state asks you to decide; a Wisp is bought before
             // anything else. Not in next_round (that shop is not known yet).
  "sell":    [unit ids],
  "open_container": [bench slots 0-8],    // optional: the controller opens a choice container (item
             // anvil, artifact anvil, ...) by itself as soon as it sees one and no question to you is
             // running, also in combat; its offer is then shown to you and you choose. You never need
             // to list one here.
  "buy":     ["Champion", ...],           // champions in the shop, by name, in this order. A full bench (9/9)
             // takes a buy only if it is a 3rd copy (it merges); otherwise put a "sell" in the same plan.
             // Stage 1: if no plan of yours for the round has come with 3 s left, every affordable card is bought
  "xp":      <n>,                         // buy XP n times (4 gold each)
  "board":   [{"unit": <id>, "hex": [row, col]}, {"name": "Champion", "hex": [row, col]}],
             // where units go: each listed unit is moved to that hex ("name" for units bought in this plan or
             // its roll - yes, a unit the same plan's roll or fast_rr may buy: it is placed as soon as bought, and an
             // entry whose champion was not bought is skipped - and for units still arriving, an augment's or a
             // Wisp's: placed the moment they are on the bench). Units you do not list stay where they are. The board holds at most `level` units: a
             // unit placed on an occupied hex swaps with the unit standing there (one drag; that unit takes the
             // placed unit's old spot, or its own listed hex if free); to take a unit off the board otherwise,
             // list it in "bench".
  "bench":   [{"unit": <id>, "slot": <bench slot 0-8>}],
             // put these units in these bench slots (from the bench or the board; done before "board"); a
             // unit already in that slot is moved to another free slot first.
  "link":    [{"from": <unit id> | [row, col], "to": <unit id>}],
             // press on the source, hover on the target ~1 s, release - the game's drag-and-hold link (the BFF
             // or its current rider to a new rider, Taric to his ally). Done after "board"; from: a board unit or a
             // board hex.
             // Optional on board, bench and link entries: "when_owned": "Champion" waits until that champion
             // arrives. Example: before rolling for Elder Dragon, name his hex and gate the two bench moves
             // freeing his team slots with "when_owned": "Elder Dragon". Gate any dependent Taric swap/link too.
  "items":   [{"give": "<item on the item bench>", "to": <unit id>, "final": "Champion"},
              {"use": "<consumable on the item bench>", "on": <unit id>},
              {"reforge": <unit id>},
              {"transfer": "<item>", "from": <unit id>, "to": <unit id> | "bench"}],
             // ACTIONS, each once (a give whose item is no longer on the item bench but already on
             // that unit counts as done). give: two components on one unit build the completed item -
             // name each component; only what is on the item bench can be given. reforge: a Reforger
             // turns ALL the unit's items into random ones of a similar type. transfer: a Remover
             // takes ALL of the source's items off, so give EVERY item on the source a destination
             // (another unit, the same unit to put it back, or "bench"). Not during combat.
  "roll":    {"times": <n>, "buy": ["Champion", ...] | {"Champion": <star 1-3>, ...}, "sell_others": false},
             // reroll n times, buying the listed champions until each has its star level's copies on
             // board + bench (1 / 3 / 9 copies; a plain list means its auto_buy star, else 2). A Wisp in a
             // new shop stops the rolling and comes to you as its own short question; then the rest is rolled.
  "fast_rr": {"buy": ["Champion", ...] | {"Champion": <star>, ...}, "gold_floor": <g>,
              "stop_when": [{"any": ["Champion", ...], "star": <2|3>}], "sell_others": true},
             // a fast roll-down instead: reroll while gold >= g + 2, buying as above; stops early when every
             // stop_when entry has one of its champions with that star level's copies (merged or not).
             // From stage 4 every Wisp that shows up is bought on the spot, with no question.
             // sell_others (fast_rr: true unless you set false; roll: false unless you set true): before
             // the first roll, every BENCH unit outside buy, keep, board and auto_buy is sold (also in
             // combat). Rolling stops only at your own conditions (stop_when, gold_floor, times), or when
             // the game stops it: the bench is full and a wanted card cannot be bought, or the game does
             // not take a roll (you are told which). Both list buys best first: a paid roll always keeps
             // the price of the first-listed champion still short, so a shop showing it can buy it.
             // Riftbeast Brambleback: a paid refresh retains the MOST expensive unfinished named hit's
             // price after its 2 gold refresh (4-cost search: 6g before D; including 5-costs: 7g).
             // This applies even with gold_floor 0. Named star targets complete -> no more paid rolls.
             // This comp's Brambleback/Sentinel searches cap at 2-star; held upgrade copies survive
             // bench clearing. The controller also equips Brambleback's mapped carry items and Alpha
             // in his entry phase: those transfers count toward his three slots. Automatic entries
             // use Brambleback (0,6), Sentinel (3,4); an explicit board hex chooses their position.
  "keep":    [unit ids],                  // never sold by the controller
  "auto_buy": {"Champion": <star 1-3>, ...},
             // STANDING ORDER, kept all game until you change it: whenever the shop (also during a roll)
             // shows a listed champion and we own fewer copies than its star level needs (1 / 3 / 9), the
             // controller buys it at once - planning or combat, no gold reserve, no question. Adds to or
             // changes the list; 0 takes a champion off it. The state shows the list and what it bought.
  "auto_sell": true | false | ["Champion", ...],
             // STANDING ORDER, off until you turn it on: true = every BENCH unit whose champion is not on
             // auto_buy, and every spare copy beyond its auto_buy star level, is sold as soon as seen; a list =
             // only those champions. Never board units, 2- or 3-star units (only your own sell sells them),
             // units in your board/keep, or units whose identity
             // is unverified. false turns it off.
  "after_fight": {"win":  {"sell": [unit ids], "buy": ["Champion", ...]},
                  "lose": {"sell": [unit ids], "buy": ["Champion", ...]}},
             // orders for the moment THIS round's fight result shows (our HP drops, the streak flame
             // changes, or the +1 win gold arrives) - before the next round starts and pays interest on
             // the gold held then. Only the orders for the actual result run: its sells first (bench
             // units), then its buys from the shop on screen (one card per name; the shop is replaced
             // when the next planning phase opens). You are told what was done.
  "item_final": [{"unit": <id>, "item": "<item it holds>", "final": "Champion", "star": <1-3>}],
             // HOLDERS: every item has two - the unit holding it now (its current holder) and the champion it is
             // meant for (its final holder). A give names both: "to" is the current holder, "final" the final holder
             // (also for a component: the item it builds keeps it; without "final" it is the unit it goes on); add "final_star" (1-3, default 1) for the star the
             // final holder must have. A transfer changes only the current holder. item_final changes an item's final
             // holder ("final": null clears it). The moment we own an item's final holder at that star (its name read
             // off its panel) in a planning phase, the controller moves the item to it with no question to you: a
             // current holder that is not in the comp's final board is sold (its items return to the item bench, the
             // final holder takes its hex if the holder was fielded, and the item is given to the final holder); one
             // that is in it keeps its place - an Item Remover takes all its items off, the item goes to the final
             // holder and the others back onto it. A fielded holder is sold only when the final holder takes its hex
             // (else the Remover, or nothing). The final holder must be worth at least the current holder's gold
             // (cost x 3 per star: Diana 2 = 9 waits for Sentinel 2 = 12). The state lists every item's holders and
             // the star each move waits for; you are told what moved.
  "gold_floor": <g>,                      // no spending below g, except explicit xp; a roll's paid rerolls stop
                                          // at it too (free rerolls, and buying the units a roll finds, do not)
  "next_review": [{"plan_complete": true} | {"seconds_left": <n>} | {"gold_at_least": <n>} |
                  {"unit": "Champion", "star": <1-3>}]
             // optional: ask me again this planning phase as soon as any of these holds.
             // Without it you are asked again at the next planning phase (or a choice event).
}
Hexes: row 0 is the back row (next to the bench), row 3 the front row; columns 0-6 left to right.
Unit ids stay the same all game. Shop slots are valid only for the state they came with.

PLANNING AHEAD (optional, any round): a third object after strategy,
{"next_round": {...same keys as execution_plan...}}
is carried out the moment the NEXT planning phase opens, instead of asking you first: no thinking
time is spent there, execution starts at once. Without it you are asked at the start of the next
planning phase as usual. Name champions in "buy" (no shop slots: that shop is not known yet) and
use "name" in "board" for units not owned yet. When it is complete you are asked again with what
it did, unless you set its own "next_review" ([] = don't ask me this round). A newer next_round
replaces an older one; a choice or augment in between does not cancel it.

strategy - where you are going and why. Every reply must include it.
{
  "comps":     [{"name": "<MetaTFT comp name, exactly as in the briefing>", "p": 0.3}, ...],  // 1-3 comps
  "other":     0.4,
  "commit":    null,   // or "<MetaTFT comp name, exactly as in the briefing>": the comp you play from now on
                      // (COMMIT below)
  "fight_goal": "win" | "lose",   // the result you play this round's fight for; it stays until you change it
  "why": {   // for the people watching: plain words, one short sentence per part, in this order.
             // Champion and item names, places like "front row, left" - no unit ids, no hex
             // numbers, no JSON words. "-" for a part that does nothing this round.
    "goal":       "<why you play this fight for that result>",
    "sell":       "<what you sell and why>",
    "buy":        "<what you buy and why>",
    "level":      "<XP and rolls: what and why>",
    "board":      "<who is fielded where and why - every round, also when nothing moves>",
    "items":      "<which items go on whom and why>",
    "containers": "<anvils or other bench containers: open now or keep, and why>",
    "next":       "<what you expect to do next round>"
  },
  "notes":     "<anything to remember, including doubts about perception>"
}
P = the probability that this comp is the FINAL board of this game. The listed comps plus "other" must sum
to 1. Until you commit, the overlay shows these probabilities.
COMMIT - "commit" names the comp you play from now on; null = not decided yet. A commit makes it this game's
target comp, as in a fixed-comp game: its MetaTFT page and target board hexes are sent to you, auto_buy is
set to its units (to 2-star), and the overlay shows it as TARGET COMP. Before a commit nothing is bought or
placed for any comp beyond what your plans say. After a commit you can still change comp: "pivot":
{"to": "<comp>", "why": "..."} in strategy.

Augment decisions and combat reviews give their own short reply format; there too the
decision comes first and the reason after it."""


AUTO_OPEN_TEXT = {
    'order': ("""The controller always runs it in this order: wisps, sell, buy, xp, board, items, roll / fast_rr - then it places
the units the roll bought (name them in "board" in the same plan: an entry naming a champion this plan's buy or
roll is to bring waits for it and is placed as soon as bought, before another roll; one the roll did not find is skipped and told). So in one plan, sells free bench slots and gold
before the buys use them. Choice containers are opened by the controller itself.""",
              """The controller always runs it in this order: wisps, sell, buy, xp, board, items, roll / fast_rr - then it places
the units the roll bought (name them in "board" in the same plan: an entry naming a champion this plan's buy or
roll is to bring waits for it and is placed as soon as bought, before another roll; one the roll did not find is skipped and told) - and open_container. So in one plan, sells free bench
slots and gold before the buys use them. A container is opened first while 11 s or more are left in
the planning phase (you need ~9 s to choose from its offer), otherwise last."""),
    'open': ("""  "open_container": [bench slots 0-8],    // optional: the controller opens a choice container (item
             // anvil, artifact anvil, ...) by itself as soon as it sees one and no question to you is
             // running, also in combat; its offer is then shown to you and you choose. You never need
             // to list one here.""",
             """  "open_container": [bench slots 0-8],    // choice containers on the bench (the state lists them:
             // item anvil, artifact anvil, ...). Opening one sells it; its offer is then shown to
             // you and you choose. Opening now or later is your decision."""),
}


def auto_open_anvil():
    """Choice containers (anvils) on the bench: True = the controller opens them itself the moment it sees
    one and no question is running (default when the comp is fixed: the model only chooses among the
    offered options); False = the model is told and decides when to open (execution_plan.open_container)."""
    return forced_comp() is not None


def forced_comp():
    """TFT_FORCE_COMP (tools/play_comp.py): the one comp this game plays, or None."""
    name = os.environ.get('TFT_FORCE_COMP')
    return META.by_name(name) if name else None


FORCED_STRATEGY = """strategy - why. Every reply must include it.
{
  "fight_goal": "win" | "lose",   // the result you play this round's fight for; it stays until you change it
  "why": {   // for the people watching: plain words, one short sentence per part, in this order.
             // Champion and item names, places like "front row, left" - no unit ids, no hex
             // numbers, no JSON words. "-" for a part that does nothing this round.
    "goal":       "<why you play this fight for that result>",
    "sell":       "<what you sell and why>",
    "buy":        "<what you buy and why>",
    "level":      "<XP and rolls: what and why>",
    "board":      "<who is fielded where and why - every round, also when nothing moves>",
    "items":      "<which items go on whom and why>",
    "containers": "<anvils or other bench containers: open now or keep, and why>",
    "next":       "<what you expect to do next round>"
  },
  "notes":     "<anything to remember, including doubts about perception>"
}

"""


PILOT_OPENING = """You make the judgement calls in a live Teamfight Tactics game (Set <SET>, patch <PATCH>).
A rule-based controller plays one comp this game: <COMP>.
Its final board at level 8: <UNITS>.
Traits of that board: <TRAITS>.
Item builds (MetaTFT): <BUILDS>.
The controller does the economy, the levelling (MetaTFT timings: <LEVELS>), buying the comp's units,
the level-8 roll-down, the board, positions and items.

You decide:
- augments: three cards are offered; you may reroll each card once, or take one;
- Wisps (non-champion shop cards): buy or skip.
Each question brings the game state, the options and their data as text.

Reference files for this Set and patch are in your working directory (README.md first): augments,
Wisps, champions, traits, items, MetaTFT comps. The questions have a clock.

Reply with the decision as one JSON object first, then {"why": "<2-4 sentences>"}. Nothing else."""


def pilot_opening(pilot):
    """The standing brief in pilot mode: the comp, its board, traits and builds, the model's calls."""
    count = Counter(t for u in pilot.final for t in (CATALOG.unit(u).traits if CATALOG.unit(u) else ()))
    traits = ', '.join(f"{t} {n} (breakpoints {'/'.join(map(str, CATALOG.traits.get(t) or ()))})"
                       for t, n in count.most_common())
    builds = '; '.join(f"{u}: {', '.join(items)}" for u, items in pilot.builds.items()) or 'none'
    levels = ', '.join(f'{s[0]}-{s[1]}: {lv}' for s, lv in pilot.schedule)
    return (PILOT_OPENING.replace('<SET>', str(CATALOG_SET)).replace('<PATCH>', str(META.patch))
            .replace('<COMP>', pilot.comp.name).replace('<UNITS>', ', '.join(pilot.final))
            .replace('<TRAITS>', traits).replace('<BUILDS>', builds).replace('<LEVELS>', levels))


GAME_STRATEGY = """GAME STRATEGY - your strategy for this game, one paragraph in your own words. You write it while you
prepare (before the game); it is pinned on screen for the people watching (panel STRATEGY, bottom right)
and shown to you in every planning message. Send it again when your strategy changes (a new
direction, comp, tempo or item plan), and in every combat-start message (STRATEGY REWRITE: the whole
paragraph as it stands now, after what this game has shown), as one more object after strategy:
{"game_strategy": {<COMP>"text": "<the whole paragraph, at most about 400 characters>",
                   "why": "<what changed, one short sentence, or 'unchanged' (not in the first one)>"}}
Leave it out otherwise: the last one stays.<FLEX>

"""
# flex: the comp is chosen by strategy.commit (OPENING), not here
# the climb (the owner's structure, 2026-10-27): the game plan file - written in the preparation, revised only at
# checkpoints with an EV reason, saved as runs/<run>/game_plan.json; it replaces the paragraph rewritten every combat
# (game 33: "level 7 by 3-5" became "bank for level 8 at 4-2" between two combats, with no reason given)
GAME_PLAN = """GAME PLAN - your plan for this game. You write it while you prepare (before the game); it is pinned on
screen, shown to you in every planning message and given to the second conversation, which starts at 4-3. It changes
only at the checkpoints - the first planning phase of each stage (2-1, 3-1, 4-1, ...) and the augment rounds (2-1, 3-2, 4-2) - and
only when this game has shown a reason: "why" gives it as an EV comparison (what the old plan costs, what the new one
gains). A plan sent at any other time is not applied. Format, as one more object after strategy:
{"game_strategy": {"text": "<the plan in one paragraph, at most about 400 characters>",
                   "boards": {"<level>": ["<champion>", ...], ...},
                   "items": {"<champion>": ["<completed item>", ...], ...},
                   "why": "<what this game showed, and the EV of the change; 'unchanged' if you send it unchanged>"}}
boards: the board you aim for at each level; items: who holds which item (the carry first). No round is tied to a
level: when to level is decided in each round's plan, from that round's numbers. Leave it out when it is unchanged:
the last one stays.

"""
GAME_STRATEGY_FLEX = ('', '\nUntil you commit, the panel shows your comp probabilities; then it shows this paragraph.')


def opening_prompt():
    """The model's standing brief for the whole game (the user's design).
    Fixed comp: no comp probabilities or target to choose, and one line naming the comp."""
    text = OPENING
    if not auto_open_anvil():
        for k in ('order', 'open'):
            text = text.replace(AUTO_OPEN_TEXT[k][0], AUTO_OPEN_TEXT[k][1], 1)
    else:
        text = text.replace(AUTO_OPEN_TEXT['open'][0] + '\n', '', 1)
    comp = forced_comp()
    gs = GAME_STRATEGY.replace('<COMP>', '' if comp else GAME_STRATEGY_FLEX[0]).replace(
        '<FLEX>', '' if comp else GAME_STRATEGY_FLEX[1])
    i = text.index('Augment decisions and combat reviews')
    text = text[:i] + gs + text[i:]
    if comp:
        a = text.index('strategy - where you are going and why.')
        text = text[:a] + FORCED_STRATEGY + text[text.index('GAME STRATEGY - your strategy'):]
        low = [n for n in comp.units if CATALOG.unit(n) and CATALOG.unit(n).cost <= 4]      # player.comp_auto_buy
        five = [n for n in comp.units if n not in low]
        stars = auto_buy_stars()
        text += (f"\n\nTarget comp for this game: {comp.name} ({comp.id}). auto_buy starts as the 1-4 cost units of "
                 f'its target final board ({", ".join(f"{n} {stars.get(n, 2)}*" for n in low)})'
                 + (f'; its 5-costs ({", ".join(five)}) only when you add them' if five else '')
                 + '; auto_sell starts off. Both can be changed at any time.')
        if CLIMB:
            text = text.replace(gs, GAME_PLAN, 1)
    fs = None if comp else flex_set()
    if fs:                               # the climb's flex set: its comps and the commit rule (player.update_comps)
        text = text.replace(FLEX_COMMIT_TEXT, FLEX_SET_COMMIT_TEXT.format(
            names=', '.join(c.name for c in fs['comps']), p=fs['commit_p'], by=fs['commit_by']), 1)
        if CLIMB:                        # the climb's game plan, as with a fixed comp (it changes only at checkpoints)
            text = text.replace(gs, GAME_PLAN.rstrip('\n') + GAME_STRATEGY_FLEX[1] + '\n\n', 1)
    return text.replace('<SET>', str(CATALOG_SET)).replace('<PATCH>', str(META.patch))


FLEX_COMMIT_TEXT = ('Before a commit nothing is bought or\nplaced for any comp beyond what your plans say. After a commit '
                    'you can still change comp: "pivot":\n{"to": "<comp>", "why": "..."} in strategy.')
FLEX_SET_COMMIT_TEXT = ('THIS GAME\'S COMP SET: {names} - the only comps you give P for,\ncommit to or pivot to. The '
                        'controller commits the comp whose P reaches {p:.2f} in a reply, and at {by} the comp\nwith the '
                        'highest P if none is committed by then; you can commit earlier yourself. Before a commit\n'
                        'auto_buy collects the 1-4 cost units of every comp in the set (to 2-star); after it, only the '
                        'committed\ncomp\'s. After a commit you can still change comp within the set: "pivot": '
                        '{{"to": "<comp>", "why": "..."}}\nin strategy.')


SYSTEM_PROMPT = opening_prompt


class _Session:
    """The Claude Code CLI conversation, kept open through the 4-2 fight, then replaced once at 4-3. Each
    question is a new message; the second conversation opens with a digest and handover. If the
    process dies, the next question starts a fresh session."""

    def __init__(self):
        self._proc = None
        self._results = None
        self._lock = threading.Lock()        # one question at a time
        self._write = threading.Lock()       # stdin: a question, or an interrupt while one is in flight
        self.cut = False                     # the reply in flight was interrupted (cut_preparation)
        self._partial = None                 # text chunk callback of the question in flight
        self.session_id = None               # of the last answer: lets a terminal resume it

    def _start(self, exe):
        import queue
        # the reference files are the working directory; the model may read/search them
        d = player_cwd()
        cmd = [exe, '-p', '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose',
               '--model', CLAUDE_MODEL, '--effort', CLAUDE_EFFORT, '--strict-mcp-config',
               '--system-prompt', opening_prompt(),
               '--tools', 'Read,Grep,Glob', '--allowedTools', 'Read,Grep,Glob',
               '--include-partial-messages']      # text as it is written: act on the first answer
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, cwd=d, env=cli_env(),
                                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self._results = queue.Queue()
        proc, results = self._proc, self._results

        def reader():
            for line in proc.stdout:
                if b'"text_delta"' in line:
                    try:
                        ev = json.loads(line.decode('utf-8', 'replace')).get('event') or {}
                        chunk = ev.get('delta', {}).get('text', '')
                        LIVE['text'] += chunk                 # the reply as it is written, for the overlay
                        if self._partial:
                            self._partial(chunk)
                    except Exception:
                        pass
                elif b'"type":"result"' in line:
                    try:
                        results.put(json.loads(line.decode('utf-8', 'replace')))
                    except ValueError:
                        pass
            results.put(None)                  # process ended

        threading.Thread(target=reader, daemon=True).start()

    def warm(self, exe):
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start(exe)

    def close(self):
        """End the conversation's process: the next question starts a new conversation."""
        with self._lock:
            self._state_sent = {}
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
            self._proc = None

    def interrupt(self):
        """Cut the reply in flight short (the CLI's interrupt request): it ends at once as an error result,
        and the conversation goes on with what the model read so far."""
        with self._write:
            if self._proc is not None and self._proc.poll() is None:
                self.cut = True
                req = {'type': 'control_request', 'request_id': f'cut-{time.time():.3f}', 'request': {'subtype': 'interrupt'}}
                self._proc.stdin.write((json.dumps(req) + chr(10)).encode())
                self._proc.stdin.flush()

    def ask(self, exe, content, timeout=90, on_first=None, call_n=None, fresh=False, raw=False):
        """on_first(obj): called (from the reader thread) with the reply's first
        complete JSON object as soon as it has been written - the decision - while
        the model is still writing the rest (the explanation). fresh: content opens a new conversation as it is
        (the second conversation, opened early in the 4-2 fight). raw continues with internal text as given."""
        asked = time.time()
        with self._lock:
            # the seconds this question waited for the one before it (one question at a time): request e3, game 35 -
            # half of the late updates sat 4-9 s behind a Wisp question, counted in their answer time
            self.queued_s = round(time.time() - asked, 1)
            if STOPPED or not _still_wanted():
                raise ModelStopped()
            content = age_prompt(content, self.queued_s)
            if getattr(self, '_digest_pending', False):
                self._digest_pending = False
                digest = stage_digest(_STAGE['now'], backend_switch=True)
                _save_text(call_n, 'digest.txt', digest)
                content = [dict(c, text=digest + c['text']) if i == 0 else c for i, c in enumerate(content)]
                content = _with_stage_strategy(content, call_n)
                fresh = True
            content, new = (content, fresh) if fresh or raw else _stage_turnover(content, call_n)
            content = _outgoing(content, self, fresh=new or self._proc is None or self._proc.poll() is not None, call_n=call_n)
            if new and self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()               # the last stage's conversation ends
            if new or self._proc is None or self._proc.poll() is not None:
                self._start(exe)
            self._partial = _first_object_watcher(on_first) if on_first else None
            LIVE.update(what=_question_line(content), t0=time.time(), text='', took=None)
            msg = json.dumps({'type': 'user', 'message': {'role': 'user', 'content': content}})
            self.cut = False
            try:
                with self._write:
                    self._proc.stdin.write((msg + chr(10)).encode())
                    self._proc.stdin.flush()
                r = self._results.get(timeout=timeout)
            except Exception as e:
                self._proc = None
                raise RuntimeError(f'session failed: {e}')
            if r is None:
                self._proc = None
                raise RuntimeError('session ended')
            self._partial = None
            LIVE['took'] = time.time() - LIVE['t0']
            self.session_id = r.get('session_id') or self.session_id      # also after an error: the conversation exists
            cut = self.cut and r.get('is_error')
            if r.get('is_error') and not cut:
                raise RuntimeError(r.get('result'))
            # turns > 1 means the model used its tools (read/searched the reference files)
            self.last_meta = {k: r.get(k) for k in ('num_turns', 'duration_ms', 'duration_api_ms', 'total_cost_usd',
                                                    'usage', 'modelUsage', 'stop_reason', 'session_id')}
            self.last_meta['queued_s'] = getattr(self, 'queued_s', None)
            self.last_meta['call'] = call_n
            if CALL_DIR is not None:
                try:
                    (CALL_DIR / f'{call_n or 0:04d}-result.json').write_text(json.dumps(r, ensure_ascii=False), encoding='utf-8')
                except Exception:
                    pass
            self.session_id = r.get('session_id') or self.session_id
            return LIVE['text'] if cut else r.get('result', '')      # cut: the text written until then

    def open_terminal(self, exe, locked=True):
        """After the game: stop the headless process and resume this very conversation in a
        console window, so the model can be asked why it did what it did. Returns True if opened."""
        import contextlib
        with (self._lock if locked else contextlib.nullcontext()):     # F8: a question may hold the lock
            if not self.session_id:
                return False
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            self._proc = None
            d = str(REFERENCE) if REFERENCE.exists() else tempfile.gettempdir()
            # its own console, outside the launcher's job: stopping the launcher (or its time limit)
            # must not close the conversation the user is reading (run27's terminal was killed)
            cmd = [exe, '--resume', self.session_id, '--model', CLAUDE_MODEL,
                              '--effort', CLAUDE_EFFORT, '--strict-mcp-config',
                              '--system-prompt', opening_prompt(), '--tools', 'Read,Grep,Glob']
            try:
                subprocess.Popen(cmd, cwd=d, env=cli_env(), creationflags=TERMINAL_FLAGS)
            except OSError:              # the launcher's job forbids breakaway: a plain new console
                subprocess.Popen(cmd, cwd=d, env=cli_env(),
                                 creationflags=getattr(subprocess, 'CREATE_NEW_CONSOLE', 0))
            return True


def age_prompt(content, waited):
    """A question that waited for the one before it says the time left as it is when it is sent, not when it was
    written (request e3: in games 31-36, 12 of 18 late answers had waited 1-13 s behind another question, the "~N s
    left" in their state that much too high). content: a string or the CLI's content blocks."""
    if not waited or waited < 1:
        return content
    fix = lambda m: f"~{max(0, int(m.group(1)) - round(waited))}{m.group(2)}"
    sub = lambda t: re.sub(r'~(\d+)(\s?s(?: of planning)? left)', fix, t)
    if isinstance(content, str):
        return sub(content)
    return [dict(b, text=sub(b['text'])) if isinstance(b, dict) and b.get('type') == 'text' else b for b in content]


_SESSION = _Session()


def json_objects(text, repair=True):
    """The complete top-level JSON objects in text, in order (strings respected). repair=False:
    only objects whose closing brace is in the text (a reply still streaming in must never have
    its unfinished object closed for it - run26 3-6: a plan cut to 2 of its 7 board units was
    installed, and the controller benched Varus, Kog'Maw and Shen).
    A reply that forgot closing brackets is repaired: an object that starts at the very start
    of a line while another is still open closes the open one first (run19: the plan line
    lacked its last brace, so every plan from 4-5 on was lost); the end of the text closes
    whatever is open (seen with GPT-6 Sol)."""
    out, stack, start, in_str, esc = [], [], None, False, False
    nl = chr(10)
    src = text or ''

    def add(chunk):
        # as written first: stripping // comments cut the https:// of every link in a string (the Strategist's
        # study requests after game 40 were lost that way)
        for c in (chunk, re.sub(r'//[^' + nl + r']*', '', chunk)):
            try:
                out.append(json.loads(c))
                return
            except ValueError:
                pass

    for i, ch in enumerate(src):
        if in_str:
            if esc:
                esc = False
            elif ch == chr(92):
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and stack:
            in_str = True
        elif repair and ch == '{' and stack and i > 0 and src[i - 1] == nl:
            # a new top-level object begins on its own line: close the one left open
            add(src[start:i].rstrip() + ''.join(reversed(stack)))
            stack, start = ['}'], i
        elif ch in '{[':
            if not stack:
                start = i
            if ch == '{' or stack:
                stack.append('}' if ch == '{' else ']')
        elif ch in '}]' and stack:
            stack.pop()
            if not stack:
                add(src[start:i + 1])
    if repair and stack and start is not None:
        add(src[start:].rstrip() + ''.join(reversed(stack)))
    return out


def _first_object_watcher(on_first):
    """Feed text chunks; calls on_first once, with the first complete JSON object."""
    buf, fired = [], [False]

    def feed(chunk):
        if fired[0] or not chunk:
            return
        buf.append(chunk)
        if '}' in chunk:
            # the deliberation comes first in a planning reply; the decision is the object after it
            objs = [o for o in json_objects(''.join(buf), repair=False)
                    if not (isinstance(o, dict) and set(o) == {'deliberation'})]
            if objs:
                fired[0] = True
                try:
                    on_first(objs[0])
                except Exception as e:
                    print('first-answer handler failed:', e)
    return feed


def _images(x):
    """One JPEG, a list of them, or None -> list."""
    if x is None:
        return []
    return [i for i in x if i] if isinstance(x, (list, tuple)) else [x]


PREP_HEAD = ('PREPARATION - sent as we enter the queue for the game. The queue takes about 1-3 minutes, then the '
             'game loads for about 1 minute; no game clock runs until then. When the game asks its first question '
             '(in round 1-2), this reply is cut off: the files you have read stay in the conversation through '
             'the 4-2 augment, planning and fight; what you have not written yet is lost.\n\n')
# what the preparation is for (the user, 2026-10-12): read everything on the comp, then write the game strategy
PREP_TASK_FIXED = ('\n\nPREPARATION TASK: read all the information on this comp in the reference files - its MetaTFT '
                   'page (metatft/comps/<ID>.md), each of its champions and their abilities (champions.md), its traits '
                   '(traits.md), the items of its builds (items.md) and the augments (augments.md). Then write your '
                   'strategy for this game as one paragraph: {"game_strategy": {"text": "..."}} (format in your '
                   'instructions). Topics: tempo, augments, items, board and the comp\'s mechanics - in your own words '
                   'and weight.')
PREP_TASK_PLAN = ('\n\nPREPARATION TASK: read all the information on this comp in the reference files - its MetaTFT '
                  'page (metatft/comps/<ID>.md), each of its champions and their abilities (champions.md), its traits '
                  '(traits.md), the items of its builds (items.md) and the augments (augments.md). Then write your game '
                  'plan: {"game_strategy": {"text": ..., "boards": ..., "items": ...}} (format in your instructions).')
PREP_TASK_MEMORY = (' Your memory above comes first: it is what you learned from your earlier games; this game\'s '
                    'strategy is yours to set with it.')
PREP_TASK_FLEX = ('\n\nPREPARATION TASK: read the reference files (README.md lists them) together with the MetaTFT comp '
                  'ranking above. Then write your overall strategy for this game as one paragraph: {"game_strategy": '
                  '{"text": "..."}} (no "comp" yet; format in your instructions), and your comp probabilities: {"strategy": '
                  '{"comps": [{"name": "<MetaTFT comp>", "p": 0.3}, ...], "other": 0.4, "commit": null}} ("commit": a comp '
                  'name instead if you already commit to one).')


def briefing_text(before_queue=False):
    """The first message of every game: the rules, traits, items, the augment tier list and
    every MetaTFT comp with its exact stats. Read in full now; the per-comp pages come later,
    when a direction is chosen. Fixed comp: only that comp (see forced_briefing).
    before_queue: sent before we queue (prepare), with no game clock; otherwise the game is
    already running (the player was started into it)."""
    comp = forced_comp()
    if comp:
        if CLIMB:
            try:                                 # counted now, not at the 2-1 augment's clock (comp_reach, x32.7)
                from tft_player.comp_reach import reach
                reach(comp.name)
            except Exception as e:
                print(f'comp reach: not counted ({e.__class__.__name__}: {e})', flush=True)
        mem = memory_block()
        first = stage_strategy_text('1')
        return (PREP_HEAD + forced_briefing() + (f'\n\n{mem}' if mem else '') + (f'\n\n{first}' if first else '')
                + (PREP_TASK_PLAN if CLIMB else PREP_TASK_FIXED).replace('<ID>', comp.id)
                + (PREP_TASK_MEMORY if mem else '') if before_queue
                else forced_briefing())
    ref = REFERENCE
    read = lambda f: (ref / f).read_text(encoding='utf-8') if (ref / f).exists() else f'({f} missing)'
    tiers = {}
    for n, a in sorted(META.augments.items()):
        if a.get('tier'):
            tiers.setdefault(a['tier'], []).append(n)
    aug = '\n'.join(f"{t}: {', '.join(tiers[t])}" for t in ('S', 'A', 'B', 'C') if t in tiers)
    fs = flex_set()                      # the climb's flex set: only its comps
    comps = sorted((c for c in (fs['comps'] if fs else META.comps.values()) if c.avg), key=lambda c: c.avg)
    rows = [f"- {c.name} [{c.id}]: avg place {c.avg:.2f}, pick {c.pick}, win {c.win:.1%}, top 4 {c.top4:.1%}, "
            f"{c.levelling}; units: {', '.join(c.units)}"
            if c.win is not None else f"- {c.name} [{c.id}]: avg place {c.avg:.2f}, {c.levelling}; units: {', '.join(c.units)}"
            for c in comps]
    first = flex_choice() if fs and before_queue else ''         # the climb's flex set: how to choose the comp
    pages = set_pages(fs) if fs else ''
    return ((PREP_HEAD + 'BRIEFING - data for this game.\n\n' if before_queue else
             'BRIEFING - data for this game, sent while it runs.\n\n') +
            '=== GAME MECHANICS ===\n' + read('game-mechanics.md') + '\n\n'
            '=== TRAITS ===\n' + read('traits.md') + '\n\n'
            '=== ITEMS ===\n' + read('items.md') + '\n\n'
            f'=== AUGMENT TIER LIST (MetaTFT expert list, S best; patch {META.patch}) ===\n' + aug + '\n\n'
            + (f"=== THIS GAME'S COMP SET (the comps this game chooses between; MetaTFT Ranked, Platinum+, last 3 "
               f"days, patch {META.patch}; best average place first; pick = times played per lobby) ===\n"
               if fs else
               f'=== METATFT COMPS (Ranked, Platinum+, last 3 days, patch {META.patch}; best average place first; '
               'pick = times played per lobby) ===\n') + '\n'.join(rows) + '\n\n'
            + (pages + '\n\n' if pages else
               'Each comp also has a detailed MetaTFT page (positioning, levelling and rolls, carousel priority, '
               'unit and item stats, traits, augments). When a comp becomes your target, its page is sent to you '
               'as text. ') +
            'More text data: the reference files in your working directory (metatft/comps/<id>.md).' +
            (f'\n\n{memory_block()}' if before_queue and memory_block() else '') +
            (f'\n\n{first}' if first else '') +
            ((PREP_TASK_FLEX.replace('the MetaTFT comp ranking above', "this game's comp set above") if fs else
              PREP_TASK_FLEX) + (PREP_TASK_MEMORY if memory_block() else '') if before_queue
             else '\n\nReply with ONLY {"ready": true}.'))


_FORCED_TEXT = {}


def set_pages(fs):
    """The climb's flex set, before a commit (the owner, 2026-11-07: "before commitment the player should see the
    information about the 3 comps, and decide on what could make switch to which comp, like the MetaTFT data from each
    comp"): every set comp's MetaTFT page as text - the transcription the fixed-comp briefing uses, else our own text
    page of it."""
    out = []
    for c in fs['comps']:
        text = ''
        try:
            from tft_player.metatft_shots import comp_brief
            text = comp_brief(c.name)
        except Exception as e:
            print(f'MetaTFT comp transcription of {c.name} failed ({e}): our text page instead', flush=True)
        if not text:
            f = REFERENCE / 'metatft' / 'comps' / f'{c.id}.md'
            text = f.read_text(encoding='utf-8') if f.exists() else ''
        if text:
            out.append(f"=== {c.name} [{c.id}]: its MetaTFT page as text ===\n{text.strip()}")
    return '\n\n'.join(out)


def comp_curve_line(comp):
    """The comp's own levelling line, as MetaTFT text fetched for the briefing (cached)."""
    m = re.search(r'^Levelling[^\n]*$', _FORCED_TEXT.get(comp.id, ''), re.M)
    return m.group(0) if m else ''


def forced_briefing():
    """Fixed comp: one line and the text transcription of its MetaTFT page."""
    comp = forced_comp()
    try:
        # The complete transcription, refreshed after 48 h.
        from tft_player.metatft_shots import comp_brief
        text = ("The comp's MetaTFT page (tabs Options & Quick Start, Units & Items, Pro Augments and Tips) "
                "as text:\n\n" + comp_brief(comp.name))
    except Exception as e:
        print(f'MetaTFT comp transcription failed ({e}): page text unavailable')
        text = ''
    _FORCED_TEXT[comp.id] = text
    from tft_player.trait_text import choices_text
    traits = [t for u in comp.units for t in (CATALOG.unit(u).traits if CATALOG.unit(u) else ())]
    choices = choices_text(traits)
    if choices:
        # what each option does, and the model's choice made now - at the choice screen it has seconds
        text += ('\n\nTrait choices on the target final board (the game shows only the names on the cards):\n' + choices +
                 '\nThe choice screen has a short timer, so the picks are asked now. End your reply with '
                 '{"trait_choices": {"<trait>": ["<first pick>", "<second pick>"]}} using the names on the cards; '
                 'they are shown to you again when the choice appears.')
    return f'Target comp for this game: {comp.name} ({comp.id}).' + (f'\n\n{text}' if text else '')


STOPPED = False          # set when we are eliminated: ask_model answers None at once


class ModelStopped(Exception):
    """A question that waited for the conversation until the game was over, or until it was no longer wanted (its
    round passed): it is not sent (benchmark game 20261020-151221: Argon's ~400 s answers queued in the game went on
    asking after the elimination, and the process waited on them)."""


_WANTED = threading.local()          # the asking thread's still_wanted (ask_model)


def _still_wanted():
    """The question of this thread is still worth sending (ask_model's still_wanted; none given: yes)."""
    f = getattr(_WANTED, 'fn', None)
    try:
        return f is None or bool(f())
    except Exception:
        return True


CALL_DIR = None          # set by the player: every model call's screenshots and raw result go here (benchmark data)
_CALL_N = [0]


def _save_text(n, name, text):
    """A diagnostic file next to call n's inputs (calls/NNNN-<name>)."""
    if CALL_DIR is None or not n:
        return
    try:
        (CALL_DIR / f'{n:04d}-{name}').write_text(text, encoding='utf-8')
    except OSError:
        pass


def _save_call_inputs(prompt, imgs):
    """Save the screenshots (and prompt) of a model call as calls/NNNN-*.jpg; returns the call number."""
    if CALL_DIR is None:
        return None
    try:
        _CALL_N[0] += 1
        n = _CALL_N[0]
        CALL_DIR.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(imgs if os.environ.get('TFT_NO_SCREENSHOTS') != '1' else ()):   # --no-screenshots
            (CALL_DIR / f'{n:04d}-{i}.jpg').write_bytes(img)
        (CALL_DIR / f'{n:04d}-prompt.txt').write_text(prompt, encoding='utf-8')
        _save_strategies()
        return n
    except Exception:
        return None


def _compact_state(text, seen):
    """Omit only exact repeats of STATE sections in this conversation, naming every omission."""
    def state(match):
        sections = []
        for line in match.group()[len('\nSTATE\n'):].splitlines(keepends=True):
            if sections and (not line.strip() or line[:1].isspace() or line.startswith('- ')):
                sections[-1] += line
            else:
                sections.append(line)
        unchanged, full = [], []
        # Compare against what preceded this message, including when a key occurs twice in one STATE.
        previous = dict(seen)
        for section in sections:
            key = re.split(r'[:(]', section.splitlines()[0], maxsplit=1)[0].strip()
            if not key:
                full.append(section)
                continue
            if previous.get(key) == section.rstrip('\n'):
                unchanged.append(key)
            else:
                full.append(section)
            seen[key] = section.rstrip('\n')
        notice = ('Unchanged since you last saw them (not repeated): ' + ', '.join(unchanged) + '\n'
                  if unchanged else '')
        return '\nSTATE\n' + notice + ''.join(full)
    return _STATE_BLOCK.sub(state, text)


def _outgoing(content, session=None, fresh=False, call_n=None):
    """The common send gate for every backend and speed check. Evidence is saved before this gate.

    Only text reaches a model. STATE memory belongs to the actual backend conversation and is
    updated inside its lock, after queued questions have been checked and conversation turnover applied.
    """
    seen = {} if session is None or fresh else getattr(session, '_state_sent', {})
    if session is not None:
        session._state_sent = seen
    out = [dict(c, text=_compact_state(c.get('text') or '', seen))
           for c in content if c.get('type') == 'text']
    sent = '\n'.join(c['text'] for c in out)
    if CALL_DIR is not None and call_n:
        try:
            full = (CALL_DIR / f'{call_n:04d}-prompt.txt').read_text(encoding='utf-8')
            if sent != full:
                _save_text(call_n, 'sent.txt', sent)
        except OSError:
            pass
    return out


def _claude_cli_call(exe, prompt, image_jpeg=None, on_first=None, timeout=90):
    """Claude through the Claude Code CLI (your Claude subscription), as a new
    message in the game's open conversation. The screenshot is an inline image
    block; a short system prompt and no tools keep answers at ~3-6s."""
    import base64
    content = []
    for img in _images(image_jpeg):
        content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                                    'data': base64.standard_b64encode(img).decode()}})
    content.append({'type': 'text', 'text': prompt})
    n = _save_call_inputs(prompt, _images(image_jpeg))
    try:
        return _SESSION.ask(exe, content, timeout=timeout, on_first=on_first, call_n=n)
    finally:
        if n:
            _SESSION.last_meta = dict(getattr(_SESSION, 'last_meta', None) or {}, call=n)


class _ORSession:
    """ONE OpenRouter conversation per game (any model there, e.g. GPT-6 Sol), with
    the same system prompt, the screenshot on each question (only the latest kept
    in the history), streaming (on_first gets the decision as soon as it is
    written) and two tools to read/search the reference files."""

    TOOLS = [
        {'type': 'function', 'function': {
            'name': 'read_file', 'description': 'Read a reference file (path relative to the reference folder).',
            'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}},
        {'type': 'function', 'function': {
            'name': 'grep', 'description': 'Search the reference files for a regular expression; returns matching lines.',
            'parameters': {'type': 'object', 'properties': {'pattern': {'type': 'string'}}, 'required': ['pattern']}}},
    ]

    def __init__(self):
        self.messages = None
        self._lock = threading.Lock()

    def _tool(self, name, args):
        base = REFERENCE.resolve()
        if name == 'read_file':
            p = (base / str(args.get('path', ''))).resolve()
            if base not in p.parents and p != base or not p.is_file():
                return 'no such file'
            return p.read_text(encoding='utf-8')[:20000]
        if name == 'grep':
            try:
                rx = re.compile(str(args.get('pattern', '')), re.I)
            except re.error as e:
                return f'bad pattern: {e}'
            hits = []
            for f in sorted(base.rglob('*.md')):
                for line in f.read_text(encoding='utf-8').splitlines():
                    if rx.search(line):
                        hits.append(f'{f.relative_to(base).as_posix()}: {line[:300]}')
                        if len(hits) >= 60:
                            return '\n'.join(hits)
            return '\n'.join(hits) or 'no match'
        return 'unknown tool'

    def _stream(self, on_text):
        import urllib.error
        import urllib.request
        # max_tokens as on the Anthropic path: without it OpenRouter reserves the model's whole output length
        # against the key's credit and refuses the call (402) long before the credit is used up
        host = api_host()
        base, _, label = API_HOSTS[host]
        body = {'model': OR_MODEL, 'messages': self.messages, 'stream': True, 'tools': self.TOOLS}
        if host == 'openai':                       # OpenAI's names for the same two settings
            body['max_completion_tokens'] = 16000
            if OR_EFFORT:
                body['reasoning_effort'] = OR_EFFORT
        else:
            body['max_tokens'] = 16000
            if OR_EFFORT:
                body['reasoning'] = {'effort': OR_EFFORT}
        req = urllib.request.Request(base + '/chat/completions', json.dumps(body).encode(),
                                     {'Authorization': 'Bearer ' + api_key(),
                                      'Content-Type': 'application/json'})
        text, calls = [], {}
        try:
            r = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'{label} {e.code}: {e.read().decode("utf-8", "replace")[:300]}') from None
        with r:
            for raw in r:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:') or line == 'data: [DONE]':
                    continue
                try:
                    ch = json.loads(line[5:]).get('choices') or [{}]
                except ValueError:
                    continue
                d = ch[0].get('delta') or {}
                if d.get('content'):
                    text.append(d['content'])
                    on_text(d['content'])
                for tc in d.get('tool_calls') or []:
                    c = calls.setdefault(tc.get('index', 0), {'id': '', 'name': '', 'args': ''})
                    c['id'] = tc.get('id') or c['id']
                    fn = tc.get('function') or {}
                    c['name'] += fn.get('name') or ''
                    c['args'] += fn.get('arguments') or ''
        return ''.join(text), [calls[k] for k in sorted(calls)]

    def ask(self, content, on_first=None, call_n=None, fresh=False, raw=False):
        with self._lock:
            if STOPPED or not _still_wanted():
                raise ModelStopped()
            content, new = (content, fresh) if fresh or raw else _stage_turnover(content, call_n)
            content = _outgoing(content, self, fresh=new or self.messages is None, call_n=call_n)
            if self.messages is None or new:
                self.messages = [{'role': 'system', 'content': opening_prompt()}]
            # keep only the newest screenshot in the history (size and cost)
            for m in self.messages:
                if isinstance(m.get('content'), list):
                    m['content'] = [c for c in m['content'] if c.get('type') != 'image_url'] or ''
            self.messages.append({'role': 'user', 'content': content})
            feed = _first_object_watcher(on_first) if on_first else (lambda t: None)
            t0, turns = time.time(), 0
            while True:
                turns += 1
                text, calls = self._stream(feed)
                msg = {'role': 'assistant', 'content': text}
                if calls:
                    msg['tool_calls'] = [{'id': c['id'], 'type': 'function',
                                          'function': {'name': c['name'], 'arguments': c['args']}} for c in calls]
                self.messages.append(msg)
                if not calls or turns > 6:
                    self.last_meta = {'num_turns': turns, 'duration_ms': round((time.time() - t0) * 1000)}
                    return text
                for c in calls:
                    try:
                        args = json.loads(c['args'] or '{}')
                    except ValueError:
                        args = {}
                    self.messages.append({'role': 'tool', 'tool_call_id': c['id'], 'content': self._tool(c['name'], args)})


_OR = _ORSession()


class _CodexSession:
    """One of the game's two Codex CLI conversations (ChatGPT login). Each question is one `codex exec` run that
    resumes the conversation by its id (never "the last session": the user's own Codex sessions stay apart). The
    game's system prompt replaces Codex's own instructions; the user's config is not loaded (no plugins or MCP
    servers); the model reads the reference files with its shell in a read-only sandbox. Codex prints each message
    whole, not as it is written: the decision is acted on when its message is complete."""

    def __init__(self):
        self.thread_id = None
        self._proc = None
        self._lock = threading.Lock()
        self._dispatch = threading.Lock()  # includes limit handling before the next queued call
        self.cut = False
        self.session_id = None               # = thread_id: lets a terminal resume it
        self.last_meta = None
        self._instr = None

    def _base(self, exe):
        cmd = [exe, 'exec', '--json', '--ignore-user-config', '--ignore-rules', '--skip-git-repo-check',
               '-s', 'read-only', '-c', 'approval_policy="never"', '-m', CODEX_MODEL,
               '-c', f'model_reasoning_effort="{CODEX_EFFORT}"',
               '-c', f'model_instructions_file="{self._instr.as_posix()}"']
        sb = codex_sandbox()
        if sb:
            cmd += ['-c', f'windows.sandbox="{sb}"']
        tier = codex_service_tier()
        if tier:
            cmd += ['-c', f'service_tier="{tier}"']
        return cmd

    def close(self):
        with self._lock:
            self._state_sent = {}
            self.thread_id = None

    def interrupt(self):
        """Cut the reply in flight short: its process ends; the conversation goes on from what it holds."""
        p = self._proc
        if p is not None and p.poll() is None:
            self.cut = True
            p.kill()

    def ask(self, exe, content, timeout=90, on_first=None, call_n=None, fresh=False, raw=False):
        # Queued calls on this old session must also follow the game's fallback after a limit.
        with self._dispatch:
            if _GAME_FALLBACK:
                return _SESSION.ask(claude_cli(), content, timeout=timeout, on_first=on_first,
                                    call_n=call_n, fresh=fresh, raw=raw)
            try:
                return self._ask(exe, content, timeout, on_first, call_n, fresh, raw)
            except RuntimeError as e:
                from agent import backends
                if not backends.LIMIT_RX.search(str(e)):
                    raise
                _switch_backend(str(e), call_n)
                return _SESSION.ask(claude_cli(), content, timeout=timeout, on_first=on_first,
                                    call_n=call_n, raw=raw)

    def _ask(self, exe, content, timeout=90, on_first=None, call_n=None, fresh=False, raw=False):
        with self._lock:
            if STOPPED or not _still_wanted():
                raise ModelStopped()
            content, new = (content, fresh) if fresh or raw else _stage_turnover(content, call_n)
            content = _outgoing(content, self, fresh=new or self.thread_id is None or self._instr is None, call_n=call_n)
            if new or self.thread_id is None or self._instr is None:
                self.thread_id = None
                self._instr = Path(tempfile.gettempdir()) / f'tft-codex-instructions-{os.getpid()}.md'
                self._instr.write_text(opening_prompt(), encoding='utf-8')
            text = next((c.get('text') or '' for c in content if c.get('type') == 'text'), '')
            imgs = [c['jpeg'] for c in content if c.get('type') == 'jpeg']
            d = Path(tempfile.mkdtemp(prefix='tft-codex-'))
            files = []
            for i, img in enumerate(imgs):
                f = d / f'screen-{i}.jpg'
                f.write_bytes(img)
                files.append(str(f))
            cmd = self._base(exe)
            if self.thread_id:
                cmd += ['resume', self.thread_id]
            for f in files:
                cmd += ['-i', f]
            # -i accepts multiple paths: terminate options so stdin '-' is the prompt, never an image (e126).
            cmd += ['--', '-']
            feed = _first_object_watcher(on_first) if on_first else None
            LIVE.update(what=_question_line(content), t0=time.time(), text='', took=None)
            self.cut, said, usage, turns, err, proc = False, [], None, 1, None, None
            raw, began = [], time.time()
            cwd = player_cwd()
            try:
                self._proc = proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                     stderr=subprocess.DEVNULL, cwd=cwd,
                                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                proc.stdin.write(text.encode('utf-8'))
                proc.stdin.close()
                timer = threading.Timer(timeout, proc.kill)
                timer.start()
                try:
                    for line in proc.stdout:
                        if len(raw) < 400:
                            raw.append(line.decode('utf-8', 'replace')[:2000])
                        try:
                            ev = json.loads(line.decode('utf-8', 'replace'))
                        except ValueError:
                            continue
                        kind = ev.get('type')
                        item = ev.get('item') or {}
                        if kind == 'thread.started':
                            self.thread_id = self.session_id = ev.get('thread_id') or self.thread_id
                        elif kind == 'item.completed' and item.get('type') == 'agent_message':
                            chunk = (item.get('text') or '') + '\n'
                            said.append(chunk)
                            LIVE['text'] += chunk
                            if feed:
                                feed(chunk)
                        elif kind == 'item.completed' and item.get('type') == 'command_execution':
                            turns += 1                 # a file read or search
                        elif kind == 'turn.completed':
                            usage = ev.get('usage')
                        elif kind in ('turn.failed', 'error'):
                            err = (ev.get('error') or {}).get('message') or ev.get('message') or err
                    proc.wait(timeout=10)
                finally:
                    timer.cancel()
            except OSError as e:
                if not self.cut:             # cut while its question was still being written: a cut, not a fault
                    raise RuntimeError(f'Codex did not start: {e}')
            finally:
                self._proc = None
                for f in files:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
                try:
                    d.rmdir()
                except OSError:
                    pass
            LIVE['took'] = time.time() - LIVE['t0']
            answer = ''.join(said).strip()
            # A failed turn is an error even when it emitted some text; never recover an older answer over it.
            error = err or ('Codex turn interrupted by harness' if self.cut else None)
            if error or not answer:
                _save_text(call_n, 'codex-stdout.txt', ''.join(raw))
                _save_text(call_n, 'codex-error.txt', error or f'no agent_message (exit code {proc.returncode})')
            if err:
                raise RuntimeError(err)
            if self.cut and not answer:
                raise RuntimeError(error)
            if not answer and not self.cut:
                # run 20261018-092528 call 9: the turn completed with an answer in Codex's own session file, but none
                # reached us - take it from there, and keep what we did read for the diagnosis
                _save_text(call_n, 'codex-stdout.txt', ''.join(raw))
                answer = codex_rollout_answer(self.thread_id, began) or ''
                if not answer:
                    raise RuntimeError(err or f'Codex ended without an answer (exit code {proc.returncode})')
                print(f'Codex answer read from its session file (none on its output): {answer[:80]}', flush=True)
            self.last_meta = {'backend': 'codex', 'num_turns': turns, 'duration_ms': round(LIVE['took'] * 1000), 'usage': usage,
                              'session_id': self.thread_id}
            if CALL_DIR is not None:
                try:
                    (CALL_DIR / f'{call_n or 0:04d}-result.json').write_text(
                        json.dumps(dict(self.last_meta, result=answer), ensure_ascii=False), encoding='utf-8')
                except Exception:
                    pass
            return answer

    def open_terminal(self, exe, locked=True):
        """After the game: resume this conversation in a console window (the user's own Codex settings)."""
        import contextlib
        with (self._lock if locked else contextlib.nullcontext()):
            if not self.thread_id:
                return False
            cmd = [exe, 'resume', self.thread_id, '-s', 'read-only']
            d = str(REFERENCE) if REFERENCE.exists() else tempfile.gettempdir()
            try:
                subprocess.Popen(cmd, cwd=d, creationflags=TERMINAL_FLAGS)
            except OSError:
                subprocess.Popen(cmd, cwd=d, creationflags=getattr(subprocess, 'CREATE_NEW_CONSOLE', 0))
            return True


_CODEX = _CodexSession()


def _codex_call(exe, prompt, image_jpeg=None, on_first=None, timeout=90):
    content = [{'type': 'text', 'text': prompt}] + [{'type': 'jpeg', 'jpeg': img} for img in _images(image_jpeg)]
    n = _save_call_inputs(prompt, _images(image_jpeg))
    try:
        return _CODEX.ask(exe, content, timeout=timeout, on_first=on_first, call_n=n)
    finally:
        if n:
            _CODEX.last_meta = dict(_CODEX.last_meta or {}, call=n)


GROK_MODEL = os.environ.get('TFT_GROK_MODEL', 'grok-4.7')
GROK_EFFORT = os.environ.get('TFT_GROK_EFFORT', 'medium')


def grok_cli():
    """Grok Build's CLI (x.ai/cli installs it in ~/.grok/bin), or None."""
    p = Path.home() / '.grok' / 'bin' / 'grok.exe'
    return str(p) if p.exists() else shutil.which('grok')


class _GrokSession:
    """Grok on the user's Grok login, through Grok Build's agent over ACP (`grok agent stdio`, JSON-RPC on its stdin
    and stdout): one agent process for the game, two ACP sessions per game (the second starts at 4-3,
    as the other CLIs' conversations do). The screenshot goes in as an image block; ACP has no system prompt, so
    the game's system prompt leads a session's first message. The reply streams in (agent_message_chunk): the
    decision is acted on the moment it is written. The agent works in the reference folder: a read or a search there
    is allowed, anything else it asks permission for (a write, a command, the web) is refused."""

    ALLOW = ('read', 'search')               # ACP tool kinds answered with "allow"

    def __init__(self):
        self._proc, self._q, self._id, self._key = None, None, 0, None
        self._lock, self._write = threading.Lock(), threading.Lock()
        self.session_id = self.thread_id = None
        self.last_meta, self.cut, self._pending = None, False, None

    def _start(self, exe):
        import queue
        self.stop()
        cmd = [exe, 'agent', '-m', GROK_MODEL, '--reasoning-effort', GROK_EFFORT, '--no-leader', 'stdio']
        cwd = str(REFERENCE) if REFERENCE.exists() else tempfile.gettempdir()
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      cwd=cwd, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self._q, self._key, self.session_id = queue.Queue(), (GROK_MODEL, GROK_EFFORT), None
        q, out = self._q, self._proc.stdout

        def read():
            for line in out:
                try:
                    q.put(json.loads(line.decode('utf-8', 'replace')))
                except ValueError:
                    pass
            q.put(None)                       # the agent ended
        threading.Thread(target=read, daemon=True).start()
        self._rpc('initialize', {'protocolVersion': 1, 'clientCapabilities': {
            'fs': {'readTextFile': False, 'writeTextFile': False}, 'terminal': False}}, 60)

    def stop(self):
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            p.kill()

    def _send(self, obj):
        with self._write:
            self._proc.stdin.write((json.dumps(obj) + '\n').encode('utf-8'))
            self._proc.stdin.flush()

    def _rpc(self, method, params, timeout, on_update=None):
        """A request to the agent and its result. Meanwhile its session updates go to on_update and its own
        requests are answered (permissions by tool kind, anything else with an error)."""
        import queue
        self._id += 1
        rid = self._id
        self._pending = rid
        self._send({'jsonrpc': '2.0', 'id': rid, 'method': method, 'params': params})
        end = time.time() + timeout
        while True:
            try:
                m = self._q.get(timeout=max(0.05, min(1.0, end - time.time())))
            except queue.Empty:
                if time.time() > end:
                    raise TimeoutError(f'Grok gave no answer to {method} in {timeout:.0f} s')
                continue
            if m is None:
                raise RuntimeError('the Grok agent ended')
            if m.get('id') == rid and ('result' in m or 'error' in m):
                if 'error' in m:
                    raise RuntimeError(f"Grok {method}: {(m['error'] or {}).get('message')}")
                return m['result'] or {}
            if m.get('method') == 'session/update':
                if on_update:
                    on_update((m.get('params') or {}).get('update') or {})
            elif m.get('method') and 'id' in m:
                self._send({'jsonrpc': '2.0', 'id': m['id'], **self._answer(m)})

    def _answer(self, m):
        if m['method'] == 'session/request_permission':
            p = m.get('params') or {}
            ok = ((p.get('toolCall') or {}).get('kind') in self.ALLOW)
            want = ('allow_once', 'allow_always') if ok else ('reject_once', 'reject_always')
            opt = next((o for o in p.get('options') or [] if o.get('kind') in want), None)
            outcome = {'outcome': 'selected', 'optionId': opt['optionId']} if opt else {'outcome': 'cancelled'}
            return {'result': {'outcome': outcome}}
        return {'error': {'code': -32601, 'message': f"{m['method']} is not offered by this client"}}

    def close(self):
        with self._lock:
            self._state_sent = {}
            self.session_id = self.thread_id = None

    def interrupt(self):
        """Cut the reply in flight short (session/cancel): the turn ends; the session keeps what it read."""
        if self._proc is not None and self._proc.poll() is None and self.session_id:
            self.cut = True
            try:
                self._send({'jsonrpc': '2.0', 'method': 'session/cancel', 'params': {'sessionId': self.session_id}})
            except OSError:
                pass

    def ask(self, exe, content, timeout=90, on_first=None, call_n=None, fresh=False, raw=False):
        import base64
        with self._lock:
            if STOPPED or not _still_wanted():
                raise ModelStopped()
            content, new = (content, fresh) if fresh or raw else _stage_turnover(content, call_n)
            if self._proc is None or self._proc.poll() is not None or self._key != (GROK_MODEL, GROK_EFFORT):
                self._start(exe)
            content = _outgoing(content, self, fresh=new or not self.session_id, call_n=call_n)
            text = next((c.get('text') or '' for c in content if c.get('type') == 'text'), '')
            if new or not self.session_id:
                cwd = str(REFERENCE) if REFERENCE.exists() else tempfile.gettempdir()
                self.session_id = self.thread_id = self._rpc('session/new', {'cwd': cwd, 'mcpServers': []}, 60)['sessionId']
                text = opening_prompt() + '\n\n' + text      # ACP has no system prompt: it leads the session
            prompt = [{'type': 'image', 'mimeType': 'image/jpeg', 'data': base64.standard_b64encode(c['jpeg']).decode()}
                      for c in content if c.get('type') == 'jpeg'] + [{'type': 'text', 'text': text}]
            feed = _first_object_watcher(on_first) if on_first else None
            LIVE.update(what=_question_line(content), t0=time.time(), text='', took=None)
            said, turns = [], [1]
            self.cut = False

            def update(u):
                kind = u.get('sessionUpdate')
                if kind == 'agent_message_chunk':
                    chunk = (u.get('content') or {}).get('text') or ''
                    said.append(chunk)
                    LIVE['text'] += chunk
                    if feed:
                        feed(chunk)
                elif kind == 'tool_call':
                    turns[0] += 1                       # a file read or search
            try:
                r = self._rpc('session/prompt', {'sessionId': self.session_id, 'prompt': prompt}, timeout, update)
            except TimeoutError:
                self.interrupt()
                raise
            LIVE['took'] = time.time() - LIVE['t0']
            answer = ''.join(said).strip()
            if not answer and not self.cut:
                raise RuntimeError(f"Grok ended without an answer (stop reason {r.get('stopReason')})")
            self.last_meta = {'num_turns': turns[0], 'duration_ms': round(LIVE['took'] * 1000),
                              'usage': (r.get('_meta') or {}).get('usage') or r.get('usage'),
                              'session_id': self.session_id, 'stop_reason': r.get('stopReason')}
            if CALL_DIR is not None:
                try:
                    (CALL_DIR / f'{call_n or 0:04d}-result.json').write_text(
                        json.dumps(dict(self.last_meta, result=answer), ensure_ascii=False), encoding='utf-8')
                except Exception:
                    pass
            return answer

    def open_terminal(self, exe, locked=True):
        """After the game: this stage's session in Grok Build's own window, to ask it about its play."""
        import contextlib
        with (self._lock if locked else contextlib.nullcontext()):
            if not self.session_id:
                return False
            d = str(REFERENCE) if REFERENCE.exists() else tempfile.gettempdir()
            try:
                subprocess.Popen([exe, '--resume', self.session_id], cwd=d, creationflags=TERMINAL_FLAGS)
            except OSError:
                return False
            return True


_GROK = _GrokSession()


def _grok_call(exe, prompt, image_jpeg=None, on_first=None, timeout=90):
    content = [{'type': 'text', 'text': prompt}] + [{'type': 'jpeg', 'jpeg': img} for img in _images(image_jpeg)]
    n = _save_call_inputs(prompt, _images(image_jpeg))
    try:
        return _GROK.ask(exe, content, timeout=timeout, on_first=on_first, call_n=n)
    finally:
        if n:
            _GROK.last_meta = dict(_GROK.last_meta or {}, call=n)


def _use_grok(mid, effort):
    """Grok on the user's Grok login, through Grok Build (installed and signed in with `grok login`)."""
    global GROK_MODEL, GROK_EFFORT
    exe = grok_cli()
    if not exe:
        return 'Grok Build was not found: install it from x.ai/cli and sign in with "grok login"'
    try:
        out = subprocess.run([exe, 'models'], capture_output=True, timeout=60,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        said = (out.stdout + out.stderr).decode('utf-8', 'replace')
    except (OSError, subprocess.TimeoutExpired) as e:
        return f'Grok Build did not start: {e}'
    if 'not authenticated' in said.lower() or 'logged in' not in said.lower():
        return 'Grok Build is not signed in: run "grok login"'
    os.environ['TFT_COACH_BACKEND'] = 'grok'
    os.environ['TFT_GROK_MODEL'] = GROK_MODEL = mid or GROK_MODEL
    if effort:
        os.environ['TFT_GROK_EFFORT'] = GROK_EFFORT = effort
    return None


AGY_MODEL = os.environ.get('TFT_AGY_MODEL', 'gemini-3.8-flash')
AGY_EFFORT = os.environ.get('TFT_AGY_EFFORT', 'medium')


def agy_cli():
    """The Antigravity CLI (antigravity.google/cli installs it in %LOCALAPPDATA%\\agy\\bin), or None."""
    p = Path(os.environ.get('LOCALAPPDATA', '')) / 'agy' / 'bin' / 'agy.exe'
    return str(p) if p.exists() else shutil.which('agy')


def agy_model():
    """The model id the CLI takes: Gemini's effort is part of its name there (gemini-3.8-flash-medium)."""
    return AGY_MODEL if re.search(r'-(low|medium|high)$', AGY_MODEL) else f'{AGY_MODEL}-{AGY_EFFORT}'


class _AgySession:
    """Gemini on the user's Google AI login, through the Antigravity CLI (it replaced Gemini CLI for personal
    accounts, 2026-06): two `agy -p` conversations per game, questions written to its stdin one per line
    (stream-json) and its reply streamed back (text_delta), so the decision is acted on as it is written. Its input
    takes text only: the screenshot is saved as a file the model views with its own file tool. It has no system
    prompt option, so the game's system prompt leads a conversation's first message. It works in the reference
    folder; tools that need a permission headless mode cannot ask for are denied (the user's Antigravity settings
    allow the read-only ones)."""

    def __init__(self):
        self._proc, self._q, self._lock = None, None, threading.Lock()
        self.session_id = self.thread_id = None
        self.last_meta, self.cut, self._dir, self._n = None, False, None, 0

    def _start(self, exe):
        import queue
        self.stop()
        self._dir = Path(tempfile.mkdtemp(prefix='tft-agy-'))
        cwd = str(REFERENCE) if REFERENCE.exists() else str(self._dir)
        cmd = [exe, '--model', agy_model(), '--add-dir', str(self._dir), '--input-format', 'stream-json',
               '--output-format', 'stream-json', '-p', '']
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      cwd=cwd, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self._q, out = queue.Queue(), self._proc.stdout
        q = self._q

        def read():
            for line in out:
                try:
                    q.put(json.loads(line.decode('utf-8', 'replace')))
                except ValueError:
                    pass
            q.put(None)
        threading.Thread(target=read, daemon=True).start()
        self.session_id = self.thread_id = None

    def stop(self):
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            p.kill()

    def close(self):
        with self._lock:
            self._state_sent = {}
            self.stop()                       # the next question starts a new conversation

    def interrupt(self):
        """Cut the reply in flight short: the process ends (the next question starts a new conversation)."""
        if self._proc is not None and self._proc.poll() is None:
            self.cut = True
            self.stop()

    def ask(self, exe, content, timeout=90, on_first=None, call_n=None, fresh=False, raw=False):
        import queue
        with self._lock:
            if STOPPED or not _still_wanted():
                raise ModelStopped()
            content, new = (content, fresh) if fresh or raw else _stage_turnover(content, call_n)
            content = _outgoing(content, self, fresh=new or self._proc is None or self._proc.poll() is not None, call_n=call_n)
            text = next((c.get('text') or '' for c in content if c.get('type') == 'text'), '')
            if new or self._proc is None or self._proc.poll() is not None:
                self._start(exe)
                text = opening_prompt() + '\n\n' + text      # no system prompt option: it leads the conversation
            files = []
            for c in content:
                if c.get('type') == 'jpeg':
                    self._n += 1
                    f = self._dir / f'screen-{self._n}.jpg'
                    f.write_bytes(c['jpeg'])
                    files.append(f)
            if files:
                text += ('\n\nThe live game screen right now is the image file ' + ', '.join(str(f) for f in files)
                         + ': view it.')
            feed = _first_object_watcher(on_first) if on_first else None
            LIVE.update(what=_question_line(content), t0=time.time(), text='', took=None)
            said, turns, res, self.cut = [], 1, None, False
            self._proc.stdin.write((json.dumps({'event': 'user', 'message': {'role': 'user', 'content': [
                {'type': 'text', 'text': text}]}}) + '\n').encode('utf-8'))
            self._proc.stdin.flush()
            end = time.time() + timeout
            while res is None:
                try:
                    ev = self._q.get(timeout=max(0.05, min(1.0, end - time.time())))
                except queue.Empty:
                    if time.time() > end:
                        self.interrupt()
                        raise TimeoutError(f'Gemini gave no answer in {timeout:.0f} s')
                    continue
                if ev is None:
                    if self.cut:
                        break
                    raise RuntimeError('the Antigravity CLI ended')
                su = ev.get('step_update') or {}
                if ev.get('event') == 'init':
                    self.session_id = self.thread_id = ev.get('conversation_id')
                elif su.get('text_delta') and su.get('step_type') == 'agent_response':
                    chunk = su['text_delta']
                    said.append(chunk)
                    LIVE['text'] += chunk
                    if feed:
                        feed(chunk)
                elif su.get('step_type') == 'tool' and su.get('state') == 'DONE':
                    turns += 1                         # a file view or read
                elif ev.get('event') == 'result':
                    res = ev.get('result') or {}
            for f in files:
                try:
                    f.unlink()
                except OSError:
                    pass
            LIVE['took'] = time.time() - LIVE['t0']
            answer = ''.join(said).strip() or (res or {}).get('response', '').strip()
            if not answer and not self.cut:
                denied = (res or {}).get('denied_actions')
                raise RuntimeError(f"Gemini ended without an answer ({(res or {}).get('status')}"
                                   + (f'; denied: {json.dumps(denied)[:200]}' if denied else '') + ')')
            self.last_meta = {'num_turns': turns, 'duration_ms': round(LIVE['took'] * 1000),
                              'usage': (res or {}).get('usage'), 'session_id': self.session_id}
            if CALL_DIR is not None:
                try:
                    (CALL_DIR / f'{call_n or 0:04d}-result.json').write_text(
                        json.dumps(dict(self.last_meta, result=answer), ensure_ascii=False), encoding='utf-8')
                except Exception:
                    pass
            return answer

    def open_terminal(self, exe, locked=True):
        """After the game: this stage's conversation in the Antigravity CLI's own window."""
        import contextlib
        with (self._lock if locked else contextlib.nullcontext()):
            if not self.session_id:
                return False
            d = str(REFERENCE) if REFERENCE.exists() else tempfile.gettempdir()
            try:
                subprocess.Popen([exe, '--conversation', self.session_id], cwd=d, creationflags=TERMINAL_FLAGS)
            except OSError:
                return False
            return True


_AGY = _AgySession()


def _agy_call(exe, prompt, image_jpeg=None, on_first=None, timeout=90):
    content = [{'type': 'text', 'text': prompt}] + [{'type': 'jpeg', 'jpeg': img} for img in _images(image_jpeg)]
    n = _save_call_inputs(prompt, _images(image_jpeg))
    try:
        return _AGY.ask(exe, content, timeout=timeout, on_first=on_first, call_n=n)
    finally:
        if n:
            _AGY.last_meta = dict(_AGY.last_meta or {}, call=n)


def _use_agy(mid, effort):
    """Gemini on the user's Google AI login, through the Antigravity CLI (installed and signed in)."""
    global AGY_MODEL, AGY_EFFORT
    exe = agy_cli()
    if not exe:
        return 'the Antigravity CLI was not found: install it from antigravity.google/cli and sign in (run agy once)'
    try:
        out = subprocess.run([exe, 'models'], capture_output=True, timeout=90,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        said = (out.stdout + out.stderr).decode('utf-8', 'replace')
    except (OSError, subprocess.TimeoutExpired) as e:
        return f'the Antigravity CLI did not start: {e}'
    if 'sign in' in said.lower():
        return 'the Antigravity CLI is not signed in: run agy once and sign in'
    os.environ['TFT_COACH_BACKEND'] = 'agy'
    os.environ['TFT_AGY_MODEL'] = AGY_MODEL = mid or AGY_MODEL
    if effort:
        os.environ['TFT_AGY_EFFORT'] = AGY_EFFORT = effort
    if agy_model() not in said:
        return f'the Antigravity CLI offers no model {agy_model()} on this account'
    return None


CURSOR_MODEL = os.environ.get('TFT_CURSOR_MODEL', 'muse-wonder-1.0')
CURSOR_EFFORT = os.environ.get('TFT_CURSOR_EFFORT', 'default')


def cursor_cli():
    """The newest version folder of Cursor's agent CLI (%LOCALAPPDATA%\\cursor-agent\\versions\\<version>, holding
    node.exe and index.js: run directly, not through its PowerShell launcher, ~1 s less a call), or None. Only a
    plain version folder: its updater leaves '<version>.previous' beside it (picked before: the same version number,
    later in the listing), which a later update removes."""
    base = Path(os.environ.get('LOCALAPPDATA', '')) / 'cursor-agent' / 'versions'

    def ver(d):
        m = re.fullmatch(r'(\d{4})\.(\d{1,2})\.(\d{1,2})-[0-9a-f]+', d.name)
        return tuple(int(x) for x in m.groups()) if m else None
    vs = sorted((d for d in base.glob('*') if ver(d) and (d / 'node.exe').exists() and (d / 'index.js').exists()),
                key=ver)
    return str(vs[-1]) if vs else None


def cursor_model():
    """The model id Cursor takes: its effort is part of the name (gemini-4-argon-medium); 'default' leaves the
    name bare and Cursor applies its own default for the model (muse-wonder-1.0 -> Muse Wonder High)."""
    return CURSOR_MODEL if CURSOR_EFFORT in ('', 'default') else f'{CURSOR_MODEL}-{CURSOR_EFFORT}'


class _CursorSession:
    """A model on the user's Cursor plan (models no other login of the user's has: Meta's Muse, Gemini 4 Argon),
    through Cursor's agent CLI: each question resumes the game's chat by its id; 4-3 starts the second chat.
    The question goes on stdin (the digest is longer than a command line may be), in
    read-only ask mode in the reference folder; the reply streams (--stream-partial-output), so the decision is acted
    on as it is written. The CLI takes no image: the screenshot is a file the model reads with its read tool. It has
    no system prompt option, so the game's system prompt leads a chat's first message. Each run starts the CLI
    (~6-9 s a call on this PC, in its latency)."""

    def __init__(self):
        self._proc, self._lock = None, threading.Lock()
        self.session_id = self.thread_id = None
        self.last_meta, self.cut, self._dir, self._n = None, False, None, 0

    def close(self):
        with self._lock:
            self._state_sent = {}
            self.session_id = self.thread_id = None

    def interrupt(self):
        p = self._proc
        if p is not None and p.poll() is None:
            self.cut = True
            p.kill()

    def ask(self, exe, content, timeout=90, on_first=None, call_n=None, fresh=False, raw=False):
        with self._lock:
            if STOPPED or not _still_wanted():
                raise ModelStopped()
            content, new = (content, fresh) if fresh or raw else _stage_turnover(content, call_n)
            content = _outgoing(content, self, fresh=new or not self.session_id, call_n=call_n)
            text = next((c.get('text') or '' for c in content if c.get('type') == 'text'), '')
            if new or not self.session_id:
                self.session_id = self.thread_id = None
                text = opening_prompt() + '\n\n' + text      # no system prompt option: it leads the chat
            if self._dir is None or not self._dir.exists():
                self._dir = Path(tempfile.mkdtemp(prefix='tft-cursor-'))
            files = []
            for c in content:
                if c.get('type') == 'jpeg':
                    self._n += 1
                    f = self._dir / f'screen-{self._n}.jpg'
                    f.write_bytes(c['jpeg'])
                    files.append(f)
            if files:
                text += ('\n\nThe live game screen right now is the image file ' + ', '.join(str(f) for f in files)
                         + ': read it.')
            cwd = str(REFERENCE) if REFERENCE.exists() else str(self._dir)
            cmd = [str(Path(exe) / 'node.exe'), str(Path(exe) / 'index.js'), '-p', '--output-format', 'stream-json',
                   '--stream-partial-output', '--model', cursor_model(), '--mode', 'ask', '--trust',
                   '--workspace', cwd, '--add-dir', str(self._dir),
                   # no self-update: its waiter loops every ~5 s while an agent runs, each check (tasklist, find, ping)
                   # in a new console - Windows Terminal showed them over the game (benchmark game 20261020-120844:
                   # 31 focus losses; the screen read froze at 2-3 while the game played on to 4-2)
                   '--disable-auto-update']
            if self.session_id:
                cmd += ['--resume', self.session_id]
            env = dict(os.environ, NODE_COMPILE_CACHE=os.path.join(os.environ.get('LOCALAPPDATA', ''), 'cursor-compile-cache'),
                       CURSOR_INVOKED_AS='cursor-agent.cmd')
            feed = _first_object_watcher(on_first) if on_first else None
            LIVE.update(what=_question_line(content), t0=time.time(), text='', took=None)
            said, final, res, turns, self.cut = [], None, None, 1, False
            try:
                self._proc = proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                     stderr=subprocess.DEVNULL, cwd=cwd, env=env,
                                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                proc.stdin.write(text.encode('utf-8'))
                proc.stdin.close()
                timer = threading.Timer(timeout, proc.kill)
                timer.start()
                try:
                    for line in proc.stdout:
                        try:
                            ev = json.loads(line.decode('utf-8', 'replace'))
                        except ValueError:
                            continue
                        kind = ev.get('type')
                        if kind == 'system' and ev.get('session_id'):
                            self.session_id = self.thread_id = ev['session_id']
                        elif kind == 'assistant':
                            chunk = ''.join(c.get('text') or '' for c in (ev.get('message') or {}).get('content') or []
                                            if c.get('type') == 'text')
                            if 'timestamp_ms' in ev:              # a streamed part
                                said.append(chunk)
                                LIVE['text'] += chunk
                                if feed:
                                    feed(chunk)
                            else:
                                final = chunk                    # the whole message, at its end
                        elif kind == 'tool_call' and ev.get('subtype') == 'completed':
                            turns += 1                           # the screenshot or a reference file read
                        elif kind == 'result':
                            res = ev
                    proc.wait(timeout=10)
                finally:
                    timer.cancel()
            except OSError as e:
                if not self.cut:
                    raise RuntimeError(f'the Cursor CLI did not start: {e}')
            finally:
                self._proc = None
                for f in files:
                    try:
                        f.unlink()
                    except OSError:
                        pass
            LIVE['took'] = time.time() - LIVE['t0']
            answer = (''.join(said).strip() or (final or '').strip() or ((res or {}).get('result') or '').strip())
            if not ''.join(said).strip() and answer and feed:
                feed(answer)                                     # nothing streamed: the decision from the whole reply
            if not answer and not self.cut:
                raise RuntimeError(f"Cursor ended without an answer ({(res or {}).get('subtype') or 'no result'})")
            self.last_meta = {'num_turns': turns, 'duration_ms': round(LIVE['took'] * 1000),
                              'duration_api_ms': (res or {}).get('duration_api_ms'), 'usage': (res or {}).get('usage'),
                              'session_id': self.session_id}
            if CALL_DIR is not None:
                try:
                    (CALL_DIR / f'{call_n or 0:04d}-result.json').write_text(
                        json.dumps(dict(self.last_meta, result=answer), ensure_ascii=False), encoding='utf-8')
                except Exception:
                    pass
            return answer

    def open_terminal(self, exe, locked=True):
        """After the game: this stage's chat in the Cursor CLI's own window."""
        import contextlib
        with (self._lock if locked else contextlib.nullcontext()):
            if not self.session_id:
                return False
            launcher = Path(os.environ.get('LOCALAPPDATA', '')) / 'cursor-agent' / 'cursor-agent.cmd'
            d = str(REFERENCE) if REFERENCE.exists() else tempfile.gettempdir()
            try:
                subprocess.Popen(['cmd', '/c', str(launcher), '--resume', self.session_id, '--disable-auto-update'],
                                 cwd=d, creationflags=TERMINAL_FLAGS)
            except OSError:
                return False
            return True


_CURSOR = _CursorSession()


def _cursor_call(exe, prompt, image_jpeg=None, on_first=None, timeout=90):
    content = [{'type': 'text', 'text': prompt}] + [{'type': 'jpeg', 'jpeg': img} for img in _images(image_jpeg)]
    n = _save_call_inputs(prompt, _images(image_jpeg))
    try:
        return _CURSOR.ask(exe, content, timeout=timeout, on_first=on_first, call_n=n)
    finally:
        if n:
            _CURSOR.last_meta = dict(_CURSOR.last_meta or {}, call=n)


def _use_cursor(mid, effort):
    """A model on the user's Cursor plan, through Cursor's agent CLI (installed and signed in)."""
    global CURSOR_MODEL, CURSOR_EFFORT
    exe = cursor_cli()
    if not exe:
        return 'the Cursor CLI was not found: install it (cursor.com/cli) and sign in with "cursor-agent login"'
    node = [str(Path(exe) / 'node.exe'), str(Path(exe) / 'index.js')]
    try:
        said = ''.join(subprocess.run(node + [c], capture_output=True, timeout=120, text=True, encoding='utf-8',
                                      errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout
                       for c in ('status', 'models'))
    except (OSError, subprocess.TimeoutExpired) as e:
        return f'the Cursor CLI did not start: {e}'
    if 'logged in' not in said.lower():
        return 'the Cursor CLI is not signed in: run "cursor-agent login"'
    os.environ['TFT_COACH_BACKEND'] = 'cursor'
    os.environ['TFT_CURSOR_MODEL'] = CURSOR_MODEL = mid or CURSOR_MODEL
    if effort:
        os.environ['TFT_CURSOR_EFFORT'] = CURSOR_EFFORT = effort
    if CURSOR_EFFORT not in ('', 'default') and f'{cursor_model()} - ' not in said:
        return f'Cursor offers no model {cursor_model()} on this account'
    return None


def active_session():
    """The conversation object of the game's model backend (its last_meta, open_terminal)."""
    kind = _cli()[0]
    return {'openrouter': _OR, 'codex': _CODEX, 'grok': _GROK, 'agy': _AGY, 'cursor': _CURSOR}.get(kind, _SESSION)


# the pre-game speed check (speed_check): seconds to the decision of each recorded question, and the ones cut at the
# timeout - {'augment': s, 'plan': s, 'cut': [...]}; empty until it ran
SPEED = {}
SPEED_DIR = Path(__file__).resolve().parent.parent / 'data' / 'speed-check'
PLAN_WINDOW = 30         # s: a planning phase (a round plan must be carried out in it)
AUGMENT_WINDOW = 45      # s: the augment choice at 2-1 (~48 s on screen when it opens)


def speed_prompt(what):
    """Text transcription of the saved benchmark questions (20261018-153420)."""
    text = (SPEED_DIR / f'{what}.txt').read_text(encoding='utf-8')
    text = text.replace(' [UNVERIFIED: named by the model from the screenshot]',
                        ' [UNVERIFIED: identity not confirmed by an info-panel read]')
    text = text.replace('The attached image is the live game screen right now.', '')
    text = text.replace('The attached screenshot shows the augment choice: three cards, 0 = left, '
                        '1 = middle, 2 = right. Read the cards from the screenshot.',
                        augment_cards(['Warpath', 'Late Game Scaling', 'Birthday Reunion']))
    return text.replace('explain every card on the screen, using the numbers printed on the card',
                        'explain every offered card, using its supplied card text')


def speed_check(log=print, timeout=120):
    """Before the game (the user, 2026-10-20: a model too slow for the game is told before it plays): two questions
    of a real game - the 2-1 augment and the round plan of 4-1, each a stage's first message (its digest, the question,
    the screenshot with the player names blurred), from run 20261018-153420 - go to this game's model and effort in
    fresh conversations, as a stage's first question does, and are timed to the decision: the first JSON object,
    which the controller acts on. The result is in SPEED (and the run log); the augment steps use it until the
    game's own answers are timed. -> SPEED."""
    kind, exe = _cli()
    if kind not in ('claude-cli', 'codex', 'openrouter', 'grok', 'agy', 'cursor'):
        return SPEED
    session = active_session()
    out = {'cut': []}
    for what in ('augment', 'plan'):
        text = speed_prompt(what)
        jpg = SPEED_DIR / f'{what}.jpg'
        n = _save_call_inputs(text, [jpg.read_bytes()] if jpg.exists() else [])
        content = _outgoing([{'type': 'text', 'text': text}])
        got, t0 = [], time.time()
        first = lambda obj: got or got.append(time.time() - t0)
        _reset(session)                            # a fresh conversation, as a stage's first question has
        try:
            if kind == 'openrouter':
                session.ask(content, first, call_n=n)
            else:
                session.ask(exe, content, timeout=timeout, on_first=first, call_n=n)
        except Exception as e:
            if not got:
                out['cut'].append(what)
                log(f'speed check: the {what} question gave no decision in {time.time() - t0:.0f} s ({str(e)[:80]})')
                if time.time() - t0 < timeout - 5:
                    out.setdefault('refused', {})[what] = str(e)[:200]      # an error, not the time limit
        out[what] = round(got[0] if got else time.time() - t0, 1)
        _reset(session)                            # the game starts its own conversation
    SPEED.clear()
    SPEED.update(out)
    return SPEED


def speed_refused(speed=None):
    """Why no game can be played with this model: it refused both speed-check questions - an error well before the
    time limit, not a slow answer (2026-10-20: Grok Build on a free account, its daily usage spent, said "Rate limited"
    in 4 s; a game would have been played with no model). None otherwise."""
    r = (SPEED if speed is None else speed).get('refused') or {}
    if {'augment', 'plan'} <= set(r):
        return f"the model refused both speed-check questions ({r['plan'][:120]}): no game is started"
    return None


def _reset(session):
    session._state_sent = {}
    if session is _OR:
        session.messages = None
    else:
        session.close()


def speed_note(speed=None):
    """The speed check as one line for the player: each decision time against the window it has to fit."""
    s = speed if speed is not None else SPEED
    if not s:
        return None
    a, p, cut = s.get('augment'), s.get('plan'), s.get('cut') or []
    fits = int(max(0, AUGMENT_WINDOW - 4) // a) if a else 0
    line = (f"speed check: augment decision {'over ' if 'augment' in cut else ''}{a:.0f} s (the 2-1 choice has "
            f"~{AUGMENT_WINDOW} s: {min(fits, 3)} of its 3 steps fit), round plan {'over ' if 'plan' in cut else ''}"
            f"{p:.0f} s (a planning phase is ~{PLAN_WINDOW} s)")
    slow = [w for w, ok in (('augments', fits >= 1), ('round plans', p < PLAN_WINDOW - 5)) if not ok]
    return line + (f" - TOO SLOW for this game's {' and '.join(slow)}: they will often come too late" if slow else '')


def _openrouter_call(prompt, image_jpeg=None, on_first=None):
    import base64
    content = [{'type': 'text', 'text': prompt}]
    for img in _images(image_jpeg):
        content.append({'type': 'image_url', 'image_url': {
            'url': 'data:image/jpeg;base64,' + base64.standard_b64encode(img).decode()}})
    n = _save_call_inputs(prompt, _images(image_jpeg))        # benchmark data: the screenshots and prompt of every call
    try:
        return _OR.ask(content, on_first, call_n=n)
    finally:
        if n:
            _OR.last_meta = dict(getattr(_OR, 'last_meta', None) or {}, call=n)


# the question in flight, for the overlay: what is asked, since when, the reply as it streams in, and how
# long the last one took (None while one is running)
LIVE = {'what': None, 't0': 0.0, 'text': '', 'took': None}
_GAME_FALLBACK = None
BACKEND_LOG = None


def _switch_backend(error, call_n):
    """Usage exhaustion is sticky for this game; the existing between-game mark covers later sessions."""
    global _GAME_FALLBACK, _SESSION
    from agent import backends
    cfg = json.loads((AGENT / 'config.json').read_text(encoding='utf-8'))
    model = cfg.get('fallback_model') or backends.FALLBACK[0]
    effort = cfg.get('fallback_effort') or backends.FALLBACK[1]
    backends.mark(error, f'player call {call_n} ({CALL_DIR})')
    why = use_model(model, effort)
    if why:
        raise RuntimeError(f'{error}; fallback unavailable: {why}')
    with _SWITCH_LOCK:
        # Speculative Codex work must never be adopted after the backend has changed.
        work = _SWITCH.copy()
        _SWITCH.clear()
        if work.get('candidate') is not None:
            _retire_session(work['candidate'])
        _STAGE['early'] = None
    _SESSION = _Session()
    _SESSION._digest_pending = True
    _STRATS['sent'] = set()
    _GAME_FALLBACK = dict(model=model, effort=effort, error=error, call=call_n)
    LIVE['backend_switch'] = f'Codex usage limit: switched to {model} ({effort})'
    event = dict(ev='backend_switch', stage=_STAGE['now'], backend='claude-cli', model=model,
                 effort=effort, error=error, call=call_n)
    if BACKEND_LOG:
        BACKEND_LOG(**event)
    elif CALL_DIR is not None:
        with (CALL_DIR.parent / 'log.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(event, t=time.time())) + '\n')


def _question_line(content):
    """The first line of a question's text: what the model is being asked ('DECISION: augment, stage 4-2')."""
    try:
        text = next((c.get('text') for c in content if isinstance(c, dict) and c.get('type') == 'text'), '')             if isinstance(content, list) else str(content)
    except Exception:
        text = ''
    return next((ln.strip() for ln in (text or '').splitlines() if ln.strip()), '')[:140]


_PRIORITY = [0]              # urgent questions waiting or running (a choice screen is open and timed)
_RESERVED = [0]              # the model is held for an urgent question about to be asked (a container just opened)


def reserve_model():
    """Hold the model for an urgent question that is about to be asked: True only if nothing is running or
    reserved, and from then on no other question starts until release_model(). The caller then does what
    produces the question (opens the container) - it never opens and then waits for the model."""
    if _RESERVED[0] or _PRIORITY[0] or model_busy():
        return False
    _RESERVED[0] = 1
    return True


def release_model():
    _RESERVED[0] = 0


def plain(text):
    """The roster's log shorthand (Varus*@b3, Kha'Zix*@h33[Gargoyle]) in the words the state uses."""
    text = re.sub(r'(\*+)@b(\d)', r'\1 at bench slot \2', text)
    return re.sub(r'(\*+)@h(\d)(\d)', r'\1 at hex (\2,\3)', text)


def ask_model(prompt, image_jpeg=None, on_first=None, timeout=None, priority=False, still_wanted=None):
    """One question to the game's model (blocking). None when no model is connected.
    on_first(obj) gets the reply's first JSON object (the decision) as soon as it is
    written; the call itself returns the whole reply (decision + explanation).
    priority: an urgent question (a choice screen is open and the game's own timer runs): other
    questions wait for it instead of taking the conversation first."""
    if STOPPED:
        return None              # eliminated: nothing is asked any more
    from tft_player.game import PAUSED
    if PAUSED.is_set():
        return None              # F7: no questions while paused; on continue the player asks again
    prompt = plain(prompt)
    _remember(prompt)
    if _COMMIT['brief'] and not priority:
        # flex: the comp just committed to - its brief goes ahead of the next question (a timed choice waits for
        # the one after); later stages' digests carry it in place of the flex briefing
        prompt, _COMMIT['brief'] = _COMMIT['brief'] + COMMIT_SEP + prompt, None
    cut_preparation("the game's first question came")
    if _PREP['cut'] and not _PREP['told']:
        _PREP['told'] = True
        prompt = (f"(Your preparation reply was cut off: {_PREP['cut']}. What you had not written is lost; if your "
                  f"game_strategy is not written yet, send it in this reply.)\n\n" + prompt)
    # still_wanted() is asked the moment the question gets the conversation: one whose round passed while it waited
    # behind others is not sent (benchmark game 20261021-002815: Argon's answers came 53 s late at 1-3 and 338 s at
    # 4-2 - each question waited for the ones before it, all about rounds already over)
    _WANTED.fn = still_wanted
    try:
        if priority:
            _PRIORITY[0] += 1
            try:
                return _remember(answer=_ask(prompt, image_jpeg, on_first, timeout))
            finally:
                _PRIORITY[0] -= 1
        end = time.time() + 25
        while (_PRIORITY[0] > 0 or _RESERVED[0]) and time.time() < end:
            time.sleep(0.05)
        return _remember(answer=_ask(prompt, image_jpeg, on_first, timeout))
    except ModelStopped:
        if not STOPPED:
            print(f'question not sent: its round passed while it waited ({prompt[:60]!r})', flush=True)
        return None
    finally:
        _WANTED.fn = None


def _ask(prompt, image_jpeg, on_first, timeout):
    kind, exe = _cli()
    _activate_second(kind)
    if kind not in _SESSION_NAMES and kind != 'claude':
        return None
    # Archive first. No backend receives an image, an image path, or permission to open one.
    n = _save_call_inputs(prompt, _images(image_jpeg))
    if kind == 'claude':
        return _claude_call(prompt, call_n=n)
    session = active_session()
    content = [{'type': 'text', 'text': prompt}]
    try:
        if kind == 'openrouter':
            return session.ask(content, on_first, call_n=n)
        return session.ask(exe, content, on_first=on_first, call_n=n,
                           **({'timeout': timeout} if timeout else {}))
    finally:
        if n:
            session.last_meta = dict(getattr(session, 'last_meta', None) or {}, call=n)


def model_busy():
    """True while a question to the model is in flight (the conversation answers one at a time)."""
    try:
        return _SESSION._lock.locked() or _OR._lock.locked() or _CODEX._lock.locked() or _GROK._lock.locked() or _AGY._lock.locked() or _CURSOR._lock.locked()
    except Exception:
        return False


_PREP = {'running': False, 'cut': None, 'told': True}   # the preparation in flight; why it was cut off

# Owner o129/e131 (2026-11-02), experiments/README.md E1.7: keep intent through both later augments.
SECOND_CONVERSATION_ROUND = '4-3'
_STAGE = {'now': None, 'conv': None, 'early': None}  # conv: first / second
_SWITCH_LOCK = threading.Lock()
_SWITCH = {}                         # private early opening; published only when fully answered
_SESSION_NAMES = {'claude-cli': '_SESSION', 'codex': '_CODEX', 'openrouter': '_OR',
                  'grok': '_GROK', 'agy': '_AGY', 'cursor': '_CURSOR'}
_MEMO = {}                               # the second conversation's digest and handover
_STATE_BLOCK = re.compile(r'\nSTATE\n.*?(?=\n\nReply\b|\n\nThe reference files|\n\n=== THE QUESTION ===|\Z)', re.S)


_COMMIT = {'brief': None}        # flex: the committed comp's brief, waiting for the next question
COMMIT_SEP = '\n\n=== THE QUESTION ===\n'


def commit_brief(text):
    """Flex: the model committed to a comp (now forced_comp()). Its brief (forced_briefing: the comp's MetaTFT page
    and trait choices) goes ahead of the next question and replaces the flex briefing in later digests."""
    _MEMO['brief'] = text
    _MEMO.pop('comps', None)
    _COMMIT['brief'] = text


def set_stage(stage):
    """The round as the player reads it; only SECOND_CONVERSATION_ROUND changes conversations."""
    _STAGE['now'] = stage


def _remember(prompt=None, answer=None):
    """Keep what a later stage's first message repeats: from questions the briefing, the last state and the target
    comp's MetaTFT data (flex); from answers the trait picks, game_strategy, notes and comp probabilities."""
    if prompt:
        if prompt.startswith((PREP_HEAD, 'BRIEFING', 'Target comp for this game')):
            brief = prompt.replace(PREP_HEAD, '')
            for cut in ('\n\nTrait choices on the target final board', '\n\nYOUR MEMORY - ', '\n\nPREPARATION TASK',
                        '\n\nReply with ONLY'):
                brief = brief[:brief.find(cut)] if cut in brief else brief
            _MEMO['brief'] = brief.strip()
        if prompt.startswith('MetaTFT data of the target comp,'):
            _MEMO['page'] = prompt.split('\n\nDECISION:', 1)[0]
        m = _STATE_BLOCK.search(prompt)
        if m:
            _MEMO['state'] = m.group(0).strip()
    for o in json_objects(answer or '') if isinstance(answer, str) else ():
        if not isinstance(o, dict):
            continue
        if isinstance(o.get('trait_choices'), dict):
            _MEMO['traits'] = o['trait_choices']
        if isinstance(o.get('game_strategy'), dict) and o['game_strategy'].get('text'):
            _MEMO['strategy'] = o['game_strategy']['text']
        st = o.get('strategy')
        if isinstance(st, dict):
            _MEMO['notes'] = st.get('notes') or _MEMO.get('notes')
            if isinstance(st.get('comps'), list):
                _MEMO['comps'] = {'comps': st['comps'], 'other': st.get('other'), 'target': st.get('target')}
    return answer


def stage_digest(stage, with_state=True, backend_switch=False):
    """The second conversation's digest, before its first question or early acknowledgement."""
    m = _MEMO
    parts = [f'NEW CONVERSATION FROM {SECOND_CONVERSATION_ROUND} - this game uses two conversations: preparation through '
             'the 4-2 augment, planning phase and fight, then 4-3 to the end. The first conversation is not repeated '
             'here; this is what you need from it.',
             m.get('brief') or '(no briefing was sent)',
             f"Your trait choices (made in your preparation): {json.dumps(m.get('traits'))}\n"
             + (f"Your game plan as it stands (pinned on screen; it changes only at checkpoints): {m['plan']}\n"
                if m.get('plan') else f"Your game_strategy as it stands (pinned on screen): {m.get('strategy')}\n")
             + f"Your notes in your last reply: {m.get('notes')}"
             + (f"\nYour comp probabilities in your last reply: {json.dumps(m['comps'])}" if m.get('comps') else '')]
    if m.get('handover'):
        parts.append(f"Your handover note, written at the end of the first conversation: {m['handover']}")
    if m.get('page'):
        parts.append(m['page'])
    mem = memory_block()
    if mem:
        parts.append(mem)
    if with_state and m.get('state'):
        parts.append(f"The last game state you were sent (before this question):\n{m['state']}")
    parts.append('The reference files are in your working directory as before (README.md first).')
    if backend_switch:
        parts[0] = ('NEW CONVERSATION - the game switched from Codex to Claude after a usage/rate limit. '
                    'Continue this game from the digest and answer the question below.')
    return '\n\n'.join(parts) + '\n\n=== THE QUESTION ===\n'


def _stage_turnover(content, call_n=None):
    """Send the strategy in this stage's first message in each conversation.
    The owner's 2026-10-31 "inputted to the model for all the model calls during stage 3" was replaced by
    the owner's 2026-11-02 "stage strategy only sent once".
    """
    content, new = _new_stage_turnover(content, call_n)
    return _with_stage_strategy(content, call_n), new


def _with_stage_strategy(content, call_n=None, stage=None, sent=None):
    stage = stage or _STAGE['now']
    sent = _STRATS.setdefault('sent', set()) if sent is None else sent
    block = stage_strategy_text('1' if _PREP['running'] else stage) if isinstance(content, list) else ''
    key = (_STAGE['conv'] or 'first', block)
    if not block or key in sent:
        return content
    # Preparation already contains stage 1's block; count that copy when it is actually sent.
    if any(c.get('type') == 'text' and block in (c.get('text') or '') for c in content):
        sent.add(key)
        _save_text(call_n, 'strategy.txt', block)
        return content
    if _PREP['running']:
        return content
    out, done = [], False
    for c in content:
        if c.get('type') == 'text' and not done:
            text = c.get('text') or ''
            cut = text.find('=== THE QUESTION ===\n')
            at = cut + len('=== THE QUESTION ===\n') if cut >= 0 else 0
            c, done = dict(c, text=text[:at] + block + '\n\n' + text[at:]), True
        out.append(c)
    if done:
        sent.add(key)
        _save_text(call_n, 'strategy.txt', block)
    return out


def _new_stage_turnover(content, call_n=None):
    """The single round boundary; called inside the backend's session lock."""
    now = _STAGE['now']
    if not now or _PREP['running'] or not isinstance(content, list):
        return content, False
    second = stage_key(now) >= stage_key(SECOND_CONVERSATION_ROUND) if stage_key(now) else False
    if not second or _STAGE['conv'] == 'second':
        if _STAGE['conv'] is None:
            _STAGE['conv'] = 'first'
        return content, False
    _STAGE['conv'] = 'second'
    text = next((c.get('text') or '' for c in content if c.get('type') == 'text'), '')
    if _MEMO.get('brief') and text.startswith(_MEMO['brief'] + COMMIT_SEP):
        # a commit brief ahead of this question: the new conversation's digest carries it already
        content = [dict(c, text=c['text'][len(_MEMO['brief'] + COMMIT_SEP):])
                   if c.get('type') == 'text' and c.get('text') == text else c for c in content]
        text = text[len(_MEMO['brief'] + COMMIT_SEP):]
    digest = stage_digest(SECOND_CONVERSATION_ROUND.split('-')[0], with_state='\nSTATE\n' not in text)
    if CALL_DIR is not None and call_n:
        try:
            (CALL_DIR / f'{call_n:04d}-digest.txt').write_text(digest, encoding='utf-8')
        except OSError:
            pass
    out, done = [], False
    for c in content:
        if c.get('type') == 'text' and not done:
            c, done = dict(c, text=digest + c['text']), True
        out.append(c)
    return out, True


def new_game(call_dir):
    """Before each game: a new conversation (the last game's process ends), questions are answered again
    (STOPPED is set at elimination) and every call's files go to this game's run folder."""
    global STOPPED, CALL_DIR, _GAME_FALLBACK, BACKEND_LOG
    _GAME_FALLBACK, BACKEND_LOG = None, None
    LIVE.pop('backend_switch', None)
    STOPPED, CALL_DIR, _CALL_N[0] = False, call_dir, 0
    sync_player_dir()
    _PREP.update(running=False, cut=None, told=True)
    with _SWITCH_LOCK:
        _cancel_switch()
    _STAGE.update(now=None, conv=None, early=None)
    _STRATS.clear()
    _OPENING_HOLD.update(stage=None, until=0.0, answered=False)
    _MEMO.clear()
    for name in _SESSION_NAMES.values():
        globals()[name]._state_sent = {}
    _COMMIT['brief'] = None
    _SESSION.close()
    _CODEX.close()
    _OR.messages = None


PREP_WAIT = 3600         # s: only a hung CLI reaches this - the game's first question cuts the preparation off


def prepare():
    """The briefing, sent as we enter the queue (no game clock yet). It runs until the model is done or the
    game's first question comes - then it is cut off (cut_preparation) and the conversation goes on with
    what the model read. Returns {'prompt', 'answer', 'error', 'cut', 'seconds', 't', 'meta'}; answer None
    when it failed."""
    t0, text, error, answer = time.time(), briefing_text(before_queue=True), None, None
    _PREP.update(running=True, cut=None, told=False)
    _remember(plain(text))
    try:
        answer = _remember(answer=_ask(plain(text), None, None, PREP_WAIT))
    except Exception as e:
        error = str(e)
    finally:
        _PREP['running'] = False
    if answer is None and not error:
        error = 'no answer (no model connected)'
    meta = getattr(active_session(), 'last_meta', None)
    return {'prompt': text, 'answer': answer, 'error': error, 'cut': _PREP['cut'], 'seconds': time.time() - t0,
            't': time.time(), 'meta': meta}


def cut_preparation(why):
    """End the preparation reply now (the game needs the model, or its time is up); a no-op when none runs."""
    if _PREP['running'] and not _PREP['cut']:
        _PREP['cut'] = why
        _interrupt()


def _interrupt():
    """Cut the reply in flight short, in the game model's backend."""
    s = {'claude-cli': _SESSION, 'codex': _CODEX, 'grok': _GROK, 'agy': _AGY, 'cursor': _CURSOR}.get(_cli()[0])
    if s is not None:
        s.interrupt()


def stop_model():
    """The game is over: no question is sent any more - the one in flight is cut, the ones waiting for the
    conversation are dropped - so the run ends with the game (benchmark game 20261020-151221: Argon's queued answers
    went on for minutes after the elimination)."""
    global STOPPED
    if not STOPPED:
        STOPPED = True
        _interrupt()
        with _SWITCH_LOCK:
            _cancel_switch()


def warm_up():
    """Open the game's conversation now so the first question is fast too."""
    kind, exe = _cli()
    if kind == 'claude-cli':
        threading.Thread(target=_SESSION.warm, args=(exe,), daemon=True).start()


HANDOVER = ("HANDOVER - the first conversation ends with this fight. From 4-3 the game goes on in a new "
            "conversation that sees only a digest (the comp brief, your trait choices, your game plan, the notes "
            "of your last reply and the last state), not this conversation. Write what it needs to carry on your "
            "plan: what you intend for the rest of the game (levels and roll-downs with rounds and gold; units "
            "to buy, hold or sell), what each item, component and consumable you hold is for (which unit, and "
            "when), what you are waiting for, and what this game has shown that the state does not; plain text, "
            'at most 1200 characters, in one JSON object {"handover": "..."}.')
EARLY_NOTE = ('No question yet: read the above now and think ahead to 4-3 (levels, rolls, items). '
              'Reply only {"ready": true}.')
_OPENING_HOLD = {'stage': None, 'until': 0.0, 'answered': False}
OPENING_HOLD_MAX = 30


def hold_stage_opening(stage, secs=OPENING_HOLD_MAX):
    """Only the 4-2 combat-start answer must precede the handover."""
    if is_handover_round(stage):
        _OPENING_HOLD.update(stage=stage, until=time.time() + secs, answered=False)


def release_stage_opening(stage=None, answered=False):
    if stage is None or _OPENING_HOLD['stage'] == stage:
        _OPENING_HOLD.update(stage=None, until=0.0, answered=answered)


def _handover_round():
    s, r = stage_key(SECOND_CONVERSATION_ROUND)
    return s, r - 1


def is_handover_round(stage):
    return stage_key(stage) == _handover_round()


def _cancel_switch():
    """No foreground question waits for speculative work's session lock."""
    work = _SWITCH.copy()
    _SWITCH.clear()
    if work:
        work['cancelled'].set()
        for session in (work.get('outgoing'), work.get('candidate')):
            if session is not None:
                _retire_session(session)


def _retire_session(session):
    """Cancel promptly, then release obsolete processes without waiting in the action/question path."""
    try:
        if hasattr(session, 'interrupt'):
            session.interrupt()
    except (OSError, ValueError):
        pass

    def close():
        if hasattr(session, 'close'):
            session.close()
        if hasattr(session, 'stop'):
            session.stop()
    threading.Thread(target=close, daemon=True).start()


def _activate_second(kind):
    if kind not in _SESSION_NAMES or (stage_key(_STAGE['now']) or (0, 0)) < stage_key(SECOND_CONVERSATION_ROUND):
        return
    with _SWITCH_LOCK:
        if not _SWITCH:
            return
        name = _SESSION_NAMES[kind]
        ready = _SWITCH.get('ready')
        if ready is not None:
            outgoing = _SWITCH['outgoing']
            globals()[name] = ready
            _STAGE['conv'] = 'second'
            _STRATS.setdefault('sent', set()).add(('second', stage_strategy_text(SECOND_CONVERSATION_ROUND)))
            _SWITCH.clear()
            _retire_session(outgoing)
        else:
            # The handover/opening missed the deadline. Interrupt it and open from the current digest in this call.
            outgoing = _SWITCH['outgoing']
            _cancel_switch()
            globals()[name] = type(outgoing)()


def open_second_conversation(stage, log=None):
    """After the 4-2 combat answer: handover in the outgoing session, then a private fresh session.
    A foreground 4-3 call adopts it only after its acknowledgement; otherwise it opens from the digest itself.
    """
    kind, exe = _cli()
    if (kind not in _SESSION_NAMES or STOPPED or stage_key(stage) != _handover_round()
            or _STAGE.get('early') or _STAGE['conv'] == 'second'):
        return False
    _STAGE['early'] = 'second'
    work = {'cancelled': threading.Event(), 'outgoing': active_session()}
    with _SWITCH_LOCK:
        _SWITCH.update(work)

    def wanted():
        return (not work['cancelled'].is_set() and not STOPPED
                and stage_key(_STAGE['now']) == _handover_round())

    def ask(session, text, n, **kw):
        content = [{'type': 'text', 'text': text}]
        if kind == 'openrouter':
            return session.ask(content, call_n=n, **kw)
        return session.ask(exe, content, timeout=30, call_n=n, **kw)

    def run():
        _WANTED.fn = wanted
        try:
            # Do not let an opening steal the combat question, even if another question held it up.
            while wanted() and (_OPENING_HOLD['stage'] == stage or _PRIORITY[0] or _RESERVED[0]):
                time.sleep(0.05)
            if not wanted():
                return
            for attempt in range(2):
                t0, answer, error = time.time(), None, None
                n = _save_call_inputs(HANDOVER, [])
                try:
                    answer = ask(work['outgoing'], HANDOVER, n, raw=True)
                    note = next((o['handover'] for o in reversed(json_objects(answer or '', repair=False))
                                 if isinstance(o, dict) and isinstance(o.get('handover'), str) and o['handover'].strip()), None)
                    if not note:
                        raise ValueError('no handover in reply')
                    with _SWITCH_LOCK:
                        if not work['cancelled'].is_set() and not STOPPED:
                            _MEMO['handover'] = note.strip()[:1200]
                except Exception as e:
                    error = str(e)[:200]
                if log:
                    log(ev='handover', stage=stage, text=answer, error=error,
                        seconds=round(time.time() - t0, 1), call=n, attempt=attempt + 1)
                if not error or not wanted():
                    break
            if error or not wanted() or _cli()[0] != kind:
                return
            candidate = work['candidate'] = type(work['outgoing'])()
            with _SWITCH_LOCK:
                if not wanted():
                    return
                _SWITCH['candidate'] = candidate
            digest = stage_digest(SECOND_CONVERSATION_ROUND.split('-')[0])
            n = _save_call_inputs(digest + EARLY_NOTE, [])
            _save_text(n, 'digest.txt', digest)
            content = _with_stage_strategy([{'type': 'text', 'text': digest + EARLY_NOTE}], n,
                                           stage=SECOND_CONVERSATION_ROUND, sent=set())
            t0 = time.time()
            try:
                answer = ask(candidate, content[0]['text'], n, fresh=True)
                if not any(isinstance(o, dict) and o.get('ready') is True for o in json_objects(answer or '', repair=False)):
                    raise ValueError('no acknowledgement in reply')
                with _SWITCH_LOCK:
                    if not work['cancelled'].is_set() and not STOPPED and _cli()[0] == kind:
                        _SWITCH['ready'] = candidate
                        how = 'opened'
                    else:
                        how = 'skipped'
            except Exception as e:
                how = f'failed: {e}'[:200]
            if log:
                log(ev='conversation_opened_early', stage=stage, next=SECOND_CONVERSATION_ROUND,
                    how=how, seconds=round(time.time() - t0, 1), call=n)
        finally:
            _WANTED.fn = None

    threading.Thread(target=run, daemon=True).start()
    return True


CLAUDE_MODEL = os.environ.get('TFT_CLAUDE_MODEL', 'claude-opus-5-5')
CLAUDE_EFFORT = os.environ.get('TFT_CLAUDE_EFFORT', 'low')        # latency matters: low by default
CODEX_MODEL = os.environ.get('TFT_CODEX_MODEL', 'gpt-6.1-sol')
CODEX_EFFORT = os.environ.get('TFT_CODEX_EFFORT', 'high')


def _claude_call(prompt, image_jpeg=None, call_n=None):
    """Stateless text question to Claude, with server-side refusal fallbacks."""
    import anthropic
    client = anthropic.Anthropic(max_retries=1, timeout=90.0)
    n = call_n or _save_call_inputs(prompt, _images(image_jpeg))
    content = _outgoing([{'type': 'text', 'text': prompt}], call_n=n)
    resp = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        thinking={'type': 'adaptive'},
        output_config={'effort': CLAUDE_EFFORT},
        betas=['server-side-fallback-2026-07-01'],
        extra_body={'fallbacks': 'default'},
        system=opening_prompt(),
        messages=[{'role': 'user', 'content': content}],
    )
    if resp.stop_reason == 'refusal':
        raise RuntimeError(f'refused: {getattr(resp, "stop_details", None)}')
    return ''.join(b.text for b in resp.content if b.type == 'text')


def augment_text(name, full=False):
    """The augment's card text from game data."""
    n = None if full else 220
    if name in META.augments:
        desc = META.augments[name].get('desc', '')
    else:
        entries = CATALOG.augments.get(name) or []
        desc = entries[0].get('description', '') if entries else ''
        desc = re.sub(r'@\w+@', 'N', desc)
    from tft_player.patch_data import with_note
    return with_note('augments', name, desc[:n])


def owner_banned_augments(comp=None):
    """The owner's bans for this game's comp, read from the authoritative rules file."""
    comp = comp or os.environ.get('TFT_FORCE_COMP') or getattr(forced_comp(), 'name', '')
    slug = re.sub(r'[^A-Za-z0-9]+', '_', comp).strip('_')
    path = Path(__file__).resolve().parent.parent / 'agent' / 'expert' / 'rules' / f'{slug}.json'
    if not slug or not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding='utf-8')).get('banned_augments') or [])


def augment_cards(names, raw=()):
    from tft_player.augment_harness import card_mark

    lines = ['Augment cards (0 = left, 1 = middle, 2 = right):']
    banned = owner_banned_augments()
    for i, name in enumerate(names):
        if name and name not in ('left card', 'middle card', 'right card'):
            mark = card_mark(name)
            if name in banned:
                mark += ' [never picked in this comp: do not pick it; you may reroll this card]'
            lines.append(f'{i}: {name}{mark} - "{augment_text(name, full=True) or "card text not available"}"')
        else:
            lines.append(f'{i}: name not read; raw OCR: {raw[i] if i < len(raw) else "not read"!r}; '
                         'card text not available [controller capability: unrated]')
    return '\n'.join(lines)


def keyword_augment(names, roster):
    """Last resort when an augment is unknown to the meta snapshot."""
    traits = Counter(t for u in roster.units if u.info for t in u.info.traits)
    def score(n):
        d = augment_text(n).lower() + ' ' + (n or '').lower()
        s = 0.0
        for t, c in traits.items():
            if t.lower() in d:
                s += 2 + c
        for kw, v in (('gold', 1.5), ('item', 1.2), ('component', 1.2), ('emblem', 1.0),
                      ('health', 0.8), ('attack speed', 0.8), ('ability power', 0.6),
                      ('reroll', 0.8), ('xp', 0.8)):
            if kw in d:
                s += v
        if re.search(r'after \d+|in \d+ rounds|at stage [4-7]', d):
            s -= 1.0            # delayed payoff
        return s
    scores = [score(n) for n in names]
    return max(range(len(names)), key=lambda i: scores[i])


def stage_key(stage):
    try:
        a, b = str(stage).split('-')
        return int(a), int(b)
    except ValueError:
        return None


def local_augment(names, roster, stage, rerollable, plan):
    """Default decision: ('pick', i) or ('reroll', i), plus a reason."""
    known = [n in META.augments for n in names]
    if not META.available or not any(known):
        return ('pick', keyword_augment(names, roster)), 'keywords'
    cands = META.candidates(roster, prefer=plan.comp)
    w, rest = META.comp_weights(cands, stage_key(stage))
    # never reroll off a misread: only when all three names are recognised
    decision, vals = META.augment_decision(names, rerollable if all(known) else [], w, rest)
    return decision, 'meta ' + '/'.join(f'{v:.2f}' for v in vals)


# the augment answer format (the user, 2026-10-12): each card's COMBAT and ECONOMY value in numbers, then the
# decision; the example's card texts are copied from the cards on screen (run 20261010-235723, stage 4-2)
AUGMENT_EXPLAIN = (
    'Before deciding, explain every offered card, using its supplied card text (unread facts remain unknown):\n'
    'COMBAT: the exact bonuses on your current board and on the {final}.\n'
    'ECONOMY: what it gives now and later.\n'
    '{synergy}'
    'VERDICT: one short sentence - why you take this card, or why not (shown on screen under the card).\n'
    'Write "none" for a part where the card gives nothing.\n\n'
    'Example of the format, from an earlier game (its board and gold are not yours):\n\n'
    'Card 1 (middle): Bronze For Life II - "Your team gains 2.5% Damage Amp and 3 Armor and Magic Resist for '
    'each Bronze-tier trait."\n'
    'COMBAT:\n'
    '  Current board: +12.5% Damage Amp, +15 Armor and MR on all 7 units.\n'
    '  {final_label}: +15% Damage Amp, +18 Armor and MR on all 7 units.\n'
    'ECONOMY: none.\n'
    '{synergy_none}'
    'VERDICT: not taken - Invested++ gives more in that game.\n\n'
    'Card 0 (left): Invested++ - "Gain 45 gold. At the start of each round, gain 1 Shop reroll for every 10 '
    'gold above 50 (max 80 gold)."\n'
    'COMBAT: none.\n'
    'ECONOMY:\n'
    '  Now: +45 gold (16 -> 61).\n'
    '  Later: +4 interest a round, and 1-3 free rerolls a round while holding 60-80 gold.\n'
    '{synergy_ex}'
    'VERDICT: taken - 45 gold now and free rerolls while holding 60+.\n\n')

# what a card adds to the rest of the plan. The owner (2026-10-23, agent/history/owner-notes.md): "the augments synergy
# is very important, and should be thought well for augment selection" - U.R.F (holders of Spatula and Frying Pan items
# +20% AS, 3 mana regen) with Branching Out and every augment that gives emblems. Their demonstration 20261023-090521:
# Birthday Reunion's level-9 2-star 5-cost, Invested+ at 3-2 to pay for 9 at 4-1, Booster Pack++ once the gold was
# spent. In g1-g21, 1 of the 96 answers given while an earlier augment was held named it, and an emblem card was taken
# once in 18 offers. Every card from 2-1 on: with nothing held yet, the part is about the choices still to come.
AUGMENT_SYNERGY = {
    'synergy': 'SYNERGY: one or two short sentences - what it adds to, or needs from, your other augments ({later}), '
               'your items and emblems, your units and your level plan, by name.\n',
    'synergy_none': 'SYNERGY: none.\n',
    'synergy_ex': 'SYNERGY: Birthday Reunion (held) gives a 2-star 5-cost at level 9; from level 7 at 4-1, level 9 is '
                  '100 XP (25 buys, 100 gold): this gold and the free rerolls pay for it.\n'}
AUGMENT_ROUNDS = ('2-1', '3-2', '4-2')
# the owner (x16.1, 2026-11-06): "if the model wants to take an artifact augment, it should give a list of the wanted
# ranked artifacts as well" - the artifact choice it opens is then taken at once (Player.planned_choice)
ARTIFACT_ASK = (' If the card you take gives artifacts to choose from, add "artifacts": the artifacts you want from '
                'that choice, best first (the first one offered is taken at once, without asking you again).')
ARTIFACT_KEY = ', "artifacts": ["<artifact>", ...]'


def later_augments(stage):
    """SYNERGY's augments: the ones held and the choices still to come after this stage's."""
    now = stage_key(stage)
    rest = [s for s in AUGMENT_ROUNDS if now and stage_key(s) > now]
    return (f"the ones you hold, and the {' and '.join(rest)} choice{'s' if len(rest) > 1 else ''} still to come" if rest
            else 'the ones you hold; no augment choice comes after this one')


def augment_explain(stage, forced):
    """The answer format of an augment question at this stage."""
    syn = dict(AUGMENT_SYNERGY, synergy=AUGMENT_SYNERGY['synergy'].format(later=later_augments(stage)))
    level8 = False
    if forced:
        from tft_player.comp_reach import reach
        level8 = bool(reach(getattr(forced, 'name', None)))
    final = ('level-8 board above (the one stage 4 and most of our fights are played with)' if level8 else
             'target final board' if forced else 'final board you are building toward')
    return AUGMENT_EXPLAIN.format(final=final, final_label='Level-8 board' if level8 else 'Target final board', **syn)


def augment_context(held, units=()):
    """The lines an augment question adds after its cards: the augments held, with their text, and the emblems that add
    a trait level to the board the game builds toward (what makes an emblem card or a Spatula worth taking)."""
    from tft_player.emblems import board_line
    return [line for line in (held_augments_line(held), board_line(units)) if line]


def held_augments_line(held):
    """'Augments you hold: <name> - "<card text>"; ...' for the augment question; '' while none is held (a card the
    game picked by its timer is not named: "one of A / B / C")."""
    names = [str(n) for n in held or () if n and not str(n).startswith('one of ')]
    if not names:
        return ''
    return 'Augments you hold: ' + '; '.join(
        f'{n} - "{augment_text(n, full=True)}"' if augment_text(n, full=True) else n for n in names) + '.'


def card_sections(text):
    """{card index: its explanation block ('Card 1 (middle): ... VERDICT: ...')} from an augment answer."""
    parts = re.split(r'(?m)^(?=[ *#]*Card \d\s*\((?:left|middle|right)\))', text or '')
    out = {}
    for part in parts:
        m = re.match(r'[ *#]*Card (\d)', part)
        if m and 0 <= int(m.group(1)) <= 2:
            out[int(m.group(1))] = re.split(r'(?m)^\s*\{', part)[0].strip()      # the JSON answer is not part of it
    return out


class Coach:
    def __init__(self, brain, log=print, model=None, on_call=None, on_answer=None):
        self.brain, self.log = brain, log
        self.on_call = on_call          # (kind, prompt, raw_answer, latency) for the live view / log
        self.on_answer = on_answer      # the parsed answer (e.g. its comp probabilities)
        self.model = model or os.environ.get('TFT_COACH_MODEL') or 'auto'
        self.kind, self.exe = _cli()
        self.busy = False
        self.pending_augment = None       # augment names the current call is about
        self.answers = []
        self.finished = 0                 # augment calls ended (answered, unusable or failed)
        # this stage's card explanations so far and the cards the last answer rerolled: after a reroll only the new
        # card is explained again (the user, 2026-10-16)
        self.aug_memory = {'stage': None, 'cards': {}, 'reroll': []}

    @property
    def available(self):
        return self.exe is not None

    def ask(self, scr, roster, augments=None, rerollable=(), default=None, image=None, note=None, timing=None,
            held=(), raw=()):
        """An augment decision, as a message in the game's conversation: the three
        augments as displayed, and which to reroll (or which to take). note: why the
        question comes again (the last answer could not be carried out). timing: (seconds left, this model's answer
        time) when rerolls are left but the time fits only this answer - it takes a card. held: the augments taken
        earlier this game - their card text is shown; every card is explained with its SYNERGY (augment_context,
        augment_explain)."""
        if not self.available or self.busy or not augments:
            return False
        self._image = image
        cards = augment_cards(augments, raw)
        forced = forced_comp()
        why = '"why": "<one sentence>"'
        if len(rerollable) == len(augments):
            ask = ('Choose the ONE card you keep. The other two are then rerolled (free) and you choose again '
                   'from the new three.' + ('' if forced else ' Also choose your path now: P for 1-3 comps '
                   '(P = the probability it is the final board of this game; comps + other = 1), and "commit": '
                   'a comp name if you commit to it now (COMMIT in your instructions), else null.'))
            decision = '{"keep": <0-2>, "name": "<supplied card name, or unknown>", ' + why + '}'
        elif rerollable:
            # run23: "Rerolls are used" was said while the kept card still had its reroll
            ask = (f'Cards you can still reroll (each once): {", ".join(str(i) for i in rerollable)}'
                   + (' - the card you kept still has its own reroll' if len(augments) - len(rerollable) == 2 else '')
                   + '. Reroll one of them, or take a card. Taking a card while another one still has its '
                   'reroll: that reroll is used first (it is free) and you are asked again with the new card.'
                   + ARTIFACT_ASK)
            decision = ('{"reroll": [<card>]} or {"pick": <0-2>, "name": "<supplied card name, or unknown>", '
                        + why + ARTIFACT_KEY + '}')
        else:
            ask = 'No rerolls left. Which augment do you take?' + ARTIFACT_ASK
            decision = '{"pick": <0-2>, "name": "<supplied card name, or unknown>", ' + why + ARTIFACT_KEY + '}'
        # the explanation comes first, as plain text: the decision object is still the reply's first JSON
        # object, so on_first applies it the moment it is written
        units = list(forced.units) if forced else [u.name for u in roster.board() if u.name] if roster else []
        context = augment_context(held, units)
        explain = augment_explain(scr.stage, forced)
        mem = getattr(self, 'aug_memory', None) or {'stage': None, 'cards': {}, 'reroll': []}
        if mem['stage'] != scr.stage:
            self.aug_memory = mem = {'stage': scr.stage, 'cards': {}, 'reroll': []}
        new = [i for i in mem['reroll'] if 0 <= i < len(augments)]
        kept = [i for i in range(len(augments)) if i not in new and i in mem['cards']]
        if new and kept:
            # after a reroll only the new card(s) are explained: the unchanged cards' explanations stand (the user,
            # 2026-10-16 - the model wrote all three again)
            names = lambda ix: ' and '.join(f'card {i}' for i in ix)
            explain = explain.replace('explain every offered card',
                                      f'explain only the new {names(new)} (just rerolled)', 1)
            explain = (f'{names(kept).capitalize()} did not change: your explanation of '
                       f'{"it" if len(kept) == 1 else "them"} from the last question stands - do not write it again:\n\n'
                       + '\n\n'.join(mem['cards'][i] for i in kept) + '\n\n' + explain)
        ask += '\n\n' + explain
        if forced:                                  # fixed comp: no path to choose, no P
            ask += f'Then end your reply with one JSON object:\n{decision}'
        else:
            ask += ('Then end your reply with two JSON objects:\n' + decision + '\nthen {"comps": [{"name": '
                    '"<MetaTFT comp>", "p": 0.3}, ...], "other": 0.4, "commit": null, '
                    '"why": "<3-5 sentences>"}')
        if forced:
            traits = Counter(t for u in forced.units for t in (CATALOG.unit(u).traits if CATALOG.unit(u) else ()))
            # judged on the level-8 board our games fight with; MetaTFT's final board is the best case (the owner, x32.7)
            from tft_player.comp_reach import board_lines
            boards = board_lines(forced, forced.units)
            ask = (boards + (f"MetaTFT's final board of {forced.name} (when everything goes right): " if boards else
                             f'Target final board of {forced.name} (MetaTFT): ')
                   + f'{", ".join(forced.units)}; its traits: '
                   + ', '.join(f'{t} {n}' for t, n in traits.most_common()) + '.\n' + ask)
        prompt = (f'DECISION: augment, stage {scr.stage} (~{scr.timer}s left).\n' + (f'{note}\n' if note else '') +
                  f'{cards}\n' + ''.join(f'{line}\n' for line in context) + f'\n{ask}')
        self.busy = True
        self.pending_augment = augments
        self._kind = 'augment'
        self._decided = None              # when this answer's pick or reroll was carried out (its decision time)
        threading.Thread(target=self._run, args=(prompt, augments), daemon=True).start()
        return True

    def _run(self, prompt, augments):
        t0 = time.time()
        try:
            if self.kind in ('claude', 'claude-cli', 'openrouter', 'codex', 'grok', 'agy', 'cursor'):
                # the pick/reroll is applied the moment its object is written, after the per-card explanation
                text = ask_model(prompt,
                                 getattr(self, '_image', None), on_first=lambda obj: self._apply(obj, augments))
            else:
              with tempfile.TemporaryDirectory(prefix='tft-coach-', ignore_cleanup_errors=True) as d:
                if self.kind == 'cursor':
                    cmd = [self.exe, '--print', '--output-format', 'json', '--workspace', d,
                           '--trust', '--mode', 'ask'] + (['--model', self.model] if self.model else [])
                else:
                    cmd = [self.exe, 'exec', '--skip-git-repo-check', '--sandbox', 'read-only', '-']
                out = subprocess.run(cmd, input=prompt.encode(), capture_output=True, cwd=d,
                                     timeout=60, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                text = out.stdout.decode('utf-8', 'replace')
                if out.returncode != 0:
                    raise RuntimeError(out.stderr.decode('utf-8', 'replace').strip()[-300:])
            try:
                text = json.loads(text).get('result', text)
            except Exception:
                pass
            ans = {}
            for obj in json_objects(text):
                if isinstance(obj, dict):
                    ans.update(obj)
            if self._kind == 'augment':
                mem = self.__dict__.setdefault('aug_memory', {'stage': None, 'cards': {}, 'reroll': []})
                mem['cards'].update(card_sections(text))
                aug = ans.get('augment') if isinstance(ans.get('augment'), dict) else ans
                if isinstance(aug.get('keep'), int):
                    mem['reroll'] = [i for i in range(len(augments)) if i != aug['keep']]
                else:
                    r = aug.get('reroll')
                    mem['reroll'] = [r] if isinstance(r, int) else [i for i in r or [] if isinstance(i, int)]
            self._apply(ans, augments)
            if self.on_answer:
                self.on_answer(ans)
            # the time to the decision (the pick or reroll was carried out then), not to the reply's end: it is what
            # an augment window has to fit (answer_time)
            self.answers.append(((getattr(self, '_decided', None) or time.time()) - t0, ans))
            self.log(f'[coach {time.time() - t0:.0f}s] ' + json.dumps(ans, ensure_ascii=True))
            if self.on_call:
                self.on_call(self._kind, prompt, text, time.time() - t0)
        except Exception as e:
            self.log(f'[coach failed after {time.time() - t0:.0f}s] {e}')
            if self.on_call:
                self.on_call(self._kind, prompt, f'FAILED: {e}', time.time() - t0)
        finally:
            self.finished += 1
            self.busy = False

    def _apply(self, ans, augments):
        p = self.brain.plan
        aug = ans.get('augment') if isinstance(ans.get('augment'), dict) else ans
        if isinstance(aug.get('keep'), int) and augments and 0 <= aug['keep'] < len(augments):
            aug = {'reroll': [i for i in range(len(augments)) if i != aug['keep']]}
        if augments and self.pending_augment == augments:
            pick, reroll = aug.get('pick'), aug.get('reroll')
            reroll = [reroll] if isinstance(reroll, int) else reroll if isinstance(reroll, list) else []
            reroll = [i for i in reroll if isinstance(i, int) and 0 <= i < len(augments)]
            if reroll:
                p.augment_reroll = reroll
            elif isinstance(pick, int) and 0 <= pick < len(augments):
                p.augment_pick = pick
                arts = aug.get('artifacts')
                p.augment_artifacts = [str(a) for a in arts if a] if isinstance(arts, list) else None
            if (reroll or isinstance(pick, int)) and getattr(self, '_decided', 0) is None:
                self._decided = time.time()
            if isinstance(aug.get('name'), str):
                p.augment_name = aug['name']        # the card as the model read it
