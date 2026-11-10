"""Ask a recorded game's model questions again under other conversation methods, and compare the answers.

Every question is sent as it was in the game (same prompt, screenshot, model, effort, system prompt, tools,
Claude CLI on the Claude login); only what comes before it changes:

  A  original: the game's own conversation up to the question (the CLI resumes a cut copy of its transcript)
  D  original history with only the newest screenshot (older screenshots and thinking removed)
  B  a new conversation per stage: a digest (comp brief, trait picks, game_strategy, notes, last state) and
     the stage's earlier messages only
  C  B with every screenshot at 1100x688 (about half the image tokens)
  N  B with no screenshots at all (text state only)
  T  B with screenshots but no STATE text (screenshot only)
  M  B with the climb agent's memory as it is NOW (agent/memory, as coach.memory_block gives it) in the digest,
     the comp plan's level line from agent/config.json as it is now, and no live-review note: what the player
     would decide at this moment with the memory written since - the coach's offline check

usage:
  python tools/replay_decisions.py --calls 10,11,34 A1 B1 B2      tag = condition letter + sample number
  python tools/replay_decisions.py --compare                      table of all replies so far (markdown)
  --run runs/<id> picks the game (default: the 1st-place Lunar Aphelios game 20261015-132144); replies go to
  experiments/token-cost/replays/<id>/<tag>/<call>.json. One question at a time (the Claude login is shared
  with any live game). Replays cost tokens (A re-reads the whole game history): screen with
  --model claude-sonnet-5-5 --effort low (answers go to replays/<id>@<model>-<effort>/), then confirm only the
  differences with the game's model.
"""
import argparse
import base64
import copy
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TFT_CLAUDE_AUTH', 'login')
os.environ.setdefault('TFT_CLIMB', '1')          # memory_block for condition M (nothing else here reads it)
from tft_player.coach import PLAYER_DIR, REFERENCE, claude_cli, cli_env, json_objects  # noqa: E402

KEEP = {'user', 'assistant', 'attachment', 'system'}
NAMES = {'O': 'game', 'A': 'A original', 'D': 'D newest screenshot only', 'B': 'B per stage',
         'C': 'C per stage, half-size screenshots', 'N': 'N per stage, no screenshots',
         'T': 'T per stage, screenshot only (no STATE text)', 'M': 'M per stage, with the agent memory as it is now'}
STATE_RX = re.compile(r'\nSTATE\n.*?(?=\n\nReply with|\n\nYour game_strategy|\Z)', re.S)


class Game:
    def __init__(self, run):
        self.run = Path(run)
        self.calls = self.run / 'calls'
        first = json.loads((self.calls / '0002-result.json').read_text(encoding='utf-8'))
        # the player's CLI ran in the reference folder, or in PLAYER_DIR in climb games: its transcript is under that
        slugs = [Path.home() / '.claude' / 'projects' / re.sub(r'[^A-Za-z0-9]', '-', str(d)) for d in (REFERENCE, PLAYER_DIR)]
        self.proj = next((d for d in slugs if (d / f"{first['session_id']}.jsonl").exists()), slugs[0])
        self.lines = [json.loads(l) for l in open(self.proj / f"{first['session_id']}.jsonl", encoding='utf-8')]
        m = re.search(r'Target comp for this game: (.+?) \((\d+)\)', self.prompt(1))
        if m:
            os.environ.setdefault('TFT_FORCE_COMP', m.group(1))      # the comp notes in memory_block (condition M)
        self.system = next(s for e in self.lines if e.get('type') == 'attachment'
                           and e['attachment'].get('type') == 'prompt_snapshot'
                           for s in e['attachment']['systemPrompt'] if s.startswith('You are'))
        self.model = next((e['message'].get('model') for e in self.lines if e.get('type') == 'assistant'), 'claude-opus-5-5')
        self.effort = next((e.get('effort') for e in self.lines if e.get('type') == 'assistant' and e.get('effort')), 'high')
        self.game_model, self.game_effort = self.model, self.effort
        self.call_line = {}
        for i, e in enumerate(self.lines):
            if e.get('type') == 'user' and not e.get('isMeta'):
                c = e['message']['content']
                if isinstance(c, str) or not any(b.get('type') == 'tool_result' for b in c):
                    self.call_line[len(self.call_line) + 1] = i
        self.header = [e for e in self.lines[:self.call_line[2]] if e.get('type') == 'attachment'
                       and e['attachment'].get('type') in ('environment', 'model', 'instructions', 'session_context',
                                                           'date', 'credential_org', 'prompt_snapshot')]
        self._stage = {}

    def prompt(self, n):
        return (self.calls / f'{n:04d}-prompt.txt').read_text(encoding='utf-8')

    def reply(self, n):
        """The game's answer to call n; '' for a call that has no result file (g7's call 45: the digest of a later
        call reads every earlier answer)."""
        f = self.calls / f'{n:04d}-result.json'
        return (json.loads(f.read_text(encoding='utf-8')).get('result') or '') if f.exists() else ''

    def stage(self, n):
        if n not in self._stage:
            m = re.search(r'[Ss]tage (\d)-\d', self.prompt(n))
            self._stage[n] = m.group(1) if m else (self.stage(n - 1) if n > 1 else '1')
        return self._stage[n]

    def stage_start(self, n):
        k = n
        while k > 2 and self.stage(k - 1) == self.stage(n):
            k -= 1
        return k

    def seed(self, s, state=True, memory=False):
        """The first message of a new conversation that starts at call s (memory: with the agent's memory now)."""
        brief = self.prompt(1)
        brief = brief[brief.find('Target comp for this game'):] if 'Target comp for this game' in brief else brief
        cut = brief.find('\n\nTrait choices on the target final board')
        brief = brief[:cut] if cut > 0 else brief
        cut = brief.find('\n\nPREPARATION TASK')
        brief = brief[:cut] if cut > 0 else brief
        traits = gs = notes = None
        for k in range(1, s):
            for o in json_objects(self.reply(k)):
                if isinstance(o, dict):
                    traits = o.get('trait_choices', traits)
                    if isinstance(o.get('game_strategy'), dict):
                        gs = o['game_strategy'].get('text') or gs
                    if isinstance(o.get('strategy'), dict) and o['strategy'].get('notes'):
                        notes = o['strategy']['notes']
        last = ''
        if state and '\nSTATE\n' not in self.prompt(s):
            for k in range(s - 1, 0, -1):
                m = STATE_RX.search(self.prompt(k))
                if m:
                    last = f'\n\nThe last game state you were sent (before this question):{m.group(0)}'
                    break
        return (f'NEW CONVERSATION FOR STAGE {self.stage(s)} - each stage of this game is played in a new conversation, '
                'to keep messages short. The earlier stages are not repeated here; this is what you need from them.\n\n'
                f'{brief.strip()}\n\n'
                f'Your trait choices (made in your preparation): {json.dumps(traits)}\n'
                f'Your game_strategy as it stands (pinned on screen): {gs}\n'
                f'Your notes in your last reply: {notes}'
                + (f'\n\n{memory_now()}' if memory and memory_now() else '') + f'{last}\n\n'
                'The reference files are in your working directory as before (README.md first).\n\n'
                '=== THE QUESTION ===\n')

    def history(self, cond, n):
        """-> (transcript entries to resume, or None for a fresh conversation; the question's content blocks)."""
        if n not in self.call_line:
            # a later stage's conversation is another transcript: only a stage's first question, asked in a fresh
            # conversation, is rebuilt - from the call's own files (its question text and screenshot)
            if cond in ('A', 'D') or (self.stage_start(n) != n and not getattr(self, 'fresh', False)):
                raise SystemExit(f'call {n}: not in the first transcript - only a stage\'s first question can be '
                                 f'asked again (B/M/C/N/T; stage {self.stage(n)} starts at call {self.stage_start(n)}), '
                                 f'or any with --fresh')
            fix = {'B': lambda c: c, 'M': self.now, 'C': small, 'N': no_images, 'T': no_state}[cond]
            sd = self.seed(n, state=cond != 'T', memory=cond == 'M')
            imgs = sorted(self.calls.glob(f'{n:04d}-*.jpg'))
            q = [{'type': 'text', 'text': sd + self.prompt(n)}] + [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                             'data': base64.b64encode(p.read_bytes()).decode()}} for p in imgs]
            return None, fix(q)
        q = copy.deepcopy(self.lines[self.call_line[n]]['message']['content'])
        q = [{'type': 'text', 'text': q}] if isinstance(q, str) else q
        if cond == 'A':
            es = [e for e in self.lines[:self.call_line[n]] if e.get('type') in KEEP]
            # thinking is signed by the game's model: another model gets the history without it
            return (es if self.model == self.game_model else rechain(no_thinking(copy.deepcopy(es)))), q
        if cond == 'D':
            es = copy.deepcopy([e for e in self.lines[:self.call_line[n]] if e.get('type') in KEEP])
            return rechain(no_thinking(map_content(es, no_images))), q
        s = self.stage_start(n)
        sd = self.seed(s, state=cond != 'T', memory=cond == 'M')
        fix = {'B': lambda c: c, 'M': self.now, 'C': small, 'N': no_images, 'T': no_state}[cond]
        if s == n:
            return None, fix([{'type': 'text', 'text': sd + b['text']} if b.get('type') == 'text' else b for b in q])
        first = copy.deepcopy(self.lines[self.call_line[s]])
        c = first['message']['content']
        c = [{'type': 'text', 'text': c}] if isinstance(c, str) else c
        first['message']['content'] = [{'type': 'text', 'text': sd + b['text']} if b.get('type') == 'text' else b for b in c]
        rest = copy.deepcopy([e for e in self.lines[self.call_line[s] + 1:self.call_line[n]] if e.get('type') in KEEP])
        es = rechain(no_thinking(map_content([first] + copy.deepcopy(self.header) + rest, fix)))
        return es, fix(q)

    def held(self, stage):
        """The augments this game had taken before that stage (its log's picks), for an augment question asked now."""
        from tft_player.coach import stage_key
        picks = []
        for line in open(self.run / 'log.jsonl', encoding='utf-8'):
            if '"augment"' in line:
                e = json.loads(line)
                if e.get('ev') == 'augment' and stage_key(e.get('stage')) and stage_key(e['stage']) < stage_key(stage):
                    picks.append(e.get('name'))
        return picks

    def now(self, content):
        """Condition M: the question as the code and settings ask it now (settings_now, augment_now)."""
        return settings_now(content, self.held)

    def ask(self, entries, content, timeout=300, cwd=None):
        """cwd: the CLI's working directory - by default the climb player's folder (or the reference files); the
        Judge passes a folder of its own whose memory/ holds the playbook under test."""
        cwd = str(cwd) if cwd else (str(PLAYER_DIR) if PLAYER_DIR.exists() else str(REFERENCE))
        proj = Path.home() / '.claude' / 'projects' / re.sub(r'[^A-Za-z0-9]', '-', cwd)   # where --resume looks
        proj.mkdir(parents=True, exist_ok=True)
        sid = None
        if entries is not None:
            sid = str(uuid.uuid4())
            with open(proj / f'{sid}.jsonl', 'w', encoding='utf-8') as f:
                for e in entries:
                    f.write(json.dumps(dict(e, sessionId=sid), ensure_ascii=False) + '\n')
        cmd = [claude_cli(), '-p', '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose',
               '--model', self.model, '--effort', self.effort, '--strict-mcp-config', '--system-prompt', self.system,
               '--tools', 'Read,Grep,Glob', '--allowedTools', 'Read,Grep,Glob'] + (['--resume', sid] if sid else [])
        t0 = time.time()
        # outside the repo (the climb player folder): inside it the CLI loads the repo's CLAUDE.md
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             cwd=cwd, env=cli_env(), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        p.stdin.write((json.dumps({'type': 'user', 'message': {'role': 'user', 'content': content}}) + '\n').encode())
        p.stdin.flush()
        result, tools = None, []
        killer = threading.Timer(timeout, p.kill)
        killer.start()
        try:
            for line in p.stdout:
                if b'"tool_use"' in line and b'"type":"assistant"' in line:
                    try:
                        tools += [f"{b['name']} {json.dumps(b['input'])[:120]}" for b in json.loads(line)['message']['content']
                                  if b.get('type') == 'tool_use']
                    except (ValueError, KeyError):
                        pass
                if b'"type":"result"' in line:
                    result = json.loads(line)
                    break
        finally:
            killer.cancel()
            try:
                p.stdin.close()
                p.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                p.kill()
        err = p.stderr.read().decode('utf-8', 'replace')[-500:] if result is None else ''
        for s in {sid, (result or {}).get('session_id')} - {None}:       # the replay's own transcript
            try:
                (proj / f'{s}.jsonl').unlink()
            except OSError:
                pass
        return result, tools, time.time() - t0, err


def memory_now():
    from tft_player.coach import memory_block
    return memory_block()


def rechain(entries):
    prev = None
    for e in entries:
        e['parentUuid'], prev = prev, e['uuid']
    return entries


def no_thinking(entries):
    out = []
    for e in entries:
        if e.get('type') == 'assistant' and isinstance(e['message'].get('content'), list):
            c = [b for b in e['message']['content'] if b.get('type') not in ('thinking', 'redacted_thinking')]
            if not c:
                continue
            e['message']['content'] = c
        out.append(e)
    return out


def map_content(entries, fn):
    for e in entries:
        if e.get('type') == 'user' and isinstance(e['message']['content'], list) and \
                not any(b.get('type') == 'tool_result' for b in e['message']['content']):
            e['message']['content'] = fn(e['message']['content'])
    return entries


def no_images(content):
    return [b for b in content if b.get('type') != 'image'] or [{'type': 'text', 'text': '(no screenshot)'}]


LEVELS_RX = re.compile(r"(Levels of your comp plan \([^)]*\): ).*?( You: level \d+\.)")
LIVE_RX = re.compile(r'\n*YOUR LIVE REVIEW - .*?\(after round \d-\d\)', re.S)


def settings_now(content, held=None):
    """M: the question as it would read now - the comp plan's level line from agent/config.json as it is now, no
    floor line (removed 28 Oct), no live-review note (the live coach that wrote it read the old memory; g16's said "no XP until Elise 3"), and an
    augment question as the code asks it now (augment_now; held(stage) -> the augments taken before that stage)."""
    from tft_player.coach import climb_settings
    lv = climb_settings().get('levelling')
    out = []
    for b in content:
        if b.get('type') == 'text':
            t = LEVELS_RX.sub(lambda m: m.group(1) + lv + m.group(2), b['text']) if lv else b['text']
            t = augment_now(LIVE_RX.sub('', t), held)
            t = FLOOR_RX.sub('', t)
            t = current_board_facts(t)
            b = dict(b, text=t)
        out.append(b)
    return out


def current_board_facts(text):
    """Current-prompt replay: discard retired tier-ranked suggestions, retaining the recorded facts (e149)."""
    from tft_player.coach import current_harness_text
    out = []
    for line in current_harness_text(text).split('\n'):
        if line.startswith('Most trait bonuses your units make') or re.match(r'With .+ bought from the shop:', line):
            continue
        head, sep, ranked = line.partition(' The most trait tiers your units make')
        if sep:
            owned = re.search(r'\. ([^:\n]+: \d+ distinct owned.*)$', ranked)
            line = head + (' ' + owned.group(1) if owned else '')
        # An archived event may quote the retired ranking. Keep the event as historical execution evidence.
        line = line.replace("the 'Most trait bonuses your units make' board", 'the former trait-count ranking')
        out.append(line)
    return '\n'.join(out)


AUG_RX = re.compile(r'DECISION: augment, stage (\d-\d)')
# the owner's floor (games 32-35) was removed on 28 Oct: its calculator line and its refusals are not asked now
FLOOR_RX = re.compile(r"\n[^\n]*(?:Holding everything you have \d+ gold at|refused by the floor|the floor \(the owner's "
                      r"rule\))[^\n]*")
CARDS_LINE = 'Read the cards from the screenshot.\n'
EXPLAIN_RX = re.compile(r'Before deciding, explain .*?(?=Then end your reply)', re.S)


def augment_now(text, held=None):
    """An augment question recorded with an older format, as coach.Coach.ask writes it now: the augments held (with
    their text) and the emblem line after the cards sentence, and today's answer format (augment_explain), with the
    recorded cards, rerolls, timing and kept explanations. Other text unchanged."""
    from tft_player.coach import augment_context, augment_explain, forced_comp
    m = AUG_RX.search(text)
    if not m or CARDS_LINE not in text[m.start():]:
        return text
    head, q = text[:m.start()], text[m.start():]
    q = '\n'.join(line for line in q.split('\n') if not line.startswith(('Augments you hold: ', 'Emblems for ')))
    forced = forced_comp()
    context = augment_context(held(m.group(1)) if held else [], list(forced.units) if forced else [])
    q = q.replace(CARDS_LINE, CARDS_LINE + ''.join(f'{line}\n' for line in context), 1)
    old = EXPLAIN_RX.search(q)
    if old:
        new = augment_explain(m.group(1), forced)
        only = re.search(r'explain only the new .*? \(just rerolled\)', old.group(0))
        if only:
            new = new.replace('explain every card on the screen', only.group(0), 1)
        q = q[:old.start()] + new + q[old.end():]
    return head + q


def no_state(content):
    return [dict(b, text=STATE_RX.sub('\n(STATE text left out: read the game from the screenshot)', b['text']))
            if b.get('type') == 'text' else b for b in content]


def small(content):
    from PIL import Image
    out = []
    for b in content:
        if b.get('type') == 'image':
            im = Image.open(io.BytesIO(base64.b64decode(b['source']['data']))).convert('RGB').resize((1100, 688), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, 'JPEG', quality=85)
            b = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                             'data': base64.b64encode(buf.getvalue()).decode()}}
        out.append(b)
    return out


def run(game, out, tag, calls):
    for n in calls:
        f = out / tag / f'{n:04d}.json'
        if f.exists():
            continue
        if tag[0] == 'T' and '\nSTATE\n' not in game.prompt(n):
            continue                                  # nothing to leave out
        entries, content = game.history(tag[0], n)
        r, tools, secs, err = game.ask(entries, content)
        if r is None or r.get('is_error'):
            why = (r or {}).get('result') or err
            print(f'{tag} {n:04d} no answer, not saved: {why}', flush=True)
            if 'limit' in str(why).lower():
                sys.exit('usage limit reached: stopping')
            continue
        rec = {'tag': tag, 'call': n, 'seconds': round(secs, 1), 'tools': tools, 'error': err,
               'history_entries': None if entries is None else len(entries)}
        if r:
            rec.update({k: r.get(k) for k in ('result', 'num_turns', 'duration_ms', 'total_cost_usd', 'usage', 'is_error')})
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
        u = (r or {}).get('usage') or {}
        print(f"{tag} {n:04d} {secs:5.1f}s ctx={context(u)} ${(r or {}).get('total_cost_usd') or 0:.3f} "
              f"{summary(game, n, (r or {}).get('result'))} {err}", flush=True)


def context(u):
    u = u or {}
    return u.get('input_tokens', 0) + u.get('cache_read_input_tokens', 0) + u.get('cache_creation_input_tokens', 0)


# ---- answers in short words -------------------------------------------------------------------------------------

def unit_names(text):
    return {int(i): f"{n.strip()}{s.replace(' ', '')}" for i, n, s in re.findall(r'\[id (\d+)\] ([^(*\n]+?) ?(\**) \(', text)}


def summary(game, n, text):
    """The decision in a reply, in short words (unit ids as names from the question's state)."""
    objs = [o for o in json_objects(text or '') if isinstance(o, dict)]
    if not objs:
        return 'NO ANSWER'
    names = {}
    for k in range(game.stage_start(n), n + 1):
        names.update(unit_names(game.prompt(k)))
    nm = lambda i: names.get(i, f'id {i}?') if isinstance(i, int) else str(i)
    o = objs[0]
    if {'keep', 'pick', 'reroll'} & set(o):
        return ' '.join(f'{k} {o[k]}' for k in ('keep', 'pick', 'reroll') if k in o) + (f" ({o['name']})" if o.get('name') else '')
    if 'carousel_priority' in o:
        return 'carousel: ' + ' > '.join(o['carousel_priority'][:4])
    if 'option' in o:
        return f"option {o['option']} ({o.get('name')})"
    if 'buy' in o and len(o) == 1:
        return 'wisp: ' + ('buy' if o['buy'] else 'skip')
    ep = next((x['execution_plan'] for x in objs if isinstance(x.get('execution_plan'), dict)), None)
    if ep is None:
        return 'other: ' + json.dumps(o)[:100]
    parts = []
    for w in ep.get('wisps') or []:
        parts.append(f"wisp {'buy' if w.get('buy') else 'skip'}")
    if ep.get('sell'):
        parts.append('sell ' + ', '.join(nm(i) for i in ep['sell']))
    if ep.get('buy'):
        parts.append('buy ' + ', '.join(map(str, ep['buy'])))
    if ep.get('xp'):
        parts.append(f"xp x{ep['xp']}")
    for k in ('roll', 'fast_rr'):
        if ep.get(k):
            r = ep[k]
            want = r.get('buy')
            want = ', '.join(f'{a} {b}*' for a, b in want.items()) if isinstance(want, dict) else ', '.join(want or [])
            parts.append(f"{k} " + (f"x{r['times']} " if r.get('times') else '') +
                         (f"floor {r['gold_floor']} " if 'gold_floor' in r else '') + f'for {want}')
    if ep.get('bench'):
        parts.append('bench ' + ', '.join(nm(b.get('unit')) for b in ep['bench']))
    if ep.get('board'):
        parts.append('board ' + ', '.join(f"{nm(b['unit']) if 'unit' in b else b.get('name')}@{tuple(b.get('hex') or ())}"
                                          for b in ep['board']))
    for it in ep.get('items') or []:
        if 'give' in it:
            parts.append(f"give {it['give']}>{nm(it.get('to'))}")
        elif 'use' in it:
            parts.append(f"use {it['use']}>{nm(it.get('on'))}")
        elif 'transfer' in it:
            parts.append(f"move {it['transfer']} {nm(it.get('from'))}>{nm(it.get('to'))}")
        elif 'reforge' in it:
            parts.append(f"reforge {nm(it['reforge'])}")
        else:
            parts.append(json.dumps(it)[:60])
    for k in ('identify', 'auto_buy', 'auto_sell', 'after_fight', 'item_final', 'link', 'open_container'):
        if ep.get(k) not in (None, [], {}):
            parts.append(f'{k} ' + json.dumps(ep[k])[:80])
    ids = [i for k in ('sell', 'keep') for i in ep.get(k) or []] + \
          [b.get('unit') for k in ('board', 'bench') for b in ep.get(k) or [] if 'unit' in b] + \
          [it.get('to') or it.get('on') for it in ep.get('items') or []]
    bad = sorted({i for i in ids if isinstance(i, int) and i not in names})
    nxt = next((x['next_round'] for x in objs if isinstance(x.get('next_round'), dict)), None)
    return ('; '.join(parts) or 'nothing') + (f' [UNKNOWN unit ids {bad}]' if bad else '') + \
        (f" | next_round: {', '.join(k for k, v in nxt.items() if v not in ([], {}, None))}" if nxt else '')


def compare(game, out):
    rows, cost = [], {}
    calls = sorted({int(p.stem) for p in out.glob('*/*.json')})
    conds = [c for c in 'ADBCNT' if any(out.glob(f'{c}*/'))]
    for n in calls:
        orig = json.loads((game.calls / f'{n:04d}-result.json').read_text(encoding='utf-8'))
        first = game.prompt(n).strip().splitlines()[0]
        cells = [summary(game, n, orig['result'])]
        for c in conds:
            got = {}
            for f in sorted(out.glob(f'{c}*/{n:04d}.json')):
                r = json.loads(f.read_text(encoding='utf-8'))
                s = summary(game, n, r.get('result'))
                got[s] = got.get(s, 0) + 1
                k = cost.setdefault(c, [0, 0, 0.0, 0.0])
                k[0] += 1
                k[1] += context(r.get('usage'))
                k[2] += r.get('total_cost_usd') or 0
                k[3] += (r.get('duration_ms') or 0) / 1000
            total = sum(got.values())
            cells.append('<br>'.join(f'{s} ({v}/{total})' if total > 1 else s for s, v in got.items()) or '-')
        rows.append(f"| {n:04d} | {first[:60]} | " + ' | '.join(cells) + ' |')
    head = ['| call | question | ' + ' | '.join(NAMES[c] for c in ['O'] + conds) + ' |', '|' + '---|' * (3 + len(conds))]
    foot = ['', '| method | answers | average context (tokens) | average cost per answer (list, uncached) | average seconds |',
            '|---|---|---|---|---|'] + [f'| {NAMES[c]} | {k[0]} | {k[1] // max(k[0], 1):,} | ${k[2] / max(k[0], 1):.3f} | '
                                         f'{k[3] / max(k[0], 1):.1f} |' for c, k in cost.items()]
    return '\n'.join(head + rows + foot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='*')
    ap.add_argument('--run', default=str(ROOT / 'runs' / '20261015-132144'))
    ap.add_argument('--calls', default='')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--model', help="another model for cheap tests, e.g. claude-sonnet-5-5 (default: the game's)")
    ap.add_argument('--effort', help="low / medium / high (default: the game's)")
    ap.add_argument('--fresh', action='store_true',
                    help='B/M/C/N/T: a later question of a stage in a new conversation - the stage seed and the question '
                         '(its STATE is complete; the earlier messages of the stage are not there). Answers go to <run>-fresh/')
    a = ap.parse_args()
    game = Game(a.run)
    game.fresh = a.fresh
    game.model, game.effort = a.model or game.model, a.effort or game.effort
    out = ROOT / 'experiments' / 'token-cost' / 'replays' / Path(a.run).name
    if (game.model, game.effort) != (game.game_model, game.game_effort):     # kept apart from the game model's answers
        out = out.with_name(f'{out.name}@{game.model}-{game.effort}')
    if a.fresh:
        out = out.with_name(f'{out.name}-fresh')
    if a.compare:
        print(compare(game, out))
        return
    calls = [int(x) for x in a.calls.split(',') if x]
    for t in a.tags:                                   # one question at a time
        run(game, out, t, calls)


if __name__ == '__main__':
    main()
