"""The judge (agent/FRAMEWORK.md): two knowledge versions answer the same held-out test cases; code grades them against
the expert's choice; a proposal is accepted only if it does at least as well. No model grades anything.

    python agent/judge.py score --version HEAD [--version 2af886c] [--version none]   test-set scores
    python agent/judge.py proposal agent/proposals/<id>.json                          judge one proposal
    python agent/judge.py compare <version A> <version B>                             paired table
    python agent/judge.py approve agent/proposals/<id>.json [--note ...]              the owner's OK: ship it

A version is a commit of this repo (its agent/memory), "none" (no memory) or "dir:<path>" (a memory folder). Each answer
is cached under agent/judge-cache/<version hash>/<case>.json, so a case is asked once per version.

Grading, per kind:
- tempo: 1 point for the level at the end of the phase, 1 for the gold kept (within GOLD_TOL of the expert's, or both at
  or above 50 - the interest cap); gold is not graded at 2-1 (augment gold) or on PvE rounds (x-7: loot).
- augment: 2 points for the expert's card.
"""
import argparse
import base64
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
AGENT = ROOT / 'agent'
CACHE = AGENT / 'judge-cache'
VERDICTS = AGENT / 'verdicts'
# the Player's own model and effort (agent/config.json), so a verdict measures what the games would play
_PLAYER = json.loads((AGENT / 'config.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(AGENT))
import backends  # noqa: E402  (Codex used up: the fallback, as the games then play it)
_MODEL, EFFORT = backends.choose(_PLAYER['model'], _PLAYER['effort'], _PLAYER)
MODEL = _MODEL.split(':', 1)[-1]
CODEX = _MODEL.startswith('codex:')
GOLD_TOL = 10
# nothing ships on a test set this small: with one game's correlated cases, repeated proposals would fit that game
MIN_TEST_GAMES, MIN_TEST_CASES = 3, 60
SAMPLES = 3                   # repeat samples per version, averaged before any comparison (E2.3)
WORKERS = 4                   # questions at once: 72 in 158 s against ~8 s each one by one, same answers' cost (E1.6)
QUESTION_VERSION = 2          # 2: the expert's line in the question (E2.2: the judge rewarded one style otherwise)
MEMORY_FILES = [('playbook.md', 'PLAYBOOK'), ('comps', 'COMP NOTES'), ('harness.md', 'HARNESS NOTES')]

SYSTEM = ('You are the player of a Teamfight Tactics agent (Set 18). You are shown one moment of a Ranked game - its '
          'screenshot and the numbers read from it - and decide what you would do, as in your own games, using your '
          'memory when you have one. The game may be another player\'s, on another server (its client language may '
          'differ: names are the same champions, traits and items). Answer with the JSON object asked for last.')


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, encoding='utf-8').stdout


def split_version(version):
    """'<memory>+cases' -> ('<memory>', True): the memory, and whether expert cases (agent/retrieval.py) come with each
    question. A trailing '#<n>' (a repeat sample: the same questions asked again, its own cache) is dropped here."""
    version = version.split('#')[0]
    return (version[:-len('+cases')], True) if version.endswith('+cases') else (version, False)


def sample_of(version):
    return version.split('#')[1] if '#' in version else ''


def memory_text(version):
    """The memory block the player would read for this version (the same layout as tft_player.coach.memory_block)."""
    version = split_version(version)[0]
    if version == 'none':
        return ''
    parts = []
    if version.startswith('dir:'):
        base = Path(version[4:])
        files = [(p.relative_to(base).as_posix(), p.read_text(encoding='utf-8')) for p in sorted(base.rglob('*.md'))]
    else:
        names = [n for n in git('ls-tree', '-r', '--name-only', version, 'agent/memory').splitlines() if n.endswith('.md')]
        files = [(n[len('agent/memory/'):], git('show', f'{version}:{n}')) for n in names]
    for key, title in MEMORY_FILES:
        for name, text in files:
            if (name == key or name.startswith(key + '/')) and text.strip():
                parts.append(f'=== {title} ({Path(name).name}) ===\n{text.strip()}')
    if not parts:
        return ''
    return ('YOUR MEMORY - written by you (the same model, as your own coach) after reviewing your earlier games on '
            'this account.\n\n' + '\n\n'.join(parts))


def version_key(version):
    """A short hash of what the player reads for this version (equal memories share answers)."""
    extra = ''
    if split_version(version)[1]:
        from retrieval import fingerprint
        extra = '+cases' + fingerprint()
    extra += ('#' + sample_of(version)) if sample_of(version) else ''
    return hashlib.sha1((MODEL + EFFORT + SYSTEM + str(QUESTION_VERSION) + memory_text(version) + extra).encode()).hexdigest()[:12]


def resolve(version):
    base, cases = split_version(version)
    if not (base == 'none' or base.startswith('dir:')):
        base = git('rev-parse', '--short', base).strip() or base
    return base + ('+cases' if cases else '') + (('#' + sample_of(version)) if sample_of(version) else '')


def question(case):
    st = case['state']
    if case['kind'] == 'tempo':
        hud = (f"Stage {case['stage']}, the start of its planning phase. Level {st.get('level')}, XP "
               f"{'/'.join(map(str, st['xp'])) if st.get('xp') else '?'}, gold {st.get('gold')}"
               + (f", HP {st['hp']}" if st.get('hp') is not None else '') + '.')
        line = f" This game's board is built toward: {st['line']}." if st.get('line') else ''
        return (hud + line + ' The board, bench, items, shop and the other players are in the screenshot.\n\n'
                'This planning phase: what level do you end it at, and how much gold do you keep at its end '
                '(after your buys, rerolls and XP)? Reply with one line of JSON: '
                '{"level_end": <level>, "gold_end": <gold>, "why": "<one sentence>"}')
    cards = ' / '.join(st.get('cards') or [])
    note = f" {st['note']}" if st.get('note') else ''
    return (f"Stage {case['stage']}: the augment choice. The cards on screen: {cards}.{note}\n\n"
            'Which card do you take? Reply with one line of JSON: {"pick": "<the card as written>", "why": "<one sentence>"}')


def ask(prompt, image, timeout=300, cwd=None):
    """One question to the player model in a fresh conversation (its CLI, as the game uses it). cwd: its
    working directory (default the player's folder; the Judge's held-out test passes one without memory/)."""
    from tft_player.coach import PLAYER_DIR, REFERENCE, claude_cli, cli_env
    if CODEX:
        return ask_codex(prompt, image, timeout, cwd)
    content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                            'data': base64.standard_b64encode(Path(image).read_bytes()).decode()}},
               {'type': 'text', 'text': prompt}]
    cmd = [claude_cli(), '-p', '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose',
           '--model', MODEL, '--effort', EFFORT, '--strict-mcp-config', '--system-prompt', SYSTEM,
           '--tools', 'Read,Grep,Glob', '--allowedTools', 'Read,Grep,Glob']
    cwd = str(cwd) if cwd else (str(PLAYER_DIR) if PLAYER_DIR.exists() else str(REFERENCE))
    env = cli_env()
    for k in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN'):
        env.pop(k, None)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    p.stdin.write((json.dumps({'type': 'user', 'message': {'role': 'user', 'content': content}}) + '\n').encode())
    p.stdin.flush()
    result = None
    killer = threading.Timer(timeout, p.kill)
    killer.start()
    try:
        for line in p.stdout:
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
    return result


def ask_codex(prompt, image, timeout=300, cwd=None):
    """ask() for a Codex player (agent/config.json "codex:<model>"): one `codex exec` run with the game's flags
    (tft_player.coach._CodexSession), the screenshot attached. -> {'result': its answer} like the Claude CLI's."""
    import tempfile
    from tft_player.coach import PLAYER_DIR, REFERENCE, codex_cli, codex_sandbox, codex_service_tier
    instr = Path(tempfile.gettempdir()) / f'tft-judge-instructions-{os.getpid()}.md'
    instr.write_text(SYSTEM, encoding='utf-8')
    cmd = [codex_cli(), 'exec', '--json', '--ignore-user-config', '--ignore-rules', '--skip-git-repo-check',
           '-s', 'read-only', '-c', 'approval_policy="never"', '-m', MODEL, '-c', f'model_reasoning_effort="{EFFORT}"',
           '-c', f'model_instructions_file="{instr.as_posix()}"']
    if codex_sandbox():
        cmd += ['-c', f'windows.sandbox="{codex_sandbox()}"']
    if codex_service_tier():
        cmd += ['-c', f'service_tier="{codex_service_tier()}"']
    cmd += ['-i', str(image), '-']
    cwd = str(cwd) if cwd else (str(PLAYER_DIR) if PLAYER_DIR.exists() else str(REFERENCE))
    try:
        p = subprocess.run(cmd, input=prompt.encode('utf-8'), capture_output=True, cwd=cwd, timeout=timeout,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired:
        return None
    said = []
    for line in p.stdout.decode('utf-8', 'replace').splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        item = ev.get('item') or {}
        if ev.get('type') == 'item.completed' and item.get('type') == 'agent_message':
            said.append(item.get('text') or '')
    return {'result': '\n'.join(said).strip()} if said else None


def parse(text):
    for m in re.finditer(r'\{[^{}]*\}', text or ''):
        try:
            return json.loads(m.group())
        except ValueError:
            continue
    return {}


def grade(case, ans):
    """(points, out of) for one answer against the expert."""
    ex = case['expert']
    if case['kind'] == 'augment':
        pick = str(ans.get('pick') or '').strip()
        return (2 if pick and pick == ex['pick'] else 0), 2
    pts, of = 0, 1
    try:
        pts += int(ans.get('level_end')) == ex['level_end']
    except (TypeError, ValueError):
        pass
    a, b = map(int, case['stage'].split('-'))
    if not (case['stage'] == '2-1' or b == 7):
        of += 1
        try:
            g = int(ans.get('gold_end'))
            pts += abs(g - ex['gold_end']) <= GOLD_TOL or (g >= 50 and ex['gold_end'] >= 50)
        except (TypeError, ValueError):
            pass
    return pts, of


def answers(version, cases, verbose=True):
    """{case id: (answer dict, points, out of)} for this version, asking only the cases not cached yet."""
    from cases import KEYS
    key = version_key(version)
    folder = CACHE / key
    folder.mkdir(parents=True, exist_ok=True)
    mem = memory_text(version)

    def asked(c):
        cases_block = ''
        if split_version(version)[1] and c['kind'] == 'tempo':
            from retrieval import block
            cases_block = block(c['stage'], c['state'].get('level') or 1, c['state'].get('gold') or 0)
        prompt = (mem + '\n\n' if mem else '') + (cases_block + '\n\n' if cases_block else '') + question(c)
        t0 = time.time()
        res = ask(prompt, KEYS / c['game'] / c['state']['keyframe']) or {}
        rec = {'case': c['id'], 'version': version, 'text': res.get('result') or '', 'cost': res.get('total_cost_usd'),
               'seconds': round(time.time() - t0, 1)}
        (folder / f"{c['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
        if verbose:
            print(f"  {version} {c['id']}: {rec['seconds']} s", flush=True)

    todo = [c for c in cases if not (folder / f"{c['id']}.json").exists()]
    if todo:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(asked, todo))
    out = {}
    for c in cases:
        f = folder / f"{c['id']}.json"
        rec = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {'text': ''}
        ans = parse(rec['text'])
        out[c['id']] = (ans, *grade(c, ans))
    return out


def test_cases():
    from cases import load, split_of
    return [c for c in load() if split_of(c['game']) == 'test']


def totals(cases, got):
    per = {}
    for c in cases:
        _, p, o = got[c['id']]
        k = per.setdefault(c['kind'], [0, 0])
        k[0] += p
        k[1] += o
    per = {k: [round(v[0], 2), v[1]] for k, v in per.items()}
    return {'total': [round(sum(v[0] for v in per.values()), 2), sum(v[1] for v in per.values())], **per}


def sign_p(better, worse):
    """Two-sided sign test p-value for better vs worse paired cases."""
    n = better + worse
    if not n:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def decide(ta, tb, lost, games, n):
    """(accept, provisional, reason): more points in total, no kind worse, no anchor lost, and a test set large enough."""
    small = games < MIN_TEST_GAMES or n < MIN_TEST_CASES
    kinds_ok = all(tb.get(k, [0])[0] >= v[0] for k, v in ta.items() if k != 'total')
    accept = tb['total'][0] > ta['total'][0] and kinds_ok and not lost and not small
    reason = ('accepted: more points on the test cases, no kind worse, no anchor lost' if accept else
              f'provisional: the test set has {n} cases from {games} game(s); nothing ships below '
              f'{MIN_TEST_CASES} cases from {MIN_TEST_GAMES} games' if small else
              'rejected: ' + ('an anchor case lost: ' + ', '.join(lost) if lost else
                              'not more points than the current version' if tb['total'][0] <= ta['total'][0] else
                              'a kind of decision scored worse'))
    return accept, small, reason


def label(version):
    """A file-name-safe short name for a version (a proposal's temporary memory folder by its content hash)."""
    base, cases = split_version(version)
    if base.startswith('dir:'):
        base = 'memory-' + version_key(base)[:8]
    return re.sub(r'[^A-Za-z0-9_.+#-]', '_', base + ('+cases' if cases else ''))


def mean_answers(version, cases, samples):
    """{case id: (first answer, mean points, out of)} over `samples` repeat samples of a version (E2.3: one sample of
    the same memory scored 28, 34 and 30 of 43 - a single run cannot tell apart differences under ~6 points)."""
    runs = [answers(version if s == 0 else f'{version}#{s + 1}', cases) for s in range(samples)]
    return {c['id']: (runs[0][c['id']][0], sum(r[c['id']][1] for r in runs) / samples, runs[0][c['id']][2]) for c in cases}


def compare(base, cand, cases, proposal=None, anchors=(), samples=None):
    samples = samples or SAMPLES
    a, b = mean_answers(base, cases, samples), mean_answers(cand, cases, samples)
    ta, tb = totals(cases, a), totals(cases, b)
    better = sum(b[c['id']][1] > a[c['id']][1] for c in cases)
    worse = sum(b[c['id']][1] < a[c['id']][1] for c in cases)
    lost = [x for x in anchors if x in a and a[x][1] == a[x][2] and b[x][1] < a[x][1]]
    accept, small, reason = decide(ta, tb, lost, len({c['game'] for c in cases}), len(cases))
    return {'id': f"{datetime.now():%Y%m%d-%H%M%S}-{label(base)}-vs-{label(cand)}",
            'proposal': proposal, 'base': base, 'candidate': cand, 'model': f'{MODEL} {EFFORT}', 'cases': len(cases), 'samples': samples,
            'score': {'base': ta, 'candidate': tb},
            'paired': {'better': better, 'worse': worse, 'same': len(cases) - better - worse, 'sign_p': round(sign_p(better, worse), 3)},
            # ids and points only: verdicts are read by the proposer, so they carry no test state, label or answer
            # (those stay in agent/judge-cache/, which no model reads)
            'per_case': [{'case': c['id'], 'base_points': a[c['id']][1], 'candidate_points': b[c['id']][1],
                          'of': a[c['id']][2]} for c in cases],
            'accept': accept, 'provisional': small, 'question_version': QUESTION_VERSION, 'reason': reason,
            'when': datetime.now().isoformat(timespec='seconds')}


def write_verdict(v):
    import jsonschema
    jsonschema.validate(v, json.loads((AGENT / 'schemas' / 'verdict.schema.json').read_text(encoding='utf-8')))
    VERDICTS.mkdir(parents=True, exist_ok=True)
    path = VERDICTS / f"{v['id']}.json"
    path.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding='utf-8')
    sys.path.insert(0, str(AGENT))
    from channel import post
    b, c = v['score']['base']['total'], v['score']['candidate']['total']
    what = f"proposal {v['proposal']}" if v.get('proposal') else f"version {label(v['candidate'])}"
    outcome = ('Shipped.' if v.get('shipped', v['accept']) else 'Accepted; it reaches the games when the owner approves it.'
               if v.get('awaiting') == 'owner' else 'Provisional: the test set is still too small to ship anything.'
               if v.get('provisional') else 'Rejected: it did not do better.')
    post('Judge', f"I measured {what} against {label(v['base'])} on {v['cases']} held-out decisions "
                  f"({v.get('samples', 1)} sample{'s' if v.get('samples', 1) > 1 else ''}): {b[0]} -> {c[0]} of {b[1]} points, "
                  f"{v['paired']['better']} better and {v['paired']['worse']} worse. {outcome}",
         refs=[str(path.relative_to(ROOT))])
    return path


def ship(prop, path, message, extra=()):
    (ROOT / prop['file']).write_text(prop['new_text'], encoding='utf-8')
    subprocess.run(['git', 'add', prop['file'], str(Path(path).resolve().relative_to(ROOT)), *extra], cwd=ROOT)
    subprocess.run(['git', 'commit', '-q', '-m', message], cwd=ROOT)


def judge_proposal(path):
    """Apply the proposal to a copy of the current memory and compare on the test cases. An accepted change waits for
    the owner's OK (`judge.py approve`, agent/FRAMEWORK.md rule 1: the owner decides what reaches the games); the
    owner's own proposals (author "owner") ship at once, before the comparison, which then only measures them."""
    import shutil
    import tempfile
    import jsonschema
    prop = json.loads(Path(path).read_text(encoding='utf-8'))
    jsonschema.validate(prop, json.loads((AGENT / 'schemas' / 'proposal.schema.json').read_text(encoding='utf-8')))
    base = resolve('HEAD')
    owner = prop['author'] == 'owner'
    if owner:
        ship(prop, path, f"Owner's rule shipped: {prop['id']} (the judge measures it next)")
    tmp = Path(tempfile.mkdtemp(prefix='judge-'))
    shutil.copytree(AGENT / 'memory', tmp / 'memory')
    target = tmp / 'memory' / prop['file'][len('agent/memory/'):]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(prop['new_text'], encoding='utf-8')
    v = compare(base, f"dir:{tmp / 'memory'}", test_cases(), proposal=prop['id'], anchors=prop.get('anchors') or [])
    v['candidate'] = f"proposal {prop['id']} on {base}"
    v['shipped'] = owner
    if v['accept'] and not owner:
        v['awaiting'] = 'owner'
    if owner:
        v['reason'] = "the owner's rule: shipped by their authority; measured only (" + v['reason'] + ')'
    out = write_verdict(v)
    if owner:
        subprocess.run(['git', 'add', str(out.relative_to(ROOT))], cwd=ROOT)
        subprocess.run(['git', 'commit', '-q', '-m', f"Judge: measured the owner's rule {prop['id']}"], cwd=ROOT)
    else:
        subprocess.run(['git', 'add', str(out.relative_to(ROOT))], cwd=ROOT)
        subprocess.run(['git', 'commit', '-q', '-m', f"Judge: verdict on {prop['id']} ({v['reason'][:80]})"], cwd=ROOT)
    shutil.rmtree(tmp, ignore_errors=True)
    return v, out


def approve(proposal_path, note=''):
    """The owner's OK for a change the judge accepted: write it into agent/memory and commit (the only way a model's
    knowledge change reaches the games). Refused when its verdict did not accept it or the memory changed since."""
    prop = json.loads(Path(proposal_path).read_text(encoding='utf-8'))
    vs = sorted(VERDICTS.glob('*.json'))
    vf = next((f for f in reversed(vs) if json.loads(f.read_text(encoding='utf-8')).get('proposal') == prop['id']), None)
    if not vf:
        sys.exit(f"no verdict for {prop['id']}")
    v = json.loads(vf.read_text(encoding='utf-8'))
    if v.get('awaiting') != 'owner':
        sys.exit(f"{prop['id']}: not waiting for the owner ({v['reason']})")
    base = v['candidate'].rsplit(' on ', 1)[-1]
    if memory_text(base) != memory_text(resolve('HEAD')):
        sys.exit(f"{prop['id']}: the memory changed since its verdict (judged on {base}); judge it again first")
    v.pop('awaiting')
    v['shipped'] = True
    v['owner'] = 'approved' + (f': {note}' if note else '')
    vf.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding='utf-8')
    ship(prop, proposal_path, f"Owner approved {prop['id']}; shipped (verdict {vf.stem})", [str(vf.relative_to(ROOT))])
    sys.path.insert(0, str(AGENT))
    from channel import post
    post('Owner', f"I approved {prop['id']}; it is in the memory the next games use." + (f' {note}' if note else ''),
         refs=[str(vf.relative_to(ROOT))])


STAGE_SAMPLES = 3            # each decision asked 3 times per version (the owner's master plan, 2026-11-01)
# a stage strategy may not lose more than 5% of a set's points: three samples of the same text still vary about 8%
# (E2.3), and the first real test (stage 3, 2026-11-01) lost 1.3 of 40 and 1 of 22 points at sign-test p 0.38 and 1.0
STAGE_TOL = 0.05


def comp_cases(comp, stage):
    """(in-sample, held-out) recorded decisions of the comp's strong players' games in this stage (stages 5-7 count
    as 5). Held-out = the comp's games agent/stages.py keeps from the coach and the Strategist."""
    import cases as cases_mod
    import comp_coach
    import stages
    games = {g['game'] for g in comp_coach.games_of(comp)}
    stages.heldout(comp, games)
    base = lambda g: re.sub(r'-g\d+$', '', g)
    mine = [c for c in cases_mod.load() if base(c['game']) in games and c.get('kind') in ('tempo', 'augment')
            and stages.stage_of(c['stage']) == str(stage)]
    held = [c for c in mine if stages.is_heldout(comp, c['game'])]
    return [c for c in mine if c not in held], held


def stage_version(comp, stage, text, v='candidate'):
    """A memory folder the player reads for one stage strategy text: the comp notes as the games give them, then the
    STAGE n STRATEGY block (the same words a game's call carries). -> 'dir:<folder>' (cached by its content)."""
    import comp_coach
    import stages
    notes = AGENT / 'memory' / 'comps' / f'{comp_coach.slug(comp)}.md'
    body = (notes.read_text(encoding='utf-8').strip() + '\n\n') if notes.exists() else ''
    if text:
        body += stages.block(comp, str(stage), {'v': v, 'text': text})
    # the owner's rules for every comp, as a game gives them (agent/memory/general.md; x19.2)
    general = AGENT / 'memory' / 'general.md'
    every = general.read_text(encoding='utf-8') if general.exists() else ''
    key = hashlib.sha1((every + body).encode('utf-8')).hexdigest()[:12]
    folder = CACHE / 'stage-versions' / key / 'comps'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{comp_coach.slug(comp)}.md').write_text(body or ' ', encoding='utf-8')
    if every:
        (folder.parent / 'general.md').write_text(every, encoding='utf-8')
    return f'dir:{folder.parent}'


def stage_gate(comp, stage, base_text, cand_text, samples=None):
    """The Judge on a stage strategy edit: the current and the new text answer the same recorded decisions of the
    comp's strong players in that stage, each asked `samples` times; graded by code against the strong player's choice.
    Accepted when the new text scores no lower on the held-out games and no lower on the rest. -> verdict dict."""
    samples = samples or STAGE_SAMPLES
    ins, held = comp_cases(comp, stage)
    a, b = stage_version(comp, stage, base_text, 'current'), stage_version(comp, stage, cand_text)
    out = {'comp': comp, 'stage': str(stage), 'model': f'{MODEL} {EFFORT}', 'samples': samples,
           'when': datetime.now().isoformat(timespec='seconds')}
    for name, cs in (('in_sample', ins), ('held_out', held)):
        if not cs:
            out[name] = {'cases': 0}
            continue
        ra, rb = mean_answers(a, cs, samples), mean_answers(b, cs, samples)
        ta, tb = totals(cs, ra), totals(cs, rb)
        better = sum(rb[c['id']][1] > ra[c['id']][1] for c in cs)
        worse = sum(rb[c['id']][1] < ra[c['id']][1] for c in cs)
        out[name] = {'cases': len(cs), 'games': len({c['game'] for c in cs}), 'base': ta['total'],
                     'candidate': tb['total'], 'better': better, 'worse': worse,
                     'sign_p': round(sign_p(better, worse), 3)}
    tested = [k for k in ('in_sample', 'held_out') if out[k].get('cases')]
    drop = [k for k in tested if out[k]['candidate'][0] < out[k]['base'][0] - STAGE_TOL * out[k]['base'][1]]
    out['tested'] = bool(tested)
    out['accept'] = not drop
    out['reason'] = ('no recorded decisions of this stage to test it on: ships untested' if not tested else
                     'accepted: no more than 5% fewer points (sampling noise) than the current version on '
                     + ' and '.join(tested) if not drop else
                     'rejected: more than 5% fewer points than the current version on ' + ' and '.join(drop))
    # the round cases of our own games decide first (the owner, 2026-11-10: a rewrite is accepted when it passes the
    # cases it targets and breaks none of the others): a case of the stage that passes with the current text and not
    # with the new one rejects it; one it makes pass, with none broken, accepts it
    rc = round_case_gate(comp, stage, base_text, cand_text, samples)
    out['round_cases'] = rc
    if rc.get('broken'):
        out['accept'] = False
        out['reason'] = (f"rejected: round case(s) {', '.join(rc['broken'])} pass with the current text and fail with "
                         f"the new one")
    elif rc.get('fixed'):
        out['accept'] = True
        out['tested'] = True
        out['reason'] = (f"accepted: it makes round case(s) {', '.join(rc['fixed'])} pass and breaks none of the "
                         f"stage's {rc.get('cases')} (strong players' decisions: {out['reason']})")
    elif rc.get('error'):
        out['reason'] += f" (round cases not run: {rc['error'][:200]})"
    return out


def round_case_gate(comp, stage, base_text, cand_text, samples):
    """The stage's decision cases (agent/round_cases.py stage) with both texts, in a fresh process (the code on disk)
    -> {'cases', 'base', 'candidate', 'broken', 'fixed'} or {'error'}."""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix='tft-judge-'))
    (d / 'base.txt').write_text(base_text or '', encoding='utf-8')
    (d / 'candidate.txt').write_text(cand_text or '', encoding='utf-8')
    p = subprocess.run([sys.executable, str(AGENT / 'round_cases.py'), 'stage', '--comp', comp, '--stage', str(stage),
                        '--base', str(d / 'base.txt'), '--candidate', str(d / 'candidate.txt'), '--samples',
                        str(samples)], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {'error': (p.stdout + p.stderr)[-600:] or 'no output'}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.path.insert(0, str(AGENT))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score')
    s.add_argument('--version', action='append', required=True)
    c = sub.add_parser('compare')
    c.add_argument('base')
    c.add_argument('candidate')
    p = sub.add_parser('proposal')
    p.add_argument('path')
    o = sub.add_parser('approve')
    o.add_argument('path')
    o.add_argument('--note', default='')
    g = sub.add_parser('stage')
    g.add_argument('comp')
    g.add_argument('stage')
    g.add_argument('candidate', help='a file with the new text')
    g.add_argument('--samples', type=int, default=STAGE_SAMPLES)
    a = ap.parse_args()
    if a.cmd == 'stage':
        import stages
        cur = stages.current(a.comp).get(a.stage)
        v = stage_gate(a.comp, a.stage, cur['text'] if cur else '', Path(a.candidate).read_text(encoding='utf-8'),
                       a.samples)
        print(json.dumps(v, ensure_ascii=False, indent=1))
        return
    if a.cmd == 'approve':
        return approve(a.path, a.note)
    cases = test_cases()
    if a.cmd == 'score':
        for ver in a.version:
            r = resolve(ver)
            t = totals(cases, answers(r, cases))
            print(f'{ver} ({r}): ' + ', '.join(f'{k} {v[0]}/{v[1]}' for k, v in t.items()))
    elif a.cmd == 'compare':
        v = compare(resolve(a.base), resolve(a.candidate), cases)
        out = write_verdict(v)
        print(json.dumps({k: v[k] for k in ('score', 'paired', 'accept', 'reason')}, ensure_ascii=False), '->', out)
    else:
        v, out = judge_proposal(a.path)
        print(json.dumps({k: v[k] for k in ('score', 'paired', 'accept', 'reason')}, ensure_ascii=False), '->', out)


if __name__ == '__main__':
    main()
