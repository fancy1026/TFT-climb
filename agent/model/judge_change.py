"""The Judge (the owner's structure v3, section 6): a proposed change to the game model is tested before it goes live.
Code scores every answer; no model grades its own proposal, and the bounds are fixed here, before any test runs.

1. Reproduction, on the games the change came from: each source decision (a run and a round) is asked again as the
   game asked it (tools/replay_decisions.py condition M: that stage's conversation rebuilt, with the memory as given),
   twice with the current playbook (what it gives by chance) and twice with the candidate. Each answer is checked
   against the proposal's stated expectations ("expect": checks on the reply's JSON). Passes when the candidate meets
   them in at least half of its answers and in more of them than the current playbook.
2. Held out, on decisions it never saw: strong players' held-out decisions (agent/model/replay.py), each answered
   three times by the current model and once with the change. Fails when it is worse than another answer of the
   current model more often than better beyond chance (sign test, p <= 0.05), or its side effects are worse.
3. The verdict -> agent/model/replays/<id>.json and .md; accepted -> into the game model (active), the playbook
   rebuilt (agent/memory/playbook.md) and committed - the next game's player reads it.

    python agent/model/judge_change.py agent/model/proposals/<id>.json [--no-apply] [--skip-held-out]

A proposal's reproduction cases: "reproduce": [{"run": "<run id>", "round": "3-5", "call": <n, optional>,
"expect": [{"path": "execution_plan.xp", "op": ">=", "value": 6}, ...], "why": "..."}]. Paths go into the merged
JSON objects of the reply (execution_plan, next_round, strategy, ...); ops: == != >= <= > < (a missing number counts
as 0), contains / not_contains (a list holding the value, or a dict holding it among its values, any depth, case
ignored), present / absent. "board.<...>" paths check the board the reply's plan would make from the question's state
(agent/model/plan_board.py): board.units (contains "Sentinel"), board.trait.<Trait> (>= 3), board.active_traits
(traits at a breakpoint of 2 or more), board.items.<Champion> (contains "Warmog's Armor").
"""
import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
if __name__ == '__main__':                      # the climb's memory block (not when a test imports this file)
    os.environ.setdefault('TFT_CLIMB', '1')
import decide  # noqa: E402
import modelfile  # noqa: E402
import replay  # noqa: E402

OUT = HERE / 'replays'
SAMPLES = 3                     # answers per decision, current and candidate (2 until 29 Oct: the Director found
                                # 1 proposal in 9 accepted on 6-10 answers against a 13-19% flip rate)
PASS_SHARE = 0.5                # the candidate meets the expectations in at least this share of its answers
HARM_P = 0.05                   # held out: worse than better beyond this is harm


# --- checks on a reply ----------------------------------------------------------------------------------------

def reply_objects(text):
    from tft_player.coach import json_objects
    out = {}
    for o in json_objects(text or ''):
        if isinstance(o, dict):
            out.update(o)
    return out


def get(obj, path):
    cur = obj
    for part in path.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def holds(container, value):
    v = str(value).lower()
    if isinstance(container, dict):
        return any(holds(x, value) for x in container.values())
    if isinstance(container, list):
        return any(holds(x, value) for x in container)
    return container is not None and str(container).lower() == v


def number(x):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, dict) and 'times' in x:                 # a roll: {"times": n}
        return number(x['times'])
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0


def check(obj, c):
    """One expectation against a reply's merged JSON -> (met, what was found)."""
    val = get(obj, c['path'])
    op = c.get('op', '==')
    if op == 'present':
        return val is not None, val
    if op == 'absent':
        return val is None, val
    if op in ('contains', 'not_contains'):
        h = holds(val, c['value'])
        return (h if op == 'contains' else not h), val
    a, b = number(val), number(c['value'])
    return {'==': a == b, '!=': a != b, '>=': a >= b, '<=': a <= b, '>': a > b, '<': a < b}[op], val


def meets(text, expect, plan='execution_plan', question=None):
    """All checks against a reply; a path starting "plan." means the plan this question decides the round with -
    execution_plan, or next_round for a round run from a plan written ahead. A path starting "board." is the board
    that plan would make from the question's state (agent/model/plan_board.py): board.units, board.trait.<Trait>,
    board.active_traits, board.items.<Champion>."""
    obj = reply_objects(text)
    if question and any(str(c.get('path', '')).startswith('board.') for c in expect):
        import plan_board
        obj = dict(obj, board=plan_board.view(question, obj.get(plan)))
    found = [check(obj, dict(c, path=plan + c['path'][4:] if c['path'].startswith('plan.') else c['path']))
             for c in expect]
    return all(ok for ok, _ in found), [v for _, v in found]


# --- reproduction -----------------------------------------------------------------------------------------------

ROUND_RX = re.compile(r'STATE\s+Stage (\d-\d)[.,]')


def call_round(game, n):
    m = ROUND_RX.search(game.prompt(n))
    return m.group(1) if m else None


def round_index(rnd):
    a, b = (int(x) for x in rnd.split('-'))
    return a * 10 + b


def decision_call(game, rnd):
    """(call, plan key) that decided the round: its 'DECISION: planning phase' question (execution_plan); for a round
    run from a plan written ahead (game 35: no planning question from 4-1 to 4-6), the last earlier question whose
    reply carried a next_round, with no planning question between it and the round (next_round). (None, None) if
    neither."""
    calls = sorted(int(p.name[:4]) for p in game.calls.glob('*-prompt.txt') if p.name[:4].isdigit())
    rounds = {n: call_round(game, n) for n in calls}
    planning = [n for n in calls if game.prompt(n).startswith('DECISION: planning phase')]
    for n in planning:
        if rounds[n] == rnd:
            return n, 'execution_plan'
    ahead = [n for n in calls if rounds[n] and round_index(rounds[n]) < round_index(rnd)
             and '"next_round"' in game.reply(n)]
    if ahead and not any(ahead[-1] < n and rounds[n] and round_index(rounds[n]) < round_index(rnd) for n in planning):
        return ahead[-1], 'next_round'
    return None, None


def player_folder(memory_dir=None):
    """A working directory for the CLI like the player's (the reference files and memory/), with this memory - the
    candidate's answers must not find the current playbook under memory/ (coach.memory_block tells the player it is
    there)."""
    import shutil
    import tempfile
    from tft_player.coach import MEMORY, REFERENCE
    d = Path(tempfile.mkdtemp(prefix='judge-player-'))
    shutil.copytree(REFERENCE, d, dirs_exist_ok=True)
    shutil.copytree(memory_dir or MEMORY, d / 'memory')
    return d


def transformed(content, transform):
    """A state proposal's change applied to the question's text blocks: [{"old": "<regex>", "new": "<text>"}], each
    re.sub'd once (\\1 etc. as in re); a change that matches nothing leaves the question as it was."""
    out = []
    for b in content:
        if b.get('type') == 'text':
            t = b['text']
            for tr in transform or []:
                t = re.sub(tr['old'], tr['new'], t, count=1, flags=re.S)
            b = dict(b, text=t)
        out.append(b)
    return out


def reproduce(prop, cand_dir, workers=2):
    """-> {'cases': [...], 'current': hits, 'candidate': hits, 'answers': n each, 'pass': bool}."""
    import replay_decisions as rd
    jobs, cases = [], []
    folders = {'current': player_folder(),
               'candidate': player_folder(None if prop.get('type') == 'state' else cand_dir)}
    for case in prop.get('reproduce') or []:
        run = ROOT / 'runs' / case['run']
        if not (run / 'calls').exists():
            cases.append({**case, 'error': f"no recorded questions in runs/{case['run']}"})
            continue
        os.environ.pop('TFT_FORCE_COMP', None)      # the game's own comp (Game reads it off the brief)
        game = rd.Game(run)
        game.fresh = True                    # a later stage's question is asked in a new conversation with its digest
        n, plan = (case['call'], case.get('plan', 'execution_plan')) if case.get('call') else \
            decision_call(game, case['round'])
        if n is None:
            cases.append({**case, 'error': f"no question decided {case['round']} in {case['run']}"})
            continue
        row = {**case, 'call': n, 'plan': plan, 'current': [], 'candidate': []}
        cases.append(row)
        state = prop.get('type') == 'state'
        for side, mem in (('current', None), ('candidate', None if state else str(cand_dir))):
            if mem:
                os.environ['TFT_MEMORY_DIR'] = mem
            else:
                os.environ.pop('TFT_MEMORY_DIR', None)
            entries, content = game.history('M', n)          # built now: the memory folder is read here
            if state and side == 'candidate':
                content = transformed(content, prop.get('transform'))
                if json.dumps(content) == json.dumps(game.history('M', n)[1]):
                    row['unchanged'] = 'the state change matches nothing in this question'
            for _ in range(SAMPLES):
                jobs.append((row, side, game, entries, content))
    os.environ.pop('TFT_MEMORY_DIR', None)

    def ask(job):
        row, side, game, entries, content = job
        result, _, secs, err = game.ask(entries, content, cwd=folders[side])
        text = (result or {}).get('result') or ''
        ok, found = meets(text, row['expect'], row['plan'], question=game.prompt(row['call'])) if text else (False, None)
        st = reply_objects(text).get('strategy') if text else None
        return row, side, {'met': ok, 'found': found, 'seconds': round(secs), 'error': err[-200:] if not text else '',
                           'cost': (result or {}).get('total_cost_usd'),
                           'why': st.get('why') if isinstance(st, dict) else None}

    with ThreadPoolExecutor(workers) as ex:
        for row, side, ans in ex.map(ask, jobs):
            row[side].append(ans)
    cur = sum(a['met'] for c in cases for a in c.get('current', []))
    cand = sum(a['met'] for c in cases for a in c.get('candidate', []))
    total = sum(len(c.get('candidate', [])) for c in cases)
    ok = total > 0 and cand >= math.ceil(PASS_SHARE * total) and cand > cur
    why = [] if ok else [f'reproduction: the candidate met the expectations in {cand} of {total} answers, the current '
                         f'playbook in {cur} (needs at least {math.ceil(PASS_SHARE * total)} and more than the current)'
                         + ('; the failure does not reproduce now - the current playbook already decides this way, so '
                            'the game lost it elsewhere (a harness fault, timing, a rule since removed)'
                            if total and cur == total else '')]
    return {'cases': cases, 'current': cur, 'candidate': cand, 'answers': total, 'pass': ok, 'reasons': why,
            'cost_usd': round(sum(a.get('cost') or 0 for c in cases for s in ('current', 'candidate')
                                  for a in c.get(s, [])), 2)}


# --- held out ---------------------------------------------------------------------------------------------------

def playbook_only(model_text):
    """The memory for strong players' held-out decisions: the playbook alone. Our harness notes and comp notes say
    nothing about their games, and with them in the prompt every edit to them made every cached answer of the current
    model stale (29 Oct: 0 of 40 cached after a harness-notes edit) - about 560 model calls before a test could run."""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix='playbook-'))
    (d / 'playbook.md').write_text(modelfile.playbook(model_text), encoding='utf-8')
    return decide.judge.memory_text(f'dir:{d}')


MISSING_MAX = 0.05             # held out: more answers missing than this and the test has not run


def neutral_folder():
    """The held-out questions' working directory: the reference files, no memory/ - the memory under test is in the
    prompt, and the player's folder holds the current playbook (the structure review, 29 Oct)."""
    import shutil
    import tempfile
    from tft_player.coach import REFERENCE
    d = Path(tempfile.gettempdir()) / 'tft-judge-heldout'
    if not (d / 'README.md').exists():
        shutil.copytree(REFERENCE, d, dirs_exist_ok=True)
    return d


def answers(mem, rounds, samples, workers=4):
    """[[answer per sample] per round] for one memory (cached; a missing answer is retried on the next call)."""
    cwd = neutral_folder()
    jobs = [(r, s) for s in samples for r in rounds]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda j: decide.ask(mem, j[0][0], j[0][1], j[1], cwd=cwd), jobs))
    n = len(rounds)
    return [[res[k * n + i] for k in range(len(samples))] for i in range(n)]


def warm(workers=4):
    """The current playbook's answers to every held-out decision (BASE_SAMPLES each): run before a test needs them."""
    rounds = replay.held_out()
    got = answers(playbook_only(modelfile.read()), rounds, range(1, replay.BASE_SAMPLES + 1), workers)
    missing = sum(not a['answer'] for row in got for a in row)
    return {'decisions': len(rounds), 'answers': len(rounds) * replay.BASE_SAMPLES, 'missing': missing,
            'cost_usd': round(sum(a.get('cost') or 0 for row in got for a in row), 2)}


def held_out(prop, text, new_text, rid, workers=4):
    base_mem, cand_mem = playbook_only(text), playbook_only(new_text)
    rounds = replay.held_out()
    base_recs = answers(base_mem, rounds, range(1, replay.BASE_SAMPLES + 1), workers)
    cand_recs = [row[0] for row in answers(cand_mem, rounds, [1], workers)]
    res = [a for row in base_recs for a in row] + cand_recs
    n = len(rounds)
    base = [[a['answer'] or {} for a in row] for row in base_recs]
    cand = [a['answer'] or {} for a in cand_recs]
    missing = sum(not a['answer'] for a in res)
    if missing > MISSING_MAX * len(res):
        # not a pass: with empty answers nothing is better or worse, and harm would read as none (the review, 29 Oct)
        return {'decisions': n, 'better': 0, 'worse': 0, 'p_harm': 1.0, 'named': {'better': 0, 'worse': 0, 'n': 0},
                'side_effects': {'better': 0, 'worse': 0}, 'harm': True, 'missing': missing,
                'reasons': [f'held out: {missing} of {len(res)} answers did not come (a usage limit or timeouts) - '
                            f'the test did not run; the proposal waits for the next run'],
                'cost_usd': round(sum(r.get('cost') or 0 for r in res), 2)}
    rep = replay.score(prop, rid, rounds, base, cand)
    better, worse = rep['better'], rep['worse']
    p_harm = replay.sign_p(worse, better)
    why = []
    if worse > better and p_harm <= HARM_P:
        why.append(f'held out: worse than another answer of the current model in {worse}, better in {better} '
                   f'(p={p_harm:.2f})')
    se = rep['side_effects']
    if se['worse'] > se['better'] + 1:
        why.append(f"held out: side effects where it does not name itself {se['worse']} worse, {se['better']} better")
    rep.update(p_harm=round(p_harm, 4), harm=bool(why), reasons=why,
               cost_usd=round(sum(r.get('cost') or 0 for r in res), 2))
    # the held-out games stay unseen: no decision, no player's choice in the verdict the Strategist reads - counts only
    rep.pop('changed', None)
    return rep


# --- the verdict ------------------------------------------------------------------------------------------------

def invalid(prop):
    """Why a proposal cannot be tested as written ([] if it can): its fields, and the structure's evidence rule (3+
    games, or a calculator proof)."""
    if prop.get('type') == 'comp_rule':                # a hard rule for one comp: kept on its evidence (climb.judge_all)
        return [f'no "{k}"' for k in ('id', 'comp', 'rule', 'why', 'sources') if not prop.get(k)]
    if prop.get('type') == 'state':
        bad = [f'no "{k}"' for k in ('id', 'description', 'transform') if not prop.get(k)]
        for t in prop.get('transform') or []:
            if not (isinstance(t, dict) and isinstance(t.get('old'), str) and isinstance(t.get('new'), str)):
                bad.append(f'a transform entry needs "old" (a regex) and "new": {str(t)[:80]}')
            else:
                try:
                    re.compile(t['old'])
                except re.error as e:
                    bad.append(f'"old" is not a regex: {e}')
        kind = None
    else:
        bad = [f'no "{k}"' for k in ('id', 'kind') if not prop.get(k)]
        kind = prop.get('kind')
    if kind and kind not in ('add', 'change', 'remove'):
        bad.append(f'kind "{kind}" is not add, change or remove')
    if kind == 'add' and not (prop.get('section') and prop.get('principle')):
        bad.append('an add needs "section" and "principle"')
    if kind == 'change' and not prop.get('edits'):
        bad.append('a change needs "edits"')
    if kind in ('change', 'remove') and not prop.get('rid'):
        bad.append(f'a {kind} needs "rid"')
    games = {str(g) for g in prop.get('games') or []}
    if len(games) < 3 and not prop.get('calculator'):
        bad.append(f'evidence from {len(games)} game(s): a strategy change needs 3+ games or a calculator proof')
    for c in prop.get('reproduce') or []:
        if not (isinstance(c, dict) and c.get('run') and (c.get('round') or c.get('call')) and c.get('expect')):
            bad.append(f'a reproduction case needs "run", "round" and "expect": {str(c)[:80]}')
    return bad


def judge(path, apply=True, skip_held_out=False):
    """The verdict on one proposal file; it is written as agent/model/replays/<the file's name>.json and .md, whatever
    the proposal's own id (the check and the loop find a verdict by the file's name - the structure review, 29 Oct)."""
    path = Path(path)
    name = path.stem
    verdict = {'proposal': name, 'kind': None, 'rid': None, 'when': datetime.now().isoformat(timespec='seconds'),
               'bounds': {'samples': SAMPLES, 'pass_share': PASS_SHARE, 'harm_p': HARM_P}}
    try:
        prop = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(prop, dict):
            raise ValueError('not a JSON object')
    except (OSError, ValueError) as e:
        why = [f'the proposal file is not a JSON object: {e}'[:300]]
        verdict.update(reproduction={'pass': False, 'reasons': why}, accept=False, reasons=why)
        write(name, {}, verdict)
        return verdict
    verdict.update(kind=prop.get('kind'), rid=prop.get('rid'), id=prop.get('id'))
    bad = invalid(prop)
    if bad:
        verdict.update(reproduction={'pass': False, 'reasons': bad}, accept=False, reasons=bad)
        write(name, prop, verdict)
        return verdict
    if prop.get('type') == 'state':
        verdict['kind'] = 'state'
        verdict['reproduction'] = reproduce(prop, None) if prop.get('reproduce') else \
            {'pass': False, 'reasons': ['no reproduction cases: a change must fix the game it came from']}
        reasons = list(verdict['reproduction']['reasons'])
        verdict['accept'] = not reasons
        verdict['reasons'] = reasons or ['accepted: the Editor builds it into the state']
        verdict['build'] = prop.get('description') if verdict['accept'] else None
        write(name, prop, verdict)
        sys.path.insert(0, str(decide.AGENT))
        from channel import post
        r = verdict['reproduction']
        post('Judge', f"{name} (a state change): the candidate met its checks in {r.get('candidate', 0)} of "
                      f"{r.get('answers', 0)} answers, as recorded {r.get('current', 0)}. "
                      + ('Accepted: a request for the Editor.' if verdict['accept'] else 'Rejected: ' + '; '.join(reasons)
                         + '.'), refs=[f"agent/model/replays/{name}.md"])
        return verdict
    text = modelfile.read()
    try:
        new_text, rid = replay.candidate_model(prop, text)
    except Exception as e:                               # an edit whose old text is not in the model, a bad section
        why = [f'cannot be applied to the game model: {e.__class__.__name__}: {e}'[:300]]
        verdict.update(reproduction={'pass': False, 'reasons': why}, accept=False, reasons=why)
        write(name, prop, verdict)
        return verdict
    verdict['rid'] = rid
    cand_dir = decide.memory_dir(new_text)
    if not prop.get('reproduce'):
        verdict['reproduction'] = {'pass': False, 'reasons': ['no reproduction cases: a change must fix the game it '
                                                              'came from']}
    else:
        verdict['reproduction'] = reproduce(prop, cand_dir)
    reasons = list(verdict['reproduction']['reasons'])
    if verdict['reproduction']['pass'] and not skip_held_out:
        verdict['held_out'] = held_out(prop, text, new_text, rid)
        reasons += verdict['held_out']['reasons']
    elif skip_held_out:
        reasons.append('held-out test skipped')
    verdict['accept'] = not reasons
    verdict['reasons'] = reasons or ['accepted']
    write(name, prop, verdict)
    sys.path.insert(0, str(decide.AGENT))
    from channel import post
    r = verdict['reproduction']
    post('Judge', f"{name}: reproduction on {len(prop.get('reproduce') or [])} source decision(s) - the candidate "
                  f"met them in {r.get('candidate', 0)} of {r.get('answers', 0)} answers, the current playbook in "
                  f"{r.get('current', 0)}. " + ('Accepted: it is in the playbook for the next game.' if verdict['accept']
                                                else 'Rejected: ' + '; '.join(reasons) + '.'),
         refs=[f"agent/model/replays/{name}.md"])
    if apply and verdict['accept']:
        replay.accept([{'accept': True, 'path': str(path)}])
    return verdict


def write(name, prop, v):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(v, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    r = v['reproduction']
    md = [f"# Judge: {name} ({prop.get('kind') or '?'} {v.get('rid') or ''})\n",
          f"{prop.get('principle') or prop.get('edits') or ''}\n",
          f"**{'ACCEPTED' if v['accept'] else 'REJECTED'}**: {'; '.join(v['reasons'])}.\n",
          f"Reproduction: the candidate met the expectations in {r.get('candidate', 0)} of {r.get('answers', 0)} "
          f"answers, the current playbook in {r.get('current', 0)}.\n",
          '| run | round | call | expect | current | candidate |', '|---|---|---|---|---|---|']
    for c in r.get('cases', []):
        ex = '; '.join(f"{x.get('path')} {x.get('op', '==')} {x.get('value', '')}" for x in c.get('expect') or []
                       if isinstance(x, dict))
        cell = lambda s: ' / '.join(('met' if a['met'] else 'not') + f" {a['found']}" for a in c.get(s, [])) or c.get('error', '')
        md.append(f"| {c.get('run')} | {c.get('round', '')} | {c.get('call')} | {ex} | {cell('current')} | "
                  f"{cell('candidate')} |")
    h = v.get('held_out')
    if h:
        md.append(f"\nHeld out: {h['decisions']} decisions; against another answer of the current model better in "
                  f"{h['better']}, worse in {h['worse']} (harm p={h['p_harm']:.2f}); where it names itself "
                  f"{h['named']['better']} better / {h['named']['worse']} worse; side effects {h['side_effects']}."
                  + (f" {h['missing']} answers missing." if h.get('missing') else ''))
    (OUT / f"{name}.md").write_text('\n'.join(md) + '\n', encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('proposals', nargs='*')
    ap.add_argument('--no-apply', action='store_true')
    ap.add_argument('--skip-held-out', action='store_true', help='reproduction only (a test of the Judge itself)')
    ap.add_argument('--warm', action='store_true', help="the current playbook's held-out answers, before a test needs "
                                                        "them")
    a = ap.parse_args()
    if a.warm:
        print(json.dumps(warm()))
    for p in a.proposals:
        v = judge(p, apply=not a.no_apply, skip_held_out=a.skip_held_out)
        print(json.dumps({k: v.get(k) for k in ('proposal', 'accept', 'reasons')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
