"""The gate for a harness change (agent/COACH.md step 5): everything a game needs must still load and render.

    python agent/check.py            compile, import every module, render the player's prompts, run the tests
    python agent/check.py --quick    without the tests

Runs at below-normal priority, so it can run beside a live game. Exit code 0 = pass.
"""
import json
import os
import py_compile
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def low_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


RENDER = r'''
import json, os, sys
sys.path.insert(0, r"%s")
os.environ["TFT_CLIMB"] = "1"
cfg = json.load(open(r"%s", encoding="utf-8"))
if cfg["comp"].lower() != "flex":
    os.environ["TFT_FORCE_COMP"] = cfg["comp"]
import importlib, pkgutil, tft_player
for m in pkgutil.iter_modules(tft_player.__path__):
    importlib.import_module("tft_player." + m.name)
from tft_player import coach
assert cfg["comp"].lower() == "flex" or coach.forced_comp() is not None, "no MetaTFT comp named %%r" %% cfg["comp"]
system = coach.opening_prompt()
prep = coach.briefing_text(before_queue=True)
coach._remember(coach.plain(prep))
digest = coach.stage_digest("2")
mem = coach.memory_block()
assert len(system) > 2000 and "REPLY FORMAT" in system
assert prep.startswith(coach.PREP_HEAD) and "PREPARATION TASK" in prep
assert (mem in prep and mem in digest) if mem else True
print("prompts: system %%d chars, preparation %%d, stage digest %%d, memory %%d" %% (len(system), len(prep), len(digest), len(mem)))
''' % (ROOT, ROOT / 'agent' / 'config.json')


def framework():
    """The framework's contracts (agent/FRAMEWORK.md): cases, proposals and verdicts match their schemas; every case's
    game has a split; no game is both train and test; a game a review learned from (the owner's demonstrations) is
    never test; the role files exist. -> [problems]"""
    import jsonschema
    agent = ROOT / 'agent'
    bad = []
    schema = lambda n: json.loads((agent / 'schemas' / f'{n}.schema.json').read_text(encoding='utf-8'))
    for name in ('case', 'proposal', 'verdict'):
        jsonschema.Draft202012Validator.check_schema(schema(name))
    cases_file = agent / 'decisions' / 'cases.jsonl'
    splits = json.loads((agent / 'decisions' / 'splits.json').read_text(encoding='utf-8')) \
        if (agent / 'decisions' / 'splits.json').exists() else {'test': [], 'train': []}
    test, train = set(splits.get('test', [])), set(splits.get('train', []))
    if test & train:
        bad.append(f'games in both splits: {sorted(test & train)}')
    taught = {f'demo-{json.loads(l)["run"]}' for l in (agent / 'history' / 'demos.jsonl').read_text(encoding='utf-8').splitlines()
              if l.strip()} if (agent / 'history' / 'demos.jsonl').exists() else set()
    if taught & test:
        bad.append(f'demonstrations a review learned from are test: {sorted(taught & test)}')
    ids = set()
    if cases_file.exists():
        cs = schema('case')
        for i, line in enumerate(cases_file.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            c = json.loads(line)
            try:
                jsonschema.validate(c, cs)
            except jsonschema.ValidationError as e:
                bad.append(f'cases.jsonl line {i}: {e.message[:200]}')
            if c.get('id') in ids:
                bad.append(f'duplicate case {c.get("id")}')
            ids.add(c.get('id'))
            if c.get('game') not in test | train:
                bad.append(f'case {c.get("id")}: its game has no split')
    for folder, name in (('proposals', 'proposal'), ('verdicts', 'verdict')):
        for f in sorted((agent / folder).glob('*.json')) if (agent / folder).exists() else []:
            try:
                jsonschema.validate(json.loads(f.read_text(encoding='utf-8')), schema(name))
            except (ValueError, jsonschema.ValidationError) as e:
                bad.append(f'{folder}/{f.name}: {str(e)[:200]}')
    # proposals waiting for the Judge (no verdict yet) must be testable as written (agent/model/judge_change.invalid)
    model = agent / 'model'
    if (model / 'proposals').exists():
        sys.path.insert(0, str(model))
        try:
            from judge_change import invalid
        except Exception as e:                       # the Judge itself does not import: that is the problem
            bad.append(f'agent/model/judge_change.py does not import: {e}')
            invalid = None
        for f in sorted((model / 'proposals').glob('*.json')) if invalid else []:
            if (model / 'replays' / f.name).exists():
                continue
            try:
                why = invalid(json.loads(f.read_text(encoding='utf-8')))
            except ValueError as e:
                why = [f'not JSON: {e}']
            bad += [f'agent/model/proposals/{f.name}: {w}' for w in why]
    for role in ('reviewer', 'harness_fixer', 'strategist', 'comp_coach'):
        if not (agent / 'roles' / f'{role}.md').exists():
            bad.append(f'agent/roles/{role}.md is missing')
    return bad


def main():
    low_priority()
    quick = '--quick' in sys.argv
    t0, failed = time.time(), []
    files = [p for d in ('tft_player', 'tools', 'agent') for p in (ROOT / d).glob('*.py')]
    for f in files:
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as e:
            failed.append(f'compile {f.name}: {e.msg.strip()[:300]}')
    print(f'compiled {len(files)} files' + (f', {len(failed)} failed' if failed else ''))
    contract = framework()
    print('framework contracts: ' + ('ok' if not contract else f'{len(contract)} problems'))
    failed += [f'framework: {p}' for p in contract]
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    r = subprocess.run([PY, '-c', RENDER], cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', env=env, timeout=300)
    print((r.stdout or '').strip()[-500:])
    if r.returncode:
        failed.append('imports/prompts: ' + (r.stderr or '').strip()[-1500:])
    # no tests (the owner, 2026-11-07: "do not do any tests. Do not do any tests from now on"): the check is the
    # compile, the imports and the player's prompts; the test files stay, run only by hand
    if not quick and '--tests' in sys.argv:
        # the tests set the climb's switches themselves: a check started from a shell that has TFT_CLIMB=1 failed a
        # test that is right without it, and took back the Editor's working fix (game 35's loop, 29 Oct)
        test_env = {k: v for k, v in env.items() if k not in ('TFT_CLIMB', 'TFT_FORCE_COMP', 'TFT_MEMORY_DIR')}
        r = subprocess.run([PY, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', 'tests'], cwd=ROOT,
                           capture_output=True, text=True, encoding='utf-8', errors='replace', env=test_env,
                           timeout=1800)
        tail = (r.stdout or '').strip().splitlines()[-15:]
        print('\n'.join(tail))
        if r.returncode:
            failed.append('tests: ' + ' | '.join(tail[-8:]))
    print(f'{time.time() - t0:.0f} s')
    if failed:
        print('CHECK FAILED')
        for f in failed:
            print(' -', f)
        sys.exit(1)
    print('CHECK PASSED')


if __name__ == '__main__':
    main()
