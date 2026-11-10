"""Step 4 - the replay test: a proposed change to the game model (agent/model/proposals/<id>.json, by the Curator) is
applied to a copy of the model, and every held-out decision of strong players' games (the test split: no rule is
learned from them) is asked again with and without it. Code scores it; no model grades its own proposal.

- Noise first: the current model answers each decision three times; the third answer is what the current model gives
  by chance, and how often it differs from both of the first two is its flip rate.
- The change: the candidate model answers once. Each decision is scored by agreement with his choice (level, gold
  kept, XP buys): the candidate's answer against the current model's third answer - better, worse or the same.
- Explanation: a decision whose answer names the new principle is the principle working; one that does not is a side
  effect.
- One decision at a time, from its recorded state: what a change does to later rounds needs live games.

Accepted when: better more often than worse beyond chance (sign test, p <= 0.05), better than worse where it names
itself (at least two decisions), and the side effects not worse than even. Accepted changes go into the model as
active and the playbook is rebuilt; every result is written to agent/model/replays/<id>.md.

    python agent/model/replay.py agent/model/proposals/<id>.json [<id2>.json ...] [--no-apply]
"""
import argparse
import json
import re
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decide  # noqa: E402
import modelfile  # noqa: E402

OUT = HERE / 'replays'
BASE_SAMPLES = 3


def held_out():
    from cases import splits
    out = []
    for g in splits().get('test', []):
        try:
            out += [r for r in decide.rounds(g) if r[1]]
        except FileNotFoundError:
            continue
    return out


def candidate_model(prop, text):
    """(model text with the proposal applied and active, the principle's id)."""
    kind = prop['kind']
    if kind == 'change':
        return modelfile.apply_edits(text, prop['edits']), prop.get('rid')
    if kind == 'remove':
        return modelfile.remove(text, prop['rid']), prop['rid']
    rid = f"R{modelfile.next_id(text)}"
    n = re.match(r'\s*(\d+)', prop['section']).group(1)
    m = re.search(rf'^## {n}\. .*?(?=^## |\Z)', text, re.S | re.M)
    if not m:
        raise ValueError(f"no section {prop['section']!r}")
    block = m.group(0)
    bullet = textwrap.fill(f"- [{rid} active {prop.get('scope', 'general')}] {prop['principle'].strip()}", 120,
                           subsequent_indent='  ', break_on_hyphens=False) + '\n'
    if 'What follows: (' in block:
        new_block = re.sub(r'What follows: \([^\n]*\)\n', 'What follows:\n' + bullet, block)
    else:
        lines = block.rstrip('\n').split('\n')
        last = max(i for i, l in enumerate(lines) if l.startswith('- [R') or (l.startswith('  ') and i and any(
            x.startswith('- [R') for x in lines[:i])))
        new_block = '\n'.join(lines[:last + 1]) + '\n' + bullet + '\n'.join(lines[last + 1:]) + '\n\n'
    new_block = re.sub(r'\n{3,}', '\n\n', new_block)
    return text[:m.start()] + new_block + text[m.end():], rid


def key(a):
    """The decision as compared: level, interest step kept, XP buys (in twos), a roll or not."""
    n = decide.num
    g = n(a.get('gold_end'))
    return (n(a.get('level_end')), min(5, g // 10) if g is not None else None,
            (n(a.get('xp_buys')) or 0) // 2, (n(a.get('roll_gold')) or 0) >= 6)


def cites(a, rid):
    return rid in (a.get('principles') or []) or rid in str(a.get('why') or '')


def test(path, workers=4):
    """The replay test of one proposal against the model as it stands (nothing applied). -> the report."""
    prop = json.loads(Path(path).read_text(encoding='utf-8'))
    text = modelfile.read()
    new_text, rid = candidate_model(prop, text)
    base_mem = decide.judge.memory_text(f'dir:{decide.memory_dir(text)}')
    cand_mem = decide.judge.memory_text(f'dir:{decide.memory_dir(new_text)}')
    rounds = held_out()
    jobs = [(base_mem, r, s) for s in range(1, BASE_SAMPLES + 1) for r in rounds] + [(cand_mem, r, 1) for r in rounds]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda j: decide.ask(j[0], j[1][0], j[1][1], j[2]), jobs))
    n = len(rounds)
    base = [[res[s * n + i]['answer'] or {} for s in range(BASE_SAMPLES)] for i in range(n)]
    cand = [res[BASE_SAMPLES * n + i]['answer'] or {} for i in range(n)]
    rep = score(prop, rid, rounds, base, cand)
    rep['cost_usd'] = round(sum(r.get('cost') or 0 for r in res), 2)
    write(prop, rep)
    sys.path.insert(0, str(decide.AGENT))
    from channel import post
    post('Judge', f"Replay test of {prop['id']} on {rep['decisions']} held-out decisions, against another answer of the "
                  f"current model: better in {rep['better']}, worse in {rep['worse']} (p={rep['p']:.2f}); where it "
                  f"names itself {rep['named']['better']} better / {rep['named']['worse']} worse. "
                  + ('Accepted: it is in the model now.' if rep['accept'] else 'Rejected: ' + '; '.join(rep['reasons']) + '.'),
         refs=[f"agent/model/replays/{prop['id']}.md"])
    rep['path'] = str(path)
    return rep


def sign_p(better, worse):
    """One-sided sign test: the chance of `better` or more of better+worse decisions going better by luck."""
    from math import comb
    n = better + worse
    return sum(comb(n, k) for k in range(better, n + 1)) / 2 ** n if n else 1.0


def score(prop, rid, rounds, base, cand):
    """Each decision: the candidate's answer against the current model's third answer - one draw each, so what the
    current model gives by chance is the baseline - scored by agreement with his choice (decide.points). The
    whole-set flip rates are kept for the record; a principle that applies in a few situations changes only those, so
    it is judged by direction, not by how many decisions it changes (cycle1: all three proposals fell under a 19%
    whole-set flip rate while changing their own situations)."""
    n = len(rounds)
    noise = sum(key(b[2]) not in (key(b[0]), key(b[1])) for b in base) / n
    changed = sum(key(a) not in (key(b[0]), key(b[1])) for b, a in zip(base, cand)) / n
    rows, named, other = [], {'better': 0, 'worse': 0, 'n': 0}, {'better': 0, 'worse': 0}
    for (c, start, his), b, a in zip(rounds, base, cand):
        d = decide.points(c, a, his)[0] - decide.points(c, b[2], his)[0]
        it = rid is not None and cites(a, rid)
        side = named if it else other
        named['n'] += it
        side['better'] += d > 0
        side['worse'] += d < 0
        if d:
            rows.append({'case': c['id'], 'round': c['stage'], 'game': c['game'], 'current': [
                {k: x.get(k) for k in ('level_end', 'gold_end', 'xp_buys', 'roll_gold')} for x in b],
                'candidate': {k: a.get(k) for k in ('level_end', 'gold_end', 'xp_buys', 'roll_gold', 'why', 'principles')},
                'his': {k: his[k] for k in ('level_end', 'gold_end', 'xp_buys')},
                'direction': 'better' if d > 0 else 'worse', 'names_it': it})
    better, worse = named['better'] + other['better'], named['worse'] + other['worse']
    p = sign_p(better, worse)
    why = []
    if not (better > worse and p <= 0.05):
        why.append(f'not above chance: better than another answer of the current model in {better}, worse in {worse} '
                   f'(p={p:.2f})')
    if prop['kind'] != 'remove' and (named['n'] < 2 or named['better'] <= named['worse']):
        why.append(f"where it names itself ({named['n']} decisions) {named['better']} better, {named['worse']} worse: "
                   'it does not work where it fires')
    if other['worse'] > other['better'] + 1:
        why.append(f"side effects: where it does not name itself {other['worse']} worse, {other['better']} better")
    return {'proposal': prop['id'], 'kind': prop['kind'], 'rid': rid, 'decisions': n, 'noise_rate': round(noise, 3),
            'change_rate': round(changed, 3), 'better': better, 'worse': worse, 'p': round(p, 4), 'named': named,
            'side_effects': other, 'accept': not why, 'reasons': why or ['accepted'], 'changed': rows,
            'when': datetime.now().isoformat(timespec='seconds')}


def write(prop, rep):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{prop['id']}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    nm, ot = rep['named'], rep['side_effects']
    md = [f"# Replay test: {prop['id']} ({prop['kind']} {rep['rid'] or ''})\n",
          f"{prop.get('principle') or prop.get('edits') or ''}\n",
          f"{rep['decisions']} held-out decisions, each answered three times by the current model and once with the "
          f"change. Against the current model's third answer, by agreement with his choice: better in {rep['better']}, "
          f"worse in {rep['worse']} (sign test p={rep['p']:.2f}). Where it names itself ({nm['n']}): {nm['better']} "
          f"better, {nm['worse']} worse; elsewhere: {ot['better']} better, {ot['worse']} worse. Whole-set flip rates: "
          f"the current model {rep['noise_rate']:.0%}, the change {rep['change_rate']:.0%}. "
          f"**{'ACCEPTED' if rep['accept'] else 'REJECTED'}**: {'; '.join(rep['reasons'])}.\n",
          '| game | round | current (x3) | candidate | his | vs the 3rd | names it |', '|---|---|---|---|---|---|---|']
    for r in rep['changed']:
        cur = ' / '.join(f"L{x['level_end']} {x['gold_end']}g xp{x['xp_buys']}" for x in r['current'])
        ca = r['candidate']
        md.append(f"| {r['game']} | {r['round']} | {cur} | L{ca['level_end']} {ca['gold_end']}g xp{ca['xp_buys']} | "
                  f"L{r['his']['level_end']} {r['his']['gold_end']}g xp{r['his']['xp_buys']} | {r['direction']} | "
                  f"{'yes' if r['names_it'] else 'no'} |")
    (OUT / f"{prop['id']}.md").write_text('\n'.join(md) + '\n', encoding='utf-8')


def accept(reps):
    """Put every accepted proposal into the model (each was tested against the same model; they change different
    principles), rebuild the playbook and commit. -> the ids now in the model."""
    text, done = modelfile.read(), []
    for rep in reps:
        if not rep['accept']:
            continue
        prop = json.loads(Path(rep['path']).read_text(encoding='utf-8'))
        text, rid = candidate_model(prop, text)
        done.append(f"{prop['id']} ({rid})")
    if not done:
        return done
    modelfile.MODEL.write_text(text, encoding='utf-8')
    subprocess.run([sys.executable, str(HERE / 'build_memory.py')], check=True)
    subprocess.run(['git', 'add', 'agent/model/game-model.md', 'agent/memory/playbook.md', 'agent/model/replays',
                    'agent/model/proposals'], cwd=decide.ROOT)
    subprocess.run(['git', 'commit', '-q', '-m', f"Replay test accepted {', '.join(done)}: in the model\n\n"
                    'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'], cwd=decide.ROOT)
    return done


def run(path, apply=True):
    rep = test(path)
    if apply:
        accept([rep])
    return rep


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('proposals', nargs='+', help='tested one by one against the same model; the accepted go in together')
    ap.add_argument('--no-apply', action='store_true')
    a = ap.parse_args()
    reps = [test(p) for p in a.proposals]
    for r in reps:
        print(json.dumps({k: r[k] for k in ('proposal', 'decisions', 'better', 'worse', 'p', 'named', 'side_effects',
                                            'noise_rate', 'change_rate', 'accept', 'reasons', 'cost_usd')}, ensure_ascii=False))
    if not a.no_apply:
        print('in the model:', ', '.join(accept(reps)) or 'nothing')


if __name__ == '__main__':
    main()
