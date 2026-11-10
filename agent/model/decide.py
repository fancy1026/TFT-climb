"""One decision of a strong player's game, asked of our player and compared with his - shared by step 1 (play-along,
agent/model/playalong.py) and step 4 (the replay test, agent/model/replay.py).

The question is the one the player meets in a game: the screen at the start of the planning phase, the HUD numbers,
the CALCULATOR lines, and its memory (the game model's active principles). It answers what it does this phase. No
history, no hint of what he did: each round is asked on its own, so a decision can be judged from its recorded state.
"""
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AGENT = ROOT / 'agent'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(AGENT))
sys.path.insert(0, str(ROOT))
import calc  # noqa: E402
import judge  # noqa: E402
import modelfile  # noqa: E402
import study  # noqa: E402

CACHE = HERE / 'cache'
QV = 2


def memory_dir(model_text=None):
    """A memory folder as the player would read it, with the playbook built from this model text."""
    d = Path(tempfile.mkdtemp(prefix='model-'))
    (d / 'playbook.md').write_text(modelfile.playbook(model_text), encoding='utf-8')
    shutil.copytree(AGENT / 'memory' / 'comps', d / 'comps')
    shutil.copy2(AGENT / 'memory' / 'harness.md', d / 'harness.md')
    return d


def cases_of(game):
    rows = [json.loads(l) for l in (AGENT / 'decisions' / 'cases.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    return sorted((c for c in rows if c['game'] == game and c['kind'] == 'tempo'), key=lambda c: c['t'])


def reads_of(game):
    folder = ROOT / 'runs' / game.split('-g')[0]
    return [json.loads(l) for l in (folder / 'read' / 'reads.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]


def planning_end(reads, stage):
    """(level, xp, gold) at the end of this stage's planning phase: the reads before its fight starts (the timer jumps
    back up from near 0), gold the median of the last three - so PvE loot and fight gold are not counted as gold he
    kept (the Reasoner, cycle1: vod-BV1uYn86RErk's x-7 "gold kept" was loot). None without HUD reads."""
    rows = [r for r in reads if r.get('stage') == stage]
    last_t = None
    for i, r in enumerate(rows):
        t = r.get('timer')
        if t is not None and last_t is not None and last_t <= 5 and t > last_t + 5:
            rows = rows[:i]
            break
        if t is not None:
            last_t = t
    hud = [r for r in rows if r.get('level') and r.get('gold') is not None and (r.get('xp') or r['level'] == 10)]
    if not hud:
        return None
    gold = sorted(r['gold'] for r in hud[-3:])[len(hud[-3:]) // 2]
    return hud[-1]['level'], hud[-1]['xp'][0] if hud[-1].get('xp') else 0, gold


def rounds(game):
    """[(case, start, his)] - the start of each phase (study.phase_start) and what he did in its planning phase
    (level, gold kept, XP buys, other gold), from the reads."""
    cs, reads, out = cases_of(game), reads_of(game), []
    for i, c in enumerate(cs):
        start = study.phase_start(c, study.end_of_phase(reads, cs[i - 1]['stage']) if i else None)
        end = planning_end(reads, c['stage'])
        his = {'level_end': end[0] if end else c['expert']['level_end'],
               'gold_end': end[2] if end else c['expert']['gold_end'], 'xp_buys': None, 'other_gold': None}
        if start and end:
            raw = (study.total_xp(end[0], end[1]) - study.total_xp(start[0], start[1])) / calc.M['xp']['buy']['xp']
            if raw > -0.5:
                his['xp_buys'] = max(0, round(raw))
                his['other_gold'] = start[2] - his['gold_end'] - 4 * his['xp_buys']
        out.append((c, start, his))
    return out


def question(c, start):
    if start:
        lv, x, g, exact = start
        hud = (f"level {lv}, XP {x}/{calc.to_next(lv) or '-'}, gold {g}" + ('' if exact else
               " (computed from the previous phase's end - the HUD here was read only after he had bought - so the "
               "gold is approximate)"))
    else:
        hud = 'not read this round - read level, XP and gold from the picture'
    facts = calc.facts(c['stage'], start[0], start[1], start[2], streak=None) if start else [
        '(the HUD was not read this round)']
    return (f"Stage {c['stage']}, the start of the planning phase: {hud}. The picture was taken around the phase "
            f"change and can still show the previous phase's last seconds (its stage number, its gold, its shop, the HP "
            f"before the last fight); the numbers in this text are this phase's start - decide from them. HP, the "
            f"streak, the board, the bench and the lobby: read them from the picture.\nCALCULATOR:\n- "
            + '\n- '.join(facts) +
            "\n\nYou are this player now. What do you do this planning phase? Reply with one line of JSON: "
            '{"level_end": <level at the end>, "gold_end": <gold kept at the end>, "xp_buys": <XP buys>, '
            '"roll_gold": <gold spent on rerolls>, "buy": ["<units bought>"], "sell": ["<units sold>"], '
            '"why": "<one or two sentences>", "principles": ["<ids of the game model principles that decided it, e.g. R9>"]}')


def ask(memory, c, start, sample=1, cwd=None):
    """Our player's answer for this round (cached per memory, question, sample and folder). cwd: the CLI's working
    directory - the Judge's held-out test uses one without memory/, so a candidate cannot read the current playbook
    there. An answer that did not come (a timeout, a usage limit) is not cached: cached as {} it read as "no change"
    for that playbook from then on and hid harm (the structure review, 29 Oct)."""
    prompt = (memory + '\n\n' if memory else '') + question(c, start)
    key = hashlib.sha1((judge.MODEL + judge.EFFORT + judge.SYSTEM + str(QV) + prompt
                        + ('|no-memory-folder' if cwd else '')).encode()).hexdigest()[:16]
    f = CACHE / f"{c['id']}-{key}-s{sample}.json"
    if f.exists():
        rec = json.loads(f.read_text(encoding='utf-8'))
        if rec.get('answer'):
            return rec
    t0 = time.time()
    res = judge.ask(prompt, ROOT / 'runs' / 'keyframes' / c['game'] / c['state']['keyframe'], cwd=cwd) or {}
    ans = judge.parse(res.get('result') or '')
    rec = {'case': c['id'], 'answer': ans, 'text': res.get('result') or '', 'cost': res.get('total_cost_usd'),
           'seconds': round(time.time() - t0, 1)}
    if ans:
        CACHE.mkdir(exist_ok=True)
        f.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    return rec


def num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def differences(c, start, bot, his):
    """What differs between our answer and his choice (nothing is explained here)."""
    out = []
    bl, bg, bx = num(bot.get('level_end')), num(bot.get('gold_end')), num(bot.get('xp_buys'))
    if bl is not None and bl != his['level_end']:
        out.append(f"level: we end at {bl}, he at {his['level_end']}")
    if bg is not None and abs(bg - his['gold_end']) >= 10 and not (bg >= 50 and his['gold_end'] >= 50):
        out.append(f"gold kept: we {bg}, he {his['gold_end']}")
    elif bg is not None and calc.interest(bg) != calc.interest(his['gold_end']):
        out.append(f"interest step: we keep {bg} (+{calc.interest(bg)}), he {his['gold_end']} (+{calc.interest(his['gold_end'])})")
    if bx is not None and his['xp_buys'] is not None and abs(bx - his['xp_buys']) >= 2:
        out.append(f"XP buys: we {bx}, he {his['xp_buys']}")
    spent = (num(bot.get('roll_gold')) or 0) + 0
    if his['other_gold'] is not None and his['other_gold'] >= 10 and spent < 4 and not bot.get('buy'):
        out.append(f"he spent about {his['other_gold']} on units and rolls; we spend nothing on them")
    if his['other_gold'] is not None and his['other_gold'] <= 2 and spent >= 10:
        out.append(f"we roll {spent}; he spends about nothing on units and rolls")
    return out


def points(c, ans, his):
    """Agreement with his choice: level (1), gold kept (1, not graded at 2-1, whose start is read before the augment's
    gold), XP buys within 1 (1). PvE rounds are graded: his gold kept is read before the fight's loot (planning_end)."""
    p, of = 0, 0
    lv, g, x = num(ans.get('level_end')), num(ans.get('gold_end')), num(ans.get('xp_buys'))
    of += 1
    p += lv == his['level_end']
    if c['stage'] != '2-1':
        of += 1
        p += g is not None and (abs(g - his['gold_end']) <= judge.GOLD_TOL or (g >= 50 and his['gold_end'] >= 50))
    if his['xp_buys'] is not None:
        of += 1
        p += x is not None and abs(x - his['xp_buys']) <= 1
    return p, of
