"""One recorded game's numbers, from its run folder (runs/<id>/log.jsonl and calls/): what the benchmark compares.

  python tools/game_metrics.py runs/20261016-052413 [runs/... ...]        a summary per game
  python tools/game_metrics.py runs/20261016-052413 --json                everything, as JSON
  --reread-hp: HP read again from the run's screenshots (a game whose logged HP was misread)

- placement (runs/games.jsonl, else the log's game_result), last stage, minutes;
- HP, level and gold at each round's combat start (the `fight` events), HP 0 at the end for a lost game;
- each timed question (planning phase, planning update, augment, Wisp): the seconds left on the game's timer when
  it was asked (the prompt's "~Ns left") against the time to the decision (the first JSON object when the
  backend streams; the whole reply otherwise) - late when the decision came after the time left;
- planning phases run from a plan the model wrote ahead (next_round) instead of a question;
- rejected plan entries (the model asked for something the game state does not allow), failed steps (the
  controller tried and the game did not do it), sells that failed out of all sells, actions done;
- model cost: the Claude CLI's own list cost per conversation (cumulative per session id: the last call of each);
  Codex and OpenRouter report tokens only, so their cost is an estimate from list prices (PRICES, below);
- augments picked;
- target comp reached: the share of the comp's units (the plan's `units`) on the board at each combat start, and
  the carry's stars.
"""
import ast
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMED = {'plan', 'augment', 'wisp'}            # and 'update' asked during a planning phase
LEFT = re.compile(r'~(\d+) ?s\b')
UNIT = re.compile(r"(?:(?<=\[)|(?<=, ))([A-Z?][\w'. ]*?)(\*{1,3})@h(\d)(\d)")
# $ per million tokens (input, cached input, output) for backends that report no cost. GPT-6.1-Sol: a third-party
# price list (anotherwrapper.com, 2026-10-11: $2 in / $10 out); cached input assumed at a tenth of input
PRICES = {'gpt-6.1-sol': (2.0, 0.2, 10.0)}


def round_index(stage):
    """'1-1' -> 1 ... '1-4' -> 4, '2-1' -> 5 ... (7 rounds per stage from stage 2): an x axis for curves."""
    try:
        s, r = (int(x) for x in str(stage).split('-'))
    except ValueError:
        return None
    return r if s == 1 else 4 + (s - 2) * 7 + r


def board_units(roster):
    """[(name, stars)] on the board from a roster line 'board=[Pebbles***@h14, ...] bench=[...]'."""
    board = (roster or '').split(' bench=[')[0]
    return [(m.group(1).strip(), len(m.group(2))) for m in UNIT.finditer(board)]


def plan_target(plan):
    """(units, carry) of the target comp from the logged Plan(...) text."""
    units = re.search(r"units=(\[.*?\])", plan or '')
    carry = re.search(r"carry='([^']*)'", plan or '')
    try:
        u = ast.literal_eval(units.group(1)) if units else []
    except (ValueError, SyntaxError):
        u = []
    return u, carry.group(1) if carry else None


def placement_record(run):
    """The game's record in runs/games.jsonl (matched by folder name: the release game came from another laptop)."""
    f = ROOT / 'runs' / 'games.jsonl'
    if not f.exists():
        return None
    for line in f.read_text(encoding='utf-8').splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if Path(str(r.get('run_dir', '')).replace('\\', '/')).name == Path(run).name:
            return r
    return None


def costs(run, backend, model):
    """Model cost and tokens of a game, from calls/*-result.json (one per question). Both CLIs report totals per
    conversation (session id), so each conversation's last call holds its sum."""
    sessions = {}
    for f in sorted(glob.glob(os.path.join(run, 'calls', '*-result.json'))):
        try:
            r = json.load(open(f, encoding='utf-8'))
        except (OSError, ValueError):
            continue
        sessions.setdefault(r.get('session_id'), []).append(r)
    out = {'conversations': len(sessions), 'usd': 0.0, 'usd_source': None, 'input': 0, 'cached': 0, 'output': 0}
    for rs in sessions.values():
        if any('total_cost_usd' in r for r in rs):             # Claude CLI: its own list cost
            last = max(rs, key=lambda r: r.get('total_cost_usd') or 0)
            out['usd'] += last.get('total_cost_usd') or 0
            out['usd_source'] = 'Claude CLI list cost'
            for m in (last.get('modelUsage') or {}).values():
                out['input'] += m.get('inputTokens', 0) + m.get('cacheReadInputTokens', 0) + m.get('cacheCreationInputTokens', 0)
                out['cached'] += m.get('cacheReadInputTokens', 0)
                out['output'] += m.get('outputTokens', 0)
        else:                                                   # Codex CLI: cumulative usage per thread
            u = max((r.get('usage') or {} for r in rs), key=lambda u: u.get('input_tokens', 0))
            out['input'] += u.get('input_tokens', 0)
            out['cached'] += u.get('cached_input_tokens', 0)
            out['output'] += u.get('output_tokens', 0)
    if out['usd_source'] is None:
        p = PRICES.get((model or '').split(':')[-1])
        if p and out['input']:
            out['usd'] = ((out['input'] - out['cached']) * p[0] + out['cached'] * p[1] + out['output'] * p[2]) / 1e6
            out['usd_source'] = 'estimate from list prices'
        else:
            out['usd'] = None
    if out['usd'] is not None:
        out['usd'] = round(out['usd'], 2)
    return out


def hp_from_screens(run):
    """{stage: HP} read again from the run's planning-phase screenshots (<stage>-start.full.jpg) with today's
    reader - for a game whose logged HP was misread (release game 20261016-141957 at 1440p: HP 1 from 3-7 on).
    Only frames with the shop up (the level is read) count: carousel and augment screens cover the HUD."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ROOT))
    from tft_player.screen import ScreenReader
    reader, out = ScreenReader(), {}
    for f in glob.glob(os.path.join(run, '*-start.full.jpg')):
        s = reader.read(np.array(Image.open(f).convert('RGB')))
        if s.hp is not None and s.level is not None:
            out[os.path.basename(f).split('-start')[0]] = s.hp
    return out


def metrics(run, reread_hp=False):
    run = str(run)
    ev = []
    for line in open(os.path.join(run, 'log.jsonl'), encoding='utf-8'):
        try:
            ev.append(json.loads(line))
        except ValueError:
            pass
    first = lambda name: next((e for e in ev if e.get('ev') == name), {})
    session, config, result = first('model_session'), first('run_config'), first('game_result')
    rec = placement_record(run) or result
    backend = session.get('backend')
    model = rec.get('model') or result.get('model')
    effort = session.get('effort')
    if backend == 'codex' and str(session.get('model', '')).startswith('claude'):
        effort = None        # logged Claude's effort for a Codex game (fixed in coach.model_settings, 2026-10-17)
    out = {'run': Path(run).name, 'model': model, 'backend': backend, 'effort': effort,
           'comp': rec.get('comp_name') or config.get('comp'), 'placement': rec.get('placement'),
           'placement_source': rec.get('placement_source') or ('match history' if rec.get('placement') else None),
           'last_stage': rec.get('last_stage'), 'minutes': round((rec.get('seconds') or 0) / 60, 1),
           'queue': {1090: 'Normal', 1100: 'Ranked'}.get(rec.get('queue_id')), 'git': config.get('git'),
           # changed files that play the game (the log cuts the first letter of the first one: 'ft_player/...')
           'dirty': [f for f in config.get('dirty_files') or [] if f.endswith('.py') and ('_player/' in f or '/play' in f)],
           'patch': rec.get('patch')}

    rounds = []
    screens = hp_from_screens(run) if reread_hp else None
    out['hp_source'] = 'screenshots, read again' if reread_hp else 'game log'
    for e in ev:
        if e.get('ev') != 'fight' or not e.get('stage'):
            continue
        units, carry = plan_target(e.get('plan'))
        board = board_units(e.get('roster'))
        names = {n for n, _ in board}
        hp = screens.get(e['stage']) if screens is not None else e.get('hp')
        rounds.append({'stage': e['stage'], 'x': round_index(e['stage']), 'hp': hp, 'level': e.get('level'),
                       'gold': e.get('gold'), 'board': len(board),
                       'comp_units': sum(1 for u in units if u in names), 'comp_size': len(units),
                       'carry_stars': max((s for n, s in board if n == carry), default=0) if carry else None})
    if out['placement'] and out['placement'] > 1 and out['last_stage']:
        rounds.append({'stage': out['last_stage'], 'x': round_index(out['last_stage']), 'hp': 0, 'level': None,
                       'gold': None, 'board': None, 'comp_units': None, 'comp_size': None, 'carry_stars': None,
                       'end': True})
    # HP only rises by small heals: a reading that a later one exceeds by more than 10 was misread (the harness took
    # 'HP 4' through stage 1 of 20261018-150046 and 'HP 2' at 2-2 of 20261018-183028). The curve drops it; the logged
    # value stays as hp_logged
    out['hp_misreads'] = []
    for i, r in enumerate(rounds):
        later = [x['hp'] for x in rounds[i + 1:] if x.get('hp') is not None and not x.get('end')]
        if r.get('hp') is not None and later and max(later[:3]) > r['hp'] + 10:
            out['hp_misreads'].append(r['stage'])
            r['hp_logged'], r['hp'] = r['hp'], None
    out['rounds'] = rounds
    target = next((plan_target(e.get('plan')) for e in reversed(ev) if e.get('ev') == 'fight'), ([], None))
    out['target_units'], out['carry'] = target
    played = [r for r in rounds if r.get('comp_size')]
    last = played[-1] if played else {}
    out['comp_final'] = round(last['comp_units'] / last['comp_size'], 2) if last else None
    out['comp_best'] = max((r['comp_units'] / r['comp_size'] for r in played), default=None)
    out['carry_stars_final'] = last.get('carry_stars')
    out['level_final'] = next((r['level'] for r in reversed(rounds) if r.get('level')), None)

    answers = []
    for e in ev:
        if e.get('ev') != 'model_call':
            continue
        q = (e.get('prompt') or '').split('=== THE QUESTION ===')[-1].strip()
        head = q.splitlines()[0] if q else ''
        kind = e.get('kind')
        planning_update = kind == 'update' and head.startswith('DECISION: same planning phase')
        if kind not in TIMED and not planning_update:
            continue
        m = LEFT.search(q[:2000])
        took = e.get('decision_s') or e.get('latency')
        if m is None or took is None:
            continue
        answers.append({'kind': 'update' if planning_update else kind, 'stage': e.get('stage'),
                        'left': int(m.group(1)), 'decision': took, 'late': took > int(m.group(1))})
    out['answers'] = answers
    for k in ('plan', 'augment', 'wisp', 'update'):
        a = [x['decision'] for x in answers if x['kind'] == k]
        out[f'{k}_p50'] = round(statistics.median(a), 1) if a else None
    out['timed_questions'] = len(answers)
    out['late'] = sum(1 for a in answers if a['late'])
    out['plans_asked'] = sum(1 for e in ev if e.get('ev') == 'model_call' and e.get('kind') == 'plan')
    out['plans_ahead'] = sum(1 for e in ev if e.get('ev') == 'next_round_started')
    out['model_calls'] = sum(1 for e in ev if e.get('ev') == 'model_call')
    # the answer time of the round questions (plan and update), timed or not: the speed axis of the capability chart
    # (the same quantiles as tools/speed_check.py games())
    lat = sorted(e['latency'] for e in ev if e.get('ev') == 'model_call' and e.get('kind') in ('plan', 'update')
                 and e.get('latency'))
    out['lat_p50'], out['lat_p90'] = ((round(lat[min(len(lat) - 1, int(len(lat) * q))], 1) for q in (.5, .9))
                                      if lat else (None, None))

    out['rejected'] = sum(len(e['errors']) for e in ev if e.get('ev') == 'plan_rejected' and isinstance(e.get('errors'), list))
    out['failed_steps'] = sum(1 for e in ev if e.get('ev') == 'step_failed')
    out['actions'] = sum(1 for e in ev if e.get('ev') == 'action')
    out['actions_failed'] = sum(1 for e in ev if e.get('ev') == 'action' and e.get('ok') is False)
    sells = [e for e in ev if e.get('ev') == 'action' and str(e.get('act', '')).startswith('sell(')]
    out['sells'], out['sells_failed'] = len(sells), sum(1 for e in sells if e.get('ok') is False)
    # by phase: a `round` event opens a planning phase, a `fight` event its combat (actions carry no phase)
    phase, by = 'planning', {'planning': [0, 0], 'combat': [0, 0]}
    for e in ev:
        if e.get('ev') == 'round':
            phase = 'planning'
        elif e.get('ev') == 'fight':
            phase = 'combat'
        elif e in sells:
            by[phase][0] += 1
            by[phase][1] += e.get('ok') is False
    out['sells_by_phase'] = by
    moves = [e for e in ev if e.get('ev') == 'action' and str(e.get('act', '')).startswith('move(')]
    out['moves'], out['moves_failed'] = len(moves), sum(1 for e in moves if e.get('ok') is False)
    out['augments'] = [{'stage': e.get('stage'), 'name': e.get('name')} for e in ev if e.get('ev') == 'augment']
    # a pick that missed the timer: the game took one of the cards on its own (logged since d214937)
    out['augments_missed'] = list(dict.fromkeys(e.get('stage') for e in ev if e.get('ev') == 'augment_missed'))
    out['cost'] = costs(run, backend, model)
    return out


def summary(m):
    c = m['cost']
    usd = f"${c['usd']:.2f}" + (' (est.)' if c['usd_source'] == 'estimate from list prices' else '') if c['usd'] is not None else 'n/a'
    return (f"{m['run']}  {m['model']} ({m['effort'] or '?'})  {m['comp']}: place {m['placement']} at {m['last_stage']}, "
            f"{m['minutes']} min\n"
            f"  comp reached {m['comp_final']} (best {m['comp_best'] and round(m['comp_best'], 2)}), carry {m['carry']} "
            f"{m['carry_stars_final']}*, level {m['level_final']}\n"
            f"  decisions p50: plan {m['plan_p50']} s, update {m['update_p50']} s, augment {m['augment_p50']} s, "
            f"wisp {m['wisp_p50']} s; late {m['late']}/{m['timed_questions']}; plans asked {m['plans_asked']}, "
            f"run ahead {m['plans_ahead']}\n"
            f"  rejected plan entries {m['rejected']}, failed steps {m['failed_steps']}, sells failed {m['sells_failed']} of "
            f"{m['sells']} (planning {m['sells_by_phase']['planning'][1]}/{m['sells_by_phase']['planning'][0]}, combat "
            f"{m['sells_by_phase']['combat'][1]}/{m['sells_by_phase']['combat'][0]}), moves failed {m['moves_failed']} of "
            f"{m['moves']}, actions {m['actions']} "
            f"({m['actions_failed']} failed); cost {usd}, {c['input'] / 1e6:.1f}M in ({c['cached'] / 1e6:.1f}M cached), "
            f"{c['output'] / 1e3:.0f}k out, {c['conversations']} conversations\n"
            f"  augments: {', '.join(a['name'] or '?' for a in m['augments'])}")


def main():
    runs = [a for a in sys.argv[1:] if not a.startswith('--')]
    ms = [metrics(r, '--reread-hp' in sys.argv) for r in runs]
    if '--json' in sys.argv:
        print(json.dumps(ms, indent=1, ensure_ascii=False))
    else:
        for m in ms:
            print(summary(m))


if __name__ == '__main__':
    main()
