"""Decision cases: a state and a strong player's choice in it (agent/FRAMEWORK.md). Code, not a model, builds them and
decides each game's split, so no proposer can pick what it is tested on.

    python agent/cases.py vod runs/vod-<id> --player <name> [--client "CN zh"]   tempo cases from a read video
    python agent/cases.py demo runs/<run>                                       tempo cases from the owner's game
    python agent/cases.py augment <game> --t <s> --stage 3-2 --cards "A|B|C" --pick B [--reason ...] [--video ...]
    python agent/cases.py labels <states.json> <labels.json>                    the owner's labels of agent states
    python agent/cases.py list [--split test]

A tempo case: the first frame of a planning phase with the HUD (level, XP, gold), and what the player had at the end
of that phase (level, gold after their buys, rolls and XP). An augment case: the cards on screen and the one taken.
Keyframes stay local (runs/keyframes/<game>/, third-party pictures); a case keeps the file name and its sha1.
"""
import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEC = ROOT / 'agent' / 'decisions'
CASES = DEC / 'cases.jsonl'
SPLITS = DEC / 'splits.json'
KEYS = ROOT / 'runs' / 'keyframes'
TEST_SHARE = 0.2          # a new game is test while test cases are under this share of all cases


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def load():
    return [json.loads(l) for l in CASES.read_text(encoding='utf-8').splitlines() if l.strip()] if CASES.exists() else []


def splits():
    return json.loads(SPLITS.read_text(encoding='utf-8')) if SPLITS.exists() else {'test': [], 'train': [], 'rule': ''}


def split_of(game):
    s = splits()
    return 'test' if game in s['test'] else 'train' if game in s['train'] else None


def assign(game, taught=False):
    """The game's split, decided once when its first cases are added: train when a review has already learned from it;
    otherwise test while the test cases are under TEST_SHARE of all cases (in the order games arrive, not by content)."""
    s = splits()
    if game in s['test'] or game in s['train']:
        return split_of(game)
    cases = load()
    n_test = sum(c['game'] in s['test'] for c in cases)
    side = 'train' if taught or (cases and n_test / len(cases) >= TEST_SHARE) else 'test'
    s[side].append(game)
    s['rule'] = (f'decided by agent/cases.py when a game first adds cases: train if a review already learned from it, '
                 f'else test while test cases are under {TEST_SHARE:.0%} of all cases; never changed afterwards')
    SPLITS.write_text(json.dumps(s, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return side


def current_set():
    """'<set> <patch>' from data/reference/README.md's first line ('# Reference files - TFT Set 18, patch 18.4b')."""
    head = (ROOT / 'data' / 'reference' / 'README.md').read_text(encoding='utf-8').splitlines()[0]
    m = re.search(r'Set (\d+), patch ([\w.]+)', head)
    return f'{m.group(1)} {m.group(2)}' if m else 'unknown'


def save(new, set_tag=None):
    """Add cases (replacing ones with the same id), each tagged with the set it was played on (FRAMEWORK rule 8: the
    current set unless given), and check them against the schema."""
    import jsonschema
    schema = json.loads((ROOT / 'agent' / 'schemas' / 'case.schema.json').read_text(encoding='utf-8'))
    for c in new:
        c.setdefault('set', set_tag or current_set())
        jsonschema.validate(c, schema)
    ids = {c['id'] for c in new}
    keep = [c for c in load() if c['id'] not in ids]
    DEC.mkdir(parents=True, exist_ok=True)
    CASES.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in keep + new), encoding='utf-8')


def keyframe(video, t, game, cid):
    out = KEYS / game
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{cid}.jpg'
    subprocess.run([ffmpeg(), '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{t:.2f}', '-i', str(video),
                    '-frames:v', '1', '-vf', 'scale=-2:1080', '-q:v', '2', str(path)], check=True)
    return path


def stage_key(stage):
    a, b = map(int, stage.split('-'))
    return a, b


def split_games(rows, min_rounds=10, confirm=3):
    """A stream replay's reads as games: a new game starts where the stage falls back to stage 1-2 from stage 3 or later
    and stays there for `confirm` reads (one earlier stage between later ones is an overlay or a replay clip); runs with
    fewer than `min_rounds` stages (the end of a game the stream started in) are dropped. -> [[rows]]"""
    out, cur, last, pending = [], [], None, []
    for r in rows:
        st = r.get('stage')
        if not st:
            if cur:
                cur.append(r)
            continue
        if last and stage_key(st)[0] <= 2 and stage_key(last)[0] >= 3:
            pending.append(r)
            if len(pending) >= confirm:
                out.append(cur)
                cur, last, pending = list(pending), st, []
            continue
        cur.extend(pending)
        pending = []
        cur.append(r)
        last = st
    if cur:
        out.append(cur)
    return [g for g in out if len({r['stage'] for r in g if r.get('stage')}) >= min_rounds]


def vod_cases(folder, player, client, line=None, multi=False):
    """Tempo cases from agent/vod_read.py's reads: per planning phase, the first frame with the HUD and the level and gold
    at the phase's end (the median of the last three gold reads: one misread frame cannot set it). multi: a stream
    replay with several games, each its own game (<folder>-g<n>) for the split. -> [(game, cases)]"""
    folder = Path(folder)
    rows = [json.loads(l) for l in (folder / 'read' / 'reads.jsonl').read_text(encoding='utf-8').splitlines()]
    if not multi:
        return [(folder.name, phase_cases(folder, folder.name, rows, player, client, line))]
    return [(f'{folder.name}-g{k}', phase_cases(folder, f'{folder.name}-g{k}', g, player, client, line))
            for k, g in enumerate(split_games(rows), 1)]


def phase_cases(folder, game, rows, player, client, line):
    cur, runs = None, []
    for r in rows:
        cur = r['stage'] or cur            # a stage hidden by an overlay keeps the last one read
        if cur:
            if runs and runs[-1][0] == cur:
                runs[-1][1].append(r)
            else:
                runs.append((cur, [r]))
    by = {}
    for stage, rs in runs:                 # the longest stretch of each stage (an intro clip repeats a later one)
        if len(rs) > len(by.get(stage, [])):
            by[stage] = rs
    by = dict(sorted(by.items(), key=lambda kv: kv[1][0]['t']))
    video = folder / 'game.mp4'
    out = []
    for stage, rs in by.items():
        hud = [r for r in rs if r['level'] is not None and r['gold'] is not None]
        if len(hud) < 3:
            continue                        # a carousel or a phase the video skipped
        first = hud[0]
        cid = f'{game}-{stage}-tempo'
        kf = keyframe(video, first['t'], game, cid)
        out.append({'id': cid, 'source': 'vod', 'game': game, 'player': player, 't': first['t'], 'stage': stage,
                    'kind': 'tempo', 'client': client,
                    'state': {'keyframe': kf.name, 'keyframe_sha1': sha1(kf), 'level': first['level'],
                              'xp': first['xp'], 'gold': first['gold'], **({'line': line} if line else {})},
                    'expert': {'level_end': hud[-1]['level'],
                               'gold_end': int(statistics.median([r['gold'] for r in hud[-3:]]))}})
    return out


_READER = []


def video_frame_matching(run, t, level, gold, out):
    """Write to `out` the latest frame of the run's screen recording (game-av.mkv) within 2 s before the round event at
    `t` whose HUD shows this level and gold. -> True when one did."""
    meta = run / 'game-av.json'
    if not (t and meta.exists() and (run / 'game-av.mkv').exists()):
        return False
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ROOT))
    from tft_player import screen
    if not _READER:
        _READER.append(screen.ScreenReader())
    start = json.loads(meta.read_text(encoding='utf-8'))['start_unix']
    for back in (0.0, 0.5, 1.0, 1.5, 2.0):
        offset = t - start - back
        if offset < 0:
            break
        subprocess.run([ffmpeg(), '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{offset:.2f}', '-i',
                        str(run / 'game-av.mkv'), '-frames:v', '1', '-vf', 'scale=-2:1080', '-q:v', '2', str(out)], check=True)
        a = np.asarray(Image.open(out).convert('RGB'))
        screen.load_hud_dx(a.shape[1], a.shape[0])
        s = _READER[0].read(a)
        if s.level == level and s.gold == gold:
            return True
    return False


def demo_cases(run, line=None):
    """Tempo cases from the owner's recorded game: the planning phase's start screenshot and HUD, and the level and gold
    at its combat start (the 'fight' event)."""
    run = Path(run)
    game = f'demo-{run.name}'
    evs = [json.loads(l) for l in (run / 'log.jsonl').read_text(encoding='utf-8', errors='replace').splitlines() if l.strip()]
    start = {e['stage']: e for e in evs if e.get('ev') == 'round' and e.get('stage')}
    fight = {e['stage']: e for e in evs if e.get('ev') == 'fight' and e.get('stage')}
    rx = re.compile(r'L(\d+) xp=\((\d+), (\d+)\) g=(\d+) hp=(\w+)')
    out = []
    for stage, e in start.items():
        f, pic = fight.get(stage), run / f'{stage}-start.jpg'
        m = rx.search(e.get('screen') or '')
        if not (f and m and pic.exists() and f.get('level') and f.get('gold') is not None):
            continue
        dst = KEYS / game
        dst.mkdir(parents=True, exist_ok=True)
        cid = f'{game}-{stage}-tempo'
        kf = dst / f'{cid}.jpg'
        # the screen recording's frame at the phase start, checked against the HUD numbers of the round: the bot's
        # <stage>-start.jpg can come after the first actions (4-1: level 8 with 62 gold, after XP buys - the analyst,
        # 2026-10-24); the start screenshot only when there is no recording or no frame matches
        level, gold = int(m.group(1)), int(m.group(4))
        if not (video_frame_matching(run, e.get('t'), level, gold, kf)):
            kf.write_bytes(pic.read_bytes())
        hp = int(m.group(5)) if m.group(5).isdigit() else None
        out.append({'id': cid, 'source': 'demo', 'game': game, 'player': 'owner', 't': e.get('t') or 0, 'stage': stage,
                    'kind': 'tempo', 'client': 'NA en',
                    'state': {'keyframe': kf.name, 'keyframe_sha1': sha1(kf), 'level': int(m.group(1)),
                              'xp': [int(m.group(2)), int(m.group(3))], 'gold': int(m.group(4)), 'hp': hp, **({'line': line} if line else {})},
                    'expert': {'level_end': int(f['level']), 'gold_end': int(f['gold'])}})
    return game, out


def label_cases(states_file, labels_file):
    """The owner's labels of the agent's own states (the label page, agent/FRAMEWORK.md - DAgger): one tempo case per
    answered state, grouped by the agent game it comes from (agent-g<n>, each game split on its own). Both files are
    exports of the page's database: states.json [{id, run, stage, level, xp, gold, hp, ...}], labels.json
    {state id: {level_end, gold_end, note} | {skip: true}}."""
    states = {s['id']: s for s in json.loads(Path(states_file).read_text(encoding='utf-8'))}
    labels = json.loads(Path(labels_file).read_text(encoding='utf-8'))
    by_game = {}
    for sid, lab in labels.items():
        s = states.get(sid)
        if not s or lab.get('skip') or lab.get('level_end') is None:
            continue
        game = f"agent-g{s['game']}"
        pic = ROOT / 'runs' / s['run'] / f"{s['stage']}-start.jpg"
        dst = KEYS / game
        dst.mkdir(parents=True, exist_ok=True)
        cid = f"{game}-{s['stage']}-tempo"
        kf = dst / f'{cid}.jpg'
        kf.write_bytes(pic.read_bytes())
        by_game.setdefault(game, []).append(
            {'id': cid, 'source': 'label', 'game': game, 'player': 'owner', 't': 0, 'stage': s['stage'], 'kind': 'tempo',
             'client': 'NA en',
             'state': {'keyframe': kf.name, 'keyframe_sha1': sha1(kf), 'level': s['level'], 'xp': s.get('xp'),
                       'gold': s['gold'], 'hp': s.get('hp'), 'note': f"The agent's own game {s['game']} (it placed {s['place']})."},
             'expert': {'level_end': int(lab['level_end']), 'gold_end': int(lab['gold_end']),
                        **({'reason': lab['note']} if lab.get('note') else {})}})
    return by_game


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('vod')
    v.add_argument('folder')
    v.add_argument('--player', required=True)
    v.add_argument('--client', default='CN zh')
    v.add_argument('--line', help='what the board was built toward, from the title or narration')
    v.add_argument('--multi', action='store_true', help='a stream replay with several games: split them (split_games)')
    d = sub.add_parser('demo')
    d.add_argument('run')
    d.add_argument('--line')
    a = sub.add_parser('augment')
    a.add_argument('game')
    a.add_argument('--t', type=float, required=True)
    a.add_argument('--stage', required=True)
    a.add_argument('--cards', required=True)
    a.add_argument('--pick', required=True)
    a.add_argument('--reason')
    a.add_argument('--player', default='')
    a.add_argument('--client', default='CN zh')
    a.add_argument('--video')
    a.add_argument('--line')
    lb = sub.add_parser('labels')
    lb.add_argument('states')
    lb.add_argument('labels')
    ls = sub.add_parser('list')
    ls.add_argument('--split')
    x = ap.parse_args()
    if x.cmd == 'labels':
        for game, new in label_cases(x.states, x.labels).items():
            side = assign(game, taught=False)
            save(new)
            print(f'{len(new)} cases from {game} ({side}) -> {CASES.relative_to(ROOT)}')
        return
    if x.cmd == 'list':
        for c in load():
            sp = split_of(c['game'])
            if not x.split or sp == x.split:
                print(sp, c['id'], c['stage'], c['kind'], c['state'].get('level'), c['state'].get('gold'), '->', c['expert'])
        return
    if x.cmd == 'vod':
        sides, n = {}, 0
        for game, new in vod_cases(x.folder, x.player, x.client, x.line, x.multi):
            side = assign(game, taught=False)
            save(new)
            sides[side] = sides.get(side, 0) + 1
            n += len(new)
            print(f'{len(new)} cases from {game} ({side}) -> {CASES.relative_to(ROOT)}')
        sys.path.insert(0, str(ROOT / 'agent'))
        from channel import post
        games = sum(sides.values())
        post('Data', f"I turned {Path(x.folder).name} ({x.player}) into {games} game{'s' if games != 1 else ''} and {n} "
                     f"decision cases - each planning phase's state and what the player did with it "
                     f"({', '.join(f'{k} {v} game(s)' for k, v in sides.items())}). Analyst: the train games are ready for you; "
                     f"the test ones are kept for the Judge.", to='Strategist', refs=['agent/decisions/cases.jsonl'])
        return
    elif x.cmd == 'demo':
        game, new = demo_cases(x.run, x.line)
        taught = True                       # the demonstration review (b42dc45) learned from it
    else:
        cards = [c.strip() for c in x.cards.split('|')]
        if x.pick not in cards:
            sys.exit(f'--pick must be one of the cards: {cards}')
        cid = f'{x.game}-{x.stage}-augment'
        video = Path(x.video) if x.video else ROOT / 'runs' / x.game / 'game.mp4'
        kf = keyframe(video, x.t, x.game, cid)
        game, taught = x.game, False
        new = [{'id': cid, 'source': 'vod', 'game': x.game, 'player': x.player, 't': x.t, 'stage': x.stage,
                'kind': 'augment', 'client': x.client,
                'state': {'keyframe': kf.name, 'keyframe_sha1': sha1(kf), 'cards': cards, **({'line': x.line} if x.line else {})},
                'expert': {'pick': x.pick, **({'reason': x.reason} if x.reason else {})}}]
    side = assign(game, taught)
    save(new)
    print(f'{len(new)} cases from {game} ({side}) -> {CASES.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
