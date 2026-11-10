"""The comp coach: what strong players do with one comp, and why, learned from their games (the owner, 2026-10-29:
"each comp has a different tempo ... if it is hard to reason to learn, you could hard code it from reviewing and
learning pro games"; "you should build the system that builds it automatically"; "it is not like tempo and tips - I
just gave two examples, it has much more").

    python agent/comp_coach.py "<comp>" --find        queue strong players' games of the comp (YouTube search)
    python agent/comp_coach.py "<comp>" --material    agent/comps/<comp>/material.md from the games read
    python agent/comp_coach.py "<comp>" --apply       check the coach's proposal, write the player's comp notes

The pipeline: games of the comp are found and queued for agent/study_queue.py (download, HUD read, one keyframe a
round, captions where the site gives them); code builds the material - per game and round the pro's level, XP, gold,
shops and rerolls, the round's keyframe and the pro's own words; the coach session (agent/roles/comp_coach.md) reads
it with the game's mechanics, the comp's MetaTFT page and our own games' reviews, and proposes the comp's knowledge -
any areas it finds (tempo, boards by stage, item holders, rolling, augments, positioning, streaks, ...), each point
with its reason and its sources; code drops a point whose sources do not check, and writes the rest as the player's
comp notes (agent/memory/comps/<comp>.md, shown to the player in every game of the comp).
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
COMPS = AGENT / 'comps'
MEMORY_COMPS = AGENT / 'memory' / 'comps'
sys.path.insert(0, str(AGENT))
MIN_GAMES = 3                        # strong players' games of a comp before its notes are written
NEW_PER_FIND = 3                     # games queued per search
GENERIC = {'comp', 'reroll', 'fast', 'with', 'and', 'the'}
FIND_AGAIN_H = 6                     # a search that found nothing is not repeated sooner


def slug(comp):
    return re.sub(r'[^A-Za-z0-9]+', '_', comp).strip('_')


def state(comp):
    try:
        return json.loads((COMPS / slug(comp) / 'state.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def save_state(comp, **kw):
    d = COMPS / slug(comp)
    d.mkdir(parents=True, exist_ok=True)
    st = dict(state(comp), **kw)
    (d / 'state.json').write_text(json.dumps(st, indent=1), encoding='utf-8')
    return st


def keywords(comp):
    return [w.lower() for w in re.findall(r"[A-Za-z']+", comp) if len(w) >= 4 and w.lower() not in GENERIC]


def jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()] if path.exists() else []


def games_of(comp):
    """The strong players' games of the comp that are read (runs/vod-*/read/reads.jsonl): queued for it by the comp
    coach, or a read video whose title names it. -> [{'game', 'url', 'title', 'channel'}]"""
    import study_queue
    keys = keywords(comp)
    videos = {v.get('url'): v for v in jsonl(study_queue.VIDEOS)}
    out = {}
    for r in study_queue.rows():
        if r.get('status') != 'done' or not r.get('url'):
            continue
        v = videos.get(r['url'], {})
        title = str(v.get('title') or r.get('why') or '')
        folder = study_queue.url_folder(r['url'])
        reads = folder / 'read' / 'reads.jsonl' if folder else None
        staged = reads.read_text(encoding='utf-8').count('"stage": "') if reads and reads.exists() else 0
        if staged >= study_queue.MIN_STAGED_READS and (r.get('comp') == comp or any(k in title.lower() for k in keys)):
            out[folder.name] = {'game': folder.name, 'url': r['url'], 'title': v.get('title') or '',
                                'channel': v.get('channel') or r.get('player') or ''}
    return list(out.values())


def find(comp, n=NEW_PER_FIND):
    """Strong players' full games of the comp on YouTube, not read or queued yet, queued for the study queue
    (agent/sources/requests.jsonl, tagged with the comp). -> the urls queued."""
    import study_queue
    keys = keywords(comp)
    seen = {r.get('url') for r in study_queue.rows()}
    found = []
    for q in (f'TFT {comp}', f'TFT {comp} guide gameplay'):
        found += study_queue.search_youtube(q, 15)
    picks = []
    for v in found:
        if v['url'] in seen or v['url'] in [p['url'] for p in picks]:
            continue
        if any(k in v['title'].lower() for k in keys):
            picks.append(v)
    rows = study_queue.rows()
    for v in picks[:n]:
        rows.append({'url': v['url'], 'player': v.get('channel'), 'comp': comp, 'status': 'open',
                     'why': f"comp coach: a strong player's game of {comp} ({v['title'][:80]})",
                     'asked': datetime.now().isoformat(timespec='seconds')})
    study_queue.save(rows)
    save_state(comp, find_at=datetime.now().isoformat(timespec='seconds'), find_found=len(picks[:n]))
    return [v['url'] for v in picks[:n]]


def stage_key(s):
    m = re.match(r'(\d+)-(\d+)', str(s or ''))
    return (int(m.group(1)), int(m.group(2))) if m else None


def split_games(reads):
    """A stream holds several games: a new one starts where the stage goes back to stage 1 or 2 from a later one."""
    games, cur, last = [], [], None
    for r in reads:
        k = stage_key(r.get('stage'))
        if k and last and k[0] < last[0] and k[0] <= 2 and last[0] >= 3:
            games.append(cur)
            cur = []
        if k:
            last = k
        cur.append(r)
    if cur:
        games.append(cur)
    return [g for g in games if sum(1 for r in g if r.get('stage')) >= 50]


def captions(folder):
    """[(seconds, text)] of the video's captions (game*.vtt), rolling auto-caption repeats dropped."""
    out, last = [], ''
    for f in sorted(folder.glob('game*.vtt')):
        for m in re.finditer(r'(\d+):(\d+):(\d+)\.\d+ --> [^\n]*\n((?:.+\n?)+)', f.read_text(encoding='utf-8',
                                                                                              errors='replace')):
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            text = re.sub(r'<[^>]+>', '', m.group(4)).strip().replace('\n', ' ')
            if text and text != last and not last.endswith(text):
                out.append((t, text))
                last = text
    return out


def rounds_of(game_rows):
    """Per stage of one game: when it ran in the video, the level, XP and gold it opened and closed with, the most
    gold held, and the shops seen (a new full shop in the same planning phase is a reroll)."""
    out, cur = [], None
    for r in game_rows:
        st = r.get('stage')
        if not st:
            continue
        if not cur or cur['stage'] != st:
            cur = {'stage': st, 't0': r['t'], 't1': r['t'], 'rows': []}
            out.append(cur)
        cur['t1'] = r['t']
        cur['rows'].append(r)
    for c in out:
        rs = c.pop('rows')
        val = lambda k: [x[k] for x in rs if x.get(k) is not None]
        c['level'] = (val('level') or [None])[0], (val('level') or [None])[-1]
        c['xp'] = (val('xp') or [None])[-1]
        g = val('gold')
        c['gold'] = (g[0], max(g), g[-1]) if g else None
        shops, prev = [], None
        for x in rs:
            s = tuple(x.get('shop') or ())
            if len(s) == 5 and all(s) and s != prev:
                if prev is None or sum(a != b for a, b in zip(s, prev)) >= 3:
                    shops.append(s)
                prev = s
        c['shops'] = len(shops)
    return out


def material(comp):
    """agent/comps/<comp>/material.md: every read strong player's game of the comp, round by round, with the round's
    keyframe and the player's own words. -> (path, the games in it)."""
    import stages
    games = games_of(comp)
    stages.heldout(comp, [g['game'] for g in games])      # kept for the Judge: no role reads them
    games = [g for g in games if not stages.is_heldout(comp, g['game'])]
    d = COMPS / slug(comp)
    d.mkdir(parents=True, exist_ok=True)
    md = [f"# {comp}: strong players' games, round by round\n",
          "Per round: when it ran in the video (mm:ss), level at its start -> end, XP, gold at its start / most held / "
          "at its end, the full shops seen (more than one in a planning phase: rerolls), the keyframe (the board, bench, "
          "items and trait panel as the round is fought - open it to see them), and the player's words in that round "
          "(captions; none when the site gave none).\n"]
    used = []
    try:                                                 # MetaTFT's numbers for the comp (most of what it needs)
        from tft_player.meta import META
        c = META.by_name(comp)
    except Exception:
        c = None
    if c is not None:
        md.append(f"\n## MetaTFT: {c.name} ({c.id}), patch {META.patch}, data of {META.fetched_at}\n"
                  f"avg place {c.avg}, win {c.win}, top 4 {c.top4}, {c.games} games, tier {c.tier}\n"
                  f"The comp's page as text (levels by stage, boards by level, items, augments, positions, tips): "
                  f"data/metatft_shots/{slug(comp)}/brief.txt; its screenshots: data/metatft_shots/{slug(comp)}/*.jpg\n")
        if c.builds:
            md.append('Items by unit (MetaTFT builds): ' + '; '.join(
                f"{u}: {', '.join(b.get('items') or [])} (avg place {b.get('avg')}, {b.get('games')} games)"
                if isinstance(b, dict) else f"{u}: {b}" for u, b in list(c.builds.items())[:12]))
        top = sorted(((a, t) for a, t in (c.augments or {}).items() if t in ('S', 'A')), key=lambda x: x[1])
        if top:
            md.append('Augments rated S or A for the comp: ' + ', '.join(f'{a} ({t})' for a, t in top[:40]))
    import pro_traits
    try:
        panels = pro_traits.of_comp(comp)                # the trait panel of each strong player's round
    except Exception:
        panels = {}
    panels = {k: [(n, b) for n, b in v if not stages.is_heldout(comp, n)] for k, v in panels.items()}
    md += compare(comp, panels)
    for g in games:
        folder = ROOT / 'runs' / g['game']
        reads = jsonl(folder / 'read' / 'reads.jsonl')
        caps = captions(folder)
        keys = sorted((folder / 'read' / 'keyframes').glob('*.jpg'))
        for i, rows in enumerate(split_games(reads)):
            name = g['game'] + (f'-g{i + 1}' if i else '')
            used.append(name)
            md.append(f"\n## {name}: {g['channel']} - {g['title']}\n{g['url']}\n")
            if not caps:
                md.append('(no captions kept for this video)\n')
            for c in rounds_of(rows):
                said = ' '.join(t for s, t in caps if c['t0'] - 2 <= s <= c['t1'] + 2)[:700]
                key = next((k for k in keys if k.stem.endswith('-' + c['stage'])
                            and c['t0'] - 2 <= int(k.stem.split('-')[0]) <= c['t1'] + 2), None)
                lv, gold = c['level'], c['gold']
                tr = next((b for n, b in panels.get(c['stage']) or [] if n == name), None)
                md.append(f"- {c['stage']} ({int(c['t0']) // 60:02d}:{int(c['t0']) % 60:02d}): level {lv[0]} -> {lv[1]}, "
                          f"xp {c['xp']}, gold {'/'.join(map(str, gold)) if gold else '?'}, shops {c['shops']}"
                          + (f", traits fielded: {pro_traits.show(tr)}" if tr is not None else '')
                          + (f", keyframe {key.relative_to(ROOT).as_posix()}" if key else '')
                          + (f'\n  said: "{said}"' if said else ''))
    out = d / 'material.md'
    out.write_text('\n'.join(md) + '\n', encoding='utf-8')
    return out, used


def our_games(comp):
    """Our runs of the comp (agent/games.jsonl), newest last."""
    return [g['run'] for g in jsonl(AGENT / 'games.jsonl') if g.get('comp') == comp and g.get('run')
            and (ROOT / 'runs' / g['run']).exists()]


def compare(comp, panels, last=8):
    """Our games beside the strong players', round by round, on what both are measured on: trait tiers fielded,
    level, gold. Code lays the numbers side by side; the coach explains the gaps. -> markdown lines."""
    import pro_traits
    import rounds
    pro = {}
    for g in games_of(comp):
        reads = jsonl(ROOT / 'runs' / g['game'] / 'read' / 'reads.jsonl')
        for i, rows in enumerate(split_games(reads)):
            name = g['game'] + (f'-g{i + 1}' if i else '')
            for c in rounds_of(rows):
                tr = next((b for n, b in panels.get(c['stage']) or [] if n == name), None)
                pro.setdefault(c['stage'], []).append((pro_traits.tiers(tr) if tr is not None else None,
                                                       c['level'][1], (c['gold'] or (None,) * 3)[2]))
    ours = {}
    for run in our_games(comp)[-last:]:
        try:
            for r in rounds.game_rounds(ROOT / 'runs' / run):
                if r.get('board') is not None:
                    ours.setdefault(r['stage'], []).append((r.get('tiers_fielded'), r.get('level'), r.get('gold'),
                                                            r.get('tiers_best')))
        except Exception:
            continue
    if not pro and not ours:
        return []
    avg = lambda xs: (f"{sum(xs) / len(xs):.1f}" if xs else '-')
    col = lambda rows, k: [x[k] for x in rows if x[k] is not None]
    md = ["\n## Our games beside the strong players', round by round\n",
          f"Averages per round: trait tiers fielded (each breakpoint of 2 or more reached is one tier; read from the "
          f"strong players' trait panels and from our boards), level at the round's end, gold at its end. Ours: the "
          f"last {last} games of the comp ({', '.join(our_games(comp)[-last:]) or 'none'}); 'best' is the most trait "
          f"tiers our owned units and affordable shop cards made that round. Explain every gap you see.\n",
          "| round | strong: tiers | ours: tiers (best) | strong: level | ours: level | strong: gold | ours: gold |",
          "|---|---|---|---|---|---|---|"]
    for st in sorted(set(pro) | set(ours), key=lambda x: stage_key(x) or (99, 99)):
        p, o = pro.get(st) or [], ours.get(st) or []
        md.append(f"| {st} | {avg(col(p, 0))} | {avg(col(o, 0))} ({avg(col(o, 3))}) | {avg(col(p, 1))} | "
                  f"{avg(col(o, 1))} | {avg(col(p, 2))} | {avg(col(o, 2))} |")
    return md


def check_source(src):
    """A point's source must be a game we hold and a round that game had: a strong player's read game (runs/vod-*)
    or one of ours (runs/<run>, or its review's rounds when the run folder is gone)."""
    game, rnd = str(src.get('game') or ''), str(src.get('round') or '')
    base = re.sub(r'-g\d+$', '', game)
    folder = ROOT / 'runs' / base
    if not stage_key(rnd):
        return False
    # our games before 2026-11-03 08:38 lost their run folders; the round packets their reviews were written from
    # remain, and the Strategist cites them (games 55-57: five stage rewrites refused untested)
    rounds = AGENT / 'reviews' / f'{base}-rounds' / 'rounds.json'
    if not folder.exists():
        return rounds.exists() and f'"stage": "{rnd}"' in rounds.read_text(encoding='utf-8', errors='replace')
    reads = folder / 'read' / 'reads.jsonl'
    log = folder / 'log.jsonl'
    text = reads.read_text(encoding='utf-8') if reads.exists() else log.read_text(encoding='utf-8', errors='replace') \
        if log.exists() else ''
    return f'"stage": "{rnd}"' in text


def apply(comp, write=True):
    """The coach's proposal (agent/comps/<comp>/proposed.json) checked - each point needs what to do, why, and at least
    one source that checks - and written as the player's comp notes. -> {'kept', 'dropped', 'notes'}"""
    d = COMPS / slug(comp)
    prop = json.loads((d / 'proposed.json').read_text(encoding='utf-8'))
    kept, dropped, areas = 0, [], []
    for a in prop.get('areas') or []:
        pts = []
        for p in a.get('points') or []:
            srcs = [s for s in p.get('sources') or [] if isinstance(s, dict) and check_source(s)]
            if not (str(p.get('text') or '').strip() and str(p.get('why') or '').strip() and srcs):
                dropped.append({'area': a.get('area'), 'text': p.get('text'),
                                'why': 'no reason' if not p.get('why') else 'no source that checks'})
                continue
            pts.append(dict(p, sources=srcs))
        if pts:
            areas.append({'area': a.get('area'), 'points': pts})
            kept += len(pts)
    import stages
    staged, stage_dropped = [], []
    have = stages.current(comp)
    for s, st in (prop.get('stages') or {}).items():
        # version 1 of each stage strategy; later versions are the Strategist's, checked by the Judge
        s = stages.stage_of(f'{s}-1') if str(s).isdigit() else ''
        if not s or have.get(s) or not isinstance(st, dict):
            continue
        srcs = [x for x in st.get('sources') or [] if isinstance(x, dict) and check_source(x)
                and not stages.is_heldout(comp, x.get('game'))]
        ok, why = stages.check(st.get('text'))
        if not ok or not srcs:
            stage_dropped.append({'stage': s, 'why': why or 'no source that checks'})
        elif write:
            ver, why = stages.add_version(comp, s, st['text'], st.get('why') or 'version 1, from strong players\' games',
                                          srcs, by='comp coach')
            (staged if ver else stage_dropped).append(s if ver else {'stage': s, 'why': why})
        else:
            staged.append(s)
    if not write:                                        # --check: what would be dropped, nothing written
        return {'kept': kept, 'dropped': dropped, 'stages': staged, 'stages_dropped': stage_dropped}
    know = {'comp': comp, 'updated': datetime.now().isoformat(timespec='seconds'), 'games': prop.get('games') or [],
            'areas': areas, 'dropped': dropped, 'study_next': prop.get('study_next') or []}
    (d / 'knowledge.json').write_text(json.dumps(know, ensure_ascii=False, indent=1), encoding='utf-8')
    notes = render(comp)
    save_state(comp, games_used=prop.get('games') or [], updated=know['updated'], kept=kept, dropped=len(dropped),
               with_words=sorted(with_words(comp)), ours_used=reviewed_ours(comp), brief=brief_hash())
    return {'kept': kept, 'dropped': dropped, 'notes': str(notes.relative_to(ROOT)), 'stages': staged,
            'stages_dropped': stage_dropped}


def rules(comp):
    try:
        return json.loads((COMPS / slug(comp) / 'rules.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []


def add_rule(comp, prop):
    """A hard rule for the comp (the Strategist's "comp_rule" proposal; the owner, 2026-10-29: a decision the review
    finds the model getting wrong "could hard code this to avoid it happening again"): kept when it says what the
    harness enforces and why, from at least one round of our games or strong players' that checks. The executor
    enforces it (an Editor request), and the player's comp notes list it first. -> (the rule kept, or None, why not)."""
    srcs = [s for s in prop.get('sources') or [] if isinstance(s, dict) and check_source(s)]
    if not (str(prop.get('rule') or '').strip() and str(prop.get('why') or '').strip()):
        return None, 'a comp rule needs "rule" and "why"'
    if not srcs:
        return None, "no source that checks (a run of ours or a strong player's game, and a round it had)"
    rs = [r for r in rules(comp) if r.get('rule') != prop['rule']]
    rule = {'id': prop.get('id'), 'rule': prop['rule'], 'why': prop['why'], 'sources': srcs,
            'added': datetime.now().isoformat(timespec='seconds')}
    rs.append(rule)
    d = COMPS / slug(comp)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'rules.json').write_text(json.dumps(rs, ensure_ascii=False, indent=1), encoding='utf-8')
    render(comp)
    return rule, None


def render(comp):
    """The player's comp notes (agent/memory/comps/<comp>.md): the comp's hard rules first, then the comp coach's
    knowledge by area."""
    try:
        know = json.loads((COMPS / slug(comp) / 'knowledge.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        know = {}
    import expert
    lines = []
    # the owner's rules (agent/expert/rules): first, as rules only - the owner's words and reasons stay with the roles
    # (the owner, 2026-10-30: the feedback goes to the framework, "not the player"; 2026-11-02: "add rules")
    # (a comp's own rules here; the rules for every comp go to agent/memory/general.md - render(expert.GENERAL))
    owner = expert.rules(comp, general=False)
    if owner['rules'] or owner['banned_augments']:
        lines.append("FIRST RULES for every comp - where anything else (a comp's notes, a stage strategy) says "
                     "otherwise, these hold:" if comp == expert.GENERAL else
                     f"FIRST RULES for {comp} - where anything below says otherwise, these hold:")
        lines += [f"- {r['rule']}" for r in owner['rules']]
        if owner['banned_augments']:
            lines.append(f"- Augments never picked in this comp: {', '.join(owner['banned_augments'])}.")
        lines.append('')
    if rules(comp):
        lines.append(f"RULES for {comp} - the controller keeps them even when a plan breaks them:")
        for r in rules(comp):
            lines.append(f"- {r['rule']}\n  why: {r['why']}")
        lines.append('')
    if know.get('areas'):
        lines.append(f"What strong players do with {comp}, and why - learned from their games (and ours) by the comp "
                     f"coach, {str(know.get('updated'))[:10]}. Each point says when it applies and why; the sources "
                     f"are the games it was seen in.")
        for a in know['areas']:
            lines.append(f"\n## {a['area']}")
            for p in a['points']:
                src = '; '.join(f"{s['game']} {s['round']}" for s in p['sources'][:4])
                lines.append(f"- {p['text']}" + (f" (when: {p['when']})" if p.get('when') else '')
                             + f"\n  why: {p['why']}\n  seen in: {src}")
    MEMORY_COMPS.mkdir(parents=True, exist_ok=True)
    notes = MEMORY_COMPS.parent / 'general.md' if comp == expert.GENERAL else MEMORY_COMPS / f'{slug(comp)}.md'
    notes.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return notes


def due(comp):
    """What the comp needs next: 'find' (fewer than MIN_GAMES strong players' games read and none queued), 'coach'
    (games read that its notes have not used yet), or None."""
    import study_queue
    import stages
    games = games_of(comp)
    queued = [r for r in study_queue.rows() if r.get('comp') == comp and r.get('status') in ('open', 'needs download')]
    st = state(comp)
    # the games kept for the Judge never reach the notes, so they do not count towards MIN_GAMES (comp_select.learned
    # counts the games the notes used): 2026-11-03, three read games, one held out - no search, never learned, stop
    usable = [g for g in games if not stages.is_heldout(comp, g['game'])]
    if len(usable) < MIN_GAMES and not queued:
        last = st.get('find_at')
        recent = last and (datetime.now() - datetime.fromisoformat(last)).total_seconds() < FIND_AGAIN_H * 3600
        if not (recent and not st.get('find_found')):
            return 'find'
    if missing_words(comp):
        return 'words'
    used = set(st.get('games_used') or [])
    # the games kept for the Judge are never in the material, so never "used": counted, they made the coach due
    # after every run (game 52: ten runs in a row, each posting the owner request again)
    have = {g['game'] for g in games if not stages.is_heldout(comp, g['game'])}
    if have and not any(stages.current(comp).values()):              # no stage strategies yet: version 1 to write
        return 'coach'
    if have and not have <= {re.sub(r'-g\d+$', '', u) for u in used}:
        return 'coach'
    if with_words(comp) - set(st.get('with_words') or []):          # new words for games it used: written again
        return 'coach'
    if set(reviewed_ours(comp)) - set(st.get('ours_used') or []):    # a game of ours reviewed since: written again
        return 'coach'
    if have and st.get('brief') != brief_hash():                     # its brief asks for more since: written again
        return 'coach'
    return None


def brief_hash():
    """agent/roles/comp_coach.md's hash: the notes are written again when what the brief asks for changes."""
    import hashlib
    f = ROOT / 'agent' / 'roles' / 'comp_coach.md'
    return hashlib.sha1((f.read_bytes() if f.exists() else b'').replace(b'\r', b'')).hexdigest()[:12]


def reviewed_ours(comp):
    """Our games of the comp with a round-by-round review (agent/model/reviews/<run>.md)."""
    return [r for r in our_games(comp) if (AGENT / 'model' / 'reviews' / f'{r}.md').exists()]


def with_words(comp):
    """The comp's games with the player's words (captions or a transcription)."""
    return {g['game'] for g in games_of(comp) if list((ROOT / 'runs' / g['game']).glob('game*.vtt'))}


def missing_words(comp):
    """The comp's games read without the player's words and not tried yet."""
    tried = set(state(comp).get('no_words') or [])
    return [g for g in games_of(comp) if g['game'] not in tried and g['game'] not in with_words(comp)]


def words_for(comp):
    """Each of the comp's games without words transcribed from its audio (agent/transcribe.py); one that gives none is
    not tried again. -> {game: what happened}"""
    import study_queue
    import transcribe
    rows = {r.get('url'): r for r in study_queue.rows()}
    out = {}
    for g in missing_words(comp):
        try:
            out[g['game']] = transcribe.transcribe(ROOT / 'runs' / g['game'], g['url'],
                                                   (rows.get(g['url']) or {}).get('section'))
        except Exception as e:
            out[g['game']] = f'failed: {e.__class__.__name__}: {e}'[:300]
        if not list((ROOT / 'runs' / g['game']).glob('game*.vtt')):
            save_state(comp, no_words=sorted(set(state(comp).get('no_words') or []) | {g['game']}))
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('comp')
    ap.add_argument('--find', action='store_true')
    ap.add_argument('--material', action='store_true')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if a.find:
        print('queued:', find(a.comp))
    if a.material:
        path, used = material(a.comp)
        print(path, used)
    if a.apply:
        print(json.dumps(apply(a.comp), ensure_ascii=False, indent=1))
    if a.check:
        print(json.dumps(apply(a.comp, write=False), ensure_ascii=False, indent=1))
    if not (a.find or a.material or a.apply or a.check):
        print('due:', due(a.comp), '| games:', [g['game'] for g in games_of(a.comp)])


if __name__ == '__main__':
    main()
