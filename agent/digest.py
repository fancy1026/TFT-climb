"""A finished game as a compact review packet for the coach (agent/COACH.md).

    python agent/digest.py runs/<id> [--out agent/reviews/<id>-digest.md] [--fetch-lobby]

--fetch-lobby: the lobby's final boards from the client's match history when the game's record has none (a game
we left before the match ended): run it again in the review.

From the run folder's log.jsonl: the result, a table of every round at its combat start (HP, gold, level, the
board with stars, items and hexes, the bench, the item bench, the fight's result), augments and carousels, each
model decision with its timing and its own reasons, and the harness faults of the game. The full model
conversation stays in runs/<id>/conversation.md for anything the packet leaves out.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

# events that mean the controller did not do what the plan asked, or lost track of the game
FAULTS = ('step_failed', 'plan_rejected', 'unit_dropped', 'input_failed', 'roster_mismatch', 'unit_missing',
          'items_short', 'merge_did_not_happen', 'merge_unclear', 'buy_landing_unseen', 'auto_fill_refused',
          'bench_blocked', 'augment_missed', 'tick_error', 'game_strategy_missing', 'item_unrecognised',
          'landed_elsewhere', 'no_focus', 'hp_rejected')
KINDS = ('plan', 'update', 'augment', 'wisp', 'choice', 'carousel_prep', 'container', 'review', 'briefing')


def events(run):
    out = []
    for line in (Path(run) / 'log.jsonl').read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def objects(text):
    """The JSON objects of a model answer, in order."""
    try:
        from tft_player.coach import json_objects
        return [o for o in json_objects(text or '') if isinstance(o, dict)]
    except Exception:
        return []


def short_plan(plan):
    """The execution plan as one compact line (no 'why')."""
    if not isinstance(plan, dict):
        return ''
    s = json.dumps(plan, ensure_ascii=False, separators=(',', ':'))
    return s if len(s) < 600 else s[:600] + '...'


def state_line(prompt):
    """Gold / level / xp / streak / shop from a decision's state text."""
    p = prompt or ''
    bits = []
    m = re.search(r'Gold (\d+), level (\d+), xp \((\d+), (\d+)\)', p)
    if m:
        bits.append(f'gold {m.group(1)} lvl {m.group(2)} xp {m.group(3)}/{m.group(4)}')
    m = re.search(r'~(\d+) ?s left', p)
    if m:
        bits.append(f'{m.group(1)}s left')
    m = re.search(r'^Streak: (.+)$', p, re.M)
    if m:
        bits.append('streak ' + m.group(1))
    shop = re.findall(r'^  slot (\d): ([^\n(]+?)(?: \((\d)g)', p, re.M)
    if shop:
        bits.append('shop ' + ', '.join(f'{n.strip()}({c})' for _, n, c in shop))
    m = re.search(r'^Item bench: (.+)$', p, re.M)
    if m:
        bits.append('item bench: ' + m.group(1)[:200])
    return '; '.join(bits)


def milestones(by):
    """The stages that separated the account's good games from the bad (review of game 12): Coven 3 fielded by 3-1
    in all four top-2 games, after 3-1 in all six 5th-7th; level 8 by 4-5 in every top-2 game. From the fight
    events' board (each champion once)."""
    sys.path.insert(0, str(ROOT))
    try:
        from tft_player.catalog import CATALOG
    except Exception:
        return 'Milestones: (catalog not loaded)'
    coven = lvl8 = None
    for e in by.get('fight', []):
        st = e.get('stage')
        m = re.search(r'board=\[(.*?)\] bench=', e.get('roster') or '')
        names = set(re.findall(r"([A-Z][\w'. ]*?)\*+@h", m.group(1))) if m else set()
        n = sum(1 for x in names if CATALOG.unit(x) and 'Coven' in CATALOG.unit(x).traits)
        if coven is None and n >= 3:
            coven = st
        if lvl8 is None and (e.get('level') or 0) >= 8:
            lvl8 = st
    essence = next((e.get('stage') for e in by.get('choice_pick', []) if 'Essence' in str(e.get('name'))), None)
    return (f'Milestones: Coven 3 fielded from {coven or "never"} (top-2 games: by 3-1); first Essence choice '
            f'{essence or "none"}; level 8 from {lvl8 or "never"} (top-2 games: by 4-5)')


def three_star_lines(by):
    """The comp's 3-star targets (auto_buy at 3 in agent/config.json or in the game's standing orders) - copies owned
    at each combat start, board + bench (a 2-star = 3, a 3-star = 9) - and the gold each stage put into XP and how
    many shops it rolled. Game 15: Elise 1-star until 5-1 while 76 gold went into XP from 4-1 to 4-3; the Master+
    boards of the comp at level 8 place 1.94 with Elise 3 + Cassiopeia 3, 7.11 without a 3-star."""
    try:
        cfg = json.loads((ROOT / 'agent' / 'config.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        cfg = {}
    want = {n for n, s in (cfg.get('auto_buy_stars') or {}).items() if s == 3}
    for e in by.get('auto_orders', []):
        want |= {n for n, s in (e.get('auto_buy') or {}).items() if s == 3}
    out = []
    if want:
        rows = []
        for n in sorted(want):
            seq, last = [], 0
            for e in by.get('fight', []):
                k = sum(3 ** (len(s) - 1) for s in re.findall(re.escape(n) + r'(\*+)@[hb]', e.get('roster') or ''))
                if k != last:
                    seq.append(f"{e.get('stage')} {k}")
                    last = k
            rows.append(f"{n}: {', '.join(seq) or 'never owned'}")
        out.append('3-star target copies at combat start, of 9 (board + bench; when the count changed): ' + '; '.join(rows))
    xp, rolls = defaultdict(int), defaultdict(int)
    for e in by.get('action', []):
        m = re.match(r'level\((\d+)\)', str(e.get('act') or ''))
        if m and e.get('ok') is not False:
            xp[str(e.get('stage'))[:1]] += 4 * int(m.group(1))
    for e in by.get('rolldown', []):
        if isinstance(e.get('rolls'), int):
            rolls[str(e.get('stage'))[:1]] += e['rolls']
    wisps = wisp_buys(by)
    stages = sorted(set(xp) | set(rolls) | set(wisps))
    if stages:
        out.append('Spent per stage (XP gold / shops rolled, free rerolls included / Wisps bought and their gold): '
                   + ', '.join(f'stage {s} {xp[s]}g / {rolls[s]} / {len(wisps.get(s, []))} for '
                               f'{sum(c or 0 for _, c in wisps.get(s, []))}g' for s in stages))
    # wanted cards a roll showed and could not buy, and why the rolls stopped (game 19 4-5: Morgana, one copy from
    # her 2-star, in the last shop of two rolls that had spent the gold to 3 and to 0)
    missed = [f"{e.get('stage')} {m}" for e in by.get('rolldown', []) for m in e.get('missed') or []]
    stops = defaultdict(int)
    for e in by.get('rolldown', []):
        if e.get('stopped'):
            stops[e['stopped']] += 1
    if missed or stops:
        out.append('Roll-downs: wanted cards seen, not bought: ' + ('; '.join(missed) or 'none')
                   + ' | stops: ' + (', '.join(f'{k} {v}' for k, v in sorted(stops.items())) or 'none'))
    return out


def wisp_buys(by):
    """Wisps bought, by stage: [(name, cost)] - the roll-downs' own buys, the player's "buy" answers to Wisp questions
    and its plans' Wisp buys (one per stage and name). Game 16: ~30 gold of Wisps in stages 5-6 at level 7, where 44
    gold was level 8."""
    names = {}
    for e in by.get('wisp_asked', []):
        names[e.get('wisp')] = e.get('cost')
    for e in by.get('model_call', []):
        for m in re.finditer(r"WISP '(.+?)' \((\d+)g\)", str(e.get('prompt') or '')):
            names.setdefault(m.group(1), int(m.group(2)))
    got = {}
    for e in by.get('rolldown_wisp', []):
        if e.get('card') or e.get('cost') is not None:
            name = next((n for n in names if n and e.get('card') and n.lower() in str(e['card']).lower()), e.get('card'))
            got[(str(e.get('stage')), name)] = e.get('cost')
    for e in by.get('model_call', []):
        if e.get('kind') == 'wisp' and re.search(r'"buy":\s*true', str(e.get('answer') or e.get('text') or '')):
            m = re.search(r"Wisp '(.+?)', costs (\d+) gold", str(e.get('prompt') or ''))
            if m:
                got[(str(e.get('stage')), m.group(1))] = int(m.group(2))
    for e in by.get('action', []):
        m = re.match(r'buy\(\d\) - (?:plan: )?wisp (.+?) \(', str(e.get('act') or ''))
        if m and e.get('ok') is not False:
            got.setdefault((str(e.get('stage')), m.group(1)), names.get(m.group(1)))
    out = defaultdict(list)
    for (stage, name), cost in sorted(got.items()):
        out[stage[:1]].append((name, cost))
    return out


def stage_start_lines(by):
    """A stage's first planning phase (3-1, 4-1, 5-1, 6-1): did the last fight's next_round run from its start, how long
    its first answer took, and the level from the phase's start to its fight. Game 17: first answers of 19, 24 and
    25 s with no next_round at 3-1 and 5-1 - the 3-1 rolls and every 5-1 step never ran (the 2-7/3-7/4-7 fights now
    ask for one)."""
    rows = []
    for st in ('3-1', '4-1', '5-1', '6-1'):
        calls = [e for e in by.get('model_call', []) if e.get('stage') == st and e.get('kind') in ('plan', 'update')]
        if not calls:
            continue
        made = next((e.get('made') for e in by.get('next_round_started', []) if e.get('stage') == st and e.get('made')),
                    None)
        start = next((re.search(r' L(\d+) ', e.get('screen') or '') for e in by.get('round', []) if e.get('stage') == st),
                     None)
        fight = next((e.get('level') for e in by.get('fight', []) if e.get('stage') == st), None)
        # the x-7 fight's question that carries this phase's next_round (g18: none at 3-7, 49 s at 4-7)
        prev = f'{int(st[0]) - 1}-7'
        cs = next((e for e in by.get('combat_start_call', []) if e.get('stage') == prev), None)
        ans = cs and next((e for e in by.get('model_call', []) if e.get('kind') == 'update' and e.get('t')
                           and -1 <= e['t'] - (e.get('latency') or 0) - cs['t'] <= 5), None)
        asked = (f'{prev} question not asked' if not cs else
                 f"{prev} question answered in {ans.get('latency')} s" if ans else f'{prev} question unanswered')
        own = [e for e in calls if e is not ans]      # an x-7 answer that came late is logged under this stage
        rows.append(f"{st} next_round {'from ' + made if made else 'none'} ({asked}), first answer "
                    f"{own[0].get('latency') if own else '-'} s, level {start.group(1) if start else '?'} -> {fight}")
    return 'Stage-start phases: ' + ('; '.join(rows) or 'none')


def fight_lines(by):
    """Each stage's player fights, won or lost by the HP at the next combat start (the round table's own result is
    often '?'), the HP lost, and the gold held at the start of each lost fight at level 7+ (game 21: level 7 from 3-7,
    35, 45, 31 and 34 gold held through the lost 4-1, 4-2, 4-5 and 4-6 while saving for a level 8 that came at 5-1;
    one roll at 7, at 4-3, won the only stage-4 fight; 62 -> 16 HP)."""
    fights = [e for e in by.get('fight', []) if e.get('hp') is not None and re.match(r'[2-9]-[1-6]$', str(e.get('stage')))]
    nxt = fights       # the next player fight: an x-7 start can show the HP before the loss (game 21 4-6 -> 4-7: 29)
    rows = defaultdict(lambda: {'w': 0, 'l': 0, 'hp': 0, 'held': []})
    for e in fights:
        later = next((x for x in nxt if x['t'] > e['t'] and x.get('stage') != e.get('stage')), None)
        if later is None:
            continue
        r = rows[str(e['stage'])[0]]
        if later['hp'] < e['hp']:
            r['l'] += 1
            r['hp'] += e['hp'] - later['hp']
            if (e.get('level') or 0) >= 7 and e.get('gold') is not None:
                r['held'].append(f"{e['stage']} {e['gold']}g")
        else:
            r['w'] += 1
    if not rows:
        return 'Fights by stage: none'
    return ('Fights by stage (won / lost by the HP at the next combat start; HP lost; gold held at the start of lost '
            'fights at level 7+): ' + '; '.join(f"stage {s} {r['w']}W {r['l']}L -{r['hp']} HP"
                                                + (f" (held {', '.join(r['held'])})" if r['held'] else '')
                                                for s, r in sorted(rows.items())))


def idle_items(by):
    """Items not fighting at each combat start (review of game 13: Cassiopeia's and Elise's lone Tears waited 3-1 to
    5-2, Protector's Vow + Stoneplate sat on a benched Rakan 4-6 and 4-7): components on the item bench, lone
    components on fielded units, items on benched units. From the fight events' roster (item names cut to their
    first word) and item bench. The fourth number: completed items on the item bench (game 20: Void Staff there in the
    lost 3-5 fight after the final-holder move sold Caitlyn, Spear of Shojin and Hextech Gunblade in the lost 4-2 one
    after a Remover strip - the component counts showed neither)."""
    sys.path.insert(0, str(ROOT))
    try:
        from tft_player.items import COMPONENTS, RECIPES
    except Exception:
        return 'Idle items: (items not loaded)'
    short = {c.split()[0] for c in COMPONENTS}
    rows = []
    for e in by.get('fight', []):
        m = re.search(r'board=\[(.*?)\] bench=\[(.*)\]', e.get('roster') or '')
        if not m:
            continue
        held = lambda part: [x.split(',') for x in re.findall(r'@[hb]\d+\[([^\]]*)\]', part)]
        lone = sum(1 for items in held(m.group(1)) for i in items if i in short)
        benched = sum(len(items) for items in held(m.group(2)))
        loose = sum(1 for i in e.get('items') or [] if i in COMPONENTS)
        done = [i for i in e.get('items') or [] if i in RECIPES]
        if lone or benched or loose or done:
            rows.append(f'{e.get("stage")} {loose}/{lone}/{benched}/{len(done)}'
                        + (f" ({', '.join(done)})" if done else ''))
    return ('Idle items at combat start (components on the item bench / lone components on fielded units / items '
            'on benched units / completed items on the item bench): ' + (', '.join(rows) or 'none'))


# augment cards that give emblems or reward their holders (the owner's note 2026-10-23: U.R.F with Branching Out and
# every augment that gives emblems)
EMBLEM_CARDS = re.compile(r"U\.R\.F|Urf's|Branching Out|Spreading Roots|Trait Tree|Flexible|Hard Commit|Trait Ladder|"
                          r"Frontline Foundation|Backline Blueprint|Legion Of Threes|Tactician's Kitchen|We Stick Together|"
                          r"Cooking Pot|Consuming Flora", re.I)
CARD_RX = re.compile(r'(?m)^[ *#]*Card \d \((?:left|middle|right)\)[*:. ]*(.+)$')


def card_name(rest):
    """'Branching Out+** - "Gain a random ..."' -> 'Branching Out+' (the name as an augment answer writes it)."""
    s = rest.replace('*', '')
    for cut in (' - ', ' – ', ': ', ' "', '"', '. Card text', ' (', '. I explained'):
        s = s.split(cut)[0]
    return s.strip(' .:')


def augment_lines(by):
    """Each augment choice: the cards its answers explained, the pick, the augments held before it and which of them
    the answers named, the emblem cards; then emblem items idle on the item bench at combat start. The owner's note
    2026-10-23 (agent/reviews/owner-note-20261023-augments.md): in g1-g21, 1 of the 96 answers given while an earlier
    augment was held named it, an emblem card was taken once in 18 offers, and Spatulas or emblems sat on the item
    bench all game in g10, g13 and g18."""
    from tft_player.coach import stage_key
    picks = [(e.get('stage'), e.get('name')) for e in by.get('augment', [])]
    answers = defaultdict(list)
    for e in by.get('model_call', []):
        if e.get('kind') == 'augment':
            answers[e.get('stage')].append(e.get('answer') or '')
    out = []
    for stage in sorted(answers, key=lambda s: stage_key(s) or (9, 9)):
        texts = answers[stage]
        cards = list(dict.fromkeys(card_name(m) for t in texts for m in CARD_RX.findall(t)))
        held = [n for s, n in picks if stage_key(s) and stage_key(stage) and stage_key(s) < stage_key(stage)]
        named = [h for h in held if any(str(h).split('+')[0].strip().lower() in t.lower() for t in texts)]
        emblem = [c for c in cards if EMBLEM_CARDS.search(c)]
        out.append(f"- {stage}: took {next((n for s, n in picks if s == stage), '?')}; cards {', '.join(cards) or '?'}; "
                   f"held before: {', '.join(map(str, held)) or 'none'}"
                   + (f" - named in its answers: {', '.join(map(str, named)) or 'NONE'}" if held else '')
                   + (f"; emblem cards: {', '.join(emblem)}" if emblem else ''))
    try:
        from tft_player.emblems import is_emblem_item
    except Exception:
        return out
    runs = []                     # [first stage, last stage, combats, items]: the same items at consecutive combats
    for e in by.get('fight', []):
        em = sorted(i for i in e.get('items') or [] if i and is_emblem_item(i))
        if em and runs and runs[-1][3] == em:
            runs[-1][1], runs[-1][2] = e.get('stage'), runs[-1][2] + 1
        elif em:
            runs.append([e.get('stage'), e.get('stage'), 1, em])
    out.append('Emblems, Spatulas and Frying Pans on the item bench at combat start: ' + ('; '.join(
        f"{a}{f' to {b} ({n} combats)' if n > 1 else ''}: {', '.join(em)}" for a, b, n, em in runs) or 'none'))
    return out


def lobby_lines(res):
    """Every player's final board from the match history (game_result.lobby): what beat us. Game 14 lost five
    final-duel fights from 68 HP to a 7-HP player whose board no fight picture showed (Riftbeast 8, three 3-stars)."""
    lobby = res.get('lobby') or []
    if not lobby:
        return []
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from top_games import item_name, trait_name, unit_name
    out = ['\n## The lobby\'s final boards (match history; place, level, gold left, last round)']
    for x in sorted(lobby, key=lambda x: x.get('place') or 9):
        units = ' '.join(f"{unit_name(c)}{t}" + (f"[{', '.join(item_name(i) for i in its)}]" if its else '')
                         for c, t, its in x.get('units') or [])
        traits = ', '.join(f'{trait_name(n)} {k}' for n, k in x.get('traits') or [])
        out.append(f"- {x.get('place')}{' (us)' if x.get('us') else ''}: L{x.get('level')}, {x.get('gold_left')} gold, "
                   f"round {x.get('last_round')} | {units} | {traits}")
    return out


def fetch_lobby(res):
    """--fetch-lobby: the lobby's final boards from the client's match history, when the packet was made before the
    match ended (game 15 was out 6th while seven others played on: no lobby in its packet; fetched in the review, it
    showed two other Coven boards - Elise 3 and Elise 2 - and a 3-star carry on each of the top three)."""
    if res.get('lobby') or not res.get('started_ms'):
        return res
    sys.path.insert(0, str(ROOT))
    try:
        from tft_player.ending import fetch_placement
        from tft_player.lobby import LeagueClient
        c = LeagueClient()
        try:
            got = fetch_placement(c, int(res['started_ms']), tries=1, wait=0) or {}
        finally:
            c.close()
    except Exception as e:
        print(f'lobby not fetched: {e!r}')
        return res
    return {**res, 'lobby': got.get('lobby')} if got.get('lobby') else res


def recurring_code():
    """The code that earlier reviews changed again and again (agent/learning.jsonl, code fixes): each identifier
    and the games after which it was changed - where fixes keep landing."""
    f = ROOT / 'agent' / 'learning.jsonl'
    if not f.exists():
        return []
    seen = defaultdict(set)
    for line in f.read_text(encoding='utf-8').splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get('kind') != 'Code fix':
            continue
        for ident in set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[a-z]+_[a-z0-9_]+',
                                    str(r.get('change')) + ' ' + str(r.get('learned')))):
            if not ident.endswith(('.py', '.md', '.json', '.jsonl')) and ident not in ('tft_player', 'agent_check'):
                seen[ident.split('.')[-1]].add(r.get('game'))
    top = sorted(((len(g), k, sorted(g)) for k, g in seen.items() if len(g) >= 2), reverse=True)[:10]
    if not top:
        return []
    return ['\n## Code the reviews keep changing (agent/learning.jsonl, code fixes)',
            *[f'- {k}: changed after {n} games ({", ".join(map(str, g))})' for n, k, g in top]]


def main():
    args = sys.argv[1:]
    out = None
    if '--out' in args:
        i = args.index('--out')
        out = Path(args[i + 1])
        del args[i:i + 2]
    lobby = '--fetch-lobby' in args
    args = [a for a in args if a != '--fetch-lobby']
    run = Path(args[0])
    evs = events(run)
    by = defaultdict(list)
    for e in evs:
        by[e.get('ev')].append(e)
    L = []
    res = (by.get('game_result') or [{}])[-1]
    if lobby:
        res = fetch_lobby(res)
    cfg = (by.get('run_config') or [{}])[-1]
    sess = (by.get('model_session') or [{}])[-1]
    try:
        from game_metrics import metrics
        m = metrics(str(run))
    except Exception as ex:
        m = {'error': str(ex)}
    L.append(f"# Game {run.name}: {res.get('comp_name') or m.get('comp')} - place {res.get('placement') or m.get('placement')}")
    L.append(f"model {sess.get('model')} effort {sess.get('effort')} ({sess.get('backend')}); last stage "
             f"{res.get('last_stage')}; {round((res.get('seconds') or 0) / 60, 1)} min; code {cfg.get('git') or m.get('git')}"
             f"{' (dirty)' if cfg.get('dirty') or m.get('dirty') else ''}; result {res.get('result')}")
    if m.get('rounds'):
        hp = {r['stage']: r.get('hp') for r in m['rounds']}
        lv = {r['stage']: r.get('level') for r in m['rounds']}
        L.append('HP at combat start: ' + ', '.join(f'{s} {hp.get(s)}' for s in ('2-1', '3-1', '3-5', '4-1', '4-5', '5-1', '5-5', '6-1') if s in hp))
        L.append('Level at combat start: ' + ', '.join(f'{s} {lv.get(s)}' for s in ('2-1', '2-5', '3-2', '3-5', '4-1', '4-2', '4-5', '5-1') if s in lv))
    L.append(milestones(by))
    L.append(fight_lines(by))
    L += three_star_lines(by)
    L.append(idle_items(by))
    try:                                                 # the carry's items on other units (agent/metrics.py)
        import metrics as climb_metrics
        cm = climb_metrics.game(run)
        held = climb_metrics.carry_holders(by.get('fight', []), cm.get('carry'), cm.get('item_map'))
        if held:
            L.append(f"The carry's ({cm.get('carry')}) items on other units at combat start (each holder is one more "
                     "move when he is fielded): " + '; '.join(
                         f"{h['stage']}{'' if h['fielded'] else ' (carry not fielded)'} "
                         + ', '.join(f"{u} {'+'.join(i)}" for u, i in h['holders'].items()) for h in held))
    except Exception as ex:
        L.append(f"(the carry's item holders not measured: {ex.__class__.__name__})")
    L.append(stage_start_lines(by))
    for k in ('comp_final', 'comp_best', 'carry', 'carry_stars_final', 'level_final', 'late', 'late_plans', 'cost_usd',
              'sells_failed', 'failed_steps', 'rejected'):
        if k in m:
            L.append(f'{k}: {m[k]}')

    L += lobby_lines(res)

    # augments, carousels, choices
    L.append('\n## Augments, carousels, choices')
    for e in by.get('augment', []):
        L.append(f"- {e.get('stage')} augment: {e.get('name')} (source {e.get('source')})")
    L.append('Augment choices (the cards the answers explained; the augments held before and which the answers named):')
    L += augment_lines(by)
    for e in by.get('carousel_result', []) + by.get('carousel', []):
        L.append(f"- {e.get('stage')} carousel: " + json.dumps({k: v for k, v in e.items() if k not in ('ev', 't', 'stage')},
                                                              ensure_ascii=False)[:300])
    for e in by.get('choice_pick', []):
        L.append(f"- {e.get('stage')} choice: " + json.dumps({k: v for k, v in e.items() if k not in ('ev', 't')},
                                                            ensure_ascii=False)[:300])

    # the round table
    L.append('\n## Rounds (at combat start; HP before the fight, then the fight result)')
    results = {e.get('stage'): e for e in by.get('fight_result', [])}
    rolls = defaultdict(list)
    for e in by.get('rolldown', []):
        if e.get('rolls') is None:
            continue
        rolls[e.get('stage')].append(f"{e.get('rolls')} rolls, bought {e.get('bought')}, gold left {e.get('gold_end')}"
                                     + (f", Wisps rolled past unanswered {e['wisps_passed']}" if e.get('wisps_passed') else ''))
    for e in by.get('comp rule', []):                    # older runs: the Wisps a roll rolled past, from its notice
        m = re.search(r'rolled past the Wisp\(s\) (.+?) without', e.get('what') or '')
        if m and not any('Wisps rolled past' in r for r in rolls[e.get('stage')]):
            rolls[e.get('stage')].append(f"Wisps rolled past unanswered [{m.group(1)}]")
    for e in by.get('fight', []):
        st = e.get('stage')
        fr = results.get(st) or {}
        items = [i for i in (e.get('items') or []) if i]
        L.append(f"- {st}: HP {e.get('hp')} gold {e.get('gold')} lvl {e.get('level')} | {e.get('roster')} | "
                 f"item bench {items} | fight {fr.get('result', '?')}{' (hp ' + str(fr.get('hp0')) + '->' + str(fr.get('hp')) + ')' if fr else ''}"
                 + (f" | rolls: {'; '.join(rolls[st])}" if rolls.get(st) else ''))

    # the model's decisions
    L.append('\n## Model decisions (kind, stage, seconds to the decision vs seconds left; plan; its reasons)')
    for e in by.get('model_call', []):
        kind = e.get('kind')
        if kind not in KINDS:
            continue
        objs = objects(e.get('answer'))
        plan = next((o.get('execution_plan') for o in objs if 'execution_plan' in o), None)
        strat = next((o.get('strategy') for o in objs if 'strategy' in o), None) or {}
        why = strat.get('why') if isinstance(strat.get('why'), dict) else {}
        gs = next((o.get('game_strategy') for o in objs if 'game_strategy' in o), None)
        other = [o for o in objs if not ({'execution_plan', 'strategy', 'game_strategy', 'next_round'} & set(o))]
        nxt = next((o.get('next_round') for o in objs if 'next_round' in o), None)
        head = (f"### {e.get('stage')} {kind} - decision {e.get('decision_s')}s, answer {e.get('latency')}s; "
                + state_line(e.get('prompt')))
        L.append(head)
        if kind == 'briefing':
            L.append('(preparation reply) ' + (str(e.get('answer')) or '')[:1500])
            continue
        if plan is not None:
            L.append('plan: ' + short_plan(plan))
        if nxt:
            L.append('next_round: ' + short_plan(nxt))
        if other:
            L.append('answer: ' + ' '.join(json.dumps(o, ensure_ascii=False)[:500] for o in other))
        for part in ('level', 'items', 'board', 'buy', 'sell'):
            if why.get(part) and why.get(part) != '-':
                L.append(f'  {part}: {why[part]}')
        if strat.get('notes'):
            L.append(f"  notes: {strat['notes']}")
        if gs and isinstance(gs, dict) and gs.get('why') not in (None, 'unchanged'):
            L.append(f"  game_strategy -> {gs.get('text')} (why: {gs.get('why')})")

    # harness faults
    L.append('\n## Harness faults and controller reports')
    counts = Counter(e.get('ev') for e in evs if e.get('ev') in FAULTS)
    L.append('counts: ' + ', '.join(f'{k} {n}' for k, n in counts.most_common()) if counts else 'none')
    for k in FAULTS:
        for e in by.get(k, [])[:6]:
            L.append(f"- {e.get('stage')} {k}: " + json.dumps({x: y for x, y in e.items() if x not in ('ev', 't', 'stage')},
                                                              ensure_ascii=False)[:400])
    L += recurring_code()
    text = '\n'.join(L) + '\n'
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding='utf-8')
        print(f'{out} ({len(text) // 1000} kB)')
    else:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        print(text)


if __name__ == '__main__':
    main()
