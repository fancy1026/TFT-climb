"""The change review: every change to the harness and the game knowledge is examined in the games after it, and kept,
removed or watched (the owner, 2026-10-31: "for all changes made to the harness and the game, during the next game, or
more, the changes should be evaluated and examined to keep or remove ... after 51, the reviewer should additionally
check whether each change made improving or not"; "it could choose to monitor for additional games, and it could revert
them if additional evaluation found not useful").

Code lists the changes: each game logs the commit it ran (run_config "git"), so the changes a game is the first to show
are the commits between the previous game's code and its own, on the paths that change how a game is played (WATCH).
Changes still being watched from earlier games are listed again. The Reviewer (agent/roles/reviewer.md, Job 3) gives
each a verdict - keep, remove or monitor - from what it did in the game; code records the verdicts here
(agent/changes.jsonl) and turns each "remove" into a request to the role that owns that kind of change (ROUTE). No role
writes this file.

    python agent/changes.py <run> [--since <run or commit>]   the packet for a game (agent/reviews/<run>-changes.md)
    python agent/changes.py                                   every change's latest verdict
"""
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
LEDGER = AGENT / 'changes.jsonl'
GAMES = AGENT / 'games.jsonl'
REQUESTS = AGENT / 'model' / 'requests.jsonl'

# what decides how a game is played, by kind (a role brief changes what that role writes, so it counts too)
WATCH = (('tft_player/', 'harness'), ('tools/', 'harness'), ('agent/comps/', 'comp rules'),
         ('agent/memory/comps/', 'comp notes'), ('agent/memory/', 'playbook'), ('agent/model/game-model.md', 'playbook'),
         ('agent/config.json', 'settings'), ('agent/roles/', 'roles'))
# who removes a change of each kind (requests.jsonl kinds): the Editor reverts code and the executor's comp rules, the
# comp coach drops a note, the Strategist proposes a principle's removal to the Judge, settings and role briefs go to
# the owner's side (the Director reads structural requests)
ROUTE = {'harness': 'editor', 'comp rules': 'editor', 'comp notes': 'coach', 'playbook': 'strategist',
         'settings': 'structural', 'roles': 'structural'}
VERDICTS = ('keep', 'remove', 'monitor')
EFFECTS = ('helped', 'hurt', 'none', 'not seen')


def git(*a):
    return subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                          errors='replace').stdout.strip()


def kind_of(path):
    for prefix, kind in WATCH:
        if path.startswith(prefix) and not (prefix == 'agent/comps/' and not path.endswith('rules.json')):
            return kind
    return None


def records():
    return [json.loads(l) for l in GAMES.read_text(encoding='utf-8').splitlines() if l.strip()] if GAMES.exists() else []


def run_code(run):
    """The commit a game ran (its run_config), or None."""
    f = ROOT / 'runs' / run / 'log.jsonl'
    if not f.exists():
        return None
    with open(f, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if '"run_config"' in line:
                try:
                    return json.loads(line).get('git')
                except ValueError:
                    return None
    return None


def previous_run(run):
    """The game played before `run` (agent/games.jsonl order, else run-folder names), with a logged commit."""
    runs = [r['run'] for r in records() if r.get('run')]
    if run not in runs:
        runs = sorted(set(runs) | {run})
    before = [r for r in runs[:runs.index(run)] if run_code(r)]
    return before[-1] if before else None


def game_of(run):
    return next((r.get('n') for r in records() if r.get('run') == run), None)


def by_of(subject, body):
    s = subject + ' ' + body
    for key, who in (('(the comp coach)', 'comp coach'), ('Judge verdicts', 'Judge'), ('Strategist after', 'Strategist'),
                     ('Harness fixer changes', 'Editor'), ('Requests the Editor', 'Editor'), ('Guard:', 'guard'),
                     ("The climb's comp is", 'framework (comp choice)')):
        if key in s:
            return who
    if 'committed by agent/climb.py' in s:
        return 'framework'
    return 'Editor' if re.search(r'\be\d+\b', s) or subject.startswith('Fix') else 'Coordinator'


_SESSIONS = {}


def source_of(h):
    """Where an Editor commit came from: the requests its session completed and the game whose review it worked
    (its session log agent/reviews/<tag>-harness_fixer.jsonl names the commit; 99a8ef0 names no request itself, its
    session completed e67). -> (request ids, the reviewed run or None)."""
    if not _SESSIONS:
        for f in (AGENT / 'reviews').glob('*-harness_fixer.jsonl'):
            m = re.search(r'(\d{8})-(\d{4})', f.name)
            try:
                _SESSIONS[f.name] = (m.group(1) + m.group(2) if m else '', f.read_text(encoding='utf-8', errors='replace'))
            except OSError:
                pass
    # the session it was made in: the latest one started before the commit that names it (a later session's log names
    # it too, as the commit it started from)
    made = git('log', '-1', '--format=%cd', '--date=format:%Y%m%d%H%M', h)
    for name, (started, text) in sorted(_SESSIONS.items(), key=lambda kv: kv[1][0], reverse=True):
        if started <= made and h in text:
            done = re.findall(r'requests_done\\?"\s*:\s*\[([^\]]*)\]', text)
            ids = sorted(set(re.findall(r'\b(e\d+)\b', done[-1]))) if done else []
            run = re.match(r'(\d{8}-\d{6})-harness_fixer', name)
            return ids, run.group(1) if run else None
    return [], None


def shown(subject, body):
    """(who, subject, body) as the Reviewer sees them: the Coordinator's commits carry the owner's feedback
    (agent/expert, which no role reads), so they are judged by their diff alone."""
    who = by_of(subject, body)
    if who == 'Coordinator':
        return who, 'a Coordinator change (judge it by its diff)', ''
    return who, subject, body.strip()


def commits(since, until):
    """[{change, subject, body, by, requests, files: [(path, kind, +, -)], kinds}] between two commits, oldest first."""
    out = []
    log = git('log', '--reverse', '--format=%h%x09%s%x09%b%x1e', f'{since}..{until}')
    for rec in log.split('\x1e'):
        parts = rec.strip().split('\t', 2)
        if len(parts) < 2 or not parts[0]:
            continue
        h, subject, body = parts[0], parts[1], parts[2] if len(parts) > 2 else ''
        files = []
        for line in git('show', '--numstat', '--format=', h).splitlines():
            bits = line.split('\t')
            if len(bits) == 3 and kind_of(bits[2]):
                files.append((bits[2], kind_of(bits[2]), bits[0], bits[1]))
        if not files:
            continue
        ids = set(re.findall(r'\b(e\d+)\b', subject + ' ' + body))
        reviewed = None
        if by_of(subject, body) == 'Editor':          # (an Editor's log also names the commit it started from)
            more, reviewed = source_of(h)
            ids |= set(more)
        who, subject, body = shown(subject, body)
        out.append({'change': h, 'subject': subject, 'body': body, 'by': who,
                    'requests': sorted(ids, key=lambda x: int(x[1:])), 'reviewed': reviewed, 'files': files,
                    'kinds': sorted({f[1] for f in files})})
    return out


def ledger():
    return [json.loads(l) for l in LEDGER.read_text(encoding='utf-8').splitlines() if l.strip()] if LEDGER.exists() else []


def latest():
    """{(change, part): the latest verdict row}."""
    out = {}
    for r in ledger():
        out[(r['change'], r.get('part') or '')] = r
    return out


def watched():
    """Changes whose latest verdict on some part is "monitor": {change: [rows]}."""
    out = {}
    for (h, _), r in latest().items():
        if r.get('verdict') == 'monitor':
            out.setdefault(h, []).append(r)
    return out


def removed(kinds=None):
    """Parts the change review removed (latest verdict "remove"), of any of `kinds`: the roles are told not to build
    them again."""
    return [r for r in latest().values() if r.get('verdict') == 'remove'
            and (kinds is None or set(kinds) & set(r.get('kinds') or [r.get('kind')]))]


def request_text(rid):
    if not REQUESTS.exists():
        return ''
    for l in REQUESTS.read_text(encoding='utf-8').splitlines():
        if f'"id": "{rid}"' in l:
            try:
                return json.loads(l).get('text') or ''
            except ValueError:
                return ''
    return ''


def listed(run, since=None):
    """The changes this game's review examines: new since the previous game (or `since`, a run or a commit), then the
    ones still watched. -> (changes, since_commit)."""
    until = run_code(run) or git('rev-parse', '--short', 'HEAD')
    if since and (ROOT / 'runs' / since).exists():
        since = run_code(since)
    elif not since:
        prev = previous_run(run)
        since = run_code(prev) if prev else None
    new = commits(since, until) if since else []
    have = {c['change'] for c in new}
    old = []
    for h, rows in watched().items():
        if h in have:
            continue
        who, subject, body = shown(git('log', '-1', '--format=%s', h), git('log', '-1', '--format=%b', h))
        files = [(p, k, a, d) for p, k, a, d in (
            (b[2], kind_of(b[2]), b[0], b[1]) for b in (l.split('\t') for l in git('show', '--numstat', '--format=', h).splitlines())
            if len(b) == 3) if k]
        old.append({'change': h, 'subject': subject, 'body': body,
                    'by': who, 'requests': [], 'files': files, 'kinds': sorted({f[1] for f in files}),
                    'watched': rows})
    return new + old, since


def packet(run, since=None):
    """agent/reviews/<run>-changes.md: the Reviewer's list for Job 3. -> (path, the changes)."""
    changes, since_c = listed(run, since)
    out = AGENT / 'reviews' / f'{run}-changes.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    n = game_of(run)
    lines = [f"# Changes to examine in game {n or '?'} (run {run})", '',
             f"This game ran commit {run_code(run) or '?'}. New since the previous game's commit {since_c or '?'}: "
             f"{sum(1 for c in changes if not c.get('watched'))}; still watched from earlier games: "
             f"{sum(1 for c in changes if c.get('watched'))}. Read each change's diff (`git show <change> -- <paths>`) "
             f"and find what it did in runs/{run}/ (log.jsonl, conversation.md, the screenshots).", '']
    for c in changes:
        lines.append(f"## {c['change']} - {c['subject']}")
        lines.append(f"- kind: {', '.join(c['kinds'])}; made by: {c['by']}"
                     + (f"; from request(s) {', '.join(c['requests'])}" if c['requests'] else '')
                     + (f"; from the review of run {c['reviewed']}" if c.get('reviewed') else ''))
        for rid in c['requests'][:3]:
            t = request_text(rid)
            if t:
                lines.append(f"  - {rid}: {t[:700]}{'...' if len(t) > 700 else ''}")
        if c['body']:
            lines.append(f"- what it says it does: {c['body'][:900]}")
        lines.append('- files: ' + ', '.join(f"{p} (+{a} -{d})" for p, _, a, d in c['files'][:12])
                     + (f" and {len(c['files']) - 12} more" if len(c['files']) > 12 else ''))
        lines.append(f"- the diff: `git show {c['change']} -- {' '.join(sorted({p for p, *_ in c['files']})[:12])}`")
        for r in c.get('watched') or []:
            lines.append(f"- watched since game {r.get('game')}: part \"{r.get('part')}\", {r.get('effect')}; watch for: "
                         f"{r.get('watch')} ({r.get('games')} more game(s) asked)")
        lines.append('')
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out, changes


def verdict_file(run):
    return AGENT / 'model' / 'reviews' / f'{run}-changes.json'


def gaps(run, changes):
    """What the Reviewer's verdict file leaves out: a listed change with no verdict, or a verdict without what it needs."""
    try:
        got = json.loads(verdict_file(run).read_text(encoding='utf-8')).get('changes') or []
    except (OSError, ValueError, AttributeError):
        return ['no verdict file agent/model/reviews/%s-changes.json' % run]
    out = []
    seen = {str(v.get('change'))[:7] for v in got if isinstance(v, dict)}
    for c in changes:
        if c['change'][:7] not in seen:
            out.append(f"{c['change']} ({c['subject'][:60]}) has no verdict")
    for v in got:
        if not isinstance(v, dict):
            continue
        h = v.get('change')
        if v.get('verdict') not in VERDICTS:
            out.append(f"{h}: verdict must be one of {VERDICTS}")
        if v.get('effect') not in EFFECTS:
            out.append(f"{h}: effect must be one of {EFFECTS}")
        if v.get('verdict') == 'remove' and not str(v.get('evidence') or '').strip():
            out.append(f"{h}: a removal needs its evidence")
        if v.get('verdict') == 'monitor' and not (int(v.get('games') or 0) > 0 and str(v.get('watch') or '').strip()):
            out.append(f"{h}: monitor needs how many more games and what to watch for")
    return out


def record(run, changes):
    """The Reviewer's verdicts into agent/changes.jsonl. -> the rows written."""
    try:
        got = json.loads(verdict_file(run).read_text(encoding='utf-8')).get('changes') or []
    except (OSError, ValueError, AttributeError):
        return []
    known = {c['change'][:7]: c for c in changes}
    rows, when = [], datetime.now().strftime('%Y-%m-%d %H:%M')
    for v in got:
        if not isinstance(v, dict) or v.get('verdict') not in VERDICTS:
            continue
        c = known.get(str(v.get('change'))[:7])
        if not c:
            continue
        rows.append({'change': c['change'], 'part': str(v.get('part') or '').strip(), 'subject': c['subject'],
                     'kinds': c['kinds'], 'kind': v.get('kind') if v.get('kind') in ROUTE else c['kinds'][0],
                     'by': c['by'], 'run': run, 'game': game_of(run), 'when': when,
                     'fired': v.get('fired'), 'where': v.get('where'), 'effect': v.get('effect'),
                     'verdict': v['verdict'], 'games': int(v.get('games') or 0), 'watch': v.get('watch') or '',
                     'evidence': v.get('evidence') or '', 'why': v.get('why') or ''})
    if rows:
        with open(LEDGER, 'a', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return rows


def reverted(hashes, commit, why):
    """Changes reverted outside a review (the owner's order): every part of each ends as "remove", so none is watched
    again, and the roles are told not to build it again. -> the rows written."""
    rows, when, last = [], datetime.now().strftime('%Y-%m-%d %H:%M'), latest()
    for h in hashes:
        subject = git('log', '-1', '--format=%s', h)
        kinds = sorted({k for k in (kind_of(l.split('	')[-1]) for l in git('show', '--numstat', '--format=', h)
                                    .splitlines() if l.count('	') == 2) if k}) or ['harness']
        parts = sorted({part for (c, part) in last if c == h} | {''})
        for part in parts:
            rows.append({'change': h, 'part': part, 'subject': subject, 'kinds': kinds, 'kind': kinds[0], 'by': 'owner',
                         'run': None, 'game': None, 'when': when, 'fired': None, 'where': None, 'effect': 'none',
                         'verdict': 'remove', 'games': 0, 'watch': '', 'evidence': f'reverted in {commit}: {why}',
                         'why': why})
    with open(LEDGER, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return rows


def removal_requests(rows, open_texts=()):
    """[(request kind, text)] for each "remove" verdict, unless an open request already asks for it."""
    out = []
    for r in rows:
        if r['verdict'] != 'remove':
            continue
        tag = f"Change review of game {r.get('game') or r['run']}: remove {r['change']}"
        if any(tag in t and (r['part'] or '')[:40] in t for t in open_texts):
            continue
        what = {'editor': "Revert exactly that part (`git show {h}`), keep what does not depend on it, and add a "
                          "regression test from runs/{run}/ that shows the reverted behaviour; the check must pass.",
                'coach': "Drop that point from the comp notes and do not write it again.",
                'strategist': "Propose the principle's removal (a remove proposal, for the Judge).",
                'structural': "A setting or a role brief: the owner's side decides it."}[ROUTE[r['kind']]]
        out.append((ROUTE[r['kind']], f"{tag} ({r['subject'][:90]}) - the part: {r['part'] or 'the whole change'}. "
                                      f"Found: {r['effect']}; {r['evidence'][:600]} "
                    + what.format(h=r['change'], run=r['run'])))
    return out


def summary(rows):
    k = {v: [r for r in rows if r['verdict'] == v] for v in VERDICTS}
    return {v: len(k[v]) for v in VERDICTS} | {'removed': [f"{r['change']} {r['part'][:50]}".strip() for r in k['remove']],
                                                'watched': [f"{r['change']} {r['part'][:50]}".strip() for r in k['monitor']]}


if __name__ == '__main__':
    if len(sys.argv) >= 2 and not sys.argv[1].startswith('-'):
        since = sys.argv[sys.argv.index('--since') + 1] if '--since' in sys.argv else None
        path, ch = packet(sys.argv[1], since)
        print(path, len(ch), 'change(s)')
    else:
        for (h, part), r in latest().items():
            print(f"{h} {r['verdict']:<8} game {r.get('game')}: {r['subject'][:60]} | {part[:60]}")
