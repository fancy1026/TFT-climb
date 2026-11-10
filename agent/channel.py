"""The team channel (agent/FRAMEWORK.md): each agent's short posts - what it did, what it found, what it hands to whom -
the view the owner reads first in the control room (agent/console.py). Full sessions stay in agent/reviews/.

    python agent/channel.py "<agent>" "<text>" [--to "<agent>"] [--ref <file or commit>]...

Agents: Coordinator, Reviewer, Editor (the harness fixer), Strategist, Judge, Director, Player, Owner.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

CHANNEL = Path(__file__).resolve().parent / 'channel.jsonl'
AGENTS = {'harness_fixer': 'Editor', 'reviewer': 'Reviewer', 'strategist': 'Strategist', 'director': 'Director',
          # retired roles, for old posts
          'analyst': 'Analyst', 'proposer': 'Researcher', 'data': 'Data', 'curator': 'Curator', 'reasoner': 'Reasoner',
          'checker': 'Checker'}


def post(frm, text, to=None, refs=(), when=None, reconstructed=False):
    """Append one post. `frm` is a display name or a role key (analyst, proposer, harness_fixer)."""
    text = ' '.join(str(text or '').split())
    if not text:
        return None
    row = {'when': when or datetime.now().strftime('%Y-%m-%d %H:%M'), 'from': AGENTS.get(frm, frm), 'text': text}
    if to:
        row['to'] = AGENTS.get(to, to)
    if refs:
        row['refs'] = [str(r) for r in refs if r]
    if reconstructed:
        row['reconstructed'] = True     # written later from the agent's report, not by the agent at the time
    with open(CHANNEL, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return row


def read():
    if not CHANNEL.exists():
        return []
    return [json.loads(l) for l in CHANNEL.read_text(encoding='utf-8').splitlines() if l.strip()]


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('frm')
    ap.add_argument('text')
    ap.add_argument('--to')
    ap.add_argument('--ref', action='append', default=[])
    a = ap.parse_args()
    print(post(a.frm, a.text, a.to, a.ref))


if __name__ == '__main__':
    main()
