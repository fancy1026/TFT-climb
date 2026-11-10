"""Case memory (agent/FRAMEWORK.md): strong players' choices in states like this one, from the TRAIN cases only, as a few
lines the player reads with its question. The judge measures it as a knowledge version ("<version>+cases") before it
may reach live play.

    python agent/retrieval.py 4-1 7 45        the lines a state at stage 4-1, level 7, 45 gold would get
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEC = ROOT / 'agent' / 'decisions'


def round_index(stage):
    a, b = map(int, stage.split('-'))
    return (a - 1) * 7 + b


def train_cases():
    splits = json.loads((DEC / 'splits.json').read_text(encoding='utf-8'))
    train = set(splits.get('train', []))
    rows = [json.loads(l) for l in (DEC / 'cases.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    return [c for c in rows if c['game'] in train]


def distance(c, stage, level, gold):
    st = c['state']
    return (3 * abs(round_index(c['stage']) - round_index(stage)) + 2 * abs((st.get('level') or level) - level)
            + abs((st.get('gold') or gold) - gold) / 10)


def similar(stage, level, gold, k=3, kind='tempo'):
    pool = [c for c in train_cases() if c['kind'] == kind and c['state'].get('level') is not None]
    return sorted(pool, key=lambda c: distance(c, stage, level, gold))[:k]


def line(c):
    st, ex = c['state'], c['expert']
    who = c.get('player') or 'a strong player'
    head = (f"{who}, stage {c['stage']}, level {st.get('level')}, {st.get('gold')} gold"
            + (f", {st['hp']} HP" if st.get('hp') is not None else '') + (f" ({st['line']})" if st.get('line') else ''))
    did = f"ended the phase at level {ex['level_end']} with {ex['gold_end']} gold"
    return f'- {head}: {did}' + (f' - "{ex["reason"]}"' if ex.get('reason') else '')


def block(stage, level, gold, k=3):
    cs = similar(stage, level, gold, k)
    if not cs:
        return ''
    return ('EXPERT CASES - what strong players did in the most similar states on record (their games, their comp '
            'lines; yours may differ):\n' + '\n'.join(line(c) for c in cs))


def fingerprint():
    """What the block depends on: the train cases (a new train case changes every '+cases' answer)."""
    import hashlib
    return hashlib.sha1(json.dumps(sorted(c['id'] for c in train_cases())).encode()).hexdigest()[:8]


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print(block(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
