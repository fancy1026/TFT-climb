"""Start the climb from nothing instead of the learned strategy this repository ships with.

    python tools/start_empty.py

Takes out the expert's rules, the memory and each comp's stage strategies, code rules, notes and reach (agent/): the
first climb then learns the comp itself (MetaTFT, strong players' games). They come back with:
    git checkout -- agent
"""
import shutil
from pathlib import Path

AGENT = Path(__file__).resolve().parent.parent / 'agent'


def main():
    gone = 0
    for d in (AGENT / 'expert' / 'rules', AGENT / 'memory'):
        for p in [p for p in d.rglob('*') if p.is_file() and p.name != '.gitkeep']:
            p.unlink()
            gone += 1
    comps = AGENT / 'comps'
    for comp in [d for d in comps.iterdir() if d.is_dir()] if comps.exists() else []:
        shutil.rmtree(comp)
        gone += 1
    print(f'{gone} files and comp folders taken out: the climb starts from nothing (back: git checkout -- agent)')


if __name__ == '__main__':
    main()
