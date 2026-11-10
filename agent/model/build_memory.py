"""Write the player's playbook (agent/memory/playbook.md, injected by tft_player/coach.py memory_block) from the game
model (agent/model/game-model.md): the mechanisms and, under each, the ACTIVE principles (candidates wait for the
replay test). The model file stays the one source; the evidence (agent/model/evidence.md) is not in the playbook.

    python agent/model/build_memory.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import modelfile  # noqa: E402

PLAYBOOK = HERE.parents[1] / 'agent' / 'memory' / 'playbook.md'


def main():
    text = modelfile.playbook()
    PLAYBOOK.write_text(text, encoding='utf-8')
    print(f'{PLAYBOOK}: {len(text.split())} words, {sum(p["status"] == "active" for p in modelfile.principles())} '
          f'active principles')


if __name__ == '__main__':
    main()
