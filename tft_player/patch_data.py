"""Objective patch-note corrections for fields the local game snapshot has not caught up with."""
import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'


@lru_cache(maxsize=1)
def current_overrides():
    patch = json.loads((DATA / 'current-patch.json').read_text(encoding='utf-8'))
    sources = sorted((DATA / 'sources').glob('cdragon-tft-*.json'))
    # A new snapshot requires another audit; do not silently apply these corrections to it.
    return patch if sources and sources[-1].name == patch['snapshot'] else {}


def with_note(kind, name, text):
    patch = current_overrides()
    note = patch.get(kind, {}).get(name, '')
    return text + (f" Patch {patch['patch']}: {note}" if note else '')
