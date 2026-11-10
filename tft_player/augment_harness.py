"""Objective controller capability ratings, separate from comp bans and augment strength."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

TABLE = Path(__file__).resolve().parents[1] / 'data/reference/augment-harness.json'


@lru_cache(maxsize=1)
def ratings():
    """Load once per player process; a missing table leaves every card explicitly unrated."""
    try:
        return json.loads(TABLE.read_text(encoding='utf-8'))['augments']
    except (OSError, ValueError, KeyError):
        return {}


def card_mark(name):
    entry = ratings().get(name)
    if not entry or entry.get('plays') not in ('well', 'poorly'):
        return ' [controller capability: unrated]'
    if entry['plays'] == 'poorly':
        return (' [the controller plays this poorly: ' + entry['reason'].rstrip('.')
                + '; lower priority than a card it plays well]')
    return ''
