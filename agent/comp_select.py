"""Which comp the climb plays (the owner, 2026-10-29: "it should only play games that learned from the pro": find the
comp with the best results now, or the one strong players like to play that MetaTFT rates S or A; find several strong
players' games of it; finalise its strategy; make the player follow it).

    python agent/comp_select.py            the candidates, the comp to play and the comp to learn

Candidates: MetaTFT's S and A comps (avg place, Platinum+, the last days; not situational ones), ranked by average
place - what LP follows - with how many strong players' videos of each we have found as the second key. A comp is
learned when the comp coach has written its notes from at least comp_coach.MIN_GAMES strong players' games. The comp
played is the best learned candidate; the best candidate not learned yet is learned first (its games found, read, its
notes written) - and when none is learned, no game is played until one is.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
sys.path.insert(0, str(AGENT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
CANDIDATES = 6                        # the best S/A comps considered
META_AGE_H = 24                       # MetaTFT data refreshed when older


def refresh():
    """MetaTFT's data when older than META_AGE_H (plain downloads). -> what happened."""
    try:
        import play_match
        return play_match.update_meta(META_AGE_H)
    except Exception as e:                              # no network: the data we have
        return f'MetaTFT refresh failed ({e.__class__.__name__})'


def videos_of(comp):
    """Strong players' videos found that name the comp (agent/sources/videos.jsonl titles)."""
    import comp_coach
    import study_queue
    keys = comp_coach.keywords(comp)
    return [v for v in comp_coach.jsonl(study_queue.VIDEOS) if any(k in str(v.get('title', '')).lower() for k in keys)]


def learned(comp):
    """The comp coach's notes exist and used at least comp_coach.MIN_GAMES strong players' games."""
    import comp_coach
    try:
        st = json.loads((comp_coach.COMPS / comp_coach.slug(comp) / 'state.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    notes = comp_coach.MEMORY_COMPS / f'{comp_coach.slug(comp)}.md'
    pro = [g for g in st.get('games_used') or [] if str(g).startswith('vod-')]
    return notes.exists() and len(pro) >= comp_coach.MIN_GAMES and (st.get('kept') or 0) > 0


def unlearnable(comp):
    """The comp coach searched for its games and found none to queue (comp_coach.find's record)."""
    import comp_coach
    try:
        st = json.loads((comp_coach.COMPS / comp_coach.slug(comp) / 'state.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return st.get('find_found') == 0 and not comp_coach.games_of(comp)


def candidates():
    from tft_player.meta import META
    comps = [c for c in META.comps.values() if c.tier in ('S', 'A') and not c.situational and c.avg]
    rank = sorted(comps, key=lambda c: (round(c.avg, 2), -len(videos_of(c.name))))[:CANDIDATES]
    return [{'comp': c.name, 'id': c.id, 'tier': c.tier, 'avg': c.avg, 'win': c.win, 'top4': c.top4,
             'games': c.games, 'pro_videos': len(videos_of(c.name)), 'learned': learned(c.name),
             'unlearnable': unlearnable(c.name)} for c in rank]


def good(comp):
    """The comp is still one MetaTFT rates S or A (and not situational)."""
    from tft_player.meta import META
    c = META.by_name(comp) if comp else None
    return bool(c and c.tier in ('S', 'A') and not c.situational)


def choose(current=None):
    """One comp at a time (the owner, 2026-10-29: "it does not need to learn everything - if the framework wants to play
    a certain comp for now, like Brambleback, it could only first learn pro games on this"): the comp being played
    stays the focus while MetaTFT rates it S or A and it can be learned; only then does the best candidate take over.
    The focus is played once learned, and learned first until then. -> {'focus', 'play', 'learn', 'candidates'}"""
    cands = [c for c in candidates() if good(c['comp'])]
    focus = current if current and good(current) and not unlearnable(current) else \
        next((c['comp'] for c in cands if c['learned']), None) or \
        next((c['comp'] for c in cands if not c['unlearnable']), None)
    ok = bool(focus and learned(focus))
    return {'focus': focus, 'play': focus if ok else None, 'learn': None if ok else focus, 'candidates': cands}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print(refresh())
    ch = choose(json.loads((AGENT / 'config.json').read_text(encoding='utf-8')).get('comp'))
    for c in ch['candidates']:
        print(f"{c['tier']} {c['comp']}: avg {c['avg']}, win {c['win']}, top4 {c['top4']}, {c['games']} games; "
              f"{c['pro_videos']} strong players' videos found; {'LEARNED' if c['learned'] else 'not learned'}"
              + (' (no games found)' if c['unlearnable'] else ''))
    print('play:', ch['play'], '| learn next:', ch['learn'])


if __name__ == '__main__':
    main()
