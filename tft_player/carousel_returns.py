"""Provenance for return labels only; never changes ownership or merge execution."""


def baseline(roster, now):
    return {'started_at': now, 'merge_index': len(roster.merge_history),
            'before_copies': [(u.id, u.name, u.star) for u in roster.units
                              if u.name and not u.missing and not u.unverified and
                              (getattr(u, 'sure', False) or
                               (getattr(u, 'panel_read', {}).get('name') == u.name and
                                u.panel_read.get('stars') == u.star))]}


def purchase_merges(roster, confirmation, stage):
    """Merges fully explained by pre-carousel copies and subsequent own buys.

    An assumed merge alone is insufficient: its surviving unit must have its new
    name AND star level read from its own panel this stage. Unknown/corrected
    baseline copies and external arrivals cannot explain a purchase-only merge.
    """
    if 'merge_index' not in confirmation or 'started_at' not in confirmation:
        return set()
    accounted = set(map(tuple, confirmation.get('before_copies', [])))
    explained = set()
    for event in roster.merge_history[confirmation['merge_index']:]:
        inputs = event['inputs']
        keys = [(u['id'], u['name'], u['star']) for u in inputs]
        if len(set(keys)) != 3:
            continue
        known = all(not u['missing'] and not u['unverified'] and
                    (key in accounted or (u['ours'] and u['bought_at'] is not None
                                          and u['bought_at'] >= confirmation['started_at']))
                    for u, key in zip(inputs, keys))
        if known:
            accounted.difference_update(keys)
            key = event['unit'], event['name'], event['star']
            accounted.add(key)
            explained.add(key)
    return {u.id for u in roster.units
            if (u.id, u.name, u.star) in explained and
            getattr(u, 'panel_read', {}) == {'stage': stage, 'name': u.name, 'stars': u.star}}


def unconfirmed_merges(roster, confirmation):
    """A doubtful third copy may have merged in game even when ownership abstains.

    It cannot prove ownership, nor make a separate arrival a proven carousel return.
    Keep that label ambiguity until the units' panels settle the actual inventory.
    """
    ambiguous = set()
    for uid, before_star in confirmation.get('before_stars', {}).items():
        u = roster.resolve(uid)
        if u is None or not u.name or u.missing or u.star != before_star:
            continue
        group = roster.copies(u.name, u.star)
        if len(group) >= 3 and any(x.unverified for x in group):
            survivor = min(group, key=lambda x: (not x.on_board, x.loc, x.id))
            ambiguous.add(survivor.id)
    return ambiguous
