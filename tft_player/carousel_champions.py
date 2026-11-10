"""Local carousel appearance reads. The gallery is separate from bench/board crops.

No champion is actionable without a complete, carousel-held-out evaluation of the
same gallery. Uncalibrated candidates remain diagnostic data, never player facts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

GALLERY = Path(__file__).with_name('carousel_gallery.npz')
EVALUATION = Path(__file__).with_name('carousel_gallery_evaluation.json')
_bank = None
FEATURE_SPEC = 'dinov2-small-cls/rgb-imagenet-112x126/carousel-112x98-marker+52:145-x56/v1'


def unit_crop(frame, marker):
    """RGB model below its item marker, excluding the eye and item icon; 1080p coords."""
    h, w = frame.shape[:2]
    s = h / 1080
    x, y = w / 2 + (marker[0] - 960) * s, marker[1] * s
    c = frame[max(0, round((y + 52 * s))):min(h, round(y + 145 * s)),
              max(0, round(x - 56 * s)):min(w, round(x + 56 * s))]
    return cv2.resize(c, (112, 98), interpolation=cv2.INTER_AREA) if c.size else None


def gallery():
    global _bank
    if _bank is None:
        _bank = ([], np.zeros((0, 384), np.float32), {})
        if GALLERY.exists():
            z = np.load(GALLERY, allow_pickle=False)
            names, vectors = list(z['names'].astype(str)), z['vectors']
            policy = {}
            if EVALUATION.exists():
                e = json.loads(EVALUATION.read_text(encoding='utf-8'))
                # A threshold found after removing wrong answers is not a measured
                # 100% result. Require the declared held-out counts and no leakage.
                disjoint = not set(e.get('gallery_carousels', [])) & set(e.get('heldout_carousels', []))
                same = e.get('gallery_sha256') == hashlib.sha256(GALLERY.read_bytes()).hexdigest()
                sources = set(z['carousels'].astype(str)) if 'carousels' in z else set()
                provenance = sources == set(e.get('gallery_carousels', []))
                if (e.get('labels_complete') and disjoint and same and e.get('heldout_carousels')
                        and provenance and e.get('evaluation_scope') == 'first_offer'
                        and e.get('feature_spec') == FEATURE_SPEC):
                    policy = {n: v for n, v in e.get('champions', {}).items()
                              if v.get('accepted', 0) > 0 and v.get('correct') == v.get('accepted')}
            _bank = names, vectors, policy
    return _bank


def classify(vectors, names, bank, policy):
    out = []
    for vec in vectors:
        best = {}
        for n, score in zip(names, bank @ vec):
            best[n] = max(best.get(n, -1.0), float(score))
        top = sorted(best.items(), key=lambda p: -p[1])
        if not top:
            out.append({'champion': None, 'candidate': None, 'score': 0., 'lead': 0.})
            continue
        name, score = top[0]
        lead = score - (top[1][1] if len(top) > 1 else 0.)
        gate = policy.get(name)
        trusted = bool(gate and score >= gate['cutoff'] and lead >= gate['lead'])
        out.append({'champion': name if trusted else None, 'candidate': name,
                    'score': score, 'lead': lead})
    return out


class Offer(list):
    def __init__(self, markers, reads):
        super().__init__(markers)
        self.champions = reads


def read_offer(frame, markers):
    names, bank, policy = gallery()
    if not names:
        return Offer(markers, [{'champion': None, 'candidate': None, 'score': 0., 'lead': 0.}
                               for _ in markers])
    from tft_player.champs import embed
    crops = [unit_crop(frame, m) for m in markers]
    valid = [i for i, c in enumerate(crops) if c is not None]
    reads = [dict(champion=None, candidate=None, score=0., lead=0.) for _ in markers]
    for i, result in zip(valid, classify(embed([crops[i] for i in valid]), names, bank, policy)):
        reads[i] = result
    return Offer(markers, reads)
