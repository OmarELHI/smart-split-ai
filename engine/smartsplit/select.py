"""Etape 6 : choix des clips a partir des notes du modele de langage et du score audio.

Un clip part d'un extrait bien note, s'etend aux extraits voisins bien notes, puis est
cale sur des debuts et fins de phrases pour durer entre 15 et 60 s.
"""
from __future__ import annotations

import math

from .config import LONG_MAX, SHORT_MAX

CLIP_MIN, CLIP_MAX = 15.0, SHORT_MAX   # le montage peut ensuite allonger une histoire jusqu'a LONG_MAX
PER_HOUR = 10          # minimum de clips par heure
NOTE_MIN = 7           # note minimale d'un extrait pour devenir un clip (completee jusqu'a PER_HOUR si besoin)
W_LLM, W_AUDIO = 0.8, 0.2


def _sentences(transcript: dict) -> list[tuple[float, float]]:
    from .transcribe import sentences
    return [(s["start"], s["end"]) for s in sentences(transcript)]


def _snap(start: float, end: float, sents: list[tuple[float, float]], duration: float) -> tuple[float, float]:
    """Etend [start, end] aux phrases voisines jusqu'a 15 s minimum, coupe a 60 s maximum."""
    inside = [i for i, (a, b) in enumerate(sents) if b > start and a < end]
    if not inside:
        return max(0, start), min(duration, max(end, start + CLIP_MIN))
    i0, i1 = inside[0], inside[-1]
    while sents[i1][1] - sents[i0][0] < CLIP_MIN:
        # ajoute la phrase la plus proche, avant ou apres
        gap_prev = sents[i0][0] - sents[i0 - 1][1] if i0 > 0 else math.inf
        gap_next = sents[i1 + 1][0] - sents[i1][1] if i1 + 1 < len(sents) else math.inf
        if gap_prev == gap_next == math.inf:
            break
        if gap_next <= gap_prev:
            i1 += 1
        else:
            i0 -= 1
    while sents[i1][1] - sents[i0][0] > CLIP_MAX and i1 > i0:
        i1 -= 1
    a = max(0.0, sents[i0][0] - 0.3)
    b = min(duration, sents[i1][1] + 0.6)
    if b - a < CLIP_MIN:                       # pas assez de phrases : on complete par du temps
        pad = (CLIP_MIN - (b - a)) / 2
        a, b = max(0.0, a - pad), min(duration, b + pad)
    return round(a, 2), round(b, 2)


def select(blocks: list[dict], transcript: dict, reactions: dict | None, duration: float) -> list[dict]:
    audio = reactions["score"] if reactions else None

    def audio_peak(a: float, b: float) -> float:
        if not audio:
            return 0.0
        seg = audio[int(a):int(b) + 1]
        return max(seg) if seg else 0.0

    for b in blocks:
        b["final"] = round(W_LLM * b["note"] / 10 + W_AUDIO * audio_peak(b["start"], b["end"]), 3)
    want = max(1, math.ceil(duration / 3600 * PER_HOUR))
    ranked = sorted(blocks, key=lambda b: -b["final"])
    sents = _sentences(transcript)
    clips: list[dict] = []
    for b in ranked:
        if b["note"] < NOTE_MIN and len(clips) >= want:
            break
        if b["note"] == 0:
            continue
        # fusionne avec les voisins directs bien notes
        idx = blocks.index(b)
        a_blk, z_blk = b, b
        def good(n):  # voisin direct (moins de 3 s d'ecart) et presque aussi bien note
            return n["note"] >= max(NOTE_MIN, b["note"] - 1)
        if idx > 0 and good(blocks[idx - 1]) and b["start"] - blocks[idx - 1]["end"] < 3:
            a_blk = blocks[idx - 1]
        if idx + 1 < len(blocks) and good(blocks[idx + 1]) and blocks[idx + 1]["start"] - b["end"] < 3:
            z_blk = blocks[idx + 1]
        start, end = _snap(a_blk["start"], z_blk["end"], sents, duration)
        if any(start < c["end"] and c["start"] < end for c in clips):   # pas de chevauchement
            continue
        clips.append({"start": start, "end": end, "note": b["note"], "type": b["type"],
                      "raison": b["raison"], "score": b["final"], "text": b["text"]})
    clips.sort(key=lambda c: -c["score"])
    return clips


def add_preroll(clips: list[dict], transcript: dict) -> list[dict]:
    """Ajoute avant le clip le passage sans paroles transcrites qui le precede (jusqu'a 8 s pour une reaction).

    Une reaction repond souvent a quelque chose que la transcription ne contient pas : une replique de
    personnage, un bruit, une action en jeu. Ce moment se trouve dans le "trou" juste avant ta premiere phrase.
    """
    from .transcribe import sentences
    sents = sentences(transcript)
    out = []
    for c in clips:
        prev = [s for s in sents if s["end"] <= c["start"] + 0.3]
        prev_end = prev[-1]["end"] if prev else 0.0
        gap = max(0.0, c["start"] - prev_end - 0.4)
        reaction = c.get("type") in ("reaction", "drole")
        add = min(gap, 8.0 if reaction else 1.5)
        cap = LONG_MAX if c["end"] - c["start"] > SHORT_MAX + 2 else SHORT_MAX + 2
        add = min(add, max(0.0, cap - (c["end"] - c["start"])))      # reste sous la duree maximale
        c2 = {**c, "start": round(max(0.0, c["start"] - add), 2)} if add > 0.2 else dict(c)
        if c.get("court"):                                             # meme regle pour la version courte
            k = c["court"]
            prev = [s for s in sents if s["end"] <= k["start"] + 0.3]
            gap = max(0.0, k["start"] - (prev[-1]["end"] if prev else 0.0) - 0.4)
            add = min(gap, 8.0 if reaction else 1.5, max(0.0, SHORT_MAX + 2 - (k["end"] - k["start"])))
            if add > 0.2:
                c2["court"] = {**k, "start": round(max(0.0, k["start"] - add), 2)}
        out.append(c2)
    return out
