"""Resumes de missions : une video courte par mission jouee pendant le stream.

GTA 6 affiche le nom de la mission en rose, en bas a droite de l'ecran. La vision le recopie dans chaque
fenetre de 30 s (champ "mission"). On regroupe les fenetres d'une meme mission, puis on monte un resume :
le debut de la mission, ses meilleurs moments (tes reactions les mieux notees, les evenements a l'ecran),
et la fin, en 60 s environ.
"""
from __future__ import annotations

import re
import unicodedata

from .config import MISSION_MIN, RECAP_MAX

PART = 7.0      # duree d'un moment dans le resume (s)


def _norm(name: str) -> str:
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", "", n).strip()


def spans(wins: list[dict]) -> list[dict]:
    """Missions jouees : [{nom, start, end}], d'apres le nom affiche a l'ecran."""
    out: list[dict] = []
    for w in sorted((w for w in wins if not w.get("focus")), key=lambda w: w["start"]):
        name = (w.get("mission") or "").strip()
        key = _norm(name)
        if len(key) < 3:
            continue
        if out and out[-1]["key"] == key and w["start"] - out[-1]["end"] <= 120:
            out[-1]["end"] = w["end"]
            out[-1]["vu"] += 1
        else:
            out.append({"key": key, "nom": name, "start": w["start"], "end": w["end"], "vu": 1})
    # une mission doit durer un minimum et avoir ete lue sur plusieurs fenetres (evite les erreurs de lecture)
    return [s for s in out if s["end"] - s["start"] >= MISSION_MIN and s["vu"] >= 3]


def recap(span: dict, blocks: list[dict], events: list[dict], duration: float) -> dict | None:
    a, b = span["start"], span["end"]
    cands = [{"t": blk["start"], "fin": blk["end"], "score": blk["note"] + 0.0}
             for blk in blocks if a <= blk["start"] < b and blk.get("note", 0) >= 6]
    cands += [{"t": e["end"] - 12, "fin": e["end"], "score": e["note"] + 1.0} for e in events if a <= e["end"] <= b + 10]
    if len(cands) < 2:
        return None
    n = max(2, int((RECAP_MAX - 10) // PART))
    best = sorted(cands, key=lambda c: -c["score"])
    picked: list[dict] = []
    for c in best:
        if len(picked) >= n:
            break
        if all(abs(c["t"] - p["t"]) > PART + 2 for p in picked):
            picked.append(c)
    parts = [{"start": round(a, 2), "end": round(min(b, a + 4.0), 2), "zoom": False}]          # le debut
    for c in sorted(picked, key=lambda c: c["t"]):
        s = max(a, c["t"])
        parts.append({"start": round(s, 2), "end": round(min(b, s + PART, c["fin"] + 1.5), 2),
                      "zoom": c["score"] >= 9})
    parts.append({"start": round(max(a, b - 5.0), 2), "end": round(min(duration, b), 2), "zoom": False})  # la fin
    parts = [p for p in parts if p["end"] - p["start"] >= 1.5]
    merged: list[dict] = []
    for p in sorted(parts, key=lambda p: p["start"]):
        if merged and p["start"] <= merged[-1]["end"] + 0.5:
            merged[-1]["end"] = max(merged[-1]["end"], p["end"])
        else:
            merged.append(dict(p))
    return {"start": merged[0]["start"], "end": merged[-1]["end"], "parts": merged, "court_parts": [],
            "note": 9, "type": "resume", "monte": True, "score": 2.0,
            "duree": round(sum(p["end"] - p["start"] for p in merged), 1),
            "titre": f"Mission {span['nom']} : le résumé", "raison": f"Résumé de la mission « {span['nom']} »",
            "text": ""}


def recaps(wins: list[dict], blocks: list[dict], events: list[dict], duration: float) -> list[dict]:
    out = []
    for s in spans(wins):
        r = recap(s, blocks, events, duration)
        if r:
            out.append(r)
    return out
