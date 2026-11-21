"""Etape 7 bis : montage d'un clip.

Le choix des moments garde un "rush" plus long que le clip final (RUSH_BEFORE avant, RUSH_AFTER apres).
L'IA lit les phrases numerotees du rush et ce qui se passe a l'ecran, puis renvoie :
- les passages a garder, dans l'ordre (elle enleve les temps morts, les digressions, la logistique) ;
- ceux ou Omar reagit fort, qui recevront un zoom avant ;
- un titre d'accroche, affiche en bandeau pendant tout le clip ;
- une version courte si le montage depasse SHORT_MAX.
Ensuite, les silences de plus de CUT_SILENCE s sans action a l'ecran sont coupes a l'interieur des passages.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np

from .config import CUT_SILENCE, LONG_MAX, RUSH_AFTER, RUSH_BEFORE, SHORT_MAX
from .llm import LLM_MODEL, OLLAMA_URL, _context_suffix

MIN_PART = 1.5       # passage minimal garde (s)

PROMPT = """Tu es monteur video TikTok pour Omar, streamer francais qui joue a GTA 6. Tu recois le "rush" d'un
moment repere dans son stream : ses phrases numerotees avec leur instant, et ce qui se passe a l'ecran.
Monte un clip TikTok qui retient l'attention du debut a la fin, comme les clips GTA qui font des vues :
- garde le declencheur (ce qu'on voit ou ce qui est dit juste avant la reaction) et la reaction en entier ;
- enleve les temps morts, la logistique ("attends", "deux secondes"), les digressions, les repetitions ;
- les passages gardes restent dans l'ordre chronologique ; 2 a 8 passages, chacun d'au moins 2 phrases
  ou d'une action complete ;
- un avis ou une reaction seule : 20 a {short} s au total ; une histoire en jeu (contexte, action,
  rebondissements, reactions) : jusqu'a {long} s ;
- zoom = true pour les passages ou Omar reagit fort (surprise, cri, rire, choc) ;
- titre : 3 a 8 mots en francais, qui raconte la situation ou la reaction comme un spectateur la dirait
  (ex. "Alors la je m'attendais pas a ca", "Le yeti m'a fait la pire peur de ma vie", "POV: GTA 6 te
  fait pleurer"). Pas de hashtag, pas de nom de streamer, pas de guillemets ;
- si le total depasse {short} s, donne aussi "court" : les passages d'une version de {short} s maximum
  centree sur le moment le plus fort. Sinon, "court" est une liste vide.
Les reperes "(action a l'ecran)" permettent de couper hors des phrases.
Reponds uniquement en JSON."""

PART = {"type": "object", "properties": {"debut": {"type": "integer"}, "fin": {"type": "integer"},
                                         "zoom": {"type": "boolean"}},
        "required": ["debut", "fin", "zoom"]}
SCHEMA = {"type": "object",
          "properties": {"titre": {"type": "string"}, "passages": {"type": "array", "items": PART},
                         "court": {"type": "array", "items": PART}, "raison": {"type": "string"}},
          "required": ["titre", "passages", "court", "raison"]}


def rush(clip: dict, duration: float) -> tuple[float, float]:
    return max(0.0, clip["start"] - RUSH_BEFORE), min(duration, clip["end"] + RUSH_AFTER)


def _lines(sents: list[dict], a: float, b: float, with_marks: bool) -> list[dict]:
    near = [s for s in sents if s["end"] > a and s["start"] < b]
    if with_marks or len(near) < 4:
        marks = [{"start": float(t), "end": float(t) + 0.5, "text": "(action a l'ecran)"} for t in np.arange(a, b, 5.0)
                 if not any(s["start"] - 1 <= t <= s["end"] + 1 for s in near)]
        near = sorted(near + marks, key=lambda s: s["start"])
    return near


def _parts(lines: list[dict], items: list[dict], a: float, b: float) -> list[dict]:
    out = []
    for it in items:
        i, j = int(it.get("debut", -1)), int(it.get("fin", -1))
        if not (0 <= i <= j < len(lines)):
            continue
        s = max(a, lines[i]["start"] - 0.25)
        e = min(b, lines[j]["end"] + 0.45)
        if e - s >= MIN_PART:
            out.append({"start": round(s, 2), "end": round(e, 2), "zoom": bool(it.get("zoom"))})
    out.sort(key=lambda p: p["start"])
    merged: list[dict] = []
    for p in out:                                  # chevauchements et micro-trous : on fusionne
        if merged and p["start"] <= merged[-1]["end"] + 0.6:
            merged[-1]["end"] = max(merged[-1]["end"], p["end"])
            merged[-1]["zoom"] = merged[-1]["zoom"] and p["zoom"]
        else:
            merged.append(dict(p))
    return merged


def total(parts: list[dict]) -> float:
    return sum(p["end"] - p["start"] for p in parts)


def tighten(parts: list[dict], words: list[dict], vision: list[dict] | None) -> list[dict]:
    """Coupe, a l'interieur des passages, les silences de plus de CUT_SILENCE s sans action a l'ecran."""
    def calm(a, b):
        return not vision or all(w["intensite"] < 6 for w in vision if w["end"] > a and w["start"] < b)

    out = []
    for p in parts:
        ws = [w for w in words if w["s"] >= p["start"] - 0.1 and w["e"] <= p["end"] + 0.1]
        cur = p["start"]
        for w1, w2 in zip(ws, ws[1:]):
            gap_a, gap_b = w1["e"] + 0.3, w2["s"] - 0.3
            if gap_b - gap_a >= CUT_SILENCE - 0.6 and calm(gap_a, gap_b):
                if gap_a - cur >= MIN_PART:
                    out.append({**p, "start": round(cur, 2), "end": round(gap_a, 2)})
                cur = gap_b
        if p["end"] - cur >= MIN_PART:
            out.append({**p, "start": round(cur, 2), "end": p["end"]})
    return out


def plan(clip: dict, sents: list[dict], words: list[dict], duration: float,
         vision: list[dict] | None = None, timeout: float = 180) -> dict:
    """Plan de montage d'un clip : {"titre", "parts", "court", "montage"} ajoutes au clip."""
    a, b = rush(clip, duration)
    lines = _lines(sents, a, b, bool(clip.get("vision")))
    if len(lines) < 2:
        return {**clip, "parts": [{"start": clip["start"], "end": clip["end"], "zoom": False}], "titre": ""}
    text = "\n".join(f"[{i}] {s['start']:.0f}s : {s['text']}" for i, s in enumerate(lines))
    user = (f"Moment repere entre {clip['start']:.0f}s et {clip['end']:.0f}s (raison : {clip['raison']}).\n\n"
            f"Phrases du rush :\n{text}")
    if vision:
        scr = [f"{w['start']:.0f}s-{w['end']:.0f}s : {w['description']} ({w['action']}, intensite {w['intensite']})"
               for w in vision if w["end"] > a and w["start"] < b]
        if scr:
            user += "\n\nA l'ecran :\n" + "\n".join(scr)
    system = PROMPT.format(short=int(SHORT_MAX), long=int(LONG_MAX)) + _context_suffix()
    body = json.dumps({"model": LLM_MODEL, "stream": False, "format": SCHEMA, "keep_alive": "5m",
                       "options": {"temperature": 0, "num_ctx": 8192},
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res = json.loads(json.loads(r.read())["message"]["content"])
    except Exception:  # noqa: BLE001
        res = {"titre": "", "passages": [], "court": [], "raison": ""}

    parts = tighten(_parts(lines, res.get("passages", []), a, b), words, vision)
    if not 12 <= total(parts) <= LONG_MAX + 10:        # plan inutilisable : moment repere tel quel
        parts = tighten([{"start": clip["start"], "end": clip["end"], "zoom": False}], words, vision)
    court = tighten(_parts(lines, res.get("court", []), a, b), words, vision)
    if total(parts) <= SHORT_MAX + 2 or not 12 <= total(court) <= SHORT_MAX + 2:
        court = []
    return {**clip, "start": parts[0]["start"], "end": parts[-1]["end"], "parts": parts, "court_parts": court,
            "duree": round(total(parts), 1), "titre": str(res.get("titre", "")).strip().strip('"'),
            "montage": res.get("raison", "")}


def plan_all(clips: list[dict], transcript: dict, duration: float, vision=None, progress=print) -> list[dict]:
    from .transcribe import sentences
    sents = sentences(transcript)
    words = [w for s in transcript["segments"] for w in s["words"]]
    out = []
    for i, c in enumerate(clips, 1):
        out.append(plan(c, sents, words, duration, vision))
        o = out[-1]
        progress(f"  {i}/{len(clips)} {o['duree']:.0f} s en {len(o['parts'])} passages : {o['titre']}")
    return out


def render(src, parts: list[dict], analysis, out, title: str = "", zoom: float | None = None):
    """Rend chaque passage (mise en page par scene, titre, zoom), puis les colle bout a bout."""
    import subprocess
    from pathlib import Path

    from .config import TITLE, ZOOM
    from .layout import render_segments
    from .scenes import segments_for

    out = Path(out)
    zoom = ZOOM if zoom is None else zoom
    tmp_dir = out.parent / f".{out.stem}_morceaux"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for k, p in enumerate(parts):
        f = tmp_dir / f"m{k:02d}.mp4"
        render_segments(src, p["start"], p["end"], segments_for(analysis, p["start"], p["end"]), f,
                        face_x="auto", title=title if TITLE else None, zoom=zoom if p.get("zoom") else 1.0)
        names.append(f.name)
    if len(names) == 1:
        (tmp_dir / names[0]).replace(out)
    else:
        (tmp_dir / "liste.txt").write_text("".join(f"file '{n}'\n" for n in names), encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", "liste.txt", "-c", "copy", "-movflags", "+faststart", str(out.resolve())],
                       cwd=tmp_dir, check=True)
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()
    return out
