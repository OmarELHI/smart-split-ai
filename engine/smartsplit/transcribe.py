"""Etape 4 : transcription en francais, horodatee au mot, avec faster-whisper sur le GPU.

L'audio est decoupe en morceaux d'environ 10 min, coupes dans des silences, pour garder
une memoire stable sur 5 h de stream et afficher la progression. Le resultat est mis en
cache dans data\\analysis\\<video>\\transcript.json : une video n'est transcrite qu'une fois.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .audio import SR, load_audio, quiet_cuts, to_float
from .config import DATA

MODEL_NAME = "large-v3-turbo"
LANGUAGE = "fr"


def transcript_path(src: Path) -> Path:
    return DATA / "analysis" / Path(src).stem / "transcript.json"


def load_model():
    import torch  # noqa: F401  (charge les DLL CUDA/cuDNN avant ctranslate2)
    from faster_whisper import WhisperModel

    try:  # deja telecharge : on ne contacte pas Hugging Face
        return WhisperModel(MODEL_NAME, device="cuda", compute_type="float16", local_files_only=True)
    except Exception:  # noqa: BLE001
        return WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")


def transcribe(src: Path, model=None, progress=print, audio=None) -> dict:
    src = Path(src)
    t_start = time.time()
    audio = load_audio(src) if audio is None else audio
    dur = len(audio) / SR
    progress(f"  audio lu : {dur / 60:.1f} min ({time.time() - t_start:.0f} s)")
    model = model or load_model()
    cuts = quiet_cuts(audio)
    segments = []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        chunk = to_float(audio[int(a * SR):int(b * SR)])
        segs, _ = model.transcribe(chunk, language=LANGUAGE, vad_filter=True, beam_size=5,
                                   word_timestamps=True, condition_on_previous_text=False)
        for s in segs:
            words = [{"w": w.word.strip(), "s": round(a + w.start, 2), "e": round(a + w.end, 2),
                      "p": round(w.probability, 2)} for w in (s.words or []) if w.word.strip()]
            segments.append({"start": round(a + s.start, 2), "end": round(a + s.end, 2),
                             "text": s.text.strip(), "words": words})
        progress(f"  transcription : {b / 60:.1f} / {dur / 60:.1f} min ({time.time() - t_start:.0f} s)")
    return {"src": str(src), "model": MODEL_NAME, "language": LANGUAGE, "duration": dur,
            "seconds": round(time.time() - t_start, 1), "segments": segments}


def save(tr: dict) -> Path:
    out = transcript_path(Path(tr["src"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def load(src: Path) -> dict | None:
    f = transcript_path(src)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def get(src: Path, progress=print) -> dict:
    """Transcription en cache, ou calculee puis enregistree."""
    tr = load(src)
    if tr is None:
        tr = transcribe(src, progress=progress)
        save(tr)
    return tr
