"""Lecture de l'audio d'une video, partagee par la transcription (etape 4) et les reactions (etape 5)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SR = 16000  # frequence attendue par Whisper


def audio_tracks(src: Path) -> int:
    """Nombre de pistes audio de la video."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout
    return len([l for l in out.splitlines() if l.strip()])


def mic_track(src: Path) -> int:
    """Piste a analyser (voix seule). Avec plusieurs pistes OBS, le micro est sur la piste au plus petit
    numero (ex. 5) et le melange complet sur la derniere (ex. 6) : le micro est donc la 1re piste du fichier.
    Avec une seule piste (micro + jeu melanges), c'est elle."""
    return 0


def mix_track(src: Path) -> int:
    """Piste utilisee pour le son des clips : la derniere (melange complet)."""
    return max(0, audio_tracks(src) - 1)


def load_audio(src: Path, sr: int = SR, track: int | None = None) -> np.ndarray:
    """Audio mono en int16 (2 octets par echantillon : environ 580 Mo pour 5 h a 16 kHz).
    Par defaut, la piste du micro si l'enregistrement en a une a part."""
    track = mic_track(src) if track is None else track
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src), "-map", f"0:a:{track}", "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "s16le", "-"]
    p = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.int16)


def to_float(chunk: np.ndarray) -> np.ndarray:
    return chunk.astype(np.float32) / 32768.0


def rms_profile(audio: np.ndarray, sr: int = SR, hop: float = 0.1) -> np.ndarray:
    """Volume (RMS) par tranche de hop secondes."""
    n = int(sr * hop)
    usable = len(audio) // n * n
    frames = audio[:usable].astype(np.float32).reshape(-1, n) / 32768.0
    return np.sqrt((frames ** 2).mean(axis=1) + 1e-12)


def quiet_cuts(audio: np.ndarray, every: float = 600.0, search: float = 20.0, sr: int = SR) -> list[float]:
    """Points de coupe toutes les ~10 min, places dans le passage le plus calme a +/- 20 s,
    pour ne jamais couper un mot en deux entre deux morceaux transcrits."""
    rms = rms_profile(audio, sr)
    dur = len(audio) / sr
    cuts = [0.0]
    t = every
    while t < dur - search:
        a, b = int((t - search) / 0.1), int((t + search) / 0.1)
        # moyenne glissante sur 0,5 s pour viser un vrai silence, pas un creux isole
        win = np.convolve(np.pad(rms[a:b], 2, mode="edge"), np.ones(5) / 5, mode="valid")
        cuts.append(float((a + int(np.argmin(win))) * 0.1))
        t = cuts[-1] + every
    cuts.append(float(dur))
    return cuts
