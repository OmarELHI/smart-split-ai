"""Smart Split AI - mesure de la vitesse de transcription sur le GPU.

Transcrit les 3 premieres minutes du plus petit .mp4 du dossier projet,
deux fois de suite par modele : le 1er passage inclut le chargement et
la compilation des noyaux CUDA, le 2e donne la vraie vitesse.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
FFMPEG_DIR = Path(r"D:\Tools\ffmpeg\bin")
if not shutil.which("ffmpeg") and (FFMPEG_DIR / "ffmpeg.exe").exists():
    os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ["PATH"]

import numpy as np  # noqa: E402
import torch  # noqa: E402,F401  (charge les DLL CUDA/cuDNN avant ctranslate2)
from faster_whisper import WhisperModel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DURATION = 180  # secondes d'audio testees
MODELS = sys.argv[1:] or ["large-v3-turbo", "medium"]

vids = sorted(ROOT.glob("*.mp4"), key=lambda p: p.stat().st_size)
if not vids:
    sys.exit("Aucun .mp4 dans le dossier projet.")
src = vids[0]

p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src), "-t", str(DURATION),
                    "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True)
audio = np.frombuffer(p.stdout, dtype=np.float32)
secs = len(audio) / 16000
print(f"Echantillon : {src.name}, {secs:.0f} s d'audio\n")

for name in MODELS:
    t0 = time.time()
    model = WhisperModel(name, device="cuda", compute_type="float16")
    load = time.time() - t0
    runs = []
    text = ""
    for _ in range(2):
        t0 = time.time()
        segments, _ = model.transcribe(audio, language="fr", vad_filter=True, beam_size=5)
        text = " ".join(s.text.strip() for s in segments)
        runs.append(time.time() - t0)
    speed = secs / runs[1]
    five_h = 5 * 3600 / speed / 60
    print(f"== {name} ==")
    print(f"  chargement : {load:.1f} s | 1er passage : {runs[0]:.1f} s | 2e passage : {runs[1]:.1f} s")
    print(f"  vitesse : {speed:.1f}x temps reel -> environ {five_h:.0f} min pour 5 h de stream")
    print(f"  extrait : \"{text[:150]}...\"\n")
    del model
    torch.cuda.empty_cache()
