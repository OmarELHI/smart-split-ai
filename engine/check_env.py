"""Smart Split AI - Etape 0 : verifie que l'environnement est pret.

Controle : Python, PyTorch + GPU, FFmpeg (NVENC / decodage CUDA),
faster-whisper sur GPU. Utilise le plus petit .mp4 du dossier projet
comme echantillon s'il y en a un.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# FFmpeg installe par setup.ps1 : utilise meme si le PATH du terminal n'est pas a jour
FFMPEG_DIR = Path(r"D:\Tools\ffmpeg\bin")
if not shutil.which("ffmpeg") and (FFMPEG_DIR / "ffmpeg.exe").exists():
    os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ["PATH"]

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "check"
OUT.mkdir(parents=True, exist_ok=True)

results = []


def check(name, fn):
    print(f"\n[..] {name}")
    try:
        detail = fn()
        results.append((name, True, detail))
        print(f"[OK] {name} : {detail}")
    except Exception as e:  # noqa: BLE001
        results.append((name, False, str(e)))
        print(f"[ECHEC] {name} : {e}")


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip().splitlines()[-1] if p.stderr.strip() else f"code {p.returncode}")
    return p.stdout + p.stderr


def sample_video():
    vids = sorted(ROOT.glob("*.mp4"), key=lambda p: p.stat().st_size)
    return vids[0] if vids else None


# 1. Python
def py():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} trop ancien")
    return f"{v.major}.{v.minor}.{v.micro} ({sys.executable})"


# 2. PyTorch + GPU (importe en premier : ses DLL CUDA/cuDNN servent aussi a faster-whisper)
def gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA indisponible pour torch {torch.__version__}")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    a = torch.randn(2048, 2048, device="cuda")
    (a @ a).sum().item()
    return f"torch {torch.__version__}, {name}, sm_{cap[0]}{cap[1]}, {vram:.1f} Go"


# 3. FFmpeg
def ffmpeg_caps():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg introuvable dans le PATH")
    enc = run(["ffmpeg", "-hide_banner", "-encoders"])
    hw = run(["ffmpeg", "-hide_banner", "-hwaccels"])
    missing = [x for x in ("h264_nvenc", "hevc_nvenc") if x not in enc]
    if missing:
        raise RuntimeError(f"encodeurs absents : {missing}")
    if "cuda" not in hw:
        raise RuntimeError("acceleration materielle CUDA absente")
    return "h264_nvenc, hevc_nvenc et decodage CUDA disponibles"


# 4. Encodage NVENC sur une mire
def nvenc_test():
    out = OUT / "nvenc_mire.mp4"
    run(["ffmpeg", "-y", "-hide_banner", "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=60",
         "-t", "5", "-c:v", "h264_nvenc", str(out)])
    return f"{out.name} ({out.stat().st_size // 1024} Ko)"


# 5. Decodage CUDA + encodage NVENC sur l'echantillon reel
def real_video_test():
    src = sample_video()
    if not src:
        return "aucun .mp4 dans le dossier, test saute"
    out = OUT / "echantillon_30s.mp4"
    t0 = time.time()
    run(["ffmpeg", "-y", "-hide_banner", "-hwaccel", "cuda", "-i", str(src), "-t", "30",
         "-c:v", "h264_nvenc", "-c:a", "aac", str(out)])
    dt = time.time() - t0
    return f"30 s de {src.name} traitees en {dt:.1f} s ({30 / dt:.1f}x temps reel)"


# 6. faster-whisper sur GPU
def whisper_test():
    import torch  # noqa: F401  (charge les DLL CUDA/cuDNN avant ctranslate2)
    from faster_whisper import WhisperModel

    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    src = sample_video()
    if not src:
        return "modele tiny charge sur GPU (pas d'echantillon a transcrire)"
    # Audio decode par FFmpeg (pas par PyAV) : 16 kHz mono, float32
    import numpy as np

    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src), "-t", "30",
                        "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("extraction audio impossible")
    audio = np.frombuffer(p.stdout, dtype=np.float32)
    t0 = time.time()
    segments, _ = model.transcribe(audio, language="fr")
    text = " ".join(s.text.strip() for s in segments)
    dt = time.time() - t0
    return f"30 s transcrites en {dt:.1f} s : \"{text[:80]}...\""


check("Python", py)
check("PyTorch et GPU", gpu)
check("FFmpeg", ffmpeg_caps)
check("Encodage NVENC", nvenc_test)
check("Video reelle (decodage CUDA + NVENC)", real_video_test)
check("faster-whisper sur GPU", whisper_test)

print("\n========== RESUME ==========")
for name, ok, detail in results:
    print(f"{'OK    ' if ok else 'ECHEC '} {name} : {detail}")
ok_all = all(ok for _, ok, _ in results)
print("\nEnvironnement pret." if ok_all else "\nCertains tests ont echoue : copie ce resume a Claude.")
sys.exit(0 if ok_all else 1)
