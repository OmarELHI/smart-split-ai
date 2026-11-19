"""Etape 5 : score de reaction seconde par seconde, a partir du son.

Signaux (sans modele d'IA, donc rapides) :
- volume : de combien la seconde est plus forte que le niveau habituel autour (60 s) ;
- eclat : part d'energie dans les aigus (2-5 kHz), elevee dans les cris et les rires ;
- debit : nombre de mots par seconde (transcription), qui s'emballe quand tu reagis.
Le score est attenue quand tu ne parles pas, pour que les explosions ou la musique du
jeu ne passent pas pour des reactions.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .audio import SR
from .config import DATA

HOP = 0.5                 # pas d'analyse en secondes
FRAME, STEP = 1024, 800   # fenetre FFT (64 ms) et pas (50 ms) a 16 kHz
BASE_WIN = 60.0           # fenetre du niveau habituel
PEAK_GAP = 20.0           # deux pics retenus sont espaces d'au moins 20 s
PEAK_MIN = 0.45           # score minimal d'un pic


def reactions_path(src: Path) -> Path:
    return DATA / "analysis" / Path(src).stem / "reactions.json"


def _frame_features(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Volume (dB) et eclat (part 2-5 kHz) toutes les 50 ms, calcules par blocs de 60 s."""
    freqs = np.fft.rfftfreq(FRAME, 1 / SR)
    band = (freqs >= 80) & (freqs <= 8000)
    bright = (freqs >= 2000) & (freqs <= 5000)
    win = np.hanning(FRAME).astype(np.float32)
    db, br = [], []
    block = SR * 60
    for a in range(0, len(audio), block):
        x = audio[a:a + block + FRAME].astype(np.float32) / 32768.0
        if len(x) < FRAME:
            break
        idx = np.arange(0, len(x) - FRAME + 1, STEP)
        if a + block < len(audio):
            idx = idx[idx < block]
        frames = x[idx[:, None] + np.arange(FRAME)] * win
        p = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        tot = p[:, band].sum(axis=1) + 1e-10
        db.append(10 * np.log10(tot))
        br.append(p[:, bright].sum(axis=1) / tot)
    return np.concatenate(db), np.concatenate(br)


def _bin(x: np.ndarray, per: int, how=np.mean) -> np.ndarray:
    n = len(x) // per
    return how(x[: n * per].reshape(n, per), axis=1)


def _rolling_baseline(x: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Niveau habituel (mediane) et dispersion (ecart absolu median) sur une fenetre glissante."""
    step = max(1, win // 4)
    centers, meds, mads = [], [], []
    for c in range(0, len(x), step):
        seg = x[max(0, c - win // 2): c + win // 2 + 1]
        m = np.median(seg)
        centers.append(c)
        meds.append(m)
        mads.append(np.median(np.abs(seg - m)) + 1e-6)
    pos = np.arange(len(x))
    return np.interp(pos, centers, meds), np.interp(pos, centers, mads)


def _speech(words: list[dict], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Presence de parole (0 a 1) et mots par seconde, par pas de HOP."""
    pres = np.zeros(n)
    rate = np.zeros(n)
    for w in words:
        a, b = int(w["s"] / HOP), int(w["e"] / HOP)
        pres[a:min(n, b + 1)] = 1.0
        if a < n:
            rate[a] += 1 / HOP
    k = int(2 / HOP)                                   # debit moyen sur 2 s
    rate = np.convolve(rate, np.ones(k) / k, mode="same")
    pres = np.convolve(pres, np.ones(3) / 3, mode="same")
    return np.clip(pres, 0, 1), rate


def compute(audio: np.ndarray, words: list[dict] | None = None) -> dict:
    db, br = _frame_features(audio)
    per = int(HOP * SR / STEP)                         # 10 trames de 50 ms par pas
    # energie moyenne (pas moyenne des dB) puis retour en dB
    loud = 10 * np.log10(_bin(10 ** (db / 10), per) + 1e-10)
    bright = _bin(br, per)
    n = len(loud)
    win = int(BASE_WIN / HOP)
    lm, ls = _rolling_baseline(loud, win)
    bm, bs = _rolling_baseline(bright, win)
    z_loud = np.clip((loud - lm) / (1.4826 * ls), 0, 6)
    z_bright = np.clip((bright - bm) / (1.4826 * bs), 0, 6)
    if words:
        pres, rate = _speech(words, n)
        rm, rs = _rolling_baseline(rate, win)
        z_rate = np.clip((rate - rm) / (1.4826 * rs + 0.5), 0, 6)
        gate = 0.35 + 0.65 * pres
    else:                                              # sans transcription : pas de debit ni d'attenuation
        z_rate = np.zeros(n)
        gate = np.ones(n)
    raw = (0.55 * z_loud + 0.30 * z_bright + 0.15 * z_rate) * gate
    sigma = 1.0 / HOP                                  # lissage sur environ 1 s
    r = int(3 * sigma)
    ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    sm = np.convolve(np.pad(raw, r, mode="edge"), ker / ker.sum(), mode="valid")
    ref = np.percentile(sm, 99.5) if n > 20 else sm.max()
    score = np.clip(sm / max(ref, 1e-6), 0, 1)
    per_sec = _bin(score, int(1 / HOP), np.max)        # score par seconde
    return {"hop": 1.0, "score": [round(float(v), 3) for v in per_sec],
            "peaks": find_peaks(per_sec)}


def find_peaks(score: np.ndarray, gap: float = PEAK_GAP, min_score: float = PEAK_MIN) -> list[dict]:
    order = np.argsort(score)[::-1]
    taken: list[int] = []
    for i in order:
        if score[i] < min_score:
            break
        if all(abs(i - j) >= gap for j in taken):
            taken.append(int(i))
    return [{"t": float(i), "score": round(float(score[i]), 3)} for i in sorted(taken)]


def save(src: Path, res: dict) -> Path:
    out = reactions_path(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return out


def load(src: Path) -> dict | None:
    f = reactions_path(src)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def chart(src: Path, res: dict, width: int = 1600, height: int = 360) -> Path:
    """Courbe du score dans le temps, pics marques, pour verifier d'un coup d'oeil."""
    s = np.array(res["score"])
    img = np.full((height, width, 3), 255, np.uint8)
    m, top, bot = 50, 20, height - 40
    n = max(1, len(s) - 1)
    xs = (m + np.arange(len(s)) / n * (width - 2 * m)).astype(int)
    ys = (bot - s * (bot - top)).astype(int)
    for k in range(0, len(s), max(1, len(s) // 12)):     # graduations
        x = int(m + k / n * (width - 2 * m))
        cv2.line(img, (x, top), (x, bot), (235, 235, 235), 1)
        mm, ss = divmod(k, 60)
        cv2.putText(img, f"{mm // 60}:{mm % 60:02d}:{ss:02d}" if mm >= 60 else f"{mm}:{ss:02d}",
                    (x - 18, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    y_thr = int(bot - PEAK_MIN * (bot - top))
    cv2.line(img, (m, y_thr), (width - m, y_thr), (200, 200, 255), 1)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, (140, 50, 120), 1, cv2.LINE_AA)
    ranked = sorted(res["peaks"], key=lambda p: -p["score"])
    for i, p in enumerate(ranked, 1):          # numero = rang, comme dans la liste affichee
        x, y = xs[int(p["t"])], ys[int(p["t"])]
        cv2.circle(img, (x, y), 4, (40, 40, 220), -1)
        cv2.putText(img, str(i), (x - 4, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 220), 1, cv2.LINE_AA)
    out = reactions_path(src).with_suffix(".png")
    cv2.imwrite(str(out), img)
    return out
