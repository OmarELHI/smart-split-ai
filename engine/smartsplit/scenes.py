"""Etape 2 : detection des scenes OBS et de la position de la cam.

Principe
1. Passe rapide : on ne decode que les images cles (une toutes les ~4 s dans les
   enregistrements OBS), reduites en 640x360.
2. Amorce : on repere ta cam la ou un rectangle cadre (rose/bleu marine) de la taille
   habituelle revient souvent, et on en garde quelques images de reference.
3. Reconnaissance par le contenu : on cherche ces images de reference (ta chambre), a
   plusieurs tailles, dans le haut de chaque image. Ca trouve la cam quelle que soit sa
   taille ou sa place, et ignore la cam d'un autre streamer que tu regardes.
   - une zone tres ressemblante dans le haut, mais pas l'image entiere : cam + jeu
   - l'image entiere ressemble a ta chambre : full cam
   - sinon : jeu seul
4. Lissage, puis affinage de chaque changement de scene a 0,25 s pres, et mesure exacte
   des bords de chaque position de cam en pleine resolution.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import CAM_INSET, CAM_OUTER_H, CAM_OUTER_W, DATA, Rect

AW, AH = 640, 360                       # resolution d'analyse
CAM_W_REL, CAM_H_REL = CAM_OUTER_W, CAM_OUTER_H
CW, CH = round(CAM_W_REL * AW), round(CAM_H_REL * AH)   # 134 x 74 a l'echelle d'analyse
BAND = int(AH * 0.25)                   # la cam est cherchee dans le quart haut de l'image

EDGE_MIN = 25.0        # nettete minimale des 4 bords
RING_MIN = 0.05        # part minimale de pixels rose/bleu marine autour
STILL_FULLCAM = 3.0    # ecart median entre 2 images cles en dessous duquel l'image est "fixe"
STILL_MAX = 12.0       # au-dessus : trop de mouvement pour une full cam
ROOM_SIM = 0.30        # ressemblance minimale avec la piece vue dans la cam
CLUSTER_TOL = 4        # tolerance (px a 640) pour regrouper les positions de cam


# Mesures d'image

def edge_map(img: np.ndarray) -> np.ndarray:
    """Score de rectangle de taille cam a chaque position (haut-gauche), sur la bande haute."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    sh = cv2.filter2D(gy, -1, np.ones((1, CW), np.float32) / CW, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    sv = cv2.filter2D(gx, -1, np.ones((CH, 1), np.float32) / CH, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    ny, nx = min(BAND, AH - CH - 1), AW - CW - 1
    top, bot = sh[:ny, :nx], sh[CH:CH + ny, :nx]
    lef, rig = sv[:ny, :nx], sv[:ny, CW:CW + nx]
    return np.minimum(np.minimum(top, bot), np.minimum(lef, rig))


def ring_score(img: np.ndarray, x: int, y: int, pad: int = 3) -> float:
    """Part de pixels rose ou bleu marine sur le contour du rectangle (le cadre de la cam)."""
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(AW, x + CW + pad), min(AH, y + CH + pad)
    roi = img[y0:y1, x0:x1].astype(np.int16)
    b, g, r = roi[..., 0], roi[..., 1], roi[..., 2]
    mask = np.ones(b.shape, bool)
    mask[(y - y0) + 1:(y - y0) + CH - 1, (x - x0) + 1:(x - x0) + CW - 1] = False
    navy = (b > r + 20) & (b > g + 12)
    pink = (r > g + 18) & (b > g + 12)
    return float(((navy | pink) & mask).sum() / max(1, mask.sum()))


def candidates(img: np.ndarray, k: int = 12) -> list[tuple[int, int, float, float]]:
    """Meilleurs rectangles candidats (x, y, nettete, cadre) apres suppression des doublons voisins."""
    s = edge_map(img)
    peaks = (s == cv2.dilate(s, np.ones((9, 9), np.uint8))) & (s >= EDGE_MIN * 0.6)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(s[ys, xs])[::-1][:k]
    return [(int(xs[i]), int(ys[i]), float(s[ys[i], xs[i]]), ring_score(img, int(xs[i]), int(ys[i]))) for i in order]


def hsv_hist(img: np.ndarray) -> np.ndarray:
    h = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(h, h, 1, 0, cv2.NORM_L1)
    return h


def small_gray(img: np.ndarray) -> np.ndarray:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g.astype(np.float32), (5, 5), 0)


def stillness(a: np.ndarray, b: np.ndarray) -> float:
    """Ecart median entre deux images sur les bords gauche et droit (hors personne au centre)."""
    return float(np.median(np.abs(np.concatenate([a[:, :40] - b[:, :40], a[:, -40:] - b[:, -40:]], axis=1))))


# Lecture video

def _ffmpeg_frames(cmd: list[str]):
    """Lit les images BGR 640x360 d'une commande FFmpeg et leur horodatage (filtre showinfo)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    times: list[float] = []

    def read_err():
        for line in proc.stderr:
            m = re.search(rb"pts_time:\s*([0-9.]+)", line)
            if m:
                times.append(float(m.group(1)))

    t = threading.Thread(target=read_err, daemon=True)
    t.start()
    size = AW * AH * 3
    i = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        while len(times) <= i and t.is_alive():
            t.join(0.01)
        ts = times[i] if i < len(times) else None
        yield ts, np.frombuffer(buf, np.uint8).reshape(AH, AW, 3)
        i += 1
    proc.wait()


def keyframes(src: Path):
    """Images cles uniquement : rapide meme sur 5 h de video."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", "-skip_frame", "nokey",
           "-i", str(src), "-vf", f"showinfo,scale={AW}:{AH}", "-fps_mode", "passthrough",
           "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    yield from _ffmpeg_frames(cmd)


def frames_between(src: Path, t0: float, t1: float, fps: float = 4.0):
    """Toutes les images entre t0 et t1, a fps images par seconde."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}",
           "-i", str(src), "-vf", f"fps={fps},showinfo,scale={AW}:{AH}", "-an",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    for ts, img in _ffmpeg_frames(cmd):
        yield (t0 + ts if ts is not None else None), img


# Analyse

CAM_MATCH = 0.84     # ressemblance minimale entre une zone du haut de l'image et ta cam
CAM_GAP = 0.24       # la zone doit ressembler bien plus a la cam que l'image entiere (sinon : full cam)
FULL_MIN = 0.50      # ressemblance minimale de l'image entiere avec ta cam pour une full cam
SEARCH_S = 0.5       # echelle de la recherche multi-tailles
BAND_REL = 0.45      # la cam est cherchee dans les 45 % du haut
WIDTHS = np.arange(0.14, 0.37, 0.02)   # largeurs de cam essayees (part de la largeur de l'image)
CAM_ASPECT = 536 / 298   # largeur / hauteur de l'image webcam incrustee (mesuree)
INSET = 2            # retrait (px a 640) des references pour ne pas inclure le cadre


def gray(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)


class Matcher:
    """Reconnait ta cam par son contenu (ta chambre) a partir d'images de reference."""

    def __init__(self, refs: list[np.ndarray]):
        self.refs = refs
        self.thumbs = [cv2.resize(r, (64, 36), interpolation=cv2.INTER_AREA) for r in refs]
        self.tmpl = []
        for r in refs:
            for wr in WIDTHS:
                w = int(AW * wr * SEARCH_S)
                h = int(round(w * r.shape[0] / r.shape[1]))
                self.tmpl.append((cv2.resize(r, (w, h), interpolation=cv2.INTER_AREA), w, h))

    @staticmethod
    def band(g: np.ndarray) -> np.ndarray:
        return cv2.resize(g[: int(AH * BAND_REL)], None, fx=SEARCH_S, fy=SEARCH_S, interpolation=cv2.INTER_AREA)

    def cam(self, band: np.ndarray) -> tuple[float, tuple[int, int, int, int] | None]:
        """Meilleure zone ressemblant a la cam : (score, rectangle exterieur a 640x360)."""
        best, rect = -1.0, None
        for t, w, h in self.tmpl:
            if w >= band.shape[1] or h >= band.shape[0]:
                continue
            _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(band, t, cv2.TM_CCOEFF_NORMED))
            if mx > best:
                best, rect = mx, (loc[0] / SEARCH_S, loc[1] / SEARCH_S, w / SEARCH_S, h / SEARCH_S)
        if rect is None:
            return best, None
        x, y, w, h = rect
        k = w / (CW - 2 * INSET)          # la reference est la cam rentree de INSET px
        return best, (round(x - INSET * k), round(y - INSET * k), round(w + 2 * INSET * k), round(h + 2 * INSET * k))

    def full(self, thumb: np.ndarray) -> float:
        return max(float(cv2.matchTemplate(thumb, r, cv2.TM_CCOEFF_NORMED)[0, 0]) for r in self.thumbs)


@dataclass
class Sample:
    t: float
    cands: list
    still: float | None = None
    band: np.ndarray | None = None
    thumb: np.ndarray | None = None
    crops: dict = field(default_factory=dict)
    cam_score: float = 0.0
    full_score: float = 0.0
    rect: tuple | None = None
    scene: str = ""
    cam: int | None = None          # index de la position de cam


@dataclass
class Segment:
    start: float
    end: float
    scene: str
    cam: list | None = None         # x, y, w, h relatifs


@dataclass
class Analysis:
    src: str
    duration: float
    cams: list = field(default_factory=list)      # positions de cam trouvees (relatives)
    segments: list = field(default_factory=list)
    samples: list = field(default_factory=list)   # diagnostic


def _find_cam_positions(samples: list[Sample]) -> list[tuple[int, int, float]]:
    """Amorce : positions ou un rectangle cadre de la taille de la cam revient souvent."""
    pts = [(c[0], c[1], c[3], c[2], s.t) for s in samples for c in s.cands[:3]
           if c[2] >= EDGE_MIN and c[3] >= RING_MIN]
    clusters: list[list] = []
    for p in pts:
        for cl in clusters:
            if abs(cl[0][0] - p[0]) <= CLUSTER_TOL and abs(cl[0][1] - p[1]) <= CLUSTER_TOL:
                cl.append(p)
                break
        else:
            clusters.append([p])
    need = max(3, int(0.02 * len(samples)))
    keep = []
    for cl in sorted(clusters, key=len, reverse=True):
        if len(cl) >= need and np.mean([p[2] for p in cl]) >= 0.08:
            x, y = int(np.median([p[0] for p in cl])), int(np.median([p[1] for p in cl]))
            t = max(cl, key=lambda p: p[3])[4]
            if all(abs(x - q[0]) > 2 * CLUSTER_TOL or abs(y - q[1]) > 2 * CLUSTER_TOL for q in keep):
                keep.append((x, y, t))
    return keep


def _pick_refs(samples: list[Sample], boot: list[tuple[int, int, float]], n: int = 3) -> list[np.ndarray]:
    """Images de reference de ta cam, prises a la position d'amorce, reparties dans le temps."""
    if not boot:
        return []
    bx, by, _ = boot[0]
    found = []
    for s in samples:
        for (x, y), crop in s.crops.items():
            if abs(x - bx) <= CLUSTER_TOL and abs(y - by) <= CLUSTER_TOL:
                found.append(crop)
                break
    if not found:
        return []
    idx = np.linspace(0, len(found) - 1, min(n, len(found))).round().astype(int)
    return [found[i] for i in sorted(set(idx))]


def _decide(cam_score: float, full_score: float) -> str:
    if cam_score >= CAM_MATCH and cam_score - full_score >= CAM_GAP:
        return "cam_jeu"
    if full_score >= FULL_MIN:
        return "full_cam"
    return "jeu"


def _cluster_rects(samples: list[Sample]) -> list[tuple[int, int, int, int, float]]:
    """Regroupe les positions/tailles de cam trouvees : (x, y, w, h, instant le plus net)."""
    groups: list[list[Sample]] = []
    for s in samples:
        if s.scene != "cam_jeu":
            continue
        x, y, w, _ = s.rect
        for g in groups:
            gx, gy, gw, _ = g[0].rect
            if abs(gx - x) <= 8 and abs(gy - y) <= 8 and abs(gw - w) <= 12:
                g.append(s)
                break
        else:
            groups.append([s])
    out = []
    for g in groups:
        r = np.median([s.rect for s in g], axis=0)
        best = max(g, key=lambda s: s.cam_score)
        out.append((int(r[0]), int(r[1]), int(r[2]), int(r[3]), best.t))
    return out


def _nearest_cam(rect, cams) -> int | None:
    if rect is None:
        return None
    d = [abs(rect[0] - c[0]) + abs(rect[1] - c[1]) + abs(rect[2] - c[2]) for c in cams]
    return int(np.argmin(d)) if d else None


def _classify_fallback(s: Sample) -> None:
    """Sans reference de cam : full cam si l'image est fixe, sinon jeu."""
    s.scene = "full_cam" if s.still is not None and s.still < STILL_FULLCAM else "jeu"


def _label(s: Sample) -> str:
    return f"{s.scene}:{s.cam}" if s.scene == "cam_jeu" else s.scene


def _smooth(samples: list[Sample]) -> None:
    """Supprime les erreurs isolees : une image cle differente de ses deux voisines identiques."""
    labs = [_label(s) for s in samples]
    for i in range(1, len(samples) - 1):
        if labs[i - 1] == labs[i + 1] != labs[i]:
            samples[i].scene, samples[i].cam = samples[i - 1].scene, samples[i - 1].cam


def _frame_label(img: np.ndarray, matcher: Matcher | None, cams) -> str:
    if matcher is None:
        return "?"
    g = gray(img)
    sc, rect = matcher.cam(Matcher.band(g))
    scene = _decide(sc, matcher.full(cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA)))
    return f"cam_jeu:{_nearest_cam(rect, cams)}" if scene == "cam_jeu" else scene


def _refine(src: Path, a: Sample, b: Sample, matcher, cams, fps: float = 4.0) -> float:
    """Instant precis du changement de scene entre deux images cles.

    Chaque image intermediaire est classee "comme a" ou "comme b" ; on garde la coupure
    qui contredit le moins d'images (robuste a une erreur isolee).
    """
    frames = [(ts, img) for ts, img in frames_between(src, a.t, b.t + 0.5, fps)
              if ts is not None and ts <= b.t + 1.0 / fps]
    if len(frames) < 2:
        return b.t
    lab_a, lab_b = _label(a), _label(b)
    if matcher is not None:
        is_a = [(lambda lab: lab == lab_a if lab in (lab_a, lab_b) else None)(_frame_label(img, matcher, cams))
                for _, img in frames]
    else:
        ga, gb = small_gray(frames[0][1]), small_gray(frames[-1][1])
        is_a = [bool(np.mean(np.abs(small_gray(img) - ga)) <= np.mean(np.abs(small_gray(img) - gb)))
                for _, img in frames]
    best_k, best_err = len(frames) - 1, None
    for k in range(1, len(frames)):
        err = sum(v is False for v in is_a[:k]) + sum(v is True for v in is_a[k:])
        if best_err is None or err < best_err:
            best_k, best_err = k, err
    return frames[best_k][0]


def refine_cam_rect(src: Path, t: float, rect640: tuple[int, int, int, int]) -> Rect:
    """Bords exacts de la cam en pleine resolution, chaque bord cherche separement."""
    info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height", "-of", "csv=p=0", str(src)], capture_output=True, text=True)
    w, h = (int(v) for v in info.stdout.strip().split(",")[:2])
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    g = np.frombuffer(p.stdout, np.uint8)[: w * h].reshape(h, w).astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    sx, sy = w / AW, h / AH
    x0, y0 = rect640[0] * sx, rect640[1] * sy
    x1, y1 = x0 + rect640[2] * sx, y0 + rect640[3] * sy
    m = int(12 * sx)                      # marge de recherche de chaque bord
    ya, yb = int(y0 + (y1 - y0) * 0.2), int(y0 + (y1 - y0) * 0.8)
    xa, xb = int(x0 + (x1 - x0) * 0.2), int(x0 + (x1 - x0) * 0.8)

    def best_col(c):
        cols = range(max(1, int(c) - m), min(w - 1, int(c) + m + 1))
        return max(cols, key=lambda c_: gx[ya:yb, c_].mean())

    L, R = best_col(x0), best_col(x1)
    # la cam garde le format de la webcam (environ 16:9) : on cherche le haut et le bas ensemble,
    # pour ne pas confondre le bas de la cam avec le bandeau du pseudo juste dessous
    rows = gy[:, xa:xb].mean(axis=1)
    h0 = (R - L) / CAM_ASPECT
    best, T, B = -1.0, int(y0), int(y1)
    for hh in range(int(h0 * 0.97), int(h0 * 1.03) + 1):
        for t in range(max(1, int(y0) - m), min(h - hh - 1, int(y0) + m + 1)):
            sc = rows[t] + rows[t + hh]
            if sc > best:
                best, T, B = sc, t, t + hh
    ins = max(1, round(CAM_INSET * h))
    L, T, R, B = L + ins, T + ins, R - ins, B - ins
    return Rect(L / w, T / h, (R - L) / w, (B - T) / h)


def analyze(src: Path, progress=print) -> Analysis:
    src = Path(src)
    samples: list[Sample] = []
    prev_gray = None
    for n, (ts, img) in enumerate(keyframes(src)):
        if ts is None:
            continue
        cands = candidates(img)
        g = gray(img)
        sg = small_gray(img)
        s = Sample(t=ts, cands=cands, still=stillness(sg, prev_gray) if prev_gray is not None else None,
                   band=Matcher.band(g), thumb=cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA))
        if samples and s.still is not None:
            p = samples[-1]
            p.still = s.still if p.still is None else min(p.still, s.still)
        prev_gray = sg
        for x, y, e, r in cands[:3]:
            if e >= EDGE_MIN and r >= RING_MIN:
                s.crops[(x, y)] = g[y + INSET:y + CH - INSET, x + INSET:x + CW - INSET].copy()
        samples.append(s)
        if n % 200 == 0:
            progress(f"  analyse : {ts / 60:.1f} min")
    if not samples:
        raise RuntimeError("aucune image lue")
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                     "csv=p=0", str(src)], capture_output=True, text=True).stdout.strip())

    refs = _pick_refs(samples, _find_cam_positions(samples))
    matcher = Matcher(refs) if refs else None
    if matcher is None:
        progress("  aucune cam incrustee reperee : detection simplifiee (image fixe = full cam)")
        for s in samples:
            _classify_fallback(s)
        cams = []
    else:
        for s in samples:
            s.cam_score, s.rect = matcher.cam(s.band)
            s.full_score = matcher.full(s.thumb)
            s.scene = _decide(s.cam_score, s.full_score)
        cams = _cluster_rects(samples)
        for s in samples:
            s.cam = _nearest_cam(s.rect, cams) if s.scene == "cam_jeu" else None
    _smooth(samples)

    progress("  affinage des changements de scene")
    cam_rects = [refine_cam_rect(src, c[4], c[:4]) for c in cams]
    segs: list[Segment] = []
    start = 0.0
    for a, b in zip(samples, samples[1:]):
        if _label(a) != _label(b):
            cut = _refine(src, a, b, matcher, cams)
            segs.append(Segment(start, cut, a.scene, list(asdict(cam_rects[a.cam]).values()) if a.cam is not None else None))
            start = cut
    last = samples[-1]
    segs.append(Segment(start, duration, last.scene,
                        list(asdict(cam_rects[last.cam]).values()) if last.cam is not None else None))
    # fusionne les segments voisins identiques (meme scene, meme cam)
    merged: list[Segment] = []
    for sg in segs:
        if merged and merged[-1].scene == sg.scene and merged[-1].cam == sg.cam:
            merged[-1].end = sg.end
        elif sg.end - sg.start > 0.01:
            merged.append(sg)

    return Analysis(
        src=str(src), duration=duration,
        cams=[list(asdict(r).values()) for r in cam_rects],
        segments=[asdict(x) for x in merged],
        samples=[{"t": round(s.t, 2), "scene": s.scene, "cam": s.cam,
                  "cam_score": round(s.cam_score, 2), "full_score": round(s.full_score, 2),
                  "still": None if s.still is None else round(s.still, 1),
                  "rect": list(s.rect) if s.rect else None} for s in samples],
    )


def analysis_dir(src: Path) -> Path:
    return DATA / "analysis" / Path(src).stem


def save(an: Analysis) -> Path:
    d = analysis_dir(Path(an.src))
    d.mkdir(parents=True, exist_ok=True)
    out = d / "scenes.json"
    out.write_text(json.dumps(asdict(an), indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def load(src: Path) -> Analysis | None:
    f = analysis_dir(src) / "scenes.json"
    if not f.exists():
        return None
    return Analysis(**json.loads(f.read_text(encoding="utf-8")))


def segments_for(an: Analysis, start: float, end: float) -> list[tuple[float, float, str, Rect | None]]:
    """Morceaux de scene couvrant [start, end]."""
    out = []
    for s in an.segments:
        a, b = max(start, s["start"]), min(end, s["end"])
        if b - a > 0.02:
            out.append((a, b, s["scene"], Rect(*s["cam"]) if s["cam"] else None))
    return out


def contact_sheet(src: Path, an: Analysis, max_tiles: int = 120) -> Path:
    """Planche d'images legendees pour verifier la detection a l'oeil."""
    colors = {"cam_jeu": (180, 80, 220), "jeu": (60, 170, 60), "full_cam": (40, 140, 240)}
    n = len(an.samples)
    step = max(1, n // max_tiles)
    picks = an.samples[::step][:max_tiles]
    tiles = []
    for smp in picks:
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{smp['t']:.3f}", "-i", str(src), "-frames:v", "1",
                            "-vf", "scale=320:180", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
        if len(p.stdout) < 320 * 180 * 3:
            continue
        img = np.frombuffer(p.stdout, np.uint8).reshape(180, 320, 3).copy()
        k = 320 / AW
        if smp.get("rect") and smp["scene"] != "cam_jeu":
            x, y, w, h = smp["rect"]
            cv2.rectangle(img, (int(x * k), int(y * k)), (int((x + w) * k), int((y + h) * k)), (0, 255, 255), 1)
        if smp["cam"] is not None and smp["cam"] < len(an.cams):
            x, y, w, h = an.cams[smp["cam"]]
            cv2.rectangle(img, (int(x * 320), int(y * 180)), (int((x + w) * 320), int((y + h) * 180)), (255, 0, 255), 2)
        lab = smp["scene"] + (f" #{smp['cam']}" if smp["cam"] is not None else "")
        lab += f"  c{smp.get('cam_score', 0):.2f} f{smp.get('full_score', 0):.2f}"
        col = colors.get(smp["scene"], (200, 200, 200))
        cv2.rectangle(img, (0, 0), (319, 179), col, 4)
        cv2.rectangle(img, (0, 0), (319, 24), (0, 0, 0), -1)
        m, sec = divmod(int(smp["t"]), 60)
        cv2.putText(img, f"{m:02d}:{sec:02d}  {lab}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        tiles.append(img)
    cols = 8
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = analysis_dir(src) / "planche.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return out
