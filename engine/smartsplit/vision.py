"""Etape 6 bis : ce qui se passe a l'ecran, vu par le modele de langage (gemma3 sait lire les images).

Le stream est decoupe en fenetres de 30 s. Pour chacune, 4 images du jeu sont assemblees en une
mosaique et envoyees au modele, qui repond : type d'action (fusillade, accident, poursuite, mort...),
intensite de 0 a 10 et une courte description. Ces descriptions sont ensuite donnees a la notation
et au montage, et les fenetres tres intenses deviennent des clips meme si Omar ne parle pas.

Les images viennent des images cles enregistrees pendant l'analyse des scenes (dossier keyframes).
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import DATA, VISION_BACKEND, VISION_MODEL, VISION_WINDOW
from .llm import LLM_MODEL, OLLAMA_URL

ACTIONS = ["fusillade", "accident", "poursuite", "mort", "explosion", "combat", "animal", "monstre",
           "cascade", "cinematique", "conduite", "exploration", "dialogue", "menu", "rien"]

PROMPT = """Voici {n} images d'un jeu video (GTA 6), prises dans l'ordre sur 30 secondes.
Une petite webcam du joueur peut apparaitre dans un coin : ignore-la.
Regarde CHAQUE image en detail, meme si un element n'apparait que sur une seule image :
- animaux (alligator, crocodile, requin, tigre, chien...) ou creatures (yeti, monstre) ;
- armes qui tirent, sang, personnage a terre, ecran noir ou ecran de mort ;
- voitures accidentees ou qui se percutent, explosions, feu, police et gyrophares ;
- cinematique (cadrage de film, bandes noires, gros plan sur des personnages).
Puis reponds :
- action : le type le plus fort vu sur au moins une image, parmi """ + ", ".join(ACTIONS) + """ ;
- intensite de 0 a 10 pour un clip TikTok : 8 a 10 = fusillade, accident violent, explosion, mort du
  personnage, course-poursuite, attaque d'animal ou de monstre, scene spectaculaire ;
  4 a 7 = action moderee ou tres beau paysage ; 0 a 3 = conduite calme, marche, menu, dialogue ;
- description : une phrase courte et concrete en francais (15 mots maximum), qui cite l'element le plus
  marquant (ex. "Un alligator attaque le personnage dans le marais").
Reponds uniquement en JSON."""

VISION_FRAMES = 4    # images par fenetre de 30 s pour le passage general (reduites en VISION_SIZE)
VISION_SIZE = (480, 270)
FOCUS_N = 40         # passages examines de pres : les plus forts pics de reaction du stream
FOCUS_FRAMES = 6     # images en 1280x720 sur les 8 s qui precedent et suivent le pic

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "intensite": {"type": "integer"},
        "description": {"type": "string"},
    },
    "required": ["action", "intensite", "description"],
}


LAST_TIMES: list[float] = []    # duree de chaque description (pour mesurer la vitesse)


class VisionUnavailable(RuntimeError):
    """Le modele ne repond pas assez vite aux images : on continue sans la vision."""


def frames_dir(src: Path) -> Path:
    return DATA / "analysis" / Path(src).stem / "keyframes"


def vision_path(src: Path) -> Path:
    return DATA / "analysis" / Path(src).stem / "vision.json"


def ensure_keyframes(src: Path, progress=print) -> Path:
    """Images cles en 640x360 (normalement deja enregistrees par l'analyse des scenes)."""
    d = frames_dir(src)
    if d.exists() and any(d.glob("*.jpg")):
        return d
    from .scenes import keyframes
    d.mkdir(parents=True, exist_ok=True)
    for n, (ts, img) in enumerate(keyframes(src)):
        if ts is not None:
            cv2.imwrite(str(d / f"{round(ts * 10)}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if n % 500 == 0:
            progress(f"  images cles : {(ts or 0) / 60:.0f} min")
    return d


def _index(d: Path) -> list[tuple[float, Path]]:
    out = []
    for f in d.glob("*.jpg"):
        try:
            out.append((int(f.stem) / 10, f))
        except ValueError:
            pass
    return sorted(out)


def _jpg(path: Path, small: bool = True) -> bytes:
    if not small:
        return Path(path).read_bytes()
    img = cv2.resize(cv2.imread(str(path)), VISION_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def _mosaic(paths: list[Path]) -> bytes:
    imgs = [cv2.imread(str(p)) for p in paths]
    imgs = [cv2.resize(i, (640, 360)) for i in imgs if i is not None]
    cols = 3 if len(imgs) > 4 else 2
    while len(imgs) % cols or len(imgs) < cols * 2:
        imgs.append(np.zeros((360, 640, 3), np.uint8))
    rows = [cv2.hconcat(imgs[i:i + cols]) for i in range(0, len(imgs), cols)]
    return cv2.imencode(".jpg", cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


_MULTI_OK = True     # passe a False si le modele refuse plusieurs images par requete


def _post(content: str, images: list[bytes], timeout: float) -> dict:
    body = json.dumps({
        "model": VISION_MODEL, "stream": False, "format": SCHEMA, "keep_alive": "5m",
        "options": {"temperature": 0, "num_ctx": 16384},
        "messages": [{"role": "user", "content": content,
                      "images": [base64.b64encode(b).decode() for b in images]}],
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(json.loads(r.read())["message"]["content"])
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {e.code} : {detail}") from e


def describe(paths: list[Path], timeout: float = 120, small: bool = True) -> dict:
    global _MULTI_OK
    prompt = PROMPT.replace("{n}", str(len(paths)))
    res = None
    if _MULTI_OK and len(paths) > 1:
        try:
            res = _post(prompt, [_jpg(p, small) for p in paths], timeout)
        except RuntimeError as e:
            if "HTTP 400" not in str(e):
                raise
            _MULTI_OK = False          # le modele n'accepte qu'une image : on passe a la mosaique
            print(f"  [plusieurs images refusees ({e}) : passage a une mosaique]")
    if res is None:
        res = _post(prompt + "\nLes images sont assemblees en une mosaique, a lire de gauche a droite puis de haut en bas.",
                    [_mosaic(paths)], timeout)
    res["intensite"] = max(0, min(10, int(res.get("intensite", 0))))
    return res


def _scene_at(segments: list[dict], t: float) -> str:
    for s in segments:
        if s["start"] <= t < s["end"]:
            return s["scene"]
    return "jeu"


def analyze(src: Path, segments: list[dict], duration: float, start: float = 0.0, end: float | None = None,
            progress=print, resume: list[dict] | None = None, save_every=None,
            frames: list[tuple[float, Path]] | None = None) -> list[dict]:
    """Decrit chaque fenetre de VISION_WINDOW secondes ; les images en full cam sont ignorees."""
    if frames is None:
        frames = _index(ensure_keyframes(src, progress))
    end = duration if end is None else end
    done = {round(w["start"], 1): w for w in (resume or [])}
    out: list[dict] = []
    t0 = time.time()
    calls = 0
    fails = 0
    first = True
    wins = np.arange(start, end, VISION_WINDOW)
    for i, a in enumerate(wins):
        b = min(a + VISION_WINDOW, end)
        if round(a, 1) in done:
            out.append(done[round(a, 1)])
            continue
        inside = [(t, p) for t, p in frames if a <= t < b and _scene_at(segments, t) != "full_cam"]
        if len(inside) < 2:
            continue
        pick = [inside[k][1] for k in np.linspace(0, len(inside) - 1, min(VISION_FRAMES, len(inside))).round().astype(int)]
        try:
            t1 = time.time()
            if VISION_BACKEND == "clip":
                from .vision_clip import describe as clip_describe
                res = clip_describe(pick)
            else:
                res = describe(pick, timeout=600 if first else 120)   # le 1er appel charge le modele
            if first:
                progress(f"  premiere image decrite en {time.time() - t1:.0f} s (chargement du modele compris)")
            first = False
            fails = 0
            LAST_TIMES.append(time.time() - t1)
        except Exception as e:  # noqa: BLE001
            first = False
            fails += 1
            progress(f"  [fenetre {a / 60:.1f} min ignoree : {e}]")
            if fails >= 3:
                raise VisionUnavailable(f"3 echecs de suite ({e})") from e
            continue
        calls += 1
        out.append({"start": round(float(a), 1), "end": round(float(b), 1), **res})
        if calls % 25 == 0:
            el = time.time() - t0
            progress(f"  vision : {i + 1}/{len(wins)} fenetres ({el:.0f} s, {el / calls:.1f} s par image)")
            if save_every:
                save_every(out)
    if calls:
        progress(f"  vision : {calls} fenetres decrites en {time.time() - t0:.0f} s ({(time.time() - t0) / calls:.1f} s par fenetre)")
    if VISION_BACKEND != "clip" and not any(w.get("focus") for w in out):
        out += focus(src, segments, start, end, progress)
    return sorted(out, key=lambda w: w["start"])


def focus_frames(src: Path, t: float) -> list[Path]:
    """Images en 1280x720 autour d'un instant (6 s avant, 2 s apres) : de quoi voir un detail lointain."""
    import subprocess
    d = DATA / "analysis" / Path(src).stem / "focus"
    d.mkdir(parents=True, exist_ok=True)
    a = max(0.0, t - 6.0)
    pat = d / f"{int(t * 10)}_%02d.jpg"
    if not (d / f"{int(t * 10)}_01.jpg").exists():
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{a:.2f}", "-t", "8",
                        "-i", str(src), "-vf", f"fps={FOCUS_FRAMES / 8:.3f},scale=1280:720", "-q:v", "3",
                        str(pat)], check=True)
    return sorted(d.glob(f"{int(t * 10)}_*.jpg"))[:FOCUS_FRAMES]


def focus(src: Path, segments: list[dict], start: float, end: float, progress=print) -> list[dict]:
    """Examen de pres des plus forts pics de reaction (images plus grandes et plus rapprochees)."""
    from .reactions import load as load_reactions
    rx = load_reactions(src)
    if not rx or FOCUS_N <= 0:
        return []
    peaks = [p for p in rx["peaks"] if start <= p["t"] < end and _scene_at(segments, p["t"]) != "full_cam"]
    peaks = sorted(peaks, key=lambda p: -p["score"])[:FOCUS_N]
    out, t0 = [], time.time()
    for p in sorted(peaks, key=lambda p: p["t"]):
        try:
            res = describe(focus_frames(src, p["t"]), small=False)
        except Exception as e:  # noqa: BLE001
            progress(f"  [examen de pres a {p['t'] / 60:.1f} min ignore : {e}]")
            continue
        out.append({"start": round(max(0.0, p["t"] - 6), 1), "end": round(p["t"] + 2, 1), "focus": True, **res})
    if peaks:
        progress(f"  examen de pres : {len(out)} pics de reaction en {time.time() - t0:.0f} s")
    return out


def save(src: Path, wins: list[dict], complete: bool = True) -> Path:
    p = vision_path(src)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"window": VISION_WINDOW, "complete": complete, "windows": wins},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load(src: Path, partial: bool = False) -> list[dict] | None:
    """Fenetres decrites ; None si l'analyse n'est pas terminee (sauf partial=True, pour la reprise)."""
    p = vision_path(src)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["windows"] if (partial or d.get("complete", True)) else None


def screen_text(wins: list[dict], a: float, b: float) -> str:
    """Resume de ce qui se passe a l'ecran entre a et b (pour les invites de notation et de montage)."""
    parts = [f"{w['description']} ({w['action']}, intensite {w['intensite']})"
             for w in wins if w["end"] > a and w["start"] < b and w["action"] not in ("menu", "rien")]
    return " ; ".join(parts)


def action_blocks(wins: list[dict], min_intensity: int = 7) -> list[dict]:
    """Fenetres tres intenses, transformees en extraits candidats (meme sans paroles)."""
    out = []
    for w in wins:
        if w["intensite"] >= min_intensity and w["action"] not in ("cinematique", "menu", "rien", "dialogue"):
            out.append({"start": w["start"], "end": w["end"], "note": min(w["intensite"], 8), "type": "action",
                        "raison": f"A l'ecran : {w['description']} ({w['action']})", "text": "", "vision": True})
    return out
