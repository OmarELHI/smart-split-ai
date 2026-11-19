"""Reglages communs du moteur."""
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUT = DATA / "clips"

# Format de sortie TikTok
OUT_W, OUT_H = 1080, 1920
CAM_SHARE = 0.31  # part de la hauteur occupee par la cam en scene cam + jeu

# Scene cam + jeu : fond flou, puis cam, puis jeu (empiles en bas)
GAME_RATIO = 1.0       # ratio largeur / hauteur de la zone jeu (1 = carre, 16/9 = jeu entier)
BOTTOM_MARGIN = 0.0    # espace flou sous le jeu, en part de la hauteur (0 = jeu colle en bas)

# FFmpeg installe par setup.ps1 : utilise meme si le PATH du terminal n'est pas a jour
FFMPEG_DIR = Path(r"D:\Tools\ffmpeg\bin")
if not shutil.which("ffmpeg") and (FFMPEG_DIR / "ffmpeg.exe").exists():
    os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ["PATH"]

# Modeles deja telecharges : pas de verification aupres de Hugging Face
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Modeles d'IA toujours sur D: (dans data\models), meme si la variable n'est pas definie dans le terminal
os.environ.setdefault("HF_HOME", str(DATA / "models"))


@dataclass(frozen=True)
class Rect:
    """Rectangle en coordonnees relatives (0 a 1) de l'image source."""

    x: float
    y: float
    w: float
    h: float

    def to_px(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convertit en pixels pairs (exige par l'encodeur)."""
        x = int(round(self.x * width / 2)) * 2
        y = int(round(self.y * height / 2)) * 2
        w = int(round(self.w * width / 2)) * 2
        h = int(round(self.h * height / 2)) * 2
        return x, y, w, h

    @classmethod
    def parse(cls, text: str, width: int | None = None, height: int | None = None) -> "Rect":
        """Lit "x,y,w,h" en relatif (0.76,...) ou en pixels (1956,54,530,292)."""
        x, y, w, h = (float(v) for v in text.split(","))
        if max(x, y, w, h) > 1:
            if not (width and height):
                raise ValueError("dimensions de la source necessaires pour des pixels")
            return cls(x / width, y / height, w / width, h / height)
        return cls(x, y, w, h)


# Image de la cam en haut a droite, mesuree sur un enregistrement 2560x1440 :
# bord exterieur de l'image webcam 536 x 298 a (1953, 52), et zone gardee (3 px en retrait, sans le cadre rose)
CAM_OUTER_W, CAM_OUTER_H = 536 / 2560, 298 / 1440
CAM_INSET = 3 / 1440
CAM_TOP_RIGHT = Rect(1956 / 2560, 55 / 1440, 530 / 2560, 292 / 1440)
