"""Reglages communs du moteur."""
import json
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

# Texte publicitaire affiche pendant tout le clip (comme dans ton montage CapCut). "" = aucun texte.
WATERMARK = "Twitch : OmarELHI"
WATERMARK_Y = 0.15           # centre du texte : juste sous le titre, au-dessus de ta tete (cam + jeu comme full cam)
WATERMARK_SIZE = 56          # taille de police en pixels (pour une sortie 1080x1920)
WATERMARK_OPACITY = 0.6      # 1 = opaque, 0 = invisible
WATERMARK_COLOR = "E25993"   # contour et ombre roses, texte blanc
WATERMARK_IMAGE = ""         # chemin d'un PNG transparent 1080 px de large a utiliser a la place (ex. exporte de CapCut)

# Titre d'accroche (bandeau blanc, texte noir) ecrit par l'IA pour chaque clip, comme sur les clips GTA qui marchent
TITLE = True
TITLE_Y = 0.075              # haut du bandeau : sous la barre du haut de TikTok, sans coller au bord
TITLE_SIZE = 54              # taille de police en pixels
TITLE_MAX_CHARS = 22         # longueur visee d'une ligne (2 lignes maximum)

# Montage : le choix garde un "rush" plus long autour du moment, puis l'IA y choisit les passages a garder
RUSH_BEFORE = 60.0           # secondes gardees avant le moment repere
RUSH_AFTER = 45.0            # et apres
CUT_SILENCE = 1.5            # silence (sans paroles ni action) a couper a l'interieur d'un passage garde
ZOOM = 1.12                  # zoom avant sur les passages de reaction (1.0 = aucun)

# Duree des clips : un moment court (avis, reaction) reste sous 60 s ; une histoire complete
# (contexte, action, reaction) peut aller jusqu'a LONG_MAX, avec en plus une version courte.
SHORT_MAX = 60.0
LONG_MAX = 95.0

# Vision : ce qui se passe a l'ecran (fusillade, accident...), decrit par l'IA toutes les VISION_WINDOW secondes
VISION = True
VISION_WINDOW = 30.0
VISION_BACKEND = "ollama"  # "ollama" : un modele decrit les images (VISION_MODEL) ; "clip" : SigLIP, tres rapide mais rate les actions
VISION_MODEL = "qwen2.5vl:7b"   # 0,6 s par image sur la RTX 5070 (gemma3:12b : plus d'une minute)
VISION_MIN = 7       # intensite a partir de laquelle une action devient un clip candidat, meme sans paroles
MISSION_RECAPS = True  # une video resume par mission (nom lu en rose en bas a droite de l'ecran)
MISSION_MIN = 180.0    # duree minimale d'une mission pour en faire un resume (s)
RECAP_MAX = 60.0       # duree visee d'un resume (s)
CAM_BONUS = 0.15     # bonus de score pour un clip ou ta cam est visible (cam + jeu ou full cam) tout du long

# Tendances : ce qui marche sur YouTube Shorts et TikTok autour du jeu, donne a l'IA pour choisir les clips
TREND_GAME = "GTA 6"
TREND_DAYS = 3       # mise a jour automatique tous les 3 jours
TREND_TIKTOK = ["jltomylive", "saabgta", "jlnohaa", "clip.gods16"]   # comptes de clips TikTok a suivre
VISION_CLIPS = False # clips sur la vision seule (sans reaction d'Omar) : 13 rejets sur 13, desactive

# Vocabulaire propre a tes streams, donne a Whisper (option "hotwords").
# Desactive : teste le 19/11/2026, il n'a pas corrige "GTA 6" et a degrade le decoupage en phrases.
# Exemple : VOCAB = "GTA 6, Rockstar, Twitch, TikTok"
VOCAB = ""

# Corrections automatiques des erreurs recurrentes de Whisper (mot ou expression -> correction).
# Appliquees a chaque lecture de la transcription : pas besoin de retranscrire apres un ajout.
CORRECTIONS = {
    "marouquins": "marocains",
    "marouquin": "marocain",
    "j'étais à 6": "GTA 6",
    "JLTOMI": "JLTomy",
}

# Contexte donne a l'IA pour juger les extraits : personnes, jeux, habitudes du stream (a completer librement)
PROMPT_CONTEXT = """- JLTomy (ecrit parfois JLTOMI) est un tres gros streamer francais : sa venue sur le live,
  un raid, un message ou une mention de sa part est un grand moment (note 8 a 10).
- Les streamers et createurs connus qui passent sur le live ou sont mentionnes rendent un extrait plus interessant."""

# Clips qu'Omar a juges reussis (publies) : donnes a l'IA comme exemples de ce qu'il aime.
# Ajoute ici le texte (ou la description) de chaque clip que tu publies.
GOOD_EXAMPLES = [
    "Emerveillement court et spontane devant le jeu (publie, 18 s) : \"Les gars le jeu mes freres... "
    "Les gars, l'eau, la qualite de l'eau, les graphismes. Mes freres, c'est incroyable. C'est juste incroyable\"",
    "Clip de JLTomy a 205 000 vues, \"Alors la je m'attendais pas a ca\" : surprise devant un tigre qui apparait dans le jeu",
    "Clip de JLTomy a 178 000 vues : il est choque par le realisme des PNJ de GTA 6, qu'on voit a l'ecran",
    "Histoire en jeu montee a la main (92 s) : on voit le yeti courir, reaction d'Omar, un alligator l'attaque, "
    "il cherche le yeti, le trouve, puis la cinematique, avec ses reactions tout du long",
]

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


# Reglages modifies depuis l'application (data/settings.json, section "engine") : ils remplacent les valeurs ci-dessus.
SETTINGS_FILE = DATA / "settings.json"
EDITABLE = ("WATERMARK", "WATERMARK_OPACITY", "WATERMARK_Y", "TITLE", "ZOOM", "SHORT_MAX", "LONG_MAX", "CUT_SILENCE", "RUSH_BEFORE", "RUSH_AFTER",
            "VISION", "VISION_CLIPS", "VISION_MIN", "CAM_BONUS", "TREND_GAME", "MISSION_RECAPS", "PROMPT_CONTEXT")
DEFAULTS = {k: globals()[k] for k in EDITABLE}


def _apply_settings() -> None:
    try:
        engine = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).get("engine", {})
    except (OSError, ValueError):
        return
    g = globals()
    for k, v in engine.items():
        if k not in EDITABLE:
            continue
        cur = DEFAULTS[k]
        if isinstance(cur, bool):
            ok = isinstance(v, bool)
        elif isinstance(cur, (int, float)):
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            v = type(cur)(v) if ok else v
        else:
            ok = isinstance(v, str)
        if ok:
            g[k] = v


_apply_settings()
