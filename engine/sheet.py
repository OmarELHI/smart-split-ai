"""Outil de controle : planche d'images legendees avec la scene detectee (a partir de scenes.json).

Exemple (depuis D:\\Logiciels\\Smart Split) :
  python engine\\sheet.py "F:\\Videos stream\\mon-stream.mp4"
Resultat : data\\analysis\\<video>\\planche.jpg (120 images reparties sur toute la video)
"""
import sys
from pathlib import Path

from smartsplit import scenes

src = Path(sys.argv[1])
an = scenes.load(src)
if an is None:
    sys.exit("Pas d'analyse des scenes pour cette video : lance d'abord pipeline.py ou analyze.py.")
print(f"Planche : {scenes.contact_sheet(src, an)}")
