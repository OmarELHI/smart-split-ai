"""Etape 3 : verifie la detection du visage sur les passages full cam d'une video.

Exemple (depuis D:\\Logiciels\\Smart Split) :
  python engine\\face_check.py "2026-11-19 16-11-36.mp4"
Telecharge le modele au premier lancement, puis produit data\\analysis\\<video>\\visages.jpg
(une image par seconde des passages full cam, visage encadre en vert, cadrage 9:16 en rose).
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from smartsplit import scenes
from smartsplit.face import DH, DW, FaceDetector, _frames, smooth, track

src = Path(sys.argv[1])
an = scenes.load(src)
if an is None:
    sys.exit("Lance d'abord engine\\analyze.py sur cette video.")
parts = [(s["start"], s["end"]) for s in an.segments if s["scene"] == "full_cam"]
if not parts:
    sys.exit("Aucun passage full cam dans cette video.")

t0 = time.time()
det = FaceDetector()
print(f"Modele charge en {time.time() - t0:.1f} s")

tiles, found, total = [], 0, 0
crop_w = DH * 9 / 16 / DW          # largeur du cadrage 9:16 en part de l'image
for a, b in parts:
    b = min(b, a + 60)              # 60 s max par passage pour le controle
    t1 = time.time()
    pts = track(src, a, b, det)
    keys = dict(smooth(pts, b - a))
    print(f"  {a:6.1f} -> {b:6.1f} s : visage trouve {sum(c is not None for _, c in pts)}/{len(pts)} "
          f"({time.time() - t1:.1f} s)")
    for (t, img), (_, c) in zip(_frames(src, a, b, 1.0), [(p[0], p[1]) for p in pts][::5]):
        img = img.copy()
        f = det.largest(img)
        total += 1
        if f:
            found += 1
            x, y, w, h, sc = f
            cv2.rectangle(img, (int(x * DW), int(y * DH)), (int((x + w) * DW), int((y + h) * DH)), (0, 220, 0), 2)
        k = min(keys, key=lambda kt: abs(kt - t))
        cx = keys[k]
        x0 = int(min(max(cx - crop_w / 2, 0), 1 - crop_w) * DW)
        cv2.rectangle(img, (x0, 0), (x0 + int(crop_w * DW), DH - 1), (200, 60, 255), 2)
        cv2.putText(img, f"{a + t:.0f} s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (320, 180)))

cols = 6
tiles = tiles[:96]
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
out = scenes.analysis_dir(src) / "visages.jpg"
cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
print(f"\nVisage trouve sur {found}/{total} images de controle")
print(f"Planche : {out}")
