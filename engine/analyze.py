"""Etape 2 : detecte les scenes d'un enregistrement et produit la chronologie + une planche de controle.

Exemple (depuis D:\\Logiciels\\Smart Split) :
  python engine\\analyze.py "2026-11-19 16-11-36.mp4"
Resultats dans data\\analysis\\<nom de la video>\\ : scenes.json et planche.jpg
"""
import argparse
import time
from pathlib import Path

from smartsplit import scenes


def fmt(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Detection des scenes OBS")
    ap.add_argument("src", type=Path)
    ap.add_argument("--no-sheet", action="store_true", help="ne pas produire la planche d'images")
    args = ap.parse_args()

    t0 = time.time()
    an = scenes.analyze(args.src)
    out = scenes.save(an)
    print(f"\nAnalyse en {time.time() - t0:.0f} s pour {an.duration / 60:.1f} min de video")
    print(f"Positions de cam trouvees : {len(an.cams)}")
    for i, c in enumerate(an.cams):
        print(f"  #{i} : x {c[0]:.3f}  y {c[1]:.3f}  ({'droite' if c[0] > 0.5 else 'gauche'})")
    print("\nChronologie :")
    for s in an.segments:
        cam = ""
        if s["cam"]:
            cam = f"  cam {'droite' if s['cam'][0] > 0.5 else 'gauche'}"
        print(f"  {fmt(s['start'])} -> {fmt(s['end'])}  {s['scene']}{cam}")
    print(f"\nChronologie : {out}")
    if not args.no_sheet:
        print(f"Planche     : {scenes.contact_sheet(args.src, an)}")


if __name__ == "__main__":
    main()
