"""Etape 5 : calcule le score de reaction de chaque seconde et liste les pics.

Exemple (depuis D:\\Logiciels\\Smart Split) :
  python engine\\reactions.py "2026-11-19 16-11-36.mp4"
Utilise la transcription si elle existe (lance engine\\transcribe.py avant pour un meilleur resultat).
Resultats dans data\\analysis\\<video>\\ : reactions.json et reactions.png (courbe + pics numerotes)
"""
import argparse
import time
from pathlib import Path

from smartsplit import reactions, transcribe
from smartsplit.audio import load_audio


def fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Score de reaction audio")
    ap.add_argument("src", type=Path)
    args = ap.parse_args()

    t0 = time.time()
    tr = transcribe.load(args.src)
    words = [w for s in tr["segments"] for w in s["words"]] if tr else None
    if not tr:
        print("Pas de transcription : score calcule sur le son seul")
    audio = load_audio(args.src)
    res = reactions.compute(audio, words)
    out = reactions.save(args.src, res)
    png = reactions.chart(args.src, res)
    print(f"{len(res['score']) / 60:.1f} min analysees en {time.time() - t0:.1f} s\n")
    print("Pics de reaction (du plus fort au moins fort) :")
    segs = tr["segments"] if tr else []
    for rank, p in enumerate(sorted(res["peaks"], key=lambda p: -p["score"]), 1):
        said = " ".join(s["text"] for s in segs if s["end"] >= p["t"] - 4 and s["start"] <= p["t"] + 4)
        print(f"  {rank:2d}. {fmt(p['t'])}  score {p['score']:.2f}  {said[:90]}")
    print(f"\nFichiers : {out}\n           {png}")


if __name__ == "__main__":
    main()
