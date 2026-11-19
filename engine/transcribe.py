"""Etape 4 : transcrit une video et enregistre le texte horodate au mot.

Exemple (depuis D:\\Logiciels\\Smart Split) :
  python engine\\transcribe.py "2026-11-19 16-11-36.mp4"
Resultats dans data\\analysis\\<nom de la video>\\ : transcript.json et transcript.txt (lisible)
"""
import argparse
from pathlib import Path

from smartsplit import transcribe as tr


def fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcription horodatee au mot")
    ap.add_argument("src", type=Path)
    ap.add_argument("--force", action="store_true", help="refait la transcription meme si elle existe")
    args = ap.parse_args()

    if args.force or tr.load(args.src) is None:
        res = tr.transcribe(args.src)
        out = tr.save(res)
        speed = res["duration"] / max(res["seconds"], 0.1)
        print(f"\n{res['duration'] / 60:.1f} min transcrites en {res['seconds']:.0f} s ({speed:.0f}x temps reel)")
    else:
        res = tr.load(args.src)
        out = tr.transcript_path(args.src)
        print("Transcription deja faite (ajoute --force pour la refaire)")

    words = sum(len(s["words"]) for s in res["segments"])
    txt = out.with_suffix(".txt")
    txt.write_text("\n".join(f"[{fmt(s['start'])}] {s['text']}" for s in res["segments"]), encoding="utf-8")
    print(f"{len(res['segments'])} phrases, {words} mots")
    print(f"Fichiers : {out}\n           {txt}")


if __name__ == "__main__":
    main()
