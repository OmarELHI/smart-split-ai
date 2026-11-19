"""Etape 6 : l'IA note les propos du stream et propose les clips.

Exemples (depuis D:\\Logiciels\\Smart Split) :
  python engine\\clips.py "2026-11-19 16-11-36.mp4"              # notation + liste des clips
  python engine\\clips.py "2026-11-19 16-11-36.mp4" --render 3   # + rendu des 3 meilleurs
Il faut avoir lance transcribe.py (et si possible reactions.py et analyze.py) sur la video.
"""
import argparse
import json
import time
from pathlib import Path

from smartsplit import llm, reactions, scenes, transcribe
from smartsplit.config import OUTPUT
from smartsplit.layout import render_segments
from smartsplit.select import select


def fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Notation par l'IA et choix des clips")
    ap.add_argument("src", type=Path)
    ap.add_argument("--rescore", action="store_true", help="refait la notation meme si elle existe")
    ap.add_argument("--render", type=int, default=0, help="rend les N meilleurs clips en 9:16")
    args = ap.parse_args()

    tr = transcribe.load(args.src)
    if tr is None:
        raise SystemExit("Lance d'abord engine\\transcribe.py sur cette video.")
    sc = llm.load(args.src)
    if sc is None or args.rescore:
        print(f"Modele : {llm.check()}")
        sc = llm.score(tr)
        llm.save(args.src, sc)
        llm.unload()
        print(f"Notation en {sc['seconds']:.0f} s\n")

    print("Notes par extrait :")
    for b in sc["blocks"]:
        print(f"  {fmt(b['start'])}-{fmt(b['end'])}  {b['note']:2d}/10  {b['type']:8s} {b['raison'][:70]}")

    clips = select(sc["blocks"], tr, reactions.load(args.src), tr["duration"])
    out = llm.scores_path(args.src).with_name("clips.json")
    out.write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nClips proposes ({len(clips)}) :")
    for i, c in enumerate(clips, 1):
        print(f"  {i:2d}. {fmt(c['start'])} -> {fmt(c['end'])} ({c['end'] - c['start']:.0f} s)  "
              f"note {c['note']}/10  {c['type']} : {c['raison'][:60]}")
    print(f"\nListe : {out}")

    if args.render:
        an = scenes.load(args.src)
        if an is None:
            print("\nAnalyse des scenes (une seule fois par video)...")
            an = scenes.analyze(args.src)
            scenes.save(an)
        for i, c in enumerate(clips[: args.render], 1):
            dst = OUTPUT / f"{args.src.stem}_clip{i:02d}_{int(c['start'])}-{int(c['end'])}.mp4"
            t0 = time.time()
            render_segments(args.src, c["start"], c["end"], scenes.segments_for(an, c["start"], c["end"]), dst,
                            face_x="auto")
            print(f"[OK] clip {i} : {dst} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
