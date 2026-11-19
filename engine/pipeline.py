"""Etape 7 : traitement complet d'un stream en une seule commande.

  python engine\\pipeline.py "D:\\Enregistrements OBS\\mon-stream.mp4"

Phases : scenes (en parallele) | audio -> transcription -> reactions -> notation IA -> choix -> rendu.
Chaque phase enregistre son resultat : si le traitement est interrompu, la relance reprend
la ou il s'etait arrete (les clips deja rendus ne sont pas refaits).
Options : --max N (nombre de clips rendus), --no-render, --redo <phase> (refaire une phase), --force (tout refaire)
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from smartsplit import llm, reactions, scenes, transcribe
from smartsplit.audio import load_audio
from smartsplit.config import OUTPUT
from smartsplit.layout import render_segments
from smartsplit.select import select

PHASES = ["scenes", "transcription", "reactions", "notation", "choix", "rendu"]


def fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Traitement complet d'un stream")
    ap.add_argument("src", type=Path)
    ap.add_argument("--max", type=int, default=0, help="nombre maximum de clips rendus (0 = tous)")
    ap.add_argument("--no-render", action="store_true", help="s'arrete a la liste des clips")
    ap.add_argument("--redo", choices=PHASES, action="append", default=[], help="refait cette phase")
    ap.add_argument("--force", action="store_true", help="refait toutes les phases")
    args = ap.parse_args()
    src = args.src.resolve()
    if not src.exists():
        raise SystemExit(f"Fichier introuvable : {src}")
    redo = set(PHASES) if args.force else set(args.redo)
    work = scenes.analysis_dir(src)
    work.mkdir(parents=True, exist_ok=True)
    times: dict[str, float] = {}
    t_all = time.time()
    print(f"Traitement de {src.name}\n")

    # 1. Scenes : en parallele (CPU) pendant que le GPU transcrit
    def run_scenes():
        t0 = time.time()
        an = None if "scenes" in redo else scenes.load(src)
        if an is None:
            an = scenes.analyze(src, progress=lambda m: print(f"[scenes] {m.strip()}"))
            scenes.save(an)
        times["scenes"] = time.time() - t0
        return an

    pool = ThreadPoolExecutor(max_workers=1)
    fut_scenes = pool.submit(run_scenes)

    # 2. Transcription (GPU)
    t0 = time.time()
    tr = None if "transcription" in redo else transcribe.load(src)
    audio = None
    if tr is None:
        audio = load_audio(src)
        tr = transcribe.transcribe(src, audio=audio, progress=lambda m: print(f"[transcription] {m.strip()}"))
        transcribe.save(tr)
        tr = transcribe.load(src)          # recharge avec les corrections appliquees
        free_gpu()
    times["transcription"] = time.time() - t0
    print(f"[transcription] {len(tr['segments'])} phrases ({times['transcription']:.0f} s)")

    # 3. Reactions audio (CPU)
    t0 = time.time()
    rx = None if "reactions" in redo else reactions.load(src)
    if rx is None:
        audio = audio if audio is not None else load_audio(src)
        words = [w for s in tr["segments"] for w in s["words"]]
        rx = reactions.compute(audio, words)
        reactions.save(src, rx)
    del audio
    times["reactions"] = time.time() - t0
    print(f"[reactions] {len(rx['peaks'])} pics ({times['reactions']:.0f} s)")

    # 4. Notation par l'IA (GPU, Ollama)
    t0 = time.time()
    sc = None if "notation" in redo else llm.load(src)
    if sc is None:
        print(f"[notation] modele {llm.check()}")
        sc = llm.score(tr, progress=lambda m: print(f"[notation] {m.strip()}"))
        llm.save(src, sc)
        llm.unload()                       # libere le GPU pour le rendu
    times["notation"] = time.time() - t0
    print(f"[notation] {len(sc['blocks'])} extraits ({times['notation']:.0f} s)")

    # 5. Choix des clips
    t0 = time.time()
    clips = select(sc["blocks"], tr, rx, tr["duration"])
    (work / "clips.json").write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
    times["choix"] = time.time() - t0
    print(f"[choix] {len(clips)} clips proposes")

    an = fut_scenes.result()
    pool.shutdown()
    print(f"[scenes] {len(an.segments)} passages ({times['scenes']:.0f} s, en parallele)")

    # 6. Rendu
    rendered = []
    if not args.no_render:
        t0 = time.time()
        out_dir = OUTPUT / src.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = clips[: args.max] if args.max else clips
        for i, c in enumerate(todo, 1):
            st = int(c["start"])
            dst = out_dir / f"clip{i:02d}_{st // 60:02d}m{st % 60:02d}s_note{c['note']}.mp4"
            if dst.exists() and "rendu" not in redo:
                rendered.append(dst)
                continue
            t1 = time.time()
            tmp = dst.with_suffix(".part.mp4")
            render_segments(src, c["start"], c["end"], scenes.segments_for(an, c["start"], c["end"]), tmp,
                            face_x="auto")
            tmp.replace(dst)               # le fichier n'apparait qu'une fois complet
            rendered.append(dst)
            print(f"[rendu] {i}/{len(todo)} {dst.name} ({time.time() - t1:.0f} s)")
        times["rendu"] = time.time() - t0

    total = time.time() - t_all
    report = {"src": str(src), "duration": tr["duration"], "total_s": round(total, 1),
              "phases_s": {k: round(v, 1) for k, v in times.items()}, "clips": len(clips),
              "rendered": [str(p) for p in rendered]}
    (work / "pipeline.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n=== Termine en {fmt(total)} pour {fmt(tr['duration'])} de stream ===")
    for k in PHASES:
        if k in times:
            par = " (en parallele)" if k == "scenes" else ""
            print(f"  {k:14s} {times[k]:7.1f} s{par}")
    print("\nClips :")
    for i, c in enumerate(clips[: len(rendered) or len(clips)], 1):
        print(f"  {i:2d}. {fmt(c['start'])} -> {fmt(c['end'])}  note {c['note']}/10  {c['type']} : {c['raison'][:60]}")
    if rendered:
        print(f"\nDossier : {OUTPUT / src.stem}")


if __name__ == "__main__":
    main()
