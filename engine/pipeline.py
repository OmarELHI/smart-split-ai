"""Etape 7 : traitement complet d'un stream en une seule commande.

  python engine\\pipeline.py "D:\\Enregistrements OBS\\mon-stream.mp4"

Phases : scenes (en parallele) | transcription -> reactions -> vision IA (ce qui se passe a l'ecran)
-> notation IA -> choix -> montage IA -> rendu.
Chaque phase enregistre son resultat : si le traitement est interrompu, la relance reprend
la ou il s'etait arrete (les clips deja rendus ne sont pas refaits).
Options : --max N (nombre de clips rendus), --no-render, --redo <phase> (refaire une phase), --force (tout refaire)
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from smartsplit import edit, llm, missions, reactions, scenes, transcribe, trends, vision
from smartsplit.audio import load_audio
from smartsplit.config import CAM_BONUS, MISSION_RECAPS, OUTPUT, VISION, VISION_CLIPS, VISION_MIN, VISION_MODEL
from smartsplit.select import select

PHASES = ["scenes", "transcription", "reactions", "vision", "notation", "choix", "montage", "rendu"]


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
    if "scenes" in redo:
        redo.add("vision")                 # nouvelles images cles
    if "vision" in redo:
        redo |= {"notation", "montage"}    # la notation et le montage utilisent la vision
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
            an = scenes.analyze(src, progress=lambda m: print(f"[scenes] {m.strip()}"),
                                frames_dir=vision.frames_dir(src) if VISION else None)
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
        # Whisper tourne dans un processus a part : a sa fin, toute la memoire du GPU est rendue.
        # (dans le meme processus, CTranslate2 garde sa memoire et Ollama, a l'etroit, devient tres lent)
        cmd = [sys.executable, "-u", str(Path(__file__).with_name("transcribe.py")), str(src), "--force"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            if line.strip():
                print(f"[transcription] {line.strip()}")
        if p.wait() != 0:
            raise SystemExit("La transcription a echoue (voir les messages ci-dessus).")
        tr = transcribe.load(src)
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

    an = fut_scenes.result()
    pool.shutdown()
    print(f"[scenes] {len(an.segments)} passages ({times['scenes']:.0f} s, en parallele)")

    # 4. Vision : ce qui se passe a l'ecran (GPU, Ollama, gemma3 lit les images)
    vis = None
    if VISION:
        t0 = time.time()
        vis = None if "vision" in redo else vision.load(src)
        if vis is None:
            llm.check()
            partial = vision.load(src, partial=True) if "vision" not in redo else None   # reprise apres coupure
            try:
                vis = vision.analyze(src, an.segments, an.duration, resume=partial,
                                     progress=lambda m: print(f"[vision] {m.strip()}"),
                                     save_every=lambda w: vision.save(src, w, complete=False))
                vision.save(src, vis)
            except vision.VisionUnavailable as e:
                print(f"[vision] ABANDON : {e}. Suite du traitement sans la vision.")
                vis = []
            llm.unload(VISION_MODEL)       # libere le GPU pour gemma3 (notation, montage)
            redo |= {"notation", "montage"}
        times["vision"] = time.time() - t0
        strong = sum(1 for w in vis if w["intensite"] >= VISION_MIN)
        print(f"[vision] {len(vis)} fenetres decrites, {strong} actions fortes ({times['vision']:.0f} s)")

    # 5. Tendances TikTok / YouTube Shorts autour du jeu (mises a jour tous les quelques jours)
    try:
        trends.update(progress=lambda m: print(f"[tendances] {m.strip()}"))
    except Exception as e:  # noqa: BLE001
        print(f"[tendances] mise a jour impossible ({e}) : on continue sans")

    # 6. Notation par l'IA (GPU, Ollama)
    t0 = time.time()
    sc = None if "notation" in redo else llm.load(src)
    if sc is not None and VISION and not sc.get("vision"):
        sc = None                          # ancienne notation faite sans la vision : on la refait
        redo.add("montage")
    if sc is None:
        print(f"[notation] modele {llm.check()}")
        sc = llm.score(tr, progress=lambda m: print(f"[notation] {m.strip()}"), vision=vis)
        sc["vision"] = bool(vis)
        llm.save(src, sc)
    times["notation"] = time.time() - t0
    print(f"[notation] {len(sc['blocks'])} extraits ({times['notation']:.0f} s)")

    # 5. Choix des clips
    t0 = time.time()
    cand = sc["blocks"] + (vision.action_blocks(vis, VISION_MIN) if vis and VISION_CLIPS else [])
    events = vision.reaction_events(vis, rx) if vis else []      # evenement a l'ecran + ta reaction
    cand += events
    if events:
        print(f"[choix] {len(events)} evenements a l'ecran suivis d'une reaction")
    cand.sort(key=lambda b: b["start"])
    clips = select(cand, tr, rx, tr["duration"])
    for c in clips:                        # priorite aux clips ou ta cam est visible
        seg = scenes.segments_for(an, c["start"], c["end"])
        cam = sum(min(e1, c["end"]) - max(s1, c["start"]) for s1, e1, sc_, _ in seg if sc_ in ("cam_jeu", "full_cam"))
        c["cam"] = round(cam / max(1e-6, c["end"] - c["start"]), 2)
        # un evenement a l'ecran se passe souvent en jeu seul : meme bonus que la cam, pour ne pas le defavoriser
        c["score"] = round(c["score"] + CAM_BONUS * (1.0 if c.get("evenement") else c["cam"]), 4)
    clips.sort(key=lambda c: -c["score"])
    (work / "clips.json").write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
    times["choix"] = time.time() - t0
    print(f"[choix] {len(clips)} clips proposes")

    # 7. Montage : dans un rush plus long autour de chaque moment, l'IA garde les bons passages,
    #    coupe les temps morts, place les zooms et ecrit le titre d'accroche
    t0 = time.time()
    mpath = work / "clips_montes.json"
    cached = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else None
    if cached and all("parts" in c for c in cached) and not ({"montage", "choix", "notation"} & redo):
        clips = cached
    else:
        clips = [{**c, "parts": [{"start": c["start"], "end": c["end"], "zoom": False}], "monte": False}
                 for c in sorted(clips, key=lambda c: -c["score"])]
    # seuls les meilleurs moments sont montes par l'IA (2 fois le nombre de clips a rendre) : le reste
    # le sera a la demande (bouton "Rendre 10 de plus" de l'application)
    limit = args.max * 2 if args.max else len(clips)
    todo_plan = [i for i, c in enumerate(clips[:limit]) if c.get("monte") is False]
    if todo_plan:
        llm.check()
        planned = edit.plan_all([clips[i] for i in todo_plan], tr, tr["duration"], vision=vis,
                                progress=lambda m: print(f"[montage] {m.strip()}"))
        for i, c in zip(todo_plan, planned):
            clips[i] = {**c, "monte": True}
        kept: list[dict] = []
        for c in clips:                    # supprime les chevauchements crees par le montage
            if not any(c["start"] < k["end"] and k["start"] < c["end"] for k in kept):
                kept.append(c)
        clips = kept
        mpath.write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
    # resumes de missions (ajoutes apres les clips, toujours rendus)
    clips = [c for c in clips if c.get("type") != "resume"]
    if MISSION_RECAPS and vis:
        rec = missions.recaps(vis, sc["blocks"], events, tr["duration"])
        clips += rec
        mpath.write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[montage] {len(rec)} resumes de missions : " + ", ".join(r["titre"] for r in rec))
    llm.unload()                           # libere le GPU pour le rendu
    times["montage"] = time.time() - t0
    print(f"[montage] {len(clips)} clips ({times['montage']:.0f} s)")

    # 8. Rendu
    rendered = []
    if not args.no_render:
        t0 = time.time()
        out_dir = OUTPUT / src.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = clips[: args.max] if args.max else clips
        todo += [c for c in clips if c.get("type") == "resume" and c not in todo]
        for c in todo:
            i = clips.index(c) + 1
            st = int(c["start"])
            base = f"clip{i:02d}_{st // 60:02d}m{st % 60:02d}s_note{c['note']}"
            versions = [(base, c["parts"])]
            if c.get("court_parts"):       # montage long : version courte en plus
                versions.append((base + "_court", c["court_parts"]))
            for name, parts in versions:
                dst = out_dir / f"{name}.mp4"
                if dst.exists() and "rendu" not in redo:
                    rendered.append(dst)
                    continue
                t1 = time.time()
                tmp = dst.with_suffix(".part.mp4")
                edit.render(src, parts, an, tmp, title=c.get("titre", ""))
                tmp.replace(dst)           # le fichier n'apparait qu'une fois complet
                rendered.append(dst)
                print(f"[rendu] {i}/{len(todo)} {dst.name} ({edit.total(parts):.0f} s, "
                      f"{len(parts)} passages, rendu en {time.time() - t1:.0f} s)")
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
    for i, c in enumerate(clips[: args.max or len(clips)], 1):
        court = f" + courte {edit.total(c['court_parts']):.0f} s" if c.get("court_parts") else ""
        print(f"  {i:2d}. {fmt(c['start'])}  {c.get('duree', 0):.0f} s en {len(c.get('parts', []))} passages{court}"
              f"  note {c['note']}/10  {c['type']}  \"{c.get('titre', '')}\"")
    if rendered:
        print(f"\nDossier : {OUTPUT / src.stem}")


if __name__ == "__main__":
    main()
