"""Etape 1 : rend un clip vertical a partir d'un extrait et d'une scene imposee.

Exemples (depuis D:\\Logiciels\\Smart Split) :
  python engine\\render.py "2026-11-19 16-18-24.mp4" --start 60 --end 90 --scene cam_jeu
  python engine\\render.py "2026-11-19 16-18-24.mp4" --start 60 --scene cam_jeu --game-ratio 1:1 --preview
  python engine\\render.py "2026-11-19 16-11-36.mp4" --start 30 --end 50 --scene full_cam --face-x 0.6
  python engine\\render.py "2026-11-19 16-11-36.mp4" --start 0 --end 40 --scene full_cam   (suivi du visage)
  python engine\\render.py "2026-11-19 16-18-24.mp4" --demo
  python engine\\render.py "2026-11-19 16-11-36.mp4" --start 100 --end 160 --auto
"""
import argparse
import time
from pathlib import Path

from smartsplit.config import BOTTOM_MARGIN, GAME_RATIO, OUTPUT, Rect
from smartsplit import scenes
from smartsplit.layout import Scene, probe, render_clip, render_segments


def ratio(text: str) -> float:
    """Lit "4:3", "16:9" ou "1.33"."""
    if ":" in text:
        a, b = text.split(":")
        return float(a) / float(b)
    return float(text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rendu d'un clip 1080x1920")
    ap.add_argument("src", type=Path, help="video source")
    ap.add_argument("--start", type=float, default=0, help="debut en secondes")
    ap.add_argument("--end", type=float, default=30, help="fin en secondes")
    ap.add_argument("--scene", choices=[s.value for s in Scene], default=Scene.CAM_JEU.value)
    ap.add_argument("--cam", help="zone de la cam x,y,w,h (pixels ou relatif). Par defaut : en haut a droite")
    ap.add_argument("--face-x", type=lambda v: v if v == "auto" else float(v), default="auto",
                    help="full cam : 'auto' (suivi du visage, par defaut) ou position fixe de 0 a 1")
    ap.add_argument("--game-ratio", type=ratio, default=GAME_RATIO, help="cam + jeu : ratio de la zone jeu (1:1, 4:3, 16:9)")
    ap.add_argument("--bottom-margin", type=float, default=BOTTOM_MARGIN, help="cam + jeu : espace flou sous le jeu (0 a 0.3)")
    ap.add_argument("--preview", action="store_true", help="produit une image PNG au lieu d'une video")
    ap.add_argument("--auto", action="store_true", help="scene et position de cam detectees automatiquement")
    ap.add_argument("--demo", action="store_true", help="rend les 3 scenes sur le meme passage (60 a 75 s)")
    ap.add_argument("--cpu", action="store_true", help="encode sans le GPU (libx264)")
    ap.add_argument("--out", type=Path, help="fichier de sortie")
    args = ap.parse_args()

    if args.end <= args.start:
        args.end = args.start + (1 if args.preview else 30)
    if args.auto:
        an = scenes.load(args.src)
        if an is None:
            print("Analyse des scenes (une seule fois par video)...")
            an = scenes.analyze(args.src)
            scenes.save(an)
        pieces = scenes.segments_for(an, args.start, args.end)
        out = args.out or OUTPUT / f"{args.src.stem}_auto_{int(args.start)}-{int(args.end)}.mp4"
        t0 = time.time()
        render_segments(args.src, args.start, args.end, pieces, out, face_x=args.face_x,
                        encoder="libx264" if args.cpu else "h264_nvenc",
                        game_ratio=args.game_ratio, bottom_margin=args.bottom_margin)
        for a, b, sc, cam in pieces:
            pos = f" (cam {'droite' if cam.x > 0.5 else 'gauche'})" if cam else ""
            print(f"  {a:7.2f} -> {b:7.2f}  {sc}{pos}")
        print(f"[OK] auto : {out} ({time.time() - t0:.1f} s)")
        return

    info = probe(args.src)
    cam = Rect.parse(args.cam, info["width"], info["height"]) if args.cam else None
    encoder = "libx264" if args.cpu else "h264_nvenc"
    jobs = ([(s, 60.0, 75.0) for s in Scene] if args.demo
            else [(Scene(args.scene), args.start, args.end)])
    ext = ".png" if args.preview else ".mp4"

    for scene, start, end in jobs:
        name = f"{args.src.stem}_{scene.value}_{int(start)}-{int(end)}{ext}"
        out = args.out if (args.out and not args.demo) else OUTPUT / name
        t0 = time.time()
        kwargs = {"face_x": args.face_x, "encoder": encoder,
                  "game_ratio": args.game_ratio, "bottom_margin": args.bottom_margin}
        if cam:
            kwargs["cam"] = cam
        render_clip(args.src, start, end, scene, out, **kwargs)
        print(f"[OK] {scene.value} : {out} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
