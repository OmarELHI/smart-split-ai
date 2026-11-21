"""Smart Split AI : application de bureau (serveur local + fenetre d'application).

Lancement : double-clic sur "Smart Split.bat" (ou le raccourci du bureau cree par install_shortcut.ps1).
Le serveur ecoute sur http://127.0.0.1:8765 et ouvre l'interface dans une fenetre d'application Edge.
Il s'arrete tout seul quand la fenetre est fermee depuis 2 minutes et qu'aucun traitement n'est en cours.

Aucune dependance en plus du moteur : bibliotheque standard de Python uniquement.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import webbrowser
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

APP = Path(__file__).resolve().parent
ROOT = APP.parent
ENGINE = ROOT / "engine"
sys.path.insert(0, str(ENGINE))

from smartsplit import config  # noqa: E402  (leger : aucune dependance lourde)

DATA = config.DATA
OUTPUT = config.OUTPUT
ANALYSIS = DATA / "analysis"
FEEDBACK_FILE = DATA / "feedback.json"
SETTINGS_FILE = config.SETTINGS_FILE
LOG_FILE = DATA / "app.log"
HOST, PORT = "127.0.0.1", 8765
VENV_PY = ENGINE / ".venv" / "Scripts" / "python.exe"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
VIDEO_EXT = {".mp4", ".mkv", ".mov"}
PHASES = ["scenes", "transcription", "reactions", "vision", "notation", "choix", "montage", "rendu"]
# Poids de chaque phase dans la barre de progression (durees mesurees sur un stream de 5 h)
WEIGHTS = {"scenes": 0, "transcription": 270, "reactions": 5, "vision": 690, "notation": 600,
           "choix": 1, "montage": 110, "rendu": 70}
APP_DEFAULTS = {"vod_dir": r"F:\Vidéos stream", "max_clips": 10}

lock = threading.RLock()
last_ping = time.time() + 60          # delai de grace au demarrage


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    try:
        print(line, flush=True)
    except Exception:  # noqa: BLE001  (pythonw : pas de console)
        pass
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def settings() -> dict:
    s = read_json(SETTINGS_FILE, {})
    app = {**APP_DEFAULTS, **s.get("app", {})}
    return {"app": app, "engine": s.get("engine", {})}


def inside(path: Path, *roots: Path) -> bool:
    try:
        p = path.resolve()
    except OSError:
        return False
    for r in roots:
        try:
            p.relative_to(r.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


# ----------------------------------------------------------------------------- traitements (file d'attente)

class Job:
    def __init__(self, src: str, max_clips: int, no_render: bool, redo: list[str]):
        self.id = uuid.uuid4().hex[:8]
        self.src = src
        self.name = Path(src).name
        self.max_clips = max_clips
        self.no_render = no_render
        self.redo = [r for r in redo if r in PHASES]
        self.status = "attente"            # attente, en_cours, termine, erreur, annule
        self.lines: deque[str] = deque(maxlen=3000)
        self.n_lines = 0
        self.phases = {p: {"etat": "attente", "msg": ""} for p in PHASES}
        self.render = (0, 0)
        self.created = time.time()
        self.started = 0.0
        self.ended = 0.0
        self.proc: subprocess.Popen | None = None
        self.error = ""

    def cmd(self) -> list[str]:
        c = [PY, "-u", str(ENGINE / "pipeline.py"), self.src]
        if self.max_clips:
            c += ["--max", str(self.max_clips)]
        if self.no_render:
            c.append("--no-render")
        for r in self.redo:
            c += ["--redo", r]
        return c

    def feed(self, line: str) -> None:
        self.lines.append(line)
        self.n_lines += 1
        m = re.match(r"\[(\w+)\]\s*(.*)", line)
        if not m or m.group(1) not in self.phases:
            return
        ph, msg = m.group(1), m.group(2)
        st = self.phases[ph]
        st["msg"] = msg
        if st["etat"] == "attente":
            st["etat"] = "en_cours"
        if ph != "scenes":                  # les phases suivantes se font dans l'ordre
            for p in PHASES[1:PHASES.index(ph)]:
                if self.phases[p]["etat"] != "fait":
                    self.phases[p]["etat"] = "fait"
        done = {"scenes": r"passages \(", "transcription": r"phrases \(", "reactions": r"pics \(",
                "vision": r"fenetres decrites|ABANDON", "notation": r"extraits \(", "choix": r"clips proposes",
                "montage": r"^\d+ clips \("}
        if ph in done and re.search(done[ph], msg):
            st["etat"] = "fait"
        if ph == "rendu":
            r = re.match(r"(\d+)/(\d+)", msg)
            if r:
                self.render = (int(r.group(1)), int(r.group(2)))

    def progress(self) -> float:
        if self.status == "termine":
            return 1.0
        total = sum(WEIGHTS[p] for p in PHASES if not (self.no_render and p == "rendu"))
        got = 0.0
        for p in PHASES:
            if self.no_render and p == "rendu":
                continue
            st = self.phases[p]["etat"]
            if st == "fait":
                got += WEIGHTS[p]
            elif st == "en_cours":
                frac = 0.4
                if p == "rendu" and self.render[1]:
                    frac = self.render[0] / self.render[1]
                got += WEIGHTS[p] * frac
        return min(0.99, got / total) if total else 0.0

    def to_dict(self, with_log: int | None = None) -> dict:
        d = {"id": self.id, "src": self.src, "name": self.name, "status": self.status,
             "max_clips": self.max_clips, "no_render": self.no_render, "redo": self.redo,
             "phases": self.phases, "progress": round(self.progress(), 3),
             "created": self.created, "started": self.started, "ended": self.ended,
             "error": self.error, "n_lines": self.n_lines, "stem": Path(self.src).stem}
        if with_log is not None:
            skip = max(0, with_log - (self.n_lines - len(self.lines)))
            d["log"] = list(self.lines)[skip:]
        return d


jobs: list[Job] = []
wake = threading.Event()


def worker() -> None:
    while True:
        wake.wait(2)
        wake.clear()
        with lock:
            job = next((j for j in jobs if j.status == "attente"), None)
            if not job or any(j.status == "en_cours" for j in jobs):
                continue
            job.status = "en_cours"
            job.started = time.time()
        log(f"traitement : {job.src}")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        try:
            job.proc = subprocess.Popen(job.cmd(), cwd=str(ENGINE), stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                                        encoding="utf-8", errors="replace", env=env, creationflags=NO_WINDOW)
            for line in job.proc.stdout:
                line = line.rstrip()
                if line:
                    with lock:
                        job.feed(line)
            code = job.proc.wait()
        except Exception as e:  # noqa: BLE001
            code = -1
            job.error = str(e)
        with lock:
            job.ended = time.time()
            if job.status == "annule":
                pass
            elif code == 0:
                job.status = "termine"
                for p in job.phases.values():
                    if p["etat"] == "en_cours":
                        p["etat"] = "fait"
            else:
                job.status = "erreur"
                tail = [l for l in list(job.lines)[-12:] if l.strip()]
                job.error = job.error or (tail[-1] if tail else f"code {code}")
        log(f"fin ({job.status}) : {job.src}")
        wake.set()


def cancel(job: Job) -> None:
    with lock:
        if job.status == "attente":
            job.status = "annule"
            return
        if job.status != "en_cours" or not job.proc:
            return
        job.status = "annule"
    if os.name == "nt":                     # arrete aussi la transcription lancee par le pipeline
        subprocess.run(["taskkill", "/PID", str(job.proc.pid), "/T", "/F"], capture_output=True,
                       creationflags=NO_WINDOW)
    else:
        job.proc.kill()


# ----------------------------------------------------------------------------- donnees : VOD, streams, clips

def list_vods() -> dict:
    d = Path(settings()["app"]["vod_dir"])
    out = []
    if d.is_dir():
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in VIDEO_EXT:
                st = f.stat()
                rep = read_json(ANALYSIS / f.stem / "pipeline.json", None)
                out.append({"path": str(f), "name": f.name, "size": st.st_size, "mtime": st.st_mtime,
                            "traite": bool(rep), "clips": (rep or {}).get("clips", 0)})
    out.sort(key=lambda v: -v["mtime"])
    return {"dir": str(d), "exists": d.is_dir(), "vods": out}


def parts_total(parts) -> float:
    return round(sum(max(0.0, p["end"] - p["start"]) for p in parts or []), 1)


def clip_list(work: Path) -> list[dict]:
    for name in ("clips_montes.json", "clips.json"):
        c = read_json(work / name, None)
        if isinstance(c, list):
            return c
    return []


def stream_src(stem: str, rep: dict | None) -> str:
    if rep and rep.get("src") and Path(rep["src"]).exists():
        return rep["src"]
    for d in (Path(settings()["app"]["vod_dir"]), ROOT):
        for ext in VIDEO_EXT:
            p = d / f"{stem}{ext}"
            if p.exists():
                return str(p)
    return (rep or {}).get("src", "")


def list_streams() -> list[dict]:
    fb = read_json(FEEDBACK_FILE, {})
    out = []
    if not ANALYSIS.is_dir():
        return out
    for work in ANALYSIS.iterdir():
        if not work.is_dir():
            continue
        rep = read_json(work / "pipeline.json", None)
        clips = clip_list(work)
        files = [f for f in (OUTPUT / work.name).glob("clip*.mp4") if not f.name.endswith(".part.mp4")] \
            if (OUTPUT / work.name).is_dir() else []
        decided = [fb[str(f)]["decision"] for f in files if str(f) in fb]
        out.append({"stem": work.name, "src": stream_src(work.name, rep),
                    "duration": (rep or {}).get("duration", 0), "total_s": (rep or {}).get("total_s", 0),
                    "moments": len(clips), "rendus": len(files),
                    "publies": decided.count("publie"), "rejetes": decided.count("rejete"),
                    "complet": bool(rep), "mtime": work.stat().st_mtime})
    out.sort(key=lambda s: s["stem"], reverse=True)
    return out


def stream_clips(stem: str) -> dict:
    work = ANALYSIS / stem
    if not inside(work, ANALYSIS) or not work.is_dir():
        raise FileNotFoundError(stem)
    rep = read_json(work / "pipeline.json", None)
    clips = clip_list(work)
    fb = read_json(FEEDBACK_FILE, {})
    out_dir = OUTPUT / stem
    files: dict[int, dict] = {}
    if out_dir.is_dir():
        for f in sorted(out_dir.glob("clip*.mp4")):
            m = re.match(r"clip(\d+)_(\d+)m(\d+)s_note(\d+)(_court)?\.mp4$", f.name)
            if not m:
                continue
            idx, st = int(m.group(1)) - 1, int(m.group(2)) * 60 + int(m.group(3))
            # retrouve le clip par son debut (l'ordre de la liste peut changer si le montage a ete refait)
            k = next((i for i, c in enumerate(clips) if int(c.get("start", -1)) == st), None)
            if k is None and 0 <= idx < len(clips):
                k = idx
            if k is None:
                continue
            files.setdefault(k, {})["court" if m.group(5) else "long"] = f
    items = []
    for i, c in enumerate(clips):
        vers = []
        for kind, parts_key in (("long", "parts"), ("court", "court_parts")):
            f = files.get(i, {}).get(kind)
            if kind == "court" and not f and not c.get(parts_key):
                continue
            dur = parts_total(c.get(parts_key)) if c.get(parts_key) else round(c.get("duree") or
                                                                                (c["end"] - c["start"]), 1)
            v = {"kind": kind, "duree": dur, "file": str(f) if f else "", "size": f.stat().st_size if f else 0}
            if f:
                v["url"] = "/media?path=" + quote(str(f))
                v["feedback"] = fb.get(str(f))
            vers.append(v)
        items.append({"index": i, "start": c.get("start", 0), "end": c.get("end", 0), "note": c.get("note"),
                      "type": c.get("type", ""), "titre": c.get("titre", ""), "raison": c.get("raison", ""),
                      "text": c.get("text", ""), "score": c.get("score"), "passages": len(c.get("parts") or []),
                      "versions": vers, "rendu": any(v.get("file") for v in vers)})
    return {"stem": stem, "src": stream_src(stem, rep), "report": rep, "out_dir": str(out_dir),
            "clips": items, "rendus": sum(1 for it in items if it["rendu"])}


def set_feedback(body: dict) -> dict:
    f = Path(body.get("file", ""))
    if not inside(f, OUTPUT) or not f.exists():
        raise FileNotFoundError("clip introuvable")
    fb = read_json(FEEDBACK_FILE, {})
    if body.get("decision") in (None, "", "aucune"):
        fb.pop(str(f), None)
    else:
        if body["decision"] not in ("publie", "rejete"):
            raise ValueError("decision inconnue")
        fb[str(f)] = {"decision": body["decision"], "raison": (body.get("raison") or "").strip(),
                      "stream": body.get("stem", ""), "start": body.get("start"), "titre": body.get("titre", ""),
                      "type": body.get("type", ""), "text": body.get("text", ""), "duree": body.get("duree"),
                      "note": body.get("note"), "date": datetime.now().isoformat(timespec="seconds")}
    write_json(FEEDBACK_FILE, fb)
    return {"ok": True}


def feedback_stats() -> dict:
    fb = read_json(FEEDBACK_FILE, {})
    vals = list(fb.values())
    return {"publies": sum(v["decision"] == "publie" for v in vals),
            "rejetes": sum(v["decision"] == "rejete" for v in vals),
            "raisons": sorted({v["raison"] for v in vals if v.get("raison")})}


# ----------------------------------------------------------------------------- outils systeme

def pick_file(initial: str) -> str:
    code = ("import sys,tkinter as tk\nfrom tkinter import filedialog\nr=tk.Tk();r.withdraw();"
            "r.attributes('-topmost',True)\n"
            "p=filedialog.askopenfilename(title='Choisir un enregistrement',initialdir=sys.argv[1],"
            "filetypes=[('Videos','*.mp4 *.mkv *.mov'),('Tous les fichiers','*.*')])\nprint(p or '')")
    r = subprocess.run([PY, "-c", code, initial], capture_output=True, text=True, encoding="utf-8",
                       creationflags=NO_WINDOW, timeout=600)
    if r.returncode != 0:
        raise RuntimeError("selecteur de fichier indisponible : colle le chemin du fichier a la place")
    return str(Path(r.stdout.strip())) if r.stdout.strip() else ""


def pick_folder(initial: str) -> str:
    code = ("import sys,tkinter as tk\nfrom tkinter import filedialog\nr=tk.Tk();r.withdraw();"
            "r.attributes('-topmost',True)\n"
            "print(filedialog.askdirectory(title='Dossier des enregistrements',initialdir=sys.argv[1]) or '')")
    r = subprocess.run([PY, "-c", code, initial], capture_output=True, text=True, encoding="utf-8",
                       creationflags=NO_WINDOW, timeout=600)
    if r.returncode != 0:
        raise RuntimeError("selecteur de dossier indisponible")
    return str(Path(r.stdout.strip())) if r.stdout.strip() else ""


def reveal(path: str) -> None:
    p = Path(path)
    allowed = (ROOT, DATA, Path(settings()["app"]["vod_dir"]))
    if not inside(p, *allowed) or not p.exists():
        raise FileNotFoundError(path)
    if os.name == "nt":
        if p.is_dir():
            subprocess.Popen(["explorer", str(p)])
        else:
            subprocess.Popen(f'explorer /select,"{p}"')


def ollama_ok() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5):
            return True
    except Exception:  # noqa: BLE001
        return False


def disk_free() -> dict:
    out = {}
    for d in (DATA, Path(settings()["app"]["vod_dir"])):
        try:
            u = shutil.disk_usage(d.anchor if d.anchor else d)
            out[d.anchor or str(d)] = u.free
        except OSError:
            pass
    return out


# ----------------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "SmartSplit/1.0"

    def log_message(self, fmt, *args):  # noqa: D401  (pas de journal par requete)
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        global last_ping
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self.send_file(APP / "static" / "index.html")
            if u.path.startswith("/static/"):
                p = APP / "static" / unquote(u.path[8:])
                if inside(p, APP / "static"):
                    return self.send_file(p)
            if u.path == "/assets/Poppins-Bold.ttf":
                return self.send_file(ENGINE / "assets" / "Poppins-Bold.ttf")
            if u.path == "/media":
                p = Path(q.get("path", ""))
                if inside(p, DATA) and p.is_file():
                    return self.send_file(p, ranged=True)
                return self.send_error(404)
            if u.path == "/api/ping":
                last_ping = time.time()
                return self.send_json({"ok": True, "app": "smartsplit"})
            if u.path == "/api/state":
                last_ping = time.time()
                with lock:
                    js = [j.to_dict() for j in jobs if j.status in ("attente", "en_cours")
                          or time.time() - (j.ended or j.created) < 6 * 3600]
                return self.send_json({"jobs": js, "ollama": ollama_ok(), "feedback": feedback_stats()})
            if u.path == "/api/job":
                with lock:
                    j = next((j for j in jobs if j.id == q.get("id")), None)
                    if not j:
                        return self.send_json({"error": "traitement inconnu"}, 404)
                    return self.send_json(j.to_dict(with_log=int(q.get("since", 0))))
            if u.path == "/api/vods":
                return self.send_json(list_vods())
            if u.path == "/api/streams":
                return self.send_json({"streams": list_streams(), "disk": disk_free()})
            if u.path == "/api/clips":
                return self.send_json(stream_clips(q.get("stem", "")))
            if u.path == "/api/settings":
                s = settings()
                eng = {k: {"value": s["engine"].get(k, config.DEFAULTS[k]), "default": config.DEFAULTS[k]}
                       for k in config.EDITABLE}
                return self.send_json({"app": s["app"], "app_defaults": APP_DEFAULTS, "engine": eng})
            self.send_error(404)
        except FileNotFoundError as e:
            self.send_json({"error": f"introuvable : {e}"}, 404)
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception as e:  # noqa: BLE001
            log(f"erreur GET {u.path} : {e!r}")
            self.send_json({"error": str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            b = self.body()
            if u.path == "/api/jobs":
                srcs = b.get("srcs") or [b.get("src")]
                added = []
                for src in srcs:
                    p = Path(str(src or "").strip().strip('"'))
                    if not p.is_file():
                        return self.send_json({"error": f"fichier introuvable : {p}"}, 400)
                    with lock:
                        if any(j.src == str(p) and j.status in ("attente", "en_cours") for j in jobs):
                            continue
                        j = Job(str(p), int(b.get("max_clips") or 0), bool(b.get("no_render")),
                                list(b.get("redo") or []))
                        jobs.append(j)
                        added.append(j.id)
                wake.set()
                return self.send_json({"ok": True, "ids": added})
            if u.path == "/api/jobs/cancel":
                with lock:
                    j = next((j for j in jobs if j.id == b.get("id")), None)
                if j:
                    cancel(j)
                return self.send_json({"ok": True})
            if u.path == "/api/jobs/clear":
                with lock:
                    jobs[:] = [j for j in jobs if j.status in ("attente", "en_cours")]
                return self.send_json({"ok": True})
            if u.path == "/api/feedback":
                return self.send_json(set_feedback(b))
            if u.path == "/api/pick":
                init = b.get("initial") or settings()["app"]["vod_dir"]
                fn = pick_folder if b.get("folder") else pick_file
                return self.send_json({"path": fn(init if Path(init).exists() else str(Path.home()))})
            if u.path == "/api/open":
                reveal(b.get("path", ""))
                return self.send_json({"ok": True})
            if u.path == "/api/settings":
                s = settings()
                app = {**s["app"], **{k: v for k, v in (b.get("app") or {}).items() if k in APP_DEFAULTS}}
                eng = {}
                for k, v in (b.get("engine") or {}).items():
                    if k in config.EDITABLE and v != config.DEFAULTS[k]:
                        eng[k] = v
                write_json(SETTINGS_FILE, {"app": app, "engine": eng})
                return self.send_json({"ok": True})
            self.send_error(404)
        except FileNotFoundError as e:
            self.send_json({"error": f"introuvable : {e}"}, 404)
        except Exception as e:  # noqa: BLE001
            log(f"erreur POST {u.path} : {e!r}")
            self.send_json({"error": str(e)}, 500)

    def send_file(self, p: Path, ranged: bool = False):
        if not p.is_file():
            return self.send_error(404)
        size = p.stat().st_size
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if p.suffix == ".js":
            ctype = "text/javascript"
        start, end = 0, size - 1
        rng = self.headers.get("Range") if ranged else None
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") else ""))
        self.send_header("Content-Length", str(end - start + 1))
        if ranged:
            self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            with p.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (ConnectionError, BrokenPipeError):
            pass


# ----------------------------------------------------------------------------- fenetre et cycle de vie

def find_browser() -> str | None:
    cands = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    ]
    return next((str(c) for c in cands if c.exists()), None)


def open_window(url: str) -> None:
    exe = find_browser()
    if exe:
        profile = DATA / "app-profile"       # profil a part, sur D: : une vraie fenetre d'application
        profile.mkdir(parents=True, exist_ok=True)
        subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1440,920",
                          "--no-first-run", "--no-default-browser-check", "--autoplay-policy=no-user-gesture-required"])
    else:
        webbrowser.open(url)


def already_running(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "api/ping", timeout=1.5) as r:
            return json.loads(r.read()).get("app") == "smartsplit"
    except Exception:  # noqa: BLE001
        return False


def watchdog(httpd: ThreadingHTTPServer) -> None:
    while True:
        time.sleep(10)
        with lock:
            busy = any(j.status in ("attente", "en_cours") for j in jobs)
        if not busy and time.time() - last_ping > 120:
            log("fenetre fermee et aucun traitement en cours : arret du serveur")
            httpd.shutdown()
            return


def main() -> None:
    url = f"http://{HOST}:{PORT}/"
    DATA.mkdir(parents=True, exist_ok=True)
    if already_running(url):
        open_window(url)                   # deja lance (traitement en cours, par ex.) : on rouvre la fenetre
        return
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=watchdog, args=(httpd,), daemon=True).start()
    log(f"Smart Split AI sur {url}")
    if "--no-window" not in sys.argv:
        open_window(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with lock:
            running = [j for j in jobs if j.status == "en_cours"]
        for j in running:
            cancel(j)


if __name__ == "__main__":
    main()
