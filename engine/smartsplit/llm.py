"""Etape 6 : notation des propos par un modele de langage local (Ollama).

La transcription est decoupee en extraits d'environ 30 s, envoyes par paquets de 10.
Pour chacun, le modele renvoie une note de 0 a 10, un type et une courte raison.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from .config import DATA

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_EXE = Path(r"D:\Tools\Ollama\ollama.exe")       # installe par setup_ollama.ps1
OLLAMA_MODELS = Path(r"D:\Tools\ollama-models")
LLM_MODEL = "gemma3:12b"
BLOCK_S = 30.0       # duree visee d'un extrait
BATCH = 10           # extraits par requete

SYSTEM = """Tu es monteur video TikTok pour Omar, un streamer Twitch francais (il joue souvent a GTA 6).
On te donne des extraits numerotes de la transcription automatique de son stream (environ 30 s chacun).
Note chaque extrait de 0 a 10 selon son potentiel pour un clip TikTok viral.

Ce qui fait un bon clip :
- un avis personnel tranche, original ou drole (sur un jeu, GTA 6, un sujet du moment) ;
- un commentaire interessant sur ce que propose le jeu (nouveautes, details, comparaisons) ;
- une reaction forte a un evenement en jeu : mort, accident, explosion, course-poursuite, surprise ;
- un moment drole ou une anecdote racontee avec energie.

Ce qui ne fait PAS un clip (note 0 a 2) :
- des paroles de musique ou de radio du jeu, des mots repetes, du texte incomprehensible
  (la transcription automatique invente parfois des paroles sur la musique) ;
- de la logistique banale : "attendez", "j'allume la lumiere", "deux secondes", reglages techniques ;
- des phrases courtes sans contenu ("ok", "voila", "hop").

Bareme :
- 8 a 10 : accrocherait vraiment sur TikTok (avis fort, reaction intense, moment tres drole) ;
- 5 a 7 : interessant pour sa communaute (commentaire sur une fonctionnalite du jeu, avis meme bref, petite anecdote) ;
- 3 a 4 : un peu de contenu mais faible ;
- 0 a 2 : rien a garder.
La transcription automatique fait des erreurs : juge l'intention probable (ex. "j'etais a 6" peut etre "GTA 6").
Reponds uniquement en JSON."""

SCHEMA = {
    "type": "object",
    "properties": {
        "extraits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "note": {"type": "integer"},
                    "type": {"type": "string", "enum": ["avis", "jeu", "reaction", "drole", "rien"]},
                    "raison": {"type": "string"},
                },
                "required": ["id", "note", "type", "raison"],
            },
        }
    },
    "required": ["extraits"],
}


def scores_path(src: Path) -> Path:
    return DATA / "analysis" / Path(src).stem / "llm_scores.json"


def blocks(transcript: dict, block_s: float = BLOCK_S) -> list[dict]:
    """Regroupe les phrases en extraits d'environ block_s secondes, sans couper une phrase."""
    from .transcribe import sentences
    out, cur = [], []
    for s in sentences(transcript):
        if cur and (s["end"] - cur[0]["start"] > block_s or s["start"] - cur[-1]["end"] > 8):
            out.append(cur)
            cur = []
        cur.append(s)
    if cur:
        out.append(cur)
    return [{"id": i + 1, "start": b[0]["start"], "end": b[-1]["end"],
             "text": " ".join(s["text"] for s in b)} for i, b in enumerate(out)]


def _chat(messages: list[dict], timeout: float = 300) -> dict:
    body = json.dumps({"model": LLM_MODEL, "messages": messages, "stream": False, "format": SCHEMA,
                       "options": {"temperature": 0, "num_ctx": 8192}, "keep_alive": "5m"}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def unload() -> None:
    """Libere la memoire du GPU (utile avant le rendu ou Whisper)."""
    try:
        body = json.dumps({"model": LLM_MODEL, "keep_alive": 0}).encode()
        urllib.request.urlopen(urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body,
                                                      headers={"Content-Type": "application/json"}), timeout=30)
    except Exception:  # noqa: BLE001
        pass


def _alive() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3).close()
        return True
    except Exception:  # noqa: BLE001
        return False


def ensure_server(progress=print) -> None:
    """Demarre le serveur Ollama en arriere-plan s'il ne tourne pas (apres un redemarrage du PC, par ex.)."""
    if _alive():
        return
    exe = str(OLLAMA_EXE) if OLLAMA_EXE.exists() else shutil.which("ollama")
    if not exe:
        raise RuntimeError("Ollama introuvable : lance setup_ollama.ps1")
    env = dict(os.environ)
    env.setdefault("OLLAMA_MODELS", str(OLLAMA_MODELS))      # modeles toujours lus sur D:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen([exe, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    progress("  demarrage du serveur Ollama...")
    for _ in range(40):
        time.sleep(0.5)
        if _alive():
            return
    raise RuntimeError("le serveur Ollama ne repond pas apres 20 s")


def check() -> str:
    """Verifie qu'Ollama repond (le demarre au besoin) et que le modele est installe."""
    ensure_server()
    with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
        names = [m["name"] for m in json.loads(r.read()).get("models", [])]
    if not any(n == LLM_MODEL or n.startswith(LLM_MODEL + ":") for n in names):
        raise RuntimeError(f"modele {LLM_MODEL} absent d'Ollama (installes : {', '.join(names) or 'aucun'})")
    return LLM_MODEL


def score(transcript: dict, progress=print) -> dict:
    t0 = time.time()
    bl = blocks(transcript)
    results: dict[int, dict] = {}
    todo = [b for b in bl if len(b["text"].split()) >= 4]          # trop court : note 0 d'office
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        user = "\n\n".join(f"[{b['id']}] ({b['end'] - b['start']:.0f} s) {b['text']}" for b in batch)
        try:
            res = _chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
            for e in json.loads(res["message"]["content"]).get("extraits", []):
                results[int(e["id"])] = e
        except Exception as e:  # noqa: BLE001
            progress(f"  [paquet {i // BATCH + 1} ignore : {e}]")
        progress(f"  notation : {min(i + BATCH, len(todo))}/{len(todo)} extraits ({time.time() - t0:.0f} s)")
    for b in bl:
        e = results.get(b["id"], {})
        b["note"] = max(0, min(10, int(e.get("note", 0))))
        b["type"] = e.get("type", "rien")
        b["raison"] = e.get("raison", "trop court" if b not in todo else "")
    return {"model": LLM_MODEL, "seconds": round(time.time() - t0, 1), "blocks": bl}


def save(src: Path, res: dict) -> Path:
    out = scores_path(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def load(src: Path) -> dict | None:
    f = scores_path(src)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
