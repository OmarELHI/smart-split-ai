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

import numpy as np

from .config import DATA, GOOD_EXAMPLES, LONG_MAX, PROMPT_CONTEXT, SHORT_MAX

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_EXE = Path(r"D:\Tools\Ollama\ollama.exe")       # installe par setup_ollama.ps1
OLLAMA_MODELS = Path(r"D:\Tools\ollama-models")
LLM_MODEL = "gemma3:12b"
BLOCK_S = 30.0       # duree visee d'un extrait
BATCH = 10           # extraits par requete

SYSTEM = """Tu es monteur video TikTok pour Omar, un streamer Twitch francais (il joue souvent a GTA 6).
On te donne des extraits numerotes de la transcription automatique de son stream (environ 30 s chacun).
Note chaque extrait de 0 a 10 selon son potentiel pour un clip TikTok viral.

Ce qui fait un bon clip (par ordre d'importance, d'apres les vues reelles des clips GTA 6 francais) :
1. une reaction forte a quelque chose qu'on VOIT dans le jeu : surprise (un animal, un monstre), scene
   spectaculaire, realisme bluffant, mort, accident, explosion, course-poursuite. C'est ce qui fait le plus de vues ;
2. un moment drole ou une anecdote racontee avec energie ;
3. un avis personnel, seulement s'il est drole, tranche ou surprenant. Un avis calme ou une discussion
   generale sur le jeu fait peu de vues : 4 a 6 maximum.

Ce qui ne fait PAS un clip (note 0 a 2) :
- des paroles de musique ou de radio du jeu, des mots repetes, du texte incomprehensible
  (la transcription automatique invente parfois des paroles sur la musique) ;
- de la logistique banale : "attendez", "j'allume la lumiere", "deux secondes", reglages techniques ;
- des phrases courtes sans contenu ("ok", "voila", "hop") ;
- les dialogues des personnages du jeu et les cinematiques : ce n'est pas Omar qui parle. Indices :
  phrases jouees ou scenarisees, repliques adressees a un personnage (Jason, Lucia, Jay...),
  phrases en anglais ou qui ressemblent a une traduction, ton de film. Note 0 a 2 meme si c'est bien ecrit.
  EXCEPTION : si Omar reagit pendant ou juste apres la cinematique (surprise, peur, rire, commentaire),
  c'est sa reaction qui compte : note-la comme une reaction a un evenement en jeu.

Bareme :
- 8 a 10 : accrocherait vraiment sur TikTok (reaction intense a ce qu'on voit, moment tres drole, avis tres tranche) ;
- 5 a 7 : interessant pour sa communaute (commentaire sur une fonctionnalite du jeu, avis, petite anecdote) ;
- 3 a 4 : un peu de contenu mais faible ;
- 0 a 2 : rien a garder.
La transcription automatique fait des erreurs : juge l'intention probable (ex. "j'etais a 6" peut etre "GTA 6").

Certains extraits sont suivis d'une ligne "A l'ecran :" qui decrit ce qui se passe dans le jeu a ce moment
(type d'action et intensite, vus par une IA sur des images). Utilise-la :
- une action forte a l'ecran (fusillade, accident, explosion, mort, course-poursuite, monstre) AVEC une
  reaction d'Omar = 8 a 10 ;
- une action forte a l'ecran meme si Omar parle peu = au moins 6 ;
- des paroles sans rapport avec un ecran calme (conduite, menu) restent jugees sur leur contenu.
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


FEEDBACK_FILE = DATA / "feedback.json"     # ecrit par l'application : clips publies ou rejetes (avec raison)
FEEDBACK_MAX = 8                            # exemples de chaque sorte donnes a l'IA (les plus recents)


def _feedback_examples() -> tuple[list[str], list[str]]:
    """Apprentissage : les clips publies deviennent des exemples a suivre, les rejets des erreurs a eviter."""
    try:
        fb = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], []
    items = sorted(fb.values(), key=lambda f: f.get("date", ""))

    def desc(f: dict) -> str:
        t = " ".join(str(f.get("text") or f.get("titre") or "").split())
        t = t[:240] + "..." if len(t) > 240 else t
        return f"{f.get('type', 'clip')} ({float(f.get('duree') or 0):.0f} s) : \"{t}\""

    pub = [desc(f) + " (publie)" for f in items if f.get("decision") == "publie"][-FEEDBACK_MAX:]
    rej = [f"{desc(f)} -> rejete : {f['raison']}" for f in items
           if f.get("decision") == "rejete" and f.get("raison")][-FEEDBACK_MAX:]
    return pub, rej


def _context_suffix() -> str:
    out = "\n\nContexte sur le stream :\n" + PROMPT_CONTEXT if PROMPT_CONTEXT.strip() else ""
    pub, rej = _feedback_examples()
    good = list(GOOD_EXAMPLES) + pub
    if good:
        out += "\n\nExemples de clips qu'Omar a trouves reussis (vise ce genre de moment) :\n" + \
               "\n".join(f"- {e}" for e in good)
    if rej:
        out += "\n\nClips qu'Omar a rejetes, avec sa raison (evite ces erreurs) :\n" + \
               "\n".join(f"- {e}" for e in rej)
    return out


def system_prompt() -> str:
    return SYSTEM + _context_suffix()


def _chat(messages: list[dict], timeout: float = 300) -> dict:
    body = json.dumps({"model": LLM_MODEL, "messages": messages, "stream": False, "format": SCHEMA,
                       "options": {"temperature": 0, "num_ctx": 8192}, "keep_alive": "5m"}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def unload(model: str | None = None) -> None:
    """Libere la memoire du GPU (utile avant le rendu ou Whisper)."""
    try:
        body = json.dumps({"model": model or LLM_MODEL, "keep_alive": 0}).encode()
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


def score(transcript: dict, progress=print, vision: list[dict] | None = None) -> dict:
    t0 = time.time()
    bl = blocks(transcript)
    if vision:
        from .vision import screen_text
        for b in bl:
            b["ecran"] = screen_text(vision, b["start"], b["end"])
    results: dict[int, dict] = {}
    todo = [b for b in bl if len(b["text"].split()) >= 4]          # trop court : note 0 d'office
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        user = "\n\n".join(f"[{b['id']}] ({b['end'] - b['start']:.0f} s) {b['text']}"
                             + (f"\nA l'ecran : {b['ecran']}" if b.get("ecran") else "") for b in batch)
        try:
            res = _chat([{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}])
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


MONTAGE = """Tu es monteur video TikTok pour Omar, streamer francais. Un moment interessant a ete repere
dans son stream. On te donne les phrases numerotees autour de ce moment, avec leur instant en secondes.
Les repliques des personnages du jeu peuvent apparaitre dans les phrases : elles font partie du contexte.
Choisis la phrase de DEBUT et la phrase de FIN du clip :
- le clip doit se comprendre seul : inclure ce qui declenche le moment (la question posee, ce que quelqu'un
  vient de dire, l'evenement qui provoque la reaction, le sujet de l'avis) ;
- il doit contenir l'information principale en entier (l'avis ET son explication) ;
- aller droit au but : pas de phrases inutiles avant, s'arreter juste apres la chute ou la conclusion ;
- un avis ou une reaction isolee : 15 a {short} s, idealement 20 a 40 s ;
- une histoire en jeu (contexte, action, rebondissements, reactions d'Omar) : jusqu'a {long} s.
Si ton clip depasse {short} s, propose AUSSI une version courte (debut_court, fin_court) de {short} s maximum,
centree sur le moment le plus fort mais toujours comprehensible. Sinon, recopie debut et fin.
Reponds uniquement en JSON."""

MONTAGE_SCHEMA = {
    "type": "object",
    "properties": {"debut": {"type": "integer"}, "fin": {"type": "integer"},
                   "debut_court": {"type": "integer"}, "fin_court": {"type": "integer"},
                   "raison": {"type": "string"}},
    "required": ["debut", "fin", "debut_court", "fin_court", "raison"],
}


def _bounds(near: list[dict], a: int, b: int) -> tuple[float, float] | None:
    if not (0 <= a <= b < len(near)):
        return None
    return max(0.0, near[a]["start"] - 0.3), near[b]["end"] + 0.6


def refine_clip(clip: dict, sents: list[dict], before: float = 60.0, after: float = 45.0,
                vision: list[dict] | None = None) -> dict:
    """Demande a l'IA les bornes (phrases de debut et de fin) d'un clip qui se comprend seul.

    Un clip long (histoire complete, plus de SHORT_MAX s) recoit aussi une version courte dans clip["court"].
    """
    near = [s for s in sents if s["end"] >= clip["start"] - before and s["start"] <= clip["end"] + after]
    if clip.get("vision"):
        # action sans paroles : reperes de temps toutes les 5 s pour pouvoir couper entre les phrases
        marks = [{"start": t, "end": t, "text": "(action a l'ecran)"} for t in
                 np.arange(clip["start"] - 15, clip["end"] + 15, 5.0) if t >= 0]
        near = sorted(near + marks, key=lambda s: s["start"])
    if len(near) < 2:
        return clip
    lines = "\n".join(f"[{i}] {s['start']:.0f}s-{s['end']:.0f}s : {s['text']}" for i, s in enumerate(near))
    user = (f"Moment repere entre {clip['start']:.0f}s et {clip['end']:.0f}s (raison : {clip['raison']}).\n\n"
            f"Phrases :\n{lines}")
    if vision:
        lo, hi = clip["start"] - before, clip["end"] + after
        scr = [f"{w['start']:.0f}s-{w['end']:.0f}s : {w['description']} ({w['action']}, intensite {w['intensite']})"
               for w in vision if w["end"] > lo and w["start"] < hi]
        if scr:
            user += "\n\nCe qui se passe a l'ecran (inclure l'action en entier si elle fait le moment) :\n" + "\n".join(scr)
    system = MONTAGE.format(short=int(SHORT_MAX), long=int(LONG_MAX)) + _context_suffix()
    body = json.dumps({"model": LLM_MODEL, "stream": False, "format": MONTAGE_SCHEMA,
                       "options": {"temperature": 0, "num_ctx": 8192}, "keep_alive": "5m",
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            res = json.loads(json.loads(r.read())["message"]["content"])
        full = _bounds(near, int(res["debut"]), int(res["fin"]))
        short = _bounds(near, int(res.get("debut_court", -1)), int(res.get("fin_court", -1)))
    except Exception:  # noqa: BLE001
        return clip
    if not full or not 12 <= full[1] - full[0] <= LONG_MAX + 5:   # hors limites : decoupage d'origine
        return clip
    out = {**clip, "start": round(full[0], 2), "end": round(full[1], 2), "montage": res.get("raison", "")}
    out.pop("court", None)
    if full[1] - full[0] > SHORT_MAX + 2:
        # version courte : celle de l'IA si elle est valable, sinon le moment repere au depart
        if not (short and 12 <= short[1] - short[0] <= SHORT_MAX + 2 and full[0] - 1 <= short[0]
                and short[1] <= full[1] + 1):
            short = (clip["start"], min(clip["end"], clip["start"] + SHORT_MAX))
        out["court"] = {"start": round(short[0], 2), "end": round(short[1], 2)}
    return out


def refine_clips(clips: list[dict], transcript: dict, progress=print, vision: list[dict] | None = None) -> list[dict]:
    from .transcribe import sentences
    sents = sentences(transcript)
    out = []
    t0 = time.time()
    for i, c in enumerate(clips, 1):
        out.append(refine_clip(c, sents, vision=vision))
        if i % 10 == 0 or i == len(clips):
            progress(f"  montage : {i}/{len(clips)} clips ({time.time() - t0:.0f} s)")
    return out
