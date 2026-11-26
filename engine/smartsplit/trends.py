"""Tendances : ce qui marche en ce moment sur YouTube Shorts et TikTok autour du jeu.

On recupere (avec yt-dlp, sans compte ni cle) les titres et les vues :
- des Shorts YouTube les mieux places sur quelques recherches autour du jeu ;
- des dernieres videos de comptes TikTok de clips choisis (TREND_TIKTOK).
Puis l'IA (gemma3) en tire une courte liste : les sujets et les moments qui font des vues, et ce qui en fait
peu. Cette liste est donnee a la notation et au montage, a cote des retours d'Omar.
Mise a jour automatique tous les TREND_DAYS jours (en cas d'echec reseau, on garde la derniere).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from .config import DATA, TREND_DAYS, TREND_GAME, TREND_TIKTOK

# Recherches YouTube Shorts : francaises et internationales (ce qui marche ailleurs arrive souvent ensuite en France)
QUERIES = [("international", "{g} clips"), ("international", "{g} funny clips"), ("FR", "{g} clips fr"),
           ("FR", "{g} shorts fr"), ("FR", "{g} moment drole"), ("FR", "{g} reaction streamer fr"),
           ("FR", "{g} clip twitch fr"),
           ("international", "{g} funny moments shorts"), ("international", "{g} streamer reaction shorts"),
           ("international", "{g} insane moment shorts"), ("international", "{g} realistic details shorts"),
           ("international", "{g} fails shorts"), ("international", "{g} clip kick twitch")]
# Videos officielles (bandes-annonces, pubs) : enormement de vues, mais rien a voir avec des clips de stream
OFFICIAL = ("rockstar games", "playstation", "netflix", "xbox", "ign", "gamespot")
OFFICIAL_WORDS = ("trailer", "bande-annonce", "bande annonce", "official", "officiel", "out now", "now playing",
                  "concept", "fan made", "fanmade", "leak", "fake")   # concepts de fans et fuites : pas le vrai jeu
MAX_AGE_DAYS = 45   # videos YouTube gardees : mises en ligne il y a 45 jours au plus


def _path() -> Path:
    return DATA / "tendances" / f"{TREND_GAME.lower().replace(' ', '_')}.json"


def _ytdlp(args: list[str], timeout: int = 180) -> list[dict]:
    cmd = [sys.executable, "-m", "yt_dlp", "--flat-playlist", "--dump-json", "--no-warnings", "--ignore-errors", *args]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout).stdout
    except Exception:  # noqa: BLE001
        return []
    items = []
    for line in out.splitlines():
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return items


def collect(progress=print) -> list[dict]:
    """Titres + vues des videos courtes qui marchent autour du jeu."""
    seen, out = set(), []
    for zone, q in QUERIES:
        q = q.format(g=TREND_GAME)
        got = _ytdlp([f"ytsearch50:{q} #shorts"])        # "#shorts" : la recherche remonte surtout des Shorts
        why = {"trop longues": 0, "officielles ou concepts": 0, "deja vues": 0, "sans vues": 0}
        n = 0
        for it in got:
            dur, views = it.get("duration") or 0, it.get("view_count") or 0
            title = it.get("title", "")
            chan = (it.get("channel") or it.get("uploader") or "").lower()
            if it.get("id") in seen:
                why["deja vues"] += 1
                continue
            if not views:
                why["sans vues"] += 1
                continue
            if dur and dur > 180:                         # un Short dure 3 min au plus
                why["trop longues"] += 1
                continue
            if any(o in chan for o in OFFICIAL) or any(w in title.lower() for w in OFFICIAL_WORDS):
                why["officielles ou concepts"] += 1
                continue
            seen.add(it.get("id"))
            out.append({"source": f"YouTube Shorts {zone}", "titre": title, "vues": int(views), "id": it.get("id")})
            n += 1
        detail = ", ".join(f"{k} {v}" for k, v in why.items() if v)
        progress(f"  YouTube {zone} « {q} » : {n} gardees sur {len(got)}" + (f" ({detail})" if detail else ""))
    out = _recent_only(out, progress)
    for acc in TREND_TIKTOK:
        got = _ytdlp(["--impersonate", "chrome", "--playlist-end", "30", f"https://www.tiktok.com/@{acc}"])
        n = 0
        for it in got:
            views = it.get("view_count") or 0
            if it.get("id") in seen or not views:
                continue
            seen.add(it.get("id"))
            out.append({"source": f"TikTok @{acc}", "titre": (it.get("title") or it.get("description") or "")[:150],
                        "vues": int(views)})
            n += 1
        progress(f"  TikTok @{acc} : {n} videos")
    out.sort(key=lambda x: -x["vues"])
    return out


def _recent_only(videos: list[dict], progress=print) -> list[dict]:
    """La recherche YouTube ne donne pas la date : on la demande pour les 80 plus vues, et on ne garde que les
    videos recentes (sinon les concepts et les videos d'avant la sortie du jeu dominent)."""
    from datetime import timedelta
    top = sorted(videos, key=lambda v: -v["vues"])[:80]
    if not top:
        return []
    cmd = [sys.executable, "-m", "yt_dlp", "--skip-download", "--no-warnings", "--ignore-errors",
           "--print", "%(id)s|%(upload_date)s", *[f"https://www.youtube.com/shorts/{v['id']}" for v in top]]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=600).stdout
    except Exception:  # noqa: BLE001
        return top
    dates = dict(line.split("|", 1) for line in res.splitlines() if "|" in line)
    limit = (datetime.now() - timedelta(days=MAX_AGE_DAYS)).strftime("%Y%m%d")
    keep = [v for v in top if dates.get(v["id"], "0") >= limit]
    progress(f"  YouTube : {len(keep)} videos recentes sur les {len(top)} plus vues")
    return keep


def summarize(videos: list[dict]) -> dict:
    from .llm import LLM_MODEL, OLLAMA_URL
    top = videos[:60]
    low = videos[-20:] if len(videos) > 80 else []
    lines = "\n".join(f"- {v['vues']:,} vues ({v['source']}) : {v['titre']}".replace(",", " ") for v in top)
    lows = "\n".join(f"- {v['vues']:,} vues : {v['titre']}".replace(",", " ") for v in low)
    schema = {"type": "object", "properties": {
        "sujets": {"type": "array", "items": {"type": "string"}},
        "moments": {"type": "array", "items": {"type": "string"}},
        "a_eviter": {"type": "array", "items": {"type": "string"}},
        "titres": {"type": "array", "items": {"type": "string"}}},
        "required": ["sujets", "moments", "a_eviter", "titres"]}
    prompt = (f"Voici les videos courtes qui font le plus de vues en ce moment autour de {TREND_GAME} "
              f"(YouTube Shorts en France et a l'international, TikTok), avec leurs vues :\n{lines}\n"
              + (f"\nEt quelques-unes qui en font peu :\n{lows}\n" if lows else "")
              + "\nPour un streamer francais qui fait des clips de ses lives (il joue au mode histoire, pas au RP), "
                "resume en francais. Parle de ce qui se passe dans les videos, pas des chaines ni des streamers "
                "qui les publient :\n"
                "- sujets : 5 a 8 sujets du jeu qui attirent (lieux, personnages, mecaniques, evenements) ;\n"
                "- moments : 5 a 8 types de moments qui font des vues (ex. reaction a un detail realiste) ;\n"
                "- a_eviter : 3 a 5 types de contenus qui font peu de vues ;\n"
                "- titres : 3 a 5 formes de titres qui marchent pour des clips de reaction (pas des mots comme "
                "4K, Gameplay ou Concept).\n"
                "Phrases courtes et concretes. Reponds en JSON.")
    body = json.dumps({"model": LLM_MODEL, "stream": False, "format": schema, "keep_alive": "5m",
                       "options": {"temperature": 0, "num_ctx": 8192},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(json.loads(r.read())["message"]["content"])


def update(progress=print, force: bool = False) -> dict | None:
    """Met a jour les tendances si elles ont plus de TREND_DAYS jours. Renvoie les tendances (ou None)."""
    cur = load()
    if cur and not force and time.time() - cur.get("ts", 0) < TREND_DAYS * 86400:
        return cur
    progress(f"Mise a jour des tendances {TREND_GAME}...")
    videos = collect(progress)
    if len(videos) < 15:
        progress(f"  trop peu de videos recuperees ({len(videos)}) : on garde les tendances precedentes")
        return cur
    from .llm import check
    check()
    summary = summarize(videos)
    data = {"jeu": TREND_GAME, "ts": time.time(), "date": datetime.now().strftime("%d/%m/%Y"),
            "resume": summary, "top": videos[:15], "nb_videos": len(videos)}
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    progress(f"  tendances enregistrees ({len(videos)} videos analysees)")
    return data


def load() -> dict | None:
    p = _path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def prompt_text() -> str:
    """Texte ajoute aux consignes de l'IA (vide si aucune tendance)."""
    t = load()
    if not t:
        return ""
    r = t["resume"]
    out = (f"\n\nCe qui marche en ce moment sur TikTok et YouTube Shorts autour de {t['jeu']} "
           f"(analyse de {t['nb_videos']} videos, {t['date']}). Privilegie les moments qui collent a ces tendances :")
    for label, key in (("Sujets qui attirent", "sujets"), ("Moments qui font des vues", "moments"),
                       ("Fait peu de vues", "a_eviter"), ("Formes de titres qui marchent", "titres")):
        items = [x for x in r.get(key, []) if not any(w in x.lower() for w in (" rp", "rp)", "roleplay", "flashback"))]
        if items:
            out += f"\n{label} : " + " ; ".join(items[:8])
    rp = ("gtarp", "gta rp", "flashback", " rp ")
    ex, titles = [], set()
    for v in t.get("top", []):
        k = v["titre"][:40].lower()
        if k in titles or any(w in v["titre"].lower() for w in rp):
            continue
        titles.add(k)
        ex.append(f"\"{v['titre'][:70]}\" ({v['vues'] // 1000} k vues)")
    out += "\nExemples de videos tres vues : " + " | ".join(ex[:6])
    return out
