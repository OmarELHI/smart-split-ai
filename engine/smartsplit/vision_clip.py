"""Vision rapide sur le GPU : SigLIP (Google) compare chaque image a une liste de scenes decrites en mots.

Bien plus rapide que de faire decrire les images par gemma3 via Ollama (quelques millisecondes par image
au lieu d'une minute) : il ne redige pas de description libre, il dit a quelle scene l'image ressemble le plus.
Le resultat a la meme forme que celui de gemma3 : action, intensite de 0 a 10, description.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL = "google/siglip-so400m-patch14-384"   # version la plus precise de SigLIP (environ 3,5 Go)

# action : (descriptions en anglais pour le modele, description en francais, intensite de base)
# Les scenes calmes sont nombreuses et precises : sinon le modele range n'importe quelle image dans une action.
LABELS = {
    "fusillade": (["a gunfight with muzzle flashes and people shooting guns"], "Fusillade, coups de feu", 9),
    "accident": (["a car crashing violently into another car with debris flying"], "Accident de voiture violent", 9),
    "explosion": (["a huge explosion with a fireball"], "Explosion", 9),
    "poursuite": (["police cars with flashing lights chasing a car at high speed"], "Course-poursuite avec la police", 8),
    "mort": (["a red and black death screen with the word wasted"], "Mort du personnage", 8),
    "monstre": (["a giant hairy bigfoot monster standing in the forest"], "Un monstre ou un yeti", 8),
    "animal": (["an alligator attacking a man in a swamp", "a tiger attacking a man"], "Attaque d'un animal sauvage", 8),
    "cascade": (["a car flying through the air after a jump"], "Cascade, saut en voiture", 8),
    "combat": (["two men punching each other in a fist fight"], "Bagarre", 7),
    "paysage": (["a breathtaking sunset over the ocean", "a beautiful city skyline at dusk"], "Tres beau paysage", 5),
    "cinematique": (["a cinematic close-up of two characters talking"], "Cinematique", 3),
    "conduite": (["driving a pickup truck on a city street", "driving a car on a dirt road",
                  "a car parked on the side of a road"], "Conduite", 2),
    "exploration": (["a man walking through tall grass in a swamp", "a man walking in a city street",
                     "a man standing next to a car", "a man aiming a gun at the grass"], "Marche, exploration", 2),
    "menu": (["a blurred game screen with a weapon selection wheel", "a game menu or a map", "a black screen"],
             "Menu ou ecran noir", 0),
}

_model = None
OWNER = None
SIZE = 224


def _load():
    global _model
    if _model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(MODEL, dtype=torch.float16 if dev == "cuda" else torch.float32)
        model = model.to(dev).eval()
        tok = AutoTokenizer.from_pretrained(MODEL)
        texts, owner = [], []
        for i, v in enumerate(LABELS.values()):
            for d in v[0]:
                texts.append(f"a screenshot of a video game: {d}")
                owner.append(i)
        with torch.no_grad():
            t = tok(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
            tf = model.get_text_features(**t)
            tf = getattr(tf, "pooler_output", tf)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        global OWNER, SIZE
        OWNER = torch.tensor(owner, device=dev)
        SIZE = int(model.config.vision_config.image_size)
        _model = (model, None, tf, dev)
    return _model


def describe(paths: list[Path], timeout: float = 0) -> dict:  # noqa: ARG001 (meme signature que gemma3)
    import torch
    model, _, tf, dev = _load()
    # pretraitement SigLIP fait a la main (evite Pillow / torchvision) : taille du modele, valeurs ramenees dans [-1, 1]
    imgs = [cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (SIZE, SIZE),
                       interpolation=cv2.INTER_CUBIC) for p in paths]
    arr = (np.stack(imgs).astype(np.float32) / 255.0 - 0.5) / 0.5
    with torch.no_grad():
        px = torch.from_numpy(arr).permute(0, 3, 1, 2).to(dev, dtype=tf.dtype)
        f = model.get_image_features(pixel_values=px)
        f = getattr(f, "pooler_output", f)
        f = f / f.norm(dim=-1, keepdim=True)
        logits = (f @ tf.T).float() * model.logit_scale.exp().float()
        p = logits.softmax(dim=-1)                            # une ligne par image, une colonne par phrase
        probs = torch.zeros(p.shape[0], len(LABELS), device=p.device).index_add_(1, OWNER, p).cpu().numpy()
    keys = list(LABELS)
    best = probs.max(axis=0)                                  # chaque scene : l'image qui y ressemble le plus
    mean = probs.mean(axis=0)
    top = int(mean.argmax())                                  # scene dominante sur la fenetre
    key, inten = keys[top], LABELS[keys[top]][2] * min(1.0, mean[top] / 0.5)
    calm = max(best[i] for i, k in enumerate(keys) if LABELS[k][2] <= 3)
    for i, k in enumerate(keys):     # action nette sur au moins une image, plus probable que toute scene calme
        if LABELS[k][2] >= 7 and best[i] >= 0.4 and best[i] > calm:
            v = LABELS[k][2] * min(1.0, best[i] / 0.6)
            if v > inten:
                key, inten = k, v
    order = mean.argsort()[::-1][:3]
    return {"action": key, "intensite": int(round(inten)), "description": LABELS[key][1],
            "confiance": round(float(best[keys.index(key)]), 2),
            "top3": " ".join(f"{keys[i]}:{mean[i]:.2f}" for i in order)}
