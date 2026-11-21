# Smart Split AI

Application de bureau qui transforme un enregistrement de stream Twitch en clips verticaux 9:16 pour TikTok. Les moments forts sont détectés par IA et le cadrage s'adapte à la scène OBS (cam + jeu, jeu seul, full cam). Tout tourne en local sur le GPU.

## Structure

- `engine/` : moteur de traitement en Python (transcription, détection, recadrage, encodage)
- `app/` : application de bureau (serveur local `server.py` + interface `static/index.html`)
- `data/` : modèles, caches et sorties, non versionnés

## Installation (Windows, GPU NVIDIA)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Le script installe Python 3.12, FFmpeg, Git et Node.js dans `D:\Tools`, crée l'environnement Python dans `engine\.venv`, puis lance `engine\check_env.py` pour vérifier le GPU, l'encodage NVENC et la transcription.

## Lancer l'application

Double-clic sur `Smart Split.bat`. L'interface s'ouvre dans une fenêtre d'application (Edge) ; le serveur local tourne sur http://127.0.0.1:8765 et s'arrête tout seul deux minutes après la fermeture de la fenêtre, sauf si un traitement est en cours.

Pour avoir un raccourci « Smart Split AI » sur le Bureau et dans le menu Démarrer :

```powershell
powershell -ExecutionPolicy Bypass -File .\app\install_shortcut.ps1
```

En cas de problème, `Smart Split (console).bat` lance la même chose avec les messages visibles. Le journal est aussi écrit dans `data\app.log`.

- **Traitement** : choisis un ou plusieurs enregistrements (file d'attente), suis chaque étape en direct, annule si besoin.
- **Clips** : regarde les clips de chaque stream, version complète ou courte, puis publie ou rejette avec une raison. Ces décisions (`data\feedback.json`) sont données à l'IA comme exemples pour la notation et le montage suivants.
- **Réglages** : dossier des enregistrements, nombre de clips, texte publicitaire, titre, zoom, durées, vision, contexte donné à l'IA (`data\settings.json`).
