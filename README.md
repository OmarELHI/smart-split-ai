# Smart Split AI

Application de bureau qui transforme un enregistrement de stream Twitch en clips verticaux 9:16 pour TikTok. Les moments forts sont détectés par IA et le cadrage s'adapte à la scène OBS (cam + jeu, jeu seul, full cam). Tout tourne en local sur le GPU.

## Structure

- `engine/` : moteur de traitement en Python (transcription, détection, recadrage, encodage)
- `app/` : interface Vue.js + Electron (Jalon C)
- `data/` : modèles, caches et sorties, non versionnés

## Installation (Windows, GPU NVIDIA)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Le script installe Python 3.12, FFmpeg, Git et Node.js dans `D:\Tools`, crée l'environnement Python dans `engine\.venv`, puis lance `engine\check_env.py` pour vérifier le GPU, l'encodage NVENC et la transcription.
