"""Met a jour et affiche les tendances TikTok / YouTube Shorts autour du jeu (TREND_GAME dans config.py).

  python engine\trends.py
"""
from smartsplit import trends

t = trends.update(force=True)
if t:
    print(trends.prompt_text())
else:
    print("Aucune tendance recuperee.")
