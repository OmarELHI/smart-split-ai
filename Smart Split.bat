@echo off
rem Smart Split AI : lance l application (sans console)
cd /d "%~dp0"
if exist "engine\.venv\Scripts\pythonw.exe" (
  start "" "engine\.venv\Scripts\pythonw.exe" "app\server.py"
) else (
  echo Environnement Python introuvable : lance d abord setup.ps1
  pause
)
