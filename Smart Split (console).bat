@echo off
rem Smart Split AI avec console : pour voir les messages en cas de probleme
cd /d "%~dp0"
"engine\.venv\Scripts\python.exe" "app\server.py"
pause
