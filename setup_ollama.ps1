# Smart Split AI - Etape 6 : installation d'Ollama et du modele de langage, tout sur D:
# Lancement (PowerShell, dans ce dossier) :
#   powershell -ExecutionPolicy Bypass -File .\setup_ollama.ps1

$ErrorActionPreference = "Stop"
# La barre de progression de PowerShell 5 ralentit enormement les telechargements
$ProgressPreference = "SilentlyContinue"
$Tools  = "D:\Tools"
$Dir    = Join-Path $Tools "Ollama"
$Models = Join-Path $Tools "ollama-models"
$Model  = "gemma3:12b"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Magenta }

New-Item -ItemType Directory -Force -Path $Tools, $Models | Out-Null

# 1. Les modeles iront dans D:\Tools\ollama-models (environ 8 Go pour gemma3:12b)
Step "Dossier des modeles sur D:"
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $Models, "User")
$env:OLLAMA_MODELS = $Models

# 2. Installation d'Ollama dans D:\Tools\Ollama
Step "Ollama"
$exe = Join-Path $Dir "ollama.exe"
if (-not (Test-Path $exe)) {
    $setup = Join-Path $Tools "OllamaSetup.exe"
    # curl.exe (fourni avec Windows) : rapide et affiche la progression
    curl.exe -L --fail -o $setup "https://ollama.com/download/OllamaSetup.exe"
    if ($LASTEXITCODE -ne 0) { throw "telechargement de l installateur Ollama impossible" }
    Start-Process $setup -ArgumentList "/VERYSILENT", "/NORESTART", "/DIR=`"$Dir`"" -Wait
    Remove-Item $setup
}
if (-not (Test-Path $exe)) { throw "Ollama n'a pas ete installe dans $Dir" }

# 3. Redemarre le serveur Ollama pour qu'il utilise le dossier des modeles sur D:
Step "Serveur Ollama"
Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
Start-Process $exe -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 5
& $exe --version

# 4. Telechargement du modele
Step "Modele $Model"
& $exe pull $Model

Step "Termine"
Write-Host "Ollama : $exe"
Write-Host "Modeles : $Models"
