# Smart Split AI - Etape 0 : installation de l'environnement
# Tout est installe sur D: (outils dans D:\Tools, projet dans ce dossier).
# Lancement (PowerShell, dans ce dossier) :
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"
$Root  = $PSScriptRoot
$Tools = "D:\Tools"
$Data  = Join-Path $Root "data"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Magenta }
function Has($cmd)  { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}
function Add-UserPath($dir) {
    $p = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($p -notlike "*$dir*") {
        [Environment]::SetEnvironmentVariable("Path", "$p;$dir", "User")
    }
    Refresh-Path
}

New-Item -ItemType Directory -Force -Path $Tools, $Data | Out-Null

# 1. Python 3.12 dans D:\Tools\Python312
Step "Python 3.12"
$Py = Join-Path $Tools "Python312\python.exe"
if (-not (Test-Path $Py)) {
    winget install -e --id Python.Python.3.12 --scope user `
        --accept-package-agreements --accept-source-agreements `
        --override "/quiet InstallAllUsers=0 TargetDir=$Tools\Python312 PrependPath=0 Include_launcher=0 Include_test=0"
}
if (-not (Test-Path $Py)) { throw "Python n'a pas ete installe dans $Tools\Python312" }
& $Py --version

# 2. FFmpeg (build gyan.dev avec NVENC/NVDEC) dans D:\Tools\ffmpeg
Step "FFmpeg"
$FfBin = Join-Path $Tools "ffmpeg\bin"
if (-not (Test-Path (Join-Path $FfBin "ffmpeg.exe"))) {
    $zip = Join-Path $Tools "ffmpeg.zip"
    $tmp = Join-Path $Tools "ffmpeg-tmp"
    Invoke-WebRequest "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $zip
    Expand-Archive $zip -DestinationPath $tmp -Force
    $dir = Get-ChildItem $tmp -Directory | Select-Object -First 1
    Move-Item $dir.FullName (Join-Path $Tools "ffmpeg")
    Remove-Item $tmp -Recurse -Force
    Remove-Item $zip
}
Add-UserPath $FfBin
ffmpeg -hide_banner -version | Select-Object -First 1

# 3. Git et Node.js (Node servira au Jalon C, interface Vue.js)
Step "Git et Node.js"
if (-not (Has "git")) {
    winget install -e --id Git.Git --location "$Tools\Git" --accept-package-agreements --accept-source-agreements
}
if (-not (Has "node")) {
    winget install -e --id OpenJS.NodeJS.LTS --location "$Tools\nodejs" --accept-package-agreements --accept-source-agreements
}
Refresh-Path

# 4. Caches et modeles sur D:
Step "Caches sur D:"
[Environment]::SetEnvironmentVariable("PIP_CACHE_DIR", "$Tools\pip-cache", "User")
[Environment]::SetEnvironmentVariable("HF_HOME", "$Data\models", "User")
$env:PIP_CACHE_DIR = "$Tools\pip-cache"
$env:HF_HOME = "$Data\models"

# 5. Environnement virtuel Python + dependances
Step "Environnement Python (engine\.venv)"
$Venv = Join-Path $Root "engine\.venv"
if (-not (Test-Path $Venv)) { & $Py -m venv $Venv }
$VPy = Join-Path $Venv "Scripts\python.exe"
& $VPy -m pip install --upgrade pip
# PyTorch compile pour CUDA 12.8 : necessaire pour les cartes RTX 50 (Blackwell)
& $VPy -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& $VPy -m pip install -r (Join-Path $Root "engine\requirements.txt")

# 6. Depot Git
Step "Depot Git"
if (-not (Test-Path (Join-Path $Root ".git"))) {
    git -C $Root init
}

# 7. Verification
Step "Verification de l'environnement"
& $VPy (Join-Path $Root "engine\check_env.py")
