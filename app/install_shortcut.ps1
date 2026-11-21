# Smart Split AI : cree les raccourcis "Smart Split AI" sur le Bureau et dans le menu Demarrer.
# Lancement (PowerShell, dans le dossier du projet) :
#   powershell -ExecutionPolicy Bypass -File .\app\install_shortcut.ps1

$Root = Split-Path $PSScriptRoot -Parent
$Pyw  = Join-Path $Root "engine\.venv\Scripts\pythonw.exe"
$Srv  = Join-Path $Root "app\server.py"
$Icon = Join-Path $Root "app\icon.ico"
if (-not (Test-Path $Pyw)) { throw "Environnement Python introuvable ($Pyw) : lance d'abord setup.ps1" }

$Shell = New-Object -ComObject WScript.Shell
$Targets = @(
    (Join-Path ([Environment]::GetFolderPath("Desktop")) "Smart Split AI.lnk"),
    (Join-Path ([Environment]::GetFolderPath("Programs")) "Smart Split AI.lnk")
)
foreach ($lnk in $Targets) {
    $s = $Shell.CreateShortcut($lnk)
    $s.TargetPath = $Pyw
    $s.Arguments = "`"$Srv`""
    $s.WorkingDirectory = $Root
    $s.IconLocation = "$Icon,0"
    $s.Description = "Smart Split AI : clips TikTok depuis tes streams"
    $s.Save()
    Write-Host "Raccourci cree : $lnk" -ForegroundColor Magenta
}
