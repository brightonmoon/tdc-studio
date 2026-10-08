# colab_exec.ps1 - Run colab exec with command-line arguments on Windows PowerShell
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$Session,

    [Parameter(Position = 1, Mandatory = $true)]
    [string]$FilePath,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

if (-not (Test-Path $FilePath)) {
    Write-Error "File not found: $FilePath"
    exit 1
}

# If isolated profile contains the session, use it; otherwise use active profile from colab_switch.ps1
$isolatedProfile = "C:\Users\xps\.colab_munhyeongdo4"
$isolatedSessions = Join-Path $isolatedProfile ".config\colab-cli\sessions.json"
if ((Test-Path $isolatedSessions) -and ((Get-Content $isolatedSessions -Raw) -match [regex]::Escape($Session))) {
    $env:USERPROFILE = $isolatedProfile
}
# 1. Read original python script content
$originalCode = [System.IO.File]::ReadAllText($FilePath, [System.Text.Encoding]::UTF8)

# 2. Package and upload workspace zip directly via colab upload
$tempDir = Join-Path $PSScriptRoot "..\.temp_colab"
if (-not (Test-Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
}
$zipPath = Join-Path $tempDir "workspace.zip"

$pyExe = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $pyExe)) {
    $pyExe = "python"
}

# Create zip bundle locally
$zipScript = @"
import zipfile, os
from pathlib import Path
root = Path.cwd().resolve()
zip_file = Path(r'$zipPath')
targets = ['tdc_studio', 'configs', 'deploy', 'data', 'scripts', 'pyproject.toml', 'README.md', 'models']
with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
    for t in targets:
        tp = root / t
        if tp.is_file():
            zf.write(tp, arcname=t)
        elif tp.is_dir():
            for item in tp.rglob('*'):
                if '__pycache__' in item.parts or item.suffix in ('.pyc', '.pth', '.log', '.npz'):
                    continue
                if item.suffix == '.pt' and not item.name.startswith('best_model'):
                    continue
                if item.is_file() and item.stat().st_size <= 20 * 1024 * 1024:
                    rel_path = item.relative_to(root)
                    zf.write(item, arcname=str(rel_path).replace('\\', '/'))
"@
& $pyExe -c $zipScript

# Upload workspace to remote Colab VM
colab upload -s $Session $zipPath /content/workspace.zip

# 3. Prepare the injection header to override sys.argv and unpack bundle
$argList = @()
foreach ($arg in $ScriptArgs) {
    # Escape single quotes for python string literal
    $escaped = $arg -replace "'", "\'"
    $argList += "'$escaped'"
}
$argsStr = "[" + ($argList -join ", ") + "]"

$header = @"
import sys, os, zipfile
workspace = os.path.abspath('/content/tdc-studio')
if os.path.exists('/content/workspace.zip'):
    with zipfile.ZipFile('/content/workspace.zip', 'r') as zf:
        zf.extractall(workspace)
if workspace not in sys.path:
    sys.path.insert(0, workspace)
os.chdir(workspace)

# Idempotent dependency check for Colab VM
try:
    from scripts.install_deps import verify_active_environment
    _env_info = verify_active_environment()
    if not _env_info.get("all_healthy", False):
        import subprocess
        print("[Colab Auto-Setup] Unconfigured dependencies detected. Running scripts/install_deps.py...")
        subprocess.run([sys.executable, "scripts/install_deps.py"], check=False)
except Exception as _e:
    print(f"[Colab Auto-Setup Warning] {_e}")

sys.argv = ['$($FilePath -replace '\\', '/')'] + $($argsStr)
os.environ["FORCE_CLI_ARGS"] = "1"
os.environ["TDC_REMOTE_EXECUTION"] = "1"
"@


$tempFileName = "temp_exec_" + [System.IO.Path]::GetFileName($FilePath)
$tempFile = Join-Path $tempDir $tempFileName

# Combine header and original code
$combinedCode = $header + "`n`n" + $originalCode
$ansiEncoding = [System.Text.Encoding]::GetEncoding(949)
[System.IO.File]::WriteAllText($tempFile, $combinedCode, $ansiEncoding)

# 4. Execute using colab exec
Write-Host "Executing '$FilePath' on Colab session '$Session' with arguments: $($ScriptArgs -join ' ')..." -ForegroundColor Cyan
try {
    # Set encoding to UTF-8 for Colab output and file reading on Windows
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUTF8 = '1'
    colab exec -s $Session -f $tempFile --timeout 7200.0
} finally {
    # Clean up temp file
    if (Test-Path $tempFile) {
        Remove-Item -Path $tempFile -Force
    }
}
