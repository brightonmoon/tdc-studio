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

# 1. Read original python script content
$originalCode = [System.IO.File]::ReadAllText($FilePath, [System.Text.Encoding]::UTF8)

# 2. Prepare the injection header to override sys.argv
$argList = @()
foreach ($arg in $ScriptArgs) {
    # Escape single quotes for python string literal
    $escaped = $arg -replace "'", "\'"
    $argList += "'$escaped'"
}
$argsStr = "[" + ($argList -join ", ") + "]"

# Prepend sys.argv override and disable Jupyter flag
$header = @"
import sys
import os
sys.argv = ['$($FilePath -replace '\\', '\\')'] + $($argsStr)
os.environ["FORCE_CLI_ARGS"] = "1"
"@

# 3. Create a temporary file in the workspace
$tempDir = Join-Path $PSScriptRoot "..\.temp_colab"
if (-not (Test-Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
}
$tempFileName = "temp_exec_" + [System.IO.Path]::GetFileName($FilePath)
$tempFile = Join-Path $tempDir $tempFileName

# Combine header and original code
$combinedCode = $header + "`n`n" + $originalCode
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($tempFile, $combinedCode, $utf8NoBom)

# 4. Execute using colab exec
Write-Host "Executing '$FilePath' on Colab session '$Session' with arguments: $($ScriptArgs -join ' ')..." -ForegroundColor Cyan
try {
    # Set encoding to UTF-8 for Colab output
    $env:PYTHONIOENCODING = 'utf-8'
    colab exec -s $Session -f $tempFile
} finally {
    # Clean up temp file
    if (Test-Path $tempFile) {
        Remove-Item -Path $tempFile -Force
    }
}
