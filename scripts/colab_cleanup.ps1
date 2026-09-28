# colab_cleanup.ps1 - Colab Resource Cleanup & Automated Session Reclaim
# Target Account: munhyeongdo4@gmail.com
param(
    [Parameter(Position = 0)]
    [ValidateSet("list", "stop", "prune")]
    [string]$Action = "list",

    [Parameter(Position = 1)]
    [string]$Session = "",

    [switch]$All,
    [switch]$Force
)

# 1. Enforce Colab CLI profile isolation to munhyeongdo4 account
$TargetAccount = "munhyeongdo4@gmail.com"
$ProfileDir = "C:\Users\xps\.colab_munhyeongdo4"

if (-not (Test-Path $ProfileDir)) {
    Write-Warning "Profile directory '$ProfileDir' not found. Falling back to default USERPROFILE."
} else {
    $env:USERPROFILE = $ProfileDir
}

function Show-Header {
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "  Colab Resource Management & Session Cleanup Utility" -ForegroundColor Cyan
    Write-Host "  Account: $TargetAccount" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Cyan
}

function Get-ColabSessions {
    # Returns raw string output and parses session names
    $rawOutput = colab sessions 2>&1
    $sessionList = @()
    foreach ($line in $rawOutput) {
        # Sessions are formatted with bullets or table rows
        # e.g., "* dti-gpu" or "  dti-gpu (GPU, RUNNING)"
        if ($line -match "^\s*[\*\-]?\s*([a-zA-Z0-9_\-]+)\s+.*(RUNNING|IDLE|ASSIGNED|ACTIVE)") {
            $sessionList += $matches[1]
        } elseif ($line -match "^\s*[\*\-]\s*([a-zA-Z0-9_\-]+)\s*$") {
            $sessionList += $matches[1]
        }
    }
    return [PSCustomObject]@{
        RawOutput = $rawOutput
        Sessions = $sessionList
    }
}

Show-Header

switch ($Action) {
    "list" {
        Write-Host "`n[1] Checking active Colab sessions for $TargetAccount..." -ForegroundColor Yellow
        $info = Get-ColabSessions
        Write-Host ($info.RawOutput -join "`n")
        
        if ($info.Sessions.Count -eq 0) {
            Write-Host "`n✔ No active sessions currently consuming compute units." -ForegroundColor Green
        } else {
            Write-Host "`n⚠ Active Sessions Detected ($($info.Sessions.Count)): $($info.Sessions -join ', ')" -ForegroundColor Red
            Write-Host "Run '.\scripts\colab_cleanup.ps1 stop -All' to reclaim all compute resources." -ForegroundColor Cyan
        }
    }

    "stop" {
        $info = Get-ColabSessions
        $targets = @()

        if ($All) {
            if ($info.Sessions.Count -eq 0) {
                Write-Host "`n✔ No active sessions found to stop." -ForegroundColor Green
                exit 0
            }
            $targets = $info.Sessions
        } elseif ($Session) {
            $targets = @($Session)
        } else {
            Write-Error "Please specify a session name with -Session <name> or pass -All to stop all active sessions."
            exit 1
        }

        Write-Host "`n[Target Sessions to Stop]: $($targets -join ', ')" -ForegroundColor Yellow

        if (-not $Force) {
            $confirm = Read-Host "Are you sure you want to stop and release the above session(s)? (y/N)"
            if ($confirm -ne 'y' -and $confirm -ne 'Y') {
                Write-Host "Session termination cancelled by user." -ForegroundColor Gray
                exit 0
            }
        }

        foreach ($s in $targets) {
            Write-Host "Stopping Colab session '$s'..." -ForegroundColor Cyan
            colab stop -s $s
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✔ Successfully stopped '$s'." -ForegroundColor Green
            } else {
                Write-Warning "Failed to stop session '$s' or session was already terminated."
            }
        }

        # Verify final state
        Write-Host "`n[Verification] Checking remaining sessions..." -ForegroundColor Yellow
        colab sessions
    }

    "prune" {
        Write-Host "`n[1] Pruning stale Colab session cache..." -ForegroundColor Yellow
        colab sessions

        # Clean local temporary zip files and bundle cache
        $tempColab = Join-Path $PSScriptRoot "..\.temp_colab"
        if (Test-Path $tempColab) {
            Write-Host "[2] Removing local temporary packaging directory ($tempColab)..." -ForegroundColor Cyan
            Remove-Item -Path $tempColab -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "✔ Cleaned up local temporary deployment artifacts." -ForegroundColor Green
        }

        Write-Host "`n✔ Colab environment cleanup complete for $TargetAccount." -ForegroundColor Green
    }
}
