# colab_switch.ps1 - Colab Account Switcher for Windows PowerShell
param(
    [Parameter(Position = 0)]
    [ValidateSet("list", "use", "save", "new", "delete")]
    [string]$Action = "list",

    [Parameter(Position = 1)]
    [string]$AccountName
)

$ConfigDir = Join-Path $HOME ".config\colab-cli"
$TokenPath = Join-Path $ConfigDir "token.json"

if (-not (Test-Path $ConfigDir)) {
    New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null
}

function Get-ActiveAccount {
    if (Test-Path $TokenPath) {
        # Check if we have backups that match the active token's hash
        $activeHash = (Get-FileHash $TokenPath -Algorithm MD5).Hash
        $backups = Get-ChildItem -Path $ConfigDir -Filter "token_*.json"
        foreach ($b in $backups) {
            $bHash = (Get-FileHash $b.FullName -Algorithm MD5).Hash
            if ($bHash -eq $activeHash) {
                return $b.BaseName.Replace("token_", "")
            }
        }
        return "[Unknown / Unsaved Account]"
    }
    return "[No Active Login]"
}

switch ($Action) {
    "list" {
        Write-Host "--- Colab CLI Accounts ---" -ForegroundColor Cyan
        $active = Get-ActiveAccount
        Write-Host "Active Account: " -NoNewline
        Write-Host $active -ForegroundColor Green
        
        Write-Host "`nSaved Accounts:" -ForegroundColor Yellow
        $backups = Get-ChildItem -Path $ConfigDir -Filter "token_*.json"
        if ($backups.Count -eq 0) {
            Write-Host "  (No saved accounts found)" -ForegroundColor Gray
        } else {
            foreach ($b in $backups) {
                $name = $b.BaseName.Replace("token_", "")
                if ($name -eq $active) {
                    Write-Host "  * $name (active)" -ForegroundColor Green
                } else {
                    Write-Host "    $name"
                }
            }
        }
        Write-Host "`nUsage Examples:" -ForegroundColor Gray
        Write-Host "  .\scripts\colab_switch.ps1 list            - Show all accounts and active status"
        Write-Host "  .\scripts\colab_switch.ps1 save main       - Save current login as 'main'"
        Write-Host "  .\scripts\colab_switch.ps1 new sub1        - Authenticate and register a new account as 'sub1'"
        Write-Host "  .\scripts\colab_switch.ps1 use sub1        - Switch the active token to 'sub1'"
        Write-Host "  .\scripts\colab_switch.ps1 delete sub1     - Delete 'sub1' account credentials"
    }

    "use" {
        if (-not $AccountName) {
            Write-Error "Please specify an account name to use."
            exit 1
        }
        $targetFile = Join-Path $ConfigDir "token_$AccountName.json"
        if (-not (Test-Path $targetFile)) {
            Write-Error "Account '$AccountName' not found. Use 'list' to see available accounts."
            exit 1
        }
        Copy-Item -Path $targetFile -Destination $TokenPath -Force
        Write-Host "Successfully switched active token to account '$AccountName'!" -ForegroundColor Green
    }

    "save" {
        if (-not $AccountName) {
            Write-Error "Please specify an account name to save as."
            exit 1
        }
        if (-not (Test-Path $TokenPath)) {
            Write-Error "No active login found (token.json does not exist). Run a colab command first to authenticate."
            exit 1
        }
        $targetFile = Join-Path $ConfigDir "token_$AccountName.json"
        Copy-Item -Path $TokenPath -Destination $targetFile -Force
        Write-Host "Saved current login as account '$AccountName'!" -ForegroundColor Green
    }

    "new" {
        if (-not $AccountName) {
            Write-Error "Please specify a name for the new account."
            exit 1
        }
        $targetFile = Join-Path $ConfigDir "token_$AccountName.json"
        
        # Temp backup of current active token
        $tempBackup = Join-Path $ConfigDir "token.json.tmp_bak"
        if (Test-Path $TokenPath) {
            Move-Item -Path $TokenPath -Destination $tempBackup -Force
        }
        
        Write-Host "Initiating authentication flow for new account '$AccountName'..." -ForegroundColor Cyan
        Write-Host "Please follow the instructions below to authenticate." -ForegroundColor Yellow
        
        try {
            # Run a dummy command that requires auth (like colab sessions) to trigger login
            colab sessions
            
            if (Test-Path $TokenPath) {
                Copy-Item -Path $TokenPath -Destination $targetFile -Force
                Write-Host "`nSuccessfully authenticated and saved as account '$AccountName'!" -ForegroundColor Green
            } else {
                Write-Error "Authentication appeared to fail or was cancelled."
            }
        } finally {
            # Restore original active token if new login did not complete
            if (-not (Test-Path $TokenPath) -and (Test-Path $tempBackup)) {
                Move-Item -Path $tempBackup -Destination $TokenPath -Force
            } elseif (Test-Path $tempBackup) {
                Remove-Item -Path $tempBackup -Force
            }
        }
    }

    "delete" {
        if (-not $AccountName) {
            Write-Error "Please specify an account name to delete."
            exit 1
        }
        $targetFile = Join-Path $ConfigDir "token_$AccountName.json"
        if (-not (Test-Path $targetFile)) {
            Write-Error "Account '$AccountName' not found."
            exit 1
        }
        Remove-Item -Path $targetFile -Force
        Write-Host "Deleted account '$AccountName'." -ForegroundColor Green
    }
}
