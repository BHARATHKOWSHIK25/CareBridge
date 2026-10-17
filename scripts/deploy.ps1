<#
.SYNOPSIS
    CareBridge Deployment Orchestrator for Windows PowerShell.
.PARAMETER Prod
    Deploy using production compose with Nginx reverse proxy.
.PARAMETER Demo
    Deploy using standard compose in demo mode.
#>
param(
    [switch]$Prod,
    [switch]$Demo
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "          CareBridge Deployment Orchestrator              " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not in PATH. Please install Docker Desktop: https://docs.docker.com/desktop/setup/install/windows-install/"
    exit 1
}

# 2. Check compose file
$composeFile = "docker-compose.yml"
if ($Prod) {
    $composeFile = "docker-compose.prod.yml"
}

# 3. Check .env file
if (-not (Test-Path ".env")) {
    Write-Host "[!] .env file not found." -ForegroundColor Yellow
    if ($Prod) {
        Write-Host "    Copying .env.production.example to .env..." -ForegroundColor Yellow
        Copy-Item .env.production.example .env
        Write-Warning "Please edit .env to configure LLM_BASE_URL and NURSE_PASSWORD before deploying in production!"
    } else {
        Write-Host "    Copying .env.example to .env..." -ForegroundColor Yellow
        Copy-Item .env.example .env
    }
}

# 4. Read .env parameters
$envContent = Get-Content ".env" | Where-Object { $_ -match '^[^#].*=' }
$envMap = @{}
foreach ($line in $envContent) {
    $parts = $line.Split('=', 2)
    if ($parts.Length -eq 2) {
        $envMap[$parts[0].Trim()] = $parts[1].Trim()
    }
}

$runMode = if ($envMap.ContainsKey("RUN_MODE")) { $envMap["RUN_MODE"] } else { "demo" }
if ($Prod -or $runMode -eq "live") {
    Write-Host "[*] Validating live mode configuration..." -ForegroundColor Cyan
    $llmUrl = if ($envMap.ContainsKey("LLM_BASE_URL")) { $envMap["LLM_BASE_URL"] } else { "" }
    $nursePw = if ($envMap.ContainsKey("NURSE_PASSWORD")) { $envMap["NURSE_PASSWORD"] } else { "" }

    if ([string]::IsNullOrWhiteSpace($llmUrl) -or $llmUrl -eq "https://your-llm-endpoint.example.com/v1") {
        Write-Error "RUN_MODE=live requires a valid LLM_BASE_URL in .env."
        exit 1
    }
    if ([string]::IsNullOrWhiteSpace($nursePw) -or $nursePw -eq "generate_a_strong_password_here") {
        Write-Error "RUN_MODE=live requires a strong NURSE_PASSWORD in .env."
        exit 1
    }
}

# 5. Build and launch
Write-Host "[*] Using compose file: $composeFile" -ForegroundColor Cyan
Write-Host "[*] Building and starting CareBridge containers..." -ForegroundColor Cyan

docker compose -f $composeFile build
docker compose -f $composeFile up -d

# 6. Verify health
$port = if ($Prod) { "80" } else { "8000" }
if ($envMap.ContainsKey("PORT")) { $port = $envMap["PORT"] }
if ($Prod -and $envMap.ContainsKey("HTTP_PORT")) { $port = $envMap["HTTP_PORT"] }

$healthUrl = "http://127.0.0.1:$port/health"
Write-Host "[*] Probing health endpoint at $healthUrl..." -ForegroundColor Cyan

$attempts = 0
$healthy = $false
while ($attempts -lt 30) {
    try {
        $res = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($res.status -eq "ok") {
            $healthy = $true
            break
        }
    } catch {
        # Retry
    }
    Write-Host "." -NoNewline
    Start-Sleep -Seconds 2
    $attempts++
}
Write-Host ""

if ($healthy) {
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host " [OK] CareBridge is successfully deployed and healthy!" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "  - Patient Chat:        http://localhost:$port/"
    Write-Host "  - Nurse Handoff Queue:  http://localhost:$port/nurse"
    Write-Host "  - Health Endpoint:      http://localhost:$port/health"
    Write-Host "  - API Documentation:    http://localhost:$port/docs"
    Write-Host "=========================================================="
} else {
    Write-Host "==========================================================" -ForegroundColor Yellow
    Write-Host " [!] Container started, but healthcheck timed out." -ForegroundColor Yellow
    Write-Host " Check logs with: docker compose -f $composeFile logs -f" -ForegroundColor Yellow
    Write-Host "=========================================================="
}
