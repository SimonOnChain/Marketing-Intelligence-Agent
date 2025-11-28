# Start API + Streamlit servers
# Run from project root: .\scripts\start.ps1

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Marketing Intelligence Agent" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host ".env file not found. Run .\scripts\setup.ps1 first" -ForegroundColor Red
    exit 1
}

# Check AWS configuration
Write-Host "Checking AWS services..." -ForegroundColor Yellow
$awsCheck = & python -c "
from src.config.settings import get_settings
s = get_settings()
print(f'CACHE={s.cache_enabled}')
print(f'DYNAMODB={s.use_dynamodb_cache}')
print(f'BEDROCK={s.bedrock_enabled}')
print(f'BEDROCK_INTENT={s.use_bedrock_for_intent}')
" 2>$null

if ($awsCheck) {
    Write-Host ""
    Write-Host "  AWS Services Status:" -ForegroundColor White
    foreach ($line in $awsCheck) {
        $parts = $line -split "="
        $service = $parts[0]
        $status = $parts[1]
        if ($status -eq "True") {
            Write-Host "    [ON]  $service" -ForegroundColor Green
        } else {
            Write-Host "    [OFF] $service" -ForegroundColor Gray
        }
    }
    Write-Host ""
}

# Kill any existing Python/Streamlit/Uvicorn processes on our ports
Write-Host "Cleaning up old processes..." -ForegroundColor Yellow
Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -match "streamlit|uvicorn|8500|8000|8501|8502"
} | Stop-Process -Force -ErrorAction SilentlyContinue

# Also kill by port
$ports = @(8000, 8500, 8501, 8502)
foreach ($port in $ports) {
    $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conn) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "  Killed process on port $port" -ForegroundColor Gray
    }
}
Start-Sleep -Seconds 1

# Start API in background
Write-Host "Starting FastAPI server on port 8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; Write-Host 'FastAPI Server' -ForegroundColor Green; python -m uvicorn src.api.main:app --reload --port 8000"

# Wait for API to start
Start-Sleep -Seconds 3

# Start Streamlit
Write-Host "Starting Streamlit on port 8500..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; Write-Host 'Streamlit UI' -ForegroundColor Green; python -m streamlit run src/app/streamlit_app.py --server.port 8500"

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Servers Started Successfully!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Local URLs:" -ForegroundColor White
Write-Host "    API:         http://localhost:8000" -ForegroundColor Cyan
Write-Host "    API Docs:    http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "    Streamlit:   http://localhost:8500" -ForegroundColor Cyan
Write-Host ""
Write-Host "  AWS Endpoints:" -ForegroundColor White
Write-Host "    AWS Status:  http://localhost:8000/aws/status" -ForegroundColor Cyan
Write-Host "    Cache Stats: http://localhost:8000/cache/stats" -ForegroundColor Cyan
Write-Host "    CloudWatch:  https://eu-central-1.console.aws.amazon.com/cloudwatch/home?region=eu-central-1#dashboards:name=MarketingAgent" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Tips:" -ForegroundColor Yellow
Write-Host "    - First query: ~3s (calls LLM)" -ForegroundColor Gray
Write-Host "    - Cached query: ~0.01s (instant!)" -ForegroundColor Gray
Write-Host "    - Press Ctrl+C in each window to stop" -ForegroundColor Gray
Write-Host ""

# Open browser
Start-Sleep -Seconds 2
Start-Process "http://localhost:8500"

