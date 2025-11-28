# Stop all running API + Streamlit servers
# Run from project root: .\scripts\stop.ps1

Write-Host "Stopping Marketing Intelligence Agent services..." -ForegroundColor Cyan

# Kill processes by port
$ports = @(8000, 8500, 8501, 8502)
$killed = 0

foreach ($port in $ports) {
    $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conn) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "  Stopped process on port $port" -ForegroundColor Yellow
        $killed++
    }
}

# Also try to kill any lingering streamlit/uvicorn python processes
Get-Process -Name python -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
        if ($cmdLine -match "streamlit|uvicorn") {
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
            Write-Host "  Stopped Python process: $($_.Id)" -ForegroundColor Yellow
            $killed++
        }
    } catch {}
}

if ($killed -eq 0) {
    Write-Host "No running services found" -ForegroundColor Gray
} else {
    Write-Host "Stopped $killed process(es)" -ForegroundColor Green
}

