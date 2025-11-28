# Run all tests
# Run from project root: .\scripts\test.ps1

param(
    [switch]$Coverage,  # Generate coverage report
    [switch]$E2E        # Include E2E tests (requires API keys)
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "🧪 Running Tests..." -ForegroundColor Cyan

$TestArgs = @("tests/unit", "tests/integration", "-v")

if ($E2E) {
    $TestArgs += "tests/e2e"
    Write-Host "Including E2E tests (requires API keys)" -ForegroundColor Yellow
}

if ($Coverage) {
    $TestArgs += @("--cov=src", "--cov-report=html", "--cov-report=term")
    Write-Host "Coverage report will be generated in htmlcov/" -ForegroundColor Yellow
}

.\.venv\Scripts\python.exe -m pytest @TestArgs

if ($Coverage) {
    Write-Host "`nOpen htmlcov/index.html to view coverage report" -ForegroundColor Green
}

