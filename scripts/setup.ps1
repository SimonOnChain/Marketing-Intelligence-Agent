# Marketing Intelligence Agent - Full Setup Script
# Run from project root: .\scripts\setup.ps1

param(
    [switch]$SkipData,      # Skip downloading Kaggle data
    [switch]$SkipIndex,     # Skip Qdrant indexing
    [switch]$StartServers   # Start API + Streamlit after setup
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Marketing Intelligence Agent - Setup Script" -ForegroundColor Cyan
Write-Host ("=" * 50)

# Change to project root
Set-Location $ProjectRoot
Write-Host "Working directory: $ProjectRoot"

# Step 1: Check Python and uv
Write-Host ""
Write-Host "[1/7] Checking dependencies..." -ForegroundColor Yellow
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Install from: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Red
    exit 1
}
Write-Host "uv found" -ForegroundColor Green

# Step 2: Create/sync virtual environment
Write-Host ""
Write-Host "[2/7] Setting up Python environment..." -ForegroundColor Yellow
uv sync --all-extras
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to sync dependencies" -ForegroundColor Red; exit 1 }
uv pip install pytest pytest-asyncio pytest-cov
Write-Host "Dependencies installed" -ForegroundColor Green

# Step 3: Check .env file
Write-Host ""
Write-Host "[3/7] Checking environment configuration..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "Created .env from .env.example - EDIT IT WITH YOUR API KEYS!" -ForegroundColor Yellow
        Write-Host "Required keys: XAI_API_KEY, QDRANT_URL, QDRANT_API_KEY" -ForegroundColor Yellow
    } else {
        Write-Host "No .env or .env.example found" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host ".env file exists" -ForegroundColor Green
}

# Step 4: Download Kaggle data
Write-Host ""
if (-not $SkipData) {
    Write-Host "[4/7] Downloading Olist dataset..." -ForegroundColor Yellow
    if (-not (Test-Path "data/raw/olist_orders_dataset.csv")) {
        if (Get-Command kaggle -ErrorAction SilentlyContinue) {
            kaggle datasets download olistbr/brazilian-ecommerce -p data/raw --unzip
            if ($LASTEXITCODE -ne 0) { 
                Write-Host "Kaggle download failed. Download manually from:" -ForegroundColor Yellow
                Write-Host "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce" -ForegroundColor Yellow
            } else {
                Write-Host "Dataset downloaded" -ForegroundColor Green
            }
        } else {
            Write-Host "Kaggle CLI not found. Download manually from:" -ForegroundColor Yellow
            Write-Host "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce" -ForegroundColor Yellow
            Write-Host "Extract to: data/raw/" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Dataset already exists" -ForegroundColor Green
    }
} else {
    Write-Host "[4/7] Skipping data download (SkipData flag set)" -ForegroundColor Gray
}

# Step 5: Run ETL
Write-Host ""
Write-Host "[5/7] Running ETL pipeline..." -ForegroundColor Yellow
if (Test-Path "data/raw/olist_orders_dataset.csv") {
    if (-not (Test-Path "data/processed/reviews.parquet")) {
        & "$ProjectRoot\.venv\Scripts\python.exe" -m src.data.etl
        if ($LASTEXITCODE -ne 0) { Write-Host "ETL failed" -ForegroundColor Red; exit 1 }
        Write-Host "ETL complete" -ForegroundColor Green
    } else {
        Write-Host "Processed data already exists" -ForegroundColor Green
    }
} else {
    Write-Host "Skipping ETL - raw data not found" -ForegroundColor Yellow
}

# Step 6: Index to Qdrant
Write-Host ""
if (-not $SkipIndex) {
    Write-Host "[6/7] Indexing reviews to Qdrant..." -ForegroundColor Yellow
    if (Test-Path "data/processed/reviews.parquet") {
        if (-not (Test-Path "data/processed/lexical_corpus.jsonl")) {
            & "$ProjectRoot\.venv\Scripts\python.exe" -m src.retrieval.index
            if ($LASTEXITCODE -ne 0) { Write-Host "Indexing failed" -ForegroundColor Red; exit 1 }
            Write-Host "Indexing complete" -ForegroundColor Green
        } else {
            Write-Host "Already indexed (lexical_corpus.jsonl exists)" -ForegroundColor Green
        }
    } else {
        Write-Host "Skipping indexing - processed data not found" -ForegroundColor Yellow
    }
} else {
    Write-Host "[6/7] Skipping indexing (SkipIndex flag set)" -ForegroundColor Gray
}

# Step 7: Run tests
Write-Host ""
Write-Host "[7/7] Running tests..." -ForegroundColor Yellow
& "$ProjectRoot\.venv\Scripts\python.exe" -m pytest tests/unit tests/integration -v --tb=short
if ($LASTEXITCODE -ne 0) { 
    Write-Host "Some tests failed" -ForegroundColor Yellow
} else {
    Write-Host "All tests passed" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host ("=" * 50) -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application:" -ForegroundColor White
Write-Host "  API:       .\scripts\start.ps1" -ForegroundColor Gray
Write-Host ""

# Optionally start servers
if ($StartServers) {
    Write-Host "Starting servers..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; & '$ProjectRoot\.venv\Scripts\python.exe' -m uvicorn src.api.main:app --reload --port 8000"
    Start-Sleep -Seconds 3
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; & '$ProjectRoot\.venv\Scripts\python.exe' -m streamlit run src/app/streamlit_app.py --server.port 8500"
    Write-Host "Servers started in new windows" -ForegroundColor Green
    Write-Host "  API:       http://localhost:8000" -ForegroundColor Gray
    Write-Host "  Streamlit: http://localhost:8500" -ForegroundColor Gray
}
