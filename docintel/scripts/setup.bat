@echo off
REM Quick setup script for DocIntel (Windows)

echo ======================================
echo DocIntel Setup
echo ======================================
echo.

REM Check Python
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.10 or higher.
    exit /b 1
)
echo [OK] Python found

REM Create virtual environment
echo.
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate and install
echo.
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
pip install -e . >nul 2>&1
echo [OK] Dependencies installed

REM Create .env
echo.
if not exist ".env" (
    echo Creating .env file...
    copy .env.example .env >nul
    echo [OK] .env file created
    echo.
    echo IMPORTANT: Edit .env and add your API key:
    echo    - ANTHROPIC_API_KEY (for Claude)
    echo    - or OPENAI_API_KEY (for GPT)
) else (
    echo [OK] .env file already exists
)

REM Check Docker
echo.
echo Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Docker not found. You'll need to install Qdrant separately.
    echo           Or use: docker-compose up -d
) else (
    echo [OK] Docker is installed

    REM Start Qdrant
    docker ps | findstr qdrant >nul 2>&1
    if errorlevel 1 (
        echo.
        echo Starting Qdrant...
        docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant >nul 2>&1
        echo [OK] Qdrant started
    ) else (
        echo [OK] Qdrant is already running
    )
)

echo.
echo ======================================
echo Setup Complete!
echo ======================================
echo.
echo Next steps:
echo 1. Edit .env and add your API key
echo 2. Run: venv\Scripts\activate
echo 3. Test: docintel health
echo 4. Upload: docintel upload document.pdf
echo 5. Query: docintel query "Your question?"
echo.
echo Or start the API server:
echo   docintel serve
echo.
echo Documentation: http://localhost:8000/docs
echo.

pause
