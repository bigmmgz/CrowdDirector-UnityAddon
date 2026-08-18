@echo off
REM CrowdDirector sidecar - Windows launcher.
REM Creates a local venv on first run, installs requirements, then serves on ws://localhost:8765.

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv || goto :fail
    ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
    echo Installing requirements ^(this downloads PyTorch, a few hundred MB, once^)...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
)

if "%ANTHROPIC_API_KEY%"=="" (
    echo.
    echo   ANTHROPIC_API_KEY is not set.
    echo   The per-tick director does NOT need it - the model runs locally.
    echo   It is required only to generate a scene from a description and to
    echo   interpret free-text instructions.
    echo.
    echo   Set it with:  set ANTHROPIC_API_KEY=sk-ant-...
    echo.
)

".venv\Scripts\python.exe" crowd_director_server.py
goto :eof

:fail
echo.
echo Setup failed. Check that Python 3.10+ is installed and on PATH.
exit /b 1
