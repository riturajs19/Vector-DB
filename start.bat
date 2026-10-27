@echo off
echo ========================================================
echo   Starting VectorDB + RAG System
echo ========================================================

:: Check if db.exe exists
if not exist "db.exe" (
    echo [ERROR] db.exe not found! Running build.bat first...
    call build.bat
)

:: Start C++ Vector Database Server
echo Starting C++ Vector Database server on http://localhost:8080 ...
start "VectorDB Server" /min cmd /c "db.exe"

:: Wait for server initialization
timeout /t 2 /nobreak > nul

:: Start Streamlit UI
echo Starting Streamlit Web App...
streamlit run app.py
