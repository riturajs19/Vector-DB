@echo off
echo ========================================================
echo   Building C++ Vector Database Server (db.exe)
echo ========================================================

set "PATH=C:\msys64\ucrt64\bin;%PATH%"

echo Compiling main.cpp with -std=c++17 -O2 -static -lws2_32 ...
g++ -std=c++17 -O2 -static main.cpp -o db.exe -lws2_32

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] db.exe built successfully!
) else (
    echo [FAILED] Compilation failed with error code %ERRORLEVEL%
)
