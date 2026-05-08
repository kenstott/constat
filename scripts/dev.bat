@echo off
:: Development launcher - starts constat server and UI dev server
:: Usage: scripts\dev.bat [config_file]
::
:: Examples:
::   scripts\dev.bat                    Uses demo\config.yaml
::   scripts\dev.bat my\config.yaml     Uses custom config
::
:: Controls:
::   Ctrl+C - Stop both services and exit

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "CONFIG_FILE=%~1"
if "%CONFIG_FILE%"=="" set "CONFIG_FILE=demo\config.yaml"

cd /d "%PROJECT_ROOT%"

:: Create log directory
if not exist .logs mkdir .logs

set "SERVER_LOG=.logs\server.log"
set "UI_LOG=.logs\ui.log"

echo Constat Development Environment
echo ================================
echo Config: %CONFIG_FILE%
echo.

:: Start constat server in background
echo Starting constat server...
echo Server log: %SERVER_LOG%
start /b cmd /c "constat serve -c %CONFIG_FILE% --debug > %SERVER_LOG% 2>&1"

:: Give server a moment to start
timeout /t 3 /nobreak > nul

:: Start UI dev server in background
echo Starting UI dev server...
echo UI log: %UI_LOG%
start /b cmd /c "cd constat-ui && npm run dev -- --port 5173 --strictPort > ..\%UI_LOG% 2>&1"

:: Give UI a moment to start
timeout /t 3 /nobreak > nul

echo.
echo Both services running:
echo   - Server: http://localhost:8000 (API)
echo   - UI:     http://localhost:5173 (Vite)
echo.
echo Logs:
echo   - Server: %SERVER_LOG%
echo   - UI:     %UI_LOG%
echo.
echo Press Ctrl+C to stop both services
echo.

:: Wait for user to press Ctrl+C
:loop
timeout /t 2 /nobreak > nul
goto loop