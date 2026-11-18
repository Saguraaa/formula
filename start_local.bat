@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"
cd /d "%~dp0"
if exist "D:\conda\envs\labelstudioml\python.exe" (
  "D:\conda\envs\labelstudioml\python.exe" web_app.py %*
) else (
  python web_app.py %*
)
if errorlevel 1 pause
