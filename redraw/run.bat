@echo off
setlocal
set REDRAW_ADDR=127.0.0.1:8082
set HF_HUB_DISABLE_XET=1
"%~dp0.venv\Scripts\python.exe" "%~dp0server.py"
