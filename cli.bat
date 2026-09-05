@echo off
chcp 65001 >nul 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0cli.ps1" %*
exit /b %ERRORLEVEL%
