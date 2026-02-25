@echo off
chcp 65001 >nul 2>&1
powershell -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; & '%~dp0install.ps1'"
pause
