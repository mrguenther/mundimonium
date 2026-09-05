@echo off
rem Double-clickable wrapper around launch.ps1, for launching outside
rem an existing PowerShell session (Explorer, cmd.exe).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1"
