@echo off
cd /d "%~dp0"
title NeuraType AI - Next Word Prediction Server
echo ======================================================================
echo           NeuraType AI - Next Word Prediction Server
echo ======================================================================
echo.

set PYTHON_CMD=python
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)

echo Using Python: %PYTHON_CMD%
echo Starting server on http://localhost:8000 ...
echo.

%PYTHON_CMD% run_server.py

pause
