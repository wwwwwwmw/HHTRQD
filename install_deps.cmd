@echo off
setlocal

REM Install Python deps into the project's venv (no Activate.ps1 needed).

set PYEXE=%~dp0.venv\Scripts\python.exe
if not exist "%PYEXE%" (
  echo Creating venv...
  py -m venv "%~dp0.venv"
)

"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install -r "%~dp0requirements.txt"

echo Done.
