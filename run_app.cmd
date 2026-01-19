@echo off
setlocal

REM Run Streamlit app using the project's venv Python (no Activate.ps1 needed).

set PYEXE=%~dp0.venv\Scripts\python.exe
if not exist "%PYEXE%" (
  echo ERROR: Virtualenv not found at .venv\Scripts\python.exe
  echo Create it with: py -m venv .venv
  echo Then install deps: .\install_deps.cmd
  exit /b 1
)

"%PYEXE%" -m streamlit run "%~dp0streamlit_app.py" --server.port 8501
