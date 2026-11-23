@echo off
setlocal
cd /d "%~dp0"
set "UAVID_DATA_ROOT=%~dp0data\official"
set "LAB_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "LAB_PYTHON=%~dp0.venv\Scripts\python.exe"
"%LAB_PYTHON%" -c "import torch,numpy,PIL,matplotlib,jupyterlab,ipykernel"
if errorlevel 1 (
  echo Eksik ortam. docs\notebook-kullanim.md dosyasindaki kurulum adimlarini izleyin.
  pause
  exit /b 1
)
"%LAB_PYTHON%" -m jupyterlab --ServerApp.root_dir="%~dp0." notebooks/01_UAVid_Yerel_ve_Kaggle.ipynb
if errorlevel 1 pause
