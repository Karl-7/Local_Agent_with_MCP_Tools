@echo off
setlocal
cd /d "%~dp0"
title Sigge's AI Assistant

rem ============ Settings ============
rem Chat model: qwen3.5:4b (~3.4GB, smarter)  /  qwen3.5:2b (~2.7GB, lighter)  /  qwen3.5:0.8b (~1GB, fastest)
set "GAME_AI_MODEL=qwen3.5:0.8b"
rem Embedding model, only used when pruning memory: nomic-embed-text (~270MB)
set "GAME_AI_EMBED_MODEL=nomic-embed-text"
rem Content filter for web results: llama-guard3:1b (~1.6GB). Chat model + filter together need roughly 5-6GB of RAM.
set "GAME_AI_GUARD_MODEL=llama-guard3:1b"
rem The user's name: the assistant calls the user this and files its memories under it (ASCII letters)
set "GAME_AI_USER_NAME=Sigge"
rem 1 = show the assistant in its own app window, 0 = open it in the browser
set "GAME_AI_WINDOW=1"
rem Context length: bigger = remembers more, uses more RAM
set "GAME_AI_NUM_CTX=8192"
rem 1 = run on CPU only (does not steal VRAM from the game), 0 = use GPU if available
set "GAME_AI_CPU_ONLY=0"
rem Max characters of memory put into the prompt (default 3000; set low, e.g. 200, to test pruning)
set "GAME_AI_MEMORY_BUDGET=3000"
rem ==================================

rem ---- 1. Ollama ----
set "OLLAMA=ollama"
where ollama >nul 2>nul
if errorlevel 1 (
  if not exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    echo [1/7] Installing Ollama ...
    winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
  )
  set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
)

rem ---- 2. Make sure the Ollama server is running ----
curl -s http://127.0.0.1:11434 >nul 2>nul
if errorlevel 1 (
  echo [2/7] Starting Ollama ...
  start "" /min "%OLLAMA%" serve
  timeout /t 6 /nobreak >nul
)

rem ---- 3. Download the models (only slow the first time) ----
echo [3/7] Checking model %GAME_AI_MODEL% ...
"%OLLAMA%" pull %GAME_AI_MODEL%
if errorlevel 1 (
  echo Failed to download the model. Check your internet and try again.
  pause
  exit /b 1
)
echo [3/7] Checking content filter %GAME_AI_GUARD_MODEL% ...
"%OLLAMA%" pull %GAME_AI_GUARD_MODEL%
if errorlevel 1 (
  echo Failed to download the content filter. Check your internet and try again.
  pause
  exit /b 1
)
echo [3/7] Checking embedding model %GAME_AI_EMBED_MODEL% ...
"%OLLAMA%" pull %GAME_AI_EMBED_MODEL%
if errorlevel 1 (
  echo Failed to download the embedding model. Check your internet and try again.
  pause
  exit /b 1
)

rem ---- 4. uv (manages Python automatically) ----
set "UV=uv"
where uv >nul 2>nul
if errorlevel 1 (
  if not exist "%USERPROFILE%\.local\bin\uv.exe" (
    echo [4/7] Installing uv ...
    powershell -ExecutionPolicy Bypass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
  )
  set "UV=%USERPROFILE%\.local\bin\uv.exe"
)

rem ---- 5. Node.js (needed by the memory tool) ----
where npx >nul 2>nul
if errorlevel 1 (
  if not exist "%ProgramFiles%\nodejs\npx.cmd" (
    echo [5/7] Installing Node.js ...
    winget install -e --id OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
  )
)

rem ---- 6. Deno (sandbox for the Python code tool) ----
where deno >nul 2>nul
if errorlevel 1 (
  if not exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\deno.exe" (
    if not exist "%USERPROFILE%\.deno\bin\deno.exe" (
      echo [6/7] Installing Deno ...
      winget install -e --id DenoLand.Deno --accept-source-agreements --accept-package-agreements
    )
  )
)

rem ---- 7. Desktop shortcut with icon (created once) ----
powershell -NoProfile -Command "$d=[Environment]::GetFolderPath('Desktop'); $l=Join-Path $d 'Sigges AI Assistant.lnk'; if(-not (Test-Path $l)){ $s=(New-Object -ComObject WScript.Shell).CreateShortcut($l); $s.TargetPath='%~dp0start.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; if(Test-Path '%~dp0icon.jpg'){ $s.IconLocation='%~dp0icon.jpg' }; $s.Save() }" >nul 2>nul

echo.
echo [7/7] Starting the assistant, its window will open by itself.
echo You can minimize this black window. Closing the app window quits everything.
echo.
"%UV%" run --python 3.12 Sigges_AI.py
if errorlevel 1 pause