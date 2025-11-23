@echo off
setlocal enabledelayedexpansion

rem Adjust these if you have extra options
set INPUT_ROOT=datatrain\recordings
set OUTPUT_ROOT=eval_data\recordings
set RESOLUTION=128

if not exist %OUTPUT_ROOT% mkdir %OUTPUT_ROOT%

for /d %%R in (%INPUT_ROOT%\*) do (
  set "RECORDING=%%~nR"
  set "FRAME_DIR=%%R\frames"
  dir /b "!FRAME_DIR!\*.png" >nul 2>&1
  if errorlevel 1 (
    echo Skipping %%R: no PNG frames found in "frames" subdirectory.
  ) else (
    echo Processing %%R -> %OUTPUT_ROOT%\!RECORDING!-frameArray.txt
    python build_frame_dataset.py --input-dir "!FRAME_DIR!" --output "%OUTPUT_ROOT%\!RECORDING!-frameArray.txt" --resolution %RESOLUTION%
  )
)

echo Conversion complete.
