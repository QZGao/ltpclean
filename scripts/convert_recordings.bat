@echo off
setlocal enabledelayedexpansion

rem Adjust these if you have extra options
set INPUT_ROOT=datatrain\recordings
set OUTPUT_ROOT=eval_data\recordings
set RESOLUTION=128
set MAX_FRAMES=64
set CROP_SEED=42
set MAX_ACTION=6

if not exist %OUTPUT_ROOT% mkdir %OUTPUT_ROOT%

for /d %%R in (%INPUT_ROOT%\*) do (
  set "RECORDING=%%~nR"
  set "FRAME_DIR=%%R\frames"
  if not exist "!FRAME_DIR!\*.png" (
    echo Skipping %%R: no PNG frames found in "frames" subdirectory.
  ) else (
    echo Processing %%R -> %OUTPUT_ROOT%\!RECORDING!-frameArray.txt
    set CMD=python build_frame_dataset.py --input-dir "!FRAME_DIR!" --output "%OUTPUT_ROOT%\!RECORDING!-frameArray.txt" --resolution %RESOLUTION%
    if defined MAX_FRAMES set "CMD=!CMD! --max-frames %MAX_FRAMES%"
    if defined CROP_SEED set "CMD=!CMD! --crop-seed %CROP_SEED%"
    !CMD!
    if defined MAX_ACTION (
      python scripts/clamp_actions.py --input "%OUTPUT_ROOT%\!RECORDING!-frameArray.txt" --max-action %MAX_ACTION%
    )
  )
)

echo Conversion complete.
