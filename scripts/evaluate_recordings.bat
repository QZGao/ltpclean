@echo off
setlocal enabledelayedexpansion

rem Check for --skip-existing flag
set SKIP_EXISTING=0
for %%A in (%*) do (
  if /i "%%A"=="--skip-existing" set SKIP_EXISTING=1
)

rem Adjust these paths or arguments as needed
set INPUT_ROOT=eval_data\recordings3
set OUTPUT_ROOT=eval_data\metrics3
set "MODEL_CHECKPOINT=ckpt\model_epoch75_20251122_09.pth"
set "VAE_CHECKPOINT=ckpt\VAE\vae_epoch6_20251112_03.pth"
set "VAM_CHECKPOINT=ckpt\vam_model.pt"
set "EXTRA_ARGS=--performance-horizon 128 --per-frame-metrics --min-frame-change 0.003"
rem Remove --skip-existing from args passed to Python
for %%A in (%*) do (
  if /i not "%%A"=="--skip-existing" set "EXTRA_ARGS=!EXTRA_ARGS! %%A"
)
if defined MODEL_CHECKPOINT set "EXTRA_ARGS=!EXTRA_ARGS! --model-ckpt %MODEL_CHECKPOINT%"
if defined VAE_CHECKPOINT set "EXTRA_ARGS=!EXTRA_ARGS! --vae-ckpt %VAE_CHECKPOINT%"
if defined VAM_CHECKPOINT set "EXTRA_ARGS=!EXTRA_ARGS! --vam-checkpoint %VAM_CHECKPOINT%"

if not exist %OUTPUT_ROOT% mkdir %OUTPUT_ROOT%

for %%F in (%INPUT_ROOT%\*-frameArray.txt) do (
  set "BASE=%%~nF"
  set "OUTPUT=%OUTPUT_ROOT%\!BASE!-metrics.json"
  set "SHOULD_RUN=1"
  
  if !SKIP_EXISTING!==1 (
    if exist "!OUTPUT!" (
      echo Skipping %%F - output already exists: !OUTPUT!
      set "SHOULD_RUN=0"
    )
  )
  
  if !SHOULD_RUN!==1 (
    echo Evaluating %%F
    python evaluate.py --dataset "%%F" --output "!OUTPUT!" !EXTRA_ARGS!
  )
)

echo Evaluation pass complete.
