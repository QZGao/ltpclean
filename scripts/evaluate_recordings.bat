@echo off
setlocal enabledelayedexpansion

rem Adjust these paths or arguments as needed
set INPUT_ROOT=eval_data\recordings
set OUTPUT_ROOT=eval_data\metrics
set "VAM_CHECKPOINT=ckpt\vam_model.pt"
set "EXTRA_ARGS=%*"
if defined VAM_CHECKPOINT set "EXTRA_ARGS=!EXTRA_ARGS! --vam-checkpoint %VAM_CHECKPOINT%"

if not exist %OUTPUT_ROOT% mkdir %OUTPUT_ROOT%

for %%F in (%INPUT_ROOT%\*-frameArray.txt) do (
  set "BASE=%%~nF"
  set "OUTPUT=%OUTPUT_ROOT%\!BASE!-metrics.json"
  echo Evaluating %%F
  python evaluate.py --dataset "%%F" --output "!OUTPUT!" %EXTRA_ARGS%
)

echo Evaluation pass complete.
