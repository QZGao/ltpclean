@echo off
setlocal enabledelayedexpansion

rem Adjust these paths or arguments as needed
set INPUT_ROOT=eval_data\recordings
set OUTPUT_ROOT=eval_data\metrics
set "EXTRA_ARGS=%*"

if not exist %OUTPUT_ROOT% mkdir %OUTPUT_ROOT%

for %%F in (%INPUT_ROOT%\*-frameArray.txt) do (
  set "BASE=%%~nF"
  set "OUTPUT=%OUTPUT_ROOT%\!BASE!-metrics.json"
  echo Evaluating %%F -> !OUTPUT!
  python evaluate.py --dataset "%%F" --output "!OUTPUT!" %EXTRA_ARGS%
)

echo Evaluation pass complete.
