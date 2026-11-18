param(
    [ValidateSet('prepare','smoke','formal','final','all')]
    [string]$Stage = 'all',
    [string]$Python = 'D:/conda/envs/labelstudioml/python.exe'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = (Resolve-Path vendor/ocr_deps).Path
$torchLib = Join-Path (Split-Path -Parent $Python) 'Lib/site-packages/torch/lib'
$env:Path = "$torchLib;$env:Path"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = 'True'
$out = 'runs/ocr-weight-edit-v1'

function Run-Step {
    param([string]$Log, [string[]]$Arguments)
    Write-Host ($Arguments -join ' ')
    $entry = Join-Path $PSScriptRoot $Arguments[0]
    $stepArguments = $Arguments | Select-Object -Skip 1
    & $Python -u $entry @stepArguments *>&1 | Tee-Object -FilePath (Join-Path $out $Log)
    if ($LASTEXITCODE -ne 0) { throw "Failed: $($Arguments -join ' ') ($LASTEXITCODE)" }
}

if ($Stage -in 'prepare','all') {
    if (Test-Path -LiteralPath "$out/provenance.json") { throw "Experiment directory already initialized: $out" }
    New-Item -ItemType Directory -Path $out -Force | Out-Null
    Run-Step 'prepare.log' @('weight_edit.py','prepare')
}
if ($Stage -in 'smoke','all') {
    Run-Step 'smoke-training-run.log' @('train_weight_edit.py','E2-lambda-1.0','--smoke')
    foreach ($step in 0,3) {
        Run-Step "smoke-export-$step.log" @('weight_edit.py','export','--candidate','smoke','--step',"$step")
        Run-Step "smoke-validate-$step.log" @('weight_edit.py','validate','--candidate','smoke','--step',"$step")
    }
}
if ($Stage -in 'formal','all') {
    if (-not (Test-Path -LiteralPath "$out/smoke/step-003/export-verification.json")) {
        throw 'Complete the smoke export parity check first.'
    }
    foreach ($candidate in 'E1','E2-lambda-0.1','E2-lambda-1.0') {
        if (-not (Test-Path -LiteralPath "$out/$candidate/completed.json")) {
            Run-Step "$candidate-training.log" @('train_weight_edit.py',"$candidate")
        }
    }
    foreach ($candidate in 'E1','E2-lambda-0.1','E2-lambda-1.0') {
        foreach ($step in 0,50,100,200) {
            $checkpointPath = "$out/$candidate/step-$($step.ToString('000'))"
            if (-not (Test-Path -LiteralPath "$checkpointPath/inference/inference.pdiparams")) {
                Run-Step "$candidate-export-$step.log" @('weight_edit.py','export','--candidate',"$candidate",'--step',"$step")
            }
            if (-not (Test-Path -LiteralPath "$checkpointPath/val-summary.json")) {
                Run-Step "$candidate-validate-$step.log" @('weight_edit.py','validate','--candidate',"$candidate",'--step',"$step")
            }
        }
    }
}
if ($Stage -in 'final','all') {
    if (-not (Test-Path -LiteralPath "$out/selection-lock.json")) {
        Run-Step 'lock.log' @('weight_edit.py','lock')
    }
    foreach ($family in 'E0','E1','E2') {
        if (-not (Test-Path -LiteralPath "$out/final/$family/summary.json")) {
            Run-Step "$family-final.log" @('weight_edit.py','test','--family',"$family")
        }
    }
    Run-Step 'report.log' @('weight_edit.py','report')
}

