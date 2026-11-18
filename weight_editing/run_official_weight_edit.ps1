param(
    [string]$Python='D:/conda/envs/labelstudioml/python.exe',
    [switch]$Calibrate
)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH=(Resolve-Path vendor/ocr_deps).Path
$env:Path=(Join-Path (Split-Path -Parent $Python) 'Lib/site-packages/torch/lib')+';'+$env:Path
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True'
$out='runs/official-weight-edit-v1'
New-Item -ItemType Directory -Path $out -Force | Out-Null

function Run-Stage([string[]]$Parameters) {
    Write-Host ($Parameters -join ' ')
    & $Python -u (Join-Path $PSScriptRoot 'official_weight_edit.py') @Parameters
    if ($LASTEXITCODE -ne 0) { throw "Failed $($Parameters -join ' ') ($LASTEXITCODE)" }
}

if (-not (Test-Path -LiteralPath "$out/provenance.json")) { Run-Stage @('prepare') }
foreach ($component in 'yolo','ocr') {
    foreach ($mode in 'edit','full') {
        if (-not (Test-Path -LiteralPath "$out/smoke/$component-$mode/training.json")) {
            Run-Stage @("train-$component",'--mode',$mode,'--smoke')
        }
    }
}
foreach ($component in 'yolo','ocr') {
    foreach ($mode in 'edit','full') {
        if (-not (Test-Path -LiteralPath "$out/$component-$mode/training.json")) {
            Run-Stage @("train-$component",'--mode',$mode)
        }
    }
}
foreach ($mode in 'edit','full') {
    if (-not (Test-Path -LiteralPath "$out/ocr-$mode/inference/inference.pdiparams")) {
        Run-Stage @('export','--mode',$mode)
    }
}
if (-not (Test-Path -LiteralPath "$out/test-lock.json")) {
    Run-Stage @('detect','--split','val')
    Run-Stage @('eval','--split','val')
    Run-Stage @('lock')
}
if (-not (Test-Path -LiteralPath "$out/test/edit-ocr/metrics.json")) {
    Run-Stage @('detect','--split','test')
    Run-Stage @('eval','--split','test')
}
if (-not (Test-Path -LiteralPath "$out/swanlab-status.json")) { Run-Stage @('report') }

if ($Calibrate -and -not (Test-Path -LiteralPath "$out/calibrated/results.md")) {
    if (-not (Test-Path -LiteralPath "$out/calibrated/threshold-lock.json")) {
        & $Python -u (Join-Path $PSScriptRoot 'calibrate_official_weight_edit.py') val
        if ($LASTEXITCODE -ne 0) { throw 'Validation calibration failed.' }
    }
    & $Python -u (Join-Path $PSScriptRoot 'calibrate_official_weight_edit.py') test
    if ($LASTEXITCODE -ne 0) { throw 'Calibrated test detection failed.' }
    & $Python -u (Join-Path $PSScriptRoot 'calibrate_official_weight_edit.py') report
    if ($LASTEXITCODE -ne 0) { throw 'Calibrated report failed.' }
}

