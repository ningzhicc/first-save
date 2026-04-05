param(
    [ValidateSet('adapt', 'test', 'both')]
    [string]$Mode = 'both',

    [ValidateSet('legacy', 'patch_reprogram', 'semantic_reprogram')]
    [string]$StateEncoderType = 'legacy',

    [string]$PythonExe = '',
    [string]$Device = 'cuda:0',
    [string]$ExpPoolPath = 'artifacts/exp_pools/exp_pool.pkl',
    [string]$ModelDir = '',
    [string]$Trace = 'fcc-test',
    [int]$TraceNum = 100,
    [string]$Video = 'video1',
    [int]$Rank = 128,
    [int]$Window = 20,
    [double]$Gamma = 1.0,
    [double]$LearningRate = 0.0001,
    [double]$WeightDecay = 0.0001,
    [int]$WarmupSteps = 2000,
    [int]$NumEpochs = 80,
    [int]$EvalPerEpoch = 2,
    [int]$GradAccumSteps = 32,
    [double]$TargetReturnScale = 1.0,
    [int]$Seed = 100003,
    [int]$StateFeatureDim = 256,
    [int]$PatchLen = 3,
    [int]$PatchStride = 1,
    [int]$NumPrototypes = 64,
    [int]$ReprogramHeads = 4,
    [double]$ReprogramDropout = 0.1,
    [switch]$FixedOrder,
    [string[]]$ExtraArgs = @(),
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-LocalPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BaseDir,
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return $PathValue
    }
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $PathValue))
}

function Resolve-PythonExePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptDir,
        [string]$PreferredPythonExe
    )

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($PreferredPythonExe)) {
        $candidates += (Resolve-LocalPath -BaseDir $ScriptDir -PathValue $PreferredPythonExe)
    }
    $candidates += @(
        'E:\Anaconda3\envs\abr_netllm_qwen\python.exe',
        'C:\Users\Lenovo\.conda\envs\abr_netllm_qwen\python.exe'
    )

    foreach ($candidate in $candidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return $pythonCmd.Source
    }

    throw 'Cannot find a Python executable for the Qwen environment. Please pass -PythonExe explicitly.'
}

function Resolve-LatestBestModelDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptDir
    )

    $searchRoot = Join-Path $ScriptDir 'data\ft_plms\qwen_base'
    if (-not (Test-Path $searchRoot)) {
        throw "Cannot find qwen finetune directory: $searchRoot"
    }

    $bestModelDirs = Get-ChildItem $searchRoot -Recurse -Directory -Filter 'early_stop_*_best_model' -ErrorAction SilentlyContinue |
        Where-Object {
            (Test-Path (Join-Path $_.FullName 'modules_except_plm.bin')) -or
            (Test-Path (Join-Path $_.FullName 'model.bin'))
        } |
        Sort-Object LastWriteTime -Descending

    if (-not $bestModelDirs) {
        throw "Cannot find any best-model directory under $searchRoot. Please finetune first or pass -ModelDir."
    }

    return $bestModelDirs[0].FullName
}

function Invoke-RunPlm {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonPath,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDir,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $prettyCommand = @($PythonPath) + $Arguments
    Write-Host ''
    Write-Host ('>> ' + ($prettyCommand -join ' '))

    if ($DryRun) {
        return
    }

    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "run_plm.py exited with code $LASTEXITCODE"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$resolvedPythonExe = Resolve-PythonExePath -ScriptDir $scriptDir -PreferredPythonExe $PythonExe
$resolvedExpPoolPath = Resolve-LocalPath -BaseDir $scriptDir -PathValue $ExpPoolPath
$resolvedModelDir = if ([string]::IsNullOrWhiteSpace($ModelDir)) { '' } else { Resolve-LocalPath -BaseDir $scriptDir -PathValue $ModelDir }

if (-not (Test-Path $resolvedExpPoolPath) -and $Mode -in @('adapt', 'both')) {
    throw "Experience pool not found: $resolvedExpPoolPath"
}

$commonArgs = @(
    'run_plm.py',
    '--plm-type', 'qwen',
    '--plm-size', 'base',
    '--device', $Device,
    '--rank', $Rank,
    '--trace', $Trace,
    '--trace-num', $TraceNum,
    '--video', $Video,
    '--w', $Window,
    '--gamma', $Gamma,
    '--target-return-scale', $TargetReturnScale,
    '--seed', $Seed,
    '--state-encoder-type', $StateEncoderType,
    '--state-feature-dim', $StateFeatureDim
)

if ($FixedOrder) {
    $commonArgs += '--fixed-order'
}

if ($StateEncoderType -eq 'patch_reprogram') {
    $commonArgs += @(
        '--patch-len', $PatchLen,
        '--patch-stride', $PatchStride,
        '--num-prototypes', $NumPrototypes,
        '--reprogram-heads', $ReprogramHeads,
        '--reprogram-dropout', $ReprogramDropout
    )
}
elseif ($StateEncoderType -eq 'semantic_reprogram') {
    $commonArgs += @(
        '--reprogram-heads', $ReprogramHeads,
        '--reprogram-dropout', $ReprogramDropout
    )
}

if ($Mode -in @('adapt', 'both')) {
    $adaptArgs = @(
        '--adapt',
        '--exp-pool-path', $resolvedExpPoolPath,
        '--lr', $LearningRate,
        '--weight-decay', $WeightDecay,
        '--warmup-steps', $WarmupSteps,
        '--num-epochs', $NumEpochs,
        '--eval-per-epoch', $EvalPerEpoch,
        '--grad-accum-steps', $GradAccumSteps
    )
    if ($ExtraArgs.Count -gt 0) {
        $adaptArgs += $ExtraArgs
    }

    Push-Location $scriptDir
    try {
        Invoke-RunPlm -PythonPath $resolvedPythonExe -WorkingDir $scriptDir -Arguments ($commonArgs + $adaptArgs)
    }
    finally {
        Pop-Location
    }
}

if ($Mode -in @('test', 'both')) {
    if ([string]::IsNullOrWhiteSpace($resolvedModelDir)) {
        $resolvedModelDir = Resolve-LatestBestModelDir -ScriptDir $scriptDir
    }
    if (-not (Test-Path $resolvedModelDir)) {
        throw "Model directory not found: $resolvedModelDir"
    }

    Write-Host ''
    Write-Host ('Using model dir: ' + $resolvedModelDir)

    $testArgs = @(
        '--test',
        '--model-dir', $resolvedModelDir
    )
    if ($ExtraArgs.Count -gt 0) {
        $testArgs += $ExtraArgs
    }

    Push-Location $scriptDir
    try {
        Invoke-RunPlm -PythonPath $resolvedPythonExe -WorkingDir $scriptDir -Arguments ($commonArgs + $testArgs)
    }
    finally {
        Pop-Location
    }
}
