param(
    [ValidateSet("auto", "gpu", "cpu")]
    [string]$Backend = "auto",
    [int]$Port = 7860,
    [string]$HostName = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

if ($Backend -eq "auto") {
    $HasNvidia = $false
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi -L *> $null
        $HasNvidia = $LASTEXITCODE -eq 0
    }
    $Backend = if ($HasNvidia) { "gpu" } else { "cpu" }
}

$RuntimeExtra = if ($Backend -eq "gpu") { "onnx-gpu" } else { "onnx-cpu" }
$env:SIGNATURE_GRADIO_HOST = $HostName
$env:SIGNATURE_GRADIO_PORT = [string]$Port

Write-Host "Starting Gradio demo on http://$HostName`:$Port using $Backend runtime"
& uv run --extra $RuntimeExtra --extra demo signature-verifier-demo
if ($LASTEXITCODE -ne 0) {
    throw "Gradio demo failed with exit code $LASTEXITCODE"
}
