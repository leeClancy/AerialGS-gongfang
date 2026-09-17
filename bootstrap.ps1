# 源码开发启动：创建 .venv，安装轻量依赖，启动 127.0.0.1 服务。
# 不下载数 GB 的 PyTorch / COLMAP。完整离线包请用 packaging/build_portable.ps1。
param(
    [switch]$InstallOnly,
    [switch]$SkipInstall,
    [string]$Python = "python",
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating development virtual environment: .venv"
    & $Python -m venv (Join-Path $Root ".venv")
    if (-not (Test-Path $venvPython)) {
        throw "Could not create .venv. Install Python 3.10 or newer."
    }
}
if (-not $SkipInstall) {
    Write-Host "Installing lightweight development dependencies (without CUDA/PyTorch)."
    & $venvPython -m pip install -U pip
    & $venvPython -m pip install -r (Join-Path $Root "requirements-dev.txt")
}

$env:AERIALGS_ROOT = $Root
$env:AERIALGS_HOST = "127.0.0.1"
$env:AERIALGS_PORT = "$Port"
if ($InstallOnly) {
    Write-Host "Development dependencies are ready. Run .\bootstrap.ps1 to start."
    exit 0
}
Write-Host "Starting local-only server at http://127.0.0.1:$Port/"
& $venvPython (Join-Path $Root "backend\run.py") --host 127.0.0.1 --port $Port
