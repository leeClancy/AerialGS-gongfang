# 在开发机组装 NVIDIA 专用完整离线便携包。
# 下载数 GB 依赖并校验哈希。使用官方预编译 gsplat Windows wheel，不要现场编译 CUDA。
# 源码仓库不提交这些大二进制。未实际成功跑完本脚本前，不得声称已经生成发布包。
[CmdletBinding()]
param(
    [string]$OutputRoot = "",
    [switch]$SkipDownload,
    [switch]$SkipWarmup,
    [switch]$UpdateLock
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not (Test-Path (Join-Path $Root "packaging\versions.json"))) {
    throw "Missing packaging\versions.json"
}
Set-Location $Root

$VersionsPath = Join-Path $Root "packaging\versions.json"
$LockPath = Join-Path $Root "packaging\artifact-lock.json"
$Cache = Join-Path $Root "packaging\cache"
$OutRoot = if ($OutputRoot) { $OutputRoot } else { Join-Path $Root "dist" }
$Stage = Join-Path $OutRoot "AerialGS-Portable"
$ZipPath = Join-Path $OutRoot "AerialGS-Portable-win64.zip"
New-Item -ItemType Directory -Force -Path $Cache, $OutRoot | Out-Null

$Versions = Get-Content -Raw -Encoding UTF8 $VersionsPath | ConvertFrom-Json
$Lock = @{}
if (Test-Path $LockPath) {
    $Lock = Get-Content -Raw -Encoding UTF8 $LockPath | ConvertFrom-Json
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLower()
}

function Assert-NativeExit([string]$What) {
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed with exit code $LASTEXITCODE"
    }
}

function Download-Verified {
    param(
        [string]$Url,
        [string]$Dest,
        [string]$Name,
        [string]$Expected
    )
    New-Item -ItemType Directory -Force -Path (Split-Path $Dest) | Out-Null
    if (-not (Test-Path $Dest)) {
        if ($SkipDownload) { throw "Missing cached artifact: $Dest" }
        Write-Host "Downloading $Name"
        Write-Host "  $Url"
        try {
            Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
        } catch {
            Remove-Item -Force -ErrorAction SilentlyContinue $Dest
            throw
        }
    }
    $actual = Get-Sha256 $Dest
    if ($Expected) {
        if ($actual -ne $Expected.ToLower()) {
            throw "SHA256 mismatch for ${Name}: expected $($Expected.ToLower()) actual $actual"
        }
    } else {
        Write-Host "Recording unpinned $Name sha256=$actual; use -UpdateLock to persist it."
        if ($UpdateLock) {
            if ($Lock -is [hashtable]) { $Lock[$Name] = $actual }
            else { $Lock | Add-Member -NotePropertyName $Name -NotePropertyValue $actual -Force }
        }
    }
    return $actual
}

Write-Host "=== AerialGS portable build ==="
Write-Host "Source: $Root"
Write-Host "Output: $Stage"
Write-Host "Do not treat the dist zip as released until this script succeeds."

if (Test-Path $Stage) {
    Remove-Item -Recurse -Force $Stage
}
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

$pyZip = Join-Path $Cache $Versions.python.filename
$pySha = Download-Verified -Url $Versions.python.url -Dest $pyZip -Name "python" -Expected $Versions.python.sha256

$colmapZip = Join-Path $Cache $Versions.colmap.filename
$colmapSha = Download-Verified -Url $Versions.colmap.url -Dest $colmapZip -Name "colmap" -Expected $Versions.colmap.sha256

$glomapZip = Join-Path $Cache $Versions.glomap.filename
$glomapSha = Download-Verified -Url $Versions.glomap.url -Dest $glomapZip -Name "glomap" -Expected $Versions.glomap.sha256

$vocab = Join-Path $Cache $Versions.vocab_tree.filename
$vocabExpected = $Versions.vocab_tree.sha256
if (-not $vocabExpected -and $Lock.vocab_tree) { $vocabExpected = $Lock.vocab_tree }
$vocabSha = Download-Verified -Url $Versions.vocab_tree.url -Dest $vocab -Name "vocab_tree" -Expected $vocabExpected

$getPip = Join-Path $Cache "get-pip.py"
$getPipSha = Download-Verified -Url $Versions.get_pip.url -Dest $getPip -Name "get_pip" -Expected $Versions.get_pip.sha256

$gsplatName = $Versions.python_packages.gsplat_wheel_filename
if (-not $gsplatName) {
    throw "versions.json is missing python_packages.gsplat_wheel_filename"
}
$gsplatWhl = Join-Path $Cache $gsplatName
$gsplatSha = Download-Verified -Url $Versions.python_packages.gsplat_wheel_url -Dest $gsplatWhl -Name "gsplat" -Expected $Versions.python_packages.gsplat_sha256

$runtime = Join-Path $Stage "runtime"
$pythonDir = Join-Path $runtime "python"
$colmapDir = Join-Path $runtime "colmap"
$glomapDir = Join-Path $runtime "glomap"
$vocabDir = Join-Path $runtime "vocab"
$appDir = Join-Path $Stage "app"
$licenseDir = Join-Path $Stage "licenses"
New-Item -ItemType Directory -Force -Path $pythonDir, $colmapDir, $glomapDir, $vocabDir, $appDir, $licenseDir | Out-Null

Write-Host "Extracting embedded Python"
Expand-Archive -Path $pyZip -DestinationPath $pythonDir -Force
$pth = Get-ChildItem $pythonDir -Filter "python*._pth" | Select-Object -First 1
@"
python310.zip
.
Lib\site-packages
..\..\app
import site
"@ | Set-Content -Encoding ASCII $pth.FullName

Write-Host "Extracting COLMAP and GLOMAP"
$tmpColmap = Join-Path $Cache "colmap_unpacked"
$tmpGlomap = Join-Path $Cache "glomap_unpacked"
if (Test-Path $tmpColmap) { Remove-Item -Recurse -Force $tmpColmap }
if (Test-Path $tmpGlomap) { Remove-Item -Recurse -Force $tmpGlomap }
Expand-Archive -Path $colmapZip -DestinationPath $tmpColmap -Force
Expand-Archive -Path $glomapZip -DestinationPath $tmpGlomap -Force
Copy-Item -Recurse -Force (Join-Path $tmpColmap "*") $colmapDir
Copy-Item -Recurse -Force (Join-Path $tmpGlomap "*") $glomapDir
Copy-Item -Force $vocab (Join-Path $vocabDir $Versions.vocab_tree.filename)

Write-Host "Copying application sources without packaging/cache"
foreach ($name in @("backend", "trainer", "web")) {
    Copy-Item -Recurse -Force (Join-Path $Root $name) (Join-Path $appDir $name)
}
$pkgApp = Join-Path $appDir "packaging"
New-Item -ItemType Directory -Force -Path (Join-Path $pkgApp "licenses") | Out-Null
Copy-Item -Force (Join-Path $Root "packaging\versions.json") (Join-Path $pkgApp "versions.json")
Copy-Item -Force (Join-Path $Root "packaging\requirements-portable.txt") (Join-Path $pkgApp "requirements-portable.txt")
Copy-Item -Force (Join-Path $Root "packaging\licenses\NOTICE.md") (Join-Path $pkgApp "licenses\NOTICE.md")
Copy-Item -Force (Join-Path $Root "LICENSE") (Join-Path $Stage "LICENSE")
Copy-Item -Force (Join-Path $Root "README.md") (Join-Path $Stage "README.md")
Copy-Item -Force (Join-Path $Root "packaging\licenses\NOTICE.md") (Join-Path $licenseDir "NOTICE.md")
$launcher = Get-ChildItem -Path $Root -Filter "*.bat" -File | Select-Object -First 1
if (-not $launcher) { throw "Missing launcher .bat file" }
Copy-Item -Force $launcher.FullName (Join-Path $Stage $launcher.Name)

$pythonExe = Join-Path $pythonDir "python.exe"
Write-Host "Installing pip and pinned Python packages (PyTorch 2.4.1+cu124)"
& $pythonExe $getPip
Assert-NativeExit "get-pip"
& $pythonExe -m pip install --no-warn-script-location --upgrade pip
Assert-NativeExit "pip upgrade"
$req = Join-Path $Root "packaging\requirements-portable.txt"
& $pythonExe -m pip install --no-warn-script-location -r $req
Assert-NativeExit "pip install requirements-portable"
Write-Host "Installing official precompiled gsplat wheel: $gsplatName"
& $pythonExe -m pip install --no-warn-script-location --force-reinstall --no-deps $gsplatWhl
Assert-NativeExit "pip install gsplat wheel"

if (-not $SkipWarmup) {
    Write-Host "Validating the precompiled gsplat wheel on the builder NVIDIA GPU"
    $warmup = @"
import gsplat
import torch
print('gsplat', getattr(gsplat, '__version__', '?'))
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit('The builder has no usable NVIDIA CUDA device')
from gsplat.rendering import rasterization
means = torch.zeros((8,3), device='cuda')
quats = torch.tensor([[1.,0.,0.,0.]], device='cuda').repeat(8,1)
scales = torch.ones((8,3), device='cuda') * 0.01
opacities = torch.ones((8,), device='cuda') * 0.1
colors = torch.ones((8,3), device='cuda')
view = torch.eye(4, device='cuda').unsqueeze(0)
K = torch.tensor([[[100.,0.,16.],[0.,100.,16.],[0.,0.,1.]]], device='cuda')
rasterization(means, quats, scales, opacities, colors, view, K, 32, 32, packed=False, absgrad=True)
print('gsplat prebuilt wheel ok')
"@
    $warmupFile = Join-Path $Cache "warmup_gsplat.py"
    Set-Content -Encoding UTF8 $warmupFile $warmup
    & $pythonExe $warmupFile
    Assert-NativeExit "gsplat precompiled wheel validation"
}

Write-Host "Scanning text files for absolute development paths"
$scanFiles = Get-ChildItem -Path $appDir -Recurse -Include *.py,*.ps1,*.bat,*.json,*.md,*.txt,*.html,*.js,*.css
$bad = @()
foreach ($f in $scanFiles) {
    $t = Get-Content -Raw -ErrorAction SilentlyContinue $f.FullName
    if ($t -and ($t -match [regex]::Escape($Root) -or $t -match 'C:\\Users\\')) {
        $bad += $f.FullName
    }
}
if ($bad.Count -gt 0) {
    Write-Warning ("Possible absolute development paths found:`n" + ($bad -join "`n"))
}

$artifactLines = @(
    "python  $pySha  $($Versions.python.filename)",
    "colmap  $colmapSha  $($Versions.colmap.filename)",
    "glomap  $glomapSha  $($Versions.glomap.filename)",
    "vocab_tree  $vocabSha  $($Versions.vocab_tree.filename)",
    "gsplat_wheel  $gsplatSha  $gsplatName",
    "get_pip  $getPipSha  get-pip.py"
)
$artifactLines -join "`r`n" | Set-Content -Encoding ASCII (Join-Path $Stage "ARTIFACT_SHA256.txt")

$manifest = [ordered]@{
    name = "AerialGS-Portable-win64"
    built_utc = [DateTime]::UtcNow.ToString("o")
    python_sha256 = $pySha
    colmap_sha256 = $colmapSha
    glomap_sha256 = $glomapSha
    vocab_sha256 = $vocabSha
    gsplat_sha256 = $gsplatSha
    gsplat_wheel = $gsplatName
    torch = $Versions.python_packages.torch
    torchvision = $Versions.python_packages.torchvision
    relocatable = $true
    loopback = "127.0.0.1"
    note = "This manifest excludes the zip self-hash; it is stored in the outer SHA256SUMS.txt."
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $Stage "BUILD_MANIFEST.json")

Write-Host "Creating zip after writing the internal artifact manifest"
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Compress-Archive -Path $Stage -DestinationPath $ZipPath -CompressionLevel Optimal
$zipSha = Get-Sha256 $ZipPath
$outer = $artifactLines + @("portable_zip  $zipSha  AerialGS-Portable-win64.zip")
$outer -join "`r`n" | Set-Content -Encoding ASCII (Join-Path $OutRoot "SHA256SUMS.txt")

if ($UpdateLock) {
    $lockObj = @{
        python = $pySha
        colmap = $colmapSha
        glomap = $glomapSha
        vocab_tree = $vocabSha
        get_pip = $getPipSha
        gsplat = $gsplatSha
        portable_zip = $zipSha
    }
    $lockObj | ConvertTo-Json | Set-Content -Encoding UTF8 $LockPath
}

Write-Host "Portable package written: $ZipPath"
Write-Host "Outer checksums: $(Join-Path $OutRoot 'SHA256SUMS.txt')"
Write-Host "The internal ARTIFACT_SHA256.txt contains third-party hashes only."
Write-Host "Extract the package and run its launcher. A compatible NVIDIA driver is required."
