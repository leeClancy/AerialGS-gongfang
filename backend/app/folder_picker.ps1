param(
  [string]$Mode = "folders",
  [string]$OutFile = "",
  [string]$Title = ""
)

$ErrorActionPreference = "Stop"
if (-not $Title) { $Title = $env:AERIALGS_PICKER_TITLE }
if (-not $Title) { $Title = "选择文件夹" }
if (-not $OutFile) { $OutFile = $env:AERIALGS_PICKER_OUT }
if (-not $OutFile) { throw "missing OutFile" }
$multi = $env:AERIALGS_PICKER_MULTI -eq "1"

Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles() | Out-Null

$owner = New-Object System.Windows.Forms.Form
$owner.Text = $Title
$owner.TopMost = $true
$owner.ShowInTaskbar = $true
$owner.StartPosition = "CenterScreen"
$owner.Size = New-Object System.Drawing.Size(80, 40)
$owner.MinimizeBox = $false
$owner.MaximizeBox = $false
$owner.Show()
$owner.Activate()
try { [void][System.Windows.Forms.Application]::DoEvents() } catch {}

$picked = New-Object System.Collections.Generic.List[string]

function Write-Picked {
  param([string[]]$Items)
  $dir = Split-Path -Parent $OutFile
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $escaped = @()
  foreach ($item in $Items) {
    if (-not $item) { continue }
    $escaped += ('"' + ($item.Replace("\", "\\").Replace('"', '\"')) + '"')
  }
  "[" + ($escaped -join ",") + "]" | Set-Content -Path $OutFile -Encoding UTF8
}

try {
  if ($Mode -eq "file") {
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = $Title
    $dialog.CheckFileExists = $true
    $dialog.Multiselect = $false
    $dialog.Filter = "All files (*.*)|*.*"
    $result = $dialog.ShowDialog($owner)
    if ($result -eq [System.Windows.Forms.DialogResult]::OK -and $dialog.FileName) {
      $picked.Add($dialog.FileName)
    }
  } else {
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = $Title
    $dialog.ShowNewFolderButton = $true
    try { $dialog.UseDescriptionForTitle = $true } catch {}
    while ($true) {
      $result = $dialog.ShowDialog($owner)
      if ($result -ne [System.Windows.Forms.DialogResult]::OK -or -not $dialog.SelectedPath) { break }
      if (-not $picked.Contains($dialog.SelectedPath)) { $picked.Add($dialog.SelectedPath) }
      if (-not $multi) { break }
      $more = [System.Windows.Forms.MessageBox]::Show(
        $owner,
        "已选 $($picked.Count) 个文件夹。要继续添加下一个吗？",
        $Title,
        [System.Windows.Forms.MessageBoxButtons]::YesNo,
        [System.Windows.Forms.MessageBoxIcon]::Question
      )
      if ($more -ne [System.Windows.Forms.DialogResult]::Yes) { break }
    }
  }
} finally {
  try { $owner.Close() } catch {}
  try { $owner.Dispose() } catch {}
}

Write-Picked -Items $picked.ToArray()
exit 0
