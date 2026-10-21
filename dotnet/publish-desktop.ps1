# Publish the desktop app as a NativeAOT executable.
#
# Avalonia compiles XAML from CoreCompile.  An incremental `dotnet publish`
# can skip CoreCompile and hand NativeAOT an assembly without compiled XAML,
# so this script rebuilds first and then publishes with --no-build.
#
# Usage:
#   pwsh dotnet/publish-desktop.ps1
#   pwsh dotnet/publish-desktop.ps1 -Output D:\apps\indra-ocr

param(
    [string]$Output = (Join-Path $PSScriptRoot "dist\desktop")
)

$ErrorActionPreference = "Stop"
$project = Join-Path $PSScriptRoot "src\IndraOcr.Desktop"
$data = Join-Path $project "Data"
if (-not (Test-Path (Join-Path $data "det.onnx"))) {
    throw "Bundled models are missing: $data"
}

Write-Host "Building (NativeAOT preparation)..."
dotnet build $project -c Release -r win-x64 -t:Rebuild -p:IndraNativeAot=true --nologo
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (Test-Path $Output) {
    Remove-Item -Recurse -Force $Output
}

Write-Host "Publishing NativeAOT to $Output ..."
dotnet publish $project -c Release -r win-x64 --no-build -p:IndraNativeAot=true -o $Output --nologo
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

# Symbols are not needed for the shipped app (and are large for AOT builds).
Get-ChildItem $Output -Filter "*.pdb" -File -ErrorAction SilentlyContinue | Remove-Item -Force

# CopyToPublishDirectory handles this normally; keep a fallback so a publish
# with --no-build can never ship an executable without its models.
$publishedData = Join-Path $Output "Data"
if (-not (Test-Path (Join-Path $publishedData "det.onnx"))) {
    Copy-Item $data $publishedData -Recurse -Force
}

$exe = Join-Path $Output "IndraOcr.Desktop.exe"
Write-Host "NativeAOT executable: $exe"
Write-Host "Bundled models: $(Join-Path $Output 'Data')"
