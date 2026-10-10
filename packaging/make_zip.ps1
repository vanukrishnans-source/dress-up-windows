$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$stage = Join-Path $root "dist\pkg\DressUp"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item -Recurse "$root\dist\DressUp\*" $stage
Copy-Item "$root\README.md","$root\THIRD_PARTY.md","$root\LICENSE" $stage -Force
$ver = (Get-Content "$root\dressup\__init__.py" | Select-String '__version__\s*=\s*"([^"]+)"').Matches.Groups[1].Value
$zip = Join-Path $root "dist\DressUp-$ver-win64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
$sha = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
"$sha  DressUp-$ver-win64.zip" | Out-File -Encoding ascii "$root\dist\SHA256SUMS"
Write-Host "ZIP $zip $((Get-Item $zip).Length) bytes  SHA256 $sha"
