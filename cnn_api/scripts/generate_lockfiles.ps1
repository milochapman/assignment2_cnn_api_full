# Generate uv lockfile and a frozen requirements.lock.txt
# Usage: powershell -ExecutionPolicy Bypass -File scripts\generate_lockfiles.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Move to repo cnn_api root
Set-Location (Join-Path $PSScriptRoot "..")

# Check uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "uv is not installed. Install via PowerShell:"
  Write-Host "iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex"
  exit 1
}

uv lock
uv export --frozen --format requirements-txt > requirements.lock.txt

Write-Host "Generated files: uv.lock, requirements.lock.txt"