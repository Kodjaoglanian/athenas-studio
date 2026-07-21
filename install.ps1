# Athenas Studio installer for Windows PowerShell
# One-line install: irm https://athenas.studio/install.ps1 | iex
param(
    [string]$InstallDir = "$env:LOCALAPPDATA\athenas\bin",
    [string]$ConfigDir = "$env:USERPROFILE\.athenas"
)

$ErrorActionPreference = "Stop"

$Repo = "Kodjaoglanian/athenas-studio"

function Write-Info($msg) { Write-Host "  [info] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  [ok]   $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [warn] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "  [err]  $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "    ___   __   ____  _   _ _____ _____ ____     " -ForegroundColor DarkCyan
Write-Host "   / _ \ / /_ / ___|| | |  ___|_   _|  _ \    " -ForegroundColor DarkCyan
Write-Host "  / /_\_/ __|\___ \| |_| | |_    | | | |_) |   " -ForegroundColor DarkCyan
Write-Host " / /_\  \_| |____) |  _  |  _|   | | |  _ <    " -ForegroundColor DarkCyan
Write-Host " \____|\__|____/ |_| |_|_|     |_| |_| \_\   " -ForegroundColor DarkCyan
Write-Host "        Studio - Local LLM Inference" -ForegroundColor DarkCyan
Write-Host ""

# Detect architecture
$arch = $env:PROCESSOR_ARCHITECTURE
if ($arch -eq "AMD64") {
    $target = "x86_64-pc-windows-msvc"
} elseif ($arch -eq "ARM64") {
    $target = "aarch64-pc-windows-msvc"
} else {
    Write-Err "Unsupported architecture: $arch"
}

Write-Info "Detected target: $target"

# Fetch latest release
try {
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -Headers @{ "User-Agent" = "athenas-installer" }
    $version = $release.tag_name
} catch {
    Write-Err "Failed to fetch latest version: $_"
}

Write-Info "Latest version: $version"

$archiveName = "athenas-$version-$target.zip"
$downloadUrl = "https://github.com/$Repo/releases/download/$version/$archiveName"
Write-Info "Downloading: $downloadUrl"

$tmpDir = New-Item -ItemType Directory -Force -Path "$env:TEMP\athenas-install-$(Get-Random)"
$archivePath = Join-Path $tmpDir.FullName $archiveName

try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath -UseBasicParsing
} catch {
    Write-Err "Download failed: $_"
}

# Verify SHA256
$sha256Url = "$downloadUrl.sha256"
$sha256Path = Join-Path $tmpDir.FullName "$archiveName.sha256"
try {
    Invoke-WebRequest -Uri $sha256Url -OutFile $sha256Path -UseBasicParsing
    $expectedHash = (Get-Content $sha256Path).Split(" ")[0]
    $actualHash = (Get-FileHash $archivePath -Algorithm SHA256).Hash
    if ($actualHash -ne $expectedHash) {
        Write-Warn "Checksum mismatch! Expected: $expectedHash, Got: $actualHash"
    } else {
        Write-Ok "Checksum verified"
    }
} catch {
    Write-Warn "No checksum available, skipping verification"
}

# Extract
Write-Info "Extracting..."
Expand-Archive -Path $archivePath -DestinationPath $tmpDir.FullName -Force

# Install
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$binaryPath = Join-Path $InstallDir "athenas.exe"

if (Test-Path (Join-Path $tmpDir.FullName "athenas.exe")) {
    Move-Item (Join-Path $tmpDir.FullName "athenas.exe") $binaryPath -Force
} else {
    Write-Err "Binary not found in archive"
}

Write-Ok "Installed to: $binaryPath"

# Create config directories
New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
New-Item -ItemType Directory -Force -Path "$ConfigDir\models" | Out-Null
New-Item -ItemType Directory -Force -Path "$ConfigDir\cache" | Out-Null
New-Item -ItemType Directory -Force -Path "$ConfigDir\data" | Out-Null
Write-Ok "Config directory: $ConfigDir"

# Add to PATH (User level)
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("PATH", "$InstallDir;$userPath", "User")
    Write-Ok "Added to user PATH"
} else {
    Write-Info "Already in PATH"
}

# Verify
Write-Host ""
try {
    & $binaryPath --version
    Write-Host ""
    Write-Ok "Athenas Studio installed successfully!"
    Write-Host ""
    Write-Host "  Run 'athenas --help' to get started" -ForegroundColor DarkCyan
    Write-Host "  Or start the TUI with 'athenas'" -ForegroundColor DarkCyan
    Write-Host ""
    Write-Warn "Restart your terminal for PATH changes to take effect"
} catch {
    Write-Warn "Installation complete, but binary verification failed."
    Write-Warn "Try running: $binaryPath --help"
}

# Cleanup
Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
