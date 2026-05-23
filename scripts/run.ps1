<#
.SYNOPSIS
    Builds the .NET CLI, builds/starts the TTS container, waits for the
    model to load, then synthesizes a sample sentence.

.EXAMPLE
    .\scripts\run.ps1
    .\scripts\run.ps1 -Text "God morgen!" -Voice nic -Out morning.wav
    .\scripts\run.ps1 -SkipBuild -SkipDocker          # just synthesize
    .\scripts\run.ps1 -NoSynth                        # boot only, no synth call
#>
[CmdletBinding()]
param(
    [string]$Text = "Hej, hvordan går det?",
    [ValidateSet("mic", "nic")]
    [string]$Voice = "mic",
    [string]$Out = "out.wav",
    [string]$Server = "http://localhost:8000",
    [int]$HealthTimeoutSeconds = 600,
    [switch]$SkipBuild,
    [switch]$SkipDocker,
    [switch]$NoSynth
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

function Write-Step([string]$message) {
    Write-Host "==> $message" -ForegroundColor Cyan
}

if (-not $SkipBuild) {
    Write-Step "dotnet build (Release)"
    dotnet build src/DanishVoice.Cli/DanishVoice.Cli.csproj -nologo -c Release
    if ($LASTEXITCODE -ne 0) { throw "dotnet build failed" }
}

if (-not $SkipDocker) {
    Write-Step "docker compose up -d --build"
    docker compose up -d --build
    if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }
}

Write-Step "Waiting for $Server/health (up to $HealthTimeoutSeconds s)"
$deadline = (Get-Date).AddSeconds($HealthTimeoutSeconds)
$ready = $false
while (-not $ready) {
    try {
        $resp = Invoke-WebRequest "$Server/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        $body = $resp.Content | ConvertFrom-Json
        if ($body.model_loaded) {
            Write-Host "    ready: device=$($body.device) model=$($body.model)" -ForegroundColor Green
            $ready = $true
            break
        }
    }
    catch {
        # server not up yet or still loading — keep polling silently
    }
    if ((Get-Date) -gt $deadline) {
        throw "Timed out after $HealthTimeoutSeconds s waiting for $Server/health. Run 'docker compose logs tts' to see what is going on."
    }
    Start-Sleep -Seconds 3
}

if ($NoSynth) {
    Write-Step "Server is ready. Skipping synthesis (-NoSynth)."
    exit 0
}

Write-Step "Synthesizing: voice=$Voice  out=$Out"
dotnet run --project src/DanishVoice.Cli --no-build -c Release -- `
    $Text --voice $Voice --out $Out --server $Server
if ($LASTEXITCODE -ne 0) { throw "synthesis failed (exit $LASTEXITCODE)" }

$absOut = (Resolve-Path $Out).Path
Write-Host ""
Write-Host "Done. Output: $absOut" -ForegroundColor Green
Write-Host "Play it with:  Start-Process `"$absOut`""
