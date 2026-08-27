#requires -Version 5.1
[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "GetFlex"),
    [string]$ReceiptPath = (Join-Path $env:TEMP "getflex-windows-verification.json"),
    [string[]]$ExpectedMarker = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Capture([string]$File, [string[]]$Arguments) {
    $previousPreference = $ErrorActionPreference
    $exitCode = 1
    try {
        $ErrorActionPreference = "Continue"
        $output = & $File @Arguments 2>&1 | Out-String
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) { throw "$File failed (${exitCode}): $($Arguments -join ' ')`n$output" }
    return $output.Trim()
}

$compose = @(
    "compose", "--project-directory", $InstallDir,
    "-f", (Join-Path $InstallDir "compose.yaml"),
    "-f", (Join-Path $InstallDir "compose.sources.yaml")
)
function Invoke-ComposeCapture([string[]]$Arguments) {
    $dockerArguments = @($compose) + @($Arguments)
    return Invoke-Capture -File docker -Arguments $dockerArguments
}
$manifestPath = Join-Path $InstallDir "install-manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) { throw "Missing install manifest: $manifestPath" }
$parsedModules = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$modules = if ($parsedModules -is [System.Array]) {
    @($parsedModules | ForEach-Object { $_ })
} else {
    @($parsedModules)
}

$receipt = [ordered]@{
    timestamp = [DateTime]::UtcNow.ToString("o")
    windows = [Environment]::OSVersion.VersionString
    docker_version = Invoke-Capture docker @("version", "--format", "{{.Server.Version}}")
    compose_version = Invoke-Capture docker @("compose", "version", "--short")
    image = Invoke-Capture docker @("inspect", "getflex-worker", "--format", "{{.Config.Image}}")
    image_id = Invoke-Capture docker @("inspect", "getflex-worker", "--format", "{{.Image}}")
    modules = @()
    mcp_handshake = $false
    worker_restart = $false
    compose_recreate = $false
    volume_preserved = $false
    expected_markers = @()
    reboot_proof_required = $true
}

foreach ($module in $modules) {
    $orient = Invoke-Capture docker @("exec", "getflex-worker", "flex", "search", "--cell", [string]$module.cell, "@orient")
    $receipt.modules += [ordered]@{
        module = [string]$module.module
        cell = [string]$module.cell
        orient_ok = [bool]($orient.Length -gt 0)
        limitations = @($module.limitations)
    }
}

$initialize = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"getflex-windows-verify","version":"1"}}}'
$previousPreference = $ErrorActionPreference
$handshakeExit = 1
try {
    $ErrorActionPreference = "Continue"
    $handshake = $initialize | & docker exec -i getflex-worker python -m flex.serve 2>$null | Out-String
    $handshakeExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousPreference
}
if ($handshakeExit -ne 0 -or $handshake -notmatch '"id"\s*:\s*1') { throw "Flex stdio MCP handshake failed: $handshake" }
$receipt.mcp_handshake = $true

foreach ($markerSpec in $ExpectedMarker) {
    $parts = $markerSpec.Split('=', 2)
    if ($parts.Count -ne 2) { throw "ExpectedMarker must be cell=marker: $markerSpec" }
    $cell = $parts[0]
    $marker = $parts[1]
    $escaped = $marker.Replace("'", "''")
    $query = "SELECT CASE WHEN EXISTS (SELECT 1 FROM chunks WHERE instr(content, '$escaped') > 0) THEN 'FLEX_MARKER_FOUND' ELSE 'FLEX_MARKER_MISSING' END AS marker_status"
    $result = Invoke-Capture docker @("exec", "getflex-worker", "flex", "search", "--cell", $cell, $query)
    $found = $result.Contains("FLEX_MARKER_FOUND")
    $receipt.expected_markers += [ordered]@{ cell=$cell; marker=$marker; found=$found }
    if (-not $found) { throw "Marker not found in ${cell}: $marker" }
}

Invoke-ComposeCapture -Arguments @("restart", "worker") | Out-Null
foreach ($module in $modules) {
    Invoke-Capture docker @("exec", "getflex-worker", "flex", "search", "--cell", [string]$module.cell, "@orient") | Out-Null
}
$receipt.worker_restart = $true

$volumeBefore = Invoke-Capture docker @("volume", "inspect", "getflex-data", "--format", "{{.Name}}:{{.CreatedAt}}")
Invoke-ComposeCapture -Arguments @("down") | Out-Null
Invoke-ComposeCapture -Arguments @("up", "-d", "--wait", "worker") | Out-Null
$volumeAfter = Invoke-Capture docker @("volume", "inspect", "getflex-data", "--format", "{{.Name}}:{{.CreatedAt}}")
$receipt.compose_recreate = $true
$receipt.volume_preserved = ($volumeBefore -eq $volumeAfter)
if (-not $receipt.volume_preserved) { throw "getflex-data volume changed across compose down/up" }
foreach ($module in $modules) {
    Invoke-Capture docker @("exec", "getflex-worker", "flex", "search", "--cell", [string]$module.cell, "@orient") | Out-Null
}
foreach ($markerSpec in $ExpectedMarker) {
    $parts = $markerSpec.Split('=', 2)
    $cell = $parts[0]
    $marker = $parts[1]
    $escaped = $marker.Replace("'", "''")
    $query = "SELECT CASE WHEN EXISTS (SELECT 1 FROM chunks WHERE instr(content, '$escaped') > 0) THEN 'FLEX_MARKER_FOUND' ELSE 'FLEX_MARKER_MISSING' END AS marker_status"
    $result = Invoke-Capture docker @("exec", "getflex-worker", "flex", "search", "--cell", $cell, $query)
    if (-not $result.Contains("FLEX_MARKER_FOUND")) { throw "Marker lost after compose recreation in ${cell}: $marker" }
}

$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReceiptPath -Encoding UTF8
Write-Host "Windows container verification passed" -ForegroundColor Green
Write-Host "Receipt: $ReceiptPath"
Write-Warning "A separate Windows reboot/Docker Desktop restart proof is still required before beta promotion."
