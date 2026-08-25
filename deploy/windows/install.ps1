#requires -Version 5.1
<#+
.SYNOPSIS
  Installs the complete released Flex wheel through Docker Desktop/WSL2.

.DESCRIPTION
  This script does not install Docker or coding-agent CLIs. It detects known
  local provider stores, accepts additional providers from a JSON manifest,
  creates narrow source mounts (read-only by default; writable only when SQLite WAL sidecars require it), initializes every selected Flex module in a
  separate disposable container, then starts one persistent Flex worker.

  Additional module manifest shape:
  {
    "modules": [
      {
        "module": "opencode",
        "cell": "opencode",
        "args": ["--opencode-db", "/sources/opencode/opencode.db"],
        "mounts": [
          {"source":"C:\\Users\\me\\AppData\\Local\\opencode\\opencode.db",
           "target":"/sources/opencode/opencode.db"}
        ],
        "modulePackage": "C:\\path\\to\\flex-opencode"
      }
    ]
  }
#>
[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "GetFlex"),
    [string]$Image = "ghcr.io/damiandelmas/flex:0.55.0",
    [string]$ModuleManifest,
    [string[]]$SkipModule = @(),
    [switch]$SkipMcpRegistration,
    [switch]$SkipPull,
    [switch]$NonInteractive,
    [switch]$RenderOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "[ok] $Message" -ForegroundColor Green }
function Write-Warn([string]$Message) { Write-Warning $Message }
function Fail([string]$Message) { throw $Message }

function Invoke-Native {
    param([Parameter(Mandatory=$true)][string]$File, [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        Fail "$File exited with code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Quote-Yaml([string]$Value) {
    return "'" + $Value.Replace("'", "''") + "'"
}

function New-Mount([string]$Source, [string]$Target, [bool]$ReadOnly = $true) {
    return [ordered]@{
        source = [System.IO.Path]::GetFullPath($Source)
        target = $Target
        readOnly = $ReadOnly
    }
}

function New-Module {
    param(
        [string]$Module,
        [string]$Cell,
        [object[]]$Mounts,
        [string[]]$Args = @(),
        [string]$ModulePackage = "",
        [string[]]$Limitations = @()
    )
    return [ordered]@{
        module = $Module
        cell = $Cell
        args = @($Args)
        mounts = @($Mounts)
        modulePackage = $ModulePackage
        limitations = @($Limitations)
    }
}

function Test-Skipped([string]$Module) {
    $normalized = @($SkipModule | ForEach-Object { @($_ -split ',') } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    return $normalized -contains $Module
}

function Assert-ModuleName([string]$Module) {
    if ($Module -notmatch '^[a-z0-9][a-z0-9_-]*$') {
        Fail "Invalid module name '$Module'. Use lowercase letters, digits, '_' or '-'."
    }
}

function Assert-ManifestTarget([string]$Target, [string]$Module) {
    if (-not $Target.StartsWith('/sources/') -or $Target.Contains('/../') -or $Target.EndsWith('/..')) {
        Fail "Mount target for $Module must be an absolute path beneath /sources/: $Target"
    }
    foreach ($reserved in @('/root/.flex', '/module-packages', '/var/run/docker.sock')) {
        if ($Target -eq $reserved -or $Target.StartsWith($reserved + '/')) {
            Fail "Mount target for $Module uses reserved path ${reserved}: $Target"
        }
    }
}

function Add-DetectedModules([System.Collections.Generic.List[object]]$Modules) {
    $codexSessions = Join-Path $HOME ".codex\sessions"
    if ((Test-Path -LiteralPath $codexSessions -PathType Container) -and -not (Test-Skipped "codex")) {
        $mounts = [System.Collections.Generic.List[object]]::new()
        $mounts.Add((New-Mount $codexSessions "/root/.codex/sessions"))
        # state_5.sqlite is intentionally not mounted automatically: a lone
        # SQLite file mount can miss live WAL/SHM state, while mounting all of
        # .codex would expose auth.json. Users can provide a safe snapshot via
        # -ModuleManifest when title/sidecar parity is required.
        $codexIndex = Join-Path $HOME ".codex\session_index.jsonl"
        if (Test-Path -LiteralPath $codexIndex -PathType Leaf) {
            $mounts.Add((New-Mount $codexIndex "/root/.codex/session_index.jsonl"))
        }
        $Modules.Add((New-Module -Module "codex" -Cell "codex" -Mounts $mounts.ToArray() -Limitations @(
            "state_5.sqlite is not auto-mounted; configure a safe snapshot for full title/sidecar metadata"
        )))
    }

    $claudeProjects = Join-Path $HOME ".claude\projects"
    if ((Test-Path -LiteralPath $claudeProjects -PathType Container) -and -not (Test-Skipped "claude-code")) {
        $Modules.Add((New-Module "claude-code" "claude_code" @(
            (New-Mount $claudeProjects "/root/.claude/projects")
        )))
    }

    $gooseCandidates = @(@(
        (Join-Path $HOME ".local\share\goose\sessions\sessions.db"),
        (Join-Path $env:APPDATA "Block\goose\data\sessions\sessions.db"),
        (Join-Path $env:LOCALAPPDATA "goose\sessions\sessions.db")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) })
    if ($gooseCandidates.Count -gt 0 -and -not (Test-Skipped "goose")) {
        $Modules.Add((New-Module "goose" "goose" @(
            # Mount the directory, not only sessions.db, so SQLite WAL/SHM
            # sidecars remain visible while native Goose is writing.
            # SQLite readers may need to create/update -shm beside a WAL DB.
            # The Goose compiler still opens sessions.db in mode=ro; the narrow
            # directory mount is writable only for SQLite sidecar mechanics.
            (New-Mount (Split-Path -Parent $gooseCandidates[0]) "/root/.local/share/goose/sessions" $false)
        )))
    }
}

function Get-OptionalProperty($Object, [string]$Name, $Default = $null) {
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $Default }
    return $property.Value
}

function Add-ManifestModules([System.Collections.Generic.List[object]]$Modules, [string]$Path) {
    if (-not $Path) { return }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { Fail "Module manifest not found: $Path" }
    $document = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    $entries = Get-OptionalProperty $document "modules" @()
    foreach ($entry in @($entries)) {
        $moduleName = [string](Get-OptionalProperty $entry "module" "")
        if (-not $moduleName) { Fail "Every manifest module needs 'module'." }
        Assert-ModuleName $moduleName
        if (Test-Skipped $moduleName) { continue }
        $mounts = @()
        foreach ($mount in @(Get-OptionalProperty $entry "mounts" @())) {
            $mountSource = [string](Get-OptionalProperty $mount "source" "")
            $mountTarget = [string](Get-OptionalProperty $mount "target" "")
            if (-not $mountSource -or -not $mountTarget) {
                Fail "Every mount for $moduleName needs source and target."
            }
            Assert-ManifestTarget $mountTarget $moduleName
            if (-not (Test-Path -LiteralPath $mountSource)) {
                Fail "Source mount for $moduleName does not exist: $mountSource"
            }
            $readOnly = [bool](Get-OptionalProperty $mount "readOnly" $true)
            $mounts += New-Mount $mountSource $mountTarget $readOnly
        }
        $packageValue = [string](Get-OptionalProperty $entry "modulePackage" "")
        $package = if ($packageValue) { [System.IO.Path]::GetFullPath($packageValue) } else { "" }
        if ($package -and -not (Test-Path -LiteralPath $package)) {
            Fail "Module package does not exist: $package"
        }
        $cellValue = [string](Get-OptionalProperty $entry "cell" "")
        $cell = if ($cellValue) { $cellValue } else { $moduleName }
        if ($cell -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') { Fail "Invalid cell name for ${moduleName}: $cell" }
        $args = @(Get-OptionalProperty $entry "args" @() | ForEach-Object { [string]$_ })
        $limitations = @(Get-OptionalProperty $entry "limitations" @() | ForEach-Object { [string]$_ })
        $Modules.Add((New-Module $moduleName $cell $mounts $args $package $limitations))
    }
}

function Write-ComposeFiles([string]$Directory, [string]$ImageName, [object[]]$Modules) {
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
    $compose = @'
name: getflex

x-flex-common: &flex-common
  image: ${FLEX_IMAGE:?Set FLEX_IMAGE in .env}
  pull_policy: never
  environment:
    FLEX_HOME: /root/.flex
    PYTHONUNBUFFERED: "1"
  volumes:
    - type: volume
      source: getflex-data
      target: /root/.flex

services:
  init:
    <<: *flex-common
    profiles: ["init"]
    environment:
      FLEX_HOME: /root/.flex
      FLEX_RUNTIME_OWNER: external
      FLEX_SKILL_MODE: none
      PYTHONUNBUFFERED: "1"
    restart: "no"

  worker:
    <<: *flex-common
    container_name: getflex-worker
    command: ["python", "-m", "flex.daemon", "--no-refresh", "--no-background"]
    restart: unless-stopped

volumes:
  getflex-data:
    name: getflex-data
'@
    Set-Content -LiteralPath (Join-Path $Directory "compose.yaml") -Value $compose -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $Directory ".env") -Value ("FLEX_IMAGE=" + $ImageName) -Encoding UTF8

    $sourceMounts = [ordered]@{}
    $initOnlyMounts = [ordered]@{}
    foreach ($module in $Modules) {
        foreach ($mount in @($module.mounts)) {
            if ($sourceMounts.Contains($mount.target) -and $sourceMounts[$mount.target].source -ne $mount.source) {
                Fail "Conflicting mount target $($mount.target): $($sourceMounts[$mount.target].source) vs $($mount.source)"
            }
            $sourceMounts[$mount.target] = $mount
        }
        if ($module.modulePackage) {
            $target = "/module-packages/" + $module.module
            $initOnlyMounts[$target] = New-Mount $module.modulePackage $target
        }
    }

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("services:")
    foreach ($service in @("init", "worker")) {
        $lines.Add("  ${service}:")
        $lines.Add("    volumes:")
        $serviceMounts = [System.Collections.Generic.List[object]]::new()
        foreach ($mount in $sourceMounts.Values) { $serviceMounts.Add($mount) }
        if ($service -eq "init") {
            foreach ($mount in $initOnlyMounts.Values) { $serviceMounts.Add($mount) }
        }
        foreach ($mount in $serviceMounts) {
            $lines.Add("      - type: bind")
            $lines.Add("        source: " + (Quote-Yaml $mount.source))
            $lines.Add("        target: " + (Quote-Yaml $mount.target))
            $readOnlyText = if ($mount.readOnly) { "true" } else { "false" }
            $lines.Add("        read_only: " + $readOnlyText)
        }
    }
    Set-Content -LiteralPath (Join-Path $Directory "compose.sources.yaml") -Value $lines -Encoding UTF8
    ConvertTo-Json -InputObject $Modules -Depth 8 | Set-Content -LiteralPath (Join-Path $Directory "install-manifest.json") -Encoding UTF8
}

$script:ComposeFiles = @()
function Set-ComposeContext([string]$Directory) {
    $script:ComposeFiles = @(
        "compose", "--project-directory", $Directory,
        "-f", (Join-Path $Directory "compose.yaml"),
        "-f", (Join-Path $Directory "compose.sources.yaml")
    )
}

function Invoke-Compose([string[]]$Arguments) {
    $dockerArguments = @($script:ComposeFiles) + @($Arguments)
    Invoke-Native -File docker -Arguments $dockerArguments
}

function Register-McpClients {
    if ($SkipMcpRegistration) { return }
    $mcpCommand = @("docker", "exec", "-i", "getflex-worker", "python", "-m", "flex.serve")
    if (Get-Command codex -ErrorAction SilentlyContinue) {
        & codex mcp remove flex 2>$null | Out-Null
        & codex mcp add flex -- @mcpCommand
        if ($LASTEXITCODE -eq 0) { Write-Ok "Registered Flex MCP with Codex" } else { Write-Warn "Could not register Codex MCP; use: codex mcp add flex -- $($mcpCommand -join ' ')" }
    }
    if (Get-Command claude -ErrorAction SilentlyContinue) {
        & claude mcp remove --scope user flex 2>$null | Out-Null
        & claude mcp add --scope user flex -- @mcpCommand
        if ($LASTEXITCODE -eq 0) { Write-Ok "Registered Flex MCP with Claude Code" } else { Write-Warn "Could not register Claude MCP; use: claude mcp add --scope user flex -- $($mcpCommand -join ' ')" }
    }
}

if (-not $RenderOnly) {
    Write-Step "Checking Docker Desktop / WSL2"
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Fail "Docker is not installed. Install and start Docker Desktop with the WSL2 engine, then rerun this script."
    }
    Invoke-Native docker version
    Invoke-Native docker compose version
    & docker info *> $null
    if ($LASTEXITCODE -ne 0) { Fail "Docker daemon is not running. Start Docker Desktop and rerun this script." }
    Write-Ok "Docker is available"
}

$modules = [System.Collections.Generic.List[object]]::new()
Add-DetectedModules $modules
Add-ManifestModules $modules $ModuleManifest
if ($modules.Count -eq 0) {
    Fail "No provider stores were detected. Supply -ModuleManifest with explicit module/source mappings."
}

Write-Step "Selected Flex modules"
foreach ($module in $modules) {
    Write-Host ("  {0} -> cell {1}" -f $module.module, $module.cell)
}
if (-not $NonInteractive) {
    $answer = Read-Host "Initialize these modules? [Y/n]"
    if ($answer -and $answer -notmatch '^[Yy]') { Fail "Installation cancelled." }
}

Write-Step "Writing Compose configuration"
Write-ComposeFiles $InstallDir $Image $modules.ToArray()
Set-ComposeContext $InstallDir
Write-Ok "Installation files written to $InstallDir"
if ($RenderOnly) {
    Write-Ok "Render-only validation complete"
    return
}

if (-not $SkipPull) {
    Write-Step "Pulling Flex image"
    Invoke-Native -File docker -Arguments @("pull", $Image)
}

foreach ($module in $modules) {
    Write-Step ("Initializing module " + $module.module)
    if ($module.modulePackage) {
        Invoke-Compose @("run", "--rm", "init", "flex", "module", "install", ("/module-packages/" + $module.module))
    }
    $initArgs = @("run", "--rm", "init", "flex", "init", "--module", $module.module) + @($module.args)
    Invoke-Compose $initArgs
    Invoke-Compose @("run", "--rm", "init", "flex", "search", "--cell", $module.cell, "@orient")
    Write-Ok ("Initialized " + $module.module)
}

Write-Step "Starting the Flex worker"
Invoke-Compose @("up", "-d", "--wait", "worker")
Invoke-Native docker exec getflex-worker flex status --all
Register-McpClients

Write-Ok "Flex is running on Windows through Docker"
Write-Host ""
Write-Host "Universal MCP command:"
Write-Host "  docker exec -i getflex-worker python -m flex.serve"
Write-Host ""
Write-Host "Operations (run from $InstallDir):"
Write-Host "  docker compose -f compose.yaml -f compose.sources.yaml ps"
Write-Host "  docker compose -f compose.yaml -f compose.sources.yaml logs -f worker"
Write-Host "  docker compose -f compose.yaml -f compose.sources.yaml restart worker"
Write-Host "  docker pull $Image; docker compose -f compose.yaml -f compose.sources.yaml up -d --wait worker"
Write-Host "  docker compose -f compose.yaml -f compose.sources.yaml down       # retain cells"
Write-Host "  docker compose -f compose.yaml -f compose.sources.yaml down -v    # purge Flex volume"
