[CmdletBinding()]
param(
    [ValidateSet("setup", "models", "start", "verify", "status", "stop")]
    [string]$Action = "setup"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ComposeFile = Join-Path $RootDir "compose.yaml"
$EnvExample = Join-Path $RootDir ".env.selfhost.example"
$EnvFile = Join-Path $RootDir ".env"
$ComposePrefix = @("compose", "--env-file", $EnvFile, "-f", $ComposeFile)

function Invoke-DockerChecked {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker 命令失败（退出码 $LASTEXITCODE）：docker $($Arguments -join ' ')"
    }
}

function Invoke-DockerCapture {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $output = @(& docker @Arguments)
    if ($LASTEXITCODE -ne 0) {
        throw "Docker 命令失败（退出码 $LASTEXITCODE）：docker $($Arguments -join ' ')"
    }
    return ($output -join "`n").Trim()
}

function Get-EnvValue {
    param([Parameter(Mandatory = $true)][string]$Name)
    $matches = @(Get-Content -LiteralPath $EnvFile | ForEach-Object {
        if ($_ -match "^$([regex]::Escape($Name))=(.*)$") { $Matches[1].Trim() }
    })
    if ($matches.Count -ne 1) {
        throw "$EnvFile 中 $Name 必须且只能出现一次。"
    }
    return $matches[0]
}

function Initialize-Environment {
    if (-not (Test-Path -LiteralPath $ComposeFile -PathType Leaf)) {
        throw "缺少 $ComposeFile"
    }
    if (-not (Test-Path -LiteralPath $EnvExample -PathType Leaf)) {
        throw "缺少 $EnvExample"
    }
    if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
        Copy-Item -LiteralPath $EnvExample -Destination $EnvFile
        Write-Host "已从安全默认模板创建 $EnvFile。"
    }

    $portText = Get-EnvValue -Name "MUSIC_TO_MIDI_PORT"
    $gpuText = Get-EnvValue -Name "GPU_DEVICE_ID"
    $profiles = Get-EnvValue -Name "MUSIC_TO_MIDI_ENABLED_PROFILES"
    $port = 0
    if (-not [int]::TryParse($portText, [ref]$port) -or $port -lt 1 -or $port -gt 65535) {
        throw "MUSIC_TO_MIDI_PORT 必须是 1-65535。"
    }
    $gpu = 0
    if (-not [int]::TryParse($gpuText, [ref]$gpu) -or $gpu -lt 0) {
        throw "GPU_DEVICE_ID 必须是非负整数。"
    }
    if ([string]::IsNullOrWhiteSpace($profiles)) {
        throw "MUSIC_TO_MIDI_ENABLED_PROFILES 不能为空。"
    }
}

function Test-DockerPrerequisites {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "未检测到 Docker CLI。请安装 Docker Engine/Desktop、Compose v2 与 NVIDIA Container Toolkit。"
    }
    $serverText = Invoke-DockerCapture -Arguments @("version", "--format", "{{.Server.Version}}")
    $composeText = (Invoke-DockerCapture -Arguments @("compose", "version", "--short")).TrimStart("v")
    if ($serverText -notmatch '^\d+\.\d+\.\d+' -or [Version]$Matches[0] -lt [Version]"24.0.0") {
        throw "Docker Engine $serverText 过旧；要求 24.0.0+。"
    }
    if ($composeText -notmatch '^\d+\.\d+\.\d+' -or [Version]$Matches[0] -lt [Version]"2.17.0") {
        throw "Docker Compose $composeText 过旧；要求 2.17.0+。"
    }
    Invoke-DockerChecked -Arguments ($ComposePrefix + @("config", "--quiet"))
}

function Test-ProfileSelection {
    $supportedText = Invoke-DockerCapture -Arguments (
        $ComposePrefix + @(
            "--profile", "tools", "run", "--rm", "--no-deps",
            "--entrypoint", "python", "model-init", "-m", "src.model_profiles", "list"
        )
    )
    $supported = @($supportedText -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $selected = @((Get-EnvValue -Name "MUSIC_TO_MIDI_ENABLED_PROFILES").Split(",") | ForEach-Object { $_.Trim() })
    if ($selected.Count -eq 0 -or $selected -contains "") {
        throw "模型配置列表包含空项。"
    }
    if (@($selected | Select-Object -Unique).Count -ne $selected.Count) {
        throw "模型配置列表不能包含重复项。"
    }
    $unknown = @($selected | Where-Object { $_ -notin $supported })
    if ($unknown.Count -gt 0) {
        throw "后端镜像不支持模型配置：$($unknown -join ', ')"
    }
    if (($selected | Where-Object { $_ -like "muscriptor*" }).Count -gt 0 -and
        [string]::IsNullOrWhiteSpace($env:HF_TOKEN)) {
        throw "MuScriptor 是 gated 模型；请只在当前 PowerShell 会话设置 `$env:HF_TOKEN 后重试。"
    }
}

function Test-GpuRuntime {
    $probe = 'import torch; import onnxruntime as ort; assert torch.version.cuda == "12.8", torch.version.cuda; assert torch.cuda.is_available(); assert torch.cuda.device_count() > 0; assert "CUDAExecutionProvider" in ort.get_available_providers(); print(torch.cuda.get_device_name(0)); print(ort.get_available_providers())'
    Invoke-DockerChecked -Arguments (
        $ComposePrefix + @(
            "--profile", "tools", "run", "--rm", "--no-deps",
            "--entrypoint", "python", "model-init", "-c", $probe
        )
    )
}

function Initialize-Models {
    Test-ProfileSelection
    Write-Host "正在显式下载并严格校验所选模型；常驻服务不会下载模型。"
    Invoke-DockerChecked -Arguments (
        $ComposePrefix + @("--profile", "tools", "run", "--rm", "model-init")
    )
}

function Wait-Ready {
    $port = Get-EnvValue -Name "MUSIC_TO_MIDI_PORT"
    $deadline = [DateTime]::UtcNow.AddMinutes(10)
    $lastError = "尚未收到响应"
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/v1/ready" -TimeoutSec 20
            if ($response.status -ne "ready") {
                throw "readiness 返回了非 ready 状态：$($response | ConvertTo-Json -Compress)"
            }
            return
        }
        catch {
            $lastError = $_.Exception.Message
            Start-Sleep -Seconds 5
        }
    }
    & docker @($ComposePrefix + @("ps"))
    & docker @($ComposePrefix + @("logs", "--tail", "200", "backend", "gateway"))
    throw "10 分钟内未通过 readiness：$lastError"
}

function Test-Stack {
    $port = Get-EnvValue -Name "MUSIC_TO_MIDI_PORT"
    Invoke-DockerChecked -Arguments (
        $ComposePrefix + @("exec", "-T", "backend", "python", "/app/docker/healthcheck.py")
    )
    $runtime = Invoke-RestMethod -Uri "http://127.0.0.1:$port/runtime-config.json" -TimeoutSec 20
    if (-not $runtime.managed -or $runtime.expected_api_version -ne "2.0") {
        throw "前端运行时配置无效：$($runtime | ConvertTo-Json -Compress)"
    }
    $ready = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/v1/ready" -TimeoutSec 20
    if ($ready.status -ne "ready" -or $ready.api_version -ne "2.0") {
        throw "外部 readiness 无效：$($ready | ConvertTo-Json -Compress)"
    }
    $selected = @((Get-EnvValue -Name "MUSIC_TO_MIDI_ENABLED_PROFILES").Split(",") | ForEach-Object { $_.Trim() })
    $missingProfiles = @($selected | Where-Object { $_ -notin @($ready.enabled_profiles) })
    if ($missingProfiles.Count -gt 0) {
        throw "readiness 未返回所选模型配置：$($missingProfiles -join ', ')"
    }

    $backendId = Invoke-DockerCapture -Arguments ($ComposePrefix + @("ps", "-q", "backend"))
    $gatewayId = Invoke-DockerCapture -Arguments ($ComposePrefix + @("ps", "-q", "gateway"))
    if ([string]::IsNullOrWhiteSpace($backendId) -or [string]::IsNullOrWhiteSpace($gatewayId)) {
        throw "后端或网关容器未运行。"
    }
    if ((Invoke-DockerCapture -Arguments @("inspect", "--format", "{{.HostConfig.ReadonlyRootfs}}", $backendId)) -ne "true") {
        throw "后端根文件系统不是只读。"
    }
    if ((Invoke-DockerCapture -Arguments @("inspect", "--format", "{{.Config.User}}", $backendId)) -ne "10001:10001") {
        throw "后端不是固定非 root 用户。"
    }
    if ((Invoke-DockerCapture -Arguments @("inspect", "--format", "{{.HostConfig.ReadonlyRootfs}}", $gatewayId)) -ne "true") {
        throw "网关根文件系统不是只读。"
    }
    if ((Invoke-DockerCapture -Arguments @("inspect", "--format", "{{.Config.User}}", $gatewayId)) -ne "10001:10001") {
        throw "网关不是固定非 root 用户。"
    }
    if ((Invoke-DockerCapture -Arguments @("network", "inspect", "music-to-midi_inference", "--format", "{{.Internal}}")) -ne "true") {
        throw "推理网络不是 internal 网络。"
    }
    Write-Host "Docker 自托管验收通过：http://127.0.0.1:$port"
    Invoke-DockerChecked -Arguments ($ComposePrefix + @("images"))
}

Push-Location $RootDir
try {
    Initialize-Environment
    Test-DockerPrerequisites
    switch ($Action) {
        "setup" {
            Invoke-DockerChecked -Arguments ($ComposePrefix + @("pull"))
            Test-GpuRuntime
            Initialize-Models
            Invoke-DockerChecked -Arguments ($ComposePrefix + @("up", "-d", "--remove-orphans"))
            Wait-Ready
            Test-Stack
        }
        "models" {
            Invoke-DockerChecked -Arguments ($ComposePrefix + @("pull", "backend", "model-init"))
            Test-GpuRuntime
            Initialize-Models
        }
        "start" {
            Invoke-DockerChecked -Arguments ($ComposePrefix + @("up", "-d", "--remove-orphans"))
            Wait-Ready
            Test-Stack
        }
        "verify" { Test-Stack }
        "status" {
            Invoke-DockerChecked -Arguments ($ComposePrefix + @("ps"))
            Invoke-DockerChecked -Arguments ($ComposePrefix + @("images"))
        }
        "stop" {
            Invoke-DockerChecked -Arguments ($ComposePrefix + @("down"))
            Write-Host "容器已停止；模型、作业和 Caddy 命名卷均已保留。"
        }
    }
}
finally {
    Pop-Location
}
