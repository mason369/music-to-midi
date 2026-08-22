[CmdletBinding()]
param(
    [string]$PublicAddress,
    [string]$AcmeEmail,
    [string]$BasicAuthUser = "mason",
    [string]$Profiles = "yourmt3:yptf_moe_multi_nops,piano_transkun",
    [string]$GpuDeviceId = "0"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ComposeFile = Join-Path $RepoRoot "compose.production.yaml"
$EnvFile = Join-Path $RepoRoot ".env.production"
$CaddyImage = "caddy:2.11.4-alpine@sha256:5f5c8640aae01df9654968d946d8f1a56c497f1dd5c5cda4cf95ab7c14d58648"
$CudaImage = "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "命令失败（退出码 $LASTEXITCODE）：$Command $($Arguments -join ' ')"
    }
}

function Convert-SecureStringToPlainText {
    param([Parameter(Mandatory = $true)][Security.SecureString]$SecureValue)
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

function Get-BasicAuthorizationValue {
    param(
        [Parameter(Mandatory = $true)][string]$Username,
        [Parameter(Mandatory = $true)][Security.SecureString]$SecurePassword
    )
    $plain = Convert-SecureStringToPlainText -SecureValue $SecurePassword
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes("${Username}:$plain")
        return "Basic $([Convert]::ToBase64String($bytes))"
    }
    finally {
        $plain = $null
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "未检测到 Docker CLI。请先安装 Docker Engine 24+、Compose v2 与 NVIDIA Container Toolkit。"
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "未检测到 Git CLI，无法记录当前源码身份。"
}

Push-Location $RepoRoot
try {
    Invoke-NativeChecked docker version
    Invoke-NativeChecked docker compose version
    $DockerServerVersionText = (& docker version --format '{{.Server.Version}}').Trim()
    if ($LASTEXITCODE -ne 0 -or $DockerServerVersionText -notmatch '^\d+\.\d+\.\d+') {
        throw "无法读取 Docker Engine 服务端版本。"
    }
    $DockerServerVersion = [Version]([regex]::Match($DockerServerVersionText, '^\d+\.\d+\.\d+').Value)
    if ($DockerServerVersion -lt [Version]'24.0.0') {
        throw "Docker Engine 版本过低：$DockerServerVersionText；要求 24.0.0 或更高。"
    }
    $ComposeVersionText = (& docker compose version --short).Trim().TrimStart('v')
    if ($LASTEXITCODE -ne 0 -or $ComposeVersionText -notmatch '^\d+\.\d+\.\d+') {
        throw "无法读取 Docker Compose 版本。"
    }
    $ComposeVersion = [Version]([regex]::Match($ComposeVersionText, '^\d+\.\d+\.\d+').Value)
    if ($ComposeVersion -lt [Version]'2.17.0') {
        throw "Docker Compose 版本过低：$ComposeVersionText；要求 2.17.0 或更高。"
    }

    if (-not $PublicAddress) {
        $PublicAddress = (Read-Host "公网 DNS 域名（仅域名，不含 https://）").Trim()
    }
    if (-not $AcmeEmail) {
        $AcmeEmail = (Read-Host "ACME 证书联系邮箱").Trim()
    }
    $BasicAuthUser = $BasicAuthUser.Trim()
    $Profiles = $Profiles.Trim()
    $GpuDeviceId = $GpuDeviceId.Trim()

    $DnsPattern = '^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$'
    if ($PublicAddress.Length -gt 253 -or $PublicAddress -notmatch $DnsPattern) {
        throw "PublicAddress 必须是公开 DNS 域名，且不能包含协议、路径或端口。"
    }
    if ($AcmeEmail -notmatch '^[^\s@]+@[^\s@]+\.[^\s@]+$') {
        throw "AcmeEmail 格式无效。"
    }
    if ($BasicAuthUser -notmatch '^[A-Za-z0-9._-]+$') {
        throw "BasicAuthUser 只能包含字母、数字、点、下划线和连字符。"
    }
    if ($GpuDeviceId -notmatch '^\d+$') {
        throw "GpuDeviceId 必须是 nvidia-smi 显示的非负整数索引。"
    }

    $SupportedProfiles = @(
        "yourmt3:ymt3_plus",
        "yourmt3:yptf_single_nops",
        "yourmt3:yptf_multi_ps",
        "yourmt3:yptf_moe_multi_nops",
        "yourmt3:yptf_moe_multi_ps",
        "miros",
        "muscriptor",
        "muscriptor:medium",
        "muscriptor:small",
        "piano_transkun",
        "piano_transkun_v2_aug",
        "piano_aria_amt",
        "piano_bytedance_pedal",
        "vocal_split",
        "six_stem_split"
    )
    $SelectedProfiles = @($Profiles.Split(',') | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
    if ($SelectedProfiles.Count -eq 0 -or $SelectedProfiles.Count -ne (@($SelectedProfiles | Select-Object -Unique)).Count) {
        throw "Profiles 必须至少包含一个且不能重复。"
    }
    $UnknownProfiles = @($SelectedProfiles | Where-Object { $_ -notin $SupportedProfiles })
    if ($UnknownProfiles.Count -gt 0) {
        throw "不支持的模型配置：$($UnknownProfiles -join ', ')"
    }
    $Profiles = $SelectedProfiles -join ','

    if (($SelectedProfiles | Where-Object { $_ -like 'muscriptor*' }).Count -gt 0 -and [string]::IsNullOrWhiteSpace($env:HF_TOKEN)) {
        throw "所选 MuScriptor 模型是 gated 模型。请仅在当前 PowerShell 会话设置 `$env:HF_TOKEN 后重新运行；脚本不会把 token 写入文件。"
    }

    Write-Host "正在验证 GPU 容器运行时..."
    Invoke-NativeChecked docker run --rm --gpus "device=$GpuDeviceId" $CudaImage nvidia-smi

    $SecurePassword = Read-Host "设置公网 Basic Auth 密码（至少 16 个字符，不会写入明文）" -AsSecureString
    $SecurePasswordConfirmation = Read-Host "再次输入相同密码" -AsSecureString
    $PlainPassword = $null
    $PlainPasswordConfirmation = $null
    try {
        $PlainPassword = Convert-SecureStringToPlainText -SecureValue $SecurePassword
        $PlainPasswordConfirmation = Convert-SecureStringToPlainText -SecureValue $SecurePasswordConfirmation
        if ($PlainPassword.Length -lt 16) {
            throw "Basic Auth 密码至少需要 16 个字符。"
        }
        if (-not [string]::Equals($PlainPassword, $PlainPasswordConfirmation, [StringComparison]::Ordinal)) {
            throw "两次输入的 Basic Auth 密码不一致。"
        }
        $HashOutput = $PlainPassword | & docker run --rm -i $CaddyImage caddy hash-password --algorithm argon2id
        if ($LASTEXITCODE -ne 0) {
            throw "Caddy Argon2id 密码哈希生成失败，退出码 $LASTEXITCODE。"
        }
    }
    finally {
        $PlainPassword = $null
        $PlainPasswordConfirmation = $null
        $SecurePassword.Dispose()
        $SecurePasswordConfirmation.Dispose()
    }
    $BasicAuthHash = ($HashOutput | Select-Object -Last 1).Trim()
    if ($BasicAuthHash -notlike '$argon2id$*') {
        throw "Caddy 返回的不是 Argon2id 哈希，拒绝继续。"
    }

    $VcsRef = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $VcsRef -notmatch '^[0-9a-f]{40}$') {
        throw "无法读取当前 Git 提交身份。"
    }
    $BuildVersion = "local-$($VcsRef.Substring(0, 12))"
    $DirtyStatus = @(& git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "无法检查当前工作区状态。"
    }
    if ($DirtyStatus.Count -gt 0) {
        $BuildVersion = "$BuildVersion-dirty"
    }
    $PublicOrigin = "https://$PublicAddress"
    $Lines = @(
        "BACKEND_IMAGE=ghcr.io/mason369/music-to-midi-backend:v1.6.0",
        "GATEWAY_IMAGE=ghcr.io/mason369/music-to-midi-gateway:v1.6.0",
        "VCS_REF=$VcsRef",
        "BUILD_VERSION=$BuildVersion",
        "PUBLIC_ADDRESS=$PublicAddress",
        "PUBLIC_ORIGIN=$PublicOrigin",
        "ACME_EMAIL=$AcmeEmail",
        "BASIC_AUTH_USER=$BasicAuthUser",
        "BASIC_AUTH_HASH='$BasicAuthHash'",
        "GPU_DEVICE_ID=$GpuDeviceId",
        "MUSIC_TO_MIDI_ENABLED_PROFILES=$Profiles",
        "MAX_UPLOAD_BYTES=4294967296",
        "MAX_REQUEST_BODY_SIZE=4GiB",
        "MAX_QUEUED_JOBS=4",
        "MIN_FREE_BYTES=21474836480",
        "RETENTION_DAYS=7",
        "RETENTION_MAX_JOBS=50",
        "RETENTION_MAX_BYTES=107374182400",
        "SHM_SIZE=2gb",
        "LOG_LEVEL=info"
    )
    [IO.File]::WriteAllLines($EnvFile, $Lines, [Text.UTF8Encoding]::new($false))
    $CurrentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    $RestrictedAcl = [Security.AccessControl.FileSecurity]::new()
    $RestrictedAcl.SetOwner($CurrentSid)
    $RestrictedAcl.SetAccessRuleProtection($true, $false)
    $RestrictedAcl.AddAccessRule(
        [Security.AccessControl.FileSystemAccessRule]::new(
            $CurrentSid,
            [Security.AccessControl.FileSystemRights]::FullControl,
            [Security.AccessControl.AccessControlType]::Allow
        )
    )
    Set-Acl -LiteralPath $EnvFile -AclObject $RestrictedAcl
    Write-Host "已写入 $EnvFile（不含明文密码和 HF_TOKEN，ACL 仅允许当前用户）。"

    Invoke-NativeChecked docker compose --env-file $EnvFile -f $ComposeFile config --quiet
    Invoke-NativeChecked docker compose --env-file $EnvFile -f $ComposeFile build --pull

    Write-Host "正在显式下载并严格校验所选模型；推理请求本身不会下载模型..."
    Invoke-NativeChecked docker compose --env-file $EnvFile -f $ComposeFile --profile tools run --rm model-init
    Invoke-NativeChecked docker compose --env-file $EnvFile -f $ComposeFile up -d

    $Ready = $false
    $Deadline = [DateTime]::UtcNow.AddMinutes(10)
    while ([DateTime]::UtcNow -lt $Deadline) {
        & docker compose --env-file $EnvFile -f $ComposeFile exec -T backend python /app/docker/healthcheck.py 2>$null
        if ($LASTEXITCODE -eq 0) {
            $Ready = $true
            break
        }
        Start-Sleep -Seconds 5
    }
    if (-not $Ready) {
        & docker compose --env-file $EnvFile -f $ComposeFile ps
        & docker compose --env-file $EnvFile -f $ComposeFile logs --tail 200 backend gateway
        throw "后端在 10 分钟内未通过真实 readiness 门禁，部署已显式失败。"
    }

    $ValidationPassword = Read-Host "再次输入 Basic Auth 密码以验证公网 HTTPS" -AsSecureString
    try {
        $Authorization = Get-BasicAuthorizationValue -Username $BasicAuthUser -SecurePassword $ValidationPassword
        $Response = Invoke-RestMethod -Uri "$PublicOrigin/api/v1/ready" -Headers @{ Authorization = $Authorization } -TimeoutSec 30
    }
    finally {
        $Authorization = $null
        $ValidationPassword.Dispose()
    }
    if ($Response.status -ne "ready") {
        throw "公网 readiness 返回了非 ready 状态：$($Response | ConvertTo-Json -Compress)"
    }
    Write-Host "部署验证通过：$PublicOrigin（HTTPS、Basic Auth、后端 readiness 均已实测）。"
}
finally {
    Pop-Location
}
