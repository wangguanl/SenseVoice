# SenseVoice 服务一键启动脚本
$ErrorActionPreference = 'SilentlyContinue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$py    = Join-Path $root '.venv\Scripts\python.exe'
$logs  = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

# 环境变量
$env:HF_HOME            = 'E:\huggingface_cache'
$env:HF_HUB_DISABLE_XET = '1'
# ffmpeg：用于 Gradio 解码 m4a / 非标准 mp3 等 libsndfile 不支持的格式
$env:FFMPEG_BIN = 'E:\Programs\ffmpeg-master-latest-win64-gpl\bin'
if (Test-Path "$env:FFMPEG_BIN\ffmpeg.exe") {
    $env:PATH = "$env:FFMPEG_BIN;" + $env:PATH
}
$env:PATH = (Resolve-Path (Join-Path $root '.venv\Scripts')).Path + ';' + $env:PATH

function Test-Port([int]$port) {
    $c = New-Object Net.Sockets.TcpClient
    try { $c.Connect('127.0.0.1', $port); return $c.Connected } catch { return $false } finally { $c.Close() }
}

# 后台启动单个服务：stdout/stderr 分别写日志，最小化窗口（关闭该窗口即停止服务）
function Start-Svc([string]$name, [string]$script) {
    $out = Join-Path $logs "$name.log"
    $err = Join-Path $logs "$name.err.log"
    Start-Process -FilePath $py -ArgumentList $script -WorkingDirectory $root `
        -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Minimized
    Write-Host "[$name] 已启动 -> $script"
}

Write-Host ""
Write-Host "=========================================="
Write-Host "   SenseVoice 服务一键启动"
Write-Host "=========================================="
Write-Host "   ASR API : http://127.0.0.1:47825"
Write-Host "   Web UI  : http://127.0.0.1:47824"
Write-Host ""
Write-Host "   (服务窗口已最小化，关闭窗口即停止对应服务)"
Write-Host ""

# 先启动 API 并等待就绪，再启动 WebUI，避免两进程同时加载模型导致竞争卡住
if (Test-Port 47825) {
    Write-Host "[API] 端口 47825 已被占用，已跳过（可能已在运行）"
} else {
    Start-Svc 'API' 'api.py'
    Write-Host "[API] 等待模型加载并监听端口 47825 ..."
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 4
        if (Test-Port 47825) { Write-Host "[API] 已在端口 47825 就绪"; break }
    }
    if (-not (Test-Port 47825)) { Write-Host "[API] 60 秒内未就绪，请查看 logs\api.log" }
}

if (Test-Port 47824) {
    Write-Host "[WebUI] 端口 47824 已被占用，已跳过（可能已在运行）"
} else {
    Start-Svc 'WebUI' 'webui.py'
    Write-Host "[WebUI] 已后台启动，加载模型约需 30~90 秒"
}

Write-Host ""
Write-Host "日志目录: $logs"
Start-Sleep -Seconds 3
Start-Process 'http://127.0.0.1:47824'
Write-Host "已尝试打开浏览器访问 Web UI，若未就绪请稍候刷新。"
Write-Host "如需停止服务，请运行 stop_services.ps1"