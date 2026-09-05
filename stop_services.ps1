# SenseVoice 服务停止脚本（按端口定位进程）
$ErrorActionPreference = 'SilentlyContinue'

$ports = 47825, 47824
$found = $false
foreach ($port in $ports) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        $procId = $conn.OwningProcess
        $name = if ($port -eq 47825) { 'ASR API' } else { 'Web UI' }
        Write-Host ("停止 [{0}] 端口 {1} -> PID {2}" -f $name, $port, $procId)
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        $found = $true
    }
}
if (-not $found) {
    Write-Host "未发现运行中的 SenseVoice 服务（端口 47825 / 47824 均空闲）。"
} else {
    Write-Host "已停止所有 SenseVoice 服务。"
}