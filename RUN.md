# 运行命令

- 项目：SenseVoice（FunAudioLLM 音频理解 / ASR+情感+事件）
- 生成时间：2026-09-09
- 运行方式：直接运行（`.venv` + `webui.py` / `api.py`）；可选 Docker
- 硬件评估：**满足**（空卡 RTX 4080 16GB；SenseVoiceSmall 常规精度可跑）

## 环境准备

```powershell
$env:Path = "E:\Programs\ffmpeg-master-latest-win64-gpl\bin;" + $env:Path
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "E:\huggingface_cache"
# 已有 .venv（funasr/torch/gradio 可用）；若重建：
# uv venv .venv
# uv pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 启动

- 推荐：`pwsh -NoProfile -File .\start.ps1`
- 说明：多服务菜单默认选 **webui**；可多选 webui+api（会提示占卡确认）
- 等价手动命令：
  - webui：`$env:SENSEVOICE_WEBUI_PORT=47824; .\.venv\Scripts\python.exe .\webui.py`
  - api：`$env:SENSEVOICE_API_PORT=47825; .\.venv\Scripts\python.exe .\api.py`

## 验证

- WebUI：浏览器打开提示端口，上传短音频得到转写+情感/事件
- API：对监听端口发上传请求，返回文本；日志无 CUDA OOM

## 备注

- 镜像：HF=`https://hf-mirror.com`（官方 huggingface.co 探测超时）；PyPI 可用清华
- ffmpeg：`E:\Programs\ffmpeg-master-latest-win64-gpl\bin`
- 旧脚本 `start_services.ps1` 仍可用；日常请用根目录 `start.ps1`
- 本次未长时间启动 GPU 服务
