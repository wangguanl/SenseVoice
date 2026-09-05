# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os
import re
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated
from typing import List
from enum import Enum
import torch
import torchaudio
import soundfile as sf
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
from utils.device_env import resolve_sensevoice_device

TARGET_FS = 16000


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


model_dir = r"E:\huggingface_cache\hub\models--FunAudioLLM--SenseVoiceSmall\snapshots\3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
m, kwargs = SenseVoiceSmall.from_pretrained(
    model=model_dir, device=resolve_sensevoice_device()
)
m.eval()

regex = r"<\|.*\|>"

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.post("/api/v1/asr")
async def turn_audio_to_text(
    files: Annotated[List[UploadFile], File(description="wav or mp3 audios in 16KHz")],
    keys: Annotated[str, Form(description="name of each audio joined with comma")] = None,
    lang: Annotated[Language, Form(description="language of audio content")] = "auto",
    use_itn: Annotated[bool, Form(description="apply inverse text normalization")] = False,
):
    audios = []
    errors = {}
    file_keys = []  # 与 audios 逐一对应的文件名
    for file in files:
        try:
            file_io = BytesIO(await file.read())
            data_or_path_or_list, audio_fs = sf.read(file_io, dtype='float32')

            # transform to target sample
            if len(data_or_path_or_list.shape) > 1:
                data_or_path_or_list = data_or_path_or_list.mean(-1)
            if audio_fs != TARGET_FS:
                resampler = torchaudio.transforms.Resample(orig_freq=audio_fs, new_freq=TARGET_FS)
                data_or_path_or_list = resampler(torch.from_numpy(data_or_path_or_list)[None, :])[0]

            # 直接推理（无 VAD）单段音频限制在 30 秒以内
            dur = data_or_path_or_list.shape[0] / TARGET_FS
            if dur > 30:
                errors[file.filename] = f"音频时长 {dur:.1f}s，超过 30 秒限制（直接推理模式），请使用 Web UI 长音频或分段上传"
                continue

            audios.append(data_or_path_or_list)
            file_keys.append(file.filename)
        except Exception as e:
            errors[file.filename] = f"音频读取失败: {type(e).__name__}: {e}"

    if not audios:
        return {"result": [], "errors": errors}

    if lang == "":
        lang = "auto"

    # key 必须与成功加载的 audios 一一对齐（剔除失败/超长的文件）
    if not keys:
        key = file_keys
    else:
        provided = keys.split(",")
        key = [provided[i] if i < len(provided) else fk for i, fk in enumerate(file_keys)]

    res = m.inference(
        data_in=audios,
        language=lang,  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=use_itn,
        ban_emo_unk=False,
        key=key,
        fs=TARGET_FS,
        **kwargs,
    )
    if len(res) == 0:
        return {"result": []}
    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
        it["text"] = rich_transcription_postprocess(it["text"])
    return {"result": res[0], "errors": errors}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("SENSEVOICE_API_PORT", "47825")))
