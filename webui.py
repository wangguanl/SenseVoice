# coding=utf-8

import os
import librosa
import base64
import io
import gradio as gr
import re

import numpy as np
import torch
import torchaudio


from funasr import AutoModel

try:
    # Try to connect internet
    model = AutoModel(
        model=r"E:\huggingface_cache\hub\models--FunAudioLLM--SenseVoiceSmall\snapshots\3847d57b6bdf2dd8875cb1508d2af43d80a16bf7",
        vad_model=r"C:\Users\wang\.cache\modelscope\models\iic--speech_fsmn_vad_zh-cn-16k-common-pytorch\snapshots\master",
        vad_kwargs={"max_single_segment_time": 30000},
        trust_remote_code=True,
    )
except Exception:
    # If it fails, use the model in cache
    local_model = os.path.expanduser("~/.cache/modelscope/hub/models/iic/SenseVoiceSmall/")
    local_vad = os.path.expanduser("~/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/")
    model = AutoModel(
        model=local_model,
        vad_model=local_vad,
        vad_kwargs={"max_single_segment_time": 30000},
        trust_remote_code=True,
        disable_update=True,  # Critical! Keep it offline
    )

import re

emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "❓",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
	"<|Cry|>": "😭",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "😷",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s


def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()

def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None
	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		#else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()
	new_s = new_s.replace("The.", " ")
	return new_s.strip()

def model_inference(input_wav, language, fs=16000):
	# task_abbr = {"Speech Recognition": "ASR", "Rich Text Transcription": ("ASR", "AED", "SER")}
	language_abbr = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko",
					 "nospeech": "nospeech"}
	
	# task = "Speech Recognition" if task is None else task
	language = "auto" if len(language) < 1 else language
	selected_language = language_abbr[language]
	# selected_task = task_abbr.get(task)
	
	# print(f"input_wav: {type(input_wav)}, {input_wav[1].shape}, {input_wav}")
	
	if isinstance(input_wav, tuple):
		fs, input_wav = input_wav
		input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
		if len(input_wav.shape) > 1:
			input_wav = input_wav.mean(-1)
		if fs != 16000:
			print(f"audio_fs: {fs}")
			resampler = torchaudio.transforms.Resample(fs, 16000)
			input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
			input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
	
	
	merge_vad = True #False if selected_task == "ASR" else True
	print(f"language: {language}, merge_vad: {merge_vad}")
	text = model.generate(input=input_wav,
						  cache={},
						  language=language,
						  use_itn=True,
						  batch_size_s=60, merge_vad=merge_vad)
	
	print(text)
	text = text[0]["text"]
	text = format_str_v3(text)
	
	print(text)
	
	return text


audio_examples = [
    ["example/zh.mp3", "zh"],
    ["example/yue.mp3", "yue"],
    ["example/en.mp3", "en"],
    ["example/ja.mp3", "ja"],
    ["example/ko.mp3", "ko"],
    ["example/emo_1.wav", "auto"],
    ["example/emo_2.wav", "auto"],
    ["example/emo_3.wav", "auto"],
    #["example/emo_4.wav", "auto"],
    #["example/event_1.wav", "auto"],
    #["example/event_2.wav", "auto"],
    #["example/event_3.wav", "auto"],
    ["example/rich_1.wav", "auto"],
    ["example/rich_2.wav", "auto"],
    #["example/rich_3.wav", "auto"],
    ["example/longwav_1.wav", "auto"],
    ["example/longwav_2.wav", "auto"],
    ["example/longwav_3.wav", "auto"],
    #["example/longwav_4.wav", "auto"],
]



html_content = """
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">语音理解模型：SenseVoice-Small</h2>
    <p style="font-size: 18px;margin-left: 20px;">SenseVoice-Small 是一个仅编码器的语音基础模型，专为快速语音理解而设计。它涵盖多种能力，包括语音识别（ASR）、语种识别（LID）、语音情感识别（SER）和声学事件检测（AED）。支持中文、英文、粤语、日语和韩语的多语言识别。此外，它还提供极低的推理延迟，比 Whisper-small 快 7 倍，比 Whisper-large 快 17 倍。</p>
    <h2 style="font-size: 22px;margin-left: 0px;">使用说明</h2> <p style="font-size: 18px;margin-left: 20px;">上传音频文件或通过麦克风输入，然后选择语言。音频会被转写为对应的文本，并附带情感（😊 开心、😡 生气/激动、😔 难过）和声音事件类型（😀 笑声、🎼 音乐、👏 掌声、🤧 咳嗽/喷嚏、😭 哭声）。事件标签显示在文本前面，情感标签显示在文本后面。</p>
	<p style="font-size: 18px;margin-left: 20px;">建议音频输入时长在 30 秒以内。超过 30 秒的音频，建议使用本地部署。</p>
	<h2 style="font-size: 22px;margin-left: 0px;">相关仓库</h2>
	<p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/QwenAudio/SenseVoice" target="_blank">SenseVoice</a>：多语言语音理解模型</p>
	<p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/modelscope/FunASR" target="_blank">FunASR</a>：基础语音识别工具包</p>
	<p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/QwenAudio/CosyVoice" target="_blank">CosyVoice</a>：高质量多语言语音合成模型</p>
</div>
"""


def launch():
	with gr.Blocks(theme=gr.themes.Soft()) as demo:
		# gr.Markdown(description)
		gr.HTML(html_content)
		with gr.Row():
			with gr.Column():
				audio_inputs = gr.Audio(label="上传音频或使用麦克风录入")
				
				with gr.Accordion("配置"):
					language_inputs = gr.Dropdown(choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
												  value="auto",
												  label="语言")
				fn_button = gr.Button("开始识别", variant="primary")
				text_outputs = gr.Textbox(label="识别结果")
			gr.Examples(examples=audio_examples, inputs=[audio_inputs, language_inputs], examples_per_page=20)
		
		fn_button.click(model_inference, inputs=[audio_inputs, language_inputs], outputs=text_outputs)

	demo.launch(server_port=int(os.getenv("SENSEVOICE_WEBUI_PORT", "47824")))


if __name__ == "__main__":
	# iface.launch()
	launch()


