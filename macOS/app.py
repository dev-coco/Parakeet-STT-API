# -*- coding: utf-8 -*-

# 禁用 TorchScript，避免冻结后 JIT 读取源码导致 OSError
import os
os.environ["PYTORCH_JIT"] = "0"

import sys
import tempfile
import time
import gc
import logging
import threading
import subprocess
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import librosa
import soundfile as sf

# 运行时资源根目录
def _get_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.abspath(os.path.dirname(__file__))

BASE_DIR = _get_base_dir()

# 模型与缓存目录
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.environ["HF_HOME"] = MODELS_DIR

# Nemo 模型
# NEMO_BUNDLE = os.path.join(MODELS_DIR, "parakeet-tdt-0.6b-v3.nemo")
FFMPEG_PATH = os.path.join(BASE_DIR, "ffmpeg/ffmpeg")

# 屏蔽冗余日志
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)
logging.disable(logging.CRITICAL)

print("正在启动中")
print("🚀 正在加载 ASR 模型（仅首次执行）...")

# 深度学习依赖放在环境变量设置之后再导入
import torch
import nemo.collections.asr as nemo_asr

# 8) 加载 Nemo ASR 模型
# ASR_MODEL = nemo_asr.models.ASRModel.restore_from(NEMO_BUNDLE)
ASR_MODEL = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
ASR_MODEL.eval()
print("✅ 模型加载完成，准备就绪。")

# 9) Flask 应用
app = Flask(__name__)

def memory_stats():
    """打印当前CUDA显存使用情况，用于调试内存占用"""
    if torch.cuda.is_available():
        print('Memory allocated:', torch.cuda.memory_allocated() / 1024 ** 2, 'MB')
        print('Memory cached:', torch.cuda.memory_reserved() / 1024 ** 2, 'MB\n')
    else:
        print("CUDA not available.\n")

def safe_json(obj):
    """递归安全转换numpy类型为Python基础类型，便于JSON序列化"""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def transcribe_audio(audio_filepath):
    """
    使用已加载的ASR模型对给定音频文件进行转录：
    1) 用 ffmpeg 转单声道 16kHz WAV；
    2) librosa 复核并重写；
    3) NeMo 模型转录并返回结果。
    """
    extracted_wav = None
    try:
        # 生成临时WAV文件路径
        extracted_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        # ffmpeg 转码
        cmd = [
            FFMPEG_PATH, "-y", "-i", audio_filepath,
            "-ac", "1", "-ar", "16000", "-vn", "-loglevel", "error", extracted_wav
        ]
        subprocess.run(cmd, check=True)

        # librosa 加载复核并重写
        data, sr = librosa.load(extracted_wav, sr=16000, mono=True)
        sf.write(extracted_wav, data, sr)

        # 语音识别
        print("🧠 开始语音识别...")
        start = time.time()
        results = ASR_MODEL.transcribe([extracted_wav], timestamps=True)
        print(f"✅ 识别完成，用时 {time.time() - start:.2f} 秒")

        if not results:
            return {"error": "无识别结果"}

        r = results[0]
        text = getattr(r, 'text', "")
        timestamps = getattr(r, 'timestamp', {})
        return {"text": text, "timestamps": safe_json(timestamps)}

    except Exception as e:
        return {"error": str(e)}

    finally:
        # 清理临时与显存
        if extracted_wav and os.path.exists(extracted_wav):
            os.remove(extracted_wav)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    接收POST请求，处理上传的音频文件，后台线程调用转录函数防止阻塞，超时300秒
    """
    if 'audio' not in request.files:
        return jsonify({"error": "未提供音频文件"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400

    filename = secure_filename(file.filename)
    print(f"收到文件: {filename}")

    temp_filepath = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            temp_filepath = tmp_file.name

        print(f"音频已保存: {temp_filepath}")

        result_container = {}

        def worker():
            result_container["data"] = transcribe_audio(temp_filepath)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=300)

        if thread.is_alive():
            print("⚠️ 转录超时（300秒）")
            return jsonify({"error": "转录超时"}), 500

        result = result_container.get("data", {"error": "未知错误"})
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(f"清理临时文件: {temp_filepath}")

if __name__ == '__main__':
    print("🌐 Flask 服务器启动中...")
    app.run(host='0.0.0.0', port=1643, debug=False)
