import os
import tempfile
import time
import gc
import logging

# 设置 PyTorch 分布式弹性多进程日志等级为错误，屏蔽提示信息
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)

import torch
from flask import Flask, request, jsonify  # Flask框架处理HTTP请求和响应
from werkzeug.utils import secure_filename  # 处理文件名安全
import numpy as np
import threading  # 用于启动后台线程
import subprocess  # 用于调用外部命令，比如ffmpeg
import librosa  # 音频处理库
import soundfile as sf  # 音频读写库
import nemo.collections.asr as nemo_asr  # NVIDIA NeMo 语音识别模型库


# 设置HF_HOME环境变量，指定模型存储路径（transformers常用环境变量）
os.environ['HF_HOME'] = './models'

# 禁用所有Python日志的严重性低于CRITICAL的日志
logging.disable(logging.CRITICAL)

print("正在启动中")  # 启动提示


# 创建Flask应用，配置最大请求体大小为100MB
app = Flask(__name__)

# === 辅助函数 ===
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


# === 加载预训练ASR模型，只执行一次，节省资源 ===
print("🚀 正在加载 ASR 模型（仅首次执行）...")
# 加载模型
ASR_MODEL = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
# ASR_MODEL = nemo_asr.models.ASRModel.restore_from("models/parakeet-tdt-0.6b-v3.nemo")
ASR_MODEL.eval()  # 设为推理模式
print("✅ 模型加载完成，准备就绪。")


# === 音频转录函数 ===
def transcribe_audio(audio_filepath):
    """
    使用已加载的ASR模型对给定音频文件进行转录

    主要流程：
    1. 使用ffmpeg将音频转换为单声道16kHz WAV格式
    2. 使用librosa加载并确认音频数据正确，重新写入确保格式
    3. 调用模型进行转录，返回文本和时间戳
    """
    try:
        # 生成临时WAV文件路径
        extracted_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        # 调用 ffmpeg 做格式转换，输出单声道 16kHz 波形文件
        cmd = [
            "ffmpeg", "-y", "-i", audio_filepath,
            "-ac", "1", "-ar", "16000", "-vn", "-loglevel", "error", extracted_wav
        ]
        subprocess.run(cmd, check=True)

        # 加载音频确保正确性，librosa 加载后的数据重新写入
        data, sr = librosa.load(extracted_wav, sr=16000, mono=True)
        sf.write(extracted_wav, data, sr)

        # 调用 NeMo 模型转录
        print("🧠 开始语音识别...")
        start = time.time()

        # 传入文件路径列表给模型，开启时间戳
        results = ASR_MODEL.transcribe([extracted_wav], timestamps=True)

        print(f"✅ 识别完成，用时 {time.time() - start:.2f} 秒")
        if not results:
            return {"error": "无识别结果"}

        # 解析第一个结果，提取文本和时间戳
        r = results[0]
        text = getattr(r, 'text', "")
        timestamps = getattr(r, 'timestamp', {})
        return {"text": text, "timestamps": safe_json(timestamps)}

    except Exception as e:
        # 出现异常返回错误信息
        return {"error": str(e)}

    finally:
        # 清理临时 WAV 文件和显存
        if os.path.exists(extracted_wav):
            os.remove(extracted_wav)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# === Flask 路由定义 ===
@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    接收POST请求，处理上传的音频文件，后台线程调用转录函数防止阻塞，超时300秒
    
    返回结构化的JSON，包含识别文本和时间戳，或错误信息
    """
    if 'audio' not in request.files:
        return jsonify({"error": "未提供音频文件"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400

    # 处理安全文件名，避免路径穿越攻击
    filename = secure_filename(file.filename)
    print(f"收到文件: {filename}")

    try:
        # 创建一个临时文件保存上传内容
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            temp_filepath = tmp_file.name

        print(f"音频已保存: {temp_filepath}")

        # 后台线程执行转录，防止阻塞主进程
        result_container = {}

        def worker():
            result_container["data"] = transcribe_audio(temp_filepath)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=300)  # 设定300秒超时

        if thread.is_alive():
            # 超时处理
            print("⚠️ 转录超时（300秒）")
            return jsonify({"error": "转录超时"}), 500

        result = result_container.get("data", {"error": "未知错误"})
        return jsonify(result)

    except Exception as e:
        # 捕获异常返回错误状态
        return jsonify({"error": str(e)}), 500

    finally:
        # 清理临时文件
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(f"清理临时文件: {temp_filepath}")


# 程序入口，启动 Flask 服务器
if __name__ == '__main__':
    print("🌐 Flask 服务器启动中...")
    app.run(host='0.0.0.0', port=1643, debug=False)
