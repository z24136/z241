"""
whisper_worker_standalone.py —— 完全独立的语音识别工作进程

关键设计：
    这个文件必须始终保持"干净"——绝对不能 import PyQt6 或 gui_app.py
    里的任何东西，也不能被 gui_app.py 用 multiprocessing.Process 直接
    拉起（那样在Windows下会连带重新导入gui_app.py，把PyQt6也带进来，
    导致和faster-whisper冲突崩溃，这是我们已经踩过的坑）。

    正确用法：主进程通过 subprocess.Popen 启动一个全新的python解释器
    来运行这个文件，两者之间只通过本地socket通信(multiprocessing.connection)，
    这样这个进程从出生开始就绝对不会加载PyQt6，物理上不可能冲突。

运行方式（由 core_engine.py 自动调用，不需要手动运行）：
    python whisper_worker_standalone.py <port> <authkey>
"""

import sys
import os
from multiprocessing.connection import Client


def main():
    port = int(sys.argv[1])
    authkey = sys.argv[2].encode()

    conn = Client(("localhost", port), authkey=authkey)

    # 避免tqdm监控线程相关的已知不稳定问题，双重保险
    import tqdm
    tqdm.tqdm.monitor_interval = 0
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    from faster_whisper import WhisperModel

    def load(size):
        return WhisperModel(size, device="cpu", compute_type="int8")

    # 第一条消息：主进程告诉我们用什么模型
    model_size = conn.recv()

    try:
        model = load(model_size)
        conn.send(("ready", None))
    except Exception as e:
        conn.send(("error", f"模型加载失败: {e}"))
        return

    while True:
        try:
            item = conn.recv()
        except EOFError:
            break  # 主进程关闭了连接，说明程序要退出了

        if item is None:
            break

        cmd, payload = item

        if cmd == "transcribe":
            audio_data, language = payload
            try:
                segments, info = model.transcribe(audio_data, language=language, beam_size=5)
                text = "".join(seg.text for seg in segments).strip()
                conn.send(("result", text))
            except Exception as e:
                conn.send(("error", f"识别失败: {e}"))

        elif cmd == "reload_model":
            try:
                model = load(payload)
                conn.send(("ready", None))
            except Exception as e:
                conn.send(("error", f"模型重新加载失败: {e}"))

    conn.close()


if __name__ == "__main__":
    main()