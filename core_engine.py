"""
core_engine.py —— 核心逻辑层（不含界面）

负责：
    1. 按键触发录音
    2. 通过 subprocess 启动完全独立的识别进程(whisper_worker_standalone.py)，
       两者用本地socket通信，物理隔离PyQt6和faster-whisper，避免底层库冲突
    3. 检测英雄联盟客户端是否在运行
    4. 转写完成后自动填入游戏聊天框（半自动，不自动发送）
"""

import os
import sys
import time
import threading
import subprocess
from multiprocessing.connection import Listener

import numpy as np
import sounddevice as sd
import keyboard
import psutil


LOL_PROCESS_NAMES = ["League of Legends.exe"]

CHAT_OPEN_KEY = "enter"
TYPE_DELAY = 0.05

# 主进程和识别子进程之间通信的共享密钥（纯本地通信，不涉及网络安全问题，
# multiprocessing.connection要求必须提供一个，固定写死即可）
AUTH_KEY = b"lol-voice-tool-local-ipc"

# 关键：区分"开发环境直接跑.py"和"打包成exe之后运行"两种情况。
# 开发时：用当前python解释器去执行 whisper_worker_standalone.py
# 打包后：sys.executable 指向的是主exe自己，这时候没有python解释器可用，
#         必须改为直接调用打包好的子exe（whisper_worker_standalone.exe）
if getattr(sys, "frozen", False):
    # 打包后运行：子exe应该和主exe放在同一目录下
    _base_dir = os.path.dirname(sys.executable)
    WORKER_COMMAND = [os.path.join(_base_dir, "whisper_worker_standalone.exe")]
else:
    # 开发环境直接运行.py
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    WORKER_SCRIPT = os.path.join(_base_dir, "whisper_worker_standalone.py")
    WORKER_COMMAND = [sys.executable, WORKER_SCRIPT]


def is_lol_running() -> bool:
    try:
        for proc in psutil.process_iter(attrs=["name"]):
            name = proc.info.get("name") or ""
            if name in LOL_PROCESS_NAMES:
                return True
    except Exception:
        pass
    return False


class VoiceEngine:
    def __init__(self, model_size="base", language="zh", record_key="v",
                 on_status=None, on_result=None):
        self.model_size = model_size
        self.language = language
        self.record_key = record_key

        self.on_status = on_status or (lambda msg: None)
        self.on_result = on_result or (lambda text: None)

        self.recording = False
        self.audio_buffer = []
        self.stream = None

        self._hotkey_handles = []
        self._running = False

        self._worker_proc = None    # subprocess.Popen 对象
        self._conn = None           # multiprocessing.connection 连接
        self._send_lock = threading.Lock()

    # ---------- 子进程管理 ----------

    def load_model(self):
        """
        启动独立子进程并加载模型。
        首次调用：启动全新子进程。
        之后调用：复用已有子进程，让它重新加载模型（更快，不用重启进程）。
        """
        if self._worker_proc is None:
            self.on_status(f"正在启动识别进程（{self.model_size}）...")
            threading.Thread(target=self._spawn_worker, daemon=True).start()
        else:
            self.on_status(f"正在切换模型（{self.model_size}）...")
            self._send(("reload_model", self.model_size))

    def _spawn_worker(self):
        try:
            listener = Listener(("localhost", 0), authkey=AUTH_KEY)
            port = listener.address[1]

            # 用全新的python解释器运行一个完全独立的脚本，
            # 这个子进程从始至终不会加载PyQt6
            self._worker_proc = subprocess.Popen(
                WORKER_COMMAND + [str(port), AUTH_KEY.decode()],
            )

            self._conn = listener.accept()  # 阻塞直到子进程连过来
            self._conn.send(self.model_size)  # 告诉子进程用什么模型

            self._listen_thread = threading.Thread(target=self._listen_conn, daemon=True)
            self._listen_thread.start()
        except Exception as e:
            self.on_status(f"[识别进程启动失败] {e}")

    def _listen_conn(self):
        """后台线程：持续接收子进程发回来的消息"""
        while True:
            try:
                kind, msg = self._conn.recv()
            except (EOFError, OSError):
                break

            if kind == "ready":
                self.on_status(f"就绪，按住 [{self.record_key.upper()}] 说话")
            elif kind == "result":
                if msg:
                    self.on_result(msg)
                    self._fill_into_game_chat(msg)
                    self.on_status(f"已识别: {msg}")
                else:
                    self.on_status("未识别到有效内容")
            elif kind == "error":
                self.on_status(f"[错误] {msg}")

    def _send(self, message):
        if self._conn is None:
            self.on_status("识别进程尚未就绪，请稍候")
            return
        try:
            with self._send_lock:
                self._conn.send(message)
        except Exception as e:
            self.on_status(f"[通信失败] {e}")

    # ---------- 录音相关 ----------

    def _audio_callback(self, indata, frames, time_info, status):
        try:
            if self.recording:
                self.audio_buffer.append(indata.copy())
        except Exception as e:
            self.on_status(f"[录音回调异常] {e}")

    def _start_recording(self):
        if self.recording:
            return
        self.on_status("正在录音...")
        self.audio_buffer = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()

    def _stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.stream.stop()
        self.stream.close()

        if not self.audio_buffer:
            self.on_status("未录到声音")
            return

        audio_data = np.concatenate(self.audio_buffer, axis=0).flatten()
        duration = len(audio_data) / 16000
        if duration < 0.3:
            self.on_status("录音过短，已忽略")
            return

        self.on_status("正在识别...")
        self._send(("transcribe", (audio_data, self.language)))

    # ---------- 填入游戏聊天框 ----------

    def _fill_into_game_chat(self, text: str):
        try:
            keyboard.send(CHAT_OPEN_KEY)
            time.sleep(TYPE_DELAY)
            keyboard.write(text)
        except Exception as e:
            self.on_status(f"填入聊天框失败: {e}")

    # ---------- 快捷键绑定/解绑 ----------

    def set_record_key(self, new_key: str):
        was_running = self._running
        if was_running:
            self._unbind_hotkey()
        self.record_key = new_key
        if was_running:
            self._bind_hotkey()

    def _bind_hotkey(self):
        def safe_start(_):
            try:
                self._start_recording()
            except Exception as e:
                self.on_status(f"[开始录音异常] {e}")

        def safe_stop(_):
            try:
                self._stop_recording()
            except Exception as e:
                self.on_status(f"[停止录音异常] {e}")

        h1 = keyboard.on_press_key(self.record_key, safe_start)
        h2 = keyboard.on_release_key(self.record_key, safe_stop)
        self._hotkey_handles = [h1, h2]

    def _unbind_hotkey(self):
        for h in self._hotkey_handles:
            try:
                keyboard.unhook(h)
            except Exception:
                pass
        self._hotkey_handles = []

    # ---------- 生命周期 ----------

    def start(self):
        if self._running:
            return
        self._running = True
        self._bind_hotkey()

    def stop(self):
        self._running = False
        self._unbind_hotkey()
        if self.recording:
            self._stop_recording()
        if self._conn is not None:
            try:
                self._conn.send(None)
                self._conn.close()
            except Exception:
                pass
        if self._worker_proc is not None:
            self._worker_proc.terminate()