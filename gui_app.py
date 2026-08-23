"""
gui_app.py —— LOL 语音转文字工具（图形界面入口）

简洁设计：
    - 主窗口：设置快捷键 / 模型大小 / 识别语言，显示LOL运行状态
    - 关闭主窗口不退出程序，缩到系统托盘继续后台工作
    - 双击托盘图标重新打开设置窗口
    - 右键托盘图标可选择"退出"

依赖安装：
    pip install faster-whisper sounddevice numpy keyboard psutil PyQt6

注意：需要以管理员权限运行终端，keyboard 库监听/模拟按键才能在游戏内生效。
"""

import sys
import threading
import multiprocessing as mp
import faulthandler

# 开启崩溃诊断：一旦发生原生层面的崩溃（比如0xC0000005访问违例），
# 会把当时各线程的调用栈写入 crash_log.txt，方便定位是哪一行代码引发的
_crash_log = open("crash_log.txt", "w", encoding="utf-8")
faulthandler.enable(file=_crash_log, all_threads=True)

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QSystemTrayIcon, QMenu
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QAction, QPixmap, QColor

import keyboard
from core_engine import VoiceEngine, is_lol_running


MODEL_OPTIONS = ["tiny", "base", "small", "medium"]
LANGUAGE_OPTIONS = [("中文", "zh"), ("英文", "en"), ("自动检测", None)]


def make_dot_icon(color: str) -> QIcon:
    """生成一个纯色小圆点图标，用作托盘图标（免去准备图片资源）"""
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    from PyQt6.QtGui import QPainter, QBrush
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QBrush(QColor(color)))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(4, 4, 24, 24)
    painter.end()
    return QIcon(pixmap)


class Bridge(QObject):
    """用于把后台线程的回调安全传回Qt主线程"""
    status_changed = pyqtSignal(str)
    result_received = pyqtSignal(str)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LOL 语音转文字")
        self.setFixedSize(320, 260)

        self.bridge = Bridge()
        self.bridge.status_changed.connect(self.on_status)
        self.bridge.result_received.connect(self.on_result)

        self.engine = VoiceEngine(
            model_size="base",
            language="zh",
            record_key="v",
            on_status=lambda msg: self.bridge.status_changed.emit(msg),
            on_result=lambda text: self.bridge.result_received.emit(text),
        )

        self._capturing_key = False
        self._build_ui()
        self._build_tray()

        # 模型加载必须在主线程完成：ctranslate2(faster-whisper底层)在部分
        # Windows环境下，如果在非主线程做初始化会触发access violation崩溃。
        # 这里用 QTimer.singleShot(0, ...) 让窗口先显示出来，
        # 再在下一次事件循环时（依然是主线程）加载模型，
        # 加载期间界面会短暂卡顿几秒，但比闪退稳定得多。
        QTimer.singleShot(0, self._load_and_start)

        # 定时检测LOL是否在运行
        self.lol_timer = QTimer(self)
        self.lol_timer.timeout.connect(self._check_lol_status)
        self.lol_timer.start(3000)
        self._check_lol_status()

    # ---------- 界面构建 ----------

    def _build_ui(self):
        layout = QVBoxLayout()

        # 状态提示
        self.status_label = QLabel("正在初始化...")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # LOL 运行状态
        self.lol_status_label = QLabel("检测中...")
        layout.addWidget(self.lol_status_label)

        # 快捷键设置
        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("录音快捷键:"))
        self.key_button = QPushButton(self.engine.record_key.upper())
        self.key_button.clicked.connect(self._start_key_capture)
        key_row.addWidget(self.key_button)
        layout.addLayout(key_row)

        # 模型选择
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("识别模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_OPTIONS)
        self.model_combo.setCurrentText("base")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_row.addWidget(self.model_combo)
        layout.addLayout(model_row)

        # 语言选择
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("识别语言:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems([label for label, _ in LANGUAGE_OPTIONS])
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_row.addWidget(self.lang_combo)
        layout.addLayout(lang_row)

        # 最近一次识别结果
        self.result_label = QLabel("识别结果会显示在这里")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("color: #666; margin-top: 8px;")
        layout.addWidget(self.result_label)

        hint = QLabel("关闭窗口不会退出程序，会缩小到系统托盘继续运行")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #999; font-size: 11px; margin-top: 8px;")
        layout.addWidget(hint)

        self.setLayout(layout)

    def _build_tray(self):
        self.tray_icon = QSystemTrayIcon(make_dot_icon("#2ecc71"), self)
        self.tray_icon.setToolTip("LOL 语音转文字 - 运行中")

        menu = QMenu()
        show_action = QAction("打开设置", self)
        show_action.triggered.connect(self._show_window)
        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self._quit_app)
        menu.addAction(show_action)
        menu.addAction(quit_action)

        self.tray_icon.setContextMenu(menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    # ---------- 事件处理 ----------

    def _load_and_start(self):
        self.engine.load_model()
        self.engine.start()

    def _on_model_changed(self, text):
        self.engine.model_size = text
        self.status_label.setText(f"模型已切换为 {text}，重新加载中...")
        # 同样必须在主线程加载，原因见 __init__ 中的说明
        QTimer.singleShot(0, self.engine.load_model)

    def _on_language_changed(self, index):
        _, code = LANGUAGE_OPTIONS[index]
        self.engine.language = code

    def _start_key_capture(self):
        if self._capturing_key:
            return
        self._capturing_key = True
        self.key_button.setText("请按键...")
        threading.Thread(target=self._capture_key_thread, daemon=True).start()

    def _capture_key_thread(self):
        """阻塞等待用户按下一个新键，然后回传给主线程更新"""
        try:
            event = keyboard.read_event(suppress=False)
            while event.event_type != keyboard.KEY_DOWN:
                event = keyboard.read_event(suppress=False)
            new_key = event.name
            self.bridge.status_changed.emit(f"__SET_KEY__:{new_key}")
        except Exception as e:
            self.bridge.status_changed.emit(f"快捷键捕获失败: {e}")

    def on_status(self, msg: str):
        # 特殊消息：用于从捕获线程回传新快捷键（避免跨线程直接改UI控件）
        if msg.startswith("__SET_KEY__:"):
            new_key = msg.split(":", 1)[1]
            self.engine.set_record_key(new_key)
            self.key_button.setText(new_key.upper())
            self._capturing_key = False
            self.status_label.setText(f"快捷键已改为 {new_key.upper()}")
            return
        self.status_label.setText(msg)

    def on_result(self, text: str):
        self.result_label.setText(f"最近识别: {text}")

    def _check_lol_status(self):
        if is_lol_running():
            self.lol_status_label.setText("英雄联盟: 运行中 ✅")
            self.tray_icon.setIcon(make_dot_icon("#2ecc71"))
        else:
            self.lol_status_label.setText("英雄联盟: 未运行 ⚪")
            self.tray_icon.setIcon(make_dot_icon("#95a5a6"))

    # ---------- 窗口/托盘行为 ----------

    def closeEvent(self, event):
        """关闭窗口时缩到托盘，而不是真正退出"""
        event.ignore()
        self.hide()
        self.tray_icon.showMessage(
            "LOL 语音转文字",
            "程序已缩小到托盘，仍在后台运行",
            QSystemTrayIcon.MessageIcon.Information,
            2000,
        )

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._show_window()

    def _show_window(self):
        self.show()
        self.activateWindow()

    def _quit_app(self):
        self.engine.stop()
        self.tray_icon.hide()
        QApplication.quit()


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # 关闭窗口不退出，靠托盘菜单退出

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    mp.freeze_support()  # Windows下使用multiprocessing必须加这行，否则子进程可能异常
    main()