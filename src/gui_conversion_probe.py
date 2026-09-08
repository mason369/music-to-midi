"""Explicit headless acceptance of the actual desktop conversion lifecycle.

This diagnostic is opt-in, including in the frozen GUI executable. It neither
replaces model inference nor modifies the normal desktop startup path.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import re
import sys
import time

GUI_CONVERSION_PROBE_SWITCH = "--self-test-gui-conversion"


def parse_probe_args(argv):
    from src.core.manual_midi import MANUAL_MIDI_ROUTES

    parser = argparse.ArgumentParser(description="真实 GUI 转换链诊断（不操作桌面）")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--route", required=True, action="append", choices=MANUAL_MIDI_ROUTES)
    args = parser.parse_args(argv)
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if not args.input.is_file():
        parser.error("输入文件不存在")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("诊断输出必须是新目录或空目录，不能覆盖既有内容")
    return args


def check_regex_identity():
    import re._compiler as compiler
    import re._constants as constants
    import re._parser as parser

    if not (compiler.BRANCH is parser.BRANCH is constants.BRANCH):
        raise RuntimeError("正则 BRANCH 常量身份不一致")
    if re.compile(r"(?:guitar|piano)-\d+").fullmatch("guitar-04") is None:
        raise RuntimeError("正则分支编译/匹配失败")
    if re.Scanner([(r"abc", lambda *_: "a"), (r"def", lambda *_: "b")]).scan("abcdef") != (
        ["a", "b"],
        "",
    ):
        raise RuntimeError("正则 Scanner 分支匹配失败")
    return {"consistent": True, "branch_identity": id(constants.BRANCH)}


def wait_for_gui(predicate, description, seconds=900):
    """Pump real Qt events, including DeferredDelete; never fake a terminal state."""
    from PyQt6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    poll = QTimer()
    deadline = QTimer()
    deadline.setSingleShot(True)
    state = {}

    def check():
        try:
            if predicate():
                state["done"] = True
                loop.quit()
        except Exception as exc:
            state["error"] = exc
            loop.quit()

    poll.timeout.connect(check)
    deadline.timeout.connect(loop.quit)
    poll.start(20)
    deadline.start(int(seconds * 1000))
    check()
    if not state:
        loop.exec()
    poll.stop()
    deadline.stop()
    if "error" in state:
        raise state["error"]
    if not state.get("done"):
        raise TimeoutError(f"GUI 诊断超时：{description}")


def execute_probe(args, report, save):
    import mido
    from PyQt6.QtCore import QTimer, QT_VERSION_STR, PYQT_VERSION_STR, qVersion
    from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QTextEdit

    from src.core.manual_midi import build_manual_midi_config
    from src.gui.main_window import MainWindow
    from src.gui.widgets.project_panel import ProjectWorker
    from src.models.data_models import Config

    app = QApplication.instance() or QApplication([])
    old_quit = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)
    report.update(qt_compiled=QT_VERSION_STR, qt_runtime=qVersion(), pyqt=PYQT_VERSION_STR)
    window = MainWindow(Config(language="zh_CN", use_gpu=True, output_dir=str(args.output)))
    window.show()
    modal_errors = []
    modal_watch = QTimer()

    def record_modal_error():
        dialog = app.activeModalWidget()
        if isinstance(dialog, QDialog):
            details = [child.text() for child in dialog.findChildren(QLabel)]
            details += [child.toPlainText() for child in dialog.findChildren(QTextEdit)]
            modal_errors.append("\n".join([dialog.windowTitle(), *details]))
            # The real error dialog is acknowledged, never suppressed as success.
            dialog.reject()

    modal_watch.timeout.connect(record_modal_error)
    modal_watch.start(20)
    try:
        for index, route in enumerate(args.route, 1):
            case = {"route": route, "stage": "starting", "ok": False}
            report["cases"].append(case)
            save()
            config = build_manual_midi_config(window.config, route)
            panel = window.track_panel
            panel.set_processing_mode(config.processing_mode)
            panel.set_transcription_backend(config.transcription_backend)
            panel.set_yourmt3_model(config.yourmt3_model)
            panel.set_muscriptor_model(config.muscriptor_model)
            output = args.output / f"route-{index:02d}"
            output.mkdir()
            window.output_dir_edit.setText(str(output))
            window.dropzone.files_selected.emit([str(args.input)])
            wait_for_gui(window.start_btn.isEnabled, "开始按钮就绪", 30)
            started = time.monotonic()
            window.start_btn.click()
            worker = window.worker
            if not isinstance(worker, ProjectWorker):
                raise RuntimeError("真实开始按钮没有创建项目处理线程")
            state = {}
            worker.completed.connect(lambda snapshot: state.update(snapshot=snapshot))
            worker.failed.connect(lambda message: state.update(error=str(message)))
            case["stage"] = "converting"
            report["stage"] = "converting"
            save()
            wait_for_gui(lambda: window.worker is None, "处理线程结束")
            if state.get("error") or modal_errors:
                raise RuntimeError(state.get("error") or modal_errors[-1])
            if "snapshot" not in state or state["snapshot"]["status"] != "succeeded":
                raise RuntimeError(f"项目没有成功完成：{state}")
            store = window.project_panel.store
            song = window.project_panel.song()
            step = song.checkpoints[song.primary_key]
            if step.status != "succeeded":
                raise RuntimeError(step.error or "项目检查点没有成功完成")
            midi_path = store.verify(next(a for a in step.artifacts if a.kind == "midi"))
            midi = mido.MidiFile(midi_path)
            notes = sum(m.type == "note_on" and m.velocity > 0 for tr in midi.tracks for m in tr)
            if notes <= 0:
                raise RuntimeError("MIDI 没有有效音符")
            case.update(stage="result_audio", notes=notes, midi=str(midi_path))
            report["stage"] = "result_audio"
            save()
            editor = window.muscriptor_result_widget
            if editor is None:
                raise RuntimeError("真实主窗口没有生成结果编辑器")
            wait_for_gui(lambda: editor._asset_worker is None, "结果音频生成")
            if editor._assets is None or not editor._playback_engine.is_configured:
                raise RuntimeError("结果音频未就绪")
            if not editor._assets.live_transcription_wav.is_file():
                raise RuntimeError("真实试听 WAV 不存在")
            # This explicit offscreen diagnostic must not disturb desktop audio.
            # The actual QAudioSink still starts and consumes the real PCM stream.
            editor._playback_engine._sink.setVolume(0.0)
            editor.play_button.click()
            until = time.monotonic() + 0.15
            wait_for_gui(lambda: time.monotonic() >= until, "实际播放", 5)
            if not editor._playing:
                raise RuntimeError("真实音频播放未保持运行")
            editor.pause()
            if editor._playing:
                raise RuntimeError("暂停没有停止播放")
            if not window.isVisible() or window.worker is not None:
                raise RuntimeError("结果完成后的主窗口或线程状态异常")
            case.update(
                stage="complete",
                ok=True,
                seconds=time.monotonic() - started,
                regex=check_regex_identity(),
                playback=True,
            )
            save()
        report["stage"] = "closing"
        save()
    finally:
        window.close()
        wait_for_gui(
            lambda: not window.isVisible()
            and window.worker is None
            and getattr(window, "_gpu_detector", None) is None,
            "主窗口及后台线程关闭",
            120,
        )
        modal_watch.stop()
        app.setQuitOnLastWindowClosed(old_quit)
        report["closed"] = True
    if modal_errors:
        raise RuntimeError(modal_errors[-1])


def run_gui_conversion_probe(argv):
    args = parse_probe_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    # Must be selected before QApplication/QtMultimedia initialization.
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    report = {
        "ok": False,
        "frozen": bool(getattr(sys, "frozen", False)),
        "pid": os.getpid(),
        "cases": [],
        "stage": "startup",
    }

    def save():
        (args.output / "gui-probe.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save()
    try:
        from src.main import _prepare_torch_runtime_before_pyqt
        from src.utils.crash_diagnostics import install_native_fault_log
        from src.utils.logger import setup_logger

        setup_logger(name="src", log_dir=str(args.output / "logs"), level=logging.DEBUG)
        install_native_fault_log(args.output / "logs")
        _prepare_torch_runtime_before_pyqt()
        execute_probe(args, report, save)
        report.update(ok=True, stage="complete")
        return 0
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}: {exc}", stage="failed")
        logging.getLogger(__name__).exception("真实 GUI 转换链诊断失败")
        return 1
    finally:
        save()
