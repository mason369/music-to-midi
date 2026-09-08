"""Project commands for source, portable and Docker CLI entry points."""

from __future__ import annotations

import json
import signal
from pathlib import Path

from src.projects import ProjectRunner, ProjectStore, Workflow
from src.projects.store import SPLIT_STEMS
from src.web_api.schemas import InferenceOptions, ManualMidiOptions


def add_parser(subparsers):
    parser = subparsers.add_parser("project", help="创建、配置、打开和继续持久化项目")
    parser.add_argument(
        "action", choices=("create", "add", "configure", "status", "run", "export", "import")
    )
    parser.add_argument("project_path", type=Path, help="项目目录或 music-to-midi-project.json")
    parser.add_argument(
        "inputs", nargs="*", help="create/add 的输入音频；export/import 的项目包路径"
    )
    parser.add_argument("--profile", type=Path, help="工作流 JSON 预设")
    parser.add_argument("--save-profile", type=Path, help="保存当前工作流为 JSON 预设")
    parser.add_argument("--mode", help="处理模式，例如 six-stem-split")
    parser.add_argument("--backend", help="SMART 后端：yourmt3、miros、muscriptor")
    parser.add_argument("--yourmt3-model")
    parser.add_argument("--muscriptor-model", choices=("large", "medium", "small"))
    parser.add_argument(
        "--stem",
        action="append",
        default=[],
        metavar="STEM=ROUTE",
        help="明确选择转写声部及模型，可重复；例如 piano=piano_transkun",
    )
    parser.add_argument("--clear-stems", action="store_true", help="清空逐声部计划，仅执行分离")
    parser.add_argument("--song", action="append", help="仅处理/配置指定歌曲 ID，可重复")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--language", default="zh_CN", choices=("zh_CN", "en_US"))
    return parser


def workflow_from_arguments(namespace, base: Workflow) -> Workflow:
    workflow = Workflow.load(namespace.profile) if namespace.profile else base.model_copy(deep=True)
    payload = workflow.model_dump(mode="json")
    for flag, key in (
        ("mode", "processing_mode"),
        ("backend", "transcription_backend"),
        ("yourmt3_model", "yourmt3_model"),
        ("muscriptor_model", "muscriptor_model"),
    ):
        value = getattr(namespace, flag)
        if value is not None:
            payload["primary"][key] = value.replace("-", "_")
    if namespace.clear_stems:
        payload["stems"] = {}
    for selection in namespace.stem:
        stem, separator, route = selection.partition("=")
        if not separator or not stem or not route:
            raise ValueError("--stem 必须为 声部=模型路线")
        previous = payload["stems"].get(stem, {})
        payload["stems"][stem] = ManualMidiOptions.model_validate(
            {**previous, "route": route}
        ).model_dump(mode="json")
    return Workflow.model_validate(payload)


def run_command(namespace, *, engine_factory, stdout, install_signal_handlers=True):
    store = ProjectStore(namespace.project_path)
    action = namespace.action
    if action == "import":
        if len(namespace.inputs) != 1:
            raise ValueError("import 需要一个 .mtmproject 项目包")
        store = ProjectStore.import_archive(namespace.inputs[0], namespace.project_path)
    elif action == "create":
        workflow = workflow_from_arguments(namespace, Workflow())
        store.create(namespace.project_path.stem, workflow)
        if namespace.inputs:
            store.add(namespace.inputs)
    elif action == "add":
        if not namespace.inputs:
            raise ValueError("add 需要输入音频")
        store.add(namespace.inputs, workflow_from_arguments(namespace, store.load().workflow))
    elif action == "configure":
        store.configure(
            workflow_from_arguments(namespace, store.load().workflow),
            namespace.song,
            clear_custom_stems=namespace.clear_stems,
        )
    elif action == "export":
        if len(namespace.inputs) != 1:
            raise ValueError("export 需要一个 .mtmproject 输出路径")
        store.export_archive(namespace.inputs[0])
    elif action == "run":
        if (
            namespace.profile
            or namespace.mode
            or namespace.stem
            or namespace.clear_stems
            or namespace.backend
            or namespace.yourmt3_model
            or namespace.muscriptor_model
        ):
            raise ValueError("先通过 project configure 保存工作流，再显式运行项目")

        def emit(snapshot):
            if namespace.json:
                print(
                    json.dumps({"event": "project", **snapshot}, ensure_ascii=False),
                    file=stdout,
                    flush=True,
                )
            else:
                progress = snapshot["progress"]
                print(
                    f"[{progress.get('index', 0)}/{progress.get('total', len(snapshot['songs']))}] "
                    f"{progress.get('percent', 0)}% {progress.get('message', '')}",
                    file=stdout,
                    flush=True,
                )

        runner = ProjectRunner(store, engine_factory=engine_factory, on_update=emit)
        previous = {}
        try:
            if install_signal_handlers:
                for number in (signal.SIGINT, signal.SIGTERM):
                    previous[number] = signal.getsignal(number)
                    signal.signal(number, lambda *_args: runner.cancel())
            result = runner.run(namespace.song, fail_fast=namespace.fail_fast)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
        if result["status"] == "cancelled":
            return 130
        if result["status"] == "failed":
            return 1
    if namespace.save_profile:
        store.load().workflow.save(namespace.save_profile)
    print(
        json.dumps(store.snapshot(), ensure_ascii=False, indent=None if namespace.json else 2),
        file=stdout,
        flush=True,
    )
    return 0
