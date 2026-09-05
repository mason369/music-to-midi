from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cli.app import MANIFEST_NAME, main
from src.models.muscriptor_instruments import MUSCRIPTOR_INSTRUMENTS


class FakeEngine:
    def __init__(
        self,
        *,
        fail_names: set[str] | None = None,
        cancel: bool = False,
        stdout_noise: bool = False,
        mutate_source: bool = False,
        mutate_source_preserve_metadata: bool = False,
        quality_warnings: list[str] | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.fail_names = fail_names or set()
        self.cancel = cancel
        self.stdout_noise = stdout_noise
        self.mutate_source = mutate_source
        self.mutate_source_preserve_metadata = mutate_source_preserve_metadata
        self.quality_warnings = list(quality_warnings or [])

    def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.stdout_noise:
            print("third-party model diagnostic")
        processor = SimpleNamespace(cancel=lambda: None)
        kwargs["processor_callback"](processor)
        kwargs["progress_callback"](
            SimpleNamespace(
                stage=SimpleNamespace(value="transcription"),
                overall_progress=0.5,
                message="working",
            )
        )
        if self.cancel:
            raise InterruptedError("test cancellation")
        source = Path(kwargs["source_path"])
        if source.name in self.fail_names:
            raise RuntimeError(f"intentional failure: {source.name}")
        artifact_path = Path(kwargs["output_dir"]) / f"{source.stem}.mid"
        artifact_path.write_bytes(b"MThd-fake-midi")
        if self.mutate_source_preserve_metadata:
            original_stat = source.stat()
            source.write_bytes(b"x" * original_stat.st_size)
            os.utime(
                source,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
        elif self.mutate_source:
            source.write_bytes(b"changed while running")
        artifact = SimpleNamespace(
            id="midi",
            kind="midi",
            path=artifact_path,
            media_type="audio/midi",
            track_id=kwargs.get("track_id"),
        )
        return SimpleNamespace(
            result={
                "mode": kwargs["options"].get("processing_mode", "manual"),
                "quality_warnings": self.quality_warnings,
            },
            artifacts=(artifact,),
        )


def _audio(path: Path, payload: bytes = b"audio") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _run(arguments: list[str], engine: FakeEngine):
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        arguments,
        engine_factory=lambda: engine,
        stdout=stdout,
        stderr=stderr,
        install_signal_handlers=False,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def _summary(output_root: Path) -> dict:
    paths = list(output_root.glob("batch-run-*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_direct_path_shortcut_normalizes_product_options_and_writes_hash_manifest(tmp_path):
    source = _audio(tmp_path / "song.wav")
    output = tmp_path / "out"
    engine = FakeEngine()

    code, stdout, stderr = _run(
        [
            str(source),
            "--output",
            str(output),
            "--mode",
            "piano-transkun",
            "--tempo-mode",
            "fixed-manual",
            "--bpm",
            "123.5",
            "--quantize",
        ],
        engine,
    )

    assert code == 0
    assert stderr == ""
    assert "完成" in stdout
    assert len(engine.calls) == 1
    options = engine.calls[0]["options"]
    assert options["processing_mode"] == "piano_transkun"
    assert options["tempo_mode"] == "fixed_manual"
    assert options["custom_bpm"] == 123.5
    assert options["quantize_notes"] is True
    assert options["quantize_grid"] == "1/32"

    manifest = json.loads((output / "song" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["source"]["sha256"]
    assert manifest["artifacts"][0]["sha256"]
    assert (output / "song" / manifest["artifacts"][0]["path"]).is_file()
    assert _summary(output)["counts"] == {
        "total": 1,
        "succeeded": 1,
        "skipped": 0,
        "failed": 0,
        "cancelled": 0,
        "not_run": 0,
    }


def test_recursive_batch_deduplicates_inputs_excludes_output_and_isolates_same_stems(tmp_path):
    source_root = tmp_path / "library"
    first = _audio(source_root / "album-a" / "song.wav", b"identical")
    second = _audio(source_root / "album-b" / "song.wav", b"identical")
    output = source_root / "cli-output"
    _audio(output / "old-output.wav", b"must not be ingested")
    engine = FakeEngine()

    code, _, stderr = _run(
        [
            "batch",
            str(source_root),
            str(first),
            "--recursive",
            "--output",
            str(output),
        ],
        engine,
    )

    assert code == 0
    assert stderr == ""
    assert [Path(call["source_path"]) for call in engine.calls] == [
        first.resolve(),
        second.resolve(),
    ]
    assert (output / "song" / MANIFEST_NAME).is_file()
    assert (output / "song-2" / MANIFEST_NAME).is_file()
    assert _summary(output)["counts"]["total"] == 2


def test_verified_resume_skips_but_corrupt_artifact_forces_preserved_rerun(tmp_path):
    source = _audio(tmp_path / "song.wav")
    output = tmp_path / "out"
    engine = FakeEngine()

    first_code, _, _ = _run(["convert", str(source), "-o", str(output)], engine)
    second_code, second_stdout, _ = _run(["convert", str(source), "-o", str(output)], engine)

    assert first_code == second_code == 0
    assert len(engine.calls) == 1
    assert "跳过" in second_stdout
    first_artifact = output / "song" / "song.mid"
    first_artifact.write_bytes(b"corrupt")

    third_stdout = io.StringIO()
    third_stderr = io.StringIO()
    third_code = main(
        ["convert", str(source), "-o", str(output)],
        engine_factory=lambda: engine,
        stdout=third_stdout,
        stderr=third_stderr,
        install_signal_handlers=False,
    )

    assert third_code == 0
    assert len(engine.calls) == 2
    assert "无法通过校验" in third_stderr.getvalue()
    assert first_artifact.read_bytes() == b"corrupt"
    assert (output / "song-2" / MANIFEST_NAME).is_file()


def test_batch_failure_is_explicit_continues_and_returns_nonzero(tmp_path):
    bad = _audio(tmp_path / "bad.wav")
    good = _audio(tmp_path / "good.wav")
    output = tmp_path / "out"
    engine = FakeEngine(fail_names={bad.name})

    code, _, stderr = _run(
        ["batch", str(bad), str(good), "--output", str(output)],
        engine,
    )

    assert code == 1
    assert len(engine.calls) == 2
    assert "intentional failure: bad.wav" in stderr
    summary = _summary(output)
    assert summary["status"] == "failed"
    assert summary["counts"]["failed"] == 1
    assert summary["counts"]["succeeded"] == 1
    failed_manifest = json.loads((output / "bad" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["error"]["type"] == "RuntimeError"


def test_fail_fast_reports_unprocessed_count(tmp_path):
    bad = _audio(tmp_path / "bad.wav")
    good = _audio(tmp_path / "good.wav")
    output = tmp_path / "out"
    engine = FakeEngine(fail_names={bad.name})

    code, _, _ = _run(
        ["batch", str(bad), str(good), "--output", str(output), "--fail-fast"],
        engine,
    )

    assert code == 1
    assert len(engine.calls) == 1
    assert _summary(output)["counts"]["not_run"] == 1


def test_manual_route_passes_explicit_route_track_and_tempo_source(tmp_path):
    track = _audio(tmp_path / "vocals.wav")
    original = _audio(tmp_path / "original.flac", b"original")
    output = tmp_path / "out"
    engine = FakeEngine()

    code, _, stderr = _run(
        [
            "track-to-midi",
            str(track),
            "--route",
            "yourmt3:yptf_moe_multi_nops",
            "--tempo-source",
            str(original),
            "--output",
            str(output),
        ],
        engine,
    )

    assert code == 0
    assert stderr == ""
    call = engine.calls[0]
    assert call["kind"] == "manual_midi"
    assert call["options"]["route"] == "yourmt3:yptf_moe_multi_nops"
    assert call["track_id"] == "vocals"
    assert call["tempo_source_path"] == original.resolve()


def test_json_mode_outputs_only_valid_json_lines(tmp_path):
    source = _audio(tmp_path / "song.wav")
    engine = FakeEngine(stdout_noise=True)

    code, stdout, stderr = _run(
        ["convert", str(source), "--output", str(tmp_path / "out"), "--json"],
        engine,
    )

    assert code == 0
    assert stderr.strip() == "third-party model diagnostic"
    events = [json.loads(line) for line in stdout.splitlines()]
    assert [event["event"] for event in events] == [
        "started",
        "progress",
        "succeeded",
        "summary",
        "summary_path",
    ]


def test_quiet_mode_hides_success_diagnostics_but_keeps_summary(tmp_path):
    source = _audio(tmp_path / "song.wav")

    class NoisySuccessfulEngine(FakeEngine):
        def run(self, **kwargs):
            print("third-party stdout progress")
            print("third-party stderr progress", file=sys.stderr)
            return super().run(**kwargs)

    code, stdout, stderr = _run(
        ["convert", str(source), "--output", str(tmp_path / "out"), "--quiet"],
        NoisySuccessfulEngine(),
    )

    assert code == 0
    assert "third-party" not in stdout
    assert "third-party" not in stderr
    assert stdout.strip() == "汇总：总数 1，成功 1，跳过 0，失败 0，取消 0，未执行 0"


def test_quiet_mode_replays_third_party_diagnostics_on_failure(tmp_path):
    source = _audio(tmp_path / "song.wav")

    class NoisyFailingEngine(FakeEngine):
        def run(self, **kwargs):
            print("third-party failure context")
            raise RuntimeError("intentional quiet failure")

    code, stdout, stderr = _run(
        ["convert", str(source), "--output", str(tmp_path / "out"), "--quiet"],
        NoisyFailingEngine(),
    )

    assert code == 1
    assert "third-party failure context" in stderr
    assert "intentional quiet failure" in stderr
    assert stdout.strip() == "汇总：总数 1，成功 0，跳过 0，失败 1，取消 0，未执行 0"


def test_cancellation_records_terminal_state_and_exit_130(tmp_path):
    source = _audio(tmp_path / "song.wav")
    output = tmp_path / "out"
    engine = FakeEngine(cancel=True)

    code, _, stderr = _run(["convert", str(source), "--output", str(output)], engine)

    assert code == 130
    assert "已取消" in stderr
    manifest = json.loads((output / "song" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    summary = _summary(output)
    assert summary["status"] == "cancelled"
    assert summary["counts"]["cancelled"] == 1


def test_dry_run_does_not_create_output_or_load_engine(tmp_path):
    source = _audio(tmp_path / "song.wav")
    output = tmp_path / "out"
    engine = FakeEngine()

    code, stdout, stderr = _run(
        ["convert", str(source), "--output", str(output), "--dry-run"],
        engine,
    )

    assert code == 0
    assert stderr == ""
    assert "未加载模型" in stdout
    assert not output.exists()
    assert engine.calls == []


def test_input_errors_are_not_silently_accepted(tmp_path):
    source = _audio(tmp_path / "notes.txt")
    engine = FakeEngine()

    code, _, stderr = _run(["convert", str(source)], engine)

    assert code == 2
    assert "unsupported audio file" in stderr
    assert engine.calls == []


def test_empty_audio_and_invalid_bpm_fail_before_model_loading(tmp_path):
    source = _audio(tmp_path / "empty.wav", b"")
    engine = FakeEngine()

    empty_code, _, empty_stderr = _run(["convert", str(source)], engine)
    bpm_code, _, bpm_stderr = _run(
        ["convert", str(_audio(tmp_path / "song.wav")), "--bpm", "500"], engine
    )

    assert empty_code == 2
    assert "empty" in empty_stderr
    assert bpm_code == 2
    assert "300" in bpm_stderr
    assert engine.calls == []


@pytest.mark.parametrize(
    "arguments, expected",
    [
        (["--mode", "six-stem-split", "--quantize"], "produces WAV tracks only"),
        (["--mode", "piano-transkun", "--backend", "miros"], "does not use"),
        (["--backend", "miros", "--yourmt3-model", "ymt3-plus"], "only valid"),
        (["--instrument", "voice"], "require --backend muscriptor"),
    ],
)
def test_irrelevant_options_fail_instead_of_being_silently_ignored(tmp_path, arguments, expected):
    source = _audio(tmp_path / "song.wav")
    engine = FakeEngine()

    code, _, stderr = _run(["convert", str(source), *arguments], engine)

    assert code == 2
    assert expected in stderr
    assert engine.calls == []


def test_manual_non_muscriptor_route_rejects_muscriptor_constraints(tmp_path):
    source = _audio(tmp_path / "vocals.wav")
    engine = FakeEngine()

    code, _, stderr = _run(
        [
            "track-to-midi",
            str(source),
            "--route",
            "miros",
            "--instrument",
            "voice",
        ],
        engine,
    )

    assert code == 2
    assert "only valid with a MuScriptor route" in stderr
    assert engine.calls == []


def test_source_change_during_inference_is_failed_not_published(tmp_path):
    source = _audio(tmp_path / "song.wav")
    output = tmp_path / "out"
    engine = FakeEngine(mutate_source=True)

    code, _, stderr = _run(["convert", str(source), "--output", str(output)], engine)

    assert code == 1
    assert "changed while conversion was running" in stderr
    manifest = json.loads((output / "song" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


def test_source_hash_change_is_detected_even_when_size_and_mtime_are_preserved(tmp_path):
    source = _audio(tmp_path / "song.wav")
    output = tmp_path / "out"
    engine = FakeEngine(mutate_source_preserve_metadata=True)

    code, _, stderr = _run(["convert", str(source), "--output", str(output)], engine)

    assert code == 1
    assert "changed while conversion was running" in stderr
    manifest = json.loads((output / "song" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


def test_runtime_preflight_failure_is_terminal_and_does_not_block_later_items(
    tmp_path, monkeypatch
):
    import src.cli.app as cli_app

    bad = _audio(tmp_path / "bad.wav")
    good = _audio(tmp_path / "good.wav")
    output = tmp_path / "out"
    engine = FakeEngine()
    original_source_identity = cli_app._source_identity

    def source_identity(path: Path) -> dict:
        if path == bad.resolve():
            raise OSError("input disappeared after discovery")
        return original_source_identity(path)

    monkeypatch.setattr(cli_app, "_source_identity", source_identity)
    code, _, stderr = _run(
        ["batch", str(bad), str(good), "--output", str(output)],
        engine,
    )

    assert code == 1
    assert "input disappeared after discovery" in stderr
    assert [Path(call["source_path"]) for call in engine.calls] == [good.resolve()]
    summary = _summary(output)
    assert summary["status"] == "failed"
    assert summary["counts"]["failed"] == 1
    assert summary["counts"]["succeeded"] == 1
    assert summary["counts"]["not_run"] == 0
    assert summary["items"][0]["status"] == "failed"
    assert "manifest" not in summary["items"][0]


def test_empty_midi_quality_warning_is_visible_without_changing_success_status(tmp_path):
    source = _audio(tmp_path / "silence.wav")
    output = tmp_path / "out"
    engine = FakeEngine(quality_warnings=["empty_midi"])

    code, stdout, stderr = _run(
        ["convert", str(source), "--output", str(output)],
        engine,
    )

    assert code == 0
    assert "完成" in stdout
    assert "MIDI 不含音符" in stderr
    manifest = json.loads((output / "silence" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["result"]["quality_warnings"] == ["empty_midi"]

    resumed_code, resumed_stdout, resumed_stderr = _run(
        ["convert", str(source), "--output", str(output)],
        engine,
    )

    assert resumed_code == 0
    assert "跳过" in resumed_stdout
    assert "MIDI 不含音符" in resumed_stderr
    assert len(engine.calls) == 1


def test_routes_json_lists_exact_shared_contract():
    stdout = io.StringIO()
    code = main(["routes", "--json"], stdout=stdout, stderr=io.StringIO())

    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["count"] == 13
    assert len(payload["routes"]) == 13
    assert "muscriptor:small" in payload["routes"]
    assert "piano_bytedance_pedal" in payload["routes"]


def test_all_public_model_and_instrument_choices_are_parser_visible(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["convert", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "piano_bytedance_pedal" in help_text
    assert "yptf_moe_multi_nops" in help_text
    assert "muscriptor" in help_text
    assert len(MUSCRIPTOR_INSTRUMENTS) == 35


def test_src_main_cli_switch_dispatches_before_gui(monkeypatch):
    import src.cli.app as cli_app
    import src.main as main_module

    received: list[str] = []
    monkeypatch.setattr(main_module.sys, "argv", ["src.main", "--cli", "routes", "--json"])
    monkeypatch.setattr(main_module, "setup_chinese_environment", lambda: None)
    monkeypatch.setattr(cli_app, "main", lambda arguments: received.extend(arguments) or 17)

    with pytest.raises(SystemExit) as exit_info:
        main_module.main()

    assert exit_info.value.code == 17
    assert received == ["routes", "--json"]


def test_frozen_cli_executable_name_is_a_first_class_entry(monkeypatch, tmp_path):
    import src.main as main_module

    monkeypatch.setattr(main_module.sys, "argv", ["MusicToMidiCLIXpu.exe", "routes"])
    monkeypatch.setattr(main_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(main_module.sys, "executable", str(tmp_path / "MusicToMidiCLIXpu.exe"))

    assert main_module._is_cli_runtime() is True
