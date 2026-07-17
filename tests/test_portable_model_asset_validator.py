from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from tools import validate_portable_model_assets as validator


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(root: Path, relative_path: Path | str, data: bytes) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _patch_identity(monkeypatch, module, prefix: str, data: bytes) -> None:
    monkeypatch.setattr(module, f"{prefix}_SIZE", len(data))
    monkeypatch.setattr(module, f"{prefix}_SHA256", _sha256(data))


def _prepare_valid_assets(tmp_path: Path, monkeypatch) -> dict[str, object]:
    audio_separator_dir = tmp_path / "audio-separator"
    yourmt3_dir = tmp_path / "yourmt3_all"
    yourmt3_source_dir = tmp_path / "YourMT3" / "amt" / "src"
    aria_amt_dir = tmp_path / "aria_amt"
    bytedance_piano_dir = tmp_path / "bytedance_piano"
    miros_dir = tmp_path / "ai4m-miros"
    files: dict[str, Path] = {}

    leap_checkpoint = b"pinned-leap-checkpoint"
    leap_config = b"pinned-leap-config"
    files["leap_checkpoint"] = _write(
        audio_separator_dir, validator.leap.LEAP_CHECKPOINT_NAME, leap_checkpoint
    )
    files["leap_config"] = _write(audio_separator_dir, validator.leap.LEAP_CONFIG_NAME, leap_config)
    _patch_identity(monkeypatch, validator.leap, "LEAP_CHECKPOINT", leap_checkpoint)
    _patch_identity(monkeypatch, validator.leap, "LEAP_CONFIG", leap_config)

    polar_checkpoint = b"pinned-polarformer-onnx"
    polar_config = b"pinned-polarformer-config"
    files["polar_checkpoint"] = _write(
        audio_separator_dir,
        validator.polarformer.POLARFORMER_ONNX_NAME,
        polar_checkpoint,
    )
    files["polar_config"] = _write(
        audio_separator_dir,
        validator.polarformer.POLARFORMER_CONFIG_NAME,
        polar_config,
    )
    _patch_identity(monkeypatch, validator.polarformer, "POLARFORMER_ONNX", polar_checkpoint)
    _patch_identity(monkeypatch, validator.polarformer, "POLARFORMER_CONFIG", polar_config)

    yourmt3_keys = tuple(validator.yourmt3.OFFICIAL_YOURMT3_MODEL_KEYS)
    assert len(yourmt3_keys) == 5
    yourmt3_identities = {}
    for index, model_key in enumerate(yourmt3_keys):
        payload = f"pinned-yourmt3-{model_key}".encode("utf-8")
        identity_name = "last.ckpt" if index == 3 else "model.ckpt"
        relative_path = Path("amt") / "logs" / model_key / "checkpoints" / identity_name
        staged_path = relative_path.with_name("model.ckpt")
        files[f"yourmt3_{index}"] = _write(yourmt3_dir, staged_path, payload)
        yourmt3_identities[model_key] = {
            "filename": relative_path.as_posix(),
            "size": len(payload),
            "sha256": _sha256(payload),
        }
    monkeypatch.setattr(validator.yourmt3, "YOURMT3_MODEL_IDENTITIES", yourmt3_identities)

    files["yourmt3_source"] = _write(
        yourmt3_source_dir, "inference.py", b"PINNED_IMPLEMENTATION = True\n"
    )
    _write(yourmt3_source_dir, "config.py", b"MODEL_FAMILY = 'YourMT3'\n")
    source_manifest, source_file_count = validator.yourmt3_source.calculate_yourmt3_source_manifest(
        yourmt3_source_dir
    )
    monkeypatch.setattr(
        validator.yourmt3_source,
        "PATCHED_YOURMT3_MANIFEST_SHA256",
        source_manifest,
    )
    monkeypatch.setattr(
        validator.yourmt3_source,
        "PATCHED_YOURMT3_MANIFEST_FILE_COUNT",
        source_file_count,
    )

    aria_checkpoint = b"pinned-aria-checkpoint"
    files["aria"] = _write(
        aria_amt_dir, validator.aria_amt.ARIA_AMT_CHECKPOINT_NAME, aria_checkpoint
    )
    _patch_identity(monkeypatch, validator.aria_amt, "ARIA_AMT_CHECKPOINT", aria_checkpoint)

    bytedance_checkpoint = b"pinned-bytedance-checkpoint"
    files["bytedance"] = _write(
        bytedance_piano_dir,
        validator.bytedance_piano.BYTEDANCE_PIANO_CHECKPOINT_NAME,
        bytedance_checkpoint,
    )
    _patch_identity(
        monkeypatch,
        validator.bytedance_piano,
        "BYTEDANCE_PIANO_CHECKPOINT",
        bytedance_checkpoint,
    )

    files["miros_source"] = _write(miros_dir, "main.py", b"print('pinned miros')\n")
    _write(miros_dir, "transcribe.py", b"PINNED = True\n")
    miros_pretrained = b"pinned-miros-pretrained"
    miros_finetuned = b"pinned-miros-finetuned"
    files["miros_pretrained"] = _write(
        miros_dir,
        validator.miros.MirosTranscriber.PRETRAINED_REL_PATH,
        miros_pretrained,
    )
    files["miros_finetuned"] = _write(
        miros_dir,
        validator.miros.MirosTranscriber.CHECKPOINT_REL_PATH,
        miros_finetuned,
    )
    monkeypatch.setattr(validator.miros, "MIROS_PRETRAINED_EXACT_BYTES", len(miros_pretrained))
    monkeypatch.setattr(validator.miros, "MIROS_PRETRAINED_SHA256", _sha256(miros_pretrained))
    monkeypatch.setattr(validator.miros, "MIROS_FINETUNED_EXACT_BYTES", len(miros_finetuned))
    monkeypatch.setattr(validator.miros, "MIROS_FINETUNED_SHA256", _sha256(miros_finetuned))
    monkeypatch.setattr(
        validator.miros,
        "MIROS_PATCHED_SOURCE_SHA256",
        validator.miros.compute_miros_source_tree_sha256(miros_dir),
    )

    return {
        "audio_separator_dir": audio_separator_dir,
        "yourmt3_dir": yourmt3_dir,
        "yourmt3_source_dir": yourmt3_source_dir,
        "aria_amt_dir": aria_amt_dir,
        "bytedance_piano_dir": bytedance_piano_dir,
        "miros_dir": miros_dir,
        "files": files,
    }


def _validate(layout: dict[str, object]) -> dict[str, tuple[Path, ...]]:
    return validator.validate_portable_model_assets(
        audio_separator_dir=layout["audio_separator_dir"],
        yourmt3_dir=layout["yourmt3_dir"],
        yourmt3_source_dir=layout["yourmt3_source_dir"],
        aria_amt_dir=layout["aria_amt_dir"],
        bytedance_piano_dir=layout["bytedance_piano_dir"],
        miros_dir=layout["miros_dir"],
    )


def _mutate_without_changing_size(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    assert payload
    payload[0] ^= 0xFF
    path.write_bytes(payload)


def test_validator_accepts_every_pinned_portable_asset(tmp_path, monkeypatch):
    layout = _prepare_valid_assets(tmp_path, monkeypatch)

    validated = _validate(layout)

    assert set(validated) == {
        "leap_xe",
        "polarformer",
        "yourmt3",
        "yourmt3_source",
        "aria_amt",
        "bytedance_piano",
        "miros",
    }
    assert len(validated["leap_xe"]) == 2
    assert len(validated["polarformer"]) == 2
    assert len(validated["yourmt3"]) == 5
    assert len(validated["yourmt3_source"]) == 1
    assert len(validated["aria_amt"]) == 1
    assert len(validated["bytedance_piano"]) == 1
    assert len(validated["miros"]) == 4


@pytest.mark.parametrize(
    ("asset_key", "expected_error"),
    [
        ("leap_checkpoint", "Leap XE checkpoint SHA-256 mismatch"),
        ("leap_config", "Leap XE config SHA-256 mismatch"),
        ("polar_checkpoint", "PolarFormer ONNX checkpoint SHA-256 mismatch"),
        ("polar_config", "PolarFormer config SHA-256 mismatch"),
        ("yourmt3_0", "YourMT3 ymt3_plus checkpoint SHA-256 mismatch"),
        ("yourmt3_1", "YourMT3 yptf_single_nops checkpoint SHA-256 mismatch"),
        ("yourmt3_2", "YourMT3 yptf_multi_ps checkpoint SHA-256 mismatch"),
        ("yourmt3_3", "YourMT3 yptf_moe_multi_nops checkpoint SHA-256 mismatch"),
        ("yourmt3_4", "YourMT3 yptf_moe_multi_ps checkpoint SHA-256 mismatch"),
        ("yourmt3_source", "Patched YourMT3 source manifest mismatch"),
        ("aria", "Aria-AMT checkpoint SHA-256 mismatch"),
        ("bytedance", "ByteDance Piano checkpoint SHA-256 mismatch"),
        ("miros_source", "MIROS patched source tree SHA256 mismatch"),
        ("miros_pretrained", "MIROS MusicFM pretrained weight SHA256 mismatch"),
        ("miros_finetuned", "MIROS fine-tuned checkpoint SHA256 mismatch"),
    ],
)
def test_validator_rejects_same_size_wrong_identity(
    tmp_path, monkeypatch, asset_key, expected_error
):
    layout = _prepare_valid_assets(tmp_path, monkeypatch)
    files = layout["files"]
    _mutate_without_changing_size(files[asset_key])

    with pytest.raises(RuntimeError, match=re.escape(expected_error)):
        _validate(layout)


def test_cli_returns_nonzero_and_reports_identity_failure(tmp_path, monkeypatch, capsys):
    layout = _prepare_valid_assets(tmp_path, monkeypatch)
    _mutate_without_changing_size(layout["files"]["aria"])

    exit_code = validator.main(
        [
            "--audio-separator-dir",
            str(layout["audio_separator_dir"]),
            "--yourmt3-dir",
            str(layout["yourmt3_dir"]),
            "--yourmt3-source-dir",
            str(layout["yourmt3_source_dir"]),
            "--aria-amt-dir",
            str(layout["aria_amt_dir"]),
            "--bytedance-piano-dir",
            str(layout["bytedance_piano_dir"]),
            "--miros-dir",
            str(layout["miros_dir"]),
            "--label",
            "test bundle",
        ]
    )

    assert exit_code == 1
    assert "[error] test bundle identity validation failed" in capsys.readouterr().err


def test_validator_rejects_missing_required_asset(tmp_path, monkeypatch):
    layout = _prepare_valid_assets(tmp_path, monkeypatch)
    layout["files"]["leap_config"].unlink()

    with pytest.raises(RuntimeError, match="Leap XE config is missing"):
        _validate(layout)


def test_validator_rejects_truncated_required_asset(tmp_path, monkeypatch):
    layout = _prepare_valid_assets(tmp_path, monkeypatch)
    checkpoint = layout["files"]["bytedance"]
    checkpoint.write_bytes(checkpoint.read_bytes()[:-1])

    with pytest.raises(RuntimeError, match="ByteDance Piano checkpoint size mismatch"):
        _validate(layout)


def test_validator_rejects_invalid_duplicate_yourmt3_checkpoint_alias(tmp_path, monkeypatch):
    layout = _prepare_valid_assets(tmp_path, monkeypatch)
    valid_checkpoint = layout["files"]["yourmt3_3"]
    invalid_alias = valid_checkpoint.with_name("last.ckpt")
    invalid_alias.write_bytes(b"x" * valid_checkpoint.stat().st_size)

    with pytest.raises(
        RuntimeError,
        match="YourMT3 yptf_moe_multi_nops checkpoint SHA-256 mismatch",
    ):
        _validate(layout)


def _patch_valid_runtime_identities(monkeypatch):
    versions = {
        "audio-separator": validator.AUDIO_SEPARATOR_PACKAGE_VERSION,
        "onnxruntime-gpu": validator.ONNXRUNTIME_GPU_PACKAGE_VERSION,
    }
    monkeypatch.setattr(validator.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(validator.importlib, "import_module", lambda _name: object())
    monkeypatch.setattr(validator.aria_amt, "get_aria_amt_runtime_unavailable_reason", lambda: "")
    monkeypatch.setattr(
        validator.bytedance_piano.ByteDancePianoTranscriber,
        "get_unavailable_reason",
        staticmethod(lambda: ""),
    )
    monkeypatch.setattr(
        validator.transkun.TranskunTranscriber,
        "get_unavailable_reason",
        staticmethod(lambda: ""),
    )
    monkeypatch.setattr(
        validator.transkun.TranskunTranscriber,
        "is_model_available",
        lambda _self: True,
    )
    return versions


def test_runtime_validator_accepts_exact_pinned_packages_and_sources(monkeypatch):
    expected_versions = _patch_valid_runtime_identities(monkeypatch)

    identities = validator.validate_portable_runtime_identities()

    assert identities["audio-separator"] == expected_versions["audio-separator"]
    assert identities["onnxruntime-gpu"] == expected_versions["onnxruntime-gpu"]
    assert identities["aria-amt-source"] == validator.aria_amt.ARIA_AMT_SOURCE_REVISION
    assert identities["piano-transcription-inference"] == "0.0.6"
    assert identities["transkun"] == "2.0.1"


@pytest.mark.parametrize("package_name", ["audio-separator", "onnxruntime-gpu"])
def test_runtime_validator_rejects_wrong_distribution_version(monkeypatch, package_name):
    versions = _patch_valid_runtime_identities(monkeypatch)
    versions[package_name] = "0.0.0-wrong"

    with pytest.raises(RuntimeError, match=f"version mismatch for {package_name}"):
        validator.validate_portable_runtime_identities()


@pytest.mark.parametrize(
    ("runtime_name", "patch_target"),
    [
        ("Aria-AMT", "aria"),
        ("ByteDance Piano", "bytedance"),
        ("TransKun", "transkun"),
    ],
)
def test_runtime_validator_rejects_backend_source_or_package_mismatch(
    monkeypatch, runtime_name, patch_target
):
    _patch_valid_runtime_identities(monkeypatch)
    if patch_target == "aria":
        monkeypatch.setattr(
            validator.aria_amt,
            "get_aria_amt_runtime_unavailable_reason",
            lambda: "wrong direct_url revision",
        )
    elif patch_target == "bytedance":
        monkeypatch.setattr(
            validator.bytedance_piano.ByteDancePianoTranscriber,
            "get_unavailable_reason",
            staticmethod(lambda: "wrong package version"),
        )
    else:
        monkeypatch.setattr(
            validator.transkun.TranskunTranscriber,
            "get_unavailable_reason",
            staticmethod(lambda: "wrong package version"),
        )

    with pytest.raises(RuntimeError, match=f"{runtime_name} runtime identity"):
        validator.validate_portable_runtime_identities()


def test_runtime_validator_rejects_wrong_transkun_packaged_resources(monkeypatch):
    _patch_valid_runtime_identities(monkeypatch)
    monkeypatch.setattr(
        validator.transkun.TranskunTranscriber,
        "is_model_available",
        lambda _self: False,
    )

    with pytest.raises(RuntimeError, match="TransKun packaged V2 resources failed"):
        validator.validate_portable_runtime_identities()


def test_repository_yourmt3_source_matches_portable_pinned_manifest():
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "YourMT3" / "amt" / "src"

    manifest_sha256, file_count = validator.yourmt3_source.validate_patched_yourmt3_source(
        source_dir
    )

    assert manifest_sha256 == validator.yourmt3_source.PATCHED_YOURMT3_MANIFEST_SHA256
    assert file_count == validator.yourmt3_source.PATCHED_YOURMT3_MANIFEST_FILE_COUNT
    spec = (repo_root / "MusicToMidi.spec").read_text(encoding="utf-8")
    assert '"YourMT3", "amt", "src"' in spec


def test_build_portable_validates_source_and_staged_assets_before_packaging():
    script = (Path(__file__).resolve().parents[1] / "build_portable.ps1").read_text(
        encoding="utf-8"
    )

    source_check = script.index('-Label "portable source assets"')
    first_model_copy = script.index("Copy-Tree -Source $AudioSeparatorSource")
    staged_check = script.index('-Label "staged portable model assets"')
    last_model_copy = script.index("Copy-Tree -Source $MirosSource")
    pyinstaller = script.index("-m PyInstaller")

    assert source_check < first_model_copy
    assert last_model_copy < staged_check < pyinstaller
    assert script.count("tools\\validate_portable_model_assets.py") == 1
    assert "Assert-SixStemAssets -ModelDir $AudioSeparatorSource" in script
    assert "Assert-SixStemAssets -ModelDir $AudioSeparatorBundle" in script
    assert "validate_transkun_v2_aug_model_files" in script

    source_block = script[source_check - 500 : source_check + 100]
    staged_block = script[staged_check - 500 : staged_check + 100]
    for source_name in (
        "$AudioSeparatorSource",
        "$YourMt3Source",
        "$YourMt3CodeSource",
        "$AriaAmtSource",
        "$ByteDancePianoSource",
        "$MirosSource",
    ):
        assert source_name in source_block
    for bundle_name in (
        "$AudioSeparatorBundle",
        "$YourMt3Bundle",
        "$YourMt3CodeSource",
        "$AriaAmtBundle",
        "$ByteDancePianoBundle",
        "$MirosBundle",
    ):
        assert bundle_name in staged_block

    repo_root = Path(__file__).resolve().parents[1]
    requirements = (repo_root / "requirements.txt").read_text(encoding="utf-8")
    installer = (repo_root / "install.ps1").read_text(encoding="utf-8")
    release_workflow = (repo_root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "audio-separator==0.44.1" in installer
    assert "onnxruntime-gpu==1.23.2" in requirements
    assert "tests/test_portable_model_asset_validator.py" in release_workflow


def test_linux_release_validates_staged_assets_before_pyinstaller():
    release_workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/release.yml"
    ).read_text(encoding="utf-8")

    staged_copy = release_workflow.index(
        'cp -a "$GITHUB_WORKSPACE/external/ai4m-miros" ' '"$BUILD_ASSET_ROOT/ai4m-miros"'
    )
    staged_validation = release_workflow.index("python tools/validate_portable_model_assets.py")
    pyinstaller = release_workflow.index("pyinstaller MusicToMidi.spec")

    assert staged_copy < staged_validation < pyinstaller
    for argument in (
        '--audio-separator-dir "$BUILD_ASSET_ROOT/audio-separator"',
        '--yourmt3-dir "$BUILD_ASSET_ROOT/yourmt3_all"',
        '--yourmt3-source-dir "$GITHUB_WORKSPACE/YourMT3/amt/src"',
        '--aria-amt-dir "$BUILD_ASSET_ROOT/aria_amt"',
        '--bytedance-piano-dir "$BUILD_ASSET_ROOT/bytedance_piano"',
        '--miros-dir "$BUILD_ASSET_ROOT/ai4m-miros"',
    ):
        assert argument in release_workflow
    assert "download_multistem_model.py" in release_workflow[staged_validation:pyinstaller]
    assert "validate_transkun_v2_aug_model_files" in release_workflow[staged_validation:pyinstaller]
