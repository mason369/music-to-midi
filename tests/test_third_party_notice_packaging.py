from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_zfturbo_mit_notice_is_present_and_attributed():
    notice = _read("THIRD_PARTY_NOTICES.md")
    assert "ZFTurbo/Music-Source-Separation-Training" in notice
    assert "Roman Solovyev" in notice
    assert "MIT License" in notice
    assert "Permission is hereby granted" in notice


def test_notice_is_in_pyinstaller_space_and_portable_release_outputs():
    spec = _read("MusicToMidi.spec")
    portable = _read("build_portable.ps1")
    sync = _read(".github/workflows/sync_to_hf.yml")
    build = _read(".github/workflows/build.yml")
    release = _read(".github/workflows/release.yml")

    assert "('THIRD_PARTY_NOTICES.md', '.')" in spec
    assert '"THIRD_PARTY_NOTICES.md"' in portable
    assert "cp THIRD_PARTY_NOTICES.md" in sync
    assert "THIRD_PARTY_NOTICES.md" in release
    assert "portable artifacts are release-only" in build
    assert "actions/upload-artifact" not in build


def test_notice_covers_every_bundled_model_family_with_pinned_provenance():
    notice = _read("THIRD_PARTY_NOTICES.md")

    for expected in (
        "mimbres/YourMT3",
        "5e66c1ea173a8186e0d20432b841d3180cc015b5",
        "pcunwa/BS-Roformer-Leap",
        "bgkb/bs_polarformer",
        "noblebarkrr/mvsepless_resources",
        "TransKun default V2 and V2 Aug",
        "EleutherAI/aria-amt",
        "loubb/aria-midi",
        "piano-transcription-inference 0.0.6",
        "10.5281/zenodo.4034264",
        "amt-os/ai4m-miros",
        "minzwon/MusicFM",
        "audio-separator 0.44.1",
        "ONNX Runtime GPU 1.23.2",
        "PyTorch, torchaudio, and torchvision",
        "PyQt6",
        "FFmpeg/ffprobe",
    ):
        assert expected in notice


def test_machine_inventory_is_closed_over_every_current_portable_component():
    notice = _read("THIRD_PARTY_NOTICES.md")
    workflow = _read(".github/workflows/release.yml")
    rows = [
        line for line in notice.splitlines() if line.startswith("PORTABLE_COMPONENT: ")
    ]
    component_ids = {line.split(" |", 1)[0].split(": ", 1)[1] for line in rows}
    expected_ids = {
        "zfturbo_adapted_source",
        "yourmt3_checkpoints",
        "yourmt3_patched_source",
        "leap_xe",
        "polarformer",
        "bs_roformer_sw_fixed",
        "audio_separator",
        "transkun_source",
        "transkun_default_v2_weight",
        "transkun_v2_aug",
        "aria_amt_source",
        "aria_amt_checkpoint",
        "bytedance_inference",
        "bytedance_checkpoint",
        "miros_source",
        "musicfm_pretrained",
        "miros_finetuned",
        "pytorch_cuda_runtime",
        "onnxruntime_gpu",
        "pyqt6_runtime",
        "ffmpeg_runtime",
        "frozen_dependency_set",
    }

    assert component_ids == expected_ids
    assert len(rows) == len(component_ids)
    for row in rows:
        assert " | bundle=" in row
        assert " | artifact=" in row
        assert " | revision=" in row
        assert " | license=" in row
        assert row.endswith((" | status=VERIFIED", " | status=BLOCKED"))
    for component_id in expected_ids:
        assert component_id in workflow
    assert "Portable license inventory count mismatch" in workflow
    assert "contains an unreviewed component ID" in workflow
    assert "Malformed portable license inventory row" in workflow


def test_unverified_or_incompatible_redistribution_terms_block_release():
    notice = _read("THIRD_PARTY_NOTICES.md")
    workflow = _read(".github/workflows/release.yml")

    blockers = [
        line
        for line in notice.splitlines()
        if line.startswith("RELEASE_BLOCKER_UNRESOLVED_LICENSE:")
    ]
    assert len(blockers) >= 9
    assert "LEAP_XE_CHECKPOINT_AND_CONFIG" in notice
    assert "BS_ROFORMER_SW_FIXED_CHECKPOINT_AND_CONFIG" in notice
    assert "YOURMT3_PATCHED_SOURCE_PROVENANCE_AND_MIXED_LICENSES" in notice
    assert "TRANSKUN_DEFAULT_V2_WEIGHT" in notice
    assert "TRANSKUN_V2_AUG_GOOGLE_DRIVE_ARTIFACT" in notice
    assert "MIROS_SOURCE_AND_FINETUNED_CHECKPOINT" in notice
    assert "ARIA_AMT_CC_BY_NC_SA_CHECKPOINT_COMPLIANCE" in notice
    assert "BYTEDANCE_INFERENCE_SOURCE_LICENSE" in notice
    assert "PYQT6_CUDA_FFMPEG_AND_FROZEN_DEPENDENCY_COMPLIANCE" in notice
    assert 'grep -n "^RELEASE_BLOCKER_UNRESOLVED_LICENSE:"' in workflow
    assert "Portable release is forbidden until every marker is resolved" in workflow


def test_verified_license_claims_name_the_primary_upstream_declaration():
    notice = _read("THIRD_PARTY_NOTICES.md")

    assert "license: apache-2.0" in notice
    assert notice.count("license: mit") >= 2
    assert "CC BY-NC-SA 4.0" in notice
    assert "CC BY 4.0" in notice
    assert "publicly accessible does not by itself grant redistribution rights" in notice
