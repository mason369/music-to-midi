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
        assert row.endswith((" | status=VERIFIED", " | status=OWNER_ACCEPTED"))
    for component_id in expected_ids:
        assert component_id in workflow
    assert "Portable license inventory count mismatch" in workflow
    assert "contains an unreviewed component ID" in workflow
    assert "Malformed portable license inventory row" in workflow


def test_release_gate_requires_verified_or_owner_accepted_components():
    notice = _read("THIRD_PARTY_NOTICES.md")
    workflow = _read(".github/workflows/release.yml")

    # The fail-closed markers are retired: every component is either VERIFIED
    # with declared license evidence or OWNER_ACCEPTED with a distribution record.
    assert "RELEASE_BLOCKER_UNRESOLVED_LICENSE:" not in notice

    rows = [
        line for line in notice.splitlines() if line.startswith("PORTABLE_COMPONENT: ")
    ]
    owner_accepted = {
        line.split(" |", 1)[0].split(": ", 1)[1]
        for line in rows
        if line.endswith(" | status=OWNER_ACCEPTED")
    }
    assert owner_accepted == {
        "leap_xe",
        "bs_roformer_sw_fixed",
        "miros_source",
        "miros_finetuned",
    }

    for component_id in sorted(owner_accepted):
        assert f"OWNER_ACCEPTED_NOTICE: {component_id}" in notice

    # Owner-accepted records keep full attribution and the takedown route.
    assert "undeclared upstream" in notice
    assert "issue tracker" in notice
    assert "No license" in notice or "no license file" in notice

    # Declared-license components carry their compliance records.
    assert "CC BY-NC-SA 4.0" in notice
    assert "GPL-3.0" in notice
    assert "NVIDIA CUDA" in notice
    assert "THIRD_PARTY_SBOM.txt" in notice
    assert "FFMPEG_BUILD_AUDIT.txt" in notice

    # The workflow enforces both statuses and the distribution-record check.
    assert "status=(VERIFIED|BLOCKED|OWNER_ACCEPTED)" in workflow
    assert '"| status=BLOCKED"' in workflow
    assert "OWNER_ACCEPTED_NOTICE: ${component_id}" in workflow
    assert 'grep -n "^RELEASE_BLOCKER_UNRESOLVED_LICENSE:"' in workflow
    assert "Portable release is forbidden until every marker is resolved" in workflow


def test_verified_license_claims_name_the_primary_upstream_declaration():
    notice = _read("THIRD_PARTY_NOTICES.md")

    assert "license: apache-2.0" in notice
    assert notice.count("license: mit") >= 2
    assert "CC BY-NC-SA 4.0" in notice
    assert "CC BY 4.0" in notice
    assert "publicly accessible does not by itself grant redistribution rights" in notice
