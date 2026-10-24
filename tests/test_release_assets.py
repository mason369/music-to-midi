"""Release failures must never expose incomplete or mixed-revision downloads."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tools import release_assets as release

TAG = "v1.10.0"
REVISION = "f" * 40
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    names = [
        "MusicToMidi-Windows-GPU-Portable.wim.001",
        "MusicToMidi-Windows-GPU-Portable.wim.002",
        "MusicToMidi-Windows-WebFrontend-Portable.zip",
        "MusicToMidi-Linux-GPU-Portable.tar.gz.partaa",
        "MusicToMidi-Linux-GPU-Portable.tar.gz.partab",
        "MusicToMidi-Linux-WebFrontend-Portable.tar.gz",
        f"MusicToMidi-Docker-{TAG}.zip",
        f"MusicToMidi-Docker-{TAG}.tar.gz",
    ]
    for name in names:
        (root / name).write_bytes(name.encode())
    for role in release.ROLES:
        release.record_assets(root, role, TAG, REVISION)
    notes = tmp_path / "notes.md"
    notes.write_text("中文发布说明 / English release notes", encoding="utf-8")
    return root, notes


def test_complete_bundle_publishes_only_downloads_and_checksums(bundle):
    root, _ = bundle
    records = release.verify_assets(root, TAG, REVISION)
    assert len(records) == 9
    assert records[-1]["name"] == "SHA256SUMS.txt"
    assert len((root / "SHA256SUMS.txt").read_text().splitlines()) == 8
    assert all("manifest" not in item["name"] for item in records)


@pytest.mark.parametrize(
    "role,old,new",
    [
        (
            "Windows",
            "MusicToMidi-Windows-GPU-Portable.wim.002",
            "MusicToMidi-Windows-GPU-Portable.wim.003",
        ),
        (
            "Linux",
            "MusicToMidi-Linux-GPU-Portable.tar.gz.partab",
            "MusicToMidi-Linux-GPU-Portable.tar.gz.partac",
        ),
    ],
)
def test_missing_middle_volume_is_rejected(bundle, role, old, new):
    root, _ = bundle
    (root / old).rename(root / new)
    with pytest.raises(ValueError, match="Missing or misnumbered"):
        release.role_assets(root, role, TAG)


@pytest.mark.parametrize("mutation", ["hash", "empty", "revision", "tag", "report", "extra", "cpu"])
def test_corruption_mixed_revision_and_reports_fail_before_upload(bundle, mutation):
    root, _ = bundle
    target = root / "MusicToMidi-Windows-GPU-Portable.wim.001"
    if mutation in {"hash", "empty"}:
        target.write_bytes(b"wrong" if mutation == "hash" else b"")
    elif mutation in {"revision", "tag"}:
        manifest = root / "release-manifest-Linux.json"
        payload = json.loads(manifest.read_text())
        payload[mutation] = "a" * 40 if mutation == "revision" else "v1.9.0"
        manifest.write_text(json.dumps(payload))
    elif mutation == "cpu":
        (root / "MusicToMidi-Windows-CPU-Portable.zip").write_text("unsupported bundle")
    else:
        (
            root
            / ("MusicToMidi-Docker-report.txt" if mutation == "report" else "test-results.json")
        ).write_text("private evidence")
    with pytest.raises(ValueError):
        release.verify_assets(root, TAG, REVISION)


def test_missing_role_manifest_cannot_be_treated_as_optional(bundle):
    root, _ = bundle
    (root / "release-manifest-Docker.json").unlink()
    with pytest.raises(FileNotFoundError):
        release.verify_assets(root, TAG, REVISION)


def install_fake_github(
    monkeypatch, bundle, *, fail_upload=False, corrupt_remote=False, public=False
):
    root, _ = bundle
    expected = {item["name"]: item for item in release.verify_assets(root, TAG, REVISION)}
    calls = []
    state = {"release": {"id": 7, "draft": False} if public else None, "assets": []}

    def fake_gh(*args, timeout=60):
        calls.append((args, timeout))
        if args[0] == "api":
            endpoint = args[1]
            if "/commits/" in endpoint:
                return REVISION
            if "/assets?" in endpoint:
                return json.dumps([state["assets"]])
            if state["release"] is None:
                raise subprocess.CalledProcessError(1, "gh", stderr="HTTP 404")
            return json.dumps(state["release"])
        if args[:2] == ("release", "create"):
            assert "--draft" in args and "--verify-tag" in args
            state["release"] = {
                "id": 7,
                "draft": True,
                "html_url": "https://github.com/example/repo/releases/tag/" + TAG,
            }
        elif args[:2] == ("release", "upload"):
            if fail_upload:
                raise subprocess.CalledProcessError(1, "gh", stderr="upload failed")
            name = Path(args[3]).name
            item = expected[name]
            state["assets"].append(
                {
                    "name": name,
                    "size": item["size"],
                    "state": "uploaded",
                    "digest": "sha256:" + ("0" * 64 if corrupt_remote else item["sha256"]),
                }
            )
        elif args[:2] == ("release", "edit"):
            assert len(state["assets"]) == len(expected)
            assert "--draft=false" in args
            state["release"]["draft"] = False
        else:
            raise AssertionError(args)
        return ""

    monkeypatch.setattr(release, "gh", fake_gh)
    return state, calls


def test_draft_becomes_public_only_after_remote_hashes_match(monkeypatch, bundle):
    state, calls = install_fake_github(monkeypatch, bundle)
    root, notes = bundle
    release.publish_assets(root, TAG, REVISION, "example/repo", notes)
    assert state["release"]["draft"] is False
    assert len([call for call, _ in calls if call[:2] == ("release", "edit")]) == 1
    assert all(timeout == 1800 for call, timeout in calls if call[:2] == ("release", "upload"))


@pytest.mark.parametrize("failure", ["upload", "remote_hash", "public"])
def test_release_stays_unpublished_after_failure(monkeypatch, bundle, failure):
    state, calls = install_fake_github(
        monkeypatch,
        bundle,
        fail_upload=failure == "upload",
        corrupt_remote=failure == "remote_hash",
        public=failure == "public",
    )
    root, notes = bundle
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        release.publish_assets(root, TAG, REVISION, "example/repo", notes)
    assert not any(call[:2] == ("release", "edit") for call, _ in calls)
    if failure == "upload":
        assert len([call for call, _ in calls if call[:2] == ("release", "upload")]) == 1
    if failure != "public":
        assert state["release"]["draft"] is True


def test_api_permission_failure_does_not_create_a_new_release(monkeypatch):
    def denied(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "gh", stderr="HTTP 403")

    monkeypatch.setattr(release, "gh", denied)
    with pytest.raises(subprocess.CalledProcessError):
        release.read_release("example/repo", TAG)


def test_actions_have_one_complete_release_owner():
    def workflow(name):
        return yaml.safe_load((ROOT / ".github/workflows" / name).read_text("utf-8"))

    main = workflow("release.yml")
    jobs = main["jobs"]
    assert set(jobs["release"]["needs"]) == {"preflight", "source", "build", "containers", "space"}
    assert jobs["containers"]["uses"] == "./.github/workflows/container.yml"
    assert jobs["source"]["uses"] == "./.github/workflows/build.yml"
    assert jobs["space"]["uses"] == "./.github/workflows/sync_to_hf.yml"
    for name in ("container.yml", "build.yml", "sync_to_hf.yml"):
        assert "workflow_call" in workflow(name)[True]
    container = (ROOT / ".github/workflows/container.yml").read_text("utf-8")
    assert "gh release create" not in container and "gh release upload" not in container
    assert main[True]["workflow_dispatch"]["inputs"]["tag"]["required"] is True
    assert "tags" not in workflow("container.yml")[True]["push"]
    artifact = next(
        step
        for step in jobs["release"]["steps"]
        if step.get("uses", "").startswith("actions/download-artifact")
    )
    assert artifact["with"]["pattern"] == "MusicToMidi-*"


def test_public_release_notes_do_not_publish_local_acceptance_evidence():
    notes = (ROOT / "docs/releases/v1.10.0.md").read_text("utf-8")
    for forbidden in ("C:/Temp", "C:\\Temp", "261022093", "dump", "7286", "7,286", "GitNexus"):
        assert forbidden not in notes
    assert "## 音乐转 MIDI v1.10.0" in notes
    assert "## Music to MIDI v1.10.0" in notes


@pytest.mark.parametrize("case", ["valid", "runtime", "tag", "commit", "notes", "invalid_tag"])
def test_actual_workflow_preflight_rejects_inconsistent_sources(tmp_path, monkeypatch, case):
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text("utf-8"))
    step = next(
        step for step in workflow["jobs"]["preflight"]["steps"] if step.get("id") == "identity"
    )
    code = step["run"].split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    (tmp_path / "src").mkdir()
    (tmp_path / "docs/releases").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.10.0"\n')
    runtime_version = "1.9.0" if case == "runtime" else "1.10.0"
    (tmp_path / "src/__init__.py").write_text(f'__version__ = "{runtime_version}"\n')
    if case != "notes":
        (tmp_path / "docs/releases/v1.10.0.md").write_text("release notes")
    tag = "v1.9.0" if case == "tag" else ("refs/heads/main" if case == "invalid_tag" else TAG)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RELEASE_TAG", tag)
    monkeypatch.setenv("GITHUB_SHA", REVISION)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))

    def tag_commit(arguments, text):
        assert arguments == ["git", "rev-parse", f"refs/tags/{TAG}^{{commit}}"]
        return "a" * 40 if case == "commit" else REVISION

    monkeypatch.setattr(subprocess, "check_output", tag_commit)
    if case == "valid":
        exec(compile(code, "release-preflight", "exec"), {})
        assert (tmp_path / "outputs").read_text() == f"tag={TAG}\n"
    else:
        with pytest.raises((SystemExit, ValueError)):
            exec(compile(code, "release-preflight", "exec"), {})
        assert not (tmp_path / "outputs").exists()
