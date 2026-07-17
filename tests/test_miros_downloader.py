import hashlib
import io
import os
import subprocess
import tempfile
import unittest
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import download_miros_model
import src.core.miros_transcriber as miros_runtime
from src.core.miros_transcriber import MirosTranscriber


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checkpoint_payload() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as checkpoint:
        checkpoint.writestr("metadata.json", "{}")
    return buffer.getvalue()


def _write_source(repo: Path, *, patched: bool) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "main.py").write_text("print('miros')\n", encoding="utf-8")
    audio_segments = (
        download_miros_model.MIROS_AUDIO_SEGMENTS_CPU_BLOCK
        if patched
        else download_miros_model.MIROS_AUDIO_SEGMENTS_GPU_BLOCK
    )
    inference_context = (
        download_miros_model.MIROS_INFERENCE_CONTEXT_NEW
        if patched
        else download_miros_model.MIROS_INFERENCE_CONTEXT_OLD
    )
    inference_batch = (
        download_miros_model.MIROS_INFERENCE_BATCH_NEW
        if patched
        else download_miros_model.MIROS_INFERENCE_BATCH_OLD
    )
    (repo / "transcribe.py").write_text(
        "\n".join((audio_segments, inference_context, inference_batch, "")),
        encoding="utf-8",
    )

    decmod = repo / download_miros_model.MIROS_DECMOD_REL_PATH
    decmod.parent.mkdir(parents=True, exist_ok=True)
    decmod_lines = [
        "if is_torch_flex_attn_available():",
        "    from torch.nn.attention.flex_attention import BlockMask",
    ]
    if not patched:
        decmod_lines.append(download_miros_model.MIROS_FLEX_ATTENTION_BAD_IMPORT)
    decmod.write_text("\n".join(decmod_lines) + "\n", encoding="utf-8")

    rope = repo / download_miros_model.MIROS_ROPE_REL_PATH
    rope.parent.mkdir(parents=True, exist_ok=True)
    rope_import = (
        download_miros_model.MIROS_ROPE_NEW_IMPORT
        if patched
        else download_miros_model.MIROS_ROPE_OLD_IMPORT
    )
    rope_decorator = (
        download_miros_model.MIROS_ROPE_NEW_DECORATOR
        if patched
        else download_miros_model.MIROS_ROPE_OLD_DECORATOR
    )
    rope.write_text(
        "\n".join(
            [
                rope_import,
                "",
                rope_decorator,
                "def apply_rotary_emb():",
                "    pass",
                "",
                "class RotaryEmbedding:",
                f"    {rope_decorator}",
                "    def forward(self):",
                "        pass",
                "",
            ]
        ),
        encoding="utf-8",
    )


@contextmanager
def _identity_overrides(repo: Path, fine_tuned: bytes, pretrained: bytes):
    source_sha256 = miros_runtime.compute_miros_source_tree_sha256(repo)
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(miros_runtime, "MIROS_PATCHED_SOURCE_SHA256", source_sha256)
        )
        stack.enter_context(
            patch.object(miros_runtime, "MIROS_FINETUNED_EXACT_BYTES", len(fine_tuned))
        )
        stack.enter_context(
            patch.object(miros_runtime, "MIROS_FINETUNED_SHA256", _sha256(fine_tuned))
        )
        stack.enter_context(
            patch.object(miros_runtime, "MIROS_PRETRAINED_EXACT_BYTES", len(pretrained))
        )
        stack.enter_context(
            patch.object(miros_runtime, "MIROS_PRETRAINED_SHA256", _sha256(pretrained))
        )
        yield


class MirosDownloaderTests(unittest.TestCase):
    def test_all_upstream_artifact_identities_are_pinned(self):
        self.assertEqual(
            "668a0aa6357bb3f09e767c9ece378956c2ffd182",
            download_miros_model.MIROS_SOURCE_COMMIT,
        )
        self.assertEqual(
            "38a067527acf6b17f458053077ff143fcedc253edc36f28b69e6ce441c0ca35d",
            download_miros_model.MIROS_UNPATCHED_SOURCE_SHA256,
        )
        self.assertEqual(
            "546287d5e3e9ea5b42a4135d1dbca96ac12a0a9c",
            download_miros_model.MIROS_PRETRAINED_COMMIT,
        )
        self.assertEqual(
            "https://huggingface.co/minzwon/MusicFM/resolve/"
            "546287d5e3e9ea5b42a4135d1dbca96ac12a0a9c/pretrained_msd.pt",
            download_miros_model.MIROS_PRETRAINED_URL,
        )
        self.assertEqual(1_316_802_088, download_miros_model.MIROS_PRETRAINED_EXACT_BYTES)
        self.assertEqual(
            "218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded",
            download_miros_model.MIROS_PRETRAINED_SHA256,
        )
        self.assertEqual(4_347_922_234, download_miros_model.MIROS_FINETUNED_EXACT_BYTES)
        self.assertEqual(
            "b1b8c167b3d2e3eaeb19202cd3fd366bb43492cd7720ff1516e1553c72e356e5",
            download_miros_model.MIROS_FINETUNED_SHA256,
        )

    def test_runtime_source_uses_bounded_inference_memory_contract(self):
        repo = Path(__file__).resolve().parents[1] / "external" / "ai4m-miros"
        transcribe_text = (repo / "transcribe.py").read_text(encoding="utf-8")

        self.assertIn(
            download_miros_model.MIROS_AUDIO_SEGMENTS_CPU_BLOCK,
            transcribe_text,
        )
        self.assertIn(
            download_miros_model.MIROS_INFERENCE_CONTEXT_NEW,
            transcribe_text,
        )
        self.assertIn(
            download_miros_model.MIROS_INFERENCE_BATCH_NEW,
            transcribe_text,
        )
        self.assertEqual(
            miros_runtime.MIROS_PATCHED_SOURCE_SHA256,
            miros_runtime.compute_miros_source_tree_sha256(repo),
        )

    def test_previous_approved_source_is_upgraded_to_bounded_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ai4m-miros"
            _write_source(repo, patched=True)
            (repo / "transcribe.py").write_text(
                "\n".join(
                    (
                        download_miros_model.MIROS_AUDIO_SEGMENTS_GPU_BLOCK,
                        download_miros_model.MIROS_INFERENCE_CONTEXT_OLD,
                        download_miros_model.MIROS_INFERENCE_BATCH_OLD,
                        "",
                    )
                ),
                encoding="utf-8",
            )
            previous_sha256 = miros_runtime.compute_miros_source_tree_sha256(repo)

            with patch.object(
                miros_runtime,
                "MIROS_PREVIOUS_PATCHED_SOURCE_SHA256",
                previous_sha256,
            ):
                download_miros_model._patch_miros_source(repo, None)

            transcribe_text = (repo / "transcribe.py").read_text(encoding="utf-8")
            self.assertIn(
                download_miros_model.MIROS_AUDIO_SEGMENTS_CPU_BLOCK,
                transcribe_text,
            )
            self.assertIn(
                download_miros_model.MIROS_INFERENCE_BATCH_NEW,
                transcribe_text,
            )

    def test_source_patcher_allows_only_deterministic_decmod_and_rope_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ai4m-miros"
            _write_source(repo, patched=False)
            unpatched_sha256 = miros_runtime.compute_miros_source_tree_sha256(repo)
            with patch.object(
                miros_runtime,
                "MIROS_UNPATCHED_SOURCE_SHA256",
                unpatched_sha256,
            ):
                download_miros_model._patch_miros_source(repo, None)
            expected_sha256 = miros_runtime.compute_miros_source_tree_sha256(repo)
            transcribe_text = (repo / "transcribe.py").read_text(encoding="utf-8")

            self.assertIn(
                download_miros_model.MIROS_AUDIO_SEGMENTS_CPU_BLOCK,
                transcribe_text,
            )
            self.assertIn(
                download_miros_model.MIROS_INFERENCE_CONTEXT_NEW,
                transcribe_text,
            )
            self.assertIn(
                download_miros_model.MIROS_INFERENCE_BATCH_NEW,
                transcribe_text,
            )
            self.assertNotIn(
                download_miros_model.MIROS_AUDIO_SEGMENTS_GPU_BLOCK,
                transcribe_text,
            )

            with patch.object(
                miros_runtime,
                "MIROS_PATCHED_SOURCE_SHA256",
                expected_sha256,
            ):
                self.assertEqual("", miros_runtime.get_miros_source_identity_error(repo))
                (repo / "unexpected.py").write_text("tampered\n", encoding="utf-8")
                error = miros_runtime.get_miros_source_identity_error(repo)

        self.assertIn("patched source tree SHA256 mismatch", error)

    def test_source_identity_rejects_wrong_git_base_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ai4m-miros"
            _write_source(repo, patched=True)
            (repo / ".git").mkdir()
            expected_sha256 = miros_runtime.compute_miros_source_tree_sha256(repo)
            completed = subprocess.CompletedProcess(
                ["git", "rev-parse", "HEAD"],
                0,
                stdout="0" * 40 + "\n",
                stderr="",
            )
            with (
                patch.object(
                    miros_runtime,
                    "MIROS_PATCHED_SOURCE_SHA256",
                    expected_sha256,
                ),
                patch.object(miros_runtime.subprocess, "run", return_value=completed),
            ):
                error = miros_runtime.get_miros_source_identity_error(repo)

        self.assertIn("source commit mismatch", error)
        self.assertIn(miros_runtime.MIROS_SOURCE_COMMIT, error)

    def test_invalid_existing_source_is_rejected_before_any_patch_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ai4m-miros"
            _write_source(repo, patched=False)
            decmod = repo / download_miros_model.MIROS_DECMOD_REL_PATH
            rope = repo / download_miros_model.MIROS_ROPE_REL_PATH
            (repo / "unexpected.py").write_text("tampered\n", encoding="utf-8")
            before = (decmod.read_bytes(), rope.read_bytes())

            with self.assertRaises(RuntimeError) as caught:
                download_miros_model._patch_miros_source(repo, None)

            self.assertEqual(before, (decmod.read_bytes(), rope.read_bytes()))

        self.assertIn("neither the pristine pinned commit", str(caught.exception))

    def test_prepare_clones_exact_commit_atomically_and_downloads_exact_weights(self):
        fine_tuned = _checkpoint_payload()
        pretrained = b"pretrained-musicfm"
        calls: list[tuple[list[str], str | None]] = []

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / ".tmp" / "ai4m-miros"

            def fake_run(command, cwd=None, **_kwargs):
                command = list(command)
                calls.append((command, cwd))
                if command[:2] == ["git", "init"]:
                    (Path(cwd) / ".git").mkdir()
                elif command[:3] == ["git", "checkout", "--detach"]:
                    _write_source(Path(cwd), patched=True)
                    miros_runtime.MIROS_PATCHED_SOURCE_SHA256 = (
                        miros_runtime.compute_miros_source_tree_sha256(Path(cwd))
                    )
                elif command[:3] == ["git", "rev-parse", "HEAD"]:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=miros_runtime.MIROS_SOURCE_COMMIT + "\n",
                        stderr="",
                    )
                elif command[0] == "curl":
                    output = Path(command[command.index("-o") + 1])
                    output.write_bytes(
                        fine_tuned if "drive.usercontent.google.com" in command[-1] else pretrained
                    )
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(miros_runtime, "MIROS_PATCHED_SOURCE_SHA256", "pending")
                )
                stack.enter_context(
                    patch.object(
                        miros_runtime,
                        "MIROS_FINETUNED_EXACT_BYTES",
                        len(fine_tuned),
                    )
                )
                stack.enter_context(
                    patch.object(
                        miros_runtime,
                        "MIROS_FINETUNED_SHA256",
                        _sha256(fine_tuned),
                    )
                )
                stack.enter_context(
                    patch.object(
                        miros_runtime,
                        "MIROS_PRETRAINED_EXACT_BYTES",
                        len(pretrained),
                    )
                )
                stack.enter_context(
                    patch.object(
                        miros_runtime,
                        "MIROS_PRETRAINED_SHA256",
                        _sha256(pretrained),
                    )
                )
                stack.enter_context(
                    patch.object(download_miros_model.shutil, "which", return_value="git")
                )
                stack.enter_context(
                    patch.object(download_miros_model.subprocess, "run", side_effect=fake_run)
                )
                result = download_miros_model.prepare_miros_model(repo)

            self.assertEqual(repo, result)
            self.assertEqual(
                fine_tuned,
                (repo / MirosTranscriber.CHECKPOINT_REL_PATH).read_bytes(),
            )
            self.assertEqual(
                pretrained,
                (repo / MirosTranscriber.PRETRAINED_REL_PATH).read_bytes(),
            )
            self.assertFalse(list(repo.parent.glob("*.download")))

        commands = [command for command, _cwd in calls]
        self.assertIn(
            [
                "git",
                "fetch",
                "--depth=1",
                "origin",
                miros_runtime.MIROS_SOURCE_COMMIT,
            ],
            commands,
        )
        self.assertIn(["git", "checkout", "--detach", "FETCH_HEAD"], commands)

    def test_invalid_cached_weight_is_rejected_without_redownload(self):
        fine_tuned = _checkpoint_payload()
        pretrained = b"correct-pretrained"
        wrong_pretrained = b"x" * len(pretrained)

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ai4m-miros"
            _write_source(repo, patched=True)
            checkpoint = repo / MirosTranscriber.CHECKPOINT_REL_PATH
            pretrained_path = repo / MirosTranscriber.PRETRAINED_REL_PATH
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            pretrained_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(fine_tuned)
            pretrained_path.write_bytes(wrong_pretrained)

            with (
                _identity_overrides(repo, fine_tuned, pretrained),
                patch.object(
                    download_miros_model.subprocess,
                    "run",
                    side_effect=AssertionError("invalid cache must not trigger a network fallback"),
                ),
            ):
                with self.assertRaises(RuntimeError) as caught:
                    download_miros_model.prepare_miros_model(repo)

            self.assertEqual(wrong_pretrained, pretrained_path.read_bytes())

        self.assertIn("SHA256 mismatch", str(caught.exception))

    def test_release_mirror_parts_require_exact_identity_before_atomic_replace(self):
        fine_tuned = _checkpoint_payload()
        pretrained = b"pretrained"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "ai4m-miros"
            mirror = root / "mirror"
            mirror.mkdir()
            _write_source(repo, patched=True)
            split_at = len(fine_tuned) // 2
            (mirror / "miros-last.ckpt.partaa").write_bytes(fine_tuned[:split_at])
            (mirror / "miros-last.ckpt.partab").write_bytes(fine_tuned[split_at:])

            def fake_run(command, **_kwargs):
                output = Path(command[command.index("-o") + 1])
                output.write_bytes(pretrained)
                return subprocess.CompletedProcess(command, 0)

            with (
                _identity_overrides(repo, fine_tuned, pretrained),
                patch.dict(
                    os.environ,
                    {download_miros_model.MIROS_MIRROR_DIR_ENV: str(mirror)},
                ),
                patch.object(download_miros_model.subprocess, "run", side_effect=fake_run),
            ):
                download_miros_model.prepare_miros_model(repo)

            self.assertEqual(
                fine_tuned,
                (repo / MirosTranscriber.CHECKPOINT_REL_PATH).read_bytes(),
            )

    def test_corrupt_release_mirror_does_not_create_destination(self):
        fine_tuned = _checkpoint_payload()
        pretrained = b"pretrained"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "ai4m-miros"
            mirror = root / "mirror"
            mirror.mkdir()
            _write_source(repo, patched=True)
            (mirror / "miros-last.ckpt.partaa").write_bytes(b"x" * len(fine_tuned))
            checkpoint_path = repo / MirosTranscriber.CHECKPOINT_REL_PATH

            with (
                _identity_overrides(repo, fine_tuned, pretrained),
                patch.dict(
                    os.environ,
                    {download_miros_model.MIROS_MIRROR_DIR_ENV: str(mirror)},
                ),
            ):
                with self.assertRaises(RuntimeError) as caught:
                    download_miros_model.prepare_miros_model(repo)

            self.assertFalse(checkpoint_path.exists())
            self.assertFalse(list(checkpoint_path.parent.glob("*.download")))

        self.assertIn("SHA256 mismatch", str(caught.exception))

    def test_google_drive_confirmation_download_is_verified_before_publish(self):
        fine_tuned = _checkpoint_payload()
        pretrained = b"pretrained"
        calls: list[list[str]] = []
        warning_page = b"""
        <!DOCTYPE html>
        <form id="download-form" action="https://drive.usercontent.google.com/download" method="get">
          <input type="hidden" name="id" value="file-id">
          <input type="hidden" name="export" value="download">
          <input type="hidden" name="confirm" value="t">
          <input type="hidden" name="uuid" value="confirm-uuid">
        </form>
        """

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ai4m-miros"
            _write_source(repo, patched=True)

            def fake_run(command, **_kwargs):
                command = list(command)
                calls.append(command)
                output = Path(command[command.index("-o") + 1])
                if len(calls) == 1:
                    output.write_bytes(warning_page)
                elif len(calls) == 2:
                    output.write_bytes(fine_tuned)
                else:
                    output.write_bytes(pretrained)
                return subprocess.CompletedProcess(command, 0)

            with (
                _identity_overrides(repo, fine_tuned, pretrained),
                patch.object(
                    download_miros_model.subprocess,
                    "run",
                    side_effect=fake_run,
                ),
            ):
                download_miros_model.prepare_miros_model(repo)

            self.assertEqual(3, len(calls))
            self.assertTrue(any("uuid=confirm-uuid" in command[-1] for command in calls))
            self.assertEqual(
                fine_tuned,
                (repo / MirosTranscriber.CHECKPOINT_REL_PATH).read_bytes(),
            )

    def test_failed_direct_download_leaves_no_partial_or_published_weight(self):
        fine_tuned = _checkpoint_payload()
        pretrained = b"pretrained"
        wrong = b"x" * len(fine_tuned)

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ai4m-miros"
            _write_source(repo, patched=True)
            checkpoint_path = repo / MirosTranscriber.CHECKPOINT_REL_PATH

            def fake_run(command, **_kwargs):
                output = Path(command[command.index("-o") + 1])
                output.write_bytes(wrong)
                return subprocess.CompletedProcess(command, 0)

            with (
                _identity_overrides(repo, fine_tuned, pretrained),
                patch.object(
                    download_miros_model.subprocess,
                    "run",
                    side_effect=fake_run,
                ),
            ):
                with self.assertRaises(RuntimeError) as caught:
                    download_miros_model.prepare_miros_model(repo)

            self.assertFalse(checkpoint_path.exists())
            self.assertFalse(list(checkpoint_path.parent.glob("*.download")))

        self.assertIn("SHA256 mismatch", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
