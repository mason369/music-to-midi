import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import download_miros_model
from src.core.miros_transcriber import MirosTranscriber


class MirosDownloaderTests(unittest.TestCase):
    def test_prepare_clones_repo_and_downloads_missing_weights(self):
        calls = []

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / ".tmp" / "ai4m-miros"

            def fake_run(command, **_kwargs):
                calls.append(command)
                if command[:2] == ["git", "clone"]:
                    repo.mkdir(parents=True)
                    (repo / "main.py").write_text("print('miros')", encoding="utf-8")
                    (repo / "transcribe.py").write_text("print('miros')", encoding="utf-8")
                    return subprocess.CompletedProcess(command, 0)
                if command[0] == "curl":
                    output = Path(command[command.index("-o") + 1])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"x" * 16)
                    return subprocess.CompletedProcess(command, 0)
                raise AssertionError(f"unexpected command: {command}")

            with patch.object(download_miros_model, "MIROS_MIN_CHECKPOINT_BYTES", 8), patch.object(
                download_miros_model,
                "MIROS_MIN_PRETRAINED_BYTES",
                8,
            ), patch.object(download_miros_model.subprocess, "run", side_effect=fake_run):
                result = download_miros_model.prepare_miros_model(repo)

            self.assertEqual(result, repo)
            self.assertTrue((repo / MirosTranscriber.CHECKPOINT_REL_PATH).is_file())
            self.assertTrue((repo / MirosTranscriber.PRETRAINED_REL_PATH).is_file())
            self.assertTrue(any(command[:2] == ["git", "clone"] for command in calls))
            self.assertEqual(2, sum(1 for command in calls if command[0] == "curl"))

    def test_prepare_rejects_incomplete_google_drive_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ai4m-miros"
            repo.mkdir()
            (repo / "main.py").write_text("print('miros')", encoding="utf-8")
            (repo / "transcribe.py").write_text("print('miros')", encoding="utf-8")

            def fake_run(command, **_kwargs):
                output = Path(command[command.index("-o") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"too small")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(download_miros_model, "MIROS_MIN_CHECKPOINT_BYTES", 1024), patch.object(
                download_miros_model.subprocess,
                "run",
                side_effect=fake_run,
            ):
                with self.assertRaises(RuntimeError) as error:
                    download_miros_model.prepare_miros_model(repo)

        self.assertIn("incomplete", str(error.exception))


if __name__ == "__main__":
    unittest.main()
