import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import download_multistem_model


class MultistemModelDownloaderTests(unittest.TestCase):
    def test_uses_verified_telknet_reference_checkpoint_source(self):
        self.assertIn(
            "noblebarkrr/mvsepless_resources",
            download_multistem_model.ROFORMER_SW_CHECKPOINT_URL,
        )
        self.assertIn(
            "noblebarkrr/mvsepless_resources",
            download_multistem_model.ROFORMER_SW_CONFIG_URL,
        )
        self.assertNotIn("jarredou", download_multistem_model.ROFORMER_SW_CHECKPOINT_URL)
        self.assertNotIn("jarredou", download_multistem_model.ROFORMER_SW_CONFIG_URL)
        self.assertNotIn("resolve/main", download_multistem_model.ROFORMER_SW_CHECKPOINT_URL)
        self.assertNotIn("resolve/main", download_multistem_model.ROFORMER_SW_CONFIG_URL)
        self.assertIn(
            download_multistem_model.ROFORMER_SW_SOURCE_REVISION,
            download_multistem_model.ROFORMER_SW_CHECKPOINT_URL,
        )
        self.assertIn(
            download_multistem_model.ROFORMER_SW_SOURCE_REVISION,
            download_multistem_model.ROFORMER_SW_CONFIG_URL,
        )
        self.assertEqual(download_multistem_model.ROFORMER_SW_DISPLAY_NAME, "BS-RoFormer SW Fixed")
        self.assertEqual(download_multistem_model.ROFORMER_SW_REGISTRY_NAME, "Roformer Model: BS-Roformer-SW-Fixed")
        self.assertEqual(download_multistem_model.ROFORMER_SW_CONFIG, "BS-Rofo-SW-Fixed.yaml")
        self.assertEqual(
            download_multistem_model.ROFORMER_SW_OFFICIAL_CONFIG,
            "BS-Rofo-SW-Fixed.official.yaml",
        )
        self.assertEqual(download_multistem_model.ROFORMER_SW_CHECKPOINT_SIZE, 699_412_152)
        self.assertEqual(
            download_multistem_model.ROFORMER_SW_CHECKPOINT_SHA256,
            "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e",
        )
        self.assertEqual(download_multistem_model.ROFORMER_SW_OFFICIAL_CONFIG_SIZE, 3_530)
        self.assertEqual(
            download_multistem_model.ROFORMER_SW_OFFICIAL_CONFIG_SHA256,
            "4678db9430a87ee33e7fad199166928c9adcd322e2df1a812b4bf03726e2a48b",
        )
        self.assertEqual(download_multistem_model.ROFORMER_SW_COMPATIBLE_CONFIG_SIZE, 3_522)
        self.assertEqual(
            download_multistem_model.ROFORMER_SW_COMPATIBLE_CONFIG_SHA256,
            "e7dc288d2456a9a186c451ca551025db408a1cbf3fff2b98c8eb0077129324c3",
        )

    def test_validate_file_checksum_accepts_exact_match(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            content = b"bs-roformer-sw"
            path.write_bytes(content)

            download_multistem_model.validate_file_checksum(
                path,
                hashlib.sha256(content).hexdigest(),
                len(content),
                "sample",
            )

    def test_validate_file_checksum_rejects_wrong_size(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            path.write_bytes(b"bad")

            with self.assertRaisesRegex(RuntimeError, "大小不匹配"):
                download_multistem_model.validate_file_checksum(
                    path,
                    hashlib.sha256(b"bad").hexdigest(),
                    999,
                    "sample",
                )

    def test_validate_file_checksum_rejects_wrong_hash(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            path.write_bytes(b"bad")

            with self.assertRaisesRegex(RuntimeError, "SHA256 不匹配"):
                download_multistem_model.validate_file_checksum(
                    path,
                    hashlib.sha256(b"other").hexdigest(),
                    3,
                    "sample",
                )

    def test_existing_corrupt_checkpoint_fails_before_downloading(self):
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            model_path = cache_dir / download_multistem_model.ROFORMER_SW_MODEL
            config_path = cache_dir / download_multistem_model.ROFORMER_SW_CONFIG
            model_path.write_bytes(b"corrupt")
            config_path.write_text("model: {}\n", encoding="utf-8")
            calls = []

            def downloader(url, output_path, description):
                calls.append((url, output_path, description))

            with self.assertRaisesRegex(RuntimeError, "大小不匹配"):
                download_multistem_model.download_multistem_model(
                    cache_dir=cache_dir,
                    downloader=downloader,
                )

            self.assertEqual(calls, [])

    def test_compatibility_patch_matches_telknet_bs_roformer_config_fields(self):
        with TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "BS-Rofo-SW-Fixed.official.yaml"
            config_path = Path(tmp) / "BS-Rofo-SW-Fixed.yaml"
            source_text = (
                "model:\n"
                "  freqs_per_bands: !!python/tuple\n"
                "    - 129\n"
                "  dim_head: 64\n"
                "  multi_stft_resolutions_window_sizes: !!python/tuple\n"
                "    - 256\n"
                "augmentations:\n"
                "  mixup_probs: !!python/tuple\n"
                "    - 0.2\n"
                "inference:\n"
                "  dim_t: 1101\n"
            )
            expected_text = (
                source_text.replace("freqs_per_bands: !!python/tuple", "freqs_per_bands:")
                .replace(
                    "multi_stft_resolutions_window_sizes: !!python/tuple",
                    "multi_stft_resolutions_window_sizes:",
                )
                .replace("mixup_probs: !!python/tuple", "mixup_probs:")
                .replace(
                    "    - 129\n  dim_head: 64\n",
                    "    - 129\n  num_subbands: 1\n  dim_head: 64\n",
                )
                + "is_roformer: true\n"
            )
            source_bytes = source_text.encode("utf-8")
            expected_bytes = expected_text.encode("utf-8")
            source_path.write_bytes(source_bytes)

            with (
                patch.object(download_multistem_model, "_REQUIRED_FREQS_PER_BANDS", (129,)),
                patch.object(download_multistem_model, "ROFORMER_SW_OFFICIAL_CONFIG_SIZE", len(source_bytes)),
                patch.object(
                    download_multistem_model,
                    "ROFORMER_SW_OFFICIAL_CONFIG_SHA256",
                    hashlib.sha256(source_bytes).hexdigest(),
                ),
                patch.object(download_multistem_model, "ROFORMER_SW_COMPATIBLE_CONFIG_SIZE", len(expected_bytes)),
                patch.object(
                    download_multistem_model,
                    "ROFORMER_SW_COMPATIBLE_CONFIG_SHA256",
                    hashlib.sha256(expected_bytes).hexdigest(),
                ),
            ):
                modified = download_multistem_model.ensure_multistem_config_compatible(
                    config_path,
                    source_path,
                )
                modified_again = download_multistem_model.ensure_multistem_config_compatible(
                    config_path,
                    source_path,
                )

            self.assertTrue(modified)
            self.assertFalse(modified_again)
            self.assertEqual(config_path.read_bytes(), expected_bytes)
            self.assertNotIn(b"\r\n", config_path.read_bytes())
            self.assertIn("dim_t: 1101", config_path.read_text(encoding="utf-8"))

    def test_check_only_reports_missing_assets(self):
        with TemporaryDirectory() as tmp:
            result = download_multistem_model.main(["--cache-dir", tmp, "--check-only"])

        self.assertEqual(result, 1)

    def test_console_encoding_is_forced_to_utf8(self):
        class FakeStream:
            def __init__(self):
                self.calls = []

            def reconfigure(self, **kwargs):
                self.calls.append(kwargs)

        stdout = FakeStream()
        stderr = FakeStream()
        original_stdout = download_multistem_model.sys.stdout
        original_stderr = download_multistem_model.sys.stderr
        try:
            download_multistem_model.sys.stdout = stdout
            download_multistem_model.sys.stderr = stderr

            download_multistem_model.configure_console_encoding()
        finally:
            download_multistem_model.sys.stdout = original_stdout
            download_multistem_model.sys.stderr = original_stderr

        self.assertEqual(stdout.calls, [{"encoding": "utf-8", "errors": "replace"}])
        self.assertEqual(stderr.calls, [{"encoding": "utf-8", "errors": "replace"}])


if __name__ == "__main__":
    unittest.main()
