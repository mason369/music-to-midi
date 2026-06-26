import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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
        self.assertEqual(download_multistem_model.ROFORMER_SW_DISPLAY_NAME, "BS-RoFormer SW Fixed")
        self.assertEqual(download_multistem_model.ROFORMER_SW_REGISTRY_NAME, "Roformer Model: BS-Roformer-SW-Fixed")
        self.assertEqual(download_multistem_model.ROFORMER_SW_CONFIG, "BS-Rofo-SW-Fixed.yaml")
        self.assertEqual(download_multistem_model.ROFORMER_SW_CHECKPOINT_SIZE, 699_412_152)
        self.assertEqual(
            download_multistem_model.ROFORMER_SW_CHECKPOINT_SHA256,
            "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e",
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
            config_path = Path(tmp) / "BS-Rofo-SW-Fixed.yaml"
            config_path.write_text("model: {}\ninference: {}\n", encoding="utf-8")

            modified = download_multistem_model.ensure_multistem_config_compatible(config_path)

            text = config_path.read_text(encoding="utf-8")
            self.assertTrue(modified)
            self.assertIn("is_roformer: true", text)
            self.assertIn("freqs_per_bands:", text)
            self.assertIn("num_subbands: 62", text)
            self.assertIn("dim_t: 1151", text)

    def test_check_only_reports_missing_assets(self):
        with TemporaryDirectory() as tmp:
            result = download_multistem_model.main(["--cache-dir", tmp, "--check-only"])

        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
