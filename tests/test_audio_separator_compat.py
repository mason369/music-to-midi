import unittest
from types import SimpleNamespace

from src.utils.audio_separator_compat import patch_separator_package_metadata


class _FakeSeparator:
    def __init__(self):
        self.calls = []

    def get_package_distribution(self, package_name):
        self.calls.append(package_name)
        if package_name == "audio-separator":
            return None
        return SimpleNamespace(version="9.9.9")


class AudioSeparatorCompatTests(unittest.TestCase):
    def test_patch_returns_placeholder_distribution_when_audio_separator_metadata_missing(self):
        separator_cls = type("FakeSeparatorForMissingMetadata", (_FakeSeparator,), {})

        patch_separator_package_metadata(separator_cls)

        separator = separator_cls()
        distribution = separator.get_package_distribution("audio-separator")

        self.assertIsNotNone(distribution)
        self.assertEqual("unknown", distribution.version)

    def test_patch_preserves_other_package_distribution_results(self):
        separator_cls = type("FakeSeparatorForOtherPackages", (_FakeSeparator,), {})

        patch_separator_package_metadata(separator_cls)

        separator = separator_cls()
        distribution = separator.get_package_distribution("onnxruntime")

        self.assertIsNotNone(distribution)
        self.assertEqual("9.9.9", distribution.version)


if __name__ == "__main__":
    unittest.main()
