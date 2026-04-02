import unittest
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch

from src.utils.audio_separator_compat import get_separator_cls, patch_separator_package_metadata


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

    def test_get_separator_cls_activates_audio_separator_runtime_before_import(self):
        separator_module = ModuleType("audio_separator.separator")
        separator_module.Separator = type("ImportedSeparator", (_FakeSeparator,), {})
        package_module = ModuleType("audio_separator")
        package_module.separator = separator_module

        with patch("src.utils.audio_separator_compat.activate_audio_separator_runtime") as activate_runtime, patch.dict(
            "sys.modules",
            {
                "audio_separator": package_module,
                "audio_separator.separator": separator_module,
            },
            clear=False,
        ):
            separator_cls = get_separator_cls()

        activate_runtime.assert_called_once()
        self.assertIs(separator_cls, separator_module.Separator)


if __name__ == "__main__":
    unittest.main()
