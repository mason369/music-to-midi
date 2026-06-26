import unittest
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch

from src.utils.audio_separator_compat import (
    execute_audio_separator_job,
    get_separator_cls,
    patch_separator_package_metadata,
)


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

    def test_execute_audio_separator_job_loads_ensemble_with_preset_constructor_kwarg(self):
        seen = {}

        class FakeSeparator:
            def __init__(self, output_dir, model_file_dir, output_format, ensemble_preset=None):
                seen["ensemble_preset"] = ensemble_preset
                seen["output_dir"] = output_dir
                self.loaded = False

            def load_model(self):
                self.loaded = True
                seen["load_model_args"] = ()

        with patch("src.utils.audio_separator_compat.get_device", return_value="cpu"):
            separator, result, used_cpu, reason = execute_audio_separator_job(
                FakeSeparator,
                separator_kwargs={
                    "output_dir": "out",
                    "model_file_dir": "models",
                    "output_format": "WAV",
                },
                model_name="ensemble:vocal_rvc",
                action=lambda active_separator: active_separator.loaded,
                logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
            )

        self.assertTrue(result)
        self.assertFalse(used_cpu)
        self.assertIsNone(reason)
        self.assertTrue(separator.loaded)
        self.assertEqual(seen["ensemble_preset"], "vocal_rvc")
        self.assertEqual(seen["load_model_args"], ())

    def test_execute_audio_separator_job_rejects_ensemble_when_runtime_lacks_kwarg(self):
        class OldSeparator:
            def __init__(self, output_dir, model_file_dir, output_format):
                pass

            def load_model(self, model_name):
                raise AssertionError(model_name)

        with patch("src.utils.audio_separator_compat.get_device", return_value="cpu"):
            with self.assertRaisesRegex(RuntimeError, "ensemble_preset"):
                execute_audio_separator_job(
                    OldSeparator,
                    separator_kwargs={
                        "output_dir": "out",
                        "model_file_dir": "models",
                        "output_format": "WAV",
                    },
                    model_name="ensemble:karaoke",
                    action=lambda active_separator: active_separator,
                    logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
                )


if __name__ == "__main__":
    unittest.main()
