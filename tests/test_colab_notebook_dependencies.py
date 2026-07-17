import ast
import json
import os
import re
import tempfile
import unittest
from pathlib import Path


def _find_function(source_text, function_name):
    matches = [
        node
        for node in ast.parse(source_text).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one {function_name!r} function, found {len(matches)}"
        )
    return matches[0]


def _node_source(source_text, node):
    segment = ast.get_source_segment(source_text, node)
    if segment is None:
        raise AssertionError(f"Unable to recover source for {type(node).__name__}")
    return segment


def _called_names(node):
    names = []
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Call):
            continue
        if isinstance(candidate.func, ast.Name):
            names.append(candidate.func.id)
        elif isinstance(candidate.func, ast.Attribute):
            names.append(candidate.func.attr)
    return names


def _assigned_collection(source_text, assignment_name):
    for node in ast.parse(source_text).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == assignment_name
            for target in node.targets
        ):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "frozenset"
            and len(value.args) == 1
            and not value.keywords
        ):
            return frozenset(ast.literal_eval(value.args[0]))
        return ast.literal_eval(value)
    raise AssertionError(f"Missing assignment for {assignment_name!r}")


def _event_bindings(source_text):
    bindings = []
    for node in ast.walk(ast.parse(source_text)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"change", "click", "input", "select", "submit"}:
            continue
        fn_value = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "fn"),
            node.args[0] if node.args else None,
        )
        if isinstance(fn_value, ast.Name):
            bindings.append((node.func.attr, fn_value.id))
        elif isinstance(fn_value, ast.Attribute):
            bindings.append((node.func.attr, fn_value.attr))
        elif isinstance(fn_value, ast.Call) and isinstance(fn_value.func, ast.Name):
            bindings.append((node.func.attr, fn_value.func.id))
    return bindings


class TestColabNotebookDependencies(unittest.TestCase):
    @staticmethod
    def _load_notebook_source_text():
        notebook_path = Path("colab_notebook.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        sources = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                sources.append("".join(cell.get("source", [])))
        return "\n".join(sources)

    @staticmethod
    def _load_notebook_all_source_text():
        notebook_path = Path("colab_notebook.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        sources = []
        for cell in notebook.get("cells", []):
            sources.append("".join(cell.get("source", [])))
        return "\n".join(sources)

    def test_restored_modes_and_dependencies_are_present(self):
        source_text = self._load_notebook_source_text()

        for restored_text in (
            "aria-amt",
            "Aria-AMT",
            "ByteDance Pedal",
            "matplotlib",
            "TransKun",
            "TransKun V2 Aug",
            "六声部分离",
            "six_stem_split",
            "piano_transkun_v2_aug",
            "piano_bytedance_pedal",
            "download_multistem_model",
            "download_vocal_model",
            "download_accompaniment_model",
            "download_transkun_v2_aug_model",
            "download_aria_amt_model",
            "download_bytedance_piano_model",
        ):
            with self.subTest(restored_text=restored_text):
                self.assertIn(restored_text, source_text)

    def test_colab_installs_audio_separator_with_same_runtime_pins(self):
        source_text = self._load_notebook_source_text()

        self.assertIn("audio_separator_runtime_packages = [", source_text)
        self.assertIn('"numpy==1.26.4"', source_text)
        self.assertIn('"onnxruntime-gpu==1.23.2"', source_text)
        self.assertIn('"audio-separator==0.44.1"', source_text)
        self.assertIn('"gradio==4.44.1"', source_text)
        self.assertIn('"fastapi==0.115.2"', source_text)
        self.assertIn('"starlette==0.40.0"', source_text)
        self.assertIn('" --no-deps"', source_text)
        self.assertNotIn('"audio-separator>=0.38.0"', source_text)
        self.assertNotIn('"gradio>=4.44.0"', source_text)

    def test_colab_installs_miros_runtime_dependency_surface(self):
        source_text = self._load_notebook_source_text()

        for package_name in (
            '"einops"',
            '"smart-open"',
            '"pretty-midi>=0.2.10"',
            '"soxr>=0.3.7"',
            '"mido"',
            '"soundfile"',
            '"h5py>=3.10,<4"',
            '"mirdata>=0.3.8,<1"',
        ):
            with self.subTest(package_name=package_name):
                self.assertIn(package_name, source_text)

    def test_colab_pins_piano_backend_packages_and_verifies_aria_source(self):
        source_text = self._load_notebook_source_text()

        self.assertIn('"transkun==2.0.1"', source_text)
        self.assertIn('"piano-transcription-inference==0.0.6"', source_text)
        self.assertIn("--no-deps --force-reinstall", source_text)
        self.assertIn("get_aria_amt_runtime_unavailable_reason", source_text)
        self.assertIn(
            "Aria-AMT source identity verified at " "a1ab73fc901d1759ec3bc173c146b3c6a3040261",
            source_text,
        )

    def test_colab_downloads_all_official_yourmt3_model_modes(self):
        source_text = self._load_notebook_source_text()

        self.assertIn("OFFICIAL_YOURMT3_MODEL_KEYS", source_text)
        self.assertIn("download_model(yourmt3_model_key", source_text)
        self.assertNotIn("download_official_yourmt3_models()", source_text)
        self.assertNotIn("download_ultimate_moe", source_text)

    def test_colab_exposes_official_yourmt3_model_selector_and_description(self):
        source_text = self._load_notebook_source_text()

        self.assertIn("OFFICIAL_YOURMT3_MODEL_KEYS", source_text)
        self.assertIn("YOURMT3_MODEL_LABEL_TO_KEY", source_text)
        self.assertIn("yourmt3_model_dropdown = gr.Dropdown", source_text)
        self.assertIn("yourmt3_model_info = gr.Markdown", source_text)
        self.assertIn("def update_yourmt3_model_info", source_text)
        self.assertIn(
            'config_kwargs = {"processing_mode": mode, "yourmt3_model": yourmt3_model_key}',
            source_text,
        )
        self.assertIn(
            'if mode == "smart":',
            source_text,
        )
        self.assertIn(
            "config_kwargs.update(transcription_backend=backend_key, "
            "multi_instrument_model=backend_key)",
            source_text,
        )
        self.assertIn('model_info.get("features_zh")', source_text)
        self.assertIn("COLAB_I18N", source_text)
        self.assertIn('COLAB_LANGUAGE.startswith("zh")', source_text)
        self.assertIn("ct('yourmt3.model_title')", source_text)
        self.assertIn("ct('yourmt3.checkpoint')", source_text)
        self.assertIn("ct('yourmt3.traits')", source_text)
        self.assertIn('model_info.get("features_en")', source_text)
        self.assertNotIn("**Checkpoint**", source_text)
        self.assertNotIn("**Features**", source_text)

    def test_colab_mode_sets_and_global_backend_controls_match_the_new_contract(self):
        source_text = self._load_notebook_source_text()

        direct_modes = _assigned_collection(source_text, "DIRECT_PROCESSING_MODES")
        split_modes = _assigned_collection(source_text, "SPLIT_PROCESSING_MODES")
        multi_instrument_modes = _assigned_collection(
            source_text, "MULTI_INSTRUMENT_PROCESSING_MODES"
        )
        mode_ids = _assigned_collection(source_text, "COLAB_MODE_IDS")

        self.assertEqual(
            direct_modes,
            frozenset(
                {
                    "smart",
                    "piano_transkun",
                    "piano_transkun_v2_aug",
                    "piano_aria_amt",
                    "piano_bytedance_pedal",
                }
            ),
        )
        self.assertEqual(split_modes, {"vocal_split", "six_stem_split"})
        self.assertEqual(multi_instrument_modes, {"smart"})
        self.assertFalse(direct_modes & split_modes)
        self.assertEqual(set(mode_ids), set(direct_modes) | set(split_modes))

        self.assertIn(
            'MULTI_BACKEND_LABEL_TO_KEY = {"YourMT3+": "yourmt3", "MIROS": "miros"}',
            source_text,
        )
        self.assertIn("backend_dropdown = gr.Dropdown", source_text)
        self.assertIn("yourmt3_model_dropdown = gr.Dropdown", source_text)
        self.assertIn("def update_backend_controls", source_text)
        self.assertIn("inputs=[mode_radio, backend_dropdown]", source_text)
        self.assertIn('is_smart = mode == "smart"', source_text)
        self.assertIn(
            'uses_yourmt3 = is_smart and backend_key == "yourmt3"', source_text
        )
        self.assertIn(
            'uses_yourmt3 = mode == "smart" and backend_key == "yourmt3"',
            source_text,
        )
        self.assertNotIn(
            'MULTI_INSTRUMENT_PROCESSING_MODES = {"smart", "vocal_split", "six_stem_split"}',
            source_text,
        )

    def test_colab_prepares_only_the_selected_route_on_demand(self):
        source_text = self._load_notebook_source_text()

        self.assertIn("def prepare_selected_models", source_text)
        self.assertIn("download_model(yourmt3_model_key", source_text)
        self.assertIn("prepare_miros_model(printer=logger.info)", source_text)
        self.assertIn("download_vocal_model(printer=logger.info)", source_text)
        self.assertIn("download_accompaniment_model(printer=logger.info)", source_text)
        self.assertNotIn("download_chorus_model", source_text)
        self.assertNotIn("download_vocal_harmony_model", source_text)
        self.assertIn("download_multistem_model(printer=logger.info)", source_text)
        self.assertIn("download_transkun_v2_aug_model(printer=logger.info)", source_text)
        self.assertNotIn("download_official_yourmt3_models()", source_text)
        self.assertIn("模型权重将在转换时按需准备", source_text)

    def test_colab_main_action_separates_split_modes_without_automatic_midi(self):
        source_text = self._load_notebook_source_text()
        handler = _find_function(source_text, "convert_audio_to_midi")
        split_branches = [
            node
            for node in ast.walk(handler)
            if isinstance(node, ast.If)
            and any(
                isinstance(name, ast.Name)
                and name.id == "SPLIT_PROCESSING_MODES"
                for name in ast.walk(node.test)
            )
            and any(isinstance(operator, ast.In) for operator in ast.walk(node.test))
        ]
        self.assertEqual(len(split_branches), 1)

        split_branch = ast.Module(body=split_branches[0].body, type_ignores=[])
        split_calls = _called_names(split_branch)
        self.assertIn("AudioSeparationService", split_calls)
        self.assertIn("process", split_calls)
        self.assertIn("_build_track_state", split_calls)
        self.assertNotIn("MusicToMidiPipeline", split_calls)
        self.assertNotIn("build_manual_midi_config", split_calls)
        self.assertNotIn("manual_midi_output_dir", split_calls)
        self.assertTrue(any(isinstance(node, ast.Return) for node in ast.walk(split_branch)))

        handler_calls = _called_names(handler)
        self.assertEqual(handler_calls.count("_build_track_state"), 1)
        self.assertEqual(handler_calls.count("MusicToMidiPipeline"), 1)
        self.assertIn(
            'return output_files, "\\n".join(status_lines), {}',
            _node_source(source_text, handler),
        )
        self.assertIn(
            "from src.core.separation_service import AudioSeparationService, SeparationResult",
            source_text,
        )

        build_track_state_source = _node_source(
            source_text, _find_function(source_text, "_build_track_state")
        )
        for required_track_name in ("vocals", "accompaniment"):
            self.assertIn(required_track_name, build_track_state_source)
        self.assertIn("STEM_KEYS", build_track_state_source)
        self.assertIn("separated_audio", build_track_state_source)

        direct_output_source = _node_source(
            source_text, _find_function(source_text, "_validate_processing_outputs")
        )
        self.assertIn("DIRECT_PROCESSING_MODES", direct_output_source)
        for split_only_attribute in (
            "separated_audio",
            "stem_midi_paths",
            "vocal_midi_path",
            "accompaniment_midi_path",
            "merged_midi_path",
        ):
            self.assertIn(split_only_attribute, direct_output_source)

    def test_colab_fixed_track_rows_use_shared_browser_mixer_and_ten_shared_routes(self):
        source_text = self._load_notebook_source_text()
        self.assertEqual(_assigned_collection(source_text, "MAX_MANUAL_TRACK_ROWS"), 12)

        from src.core.manual_midi import MANUAL_MIDI_ROUTES

        self.assertEqual(len(MANUAL_MIDI_ROUTES), 10)
        self.assertEqual(len(set(MANUAL_MIDI_ROUTES)), 10)
        self.assertIn(
            "MANUAL_MIDI_ROUTE_CHOICES = [(label, route) for label, route in "
            "MANUAL_MIDI_ROUTE_LABEL_TO_KEY.items()]",
            source_text,
        )
        self.assertIn(
            "len(MANUAL_MIDI_ROUTE_CHOICES) != len(MANUAL_MIDI_ROUTES)",
            source_text,
        )
        self.assertIn("choices=MANUAL_MIDI_ROUTE_CHOICES", source_text)

        # The fixed rows render real waveforms through the same shared browser
        # mixer runtime as the Space workbench and the desktop widget.
        self.assertIn("from src.gui.web.track_mixer_runtime import (", source_text)
        self.assertIn("build_track_mixer_html", source_text)
        self.assertIn("display_track_name", source_text)
        self.assertIn("mixer_head", source_text)
        self.assertIn("head=LOG_POLL_JS + mixer_head()", source_text)
        self.assertIn("mixer_html = gr.HTML(", source_text)
        self.assertIn('COLAB_TRANSLATOR = Translator(COLAB_LANGUAGE)', source_text)
        self.assertIn(
            "demo.launch(share=True, server_name=\"0.0.0.0\", server_port=7860, "
            "allowed_paths=[str(COLAB_OUTPUT_ROOT)])",
            source_text,
        )
        self.assertIn("def _make_track_remove_handler(track_index):", source_text)
        self.assertIn(
            ("click", "_make_track_remove_handler"),
            _event_bindings(source_text),
        )
        self.assertNotIn("TRACK_DISPLAY_NAMES", source_text)

        row_loops = []
        for node in ast.walk(ast.parse(source_text)):
            if not isinstance(node, ast.For):
                continue
            if not (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and any(
                    isinstance(argument, ast.Name)
                    and argument.id == "MAX_MANUAL_TRACK_ROWS"
                    for argument in node.iter.args
                )
            ):
                continue
            calls = set(_called_names(node))
            if "Checkbox" in calls:
                row_loops.append((node, calls))
        self.assertEqual(len(row_loops), 1)
        _row_loop, row_calls = row_loops[0]
        self.assertTrue({"Checkbox", "Dropdown", "Button", "File", "Markdown"} <= row_calls)
        self.assertNotIn("Audio", row_calls)

        render_node = _find_function(source_text, "_render_track_rows")

        class _GradioStub:
            @staticmethod
            def update(**kwargs):
                return kwargs

        namespace = {
            "gr": _GradioStub,
            "MAX_MANUAL_TRACK_ROWS": 12,
            "DEFAULT_MANUAL_MIDI_ROUTE": MANUAL_MIDI_ROUTES[0],
        }
        exec(
            compile(
                ast.Module(body=[render_node], type_ignores=[]),
                "<colab-empty-track-render-test>",
                "exec",
            ),
            namespace,
        )
        empty_updates = namespace["_render_track_rows"]({})
        self.assertEqual(len(empty_updates), 2 + 6 * 12)
        self.assertEqual(empty_updates[0], {"visible": False})
        self.assertEqual(empty_updates[1], {"value": "", "visible": False})
        for row_index in range(12):
            row_updates = empty_updates[2 + 6 * row_index : 2 + 6 * (row_index + 1)]
            self.assertEqual(len(row_updates), 6)
            self.assertEqual(row_updates[0], {"visible": False})

    def test_colab_track_selection_is_inert_until_the_explicit_start_click(self):
        source_text = self._load_notebook_source_text()
        bindings = _event_bindings(source_text)
        self.assertEqual(bindings.count(("change", "selection_handler")), 2)
        self.assertEqual(
            bindings.count(("click", "_make_track_conversion_handler")), 1
        )
        self.assertNotIn(("change", "_make_track_conversion_handler"), bindings)
        self.assertNotIn(("select", "_make_track_conversion_handler"), bindings)

        selection_node = _find_function(source_text, "_update_track_selection")
        selection_calls = set(_called_names(selection_node))
        for forbidden_call in (
            "build_manual_midi_config",
            "manual_midi_output_dir",
            "MusicToMidiPipeline",
            "process",
        ):
            self.assertNotIn(forbidden_call, selection_calls)

        selection_wrapper = _find_function(
            source_text, "_make_track_selection_handler"
        )
        self.assertIn("_update_track_selection", _called_names(selection_wrapper))
        self.assertNotIn("convert_track_to_midi", _called_names(selection_wrapper))

        conversion_node = _find_function(source_text, "convert_track_to_midi")
        conversion_calls = _called_names(conversion_node)
        self.assertEqual(conversion_calls.count("prepare_manual_midi_route"), 1)
        self.assertEqual(conversion_calls.count("manual_midi_output_dir"), 1)
        self.assertEqual(conversion_calls.count("MusicToMidiPipeline"), 1)
        self.assertEqual(conversion_calls.count("process"), 1)
        self.assertIn("_require_active_request_dir", conversion_calls)
        self.assertIn("_require_owned_request_file", conversion_calls)
        self.assertIn('if not track["enabled"]:', _node_source(source_text, conversion_node))

        preparation_node = _find_function(source_text, "prepare_manual_midi_route")
        self.assertEqual(
            _called_names(preparation_node).count("build_manual_midi_config"), 1
        )

    def test_colab_added_audio_is_copied_into_the_active_request(self):
        source_text = self._load_notebook_source_text()
        add_node = _find_function(source_text, "add_audio_tracks")
        add_source = _node_source(source_text, add_node)
        add_calls = _called_names(add_node)

        self.assertEqual(add_calls.count("copy2"), 1)
        self.assertIn("_require_source_audio", add_calls)
        self.assertIn("_require_active_request_dir", add_calls)
        self.assertIn("_require_owned_request_file", add_calls)
        self.assertIn("_normalize_track_state", add_calls)
        self.assertIn("MAX_MANUAL_TRACK_ROWS", add_source)
        self.assertIn('added_dir = request_dir / "added"', add_source)
        self.assertIn("copied = _require_owned_request_file", add_source)
        self.assertIn(("click", "add_audio_tracks"), _event_bindings(source_text))

        normalize_source = _node_source(
            source_text, _find_function(source_text, "_normalize_track_state")
        )
        self.assertIn("len(tracks) > MAX_MANUAL_TRACK_ROWS", normalize_source)
        self.assertNotIn("[:MAX_MANUAL_TRACK_ROWS]", normalize_source)

    def test_colab_request_path_guards_reject_missing_empty_and_external_paths(self):
        source_text = self._load_notebook_source_text()
        guard_names = (
            "_require_active_request_dir",
            "_require_owned_request_file",
            "_require_owned_request_output_dir",
        )
        guard_nodes = [_find_function(source_text, name) for name in guard_names]
        for guard_node in guard_nodes:
            self.assertTrue(
                any(isinstance(node, ast.Raise) for node in ast.walk(guard_node)),
                guard_node.name,
            )

        active_source = _node_source(source_text, guard_nodes[0])
        file_source = _node_source(source_text, guard_nodes[1])
        output_source = _node_source(source_text, guard_nodes[2])
        self.assertIn('request_dir.name.startswith("request-")', active_source)
        self.assertIn("request_dir.parent != session_root", active_source)
        self.assertIn("not request_dir.is_dir()", active_source)
        self.assertIn("path.relative_to(request_dir)", file_source)
        self.assertIn("not path.is_file()", file_source)
        self.assertIn("path.stat().st_size <= 0", file_source)
        self.assertIn("path.relative_to(request_dir)", output_source)

        with tempfile.TemporaryDirectory() as temporary_root:
            session_root = Path(temporary_root).resolve()
            request_dir = session_root / "request-valid"
            request_dir.mkdir()
            owned_audio = request_dir / "track.wav"
            owned_audio.write_bytes(b"audio")
            empty_audio = request_dir / "empty.wav"
            empty_audio.write_bytes(b"")
            outside_audio = session_root / "outside.wav"
            outside_audio.write_bytes(b"audio")

            namespace = {"Path": Path, "COLAB_OUTPUT_ROOT": session_root}
            exec(
                compile(
                    ast.Module(body=guard_nodes, type_ignores=[]),
                    "<colab-path-guard-test>",
                    "exec",
                ),
                namespace,
            )
            require_dir = namespace["_require_active_request_dir"]
            require_file = namespace["_require_owned_request_file"]
            require_output = namespace["_require_owned_request_output_dir"]

            self.assertEqual(require_dir(request_dir), request_dir)
            self.assertEqual(
                require_file(request_dir, owned_audio, "audio", {".wav"}),
                owned_audio,
            )
            with self.assertRaises(ValueError):
                require_dir(session_root)
            with self.assertRaises(ValueError):
                require_dir(session_root / "not-a-request")
            with self.assertRaises(ValueError):
                require_file(request_dir, outside_audio, "audio", {".wav"})
            with self.assertRaises(ValueError):
                require_file(request_dir, empty_audio, "audio", {".wav"})
            with self.assertRaises(ValueError):
                require_file(request_dir, owned_audio, "audio", {".flac"})

            owned_output = request_dir / "midi" / "miros"
            self.assertEqual(require_output(request_dir, owned_output), owned_output)
            self.assertTrue(owned_output.is_dir())
            escaped_output = session_root / "escaped-midi"
            with self.assertRaises(ValueError):
                require_output(request_dir, escaped_output)
            self.assertFalse(escaped_output.exists())

    def test_colab_rejects_unknown_backend_and_model_values(self):
        source_text = self._load_notebook_source_text()

        self.assertIn('"error.unknown_backend": "未知多乐器转写后端: {backend}"', source_text)
        self.assertIn(
            '"error.unknown_backend": "Unknown multi-instrument transcription backend: {backend}"',
            source_text,
        )
        self.assertIn(
            'raise gr.Error(ct("error.unknown_backend", backend=backend_label))', source_text
        )
        self.assertIn(
            'raise gr.Error(ct("error.unknown_yourmt3_model", model=yourmt3_model_label))',
            source_text,
        )

    def test_colab_ui_labels_are_localized_without_model_descriptions_in_text_table(self):
        source_text = self._load_notebook_source_text()

        self.assertIn("COLAB_MODE_CHOICES", source_text)
        self.assertIn(
            'audio_input = gr.Audio(label=COLAB_TRANSLATOR.t("space.ui.audio_input")',
            source_text,
        )
        self.assertIn('label=ct("ui.mode_label")', source_text)
        self.assertIn('"▶  " + COLAB_TRANSLATOR.t("toolbar.start_convert")', source_text)
        self.assertIn('"▶  " + COLAB_TRANSLATOR.t("toolbar.start_separation")', source_text)
        self.assertIn("status_output = gr.Textbox(", source_text)
        self.assertIn(
            'uses_yourmt3 = is_smart and backend_key == "yourmt3"',
            source_text,
        )
        self.assertIn("COLAB_MODE_INFO_TERMS", source_text)
        self.assertIn('return ct(f"mode_info.{mode}", **terms)', source_text)
        self.assertNotIn('return "**VOCAL_SPLIT**', source_text)
        self.assertNotIn('return "**六声部分离 + 分别转写**', source_text)
        self.assertNotIn("mode_mapping = {", source_text)

        text_table_match = re.search(
            r"COLAB_I18N = \{\n(?P<table>.*?)\n\}\n\n\n" r"def ct",
            source_text,
            flags=re.S,
        )
        self.assertIsNotNone(text_table_match)
        text_table = text_table_match.group("table")
        colab_i18n = ast.literal_eval("{\n" + text_table + "\n}")
        self.assertIn("zh_CN", colab_i18n)
        self.assertIn("en_US", colab_i18n)
        self.assertEqual(
            set(colab_i18n["zh_CN"]),
            set(colab_i18n["en_US"]),
        )
        for professional_phrase in (
            "Leap XE 90-band vocals + BS-PolarFormer 62-band accompaniment",
            "BS-RoFormer SW Fixed",
            "bass/drums/guitar/piano/vocals/other",
            "CC64",
            "Perceiver-TF",
            "multi-channel decoding",
            "pitch-shift augmentation",
            "MAESTRO",
        ):
            with self.subTest(professional_phrase=professional_phrase):
                self.assertNotIn(professional_phrase, text_table)
                if professional_phrase in {
                    "Leap XE 90-band vocals + BS-PolarFormer 62-band accompaniment",
                    "BS-RoFormer SW Fixed",
                    "bass/drums/guitar/piano/vocals/other",
                    "CC64",
                }:
                    self.assertIn(professional_phrase, source_text)

    def test_colab_removes_stale_split_merge_and_save_toggles(self):
        source_text = self._load_notebook_source_text()
        self.assertNotIn("vocal_split_merge_midi", source_text)
        self.assertNotIn("save_separated_tracks", source_text)
        self.assertNotIn("ui.vocal_merge_midi", source_text)
        self.assertNotIn("ui.save_separated_tracks", source_text)
        self.assertNotIn("_vocal_accompaniment_merged.mid", source_text)
        self.assertNotIn("_vocal.mid", source_text)
        self.assertNotIn("_accompaniment.mid", source_text)
        self.assertIn(
            "inputs=[audio_input, mode_radio, backend_dropdown, "
            "yourmt3_model_dropdown]",
            source_text,
        )

    def test_colab_backend_change_only_refreshes_the_smart_mode_card(self):
        source_text = self._load_notebook_source_text()

        self.assertIn(
            'COLAB_BACKEND_ROUTE_NAMES = {"yourmt3": "YourMT3+", "miros": "MIROS"}',
            source_text,
        )
        self.assertIn(
            "def update_mode_info(mode, backend_label=DEFAULT_MULTI_BACKEND_LABEL):",
            source_text,
        )
        self.assertIn(
            'terms["backend_route"] = COLAB_BACKEND_ROUTE_NAMES[backend_key]',
            source_text,
        )
        self.assertIn("update_mode_info(mode, backend_label),", source_text)
        self.assertIn(
            "outputs=[mode_info, yourmt3_model_dropdown, yourmt3_model_info]",
            source_text,
        )
        self.assertNotIn('"model_family": "YourMT3+"', source_text)

        selected_names = {
            "MULTI_INSTRUMENT_PROCESSING_MODES",
            "MULTI_BACKEND_LABEL_TO_KEY",
            "DEFAULT_MULTI_BACKEND_LABEL",
            "COLAB_LANGUAGE",
            "COLAB_I18N",
            "COLAB_MODE_IDS",
            "COLAB_BACKEND_ROUTE_NAMES",
            "COLAB_MODE_INFO_TERMS",
        }
        selected_functions = {"ct", "resolve_multi_backend", "update_mode_info"}
        selected_nodes = []
        for node in ast.parse(source_text).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in selected_names
                for target in node.targets
            ):
                selected_nodes.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name in selected_functions:
                selected_nodes.append(node)

        class _GradioStub:
            Error = ValueError

        namespace = {"gr": _GradioStub, "os": os}
        exec(
            compile(
                ast.Module(body=selected_nodes, type_ignores=[]),
                "<colab-mode-card-test>",
                "exec",
            ),
            namespace,
        )
        update_mode_info = namespace["update_mode_info"]

        yourmt3_card = update_mode_info("smart", "YourMT3+")
        miros_card = update_mode_info("smart", "MIROS")
        self.assertIn("YourMT3+", yourmt3_card)
        self.assertNotIn("MIROS", yourmt3_card)
        self.assertIn("MIROS", miros_card)
        self.assertNotIn("YourMT3+", miros_card)

        for mode in ("vocal_split", "six_stem_split"):
            with self.subTest(mode=mode):
                yourmt3_card = update_mode_info(mode, "YourMT3+")
                miros_card = update_mode_info(mode, "MIROS")
                self.assertEqual(yourmt3_card, miros_card)
                self.assertNotIn("MIROS", yourmt3_card)
                self.assertNotIn("YourMT3+", yourmt3_card)

    def test_colab_model_preparation_does_not_mask_required_resource_failures(self):
        source_text = self._load_notebook_source_text()

        self.assertNotIn("资源准备失败，可稍后手动运行", source_text)

    def test_existing_colab_checkout_must_be_clean_and_fast_forwarded_to_origin_master(self):
        source_text = self._load_notebook_source_text()

        self.assertIn('"status", "--porcelain"', source_text)
        self.assertIn("Existing /content/music-to-midi worktree is dirty", source_text)
        self.assertIn("git -C /content/music-to-midi pull --ff-only origin master", source_text)
        self.assertIn('"rev-parse", "HEAD"', source_text)
        self.assertIn('"rev-parse", "origin/master"', source_text)
        self.assertIn("if head_commit != origin_master_commit:", source_text)
        self.assertIn("Colab source identity mismatch", source_text)

    def test_colab_rejects_unknown_processing_mode_instead_of_falling_back(self):
        source_text = self._load_notebook_source_text()

        self.assertIn('"error.unknown_mode": "未知处理模式: {mode}"', source_text)
        self.assertIn('"error.unknown_mode": "Unknown processing mode: {mode}"', source_text)
        self.assertIn("if mode not in COLAB_MODE_IDS:", source_text)
        self.assertIn('raise gr.Error(ct("error.unknown_mode", mode=mode))', source_text)
        self.assertIn(
            'config_kwargs = {"processing_mode": mode, "yourmt3_model": yourmt3_model_key}',
            source_text,
        )
        self.assertNotIn(
            'config.processing_mode = mode if mode in COLAB_MODE_IDS else "smart"', source_text
        )

    def test_notebook_preserves_preinstalled_torch_and_avoids_reinstall(self):
        source_text = self._load_notebook_source_text()
        package_block_match = re.search(
            r"packages = \[\n(?P<block>.*?)\n\]",
            source_text,
            flags=re.S,
        )
        self.assertIsNotNone(package_block_match)
        package_block = package_block_match.group("block")

        self.assertIn(
            "检测 Colab 预装 torch 版本",
            source_text,
        )
        self.assertIn(
            'log(f"torch=={torch.__version__}")',
            source_text,
        )
        self.assertIn(
            'log(f"CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda}")',
            source_text,
        )
        self.assertNotIn('"torchaudio"', package_block)

    def test_pip_install_uses_shell_safe_quoting(self):
        source_text = self._load_notebook_source_text()
        self.assertIn("import shlex", source_text)
        self.assertIn(
            'quoted_packages = " ".join(shlex.quote(pkg) for pkg in packages)',
            source_text,
        )
        self.assertIn(
            'run_cmd("python -m pip install " + quoted_packages)',
            source_text,
        )

    def test_post_install_logs_torch_family_versions(self):
        source_text = self._load_notebook_source_text()
        self.assertIn(
            "关键包版本",
            source_text,
        )
        self.assertIn(
            'for module_name in ["torch", "torchaudio", "torchvision", "gradio", "huggingface_hub", "lightning", "librosa"]:',
            source_text,
        )
        self.assertIn(
            "importlib.import_module(module_name)",
            source_text,
        )
        self.assertIn(
            'log(f"{module_name}=={version}")',
            source_text,
        )

    def test_colab_intro_and_ui_match_restored_mode_surface(self):
        all_source_text = self._load_notebook_all_source_text()
        code_source_text = self._load_notebook_source_text()

        for expected_text in (
            "此 Colab 与桌面版、Space 使用同一套七模式工作流",
            "SMART 与四种钢琴模式",
            "主按钮只生成 2/6 个 WAV",
            "分离完成后显示真实波形音轨",
            "10 种转写路线",
            "选择或滚动控件本身不会触发转换",
            "The two split modes create WAV files only",
        ):
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, all_source_text)

        # Mode labels come from the same shared catalog as desktop and Space.
        self.assertIn(
            "COLAB_MODE_CHOICES = [(COLAB_TRANSLATOR.t(f\"main.mode.{mode_id}\"), mode_id) for mode_id in COLAB_MODE_IDS]",
            code_source_text,
        )
        self.assertNotIn("完整混音多乐器转写（SMART）", all_source_text)

        for expected_ui_text in (
            "输出说明",
            "主按钮只分离并保留 vocals/accompaniment 两个 WAV",
            "不会自动生成 MIDI 或合并 MIDI",
            "每条音轨可独立勾选、选择 10 种转写路线之一",
            "only separates and keeps vocals/accompaniment WAV files",
            "does not generate stem or merged MIDI automatically",
            "converted only by an explicit Start click",
        ):
            with self.subTest(expected_ui_text=expected_ui_text):
                self.assertIn(expected_ui_text, code_source_text)

        text_table_match = re.search(
            r"COLAB_I18N = \{\n(?P<table>.*?)\n\}\n\n\n" r"def ct",
            code_source_text,
            flags=re.S,
        )
        self.assertIsNotNone(text_table_match)
        colab_i18n = ast.literal_eval(
            "{\n" + text_table_match.group("table") + "\n}"
        )
        zh_copy = colab_i18n["zh_CN"]
        en_copy = colab_i18n["en_US"]
        for direct_key in (
            "mode_info.smart",
            "mode_info.piano_transkun",
            "mode_info.piano_transkun_v2_aug",
            "mode_info.piano_aria_amt",
            "mode_info.piano_bytedance_pedal",
        ):
            with self.subTest(direct_key=direct_key):
                self.assertIn("直接", zh_copy[direct_key])
                self.assertIn("direct", en_copy[direct_key].lower())
        self.assertIn("只生成", zh_copy["mode_info.vocal_split"])
        self.assertIn("六个 WAV", zh_copy["mode_info.six_stem_split"])
        self.assertIn("generates only", en_copy["mode_info.vocal_split"])
        self.assertIn("WAV files", en_copy["mode_info.six_stem_split"])
        self.assertIn("最多显示 {limit} 条音轨", zh_copy["error.too_many_tracks"])
        self.assertIn("At most {limit} tracks", en_copy["error.too_many_tracks"])
        # The timeline hint is the shared audio_tracks subtitle key now.
        self.assertIn(
            'COLAB_TRANSLATOR.t("dialogs.complete.audio_tracks.subtitle")',
            code_source_text,
        )
        shared_zh = json.loads(Path("src/i18n/zh_CN.json").read_text(encoding="utf-8"))
        shared_en = json.loads(Path("src/i18n/en_US.json").read_text(encoding="utf-8"))
        self.assertIn(
            "不会改变",
            shared_zh["dialogs"]["complete"]["audio_tracks"]["subtitle"],
        )
        self.assertIn(
            "scroll",
            shared_en["dialogs"]["complete"]["audio_tracks"]["subtitle"].lower(),
        )


if __name__ == "__main__":
    unittest.main()
