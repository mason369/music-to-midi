from pathlib import Path

DOCUMENTS = (
    Path("README.md"),
    Path("docs/README.md"),
    Path("docs/README_zh.md"),
    Path("space/README.md"),
)


def test_public_readmes_identify_muscriptor_and_its_score_protocol():
    combined = "\n".join(path.read_text(encoding="utf-8") for path in DOCUMENTS)

    for expected in (
        "MuScriptor Large",
        "MuScriptor Medium",
        "MuScriptor Small",
        "d73147e75e5b9b0c0a79ebe154587db4fd603e0c",
        "60.4 / 72.4 / 48.6 / 49.6 / 47.8",
        "372",
        "CC BY-NC 4.0",
        "Mirelo Studio",
        "muscriptor-model.md",
    ):
        assert expected in combined


def test_public_setup_docs_explain_all_gated_authorization_paths_and_caches():
    source_docs = (
        Path("README.md"),
        Path("docs/README.md"),
        Path("docs/README_zh.md"),
    )
    for path in source_docs:
        text = path.read_text(encoding="utf-8")
        for repo_id in (
            "MuScriptor/muscriptor-small",
            "MuScriptor/muscriptor-medium",
            "MuScriptor/muscriptor-large",
        ):
            assert repo_id in text
        assert "HF_TOKEN" in text
        assert "hf auth login" in text
        assert "${HF_HOME:-~/.cache/huggingface}/hub" in text
        assert "~/.music-to-midi/models/beat_this" in text
        assert "~/.cache/music_ai_models/fluidsynth/2.5.6" in text
        assert "venv" in text
        assert "MidiOutput" in text

    space = Path("space/README.md").read_text(encoding="utf-8")
    assert "Settings -> Variables and secrets" in space
    assert "私有 secret `HF_TOKEN`" in space
    assert "不能匿名全自动下载" in space

    colab = Path("colab_notebook.ipynb").read_text(encoding="utf-8")
    assert "ENABLE_MUSCRIPTOR" in colab
    assert 'userdata.get(\\"HF_TOKEN\\")' in colab
    assert "preflight_muscriptor_download_access" in colab


def test_detailed_muscriptor_document_keeps_benchmarks_and_limits_together():
    details = Path("docs/muscriptor-model.md").read_text(encoding="utf-8")

    for expected in (
        "1.45M",
        "170k",
        "约 11,000 小时",
        "MuScriptor Small",
        "| Medium |",
        "MuScriptor Large",
        "Bach10",
        "RWC-R",
        "跨所有 benchmark 的无条件 SOTA",
        "不生成 velocity",
        "不能被认定为",
    ):
        assert expected in details


def test_split_workflow_docs_do_not_claim_automatic_midi_generation():
    for path in DOCUMENTS[:3]:
        text = path.read_text(encoding="utf-8")
        assert "13" in text
        assert "MuScriptor" in text

    space = DOCUMENTS[3].read_text(encoding="utf-8")
    assert "十三条路线" in space
    assert "MuScriptor" in space

    combined = "\n".join(path.read_text(encoding="utf-8") for path in DOCUMENTS)
    for stale_claim in (
        "每个 stem 独立运行所选 YourMT3+ / MIROS",
        "continue to produce per-stem MIDI",
        "e2bd0fc5994f9acba7c1387ca5df67eb8d95df44",
        "package `0.2.2`",
        "包版本 `0.2.2`",
        "11 explicit MIDI routes",
        "11 routes in total",
        "11 条逐轨路线",
        "十一路线",
        "Public, not integrated",
        "未来候选 | 103M / 307M",
        "edaebd3126336bd7eb4467dcf675d77f4e7772f0",
        "PR #58",
        "2.5-second overlap",
        "2.5 秒重叠",
        "gzip restart",
        "gzip 异常重启",
    ):
        assert stale_claim not in combined
