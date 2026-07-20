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
        "60.4 / 72.4 / 48.6 / 49.6 / 47.8",
        "372",
        "CC BY-NC 4.0",
        "Mirelo Studio",
        "muscriptor-model.md",
    ):
        assert expected in combined


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
        assert "11" in text
        assert "MuScriptor" in text

    space = DOCUMENTS[3].read_text(encoding="utf-8")
    assert "十一路线" in space
    assert "MuScriptor" in space

    combined = "\n".join(path.read_text(encoding="utf-8") for path in DOCUMENTS)
    for stale_claim in (
        "每个 stem 独立运行所选 YourMT3+ / MIROS",
        "continue to produce per-stem MIDI",
    ):
        assert stale_claim not in combined
