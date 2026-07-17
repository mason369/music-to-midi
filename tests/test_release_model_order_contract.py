from pathlib import Path


def test_release_restores_miros_before_full_aggregate_model_validation():
    workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    restore_marker = "Restore and strictly verify pinned MIROS before the aggregate downloader"
    aggregate_marker = "python download_sota_models.py"
    restore_index = workflow.index(restore_marker)
    aggregate_index = workflow.index(aggregate_marker)

    assert restore_index < aggregate_index
    assert workflow.index("python download_miros_model.py", restore_index) < aggregate_index
    assert "rm -rf \"$MIROS_PORTABLE_ROOT\" \"$WORKSPACE_DIR/external/ai4m-miros\"" not in workflow[
        aggregate_index:
    ]
