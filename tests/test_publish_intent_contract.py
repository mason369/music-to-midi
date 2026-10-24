"""Source-only pushes retain CI/build checks without publishing artifacts."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GUARD = (
    "${{ inputs.release_tag != '' || !contains(github.event.head_commit.message, '[no publish]') }}"
)


def test_source_only_push_guards_every_external_publication():
    workflow = yaml.safe_load((ROOT / ".github/workflows/container.yml").read_text("utf-8"))
    jobs = workflow["jobs"]
    assert "if" not in jobs["contract"]
    assert "if" not in jobs["publish"]
    steps = jobs["publish"]["steps"]
    for name in ("登录 GHCR", "推送带 SBOM 与 provenance 的镜像", "生成注册表证明"):
        assert next(step for step in steps if step["name"] == name)["if"] == GUARD
    for name in ("构建本地验收镜像", "镜像级烟雾验收"):
        assert "if" not in next(step for step in steps if step["name"] == name)
    space = yaml.safe_load((ROOT / ".github/workflows/sync_to_hf.yml").read_text("utf-8"))
    assert space["jobs"]["sync"]["if"] == GUARD
    ci = yaml.safe_load((ROOT / ".github/workflows/build.yml").read_text("utf-8"))
    assert "[no publish]" not in str(ci)
    # Only the version-checked release caller can explicitly publish a no-publish commit.
    assert workflow[True]["workflow_call"]["inputs"]["release_tag"]["required"] is True
    assert "tags" not in workflow[True]["push"]
    assert "inputs.release_tag" in jobs["distribution"]["if"]
