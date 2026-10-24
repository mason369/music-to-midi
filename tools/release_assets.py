"""Verify complete release assets and publish one draft after all checks pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROLES = ("Windows", "Linux", "Docker")


def validate_identity(tag: str, revision: str) -> None:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise ValueError(f"Invalid stable release tag: {tag!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Release requires an exact 40-character source commit")


def role_assets(root: Path, role: str, tag: str) -> list[Path]:
    """Reject missing volumes and unexpected reports instead of glob-uploading."""
    if role not in ROLES:
        raise ValueError(f"Unknown release role: {role}")
    prefix = f"MusicToMidi-{role}-"
    files = sorted(
        path
        for path in root.iterdir()
        if path.name.startswith(prefix) and (not path.is_dir() or path.is_symlink())
    )
    if any(not path.is_file() or path.is_symlink() or path.stat().st_size == 0 for path in files):
        raise ValueError(f"{role} assets must be nonempty regular files")
    names = {path.name for path in files}
    if role == "Docker":
        expected = {f"MusicToMidi-Docker-{tag}.zip", f"MusicToMidi-Docker-{tag}.tar.gz"}
    else:
        if role == "Windows":
            volume_prefix = "MusicToMidi-Windows-GPU-Portable.wim."
            volume_pattern = re.escape(volume_prefix) + r"([0-9]{3})"
            frontend = "MusicToMidi-Windows-WebFrontend-Portable.zip"
        else:
            volume_prefix = "MusicToMidi-Linux-GPU-Portable.tar.gz.part"
            volume_pattern = re.escape(volume_prefix) + r"([a-z]{2})"
            frontend = "MusicToMidi-Linux-WebFrontend-Portable.tar.gz"
        volumes = sorted(name for name in names if re.fullmatch(volume_pattern, name))
        if not volumes:
            raise ValueError(f"Missing {role} full portable volumes")
        if role == "Windows":
            expected_volumes = [
                f"{volume_prefix}{index:03}" for index in range(1, len(volumes) + 1)
            ]
        else:
            expected_volumes = [
                f"{volume_prefix}{chr(97 + index // 26)}{chr(97 + index % 26)}"
                for index in range(len(volumes))
            ]
        if volumes != expected_volumes:
            raise ValueError(f"Missing or misnumbered {role} volume")
        expected = set(expected_volumes) | {frontend}
    if names != expected:
        raise ValueError(
            f"{role} asset mismatch: missing={expected - names}, unexpected={names - expected}"
        )
    return files


def file_record(path: Path) -> dict:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"name": path.name, "size": path.stat().st_size, "sha256": digest}


def record_assets(root: Path, role: str, tag: str, revision: str) -> Path:
    validate_identity(tag, revision)
    payload = {
        "role": role,
        "tag": tag,
        "revision": revision,
        "assets": [file_record(path) for path in role_assets(root, role, tag)],
    }
    output = root / f"release-manifest-{role}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def verify_assets(root: Path, tag: str, revision: str) -> list[dict]:
    validate_identity(tag, revision)
    records = []
    allowed = {f"release-manifest-{role}.json" for role in ROLES} | {"SHA256SUMS.txt"}
    for role in ROLES:
        manifest = json.loads((root / f"release-manifest-{role}.json").read_text("utf-8"))
        if (manifest.get("role"), manifest.get("tag"), manifest.get("revision")) != (
            role,
            tag,
            revision,
        ):
            raise ValueError(f"{role} artifact came from another tag or source commit")
        actual = [file_record(path) for path in role_assets(root, role, tag)]
        if manifest.get("assets") != actual:
            raise ValueError(f"{role} artifact hash/size/name differs from its build manifest")
        records.extend(actual)
        allowed.update(item["name"] for item in actual)
    unexpected = {path.name for path in root.iterdir()} - allowed
    if unexpected:
        raise ValueError(f"Unexpected release files: {sorted(unexpected)}")
    records.sort(key=lambda item: item["name"])
    checksum = root / "SHA256SUMS.txt"
    checksum.write_text(
        "".join(f"{item['sha256']}  {item['name']}\n" for item in records), encoding="utf-8"
    )
    return records + [file_record(checksum)]


def gh(*arguments: str, timeout: int = 60) -> str:
    result = subprocess.run(
        ["gh", *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    return result.stdout.strip()


def read_release(repository: str, tag: str) -> dict | None:
    try:
        return json.loads(gh("api", f"repos/{repository}/releases/tags/{tag}"))
    except subprocess.CalledProcessError as error:
        if "HTTP 404" in (error.stderr or ""):
            return None
        raise


def remote_assets(repository: str, release_id: int) -> list[dict]:
    pages = json.loads(
        gh(
            "api",
            f"repos/{repository}/releases/{release_id}/assets?per_page=100",
            "--paginate",
            "--slurp",
        )
    )
    return [item for page in pages for item in page]


def matches_remote(remote: dict, expected: dict) -> bool:
    return (
        remote.get("name") == expected["name"]
        and remote.get("size") == expected["size"]
        and remote.get("digest") == f"sha256:{expected['sha256']}"
        and remote.get("state") == "uploaded"
    )


def publish_assets(root: Path, tag: str, revision: str, repository: str, notes: Path) -> None:
    records = verify_assets(root, tag, revision)
    if not notes.is_file() or not notes.read_text("utf-8").strip():
        raise ValueError("Release notes are missing or empty")
    if gh("api", f"repos/{repository}/commits/{tag}", "--jq", ".sha") != revision:
        raise ValueError("Remote tag does not point to the verified source commit")
    release = read_release(repository, tag)
    if release is None:
        gh(
            "release",
            "create",
            tag,
            "--repo",
            repository,
            "--verify-tag",
            "--draft",
            "--title",
            tag,
            "--notes-file",
            str(notes),
        )
        release = read_release(repository, tag)
    if not release or not release.get("draft"):
        raise ValueError("Refusing to overwrite an already published release")
    existing = remote_assets(repository, release["id"])
    expected_by_name = {item["name"]: item for item in records}
    if len({item["name"] for item in existing}) != len(existing):
        raise ValueError("Draft contains duplicate asset names")
    for item in existing:
        if item["name"] not in expected_by_name or not matches_remote(
            item, expected_by_name[item["name"]]
        ):
            raise ValueError(f"Draft contains an unexpected or mismatched asset: {item['name']}")
    uploaded = {item["name"] for item in existing}
    for item in records:
        if item["name"] not in uploaded:
            # A failed upload leaves the release private. Do not hide it with retries.
            gh(
                "release",
                "upload",
                tag,
                str(root / item["name"]),
                "--repo",
                repository,
                timeout=1800,
            )
    final_assets = remote_assets(repository, release["id"])
    if len(final_assets) != len(records) or {item["name"] for item in final_assets} != set(
        expected_by_name
    ):
        raise ValueError("Remote release asset set is incomplete")
    if any(not matches_remote(item, expected_by_name[item["name"]]) for item in final_assets):
        raise ValueError("Remote release asset digest/size/state verification failed")
    gh(
        "release",
        "edit",
        tag,
        "--repo",
        repository,
        "--draft=false",
        "--latest",
        "--notes-file",
        str(notes),
    )
    published = read_release(repository, tag)
    if not published or published.get("draft"):
        raise ValueError("Release did not become public")
    print(published["html_url"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record", "verify", "publish"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--repository")
    parser.add_argument("--notes", type=Path)
    args = parser.parse_args()
    if args.command == "record":
        if not args.role:
            parser.error("record requires --role")
        print(record_assets(args.directory, args.role, args.tag, args.revision))
    elif args.command == "verify":
        print(
            f"Verified {len(verify_assets(args.directory, args.tag, args.revision))} release assets"
        )
    else:
        if not args.repository or not args.notes:
            parser.error("publish requires --repository and --notes")
        publish_assets(args.directory, args.tag, args.revision, args.repository, args.notes)


if __name__ == "__main__":
    main()
