import hashlib

from recoalign.data.manifest import load_dataset_manifest, verify_dataset


def test_dataset_verification(tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    content = b"recoalign"
    (data_root / "annotations.json").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""schema_version: 1
name: toy
version: v1
source: local-test
license: test-only
splits:
  test: 1
files:
  - path: annotations.json
    bytes: {len(content)}
    sha256: {digest}
""",
        encoding="utf-8",
    )
    manifest = load_dataset_manifest(manifest_path)
    assert verify_dataset(data_root, manifest) == []
