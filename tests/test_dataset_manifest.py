from recoalign.data.manifest import load_dataset_manifest, verify_dataset


def test_dataset_verification(research_config) -> None:
    manifest = load_dataset_manifest(research_config["data"]["manifest"])
    assert verify_dataset(research_config["data"]["root"], manifest) == []
