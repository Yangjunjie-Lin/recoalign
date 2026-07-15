"""Prepare reproducible local snapshots for baseline benchmarks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from recoalign.data.manifest import sha256_file

SUGARCREPE_CATEGORIES = (
    "add_att",
    "add_obj",
    "replace_att",
    "replace_obj",
    "replace_rel",
    "swap_att",
    "swap_obj",
)


def prepare_flickr30k(
    karpathy_json: str | Path,
    dataset_root: str | Path,
    *,
    manifest_output: str | Path,
    source: str,
    license_name: str,
    hash_images: bool = False,
) -> dict[str, Any]:
    """Normalize the Karpathy Flickr30K split and write a verifiable manifest."""
    return _prepare_retrieval_karpathy(
        karpathy_json,
        dataset_root,
        manifest_output=manifest_output,
        source=source,
        license_name=license_name,
        dataset_name="flickr30k",
        source_filename="dataset_flickr30k.json",
        nested_image_paths=False,
        notes="Images are expected under images/. Captions follow the Karpathy split.",
        hash_images=hash_images,
    )


def prepare_coco(
    karpathy_json: str | Path,
    dataset_root: str | Path,
    *,
    manifest_output: str | Path,
    source: str,
    license_name: str,
    hash_images: bool = False,
) -> dict[str, Any]:
    """Normalize the MS COCO Karpathy split and write a verifiable manifest."""
    return _prepare_retrieval_karpathy(
        karpathy_json,
        dataset_root,
        manifest_output=manifest_output,
        source=source,
        license_name=license_name,
        dataset_name="mscoco",
        source_filename="dataset_coco.json",
        nested_image_paths=True,
        notes="Images are expected under images/train2014 and images/val2014.",
        hash_images=hash_images,
    )


def _prepare_retrieval_karpathy(
    karpathy_json: str | Path,
    dataset_root: str | Path,
    *,
    manifest_output: str | Path,
    source: str,
    license_name: str,
    dataset_name: str,
    source_filename: str,
    nested_image_paths: bool,
    notes: str,
    hash_images: bool,
) -> dict[str, Any]:
    source = _nonempty_argument(source, "source")
    license_name = _nonempty_argument(license_name, "license_name")
    root = Path(dataset_root)
    image_root = root / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"{dataset_name} image directory does not exist: {image_root}")

    source_dir = root / "source"
    annotation_dir = root / "annotations"
    inventory_dir = root / "inventories"
    source_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    inventory_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / source_filename
    _copy_source(karpathy_json, source_path)
    payload = _load_json(source_path)
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Karpathy JSON must contain a non-empty images list")

    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    inventory_paths: set[Path] = set()
    for index, image in enumerate(images, start=1):
        if not isinstance(image, dict):
            raise ValueError(f"Karpathy image entry {index} must be an object")
        split = _normalize_karpathy_split(image.get("split"))
        if split is None:
            continue
        filename = _required_text(image, "filename", f"Karpathy image entry {index}")
        relative_image = Path(filename)
        if nested_image_paths:
            filepath = image.get("filepath")
            if filepath is not None and (
                not isinstance(filepath, str) or not filepath.strip()
            ):
                raise ValueError(f"Karpathy image entry {index}: filepath must be a string")
            if filepath:
                relative_image = Path(filepath.strip()) / filename
        if relative_image.is_absolute() or ".." in relative_image.parts:
            raise ValueError(f"Karpathy image entry {index}: unsafe image path")

        sentences = image.get("sentences")
        if not isinstance(sentences, list) or not sentences:
            raise ValueError(f"Karpathy image entry {index} has no captions")
        captions: list[str] = []
        for sentence_index, sentence in enumerate(sentences, start=1):
            if not isinstance(sentence, dict):
                raise ValueError(
                    f"Karpathy image entry {index} sentence {sentence_index} must be an object"
                )
            captions.append(
                _required_text(
                    sentence,
                    "raw",
                    f"Karpathy image entry {index} sentence {sentence_index}",
                )
            )

        image_path = image_root / relative_image
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{dataset_name} image referenced by annotations is missing: {image_path}"
            )
        if split == "test":
            inventory_paths.add(Path("images") / relative_image)
        image_id = str(
            image.get("cocoid", image.get("imgid", Path(filename).stem))
        )
        split_rows[split].append(
            {
                "image_id": image_id,
                "image": relative_image.as_posix(),
                "captions": captions,
            }
        )

    files = [_manifest_entry(source_path, root)]
    for split, rows in split_rows.items():
        output = annotation_dir / f"{split}.jsonl"
        _write_jsonl(output, rows)
        files.append(_manifest_entry(output, root))

    inventory_path = inventory_dir / "images.jsonl"
    _write_inventory(root, sorted(inventory_paths), inventory_path, hash_images=hash_images)
    files.append(_manifest_entry(inventory_path, root))

    manifest = {
        "schema_version": 1,
        "name": dataset_name,
        "version": "karpathy-split-v1",
        "source": source,
        "license": license_name,
        "downloaded_at": None,
        "splits": {name: len(rows) for name, rows in split_rows.items()},
        "files": files,
        "processing": {
            "format": "recoalign-retrieval-jsonl-v1",
            "source_annotation": f"source/{source_filename}",
            "image_inventory": "inventories/images.jsonl",
            "inventory_splits": ["test"],
            "image_hashes": hash_images,
        },
        "notes": notes,
    }
    _write_manifest(manifest_output, manifest)
    return manifest


def prepare_sugarcrepe(
    official_data_dir: str | Path,
    dataset_root: str | Path,
    *,
    manifest_output: str | Path,
    source: str,
    license_name: str,
    hash_images: bool = False,
) -> dict[str, Any]:
    """Normalize the seven official SugarCrepe categories into one JSONL file."""
    source = _nonempty_argument(source, "source")
    license_name = _nonempty_argument(license_name, "license_name")
    root = Path(dataset_root)
    image_root = root / "images" / "val2017"
    if not image_root.is_dir():
        raise FileNotFoundError(
            f"SugarCrepe expects COCO-2017 validation images under {image_root}"
        )

    source_dir = root / "source"
    annotation_dir = root / "annotations"
    inventory_dir = root / "inventories"
    source_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    inventory_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    inventory_paths: set[Path] = set()
    files: list[dict[str, Any]] = []
    official_root = Path(official_data_dir)
    category_counts: dict[str, int] = {}
    for category in SUGARCREPE_CATEGORIES:
        source_input = official_root / f"{category}.json"
        source_copy = source_dir / f"{category}.json"
        _copy_source(source_input, source_copy)
        files.append(_manifest_entry(source_copy, root))
        payload = _load_json(source_copy)
        if not isinstance(payload, dict):
            raise ValueError(f"SugarCrepe {category} file must contain a JSON object")
        count = 0
        for key in sorted(payload, key=_numeric_sort_key):
            sample = payload[key]
            if not isinstance(sample, dict):
                raise ValueError(f"SugarCrepe {category}:{key} must be an object")
            filename = _required_text(sample, "filename", f"SugarCrepe {category}:{key}")
            image_path = image_root / filename
            if not image_path.is_file():
                raise FileNotFoundError(f"SugarCrepe image is missing: {image_path}")
            inventory_paths.add(Path("images") / "val2017" / filename)
            rows.append(
                {
                    "sample_id": f"{category}:{key}",
                    "image": filename,
                    "positive_caption": _required_text(
                        sample, "caption", f"SugarCrepe {category}:{key}"
                    ),
                    "negative_caption": _required_text(
                        sample, "negative_caption", f"SugarCrepe {category}:{key}"
                    ),
                    "category": category,
                }
            )
            count += 1
        category_counts[category] = count

    annotation_path = annotation_dir / "test.jsonl"
    _write_jsonl(annotation_path, rows)
    files.append(_manifest_entry(annotation_path, root))
    inventory_path = inventory_dir / "images.jsonl"
    _write_inventory(root, sorted(inventory_paths), inventory_path, hash_images=hash_images)
    files.append(_manifest_entry(inventory_path, root))

    manifest = {
        "schema_version": 1,
        "name": "sugarcrepe",
        "version": "official-neurips-2023-v1",
        "source": source,
        "license": license_name,
        "downloaded_at": None,
        "splits": {"test": len(rows)},
        "files": files,
        "processing": {
            "format": "recoalign-pairwise-jsonl-v1",
            "categories": category_counts,
            "image_inventory": "inventories/images.jsonl",
            "inventory_splits": ["test"],
            "image_hashes": hash_images,
        },
        "notes": "Uses COCO-2017 validation images under images/val2017/.",
    }
    _write_manifest(manifest_output, manifest)
    return manifest


def _normalize_karpathy_split(value: Any) -> str | None:
    if value in {"train", "restval"}:
        return "train"
    if value == "val":
        return "validation"
    if value == "test":
        return "test"
    return None


def _write_inventory(
    root: Path,
    relative_paths: list[Path],
    output: Path,
    *,
    hash_images: bool,
) -> None:
    rows = []
    for relative_path in relative_paths:
        path = root / relative_path
        row: dict[str, Any] = {
            "path": relative_path.as_posix(),
            "bytes": path.stat().st_size,
        }
        if hash_images:
            row["sha256"] = sha256_file(path)
        rows.append(row)
    _write_jsonl(output, rows)


def _manifest_entry(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root)
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _copy_source(source: str | Path, destination: Path) -> None:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"source annotation does not exist: {source_path}")
    if source_path.resolve() == destination.resolve():
        return
    shutil.copyfile(source_path, destination)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)


def _required_text(mapping: dict[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: {key} must be a non-empty string")
    return value.strip()


def _numeric_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    return (int(text), text) if text.isdigit() else (2**31 - 1, text)


def _nonempty_argument(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()
