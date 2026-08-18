from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .data import atomic_json, read_jsonl, sha256_file, write_jsonl


def _load_frozen_model(config: Any):
    phase0b_root = config.source_phase0b_dir.resolve()
    if str(phase0b_root) not in sys.path:
        sys.path.insert(0, str(phase0b_root))
    from phase0b.features import load_quantized_llm

    return load_quantized_llm(config)


def relation_prompt(record: dict[str, Any]) -> str:
    return (
        "\nQuestion: What is the spatial relation of the first named shape, the "
        f"{record['attribute1']} {record['object1']}, to the second named shape, the "
        f"{record['attribute2']} {record['object2']}? Answer with exactly one word: "
        "left, right, above, or below. ASSISTANT:"
    )


def _choice_token_ids(
    tokenizer: Any, choices: tuple[str, ...], answer_prefix: str
) -> tuple[int, dict[str, int]]:
    result: dict[str, int] = {}
    shared_prefix_id: int | None = None
    for word in choices:
        ids = tokenizer.encode(answer_prefix + word, add_special_tokens=False)
        if len(ids) != 2:
            raise RuntimeError(
                f"Registered answer {answer_prefix + word!r} must be shared-prefix + choice: {ids}"
            )
        if shared_prefix_id is None:
            shared_prefix_id = int(ids[0])
        elif shared_prefix_id != int(ids[0]):
            raise RuntimeError("Behavioral candidates do not share an identical answer prefix token")
        result[word] = int(ids[1])
    if len(set(result.values())) != len(result):
        raise RuntimeError("Registered behavioral choices do not map to unique tokens")
    if shared_prefix_id is None:
        raise RuntimeError("No behavioral choices registered")
    return shared_prefix_id, result


def _prompt_embeddings(
    model: Any, tokenizer: Any, config: Any, record: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    prefix_ids = tokenizer(
        config.prompt_prefix, add_special_tokens=True, return_tensors="pt"
    ).input_ids.to(device)
    suffix_ids = tokenizer(
        relation_prompt(record), add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    return embedding(prefix_ids), embedding(suffix_ids)


class AttentionReducer:
    """Retain only decision-to-vision attention statistics, never full matrices."""

    def __init__(self, model: Any, visual_start: int, visual_stop: int) -> None:
        self.visual_start = visual_start
        self.visual_stop = visual_stop
        self.values: dict[int, dict[str, float]] = {}
        self.handles = [
            layer.self_attn.register_forward_hook(self._hook(layer_index))
            for layer_index, layer in enumerate(model.model.layers, start=1)
        ]

    def _hook(self, layer_index: int):
        def hook(_module: nn.Module, _inputs: Any, output: Any):
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                return output
            visual = output[1][:, :, -1, self.visual_start : self.visual_stop].detach().float()
            mass_by_head = visual.sum(dim=-1)
            normalized = visual / (mass_by_head.unsqueeze(-1) + 1e-12)
            entropy = -(normalized * (normalized + 1e-12).log()).sum(dim=-1)
            entropy = entropy / math.log(max(2, self.visual_stop - self.visual_start))
            self.values[layer_index] = {
                "visual_attention_mass": float(mass_by_head.mean().cpu()),
                "visual_attention_mass_std_heads": float(mass_by_head.std().cpu()),
                "visual_attention_entropy": float(entropy.mean().cpu()),
            }
            return (output[0], None, *output[2:])

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _open_hidden_maps(config: Any, n: int, mode: str) -> dict[int, np.memmap]:
    return {
        layer: np.lib.format.open_memmap(
            config.features_dir / f"decision_layer_{layer:02d}.npy",
            mode=mode,
            dtype=np.float16,
            shape=(n, 4096),
        )
        for layer in config.layer_indices
    }


@torch.inference_mode()
def extract_behavior_and_hidden(
    config: Any,
    records: list[dict[str, Any]],
    source_info: dict[str, Any],
    *,
    force: bool = False,
    recompute_test: bool = False,
) -> dict[str, Any]:
    """Run the frozen model once per registered scene under the behavioral query."""

    config.features_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.features_dir / "behavior_manifest.json"
    behavior_path = config.features_dir / "behavior_cache.jsonl"
    attention_path = config.features_dir / "attention_cache.jsonl"
    expected_hidden = [
        config.features_dir / f"decision_layer_{layer:02d}.npy" for layer in config.layer_indices
    ]
    if (
        not force
        and not recompute_test
        and manifest_path.exists()
        and behavior_path.exists()
        and attention_path.exists()
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("complete")
            and manifest.get("source_index_sha256") == source_info["index_sha256"]
            and all(path.exists() for path in expected_hidden)
            and len(read_jsonl(behavior_path)) == len(records)
        ):
            return manifest

    if config.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The registered Phase 0-C run requires CUDA")
    from transformers import AutoTokenizer

    model = _load_frozen_model(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, use_fast=False)
    answer_prefix_id, choice_ids = _choice_token_ids(
        tokenizer, config.choice_words, config.answer_prefix
    )
    embedding = model.get_input_embeddings()
    token_array = np.load(Path(source_info["za_tokens_path"]), mmap_mode="r")
    if token_array.shape[:2] != (len(records), 576):
        raise RuntimeError(f"Unexpected projected-token shape: {token_array.shape}")

    progress_path = config.features_dir / "behavior_progress.json"
    behavior_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    resume = 0
    mode = "w+"
    if recompute_test:
        if not behavior_path.exists() or not all(path.exists() for path in expected_hidden):
            raise RuntimeError("Cannot recompute test rows without the completed training cache")
        existing = read_jsonl(behavior_path)
        if len(existing) < config.n_train:
            raise RuntimeError("Training behavior cache is incomplete")
        behavior_rows = existing[: config.n_train]
        attention_rows = []
        resume = config.n_train
        mode = "r+"
    elif not force and progress_path.exists() and behavior_path.exists() and attention_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("source_index_sha256") == source_info["index_sha256"]:
            behavior_rows = read_jsonl(behavior_path)
            attention_rows = read_jsonl(attention_path)
            resume = len(behavior_rows)
            if resume > 0:
                mode = "r+"
    hidden_maps = _open_hidden_maps(config, len(records), mode)

    for row_index in tqdm(
        range(resume, len(records)), initial=resume, total=len(records), desc="Phase 0-C behavior"
    ):
        record = records[row_index]
        prefix, suffix = _prompt_embeddings(model, tokenizer, config, record)
        image = torch.from_numpy(np.asarray(token_array[row_index]).copy()).to(
            device=prefix.device, dtype=prefix.dtype
        )
        visual_start = int(prefix.shape[1])
        visual_stop = visual_start + int(image.shape[0])
        answer_prefix = embedding(
            torch.tensor([[answer_prefix_id]], device=prefix.device, dtype=torch.long)
        )
        inputs_embeds = torch.cat((prefix, image.unsqueeze(0), suffix, answer_prefix), dim=1)
        output = model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        choice_logits = {
            word: float(output.logits[0, -1, token_id].detach().float().cpu())
            for word, token_id in choice_ids.items()
        }
        prediction = max(config.choice_words, key=lambda word: choice_logits[word])
        sorted_logits = sorted(choice_logits.values(), reverse=True)
        behavior_rows.append(
            {
                "row": row_index,
                "image_id": record["image_id"],
                "split": record["split"],
                "true_relation": record["relation"],
                "predicted_relation": prediction,
                "correct": int(prediction == record["relation"]),
                "margin": sorted_logits[0] - sorted_logits[1],
                **{f"logit_{word}": choice_logits[word] for word in config.choice_words},
            }
        )
        for layer in config.layer_indices:
            hidden_maps[layer][row_index] = (
                output.hidden_states[layer][0, -1].detach().float().cpu().numpy().astype(np.float16)
            )
        del output
        collect_attention = record["split"] == "test"
        if collect_attention:
            reducer = AttentionReducer(model, visual_start, visual_stop)
            try:
                attention_output = model(
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=False,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
            finally:
                reducer.close()
            if len(reducer.values) != 32:
                raise RuntimeError(
                    f"Expected attention from 32 layers, observed {len(reducer.values)}"
                )
            for layer, values in sorted(reducer.values.items()):
                attention_rows.append(
                    {
                        "row": row_index,
                        "image_id": record["image_id"],
                        "true_relation": record["relation"],
                        "predicted_relation": prediction,
                        "correct": int(prediction == record["relation"]),
                        "layer": layer,
                        **values,
                    }
                )
            del attention_output, reducer
        if (row_index + 1) % 5 == 0 or row_index + 1 == len(records):
            for array in hidden_maps.values():
                array.flush()
            write_jsonl(behavior_path, behavior_rows)
            write_jsonl(attention_path, attention_rows)
            atomic_json(
                progress_path,
                {
                    "source_index_sha256": source_info["index_sha256"],
                    "next_row": row_index + 1,
                    "n_samples": len(records),
                },
            )
        del inputs_embeds, image, prefix, suffix, answer_prefix

    manifest = {
        "backend": "frozen_llava_1_5_7b_vicuna_language_model",
        "complete": True,
        "protocol_version": config.protocol_version,
        "source_index_sha256": source_info["index_sha256"],
        "n_samples": len(records),
        "n_train": config.n_train,
        "n_test": config.n_test,
        "choice_token_ids": choice_ids,
        "answer_prefix": config.answer_prefix,
        "answer_prefix_token_id": answer_prefix_id,
        "prompt_prefix": config.prompt_prefix,
        "prompt_template": relation_prompt(records[0]),
        "behavior_definition": (
            "teacher-force the tokenizer-identical shared answer prefix, then take the argmax "
            "over the four relation-token logits at the actual choice position; behavior and "
            "hidden states always use output_attentions=False"
        ),
        "availability_definition": "held-out linear probe at the same pre-answer decision position",
        "attention_definition": (
            "separate supporting forward with output_attentions=True; mean-head attention from "
            "pre-answer decision position to 576 visual tokens"
        ),
        "attention_rows": len(attention_rows),
        "quantization": config.quantization,
        "cuda_decoder_layers": config.cuda_decoder_layers,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "behavior_cache_sha256": sha256_file(behavior_path),
        "attention_cache_sha256": sha256_file(attention_path),
    }
    atomic_json(manifest_path, manifest)
    for array in hidden_maps.values():
        array.flush()
    del model, tokenizer, token_array, hidden_maps
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def select_patch_trials(
    config: Any,
    records: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    availability_predictions: dict[int, list[str]],
) -> list[dict[str, Any]]:
    """Select failures with a same-class successful donor and a matched wrong-class control."""

    by_row = {int(row["row"]): row for row in behavior_rows}
    eligible: list[dict[str, Any]] = []
    primary_predictions = availability_predictions[config.primary_availability_layer]
    for receiver_index, record in enumerate(records):
        behavior = by_row[receiver_index]
        if record["split"] != "test" or behavior["correct"]:
            continue
        if primary_predictions[receiver_index] != record["relation"]:
            continue
        positive = [
            index
            for index, candidate in enumerate(records)
            if index != receiver_index
            and candidate["composition"] == record["composition"]
            and by_row[index]["correct"]
        ]
        control = [
            index
            for index, candidate in enumerate(records)
            if candidate["object_label"] == record["object_label"]
            and candidate["attribute_label"] == record["attribute_label"]
            and candidate["relation"] == behavior["predicted_relation"]
            and by_row[index]["correct"]
        ]
        if not positive or not control:
            continue
        key = lambda index: (records[index]["split"] != "train", records[index]["image_id"])
        eligible.append(
            {
                "receiver_row": receiver_index,
                "positive_donor_row": sorted(positive, key=key)[0],
                "control_donor_row": sorted(control, key=key)[0],
                "true_relation": record["relation"],
                "baseline_prediction": behavior["predicted_relation"],
            }
        )

    # Round-robin the four true relations so the causal subset cannot be dominated by one class.
    selected: list[dict[str, Any]] = []
    queues = {
        relation: [row for row in eligible if row["true_relation"] == relation]
        for relation in config.choice_words
    }
    while len(selected) < config.max_patch_failures and any(queues.values()):
        for relation in config.choice_words:
            if queues[relation] and len(selected) < config.max_patch_failures:
                selected.append(queues[relation].pop(0))
    return selected


def _patch_hook(donor: np.ndarray, alpha: float):
    def hook(_module: nn.Module, _inputs: Any, output: Any):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        donor_tensor = torch.as_tensor(donor, device=hidden.device, dtype=hidden.dtype)
        patched[:, -1, :] = (1.0 - alpha) * hidden[:, -1, :] + alpha * donor_tensor
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    return hook


@torch.inference_mode()
def run_activation_patching(
    config: Any,
    records: list[dict[str, Any]],
    source_info: dict[str, Any],
    behavior_rows: list[dict[str, Any]],
    availability_predictions: dict[int, list[str]],
    *,
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_path = config.features_dir / "activation_patch_cache.jsonl"
    manifest_path = config.features_dir / "activation_patch_manifest.json"
    behavior_cache_sha256 = sha256_file(config.features_dir / "behavior_cache.jsonl")
    if not force and cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("complete")
            and manifest.get("source_index_sha256") == source_info["index_sha256"]
            and manifest.get("protocol_version") == config.protocol_version
            and manifest.get("behavior_cache_sha256") == behavior_cache_sha256
        ):
            return read_jsonl(cache_path), manifest

    trials = select_patch_trials(
        config, records, behavior_rows, availability_predictions
    )
    by_row = {int(row["row"]): row for row in behavior_rows}
    if not trials:
        write_jsonl(cache_path, [])
        manifest = {
            "complete": True,
            "source_index_sha256": source_info["index_sha256"],
            "eligible_trials": 0,
            "selected_trials": 0,
            "reason": "No behavioral failure had both registered donors and a correct final probe.",
        }
        atomic_json(manifest_path, manifest)
        return [], manifest

    from transformers import AutoTokenizer

    model = _load_frozen_model(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, use_fast=False)
    answer_prefix_id, choice_ids = _choice_token_ids(
        tokenizer, config.choice_words, config.answer_prefix
    )
    embedding = model.get_input_embeddings()
    token_array = np.load(Path(source_info["za_tokens_path"]), mmap_mode="r")
    hidden_maps = {
        layer: np.load(config.features_dir / f"decision_layer_{layer:02d}.npy", mmap_mode="r")
        for layer in config.patch_layers
    }
    conditions: list[tuple[int, float, str]] = []
    for layer in config.patch_layers:
        conditions.append((layer, 0.0, "baseline"))
        conditions.append((layer, 1.0, "positive"))
        conditions.append((layer, 1.0, "control"))
    for alpha in config.patch_alphas[1:-1]:
        conditions.append((config.primary_patch_layer, alpha, "positive"))
        conditions.append((config.primary_patch_layer, alpha, "control"))

    patch_rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    if not force and cache_path.exists():
        existing = read_jsonl(cache_path)
        counts: dict[str, int] = {}
        for row in existing:
            counts[row["image_id"]] = counts.get(row["image_id"], 0) + 1
        completed = {
            image_id for image_id, count in counts.items() if count == len(conditions)
        }
        patch_rows = [row for row in existing if row["image_id"] in completed]
    pending_trials = [
        trial
        for trial in trials
        if records[int(trial["receiver_row"])]["image_id"] not in completed
    ]

    for trial in tqdm(
        pending_trials,
        initial=len(completed),
        total=len(trials),
        desc="Activation patch",
    ):
        receiver = int(trial["receiver_row"])
        record = records[receiver]
        prefix, suffix = _prompt_embeddings(model, tokenizer, config, record)
        image = torch.from_numpy(np.asarray(token_array[receiver]).copy()).to(
            device=prefix.device, dtype=prefix.dtype
        )
        answer_prefix = embedding(
            torch.tensor([[answer_prefix_id]], device=prefix.device, dtype=torch.long)
        )
        inputs_embeds = torch.cat((prefix, image.unsqueeze(0), suffix, answer_prefix), dim=1)
        for layer, alpha, patch_type in conditions:
            donor_row: int | None = None
            if patch_type == "positive":
                donor_row = int(trial["positive_donor_row"])
            elif patch_type == "control":
                donor_row = int(trial["control_donor_row"])
            handle = None
            if donor_row is not None and alpha > 0:
                donor = np.asarray(hidden_maps[layer][donor_row]).copy()
                handle = model.model.layers[layer - 1].register_forward_hook(
                    _patch_hook(donor, alpha)
                )
            try:
                output = model(
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=False,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
            finally:
                if handle is not None:
                    handle.remove()
            logits = {
                word: float(output.logits[0, -1, token_id].detach().float().cpu())
                for word, token_id in choice_ids.items()
            }
            prediction = max(config.choice_words, key=lambda word: logits[word])
            patch_rows.append(
                {
                    "image_id": record["image_id"],
                    "receiver_row": receiver,
                    "true_relation": record["relation"],
                    "baseline_prediction": by_row[receiver]["predicted_relation"],
                    "availability_prediction": availability_predictions[
                        config.primary_availability_layer
                    ][receiver],
                    "layer": layer,
                    "alpha": alpha,
                    "patch_type": patch_type,
                    "donor_image_id": records[donor_row]["image_id"] if donor_row is not None else "",
                    "donor_relation": records[donor_row]["relation"] if donor_row is not None else "",
                    "patched_prediction": prediction,
                    "patched_correct": int(prediction == record["relation"]),
                    "target_logit_margin": logits[record["relation"]]
                    - max(value for word, value in logits.items() if word != record["relation"]),
                    **{f"logit_{word}": logits[word] for word in config.choice_words},
                }
            )
            del output
        write_jsonl(cache_path, patch_rows)
        del inputs_embeds, image, prefix, suffix, answer_prefix
        gc.collect()
        torch.cuda.empty_cache()

    baseline_rows = [row for row in patch_rows if row["patch_type"] == "baseline"]
    baseline_mismatches = sum(
        row["patched_prediction"] != row["baseline_prediction"] for row in baseline_rows
    )
    manifest = {
        "complete": True,
        "source_index_sha256": source_info["index_sha256"],
        "protocol_version": config.protocol_version,
        "behavior_cache_sha256": behavior_cache_sha256,
        "selected_trials": len(trials),
        "max_patch_failures": config.max_patch_failures,
        "patch_layers": list(config.patch_layers),
        "patch_alphas": list(config.patch_alphas),
        "primary_test": {
            "layer": config.primary_patch_layer,
            "alpha": config.primary_patch_alpha,
            "contrast": "positive same-composition successful donor versus matched donor whose relation equals the receiver's wrong baseline prediction",
        },
        "patch_position": "pre-answer decision token after the named decoder block",
        "patch_rule": "h_receiver <- (1-alpha)*h_receiver + alpha*h_donor",
        "baseline_rows": len(baseline_rows),
        "baseline_mismatches": baseline_mismatches,
        "baseline_reproduction_required": True,
        "quantization": config.quantization,
        "cache_sha256": sha256_file(cache_path),
    }
    atomic_json(manifest_path, manifest)
    del model, tokenizer, token_array, hidden_maps
    gc.collect()
    torch.cuda.empty_cache()
    return patch_rows, manifest
