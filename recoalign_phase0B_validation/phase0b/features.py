from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from .config import ExperimentConfig
from .data import sha256_file

VISION_REPO = "openai/clip-vit-large-patch14-336"
LLAVA_REPO = "liuhaotian/llava-v1.5-7b"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve_device(config: ExperimentConfig) -> torch.device:
    if config.device != "cuda":
        raise RuntimeError("The registered Phase 0-B run requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA PyTorch is unavailable. Use the pinned CUDA environment; do not substitute "
            "a smaller language model for the registered LLaVA-1.5-7B experiment."
        )
    return torch.device("cuda")


def _load_projector(
    *, device: torch.device, dtype: torch.dtype
) -> tuple[nn.Sequential, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    config_path = Path(hf_hub_download(LLAVA_REPO, "config.json"))
    raw = _json(config_path)
    if raw.get("mm_projector_type") != "mlp2x_gelu":
        raise RuntimeError("Expected the official LLaVA-1.5 mlp2x_gelu projector")
    projector = nn.Sequential(
        nn.Linear(int(raw["mm_hidden_size"]), int(raw["hidden_size"])),
        nn.GELU(),
        nn.Linear(int(raw["hidden_size"]), int(raw["hidden_size"])),
    )
    weight_path = Path(
        hf_hub_download(LLAVA_REPO, "mm_projector.bin", revision=config_path.parent.name)
    )
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    prefix = "model.mm_projector."
    state = {
        key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)
    }
    projector.load_state_dict(state, strict=True)
    projector.eval().requires_grad_(False).to(device=device, dtype=dtype)
    return projector, {
        "llava_repo": LLAVA_REPO,
        "llava_commit": config_path.parent.name,
        "projector_sha256": sha256_file(weight_path),
        "vision_select_layer": int(raw["mm_vision_select_layer"]),
        "vision_select_feature": raw["mm_vision_select_feature"],
        "hidden_size": int(raw["hidden_size"]),
    }


def _phase0_reference_means(config: ExperimentConfig, records: list[dict[str, Any]]) -> np.ndarray:
    index_path = config.source_phase0_dir / "features" / "main" / "index.jsonl"
    feature_path = config.source_phase0_dir / "features" / "main" / "za" / "features.npy"
    with index_path.open("r", encoding="utf-8") as handle:
        old_index = {
            json.loads(line)["image_id"]: row for row, line in enumerate(handle) if line.strip()
        }
    old_features = np.load(feature_path, mmap_mode="r")
    return np.asarray([old_features[old_index[row["image_id"]]] for row in records])


@torch.inference_mode()
def extract_projected_tokens(
    config: ExperimentConfig, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Capture all 576 official projected patch tokens before loading the 7B LLM."""

    config.features_dir.mkdir(parents=True, exist_ok=True)
    index_hash = sha256_file(config.features_dir / "index.jsonl")
    token_path = config.features_dir / "za_tokens.npy"
    mean_path = config.features_dir / "za_mean.npy"
    manifest_path = config.features_dir / "vision_manifest.json"
    if (
        not config.force_vision
        and manifest_path.exists()
        and token_path.exists()
        and mean_path.exists()
    ):
        manifest = _json(manifest_path)
        if manifest.get("index_sha256") == index_hash and manifest.get("complete"):
            return manifest

    device = _resolve_device(config)
    from transformers import CLIPImageProcessor, CLIPVisionModel

    dtype = torch.float16
    processor = CLIPImageProcessor.from_pretrained(VISION_REPO)
    vision = CLIPVisionModel.from_pretrained(VISION_REPO, torch_dtype=dtype)
    vision.eval().requires_grad_(False).to(device)
    projector, identity = _load_projector(device=device, dtype=dtype)
    select_layer = int(identity["vision_select_layer"])
    n = len(records)
    token_count = (int(vision.config.image_size) // int(vision.config.patch_size)) ** 2
    hidden_size = int(identity["hidden_size"])
    tokens = np.lib.format.open_memmap(
        token_path, mode="w+", dtype=np.float16, shape=(n, token_count, hidden_size)
    )
    means = np.lib.format.open_memmap(
        mean_path, mode="w+", dtype=np.float32, shape=(n, hidden_size)
    )

    for start in tqdm(range(0, n, config.vision_batch_size), desc="Project visual tokens"):
        batch_records = records[start : start + config.vision_batch_size]
        images = [Image.open(row["absolute_image_path"]).convert("RGB") for row in batch_records]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(
            device=device, dtype=dtype
        )
        output = vision(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        selected = output.hidden_states[select_layer][:, 1:, :]
        projected = projector(selected)
        array = projected.float().cpu().numpy()
        stop = start + len(batch_records)
        tokens[start:stop] = array.astype(np.float16)
        means[start:stop] = array.mean(axis=1)
        for image in images:
            image.close()
    tokens.flush()
    means.flush()

    reference = _phase0_reference_means(config, records)
    observed = np.asarray(np.load(mean_path, mmap_mode="r"), dtype=np.float32)
    cosine = np.sum(reference * observed, axis=1) / (
        np.linalg.norm(reference, axis=1) * np.linalg.norm(observed, axis=1) + 1e-12
    )
    identity.update(
        {
            "backend": "official_llava_1_5_7b_vision_projector",
            "vision_repo": VISION_REPO,
            "vision_model_type": type(vision).__name__,
            "device": str(device),
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "token_count": token_count,
            "index_sha256": index_hash,
            "n_samples": n,
            "phase0_mean_cosine_min": float(cosine.min()),
            "phase0_mean_cosine_mean": float(cosine.mean()),
            "phase0_mean_max_abs_difference": float(np.max(np.abs(reference - observed))),
            "complete": True,
        }
    )
    _atomic_json(manifest_path, identity)
    del tokens, means, projected, output, vision, projector
    gc.collect()
    torch.cuda.empty_cache()
    return identity


def _llama_config(model_dir: Path):
    from transformers import LlamaConfig

    raw = _json(model_dir / "config.json")
    allowed = {
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_act",
        "max_position_embeddings",
        "initializer_range",
        "rms_norm_eps",
        "use_cache",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "pretraining_tp",
        "tie_word_embeddings",
        "rope_scaling",
        "attention_bias",
        "attention_dropout",
    }
    values = {key: raw[key] for key in allowed if key in raw}
    values["use_cache"] = False
    return LlamaConfig(**values)


def _checkpoint_identity(model_dir: Path) -> dict[str, Any]:
    tree_files = sorted((model_dir / ".cache" / "huggingface" / "trees").glob("*.json"))
    if not tree_files:
        raise RuntimeError("Hugging Face source-tree metadata is missing from model_dir")
    tree = _json(tree_files[-1])
    shard_names = ("pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin")
    shards: dict[str, Any] = {}
    for name in shard_names:
        metadata = tree["files"][name]
        path = model_dir / name
        observed_size = path.stat().st_size if path.exists() else 0
        expected_size = int(metadata["lfs_size"])
        if observed_size != expected_size:
            raise RuntimeError(
                f"Incomplete checkpoint shard {name}: {observed_size} != {expected_size} bytes"
            )
        shards[name] = {
            "size": observed_size,
            "upstream_lfs_sha256": metadata["lfs_sha256"],
        }
    return {
        "repo": LLAVA_REPO,
        "commit": tree_files[-1].stem,
        "tree_metadata_sha256": sha256_file(tree_files[-1]),
        "shards": shards,
    }


def load_quantized_llm(config: ExperimentConfig):
    from transformers import BitsAndBytesConfig, LlamaForCausalLM

    if not config.model_dir.exists():
        raise FileNotFoundError(
            f"Full LLaVA checkpoint not found at {config.model_dir}. Run download_model.py."
        )
    _checkpoint_identity(config.model_dir)
    llama_config = _llama_config(config.model_dir)
    kwargs: dict[str, Any] = {
        "config": llama_config,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if config.quantization == "nf4":
        device_map: dict[str, int | str] = {
            "model.embed_tokens": 0,
            "model.rotary_emb": 0,
            "model.norm": "cpu",
            "lm_head": "cpu",
        }
        for layer in range(32):
            device_map[f"model.layers.{layer}"] = (
                0 if layer < config.cuda_decoder_layers else "cpu"
            )
        kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                ),
                "device_map": device_map,
            }
        )
    model = LlamaForCausalLM.from_pretrained(config.model_dir, **kwargs)
    model.eval().requires_grad_(False)
    return model


def _embed_prompt(
    model: Any, tokenizer: Any, config: ExperimentConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    prefix_ids = tokenizer(
        config.prompt_prefix, add_special_tokens=True, return_tensors="pt"
    ).input_ids.to(device)
    suffix_ids = tokenizer(
        config.prompt_suffix, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    return embedding(prefix_ids), embedding(suffix_ids)


class _AttentionReducer:
    def __init__(self, model: Any, visual_start: int, visual_stop: int) -> None:
        self.visual_start = visual_start
        self.visual_stop = visual_stop
        self.values: dict[int, dict[str, float]] = {}
        self.handles = []
        for layer_index, layer in enumerate(model.model.layers, start=1):
            self.handles.append(layer.self_attn.register_forward_hook(self._hook(layer_index)))

    def _hook(self, layer_index: int):
        def hook(_module: nn.Module, _inputs: Any, output: Any):
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                return output
            # Slice the one registered query before casting; materializing an
            # fp32 copy of the full 32-head O(sequence^2) matrix can exceed the
            # small runtime memory margin on a 6GB GPU.
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
            # The reducer has retained the only statistics needed for this experiment.
            # Removing the full matrix here prevents 32 O(sequence^2) tensors from
            # being retained in the model output.
            return (output[0], None, *output[2:])

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _open_feature_maps(
    config: ExperimentConfig, n: int, mode: str
) -> tuple[dict[int, np.memmap], dict[int, np.memmap]]:
    visual: dict[int, np.memmap] = {}
    decision: dict[int, np.memmap] = {}
    for layer in config.layer_indices:
        visual[layer] = np.lib.format.open_memmap(
            config.features_dir / f"visual_layer_{layer:02d}.npy",
            mode=mode,
            dtype=np.float16,
            shape=(n, 4096),
        )
        decision[layer] = np.lib.format.open_memmap(
            config.features_dir / f"decision_layer_{layer:02d}.npy",
            mode=mode,
            dtype=np.float16,
            shape=(n, 4096),
        )
    return visual, decision


@torch.inference_mode()
def extract_llm_features(config: ExperimentConfig, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run frozen projected tokens through the unmodified 32-layer Vicuna LLM."""

    device = _resolve_device(config)
    del device  # accelerate owns the exact layer placement
    index_hash = sha256_file(config.features_dir / "index.jsonl")
    manifest_path = config.features_dir / "llm_manifest.json"
    if not config.force_llm and manifest_path.exists():
        manifest = _json(manifest_path)
        expected = [
            config.features_dir / f"{kind}_layer_{layer:02d}.npy"
            for kind in ("visual", "decision")
            for layer in config.layer_indices
        ]
        if (
            manifest.get("complete")
            and manifest.get("index_sha256") == index_hash
            and all(path.exists() for path in expected)
        ):
            return manifest

    from transformers import AutoTokenizer

    model = load_quantized_llm(config)
    # The original LLaVA/Vicuna checkpoint ships a SentencePiece model and no
    # tokenizer.json. The slow tokenizer is therefore the exact, non-converted path.
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, use_fast=False)
    prefix, suffix = _embed_prompt(model, tokenizer, config)
    visual_start = int(prefix.shape[1])
    tokens = np.load(config.features_dir / "za_tokens.npy", mmap_mode="r")
    n = len(records)
    progress_path = config.features_dir / "llm_progress.json"
    resume = 0
    mode = "w+"
    if not config.force_llm and progress_path.exists():
        progress = _json(progress_path)
        if progress.get("index_sha256") == index_hash:
            resume = int(progress.get("next_row", 0))
            if resume > 0:
                mode = "r+"
    visual_maps, decision_maps = _open_feature_maps(config, n, mode)

    model_device = prefix.device
    for row in tqdm(range(resume, n), initial=resume, total=n, desc="LLM hidden states"):
        image = torch.from_numpy(np.asarray(tokens[row]).copy()).to(
            device=model_device, dtype=prefix.dtype
        )
        inputs_embeds = torch.cat((prefix, image.unsqueeze(0), suffix), dim=1)
        output = model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        visual_stop = visual_start + image.shape[0]
        for layer in config.layer_indices:
            hidden = output.hidden_states[layer].detach().float()
            visual_maps[layer][row] = (
                hidden[:, visual_start:visual_stop].mean(dim=1).cpu().numpy()[0].astype(np.float16)
            )
            decision_maps[layer][row] = hidden[:, -1].cpu().numpy()[0].astype(np.float16)
        if (row + 1) % 5 == 0 or row + 1 == n:
            for array in (*visual_maps.values(), *decision_maps.values()):
                array.flush()
            _atomic_json(
                progress_path,
                {"index_sha256": index_hash, "next_row": row + 1, "n_samples": n},
            )
        del output, inputs_embeds, image

    attention_rows: list[dict[str, Any]] = []
    per_relation = config.attention_samples // 4
    attention_indices: list[int] = []
    for relation in ("left", "right", "above", "below"):
        candidates = [
            index
            for index, record in enumerate(records)
            if record["split"] == "test" and record["relation"] == relation
        ]
        attention_indices.extend(candidates[:per_relation])
    for row in tqdm(attention_indices, desc="Attention audit"):
        image = torch.from_numpy(np.asarray(tokens[row]).copy()).to(
            device=model_device, dtype=prefix.dtype
        )
        inputs_embeds = torch.cat((prefix, image.unsqueeze(0), suffix), dim=1)
        reducer = _AttentionReducer(
            model, visual_start=visual_start, visual_stop=visual_start + image.shape[0]
        )
        try:
            model(
                inputs_embeds=inputs_embeds,
                output_hidden_states=False,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
        finally:
            reducer.close()
        for layer, values in reducer.values.items():
            attention_rows.append(
                {
                    "image_id": records[row]["image_id"],
                    "relation": records[row]["relation"],
                    "layer": layer,
                    **values,
                }
            )
        del inputs_embeds, image, reducer

    attention_path = config.features_dir / "attention.jsonl"
    with attention_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in attention_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    first_map = (
        next(iter(model.hf_device_map.values())) if hasattr(model, "hf_device_map") else "cuda"
    )
    manifest = {
        "backend": "frozen_llava_1_5_7b_vicuna_language_model",
        "model_dir": str(config.model_dir.resolve()),
        "checkpoint": _checkpoint_identity(config.model_dir),
        "quantization": config.quantization,
        "quantization_note": (
            "NF4 is a storage/inference approximation only; no parameter is updated. "
            "A positive result requires replication in higher precision."
        ),
        "device_map": getattr(model, "hf_device_map", {"": str(first_map)}),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "bitsandbytes_version": __import__("bitsandbytes").__version__,
        "cuda_decoder_layers": config.cuda_decoder_layers,
        "n_samples": n,
        "layers": list(config.layer_indices),
        "hidden_state_index_definition": (
            "0=LLM input embedding (Za at visual positions); k=state after k decoder blocks; "
            "32=post-final-norm state"
        ),
        "visual_pooling": "mean over the unchanged 576 visual-token positions",
        "decision_position": "last ASSISTANT prompt token before generation",
        "attention_definition": (
            "mean-head attention mass from decision position to 576 visual tokens"
        ),
        "prompt_prefix": config.prompt_prefix,
        "prompt_suffix": config.prompt_suffix,
        "index_sha256": index_hash,
        "attention_rows": len(attention_rows),
        "complete": True,
    }
    _atomic_json(manifest_path, manifest)
    for array in (*visual_maps.values(), *decision_maps.values()):
        array.flush()
    del model, tokenizer, prefix, suffix, tokens, visual_maps, decision_maps
    gc.collect()
    torch.cuda.empty_cache()
    return manifest
