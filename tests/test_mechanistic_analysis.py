from __future__ import annotations

import json

import numpy as np
import torch

from recoalign.analysis.mechanistic import (
    FixedGraphEncoderControl,
    apply_supervision_ablation,
    attention_summary,
    evaluate_token_interventions,
    intervene_structure_tokens,
    parameter_audit,
    parameter_matched_control,
    probe_structure_tokens,
    representation_comparison,
    run_toy_mechanistic_suite,
    similarity_matrix,
    summarize_ablation_seeds,
    token_clusters,
    validate_mechanistic_registry,
)
from recoalign.models.recoalign import ReCoAlignConfig, ReCoAlignModel


def small_model() -> ReCoAlignModel:
    return ReCoAlignModel(
        ReCoAlignConfig(
            visual_dim=8,
            interface_dim=16,
            llm_dim=12,
            num_structure_tokens=4,
            num_heads=4,
            num_object_classes=2,
            num_attribute_classes=2,
            num_relation_classes=2,
            num_composition_classes=2,
            num_answer_classes=2,
        )
    )


def test_mechanistic_registry_is_explicit() -> None:
    report = validate_mechanistic_registry()
    assert report["valid"]
    assert report["ablation_count"] == 14
    assert report["no_hidden_ablation"]


def test_interventions_preserve_shape_and_do_not_mutate() -> None:
    tokens = torch.randn(3, 4, 16)
    baseline = tokens.clone()
    removed = intervene_structure_tokens(tokens, "remove")
    shuffled = intervene_structure_tokens(tokens, "shuffle", seed=4)
    replaced = intervene_structure_tokens(tokens, "replace", replacement=tokens.roll(1, 0))
    assert removed.shape == shuffled.shape == replaced.shape == tokens.shape
    assert torch.equal(tokens, baseline)
    assert torch.count_nonzero(removed) == 0
    assert not torch.equal(replaced, tokens)


def test_causal_intervention_outputs_are_auditable() -> None:
    model = small_model()
    visual = torch.randn(4, 5, 8)
    labels = torch.tensor([0, 1, 0, 1])
    result = evaluate_token_interventions(model, visual, labels, seed=5)
    assert set(result["interventions"]) == {"remove", "shuffle", "replace"}
    assert result["oracle_graph_used"] is False
    assert all("accuracy_delta" in row for row in result["interventions"].values())


def test_representation_probe_visualization_and_attention() -> None:
    rng = np.random.default_rng(3)
    visual = rng.normal(size=(8, 5, 8))
    structure = rng.normal(size=(8, 4, 16))
    labels = {
        "object": [0, 0, 1, 1, 0, 0, 1, 1],
        "relation": [0, 1, 0, 1, 0, 1, 0, 1],
    }
    probes = probe_structure_tokens(structure, labels, seed=4)
    comparison = representation_comparison(visual, structure, labels, seed=4)
    assert set(probes["probes"]) == set(labels)
    assert "mean_delta" in comparison
    matrix = similarity_matrix(structure)
    assert matrix.shape == (8, 8)
    clusters = token_clusters(structure, threshold=0.0)
    assert 1 <= clusters["cluster_count"] <= 8
    attention = attention_summary(np.ones((8, 9)), structure_token_start=5)
    assert attention["structure_attention_mass"] > 0


def test_parameter_matched_control_is_exact() -> None:
    model = small_model()
    control, audit = parameter_matched_control(model, input_dim=8, output_dim=16)
    assert audit["absolute_count_difference"] == 0
    assert parameter_audit(control)["total_parameters"] == parameter_audit(model)[
        "total_parameters"
    ]


def test_fixed_encoder_and_supervision_controls_are_explicit() -> None:
    encoder = FixedGraphEncoderControl(8, 16, num_tokens=4, seed=3)
    visual = torch.randn(2, 5, 8)
    first = encoder(visual)
    second = encoder(visual)
    assert first.shape == (2, 4, 16)
    assert torch.equal(first, second)
    batch = {
        "visual_tokens": visual,
        "object_targets": torch.ones(2, dtype=torch.long),
        "attribute_targets": torch.ones(2, dtype=torch.long),
        "relation_targets": torch.ones(2, dtype=torch.long),
        "composition_targets": torch.ones(2, dtype=torch.long),
    }
    assert apply_supervision_ablation(batch, "weak")["relation_targets"] is not None
    assert apply_supervision_ablation(batch, "none")["relation_targets"] is None


def test_paired_statistics_and_toy_suite(tmp_path) -> None:
    stats = summarize_ablation_seeds([0.9, 0.8, 1.0], [0.7, 0.8, 0.9], seed_ids=[1, 2, 3])
    assert stats["n_seeds"] == 3
    assert stats["full_minus_ablation"]["mean"] > 0
    result = run_toy_mechanistic_suite(output_dir=tmp_path / "mechanistic", seeds=(1, 2, 3))
    assert result["registered_ablation_count"] == 14
    assert result["scientific_status"] == "toy_mechanistic_validation_only"
    saved = json.loads((tmp_path / "mechanistic" / "mechanistic_suite.json").read_text())
    assert saved["oracle_graph_used"] is False
