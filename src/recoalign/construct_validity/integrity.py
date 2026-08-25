"""Immutable-parent and prospective-design integrity checks for PIVOT_EXP_A3."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

A2_FROZEN_COMMIT = "765d502dfdbdabd143945a08591b067cddbf8b3e"
A2_PREDICTIONS = Path(
    "research/causal_separation/PIVOT_EXP_A2/results/predictions.jsonl.gz"
)
A2_FROZEN_SHA256 = {
    "research/causal_separation/PIVOT_EXP_A2/preregistration.md": (
        "47862e2760da4bf8a9f046beb6c4313773f5c8fbd42cdb1a6ebbb5345b689f18"
    ),
    "research/causal_separation/PIVOT_EXP_A2/config.yaml": (
        "33d342430e9aee899a759d3fcee4ce589ec6f656da49cfa9d3ae0ff5075ad339"
    ),
    "research/causal_separation/PIVOT_EXP_A2/freeze_manifest.yaml": (
        "a85a99ca98b302b7f2aac841ea0371a328fb1c481e3276472bae1b3729a4a858"
    ),
    "research/causal_separation/PIVOT_EXP_A2/results/metrics.json": (
        "bc798cb836d20647cf0d293b1091968b38545aaa8bd41a7774ffa80b39822a68"
    ),
    "research/causal_separation/PIVOT_EXP_A2/results/predictions.jsonl.gz": (
        "5d31a46bb6bad7630d5dfd4095195aae3fe6b6b69972eadccbd95c1c846d886c"
    ),
    "research/causal_separation/PIVOT_EXP_A2/results/decision_report.yaml": (
        "99bfa89d043e278cb5c221e4046fdcf6b591577dd60b042636aee3f53a9a9172"
    ),
    "research/causal_separation/PIVOT_EXP_A2/results/artifact_manifest.yaml": (
        "e934955088b23dc39307bb6a494b060e3d4d139b89f97e2216b07c3509a2447c"
    ),
}

_KNOWN_ONTOLOGY_VALUES = {
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "circle",
    "square",
    "triangle",
    "cube",
    "sphere",
}
_ANSWER_PREFIX = re.compile(r"^\s*(?:final\s+answer|answer)\s*[:=]", re.IGNORECASE)
_OPTION_LABEL_ONLY = re.compile(r"^\s*(?:option\s*)?[A-D1-4]\s*[.)]?\s*$", re.IGNORECASE)
_REASONING_MARKERS = re.compile(r"\b(?:because|therefore|thus|reasoning|first|then)\b", re.I)
_TRUNCATION_MARKERS = ("...", "…", "<truncated>")


def validate_a2_frozen_assets(root: str | Path) -> dict[str, Any]:
    """Recompute every registered A2 boundary hash without modifying any A2 asset."""

    repository = Path(root)
    assets = {
        relative: {
            "expected_sha256": expected,
            "actual_sha256": _sha256(repository / relative),
        }
        for relative, expected in A2_FROZEN_SHA256.items()
    }
    for metadata in assets.values():
        metadata["matches"] = metadata["actual_sha256"] == metadata["expected_sha256"]
    return {
        "frozen_commit": A2_FROZEN_COMMIT,
        "passed": all(bool(metadata["matches"]) for metadata in assets.values()),
        "assets": assets,
    }


def audit_legacy_parse_failures(
    root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Describe frozen A2 ``no_match`` outputs without evaluating them again.

    The inventory copies the frozen evaluation status and adds string-occurrence flags only.
    It never changes correctness, selects a replacement answer, or writes under PIVOT_EXP_A2.
    """

    repository = Path(root)
    source = repository / A2_PREDICTIONS
    boundary = validate_a2_frozen_assets(repository)
    if not boundary["passed"]:
        raise ValueError("PIVOT_EXP_A2 frozen assets do not match the registered SHA-256 boundary")

    with gzip.open(source, "rt", encoding="utf-8") as handle:
        all_rows = [json.loads(line) for line in handle if line.strip()]
    failures = [
        row
        for row in all_rows
        if row.get("family") == "manipulation_check"
        and row.get("evaluation", {}).get("evaluation_method") == "no_match"
    ]
    if len(all_rows) != 5300 or len(failures) != 137:
        raise ValueError(
            "legacy prediction inventory differs from the frozen 5300/137 boundary: "
            f"{len(all_rows)}/{len(failures)}"
        )

    inventory = [_audit_row(row) for row in failures]
    destination = repository / output_dir
    destination.mkdir(parents=True, exist_ok=True)
    inventory_path = destination / "legacy_parse_failure_inventory.jsonl"
    with inventory_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in inventory:
            handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")

    summary = _legacy_summary(inventory, source, boundary)
    summary_path = destination / "legacy_parse_failure_summary.yaml"
    summary_path.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    audit_path = destination / "legacy_parse_failure_audit.md"
    audit_path.write_text(_render_legacy_audit(summary), encoding="utf-8", newline="\n")
    return summary


def a2_scene_ids(root: str | Path) -> set[str]:
    source = Path(root) / A2_PREDICTIONS
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        return {
            str(row["source_scene_id"])
            for row in (json.loads(line) for line in handle if line.strip())
        }


def validate_trial_inventory(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    frozen_a2_scene_ids: set[str],
    development_scene_ids: set[str],
) -> dict[str, Any]:
    validation_seeds = {int(value) for value in config["study"]["validation_seeds"]}
    validation_scene_ids = {str(row["scene_id"]) for row in rows}
    by_seed = Counter(int(row["seed"]) for row in rows)
    expected_total = int(config["design"]["trial_counts"]["total"])
    expected_per_seed = expected_total // len(validation_seeds)
    allowed_choice_ids = {"1", "2", "3", "4"}
    filler = re.compile(r"^(?:choice|description)\s+\d+$", flags=re.IGNORECASE)
    assertions = {
        "expected_trial_count": len(rows) == expected_total,
        "unique_trial_keys": len({str(row["trial_key"]) for row in rows}) == len(rows),
        "five_validation_seeds": {int(row["seed"]) for row in rows} == validation_seeds,
        "equal_trials_per_seed": all(
            by_seed[seed] == expected_per_seed for seed in validation_seeds
        ),
        "exact_four_choice_ids": all(set(row["choices"]) == allowed_choice_ids for row in rows),
        "four_unique_choice_values": all(len(set(row["choices"].values())) == 4 for row in rows),
        "correct_ids_registered": all(
            {
                str(row["correct_choice_id"]),
                str(row["scene_truth_choice_id"]),
                str(row["declared_choice_id"]),
            }
            <= allowed_choice_ids
            for row in rows
        ),
        "no_filler_choices": all(
            not filler.fullmatch(str(value).strip())
            for row in rows
            for value in row["choices"].values()
        ),
        "option_entity_namespaces_distinct": all(
            str(value) not in allowed_choice_ids
            for row in rows
            for value in row["choices"].values()
            if str(value).startswith(("E", "Entity"))
        ),
        "relation_fact_count_zero": all(
            int(row["metadata"]["relation_fact_count"]) == 0 for row in rows
        ),
        "oracle_corrupted_fact_count_equal": all(
            int(row["metadata"]["semantic_fact_count"]) == 4 for row in rows
        ),
        "oracle_facts_scene_consistent": all(
            bool(row["metadata"]["oracle_scene_facts_consistent"])
            for row in rows
            if row["evidence_truth"] == "oracle"
        ),
        "corrupted_bindings_all_false": all(
            bool(row["metadata"]["all_corrupted_bindings_false"])
            for row in rows
            if row["evidence_truth"] == "corrupted"
        ),
        "no_query_role": all(
            row["metadata"]["contains_query_role"] is False for row in rows
        ),
        "no_reasoning_chain": all(
            row["metadata"]["contains_reasoning_chain"] is False for row in rows
        ),
        "no_final_answer_field": all(
            row["metadata"]["contains_final_answer_field"] is False for row in rows
        ),
        "validation_A2_scene_disjoint": not (
            validation_scene_ids & frozen_a2_scene_ids
        ),
        "development_validation_scene_disjoint": not (
            validation_scene_ids & development_scene_ids
        ),
        "forced_prompts_end_exact_prefix": all(
            str(row["prompt"]).endswith("FINAL_CHOICE=")
            for row in rows
            if row["response_method"] == "forced_choice"
        ),
        "secondary_prompts_do_not_supply_final_field": all(
            not str(row["prompt"]).endswith("FINAL_CHOICE=")
            for row in rows
            if row["response_method"] == "free_generation"
        ),
    }
    balance = _choice_balance(rows)
    assertions["scene_truth_choice_position_balance"] = balance["scene_truth_passed"]
    assertions["declared_choice_position_balance"] = balance["declared_passed"]
    return {
        "passed": all(bool(value) for value in assertions.values()),
        "assertions": assertions,
        "prediction_design_count": len(rows),
        "by_seed": dict(sorted(by_seed.items())),
        "choice_balance": balance,
        "scene_overlap": {
            "validation_A2": sorted(validation_scene_ids & frozen_a2_scene_ids),
            "development_validation": sorted(validation_scene_ids & development_scene_ids),
        },
    }


def validate_prediction_integrity(
    rows: list[dict[str, Any]], trial_rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    expected_keys = {str(row["trial_key"]) for row in trial_rows}
    observed_keys = {str(row["trial_key"]) for row in rows}
    forced = [row for row in rows if row["response_method"] == "forced_choice"]
    secondary = [row for row in rows if row["response_method"] == "free_generation"]
    expected_forced = int(config["design"]["trial_counts"]["forced_choice_total"])
    expected_secondary = int(config["design"]["trial_counts"]["free_generation_total"])
    assertions = {
        "all_registered_trials_present": observed_keys == expected_keys,
        "no_duplicate_trial_keys": len(observed_keys) == len(rows),
        "forced_choice_count": len(forced) == expected_forced,
        "free_generation_count": len(secondary) == expected_secondary,
        "forced_choice_measurement_valid": all(
            bool(row.get("valid_measurement"))
            and str(row.get("prediction_choice_id")) in {"1", "2", "3", "4"}
            for row in forced
        ),
        "secondary_invalid_rows_retained": len(secondary) == expected_secondary,
        "five_complete_seeds": len({int(row["seed"]) for row in rows}) == 5,
    }
    return {"passed": all(assertions.values()), "assertions": assertions}


def artifact_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return {"bytes": source.stat().st_size, "sha256": _sha256(source)}


def _choice_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scene_truth_counts: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    declared_counts: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    semantic_tasks = {
        "shape",
        "color",
        "object_identity",
        "entity_attribute_binding",
    }
    for row in rows:
        if row["response_method"] != "forced_choice" or row["task"] not in semantic_tasks:
            continue
        if row["construct"] == "CV3":
            key = (
                int(row["seed"]),
                str(row["task"]),
                str(row["manipulation"]),
                str(row["evidence_truth"]),
            )
            scene_truth_counts[key][str(row["scene_truth_choice_id"])] += 1
        if row["construct"] == "CV2":
            key = (
                int(row["seed"]),
                str(row["task"]),
                str(row["manipulation"]),
                str(row["evidence_truth"]),
            )
            declared_counts[key][str(row["declared_choice_id"])] += 1
    expected = Counter({"1": 25, "2": 25, "3": 25, "4": 25})
    return {
        "expected_per_group": dict(expected),
        "scene_truth_passed": bool(scene_truth_counts)
        and all(counts == expected for counts in scene_truth_counts.values()),
        "declared_passed": bool(declared_counts)
        and all(counts == expected for counts in declared_counts.values()),
        "scene_truth_groups": {
            "|".join(str(value) for value in key): dict(sorted(counts.items()))
            for key, counts in sorted(scene_truth_counts.items())
        },
        "declared_groups": {
            "|".join(str(value) for value in key): dict(sorted(counts.items()))
            for key, counts in sorted(declared_counts.items())
        },
    }


def _audit_row(row: dict[str, Any]) -> dict[str, Any]:
    raw = str(row["prediction"])
    choices = [str(value) for value in row["choices"]]
    answer = str(row["answer"])
    normalized_raw = _normalize(raw)
    choice_occurrences = [
        choice for choice in choices if _contains(normalized_raw, _normalize(choice))
    ]
    answer_string_appears = _contains(normalized_raw, _normalize(answer))
    alias_variants = _alias_variants(row, normalized_raw)
    ontology_mentions = sorted(
        value for value in _KNOWN_ONTOLOGY_VALUES if _contains(normalized_raw, value)
    )
    empty = not raw.strip()
    truncated = empty or any(raw.rstrip().endswith(marker) for marker in _TRUNCATION_MARKERS)
    reasoning_without_final = bool(_REASONING_MARKERS.search(raw)) and not choice_occurrences
    genuinely_off_task = bool(
        not choice_occurrences
        and not alias_variants
        and not ontology_mentions
        and not _OPTION_LABEL_ONLY.fullmatch(raw)
        and not reasoning_without_final
        and not truncated
    )
    pattern = _classify_pattern(
        raw=raw,
        choice_occurrences=choice_occurrences,
        alias_variants=alias_variants,
        ontology_mentions=ontology_mentions,
        answer_string_appears=answer_string_appears,
        truncated=truncated,
        reasoning_without_final=reasoning_without_final,
        genuinely_off_task=genuinely_off_task,
    )
    frozen = row["evaluation"]
    return {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A2",
        "audit_purpose": "PIVOT_EXP_A3_measurement_design_only",
        "trial_key": row["trial_key"],
        "trial_id": row["trial_id"],
        "source_scene_id": row["source_scene_id"],
        "seed": int(row["seed"]),
        "manipulation_task": row["task"],
        "condition": row["condition"],
        "raw_output": raw,
        "registered_choices": choices,
        "registered_answer": answer,
        "frozen_evaluation": {
            "evaluation_method": frozen["evaluation_method"],
            "correct": bool(row["correct"]),
            "matched_choice": frozen["matched_choice"],
        },
        "descriptive_format_audit": {
            "raw_output_pattern": pattern,
            "allowed_choice_string_appears": bool(choice_occurrences),
            "allowed_choice_occurrences": choice_occurrences,
            "multiple_choices_appear": len(choice_occurrences) > 1,
            "alias_variants_appear": bool(alias_variants),
            "alias_variants": alias_variants,
            "registered_answer_string_appears_in_invalid_output": answer_string_appears,
            "semantically_correct_description_but_violates_grammar": (
                answer_string_appears and frozen["evaluation_method"] == "no_match"
            ),
            "genuinely_off_task": genuinely_off_task,
            "known_ontology_values_mentioned": ontology_mentions,
        },
        "immutability": {
            "rescored": False,
            "replacement_prediction_created": False,
            "frozen_evaluation_preserved": True,
        },
    }


def _classify_pattern(
    *,
    raw: str,
    choice_occurrences: list[str],
    alias_variants: list[str],
    ontology_mentions: list[str],
    answer_string_appears: bool,
    truncated: bool,
    reasoning_without_final: bool,
    genuinely_off_task: bool,
) -> str:
    if truncated:
        return "empty_or_truncated"
    if len(choice_occurrences) > 1:
        return "multiple_choices"
    if _OPTION_LABEL_ONLY.fullmatch(raw):
        return "option_label_only"
    if _ANSWER_PREFIX.search(raw):
        return "answer_prefix"
    if reasoning_without_final:
        return "reasoning_without_final_choice"
    if len(choice_occurrences) == 1 and _normalize(raw) != _normalize(choice_occurrences[0]):
        return "full_sentence_single_choice"
    if alias_variants:
        return "alias_variant"
    if answer_string_appears:
        return "description_variant"
    if ontology_mentions:
        return "hallucinated_choice"
    if genuinely_off_task:
        return "off_task"
    return "description_variant"


def _alias_variants(row: dict[str, Any], normalized_raw: str) -> list[str]:
    target = str(row.get("metadata", {}).get("target_alias", ""))
    match = re.fullmatch(r"entity\s+([a-z])", _normalize(target))
    if not match:
        return []
    letter = match.group(1)
    candidates = (letter, f"object {letter}", f"entity-{letter}")
    return [candidate for candidate in candidates if _contains(normalized_raw, candidate)]


def _legacy_summary(
    inventory: list[dict[str, Any]], source: Path, boundary: dict[str, Any]
) -> dict[str, Any]:
    def counts(field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(item[field]) for item in inventory).items()))

    audits = [item["descriptive_format_audit"] for item in inventory]
    cross = Counter(
        (
            str(item["manipulation_task"]),
            str(item["condition"]),
            str(item["descriptive_format_audit"]["raw_output_pattern"]),
        )
        for item in inventory
    )
    seed_pattern = Counter(
        (
            int(item["seed"]),
            str(item["descriptive_format_audit"]["raw_output_pattern"]),
        )
        for item in inventory
    )
    return {
        "schema_version": 1,
        "study_id": "PIVOT_EXP_A3",
        "source_study_id": "PIVOT_EXP_A2",
        "audit_scope": "format_description_only",
        "source_predictions": A2_PREDICTIONS.as_posix(),
        "source_predictions_sha256": _sha256(source),
        "source_prediction_count": 5300,
        "selection": {
            "family": "manipulation_check",
            "evaluation.evaluation_method": "no_match",
            "selected_count": len(inventory),
        },
        "frozen_boundary_passed": bool(boundary["passed"]),
        "counts": {
            "by_manipulation_task": counts("manipulation_task"),
            "by_condition": counts("condition"),
            "by_seed": {int(key): value for key, value in counts("seed").items()},
            "by_raw_output_pattern": dict(
                sorted(Counter(audit["raw_output_pattern"] for audit in audits).items())
            ),
            "by_task_condition_pattern": [
                {"task": key[0], "condition": key[1], "pattern": key[2], "count": value}
                for key, value in sorted(cross.items())
            ],
            "by_seed_pattern": [
                {"seed": key[0], "pattern": key[1], "count": value}
                for key, value in sorted(seed_pattern.items())
            ],
        },
        "flags": {
            "allowed_choice_string_appears": sum(
                bool(audit["allowed_choice_string_appears"]) for audit in audits
            ),
            "multiple_choices_appear": sum(
                bool(audit["multiple_choices_appear"]) for audit in audits
            ),
            "alias_variants_appear": sum(
                bool(audit["alias_variants_appear"]) for audit in audits
            ),
            "semantically_correct_description_but_violates_grammar": sum(
                bool(audit["semantically_correct_description_but_violates_grammar"])
                for audit in audits
            ),
            "genuinely_off_task": sum(bool(audit["genuinely_off_task"]) for audit in audits),
        },
        "research_integrity": {
            "PIVOT_EXP_A2_metrics_modified": False,
            "PIVOT_EXP_A2_predictions_modified": False,
            "PIVOT_EXP_A2_predictions_rescored": False,
            "PIVOT_EXP_A2_unparsed_outputs_deleted": False,
            "audit_may_support_PIVOT_EXP_A2_accuracy_recalculation": False,
        },
    }


def _render_legacy_audit(summary: dict[str, Any]) -> str:
    task = summary["counts"]["by_manipulation_task"]
    condition = summary["counts"]["by_condition"]
    pattern = summary["counts"]["by_raw_output_pattern"]
    flags = summary["flags"]
    return f"""# PIVOT_EXP_A3 Legacy Parse-Failure Audit

## Immutable scope

This is a read-only format audit of the 137 frozen PIVOT_EXP_A2 manipulation-check rows whose
frozen `evaluation.evaluation_method` is `no_match`. The source SHA-256 is
`{summary['source_predictions_sha256']}`. No prediction was changed, deleted, reinterpreted, or
rescored, and this audit cannot be used to recalculate PIVOT_EXP_A2 accuracy or alter its
`INCONCLUSIVE` decision.

## Frozen selection

- Source predictions: {summary['source_prediction_count']}
- Selected `manipulation_check` / `no_match` rows: {summary['selection']['selected_count']}
- Frozen boundary passed: {str(summary['frozen_boundary_passed']).lower()}

## Distribution

- By task: `{json.dumps(task, sort_keys=True)}`
- By condition: `{json.dumps(condition, sort_keys=True)}`
- By raw-output pattern: `{json.dumps(pattern, sort_keys=True)}`
- At least one registered choice string appears: {flags['allowed_choice_string_appears']}
- Multiple registered choices appear: {flags['multiple_choices_appear']}
- Alias variants appear: {flags['alias_variants_appear']}
- Registered answer string appears inside an invalid output:
  {flags['semantically_correct_description_but_violates_grammar']}
- Genuinely off-task: {flags['genuinely_off_task']}

The last-but-one flag is string-occurrence evidence only. It does not convert a multiple-choice
list into a valid answer and is not a revised correctness judgment.

## Measurement implication for PIVOT_EXP_A3

The dominant failure is multi-option emission rather than an empty response: the old free-text
instrument mixed answer selection with response formatting. PIVOT_EXP_A3 therefore uses
conditional-likelihood forced choice as its primary measurement and a frozen exact grammar
(`FINAL_CHOICE=<1|2|3|4>`) only as a secondary external-validity measure. Numeric option IDs avoid
the historical `Entity A/B/C` alias namespace.
"""


def _contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack))


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


__all__ = [
    "A2_FROZEN_COMMIT",
    "A2_FROZEN_SHA256",
    "a2_scene_ids",
    "artifact_metadata",
    "audit_legacy_parse_failures",
    "validate_a2_frozen_assets",
    "validate_prediction_integrity",
    "validate_trial_inventory",
]
