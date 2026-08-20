"""Apply preregistered experiment rules without post-hoc threshold changes."""

from __future__ import annotations

from typing import Any


def evaluate_decision(
    experiment: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Return a machine-readable GO, NO-GO, or INCONCLUSIVE decision report."""

    experiment_id = experiment["experiment_id"]
    if result.get("experiment_id") != experiment_id:
        raise ValueError("result experiment_id does not match the registered experiment")
    if result.get("hypothesis") != experiment["hypothesis_id"]:
        raise ValueError("result hypothesis does not match the registered experiment")

    rule = experiment["decision_rule"]
    aggregate = result.get("metrics", {}).get("aggregate", {})
    statistical_result: list[dict[str, Any]] = []
    robustness_criteria: list[dict[str, Any]] = []
    incomplete = False

    for criterion in rule["criteria"]:
        metric_path = criterion["metric_path"]
        summary = aggregate.get(metric_path)
        if not isinstance(summary, dict):
            incomplete = True
            statistical_result.append(
                {
                    "metric_path": metric_path,
                    "mean": None,
                    "minimum_gain": criterion["minimum_gain"],
                    "confidence_interval": None,
                    "p_value": None,
                    "significance_level": rule["statistical_significance"],
                    "complete": False,
                    "passed": False,
                }
            )
            robustness_criteria.append(
                {
                    "metric_path": metric_path,
                    "observed_seeds": None,
                    "minimum_seeds": rule["robustness"]["minimum_seeds"],
                    "positive_seed_fraction": None,
                    "minimum_positive_seed_fraction": rule["robustness"][
                        "minimum_positive_seed_fraction"
                    ],
                    "complete": False,
                    "passed": False,
                }
            )
            continue

        mean = summary.get("mean")
        interval = summary.get("confidence_interval")
        lower = interval.get("lower") if isinstance(interval, dict) else None
        upper = interval.get("upper") if isinstance(interval, dict) else None
        test = summary.get("statistical_test")
        p_value = test.get("p_value") if isinstance(test, dict) else None
        statistical_complete = all(_is_number(value) for value in (mean, lower, upper, p_value))
        if not statistical_complete:
            incomplete = True
        statistical_passed = bool(
            statistical_complete
            and mean >= criterion["minimum_gain"]
            and (not rule["confidence_interval_excludes_zero"] or lower > 0.0)
            and p_value <= rule["statistical_significance"]
        )
        statistical_result.append(
            {
                "metric_path": metric_path,
                "mean": mean if _is_number(mean) else None,
                "minimum_gain": criterion["minimum_gain"],
                "confidence_interval": (
                    {
                        "lower": lower,
                        "upper": upper,
                        "excludes_zero_required": rule[
                            "confidence_interval_excludes_zero"
                        ],
                    }
                    if _is_number(lower) and _is_number(upper)
                    else None
                ),
                "p_value": p_value if _is_number(p_value) else None,
                "significance_level": rule["statistical_significance"],
                "complete": statistical_complete,
                "passed": statistical_passed,
            }
        )

        observed_seeds = summary.get("n_seeds")
        positive_fraction = summary.get("positive_seed_fraction")
        robustness_complete = _is_integer(observed_seeds) and _is_number(positive_fraction)
        if not robustness_complete:
            incomplete = True
        robustness_passed = bool(
            robustness_complete
            and observed_seeds >= rule["robustness"]["minimum_seeds"]
            and positive_fraction
            >= rule["robustness"]["minimum_positive_seed_fraction"]
        )
        robustness_criteria.append(
            {
                "metric_path": metric_path,
                "observed_seeds": observed_seeds if _is_integer(observed_seeds) else None,
                "minimum_seeds": rule["robustness"]["minimum_seeds"],
                "positive_seed_fraction": (
                    positive_fraction if _is_number(positive_fraction) else None
                ),
                "minimum_positive_seed_fraction": rule["robustness"][
                    "minimum_positive_seed_fraction"
                ],
                "complete": robustness_complete,
                "passed": robustness_passed,
            }
        )

    manifest_checks: list[dict[str, Any]] = []
    assertions = rule.get("required_manifest_assertions", {})
    per_seed = result.get("integrity", {}).get("per_seed", [])
    for key, expected in assertions.items():
        observed = [row.get("manifest_assertions", {}).get(key) for row in per_seed]
        complete = bool(per_seed) and all(value is not None for value in observed)
        if not complete:
            incomplete = True
        manifest_checks.append(
            {
                "assertion": key,
                "expected": expected,
                "observed": observed,
                "complete": complete,
                "passed": complete and all(value == expected for value in observed),
            }
        )

    result_complete = result.get("status") == "complete"
    registry_validated = result.get("integrity", {}).get("registry_validated") is True
    actual_model = str(result.get("model", ""))
    actual_backend = actual_model.removeprefix("frozen-")
    allowed_backends = tuple(
        str(value)
        for value in experiment["model"].get(
            "allowed_backends", (experiment["model"]["configured_backend"],)
        )
    )
    expected_dataset = f"{experiment['dataset']['name']}@{experiment['dataset']['version']}"
    model_matches_registration = (
        actual_model.startswith("frozen-") and actual_backend in allowed_backends
    )
    dataset_matches_registration = result.get("dataset") == expected_dataset
    scientific_backends = {
        str(value) for value in experiment["model"].get("scientific_backends", ())
    }
    model_eligible = bool(experiment["model"]["scientific_decision_allowed"]) or (
        actual_backend in scientific_backends
    )
    evidence_role = (
        "scientific_evidence" if model_eligible else experiment["model"]["evidence_role"]
    )
    if not all(
        (
            result_complete,
            registry_validated,
            model_matches_registration,
            dataset_matches_registration,
            model_eligible,
        )
    ):
        incomplete = True
    robustness_passed = bool(
        result_complete
        and registry_validated
        and model_matches_registration
        and dataset_matches_registration
        and model_eligible
        and all(item["passed"] for item in robustness_criteria)
        and all(item["passed"] for item in manifest_checks)
    )
    if incomplete:
        decision = "INCONCLUSIVE"
    elif all(item["passed"] for item in statistical_result) and robustness_passed:
        decision = "GO"
    else:
        decision = "NO-GO"

    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "evidence": {
            "hypothesis_id": experiment["hypothesis_id"],
            "run_id": result.get("run_id"),
            "result_status": result.get("status"),
            "model": result.get("model"),
            "dataset": result.get("dataset"),
            "evidence_role": evidence_role,
        },
        "statistical_result": statistical_result,
        "robustness_check": {
            "result_complete": result_complete,
            "registry_validated": registry_validated,
            "model_matches_registration": model_matches_registration,
            "dataset_matches_registration": dataset_matches_registration,
            "model_scientific_decision_allowed": model_eligible,
            "criteria": robustness_criteria,
            "manifest_assertions": manifest_checks,
            "passed": robustness_passed,
        },
        "final_decision": decision,
        "next_action": _next_action(decision),
    }


def render_decision_report(report: dict[str, Any]) -> str:
    """Render a human-readable mirror of the authoritative YAML report."""

    evidence = report["evidence"]
    lines = [
        f"# Decision Report — {report['experiment_id']}",
        "",
        f"- Hypothesis: `{evidence['hypothesis_id']}`",
        f"- Run: `{evidence.get('run_id')}`",
        f"- Evidence role: `{evidence['evidence_role']}`",
        f"- Final decision: **{report['final_decision']}**",
        f"- Next action: {report['next_action']}",
        "",
        "## Statistical result",
        "",
        "| Metric | Mean | Minimum | CI | p-value | Pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for item in report["statistical_result"]:
        interval = item.get("confidence_interval")
        rendered_interval = (
            f"[{interval['lower']}, {interval['upper']}]" if interval else "missing"
        )
        lines.append(
            f"| `{item['metric_path']}` | {item['mean']} | {item['minimum_gain']} | "
            f"{rendered_interval} | {item['p_value']} | "
            f"{'YES' if item['passed'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Robustness check",
            "",
            f"- Result complete: {report['robustness_check']['result_complete']}",
            f"- Registry validated: {report['robustness_check']['registry_validated']}",
            f"- Model matches registration: "
            f"{report['robustness_check']['model_matches_registration']}",
            f"- Dataset matches registration: "
            f"{report['robustness_check']['dataset_matches_registration']}",
            f"- Model eligible for scientific decision: "
            f"{report['robustness_check']['model_scientific_decision_allowed']}",
            f"- Overall robustness pass: {report['robustness_check']['passed']}",
            "",
            "This Markdown file mirrors `decision_report.yaml`, which is the authoritative record.",
            "Threshold changes require a new registration before rerunning.",
            "",
        ]
    )
    return "\n".join(lines)


def _next_action(decision: str) -> str:
    if decision == "GO":
        return "Proceed to independent replication and reviewed evidence promotion."
    if decision == "NO-GO":
        return "Record the falsifying result and revise or retire the hypothesis."
    return "Resolve missing or ineligible evidence and rerun the preregistered protocol."


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
