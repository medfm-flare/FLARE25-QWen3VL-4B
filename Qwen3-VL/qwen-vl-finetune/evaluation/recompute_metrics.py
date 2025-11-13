#!/usr/bin/env python3
"""Recompute FLARE25 metrics directly from saved prediction files."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from metrics import (
    calculate_classification_metrics,
    calculate_multilabel_metrics,
    calculate_detection_metrics,
    calculate_instance_detection_metrics,
    calculate_counting_metrics,
    calculate_regression_metrics,
    calculate_report_generation_metrics,
)
from task_utils import canonical_task_type

logger = logging.getLogger(__name__)


def _format_metric_for_log(metric_name: str, value: Any) -> str:
    if isinstance(value, (int, float)):
        if metric_name in {"valid_samples", "correct_predictions"}:
            return str(int(round(value)))

        if value == 0:
            return "0"

        magnitude = abs(value)
        if magnitude < 1e-6:
            return f"{value:.12f}"
        if magnitude < 1e-4:
            return f"{value:.10f}"
        if magnitude < 1:
            return f"{value:.8f}"
        if magnitude < 100:
            return f"{value:.6f}"
        return f"{value:.4f}"

    return str(value)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (int, float)):
        return not np.isnan(value)
    if isinstance(value, np.floating):
        return not np.isnan(value)
    return True


def _compute_task_metrics(task_type: str, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    preds = [sample.get("prediction") for sample in samples]
    refs = [sample.get("Answer") for sample in samples]

    valid_pairs = [
        (pred, ref) for pred, ref in zip(preds, refs) if _has_value(pred) and _has_value(ref)
    ]

    if not valid_pairs:
        logger.warning("No valid pairs found for task %s", task_type)
        return {"valid_samples": 0}

    pred_valid = [pred for pred, _ in valid_pairs]
    ref_valid = [ref for _, ref in valid_pairs]

    task_key = canonical_task_type(task_type).lower()

    if "classification" in task_key and "multi-label" not in task_key:
        return calculate_classification_metrics(pred_valid, ref_valid)
    if "multi-label" in task_key:
        return calculate_multilabel_metrics(pred_valid, ref_valid)
    if "detection" in task_key:
        if "instance" in task_key:
            return calculate_instance_detection_metrics(pred_valid, ref_valid)
        return calculate_detection_metrics(pred_valid, ref_valid)
    if "counting" in task_key:
        return calculate_counting_metrics(pred_valid, ref_valid)
    if "regression" in task_key:
        return calculate_regression_metrics(pred_valid, ref_valid)
    if "report generation" in task_key:
        return calculate_report_generation_metrics(pred_valid, ref_valid)

    logger.warning("Unknown task type %s, defaulting to classification metrics", task_type)
    return calculate_classification_metrics(pred_valid, ref_valid)


def _aggregate_by_task(
    per_dataset_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    task_type_groups: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)

    for dataset_key, result in per_dataset_results.items():
        display_name = result["metadata"].get("display_name", dataset_key)
        for task_type, metrics in result["metrics_by_task"].items():
            canonical_key = canonical_task_type(task_type)
            task_type_groups[canonical_key].append((display_name, metrics))

    aggregate_metrics: Dict[str, Dict[str, float]] = {}

    for task_type, dataset_results in task_type_groups.items():
        sum_metrics: Dict[str, float] = {}
        weighted_metrics: Dict[str, Tuple[float, float]] = {}
        fallback_metrics: Dict[str, Tuple[float, int]] = {}

        for _, metrics in dataset_results:
            sample_weight_raw = metrics.get("valid_samples", 0)
            sample_weight = (
                float(sample_weight_raw)
                if isinstance(sample_weight_raw, (int, float)) and sample_weight_raw
                else 0.0
            )

            for metric_name, metric_value in metrics.items():
                if not isinstance(metric_value, (int, float)):
                    continue

                if metric_name in {"valid_samples", "correct_predictions"}:
                    sum_metrics[metric_name] = sum_metrics.get(metric_name, 0.0) + float(
                        metric_value
                    )
                    continue

                if sample_weight > 0:
                    total, weight = weighted_metrics.get(metric_name, (0.0, 0.0))
                    weighted_metrics[metric_name] = (
                        total + float(metric_value) * sample_weight,
                        weight + sample_weight,
                    )
                else:
                    total, count = fallback_metrics.get(metric_name, (0.0, 0))
                    fallback_metrics[metric_name] = (
                        total + float(metric_value),
                        count + 1,
                    )

        agg_metrics: Dict[str, float] = {}

        for metric_name, total in sum_metrics.items():
            agg_metrics[metric_name] = total

        for metric_name, (total, weight) in weighted_metrics.items():
            if weight > 0:
                agg_metrics[metric_name] = total / weight

        for metric_name, (total, count) in fallback_metrics.items():
            if metric_name not in agg_metrics and count > 0:
                agg_metrics[metric_name] = total / count

        if (
            "correct_predictions" in agg_metrics
            and agg_metrics.get("valid_samples")
            and agg_metrics["valid_samples"] > 0
        ):
            agg_metrics["accuracy"] = agg_metrics["correct_predictions"] / agg_metrics[
                "valid_samples"
            ]

        for key in {"valid_samples", "correct_predictions"}:
            if key in agg_metrics:
                agg_metrics[key] = int(round(agg_metrics[key]))

        aggregate_metrics[task_type] = agg_metrics

    return aggregate_metrics


def _iter_prediction_files(results_dir: Path) -> Iterable[Path]:
    for path in sorted(results_dir.glob("*_predictions.json")):
        if path.name == "all_predictions.json":
            continue
        yield path


def process_predictions_file(path: Path) -> Tuple[str, Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Unexpected data structure in {path}")

    stem = path.stem.replace("_predictions", "")
    try:
        split, category, dataset_base = stem.split("_", 2)
    except ValueError:
        split, category, dataset_base = "unknown", "unknown", stem

    dataset_key = "::".join([split, category, dataset_base])
    display_name = "/".join([split, category, dataset_base])

    task_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    task_type_counts: Counter[str] = Counter()

    for sample in data:
        canonical = sample.get("TaskTypeCanonical")
        if not canonical:
            canonical = canonical_task_type(sample.get("TaskType", "Classification"))
        task_groups[canonical].append(sample)
        task_type_counts[canonical] += 1

    metrics_by_task: Dict[str, Dict[str, Any]] = {}
    for task_type, samples in task_groups.items():
        metrics_by_task[task_type] = _compute_task_metrics(task_type, samples)

    valid_predictions = sum(
        1
        for sample in data
        if (
            isinstance(sample.get("prediction"), str)
            and sample["prediction"].strip()
        )
        or isinstance(sample.get("prediction"), (int, float))
    )

    metadata = {
        "dataset_name": dataset_base,
        "task_types": sorted(task_groups.keys()),
        "samples_by_task": {task: int(count) for task, count in task_type_counts.items()},
        "total_samples": len(data),
        "valid_predictions": valid_predictions,
        "split": split,
        "category": category,
        "dataset_base_name": dataset_base,
        "display_name": display_name,
        "dataset_key": dataset_key,
        "source_file": str(path),
    }

    return dataset_key, {"metadata": metadata, "metrics_by_task": metrics_by_task}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute FLARE25 evaluation metrics from saved predictions"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Directory containing *_predictions.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the recomputed summary JSON",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results_dir = args.results_dir
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    per_dataset_results: Dict[str, Dict[str, Any]] = {}

    for predictions_file in _iter_prediction_files(results_dir):
        logger.info("Processing %s", predictions_file)
        dataset_key, result = process_predictions_file(predictions_file)
        per_dataset_results[dataset_key] = result

    if not per_dataset_results:
        logger.warning("No prediction files found in %s", results_dir)
        return

    aggregate_metrics = _aggregate_by_task(per_dataset_results)
    total_samples = sum(
        result["metadata"].get("total_samples", 0) for result in per_dataset_results.values()
    )

    summary = {
        "results_dir": str(results_dir.resolve()),
        "generated_at": datetime.now(UTC).isoformat(),
        "num_datasets": len(per_dataset_results),
        "total_samples": int(total_samples),
        "per_dataset_results": per_dataset_results,
        "aggregate_metrics_by_task": aggregate_metrics,
    }

    output_path = args.output or (results_dir / "recomputed_metrics_summary.json")
    output_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved recomputed metrics summary to %s", output_path)

    logger.info("\nAggregate metrics:")
    for task_type, metrics in aggregate_metrics.items():
        logger.info("  %s:", task_type)
        for metric_name, value in metrics.items():
            logger.info(
                "    %s: %s",
                metric_name,
                _format_metric_for_log(metric_name, value)
            )


if __name__ == "__main__":
    main()
