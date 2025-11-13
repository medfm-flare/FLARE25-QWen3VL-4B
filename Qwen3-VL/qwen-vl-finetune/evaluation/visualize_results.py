#!/usr/bin/env python3
"""
FLARE25 Evaluation Results Visualization
========================================

This script creates bar plot visualizations of the evaluation results.

Usage:
    python evaluation/visualize_results.py --results_dir ./evaluation_results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict


def _primary_metric_for_task(task_type: str, metrics: Dict[str, Any]):
    """Return (label, value, higher_is_better) for the task's headline metric."""
    task_key = (task_type or "").lower()

    if "classification" in task_key and "multi" not in task_key:
        value = metrics.get("accuracy") or metrics.get("balanced_accuracy")
        return "Accuracy", value, True

    if "multi" in task_key and "classification" in task_key:
        value = metrics.get("f1_macro") or metrics.get("f1_micro")
        return "F1 Macro", value, True

    if "instance" in task_key and "detection" in task_key:
        value = metrics.get("f1_at_0.3") or metrics.get("f1_primary")
        return "F1@0.3", value, True

    if "detection" in task_key:
        value = metrics.get("f1_at_0.3") or metrics.get("f1_primary")
        return "F1@0.3", value, True

    if "counting" in task_key:
        value = metrics.get("mae")
        return "MAE", value, False

    if "regression" in task_key:
        value = metrics.get("mae")
        return "MAE", value, False

    if "report" in task_key:
        value = metrics.get("bleu") or metrics.get("rouge_l")
        return "BLEU", value, True

    # Fallback to first numeric metric
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            return name, value, True

    return None, None, True


def _lighten_color(color: str, amount: float = 0.5) -> str:
    """Return a lighter variant of the given matplotlib color."""
    try:
        base = mcolors.to_rgb(color)
    except ValueError:
        base = (0.6, 0.6, 0.6)
    amount = max(0.0, min(1.0, amount))
    return mcolors.to_hex(tuple(1 - (1 - c) * amount for c in base))


def _format_metric_value(value: float, metric_label: str) -> str:
    if value is None:
        return ""
    if metric_label.upper() in {"MAE", "RMSE"}:
        return f"{value:.2f}"
    return f"{value:.3f}"


LOWER_BETTER_KEYWORDS = {"mae", "rmse", "error", "loss"}


def _is_lower_better(metric_name: str) -> bool:
    name = (metric_name or "").lower()
    return any(keyword in name for keyword in LOWER_BETTER_KEYWORDS)


def compute_metric_stats(summary: Dict[str, Any]) -> tuple[Dict[str, float], Dict[str, int]]:
    """Return mean values and sample counts for every numeric metric."""
    metric_samples: Dict[str, List[float]] = defaultdict(list)

    for result in summary.get('per_dataset_results', {}).values():
        for metrics in result.get('metrics_by_task', {}).values():
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    metric_samples[metric_name].append(float(metric_value))

    metric_means = {
        metric: float(np.mean(values))
        for metric, values in metric_samples.items()
        if values
    }
    metric_counts = {
        metric: len(values)
        for metric, values in metric_samples.items()
        if values
    }
    return metric_means, metric_counts


def load_evaluation_summary(results_dir: Path) -> Dict:
    """Load the evaluation summary JSON"""
    summary_file = results_dir / "evaluation_summary.json"

    if not summary_file.exists():
        raise FileNotFoundError(f"Evaluation summary not found: {summary_file}")

    with open(summary_file, 'r') as f:
        return json.load(f)


def create_metric_barplot(
    summary: Dict[str, Any],
    output_path: Path,
    baseline_summary: Dict[str, Any] | None = None
):
    """Render average metric values across all tasks (optionally vs baseline)."""
    metric_means, metric_counts = compute_metric_stats(summary)
    if not metric_means:
        print("No metric values available for plotting")
        return

    baseline_means = {}
    if baseline_summary:
        baseline_means, _ = compute_metric_stats(baseline_summary)

    metrics = sorted(set(metric_means) | set(baseline_means))
    if not metrics:
        print("No metrics to plot")
        return

    fine_values = [metric_means.get(metric, np.nan) for metric in metrics]
    baseline_values = [baseline_means.get(metric, np.nan) for metric in metrics]
    counts = [metric_counts.get(metric, 0) for metric in metrics]

    x_pos = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 0.6), 6))

    has_baseline = baseline_summary is not None and any(np.isfinite(baseline_values))

    if has_baseline:
        bar_width = 0.4
        baseline_bars = ax.bar(
            x_pos - bar_width / 2,
            baseline_values,
            width=bar_width,
            color='#b0c4de',
            edgecolor='black',
            linewidth=0.6,
            alpha=0.85,
            label='Baseline'
        )
        fine_bars = ax.bar(
            x_pos + bar_width / 2,
            fine_values,
            width=bar_width,
            color='#2E86AB',
            edgecolor='black',
            linewidth=0.6,
            alpha=0.95,
            label='Fine-tuned'
        )
    else:
        bar_width = 0.6
        fine_bars = ax.bar(
            x_pos,
            fine_values,
            width=bar_width,
            color='#2E86AB',
            edgecolor='black',
            linewidth=0.6,
            alpha=0.95
        )
        baseline_bars = None

    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Value', fontsize=12, fontweight='bold')
    title_suffix = ' vs Baseline' if has_baseline else ''
    ax.set_title(
        f'Average Metric Performance{title_suffix}',
        fontsize=14,
        fontweight='bold',
        pad=18
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # annotate counts and values
    def annotate(bars, values):
        if bars is None:
            return
        for bar, value, metric, count in zip(bars, values, metrics, counts):
            if value is None or np.isnan(value):
                continue
            height = bar.get_height()
            label = _format_metric_value(value, metric)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{label}\n(n={count})",
                ha='center',
                va='bottom',
                fontsize=8
            )

    if has_baseline:
        annotate(baseline_bars, baseline_values)
    annotate(fine_bars, fine_values)

    if has_baseline:
        ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metric bar plot to {output_path}")
    plt.close()


def create_metric_improvement_plot(
    summary: Dict[str, Any],
    output_path: Path,
    baseline_summary: Dict[str, Any] | None = None
):
    """Plot aggregated improvement per metric (positive means better)."""
    if not baseline_summary:
        print("Baseline summary not provided; skipping improvement plot")
        return

    fine_means, _ = compute_metric_stats(summary)
    baseline_means, _ = compute_metric_stats(baseline_summary)

    common_metrics = sorted(set(fine_means) & set(baseline_means))
    if not common_metrics:
        print("No shared metrics for improvement plot")
        return

    deltas = []
    for metric in common_metrics:
        fine_val = fine_means[metric]
        base_val = baseline_means[metric]
        if _is_lower_better(metric):
            delta = base_val - fine_val  # positive means reduction (better)
        else:
            delta = fine_val - base_val
        deltas.append(delta)

    fig, ax = plt.subplots(figsize=(max(10, len(common_metrics) * 0.6), 6))
    x_pos = np.arange(len(common_metrics))

    bars = ax.bar(
        x_pos,
        deltas,
        color=['#2E86AB' if delta >= 0 else '#BC4B51' for delta in deltas],
        edgecolor='black',
        linewidth=0.6,
        alpha=0.9
    )

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (Fine-tuned vs Baseline)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Metric Improvements (positive = better)',
        fontsize=14,
        fontweight='bold',
        pad=18
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(common_metrics, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for bar, delta, metric in zip(bars, deltas, common_metrics):
        height = bar.get_height()
        label = f"{delta:+.3f}"
        va = 'bottom' if delta >= 0 else 'top'
        offset = 0.01 if delta >= 0 else -0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            label,
            ha='center',
            va=va,
            fontsize=9,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metric improvement plot to {output_path}")
    plt.close()


def create_metric_distribution_plot(
    summary: Dict[str, Any],
    output_path: Path
):
    """Plot how many times each metric appears across datasets."""
    metric_means, metric_counts = compute_metric_stats(summary)
    if not metric_means:
        print("No metric values to summarise")
        return

    metrics = sorted(metric_counts, key=lambda m: metric_counts[m], reverse=True)
    counts = [metric_counts[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 0.5), 4.5))
    bars = ax.bar(
        np.arange(len(metrics)),
        counts,
        color='#6A994E',
        edgecolor='black',
        linewidth=0.6,
        alpha=0.85
    )

    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Occurrences', fontsize=12, fontweight='bold')
    ax.set_title('Metric Coverage Across Dataset Tasks', fontsize=14, fontweight='bold', pad=16)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for bar, metric in zip(bars, metrics):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={metric_counts[metric]}",
            ha='center',
            va='bottom',
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metric distribution plot to {output_path}")
    plt.close()


def create_summary_stats_figure(
    summary: Dict,
    output_path: Path,
    baseline_summary: Dict | None = None
):
    """Create textual overview of evaluation, highlighting improvements if available."""
    num_datasets = summary.get('num_datasets', 0)
    total_samples = summary.get('total_samples', 0)
    metric_means, metric_counts = compute_metric_stats(summary)

    baseline_means = {}
    metric_improvements: Dict[str, float] = {}
    if baseline_summary:
        baseline_means, _ = compute_metric_stats(baseline_summary)
        for metric, fine_mean in metric_means.items():
            if metric not in baseline_means:
                continue
            base_mean = baseline_means[metric]
            if _is_lower_better(metric):
                delta = base_mean - fine_mean
            else:
                delta = fine_mean - base_mean
            metric_improvements[metric] = delta

    positive_metrics = [m for m, v in metric_means.items() if not _is_lower_better(m)]
    negative_metrics = [m for m, v in metric_means.items() if _is_lower_better(m)]

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.axis('off')

    title = "FLARE25 Qwen3-VL Evaluation Summary"
    if baseline_summary:
        title += " (vs Baseline)"
    ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=18, fontweight='bold', transform=ax.transAxes)

    y_pos = 0.85
    line_height = 0.045
    stats_lines = [
        f"📊 Total Datasets Evaluated: {num_datasets}",
        f"📝 Total Samples: {total_samples:,}",
        "",
        "🎯 Average Performance (Fine-tuned):"
    ]

    for metric in sorted(positive_metrics):
        stats_lines.append(
            f"   • {metric}: {_format_metric_value(metric_means[metric], metric)} (n={metric_counts.get(metric, 0)})"
        )
    for metric in sorted(negative_metrics):
        stats_lines.append(
            f"   • {metric}: {_format_metric_value(metric_means[metric], metric)} (n={metric_counts.get(metric, 0)})"
        )

    if metric_improvements:
        stats_lines.extend(["", "🚀 Average Improvement over Baseline:"])
        for metric, delta in sorted(metric_improvements.items(), key=lambda item: item[1], reverse=True):
            direction = "↑" if delta >= 0 else "↓"
            stats_lines.append(f"   • {metric}: {direction}{delta:+.3f}")

    if metric_improvements:
        stats_lines.extend(["", "📈 Largest Metric Gains:"])
        top_improvements = sorted(metric_improvements.items(), key=lambda item: item[1], reverse=True)[:5]
        for metric, delta in top_improvements:
            stats_lines.append(f"   • {metric}: {delta:+.3f}")

    for idx, line in enumerate(stats_lines):
        ax.text(
            0.08,
            y_pos - idx * line_height,
            line,
            ha='left',
            va='top',
            fontsize=12,
            transform=ax.transAxes,
            family='monospace'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary statistics to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize FLARE25 evaluation results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./evaluation_results',
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save plots (defaults to results_dir)'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("FLARE25 Evaluation Results Visualization")
    print("="*70)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load summary
    try:
        summary = load_evaluation_summary(results_dir)
        print("✓ Loaded fine-tuned evaluation summary")
        print(f"  - Datasets: {summary.get('num_datasets', 0)}")
        print(f"  - Total samples: {summary.get('total_samples', 0):,}")
        print()
    except Exception as e:
        print(f"Error loading evaluation summary: {e}")
        sys.exit(1)

    baseline_summary = None
    baseline_dir = results_dir / "baseline"
    baseline_summary_path = baseline_dir / "evaluation_summary.json"
    if baseline_summary_path.exists():
        try:
            baseline_summary = load_evaluation_summary(baseline_dir)
            print("✓ Loaded baseline evaluation summary")
            print(f"  - Baseline model: {baseline_summary.get('model_path', 'unknown')}")
            print()
        except Exception as e:
            print(f"Warning: Could not load baseline summary ({e})")
            baseline_summary = None

    # Create visualizations
    print("Creating visualizations...")
    print()

    try:
        create_metric_barplot(
            summary,
            output_dir / "eval_results_metrics.png",
            baseline_summary=baseline_summary
        )
    except Exception as e:
        print(f"Error creating metric bar plot: {e}")

    try:
        create_metric_improvement_plot(
            summary,
            output_dir / "eval_results_metric_improvements.png",
            baseline_summary=baseline_summary
        )
    except Exception as e:
        print(f"Error creating improvement plot: {e}")

    try:
        create_metric_distribution_plot(
            summary,
            output_dir / "eval_results_metric_distribution.png"
        )
    except Exception as e:
        print(f"Error creating metric distribution plot: {e}")

    try:
        create_summary_stats_figure(
            summary,
            output_dir / "eval_results_summary.png",
            baseline_summary=baseline_summary
        )
    except Exception as e:
        print(f"Error creating summary figure: {e}")

    print()
    print("="*70)
    print("Visualization complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
