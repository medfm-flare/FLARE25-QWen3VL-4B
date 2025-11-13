#!/usr/bin/env python3
"""
FLARE25 Evaluation Script for Qwen3-VL
=======================================

This script evaluates a trained Qwen3-VL model on the FLARE25 validation-public dataset
and calculates task-specific metrics.

Usage:
    python evaluation/evaluate_flare25.py \
        --model_path ./output/qwen3vl_flare25 \
        --dataset_path /path/to/organized_dataset \
        --output_dir ./evaluation_results

Features:
- Supports all FLARE25 task types (Classification, Detection, Counting, etc.)
- Multi-image input handling
- Task-specific metrics calculation
- Detailed per-dataset and per-task reports
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Import transformers components
from transformers import AutoProcessor, AutoTokenizer
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    # Fallback if Qwen3 not available
    Qwen3VLForConditionalGeneration = None

# Add parent directory to path for metrics import
sys.path.insert(0, str(Path(__file__).parent))
from metrics import (
    calculate_classification_metrics,
    calculate_multilabel_metrics,
    calculate_detection_metrics,
    calculate_instance_detection_metrics,
    calculate_counting_metrics,
    calculate_regression_metrics,
    calculate_report_generation_metrics
)
from task_utils import canonical_task_type

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _format_metric_for_log(metric_name: str, value: Any) -> str:
    """Pretty-print metric values while preserving detail for small magnitudes."""
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


class FLARE25Evaluator:
    """Evaluator for FLARE25 datasets with Qwen3-VL"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
        batch_size: int = 1
    ):
        """
        Initialize evaluator

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for inference (currently only supports 1)
        """
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        logger.info(f"Loading model from {model_path}")
        self._load_model()

    def _load_model(self):
        """Load model, processor, and tokenizer"""
        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Load model - use Qwen3VLForConditionalGeneration for generation tasks
        if Qwen3VLForConditionalGeneration is not None:
            logger.info(f"Loading model with Qwen3VLForConditionalGeneration")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )
        else:
            raise ImportError(
                "Qwen3VLForConditionalGeneration not available. "
                "Please ensure transformers>=4.57.0 is installed."
            )

        self.model.eval()
        logger.info(f"Model loaded successfully! Model class: {self.model.__class__.__name__}")

    def prepare_messages(
        self,
        images: List[str],
        question: str
    ) -> List[Dict]:
        """
        Prepare messages in Qwen3-VL format

        Args:
            images: List of image paths
            question: Question text

        Returns:
            Messages list with images and text
        """
        # Load images
        pil_images = []
        for img_path in images:
            try:
                img = Image.open(img_path).convert('RGB')
                pil_images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")

        if not pil_images:
            return None

        # Build content with image tags
        content = []

        # Add image placeholders
        for img in pil_images:
            content.append({"type": "image", "image": img})

        # Add text
        # For Qwen3-VL, image tags are implicit in the content structure
        content.append({"type": "text", "text": question})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        return messages, pil_images

    def inference(
        self,
        images: List[str],
        question: str
    ) -> str:
        """
        Run inference on a single sample

        Args:
            images: List of image paths
            question: Question text

        Returns:
            Generated answer text
        """
        try:
            # Prepare messages
            result = self.prepare_messages(images, question)
            if result is None:
                return ""

            messages, pil_images = result

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(
                text=[text],
                images=pil_images,
                return_tensors="pt",
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Greedy decoding for reproducibility
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return ""

    def evaluate_dataset(
        self,
        dataset_path: Path,
        questions_file: Path,
        image_dir: Path
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Evaluate a single dataset

        Args:
            dataset_path: Path to dataset directory
            questions_file: Path to questions JSON file
            image_dir: Path to images directory

        Returns:
            Tuple of (predictions, metadata)
        """
        # Load questions
        with open(questions_file, 'r') as f:
            questions = json.load(f)

        logger.info(f"Evaluating {len(questions)} samples from {dataset_path.name}")

        predictions = []
        task_type_counts = Counter()

        for sample in tqdm(questions, desc=f"Evaluating {dataset_path.name}"):
            # Get task type and normalize
            task_type_raw = sample.get("TaskType", "Classification")
            task_type_canonical = canonical_task_type(task_type_raw)
            task_type_counts[task_type_canonical] += 1

            # Get images
            image_names = sample["ImageName"]
            if isinstance(image_names, str):
                image_names = [image_names]

            # Resolve image paths
            image_paths = []
            for img_name in image_names:
                # Remove common prefixes
                for prefix in ["imagesVal/", "imagesTr/", "images/", "Images/"]:
                    if img_name.startswith(prefix):
                        img_name = img_name.replace(prefix, "", 1)

                img_path = image_dir / img_name
                if not img_path.exists():
                    # Try just the filename
                    img_path = image_dir / Path(img_name).name

                if img_path.exists():
                    image_paths.append(str(img_path))
                else:
                    logger.warning(f"Image not found: {img_path}")

            if not image_paths:
                logger.warning(f"No valid images for sample, skipping")
                predictions.append({
                    **sample,
                    "TaskTypeCanonical": task_type_canonical,
                    "prediction": ""
                })
                continue

            # Run inference
            question = sample["Question"]
            prediction = self.inference(image_paths, question)

            # Store result
            predictions.append({
                **sample,
                "TaskTypeCanonical": task_type_canonical,
                "prediction": prediction
            })

        metadata = {
            "dataset_name": dataset_path.name,
            "task_types": sorted(task_type_counts.keys()),
            "samples_by_task": {task: int(count) for task, count in task_type_counts.items()},
            "total_samples": len(questions),
            "valid_predictions": sum(
                1
                for p in predictions
                if (
                    isinstance(p["prediction"], str)
                    and p["prediction"].strip()
                )
                or isinstance(p["prediction"], (int, float))
            )
        }

        return predictions, metadata

    def calculate_metrics(
        self,
        predictions: List[Dict],
        task_type: str
    ) -> Dict[str, Any]:
        """
        Calculate metrics for a dataset based on task type

        Args:
            predictions: List of prediction dictionaries
            task_type: Type of task (Classification, Detection, etc.)

        Returns:
            Dictionary of calculated metrics
        """
        # Extract predictions and references
        pred_answers = [p["prediction"] for p in predictions]
        ref_answers = [p["Answer"] for p in predictions]

        # Remove samples without predictions or references
        def _has_value(value):
            if value is None:
                return False
            if isinstance(value, str):
                return value.strip() != ""
            if isinstance(value, (int, float)):
                return not np.isnan(value)
            if isinstance(value, np.floating):
                return not np.isnan(value)
            return True

        valid_pairs = [
            (pred, ref) for pred, ref in zip(pred_answers, ref_answers)
            if _has_value(pred) and _has_value(ref)
        ]

        if not valid_pairs:
            logger.warning(f"No valid prediction-reference pairs for {task_type}")
            return {"error": "No valid pairs"}

        pred_valid = [p for p, _ in valid_pairs]
        ref_valid = [r for _, r in valid_pairs]

        # Calculate metrics based on task type
        normalized_task_type = canonical_task_type(task_type)
        task_type_key = normalized_task_type.lower()

        if "classification" in task_type_key and "multi-label" not in task_type_key:
            metrics = calculate_classification_metrics(pred_valid, ref_valid)

        elif "multi-label" in task_type_key:
            metrics = calculate_multilabel_metrics(pred_valid, ref_valid)

        elif "detection" in task_type_key:
            if "instance" in task_type_key:
                metrics = calculate_instance_detection_metrics(pred_valid, ref_valid)
            else:
                metrics = calculate_detection_metrics(pred_valid, ref_valid)

        elif "counting" in task_type_key:
            metrics = calculate_counting_metrics(pred_valid, ref_valid)

        elif "regression" in task_type_key:
            metrics = calculate_regression_metrics(pred_valid, ref_valid)

        elif "report generation" in task_type_key:
            metrics = calculate_report_generation_metrics(pred_valid, ref_valid)

        else:
            # Default to classification
            logger.warning(f"Unknown task type {task_type}, using classification metrics")
            metrics = calculate_classification_metrics(pred_valid, ref_valid)

        return metrics


def find_evaluation_datasets(dataset_path: Path) -> List[Tuple[Path, Path, Path]]:
    """
    Find all evaluation datasets following FLARE25 official evaluation logic:

    Main evaluation sets (covers all datasets and metrics):
    - validation-hidden/{category}/{dataset} with _withGT.json (priority)
    - testing/{category}/{dataset} with _test.json

    Fallback only if dataset/metric missing:
    - validation-public/{category}/{dataset} (should not be needed)

    Args:
        dataset_path: Path to organized_dataset directory

    Returns:
        List of (dataset_dir, questions_file, images_dir) tuples
    """
    datasets = []
    found_datasets = set()  # Track (category, dataset_name) to avoid duplicates

    val_hidden_path = dataset_path / "validation-hidden"
    test_path = dataset_path / "testing"
    val_public_path = dataset_path / "validation-public"

    # Process validation-hidden datasets (PRIMARY)
    logger.info("Finding validation-hidden datasets...")
    if val_hidden_path.exists():
        for questions_file in val_hidden_path.rglob("*_questions_val*.json"):
            category = questions_file.parent.parent.name
            dataset_name = questions_file.parent.name

            # Skip if already processed
            if (category, dataset_name) in found_datasets:
                continue

            # Prefer _withGT.json for organizers with ground truth
            hidden_dir = val_hidden_path / category / dataset_name
            questions_file_withGT = hidden_dir / f"{dataset_name}_questions_val_withGT.json"
            questions_file_regular = hidden_dir / f"{dataset_name}_questions_val.json"

            # Use withGT if available, otherwise regular
            if questions_file_withGT.exists():
                questions_file = questions_file_withGT
            elif questions_file_regular.exists():
                questions_file = questions_file_regular
            else:
                continue

            # Find images directory
            images_dir = hidden_dir / "imagesVal"
            if not images_dir.exists():
                images_dir = hidden_dir / "images"

            if images_dir.exists():
                datasets.append((hidden_dir, questions_file, images_dir))
                found_datasets.add((category, dataset_name))
                logger.info(f"  ✓ validation-hidden: {category}/{dataset_name}")

    # Process testing datasets (PRIMARY)
    logger.info("Finding testing datasets...")
    if test_path.exists():
        for questions_file in test_path.rglob("*_questions_test.json"):
            category = questions_file.parent.parent.name
            dataset_name = questions_file.parent.name
            test_dir = questions_file.parent

            # Find images directory (imagesTs for testing)
            images_dir = test_dir / "imagesTs"
            if not images_dir.exists():
                images_dir = test_dir / "images"

            if images_dir.exists():
                datasets.append((test_dir, questions_file, images_dir))
                logger.info(f"  ✓ testing: {category}/{dataset_name}")
            else:
                logger.warning(f"  ✗ No images found for testing/{category}/{dataset_name}")

    # Fallback to validation-public ONLY if needed (should not happen)
    logger.info("Checking validation-public as fallback...")
    fallback_count = 0
    if val_public_path.exists():
        for questions_file in val_public_path.rglob("*_questions_val.json"):
            if "withGT" in questions_file.name:
                continue

            category = questions_file.parent.parent.name
            dataset_name = questions_file.parent.name

            # Only use if not already covered by hidden or testing
            if (category, dataset_name) in found_datasets:
                continue

            public_dir = val_public_path / category / dataset_name
            images_dir = public_dir / "imagesVal"
            if not images_dir.exists():
                images_dir = public_dir / "images"

            if images_dir.exists():
                datasets.append((public_dir, questions_file, images_dir))
                found_datasets.add((category, dataset_name))
                fallback_count += 1
                logger.info(f"  ⚠ validation-public (fallback): {category}/{dataset_name}")

    logger.info(f"\n📊 Total datasets to evaluate: {len(datasets)}")
    logger.info(f"   - validation-hidden + testing: {len(datasets) - fallback_count}")
    if fallback_count > 0:
        logger.info(f"   - validation-public (fallback): {fallback_count}")
    return datasets


def run_model_evaluation(
    evaluator: "FLARE25Evaluator",
    datasets: List[Tuple[Path, Path, Path]],
    output_dir: Path,
    label: str
) -> Dict[str, Any]:
    """Execute evaluation for a given model and return the summary."""

    logger.info(f"\n{'=' * 80}")
    logger.info(f"RUNNING EVALUATION [{label}]")
    logger.info(f"{'=' * 80}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {}
    all_predictions: List[Dict[str, Any]] = []

    for dataset_dir, questions_file, images_dir in datasets:
        logger.info(f"\n{'=' * 80}")
        split_name = dataset_dir.parent.parent.name if dataset_dir.parent.parent else "dataset"
        category = dataset_dir.parent.name if dataset_dir.parent else ""
        dataset_base = dataset_dir.name
        dataset_key = "::".join(filter(None, [split_name, category, dataset_base]))
        display_name = "/".join(filter(None, [split_name, category, dataset_base]))

        logger.info(f"[{label}] Evaluating: {display_name}")
        logger.info(f"{'=' * 80}")

        predictions, metadata = evaluator.evaluate_dataset(
            dataset_dir,
            questions_file,
            images_dir
        )

        metrics_by_task: Dict[str, Dict[str, Any]] = {}
        for task_type in metadata["task_types"]:
            task_predictions = [
                sample
                for sample in predictions
                if sample.get("TaskTypeCanonical") == task_type
                or canonical_task_type(sample.get("TaskType")) == task_type
            ]
            if task_predictions:
                metrics_by_task[task_type] = evaluator.calculate_metrics(
                    task_predictions,
                    task_type
                )

        metadata['split'] = split_name
        metadata['category'] = category
        metadata['dataset_base_name'] = dataset_base
        metadata['display_name'] = display_name
        metadata['dataset_key'] = dataset_key

        all_results[dataset_key] = {
            "metadata": metadata,
            "metrics_by_task": metrics_by_task
        }

        all_predictions.extend(predictions)

        pred_filename = "_".join(filter(None, [split_name, category, dataset_base]))
        pred_file = output_dir / f"{pred_filename}_predictions.json"
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Saved predictions to {pred_file}")

        logger.info(f"\nMetrics for {display_name} [{label}]:")
        logger.info(f"Dataset: {display_name}")
        for task_type, metrics in metrics_by_task.items():
            logger.info(f"Task Type: {task_type}")
            for metric_name, metric_value in metrics.items():
                logger.info(
                    "  %s: %s",
                    metric_name,
                    _format_metric_for_log(metric_name, metric_value)
                )

    all_pred_file = output_dir / "all_predictions.json"
    with open(all_pred_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    logger.info(f"\nSaved all predictions to {all_pred_file}")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"AGGREGATE METRICS BY TASK TYPE [{label}]")
    logger.info(f"{'=' * 80}")

    task_type_groups: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for dataset_key, results in all_results.items():
        display_name = results["metadata"].get("display_name", dataset_key)
        for task_type, metrics in results["metrics_by_task"].items():
            canonical_key = canonical_task_type(task_type)
            task_type_groups[canonical_key].append((display_name, metrics))

    aggregate_metrics: Dict[str, Dict[str, float]] = {}
    for task_type, dataset_results in task_type_groups.items():
        logger.info(f"\n{task_type}:")
        logger.info(f"  Datasets: {', '.join(sorted({name for name, _ in dataset_results}))}")

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
                    sum_metrics[metric_name] = sum_metrics.get(metric_name, 0.0) + float(metric_value)
                    continue

                if sample_weight > 0:
                    total, weight = weighted_metrics.get(metric_name, (0.0, 0.0))
                    weighted_metrics[metric_name] = (
                        total + float(metric_value) * sample_weight,
                        weight + sample_weight
                    )
                else:
                    total, count = fallback_metrics.get(metric_name, (0.0, 0))
                    fallback_metrics[metric_name] = (
                        total + float(metric_value),
                        count + 1
                    )

        agg_metrics: Dict[str, float] = {}

        for metric_name, total in sum_metrics.items():
            if metric_name in {"valid_samples", "correct_predictions"}:
                agg_metrics[metric_name] = float(total)
            else:
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
            agg_metrics["accuracy"] = (
                agg_metrics["correct_predictions"] / agg_metrics["valid_samples"]
            )

        # Cast counts to integers for cleaner reporting
        for key in {"valid_samples", "correct_predictions"}:
            if key in agg_metrics:
                agg_metrics[key] = int(round(agg_metrics[key]))

        aggregate_metrics[task_type] = agg_metrics

        for metric_name, metric_value in agg_metrics.items():
            logger.info(
                "    %s: %s",
                metric_name,
                _format_metric_for_log(metric_name, metric_value)
            )

    total_samples = sum(r["metadata"]["total_samples"] for r in all_results.values())

    summary = {
        "label": label,
        "model_path": evaluator.model_path,
        "num_datasets": len(datasets),
        "total_samples": total_samples,
        "per_dataset_results": all_results,
        "aggregate_metrics_by_task": aggregate_metrics
    }

    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved evaluation summary to {summary_file}")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"EVALUATION COMPLETE [{label}]")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total datasets evaluated: {len(datasets)}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Results saved to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="FLARE25 Evaluation for Qwen3-VL")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to organized_dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--baseline_model_path",
        type=str,
        default=None,
        help="Optional reference checkpoint to evaluate for comparison"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = FLARE25Evaluator(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )

    # Find all evaluation datasets (validation + testing)
    dataset_path = Path(args.dataset_path)
    datasets = find_evaluation_datasets(dataset_path)

    if not datasets:
        logger.error("No validation datasets found")
        return

    # Run evaluation for fine-tuned model
    summaries: Dict[str, Any] = {}
    primary_summary = run_model_evaluation(
        evaluator=evaluator,
        datasets=datasets,
        output_dir=output_dir,
        label="finetuned"
    )
    primary_summary["dataset_path"] = args.dataset_path
    summaries["finetuned"] = primary_summary

    # Optionally evaluate baseline model for comparison
    if args.baseline_model_path:
        baseline_output_dir = output_dir / "baseline"
        baseline_evaluator = FLARE25Evaluator(
            model_path=args.baseline_model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens
        )

        baseline_summary = run_model_evaluation(
            evaluator=baseline_evaluator,
            datasets=datasets,
            output_dir=baseline_output_dir,
            label="baseline"
        )
        baseline_summary["dataset_path"] = args.dataset_path
        summaries["baseline"] = baseline_summary

        comparison_file = output_dir / "comparison_summary.json"
        with open(comparison_file, 'w') as f:
            json.dump(summaries, f, indent=2)
        logger.info(f"\nSaved comparison summary to {comparison_file}")

        # Optionally generate visualizations for the baseline results
        try:
            import subprocess
            viz_script = Path(__file__).parent / "visualize_results.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(viz_script),
                    "--results_dir",
                    str(baseline_output_dir)
                ],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info("✓ Baseline visualizations generated successfully")
                logger.info(result.stdout)
            else:
                logger.warning(
                    f"Baseline visualization generation failed: {result.stderr}"
                )
        except Exception as e:
            logger.warning(f"Could not generate baseline visualizations: {e}")
            logger.info(
                "You can manually generate them with: python evaluation/visualize_results.py "
                f"--results_dir {baseline_output_dir}"
            )

    # Generate visualizations
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING VISUALIZATIONS")
    logger.info(f"{'='*80}")

    try:
        import subprocess
        viz_script = Path(__file__).parent / "visualize_results.py"
        result = subprocess.run(
            [sys.executable, str(viz_script), "--results_dir", str(output_dir)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info("✓ Visualizations generated successfully")
            logger.info(result.stdout)
        else:
            logger.warning(f"Visualization generation failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {e}")
        logger.info(f"You can manually generate them with: python evaluation/visualize_results.py --results_dir {output_dir}")


if __name__ == "__main__":
    main()
