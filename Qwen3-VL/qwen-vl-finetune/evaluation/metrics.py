"""
Task-specific metrics for FLARE25 evaluation
Supports: Classification, Multi-label Classification, Detection, Counting, Regression, Report Generation
"""

import ast
import json
import importlib.util
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict

import numpy as np

logger = logging.getLogger(__name__)


_EXTERNAL_METRICS_MODULE: Optional[Any] = None
_EXTERNAL_METRICS_LOAD_FAILED: bool = False


def _get_external_report_metrics_module() -> Optional[Any]:
    """Load the FLARE25-QWen2.5VL metrics module for advanced report scoring."""
    global _EXTERNAL_METRICS_MODULE, _EXTERNAL_METRICS_LOAD_FAILED

    if _EXTERNAL_METRICS_MODULE is not None:
        return _EXTERNAL_METRICS_MODULE

    if _EXTERNAL_METRICS_LOAD_FAILED:
        return None

    metrics_path = (
        Path(__file__).resolve().parents[4]
        / "FLARE25-QWen2.5VL"
        / "utils"
        / "metrics.py"
    )

    if not metrics_path.exists():
        logger.debug("External metrics module not found at %s", metrics_path)
        _EXTERNAL_METRICS_LOAD_FAILED = True
        return None

    try:
        spec = importlib.util.spec_from_file_location(
            "flare25_external_metrics", metrics_path
        )
        if spec is None or spec.loader is None:
            raise ImportError("Unable to create import spec for external metrics module")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        _EXTERNAL_METRICS_MODULE = module
        return module
    except Exception as exc:  # pragma: no cover - defensive import handling
        logger.warning("Failed to load external report metrics module: %s", exc)
        _EXTERNAL_METRICS_LOAD_FAILED = True
        return None



# ==================================
# Classification Metrics
# ==================================

def calculate_classification_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate classification metrics (Accuracy, Balanced Accuracy)

    Args:
        predictions: List of predicted class labels (e.g., ['A', 'B', 'C'])
        references: List of reference class labels

    Returns:
        dict: Accuracy and balanced accuracy scores
    """
    if not predictions or not references or len(predictions) != len(references):
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "valid_samples": 0}

    # Normalize answers (strip whitespace, convert to uppercase)
    pred_normalized = [str(p).strip().upper() for p in predictions]
    ref_normalized = [str(r).strip().upper() for r in references]

    # Calculate accuracy
    correct = sum(1 for p, r in zip(pred_normalized, ref_normalized) if p == r)
    accuracy = correct / len(predictions)

    # Calculate balanced accuracy (for imbalanced datasets)
    from sklearn.metrics import balanced_accuracy_score
    try:
        balanced_acc = balanced_accuracy_score(ref_normalized, pred_normalized)
    except:
        balanced_acc = accuracy

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "valid_samples": len(predictions),
        "correct_predictions": correct
    }


# ==================================
# Multi-label Classification Metrics
# ==================================

def parse_multi_label_answer(answer: str) -> Set[str]:
    """Parse multi-label answer string into set of labels"""
    if not answer or answer.strip() == "":
        return set()

    # Handle multiple formats: "A,B,C", "A; B; C", "A B C"
    # Split by common delimiters
    labels = re.split(r'[,;\s]+', str(answer).strip().upper())
    return {label for label in labels if label}


def calculate_multilabel_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate multi-label classification metrics (F1, Precision, Recall)

    Args:
        predictions: List of predicted label sets (e.g., ["A,B", "C,D,E"])
        references: List of reference label sets

    Returns:
        dict: F1, precision, recall scores (macro and micro averaged)
    """
    if not predictions or not references or len(predictions) != len(references):
        return {
            "f1_macro": 0.0,
            "f1_micro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "valid_samples": 0
        }

    # Parse label sets
    pred_sets = [parse_multi_label_answer(p) for p in predictions]
    ref_sets = [parse_multi_label_answer(r) for r in references]

    # Calculate per-sample metrics
    sample_f1s = []
    sample_precisions = []
    sample_recalls = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred_set, ref_set in zip(pred_sets, ref_sets):
        if not ref_set:
            continue

        tp = len(pred_set & ref_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Sample-level metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        sample_precisions.append(precision)
        sample_recalls.append(recall)
        sample_f1s.append(f1)

    # Macro-averaged (average across samples)
    f1_macro = np.mean(sample_f1s) if sample_f1s else 0.0
    precision_macro = np.mean(sample_precisions) if sample_precisions else 0.0
    recall_macro = np.mean(sample_recalls) if sample_recalls else 0.0

    # Micro-averaged (aggregate then calculate)
    precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) \
               if (precision_micro + recall_micro) > 0 else 0.0

    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "valid_samples": len(predictions)
    }


# ==================================
# Detection Metrics
# ==================================

def parse_bbox(bbox_str: str) -> Optional[List[float]]:
    """Parse bounding box string to [x1, y1, x2, y2] format"""
    try:
        # Handle formats: "[x1,y1,x2,y2]", "x1,y1,x2,y2", "x1 y1 x2 y2"
        bbox_str = bbox_str.strip().replace('[', '').replace(']', '')
        coords = [float(x.strip()) for x in re.split(r'[,\s]+', bbox_str) if x.strip()]

        if len(coords) == 4:
            return coords
        return None
    except:
        return None


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0.0


def calculate_detection_metrics(
    predictions: List[str],
    references: List[str],
    iou_thresholds: List[float] = [0.3, 0.5, 0.75]
) -> Dict[str, float]:
    """
    Calculate detection metrics (F1, Precision, Recall at multiple IoU thresholds)

    Args:
        predictions: List of detection strings (e.g., "[10,20,30,40]; [50,60,70,80]")
        references: List of reference detection strings
        iou_thresholds: IoU thresholds to evaluate at

    Returns:
        dict: F1, precision, recall at each threshold
    """
    if not predictions or not references or len(predictions) != len(references):
        return {f"f1_at_{t}": 0.0 for t in iou_thresholds}

    results = {}

    for iou_threshold in iou_thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred_str, ref_str in zip(predictions, references):
            # Parse bounding boxes
            pred_bboxes = []
            ref_bboxes = []

            # Split by semicolon or newline for multiple boxes
            for bbox_str in re.split(r'[;\n]', str(pred_str)):
                bbox = parse_bbox(bbox_str)
                if bbox:
                    pred_bboxes.append(bbox)

            for bbox_str in re.split(r'[;\n]', str(ref_str)):
                bbox = parse_bbox(bbox_str)
                if bbox:
                    ref_bboxes.append(bbox)

            # Match predictions to references
            matched_refs = set()
            matched_preds = set()

            for i, pred_bbox in enumerate(pred_bboxes):
                best_iou = 0
                best_ref_idx = -1

                for j, ref_bbox in enumerate(ref_bboxes):
                    if j in matched_refs:
                        continue
                    iou = calculate_iou(pred_bbox, ref_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_ref_idx = j

                if best_iou > iou_threshold:
                    matched_refs.add(best_ref_idx)
                    matched_preds.add(i)
                    total_tp += 1

            total_fp += len(pred_bboxes) - len(matched_preds)
            total_fn += len(ref_bboxes) - len(matched_refs)

        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[f"f1_at_{iou_threshold}"] = f1
        results[f"precision_at_{iou_threshold}"] = precision
        results[f"recall_at_{iou_threshold}"] = recall

    # Average across thresholds (COCO-style mAP)
    results["f1_mean"] = np.mean([results[f"f1_at_{t}"] for t in iou_thresholds])
    results["valid_samples"] = len(predictions)

    return results


# ==================================
# Instance Detection Metrics (for chromosome dataset)
# ==================================

def parse_instance_detection_answer(answer: Any) -> Dict[str, List[List[float]]]:
    """Parse instance detection output into {chromosome_id: [[x1, y1, x2, y2], ...]} format."""

    def _coerce_bbox(raw_box: Any) -> Optional[List[float]]:
        if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
            try:
                return [float(coord) for coord in raw_box]
            except (TypeError, ValueError):
                return None
        if isinstance(raw_box, dict):
            coords = [raw_box.get(key) for key in ("x1", "y1", "x2", "y2")]
            if all(coord is not None for coord in coords):
                try:
                    return [float(coord) for coord in coords]
                except (TypeError, ValueError):
                    return None
        if isinstance(raw_box, str):
            return parse_bbox(raw_box)
        return None

    result: Dict[str, List[List[float]]] = {}

    if answer is None:
        return result

    # If already a dict (e.g., parsed JSON), use directly
    parsed_candidate: Optional[Any]

    if isinstance(answer, dict):
        parsed_candidate = answer
    else:
        text = str(answer).strip()
        if not text:
            return result

        parsed_candidate = None

        try:
            parsed_candidate = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed_candidate = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed_candidate = None

        if not isinstance(parsed_candidate, dict):
            trimmed = (
                text.replace('{', '')
                .replace('}', '')
                .replace('"', '')
                .replace("'", '')
            )
            parts = re.split(r'\s*([A-Za-z0-9]+)\s*:', trimmed)
            for idx in range(1, len(parts), 2):
                chromosome_id = parts[idx].strip()
                bbox_string = parts[idx + 1] if idx + 1 < len(parts) else ""
                for match in re.findall(r'\[[^\[\]]+\]', bbox_string):
                    bbox = _coerce_bbox(match)
                    if bbox:
                        result.setdefault(chromosome_id, []).append(bbox)
            return result

    if isinstance(parsed_candidate, dict):
        for chromosome_id, boxes in parsed_candidate.items():
            bbox_list: List[List[float]] = []
            if isinstance(boxes, dict):
                boxes = [boxes]

            if isinstance(boxes, (list, tuple)):
                for raw_box in boxes:
                    bbox = _coerce_bbox(raw_box)
                    if bbox:
                        bbox_list.append(bbox)

            if bbox_list:
                result[str(chromosome_id).strip()] = bbox_list

    return result


def calculate_instance_detection_metrics(
    predictions: List[str],
    references: List[str],
    iou_thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
) -> Dict[str, Any]:
    """
    Calculate instance detection metrics (for chromosome dataset)
    Matches predictions and references by chromosome identity

    Args:
        predictions: List of instance detection strings
        references: List of reference instance detection strings
        iou_thresholds: IoU thresholds to evaluate at

    Returns:
        dict: F1, precision, recall at each threshold, plus per-chromosome metrics
    """
    if not predictions or not references or len(predictions) != len(references):
        return {f"f1_at_{t}": 0.0 for t in iou_thresholds}

    # Parse predictions and references
    pred_dicts = [parse_instance_detection_answer(p) for p in predictions]
    ref_dicts = [parse_instance_detection_answer(r) for r in references]

    results = {}

    for iou_threshold in iou_thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred_dict, ref_dict in zip(pred_dicts, ref_dicts):
            # Get all chromosome IDs
            all_ids = set(pred_dict.keys()) | set(ref_dict.keys())

            for chromosome_id in all_ids:
                pred_bboxes = pred_dict.get(chromosome_id, [])
                ref_bboxes = ref_dict.get(chromosome_id, [])

                # Match predictions to references for this chromosome
                matched_refs = set()
                matched_preds = set()

                for i, pred_bbox in enumerate(pred_bboxes):
                    best_iou = 0
                    best_ref_idx = -1

                    for j, ref_bbox in enumerate(ref_bboxes):
                        if j in matched_refs:
                            continue
                        iou = calculate_iou(pred_bbox, ref_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_ref_idx = j

                    if best_iou > iou_threshold:
                        matched_refs.add(best_ref_idx)
                        matched_preds.add(i)
                        total_tp += 1

                total_fp += len(pred_bboxes) - len(matched_preds)
                total_fn += len(ref_bboxes) - len(matched_refs)

        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[f"f1_at_{iou_threshold}"] = f1
        results[f"precision_at_{iou_threshold}"] = precision
        results[f"recall_at_{iou_threshold}"] = recall

    # Primary metric for medical imaging
    results["f1_primary"] = results["f1_at_0.3"]
    results["f1_mean"] = np.mean([results[f"f1_at_{t}"] for t in iou_thresholds])
    results["valid_samples"] = len(predictions)

    return results


# ==================================
# Counting Metrics
# ==================================

def extract_number(text: str) -> Optional[float]:
    """Extract the first number from text"""
    try:
        # Find all numbers in text
        numbers = re.findall(r'-?\d+\.?\d*', str(text))
        if numbers:
            return float(numbers[0])
        return None
    except:
        return None


def calculate_counting_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate counting metrics (MAE, RMSE, Accuracy within tolerance)

    Args:
        predictions: List of predicted counts
        references: List of reference counts

    Returns:
        dict: MAE, RMSE, and accuracy metrics
    """
    if not predictions or not references or len(predictions) != len(references):
        return {"mae": float('inf'), "rmse": float('inf'), "accuracy_exact": 0.0}

    # Extract numbers
    pred_nums = [extract_number(p) for p in predictions]
    ref_nums = [extract_number(r) for r in references]

    # Filter valid pairs
    valid_pairs = [(p, r) for p, r in zip(pred_nums, ref_nums) if p is not None and r is not None]

    if not valid_pairs:
        return {"mae": float('inf'), "rmse": float('inf'), "accuracy_exact": 0.0, "valid_samples": 0}

    pred_valid = np.array([p for p, _ in valid_pairs])
    ref_valid = np.array([r for _, r in valid_pairs])

    # Calculate metrics
    mae = np.mean(np.abs(pred_valid - ref_valid))
    rmse = np.sqrt(np.mean((pred_valid - ref_valid) ** 2))

    # Accuracy (exact match and within tolerance)
    exact_match = np.mean(pred_valid == ref_valid)
    within_1 = np.mean(np.abs(pred_valid - ref_valid) <= 1)
    within_5_percent = np.mean(np.abs(pred_valid - ref_valid) / (ref_valid + 1e-6) <= 0.05)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "accuracy_exact": float(exact_match),
        "accuracy_within_1": float(within_1),
        "accuracy_within_5pct": float(within_5_percent),
        "valid_samples": len(valid_pairs)
    }


# ==================================
# Regression Metrics
# ==================================

def calculate_regression_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate regression metrics (MAE, RMSE, R²)

    Args:
        predictions: List of predicted values
        references: List of reference values

    Returns:
        dict: MAE, RMSE, R² scores
    """
    if not predictions or not references or len(predictions) != len(references):
        return {"mae": float('inf'), "rmse": float('inf'), "r2": 0.0}

    # Extract numbers
    pred_nums = [extract_number(p) for p in predictions]
    ref_nums = [extract_number(r) for r in references]

    # Filter valid pairs
    valid_pairs = [(p, r) for p, r in zip(pred_nums, ref_nums) if p is not None and r is not None]

    if not valid_pairs:
        return {"mae": float('inf'), "rmse": float('inf'), "r2": 0.0, "valid_samples": 0}

    pred_valid = np.array([p for p, _ in valid_pairs])
    ref_valid = np.array([r for _, r in valid_pairs])

    # Calculate metrics
    mae = np.mean(np.abs(pred_valid - ref_valid))
    rmse = np.sqrt(np.mean((pred_valid - ref_valid) ** 2))

    # R² score
    ss_res = np.sum((ref_valid - pred_valid) ** 2)
    ss_tot = np.sum((ref_valid - np.mean(ref_valid)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "valid_samples": len(valid_pairs)
    }


# ==================================
# Report Generation Metrics
# ==================================

def calculate_bleu_score(predictions: List[str], references: List[str], n_gram: int = 4) -> float:
    """Calculate BLEU score for generated text"""
    import math

    def get_ngrams(text: str, n: int) -> Counter:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        return Counter(ngrams)

    def modified_precision(prediction: str, reference: str, n: int) -> float:
        pred_ngrams = get_ngrams(prediction, n)
        ref_ngrams = get_ngrams(reference, n)

        if not pred_ngrams:
            return 0.0

        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())

        return overlap / total if total > 0 else 0.0

    # Brevity penalty
    total_pred_len = sum(len(p.split()) for p in predictions)
    total_ref_len = sum(len(r.split()) for r in references)

    if total_pred_len >= total_ref_len:
        bp = 1.0
    elif total_pred_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - total_ref_len / total_pred_len)

    # Calculate n-gram precisions
    precisions = []
    for n in range(1, min(n_gram + 1, 5)):
        precision_scores = [modified_precision(pred, ref, n) for pred, ref in zip(predictions, references)]
        avg_precision = np.mean(precision_scores) if precision_scores else 0.0
        if avg_precision > 0:
            precisions.append(math.log(avg_precision))
        else:
            return 0.0

    if not precisions:
        return 0.0

    bleu = bp * math.exp(sum(precisions) / len(precisions))
    return bleu


def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)"""

    def get_ngrams(text: str, n: int) -> Counter:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        return Counter(ngrams)

    def lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    rouge1_f_scores = []
    rouge2_f_scores = []
    rougeL_f_scores = []

    for pred, ref in zip(predictions, references):
        # ROUGE-1
        pred_unigrams = get_ngrams(pred, 1)
        ref_unigrams = get_ngrams(ref, 1)
        overlap = sum((pred_unigrams & ref_unigrams).values())

        precision = overlap / sum(pred_unigrams.values()) if sum(pred_unigrams.values()) > 0 else 0.0
        recall = overlap / sum(ref_unigrams.values()) if sum(ref_unigrams.values()) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rouge1_f_scores.append(f1)

        # ROUGE-2
        pred_bigrams = get_ngrams(pred, 2)
        ref_bigrams = get_ngrams(ref, 2)
        overlap = sum((pred_bigrams & ref_bigrams).values())

        precision = overlap / sum(pred_bigrams.values()) if sum(pred_bigrams.values()) > 0 else 0.0
        recall = overlap / sum(ref_bigrams.values()) if sum(ref_bigrams.values()) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rouge2_f_scores.append(f1)

        # ROUGE-L
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        lcs_len = lcs_length(pred_tokens, ref_tokens)

        precision = lcs_len / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rougeL_f_scores.append(f1)

    return {
        "rouge1_f": np.mean(rouge1_f_scores),
        "rouge2_f": np.mean(rouge2_f_scores),
        "rougeL_f": np.mean(rougeL_f_scores)
    }


def calculate_report_generation_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate report generation metrics (BLEU, ROUGE)

    Args:
        predictions: List of generated reports
        references: List of reference reports

    Returns:
        dict: BLEU and ROUGE scores
    """
    if not predictions or not references or len(predictions) != len(references):
        return {
            "green_score": 0.0,
            "bleu_score": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "clinical_efficacy": 0.0,
            "valid_samples": 0
        }

    pred_texts = [str(p) if p is not None else "" for p in predictions]
    ref_texts = [str(r) if r is not None else "" for r in references]

    external_module = _get_external_report_metrics_module()
    green_results: Dict[str, Any] = {}
    bleu_score: Optional[float] = None
    clinical_efficacy: Optional[float] = None

    if external_module is not None:
        try:
            green_results = external_module.calculate_green_score(pred_texts, ref_texts)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("External GREEN score calculation failed: %s", exc)
            green_results = {}

        try:
            bleu_score = external_module.calculate_bleu_score(pred_texts, ref_texts)
        except Exception:
            bleu_score = None

        try:
            clinical_efficacy = external_module.calculate_clinical_efficacy_score(
                pred_texts, ref_texts
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("External clinical efficacy calculation failed: %s", exc)
            clinical_efficacy = None

    if bleu_score is None:
        bleu_score = calculate_bleu_score(pred_texts, ref_texts)

    if clinical_efficacy is None:
        clinical_efficacy = 0.0

    rouge_scores = calculate_rouge_scores(pred_texts, ref_texts)

    green_metric_map = {
        "overall_mean": "green_score",
        "entity_matching_mean": "green_entity_matching",
        "location_accuracy_mean": "green_location_accuracy",
        "negation_handling_mean": "green_negation_handling",
        "temporal_accuracy_mean": "green_temporal_accuracy",
        "measurement_accuracy_mean": "green_measurement_accuracy",
        "clinical_significance_mean": "green_clinical_significance",
        "structure_completeness_mean": "green_structure_completeness",
        "severity_correlation_mean": "green_severity_correlation",
    }

    results: Dict[str, float] = {}
    for source_key, metric_name in green_metric_map.items():
        value = green_results.get(source_key, 0.0) if green_results else 0.0
        try:
            results[metric_name] = float(value)
        except (TypeError, ValueError):
            results[metric_name] = 0.0

    results.update({
        "bleu_score": float(bleu_score),
        "clinical_efficacy": float(clinical_efficacy),
        "rouge1_f": float(rouge_scores.get("rouge1_f", 0.0)),
        "rouge2_f": float(rouge_scores.get("rouge2_f", 0.0)),
        "rougeL_f": float(rouge_scores.get("rougeL_f", 0.0)),
        "valid_samples": len(pred_texts)
    })

    return results
