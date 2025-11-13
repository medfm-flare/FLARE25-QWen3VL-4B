# FLARE25 Evaluation Pipeline for Qwen3-VL

This directory contains the evaluation pipeline for assessing trained Qwen3-VL models on the FLARE25 medical imaging benchmark.

## Overview

The evaluation pipeline supports all FLARE25 task types with appropriate metrics:

| Task Type | Metrics | Description |
|-----------|---------|-------------|
| **Classification** | Accuracy, Balanced Accuracy | Single-label classification (A, B, C, etc.) |
| **Multi-label Classification** | F1 (Macro/Micro), Precision, Recall | Multiple labels per sample |
| **Detection** | F1@IoU, Precision, Recall | Bounding box detection at multiple IoU thresholds |
| **Instance Detection** | F1@IoU (per-class), mAP | Detection with class-specific matching (chromosome) |
| **Counting** | MAE, RMSE, Accuracy | Count prediction with tolerance |
| **Regression** | MAE, RMSE, R² | Continuous value prediction |
| **Report Generation** | BLEU, ROUGE-1/2/L | Medical report generation |

## Quick Start

### 1. Basic Evaluation

```bash
# From qwen-vl-finetune directory
cd /home/jma/Documents/multimodal/shuolinyin/FLARE25/Qwen3-VL/qwen-vl-finetune

# Run evaluation on your trained model
bash evaluation/run_evaluation.sh
```

By default, this evaluates:
- **Model**: `./output/qwen3vl_flare25` (your trained model)
- **Dataset**: `../../../organized_dataset` (FLARE25 validation-public)
- **Output**: `./evaluation_results`

### 2. Custom Paths

```bash
# Set custom paths via environment variables
MODEL_PATH=./output/my_model \
DATASET_PATH=/path/to/organized_dataset \
OUTPUT_DIR=./my_results \
bash evaluation/run_evaluation.sh
```

### 3. Direct Python Usage

```bash
python evaluation/evaluate_flare25.py \
    --model_path ./output/qwen3vl_flare25 \
    --dataset_path ../../../organized_dataset \
    --output_dir ./evaluation_results \
    --max_new_tokens 512 \
    --device cuda
```

## File Structure

```
evaluation/
├── README.md                    # This file
├── metrics.py                   # Task-specific metric implementations
├── evaluate_flare25.py          # Main evaluation script
├── run_evaluation.sh            # Convenience runner script
└── evaluation_results/          # Output directory (created after run)
    ├── evaluation_summary.json  # Overall results summary
    ├── all_predictions.json     # All predictions in FLARE format
    └── {dataset}_predictions.json  # Per-dataset predictions
```

## Output Format

### 1. evaluation_summary.json

```json
{
  "model_path": "./output/qwen3vl_flare25",
  "dataset_path": "../../../organized_dataset",
  "num_datasets": 19,
  "total_samples": 5842,
  "per_dataset_results": {
    "bcn20000": {
      "metadata": {
        "dataset_name": "bcn20000",
        "task_type": "Classification",
        "total_samples": 358,
        "valid_predictions": 358
      },
      "metrics": {
        "accuracy": 0.8547,
        "balanced_accuracy": 0.8423,
        "valid_samples": 358
      }
    },
    ...
  },
  "aggregate_metrics_by_task": {
    "Classification": {
      "accuracy": 0.8234,
      "balanced_accuracy": 0.8156
    },
    "Detection": {
      "f1_at_0.5": 0.6543,
      "precision_at_0.5": 0.6821,
      "recall_at_0.5": 0.6289
    },
    ...
  }
}
```

### 2. Per-dataset Predictions

Each dataset gets a `{dataset_name}_predictions.json` file:

```json
[
  {
    "TaskType": "Classification",
    "Modality": "Dermoscopy",
    "ImageName": ["imagesVal/BCN20000_03396-1.jpg", ...],
    "Question": "Based on the dermatoscopic image, what is the diagnosis? A. ...",
    "Answer": "F",
    "Split": "val",
    "prediction": "F"
  },
  ...
]
```

## Metrics Details

### Classification Metrics

**Accuracy**: Proportion of correct predictions
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Balanced Accuracy**: Average recall per class (handles class imbalance)
```
Balanced Accuracy = mean(Recall_per_class)
```

**When to use**: Single-label classification tasks (e.g., "What is the diagnosis? A/B/C")

---

### Multi-label Classification Metrics

**F1 Score (Macro)**: Average F1 across samples
- Treats each sample equally regardless of label count

**F1 Score (Micro)**: Aggregate true positives/false positives/negatives
- Better for imbalanced label distributions

**When to use**: Tasks where multiple labels can be correct (e.g., "List all abnormalities")

---

### Detection Metrics

**F1@IoU**: F1 score at specific Intersection over Union threshold
- Evaluated at IoU = 0.3, 0.5, 0.75 (medical imaging standards)
- F1@0.3 is primary metric for medical imaging
- F1@0.5 is standard computer vision benchmark

**mAP**: Mean Average Precision across IoU thresholds (COCO-style)

**When to use**: Bounding box detection tasks

---

### Instance Detection Metrics

Same as Detection but with **class-aware matching**:
- Predictions and ground truth must match both spatially (IoU) AND by class ID
- Used for chromosome detection where class identity matters

---

### Counting Metrics

**MAE (Mean Absolute Error)**: Average absolute difference
```
MAE = mean(|predicted - actual|)
```

**RMSE (Root Mean Square Error)**: Penalizes large errors more
```
RMSE = sqrt(mean((predicted - actual)²))
```

**Accuracy**: Exact match and within-tolerance matches
- `accuracy_exact`: Exactly correct count
- `accuracy_within_1`: Within ±1 of correct count
- `accuracy_within_5pct`: Within 5% of correct count

**When to use**: Cell counting, object counting tasks

---

### Regression Metrics

**MAE, RMSE**: Same as counting metrics

**R² (Coefficient of Determination)**: Proportion of variance explained
```
R² = 1 - (SS_residual / SS_total)
```
- R² = 1.0: Perfect prediction
- R² = 0.0: Predicts mean only
- R² < 0.0: Worse than mean

**When to use**: Continuous value prediction (e.g., TSH levels)

---

### Report Generation Metrics

**BLEU**: N-gram overlap with reference (common in machine translation)
- BLEU-4: Considers 1-4 gram overlaps
- Higher is better (0-1 scale)

**ROUGE-1/2/L**: Recall-oriented metrics
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence
- F1 scores reported (balance precision and recall)

**When to use**: Medical report generation tasks

## Interpreting Results

### Good Performance Benchmarks

Based on medical imaging literature and FLARE25 baselines:

| Task | Metric | Good | Very Good | Excellent |
|------|--------|------|-----------|-----------|
| Classification | Accuracy | >0.70 | >0.80 | >0.90 |
| Multi-label | F1 Macro | >0.60 | >0.70 | >0.85 |
| Detection | F1@0.3 | >0.50 | >0.65 | >0.80 |
| Detection | F1@0.5 | >0.40 | >0.55 | >0.70 |
| Counting | MAE | <2.0 | <1.0 | <0.5 |
| Regression | MAE | <0.5 | <0.2 | <0.1 |
| Report Gen | BLEU | >0.20 | >0.30 | >0.40 |

### Per-Task Considerations

**Classification**:
- Check balanced accuracy if dataset is imbalanced
- High accuracy + low balanced accuracy indicates overfitting to majority class

**Detection**:
- F1@0.3 is standard for medical imaging (relaxed localization requirement)
- F1@0.5 is standard for computer vision (stricter requirement)
- Low precision = too many false positives
- Low recall = missing true detections

**Counting**:
- MAE is more interpretable than RMSE
- Check `accuracy_within_1` for practical performance
- Very low exact accuracy may still have good MAE

**Report Generation**:
- BLEU focuses on word choice
- ROUGE focuses on content coverage
- Low scores don't always mean poor clinical quality (different valid phrasings)

## Advanced Usage

### Modify Evaluation Parameters

Edit `evaluate_flare25.py` to customize:

```python
# Line 45: Change generation parameters
def __init__(
    self,
    model_path: str,
    device: str = "cuda",
    max_new_tokens: int = 512,  # Increase for longer reports
    batch_size: int = 1
):
```

```python
# Line 184: Modify generation settings
outputs = self.model.generate(
    **inputs,
    max_new_tokens=self.max_new_tokens,
    do_sample=False,  # Set True for sampling instead of greedy
    temperature=0.7,  # Add temperature for sampling
    top_p=0.9,        # Add nucleus sampling
    pad_token_id=self.tokenizer.pad_token_id,
    eos_token_id=self.tokenizer.eos_token_id
)
```

### Add Custom Metrics

Edit `metrics.py` to add new metric functions:

```python
def calculate_custom_metric(predictions: List[str], references: List[str]) -> float:
    """Your custom metric implementation"""
    # Implementation here
    return metric_value
```

Then integrate in `evaluate_flare25.py`:

```python
# In calculate_metrics method
from metrics import calculate_custom_metric

if task_type == "your_custom_task":
    metrics = calculate_custom_metric(pred_valid, ref_valid)
```

### Evaluate on Specific Datasets

Modify `find_validation_datasets()` to filter:

```python
# Only evaluate specific modality
datasets = []
for questions_file in validation_path.rglob("*_questions_val.json"):
    if "Ultrasound" in str(questions_file):  # Only ultrasound datasets
        # ... rest of the logic
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce max tokens
MAX_NEW_TOKENS=256 bash evaluation/run_evaluation.sh

# Or modify the script to use CPU
DEVICE=cpu bash evaluation/run_evaluation.sh
```

### Image Not Found Errors

Check that:
1. `imagesVal/` directory exists in dataset folders
2. Image paths in JSON match actual filenames
3. Image extensions are correct (.jpg, .png, etc.)

### Wrong Metrics Calculated

Verify `TaskType` field in question JSON files matches expected format:
- "Classification"
- "Multi-label Classification"
- "Detection"
- "Instance Detection"
- "Counting" or "Cell Counting"
- "Regression"
- "Report Generation"

### Low Scores Across All Tasks

Potential issues:
1. **Model not loaded correctly**: Check that `model_path` contains valid checkpoint
2. **Format mismatch**: Model may not understand question format
3. **Insufficient training**: Model needs more training epochs
4. **Data leakage**: Validation set may differ from training distribution

## Citation

If you use this evaluation pipeline, please cite the FLARE25 challenge:

```bibtex
@article{flare2025,
  title={FLARE 2025: Foundation Models Meet Medical Vision-Language Understanding},
  author={FLARE Challenge Organizers},
  year={2025}
}
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the FLARE25 dataset documentation
3. Consult the Qwen3-VL official documentation
4. Open an issue on the challenge forum
