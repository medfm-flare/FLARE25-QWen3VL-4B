# FLARE25 Evaluation Results Visualization

This guide explains how to visualize the evaluation results from your trained Qwen3-VL model.

## Quick Start

After evaluation completes, visualizations are **automatically generated**. You'll find them in `evaluation_results/`:

```bash
ls evaluation_results/*.png
```

Output files:
- `eval_results_per_dataset.png` - Bar plot showing performance for each dataset
- `eval_results_by_task_type.png` - Average performance grouped by task type
- `eval_results_detailed_comparison.png` - Detailed metrics comparison
- `eval_results_summary.png` - Summary statistics and top performers

## Manual Visualization

If you need to regenerate plots or visualize results from a different directory:

```bash
# Basic usage
python evaluation/visualize_results.py --results_dir ./evaluation_results

# Custom output location
python evaluation/visualize_results.py \
    --results_dir ./evaluation_results \
    --output_dir ./my_plots
```

## Visualization Details

### 1. Per-Dataset Bar Plot (`eval_results_per_dataset.png`)

**What it shows:**
- Individual performance of each dataset
- Primary metric for each task type (accuracy, F1, MAE, etc.)
- Color-coded by task type

**How to read:**
- Taller bars = better performance (for accuracy/F1)
- Shorter bars = better performance (for MAE/RMSE)
- Each dataset is labeled on the x-axis
- Metric values are shown on top of each bar

**Color Legend:**
- Blue: Classification
- Purple: Multi-label Classification
- Orange: Detection
- Red: Instance Detection
- Green: Counting
- Dark Red: Regression
- Light Purple: Report Generation

### 2. Task Type Bar Plot (`eval_results_by_task_type.png`)

**What it shows:**
- Average performance across all datasets for each task type
- Helps identify which task types the model performs best/worst on

**How to read:**
- One bar per task type
- Value on top shows the average metric
- Metric name shown in parentheses

### 3. Detailed Comparison (`eval_results_detailed_comparison.png`)

**What it shows:**
- Grouped bar plot with 2 metrics per task type
- Allows comparing related metrics (e.g., Accuracy vs Balanced Accuracy)

**How to read:**
- Each task type has 2 bars side-by-side
- Metric names are labeled above each bar
- Useful for understanding metric relationships

### 4. Summary Statistics (`eval_results_summary.png`)

**What it shows:**
- Key evaluation statistics
- Total datasets and samples evaluated
- Average performance across all metrics
- Top 5 best performing datasets

**How to read:**
- Text-based summary figure
- Quick overview of overall performance
- Identifies strong datasets

## Interpreting Results

### Classification Tasks

**Good Performance:**
- Accuracy > 0.80
- Balanced Accuracy ≈ Accuracy (model not biased)

**Needs Improvement:**
- Accuracy < 0.70
- Large gap between Accuracy and Balanced Accuracy (class imbalance issues)

### Detection Tasks

**Good Performance:**
- F1@IoU=0.3 > 0.65
- F1@IoU=0.5 > 0.50

**Needs Improvement:**
- F1@IoU=0.3 < 0.50 (model struggles with localization)

### Counting/Regression Tasks

**Good Performance:**
- MAE < 2.0
- RMSE < 3.0

**Needs Improvement:**
- MAE > 5.0 (large counting errors)

### Report Generation

**Good Performance:**
- BLEU > 0.30
- ROUGE-L > 0.40

**Needs Improvement:**
- BLEU < 0.20 (poor text quality)

## Common Use Cases

### 1. Identify Weak Datasets

Look at the per-dataset plot to find the lowest bars:

```bash
# Open the per-dataset plot
xdg-open evaluation_results/eval_results_per_dataset.png  # Linux
# or
open evaluation_results/eval_results_per_dataset.png     # macOS
```

Datasets with low performance may need:
- More training data
- Different data augmentation
- Task-specific fine-tuning

### 2. Compare Task Performance

Check the task-type plot:

```bash
xdg-open evaluation_results/eval_results_by_task_type.png
```

If certain task types perform poorly:
- Adjust training hyperparameters for those tasks
- Increase data sampling rate for underperforming tasks
- Check data quality for those task types

### 3. Track Training Progress

After retraining with improved settings:

```bash
# Rename previous results
mv evaluation_results evaluation_results_v1

# Run new evaluation
bash evaluation/run_evaluation.sh

# Compare plots side-by-side
xdg-open evaluation_results_v1/eval_results_by_task_type.png &
xdg-open evaluation_results/eval_results_by_task_type.png &
```

### 4. Generate Custom Plots

Modify `visualize_results.py` to create custom visualizations:

```python
# Add your custom plotting function
def create_custom_plot(summary: Dict, output_path: Path):
    # Your plotting code here
    pass

# Call it in main()
create_custom_plot(summary, output_dir / "custom_plot.png")
```

## Dependencies

The visualization script requires:

```bash
pip install matplotlib numpy
```

These should already be installed if you ran the training/evaluation pipeline.

## Troubleshooting

### "No module named 'matplotlib'"

Install matplotlib:
```bash
pip install matplotlib
# or with uv
uv pip install matplotlib
```

### Plots look crowded

For datasets with many samples, adjust figure size in `visualize_results.py`:

```python
fig, ax = plt.subplots(figsize=(20, 10))  # Increase width
```

### Want different colors

Modify the `colors_map` dictionary:

```python
colors_map = {
    'Classification': '#YOUR_COLOR_HEX',
    # ...
}
```

### Need different metrics displayed

Edit the metric selection logic:

```python
# In create_per_dataset_barplot()
if task_type == 'Classification':
    primary_metric = 'balanced_accuracy'  # Change from 'accuracy'
    primary_value = metrics.get('balanced_accuracy')
```

## Export for Publications

The plots are saved at 300 DPI by default, suitable for publications.

**Convert to different formats:**

```bash
# Convert PNG to PDF
convert evaluation_results/eval_results_per_dataset.png \
        evaluation_results/eval_results_per_dataset.pdf

# Or use Python
python -c "
from PIL import Image
img = Image.open('evaluation_results/eval_results_per_dataset.png')
img.save('evaluation_results/eval_results_per_dataset.pdf', 'PDF', resolution=300)
"
```

## Integration with WandB

If you want to log plots to Weights & Biases:

```python
import wandb

# Initialize wandb
wandb.init(project="flare25-qwen3vl")

# Log plots
wandb.log({
    "per_dataset_results": wandb.Image("evaluation_results/eval_results_per_dataset.png"),
    "task_type_results": wandb.Image("evaluation_results/eval_results_by_task_type.png"),
})
```

## Questions?

Check the main evaluation guide:
```bash
cat ../../../EVALUATION_GUIDE.md
```

Or the evaluation README:
```bash
cat evaluation/README.md
```
