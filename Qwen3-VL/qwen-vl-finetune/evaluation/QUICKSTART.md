# Quick Start: Evaluate Your Trained Qwen3-VL Model

## TL;DR - Run Evaluation Now

```bash
cd ~/Documents/multimodal/shuolinyin/FLARE25/Qwen3-VL/qwen-vl-finetune
bash evaluation/run_evaluation.sh
```

**That's it!** Results will be in `./evaluation_results/`

---

## What This Does

✅ Loads your trained model from `./output/qwen3vl_flare25`
✅ Evaluates on all 19 FLARE25 validation datasets
✅ Calculates task-specific metrics (Accuracy, F1, MAE, BLEU, etc.)
✅ Saves detailed results and predictions

**Time**: ~2-3 hours
**GPU**: Recommended (CUDA), but CPU works too

---

## View Results

```bash
# Summary of all results
cat evaluation_results/evaluation_summary.json | python -m json.tool

# Quick stats
python -c "
import json
data = json.load(open('evaluation_results/evaluation_summary.json'))
print(f\"Datasets: {data['num_datasets']}\")
print(f\"Total samples: {data['total_samples']}\")
print('\nMetrics by task:')
for task, metrics in data['aggregate_metrics_by_task'].items():
    print(f'\n{task}:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.3f}')
"

# Per-dataset results
ls evaluation_results/*_predictions.json
```

---

## Key Metrics to Check

| Task Type | Primary Metric | Good Score |
|-----------|---------------|------------|
| Classification | `accuracy` | >0.80 |
| Multi-label | `f1_macro` | >0.70 |
| Detection | `f1_at_0.3` | >0.60 |
| Counting | `mae` | <2.0 |
| Report Gen | `bleu` | >0.25 |

---

## Common Issues

### CUDA Out of Memory?
```bash
# Use less memory
MAX_NEW_TOKENS=256 bash evaluation/run_evaluation.sh

# Or use CPU (slower)
DEVICE=cpu bash evaluation/run_evaluation.sh
```

### Custom paths?
```bash
MODEL_PATH=/path/to/model \
DATASET_PATH=/path/to/organized_dataset \
OUTPUT_DIR=/path/to/output \
bash evaluation/run_evaluation.sh
```

### Need help?
```bash
# Test your setup
python evaluation/test_evaluation.py

# Read full docs
cat evaluation/README.md
cat ../../../EVALUATION_GUIDE.md
```

---

## Output Files

After running:

```
evaluation_results/
├── evaluation_summary.json      # ⭐ Main results file
├── all_predictions.json         # All predictions combined
├── bcn20000_predictions.json    # Per-dataset predictions
├── retino_predictions.json
├── neojaundice_predictions.json
└── ... (one per dataset)
```

---

## Next Steps

1. **Check overall accuracy**:
   ```bash
   grep -A 3 '"Classification"' evaluation_results/evaluation_summary.json
   ```

2. **Find best/worst datasets**:
   ```bash
   python -c "
   import json
   data = json.load(open('evaluation_results/evaluation_summary.json'))
   for name, result in data['per_dataset_results'].items():
       metrics = result['metrics']
       if 'accuracy' in metrics:
           print(f'{name}: {metrics[\"accuracy\"]:.3f}')
   " | sort -t: -k2 -n
   ```

3. **Error analysis** - see full guide:
   ```bash
   cat ../../../EVALUATION_GUIDE.md
   ```

---

**Questions?** Read `evaluation/README.md` for detailed documentation.

**Ready?** Just run:
```bash
bash evaluation/run_evaluation.sh
```
