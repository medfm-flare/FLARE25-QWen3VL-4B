# FLARE25 Qwen3-VL-4B Training Guide

This guide covers the complete process of finetuning Qwen3-VL-4B-Instruct on the FLARE25 medical imaging dataset.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Conversion](#data-conversion)
4. [Training Pipeline](#training-pipeline)
5. [Testing & Validation](#testing--validation)
6. [Monitoring Training](#monitoring-training)
7. [Troubleshooting](#troubleshooting)
8. [Scaling to More Datasets](#scaling-to-more-datasets)

---

## Overview

### Training Configuration

- **Model**: Qwen/Qwen3-VL-4B-Instruct
- **Datasets**: 5 FLARE25 starter datasets (4,418 samples)
  - neojaundice (Clinical, multi-image) - 596 samples
  - retino (Retinography) - 1,113 samples
  - BUSI-det (Ultrasound, detection) - 517 samples
  - boneresorption (X-ray, regression) - 702 samples
  - bone_marrow (Microscopy) - 722 samples
- **Hardware**: Optimized for single GPU (24GB+ VRAM)
- **Memory Strategy**: DeepSpeed ZeRO-3 with CPU offloading

### Training Parameters

```bash
Model Size: 4B parameters
Learning Rate: 1e-5
Batch Size: 1 per GPU
Gradient Accumulation: 16 steps
Effective Batch Size: 16
Epochs: 1.0
Max Sequence Length: 4096
Image Resolution: 784 - 50,176 pixels
Optimizer: AdamW (weight_decay=0.01)
LR Schedule: Cosine with 3% warmup
```

---

## Prerequisites

### 1. Environment Setup

**Using `uv` for Environment Management** (Recommended)

```bash
# Navigate to training directory
cd /home/jma/Documents/multimodal/shuolinyin/FLARE25/Qwen3-VL/qwen-vl-finetune

# Install dependencies using uv (fast and efficient)
uv pip install -r requirements.txt

# Or if requirements.txt doesn't exist, install key dependencies
uv pip install torch torchvision transformers accelerate deepspeed wandb pillow

# Verify CUDA availability
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

**Alternative: Traditional pip (if uv is not available)**

```bash
pip install -r requirements.txt
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### 2. Verify Data Conversion

Ensure the FLARE25 datasets have been converted to Qwen3-VL format:

```bash
ls -lh /home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/

# Expected files:
# - neojaundice_train.json
# - retino_train.json
# - BUSI-det_train.json
# - boneresorption_train.json
# - bone_marrow_train.json
```

If conversion hasn't been done yet, see [Data Conversion](#data-conversion) section.

### 3. Verify Dataset Registry

Check that datasets are registered in the Qwen3-VL data loader:

```bash
uv run python -c "from qwenvl.data import data_dict; print('Registered FLARE datasets:'); print([k for k in data_dict.keys() if k.startswith('flare_')])"
```

Expected output:
```
Registered FLARE datasets:
['flare_neojaundice', 'flare_retino', 'flare_busi_det', 'flare_boneresorption', 'flare_bone_marrow']
```

---

## Data Conversion

If you need to convert additional datasets or re-convert existing ones:

### Quick Conversion (5 Starter Datasets)

```bash
cd /home/jma/Documents/multimodal/shuolinyin/FLARE25/data_conversion

# Convert all 5 starter datasets
uv run python convert_flare_to_qwen3vl.py

# Validate conversion
uv run python validate_conversion.py --input_dir ../converted_data
```

### Custom Conversion (Specific Datasets)

```bash
# Convert specific datasets
uv run python convert_flare_to_qwen3vl.py --datasets neojaundice retino

# Convert with specific splits
uv run python convert_flare_to_qwen3vl.py --datasets neojaundice --splits train val

# Verbose output
uv run python convert_flare_to_qwen3vl.py --datasets neojaundice --verbose
```

For full conversion documentation, see `data_conversion/README.md`.

---

## Training Pipeline

### Step 1: Test Data Loading

Before running full training, validate the data loading pipeline:

```bash
cd /home/jma/Documents/multimodal/shuolinyin/FLARE25/Qwen3-VL/qwen-vl-finetune

# Test data loading with model components
uv run python scripts/test_data_loading.py
```

This will:
- ✅ Verify all annotation files exist and are valid JSON
- ✅ Check image files exist and can be loaded
- ✅ Test sample processing with Qwen3-VL processor
- ✅ Validate tokenization and image preprocessing

**Expected Output:**
```
================================================================================
TEST SUMMARY
================================================================================

📋 Annotation Tests: ✅ PASSED
   Total samples: 4418
   Datasets: 5

📸 Image Tests: ✅ PASSED
   Images tested: 25
   Missing: 0
   Corrupted: 0

🔧 Model Tests: ✅ PASSED

================================================================================
🎉 ALL TESTS PASSED - Ready for training!
================================================================================
```

### Step 2: Dry-Run (5 Training Steps)

Run a quick dry-run to validate the end-to-end pipeline without full training:

```bash
# Run dry-run (5 steps only)
bash scripts/dry_run.sh
```

This will:
- Load the model and datasets
- Run 5 training steps
- Verify GPU memory usage
- Exit without saving checkpoint

**Success Indicators:**
- No CUDA out-of-memory errors
- Data loads correctly
- Forward/backward pass completes
- Loss values are reasonable (not NaN)

**Typical Memory Usage:**
- GPU Memory: ~18-22 GB (for 4B model with ZeRO-3 offloading)
- Duration: ~2-5 minutes for 5 steps

If dry-run succeeds, proceed to full training!

### Step 3: Full Training

```bash
# Launch full training
bash scripts/sft_qwen3_4b_flare.sh
```

**Training Configuration:**
- Output directory: `./output/qwen3vl_flare25_starter`
- Checkpoints saved every 200 steps
- Maximum 2 checkpoints retained
- WandB logging enabled (run name: `qwen3vl_flare25_starter`)

**Training Time Estimates:**
- ~10-12 hours on A100 (40GB)
- ~15-18 hours on RTX 4090 (24GB)
- ~20-25 hours on RTX 3090 (24GB)

### Step 4: Monitor Training (Optional)

While training is running, monitor progress in a separate terminal:

```bash
# Real-time monitoring (refreshes every 10 seconds)
uv run python scripts/monitor_training.py --output_dir ./output/qwen3vl_flare25_starter

# Or with custom refresh rate
uv run python scripts/monitor_training.py --output_dir ./output/qwen3vl_flare25_starter --refresh 5

# Just view last 50 lines of logs
uv run python scripts/monitor_training.py --output_dir ./output/qwen3vl_flare25_starter --tail 50
```

**Monitoring Features:**
- ⏱️ Training progress and elapsed time
- 📊 Current loss and average loss
- 🖥️ GPU utilization and memory usage
- 📉 Loss history visualization

---

## Testing & Validation

### Test Data Loading

```bash
# Test data loading pipeline
uv run python scripts/test_data_loading.py
```

Tests performed:
1. **Annotation Tests**: Verify JSON structure, check required keys
2. **Image Tests**: Validate image paths, check image integrity
3. **Model Tests**: Test tokenization and preprocessing with actual model

### Dry-Run Training

```bash
# Run 5 training steps
bash scripts/dry_run.sh
```

Use this to:
- Validate configuration before long training runs
- Test memory usage with different batch sizes
- Debug data loading issues
- Verify model initialization

### Manual Data Inspection

```bash
# Inspect a converted dataset
uv run python -c "
import json
with open('/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/neojaundice_train.json', 'r') as f:
    data = json.load(f)
    print(f'Total samples: {len(data)}')
    print(f'First sample:')
    print(json.dumps(data[0], indent=2))
"
```

---

## Monitoring Training

### Real-Time Monitoring

```bash
# Start monitoring (default: refresh every 10s)
uv run python scripts/monitor_training.py --output_dir ./output/qwen3vl_flare25_starter
```

**Display includes:**
- Training progress (steps, percentage complete)
- Time elapsed and estimated remaining time
- Current loss and moving average
- GPU utilization, memory usage, temperature
- Recent loss history

### WandB Dashboard

If WandB is configured, view training metrics at: https://wandb.ai/

Key metrics to monitor:
- `train/loss`: Should gradually decrease
- `train/learning_rate`: Should follow cosine schedule
- `train/grad_norm`: Should be stable (< 1.0 with clipping)

### TensorBoard (Alternative)

```bash
# If using TensorBoard instead of WandB
tensorboard --logdir ./output/qwen3vl_flare25_starter
```

### Manual Log Inspection

```bash
# View training logs
tail -f ./output/qwen3vl_flare25_starter/*.log

# Or view last N lines
uv run python scripts/monitor_training.py --output_dir ./output/qwen3vl_flare25_starter --tail 100
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

A. Reduce batch size in `scripts/sft_qwen3_4b_flare.sh`:
```bash
batch_size=1  # Already minimal
grad_accum_steps=8  # Reduce from 16
```

B. Reduce image resolution:
```bash
--max_pixels 25088  # Reduce from 50176
```

C. Verify DeepSpeed offloading is enabled:
```bash
# Check scripts/zero3_offload.json has:
"offload_optimizer": {"device": "cpu"}
"offload_param": {"device": "cpu"}
```

D. Use gradient checkpointing (already enabled by default):
```bash
--gradient_checkpointing True
```

#### 2. Dataset Not Found

**Error:**
```
ValueError: do not find flare_neojaundice
```

**Solutions:**

A. Verify dataset is registered in `qwenvl/data/__init__.py`:
```python
uv run python -c "from qwenvl.data import data_dict; print(data_dict.keys())"
```

B. Check annotation paths are correct:
```python
uv run python -c "
from qwenvl.data import data_dict
import os
for name, config in data_dict.items():
    if name.startswith('flare_'):
        path = config['annotation_path']
        exists = os.path.exists(path)
        print(f'{name}: {exists} - {path}')
"
```

C. Re-run data conversion if files are missing:
```bash
cd /home/jma/Documents/multimodal/shuolinyin/FLARE25/data_conversion
uv run python convert_flare_to_qwen3vl.py --datasets neojaundice
```

#### 3. Image Loading Errors

**Error:**
```
PIL.UnidentifiedImageError: cannot identify image file
```

**Solutions:**

A. Validate images:
```bash
cd /home/jma/Documents/multimodal/shuolinyin/FLARE25/data_conversion
uv run python validate_conversion.py --input_dir ../converted_data
```

B. Check for corrupted images:
```bash
uv run python -c "
import json
from PIL import Image
from pathlib import Path

with open('/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/neojaundice_train.json') as f:
    data = json.load(f)

for i, sample in enumerate(data[:10]):
    images = sample['image'] if isinstance(sample['image'], list) else [sample['image']]
    for img_path in images:
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception as e:
            print(f'Sample {i}, Image {img_path}: {e}')
"
```

#### 4. DeepSpeed Initialization Error

**Error:**
```
AssertionError: DeepSpeed requires a distributed environment
```

**Solution:**

Use `torchrun` (not `python`) to launch training:
```bash
# Correct
torchrun --nproc_per_node=1 qwenvl/train/train_qwen.py ...

# Wrong
python qwenvl/train/train_qwen.py ...
```

#### 5. Loss is NaN

**Possible Causes & Solutions:**

A. Learning rate too high:
```bash
lr=5e-6  # Reduce from 1e-5
```

B. Mixed precision issues:
```bash
--bf16  # Use bf16 instead of fp16
```

C. Gradient explosion:
```bash
--max_grad_norm 0.5  # Reduce from 1.0
```

D. Data issues - check for invalid labels:
```bash
uv run python scripts/test_data_loading.py
```

---

## Scaling to More Datasets

### Adding More FLARE Datasets

After validating the 5-starter dataset approach, you can scale to all 19 FLARE datasets.

#### Step 1: Convert Additional Datasets

```bash
cd /home/jma/Documents/multimodal/shuolinyin/FLARE25/data_conversion

# List available datasets
uv run python -c "
from dataset_configs import FLARE_DATASETS
print('Available FLARE datasets:')
for name in FLARE_DATASETS.keys():
    print(f'  - {name}')
"

# Convert all datasets
uv run python convert_flare_to_qwen3vl.py --all

# Or convert specific datasets
uv run python convert_flare_to_qwen3vl.py --datasets path polyp isic2019 ...
```

#### Step 2: Register New Datasets

Add entries to `qwenvl/data/__init__.py`:

```python
# Example for 'path' dataset
FLARE_PATH = {
    "annotation_path": "/home/jma/Documents/multimodal/shuolinyin/FLARE25/converted_data/path_train.json",
    "data_path": "",
}

data_dict = {
    # ... existing entries
    "flare_path": FLARE_PATH,
}
```

#### Step 3: Update Training Script

Modify `scripts/sft_qwen3_4b_flare.sh`:

```bash
# Add new datasets with sampling rates
datasets="flare_neojaundice%100,flare_retino%100,flare_busi_det%100,flare_boneresorption%100,flare_bone_marrow%100,flare_path%100,flare_polyp%100,..."

# May need to adjust training epochs
--num_train_epochs 2.0  # Increase from 1.0
```

#### Step 4: Adjust for Larger Dataset

With more data, you may need to adjust:

```bash
# Increase training time
--num_train_epochs 2.0

# Save checkpoints less frequently
--save_steps 500  # Increase from 200

# Adjust batch size if needed
grad_accum_steps=32  # Increase for larger effective batch
```

### Training Strategy for All Datasets

**Option 1: Single Training Run (All 19 Datasets)**
- Pros: Simple, single checkpoint
- Cons: Long training time (~50-60 hours)

**Option 2: Curriculum Learning (Progressive)**
1. Train on 5 starter datasets → checkpoint_1
2. Continue from checkpoint_1 with 10 more datasets → checkpoint_2
3. Continue from checkpoint_2 with remaining 4 datasets → final

**Option 3: Task-Specific Models**
- Train separate models for each modality/task type
- Better performance on specific tasks
- More models to manage

---

## Advanced Configuration

### Multi-GPU Training

For faster training with multiple GPUs:

```bash
# Modify in sft_qwen3_4b_flare.sh
NPROC_PER_NODE=4  # Number of GPUs

# Adjust batch size
batch_size=2  # Increase per GPU
grad_accum_steps=8  # Reduce since more GPUs
```

### Custom Image Resolution

```bash
# For higher quality (requires more memory)
--max_pixels 100352  # Increase from 50176
--min_pixels 1568    # Increase from 784

# For lower memory usage
--max_pixels 25088   # Decrease from 50176
--min_pixels 392     # Decrease from 784
```

### Learning Rate Tuning

```bash
# More aggressive training
lr=2e-5              # Increase from 1e-5
--warmup_ratio 0.05  # Increase from 0.03

# More conservative training
lr=5e-6              # Decrease from 1e-5
--warmup_ratio 0.1   # Increase from 0.03
```

### Evaluation During Training

```bash
# Enable evaluation
--eval_strategy steps
--eval_steps 500
--per_device_eval_batch_size 1

# Need to set validation dataset in data/__init__.py
```

---

## Output Files

After training completes, the output directory will contain:

```
output/qwen3vl_flare25_starter/
├── checkpoint-200/          # Intermediate checkpoint
│   ├── config.json
│   ├── model.safetensors
│   ├── processor_config.json
│   └── ...
├── checkpoint-400/          # Latest checkpoint
│   └── ...
├── trainer_state.json       # Training state
├── training_args.bin        # Training arguments
├── *.log                    # Training logs
└── wandb/                   # WandB logs (if enabled)
```

### Using the Trained Model

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Load your trained model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./output/qwen3vl_flare25_starter/checkpoint-400",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "./output/qwen3vl_flare25_starter/checkpoint-400",
    trust_remote_code=True
)

# Use for inference
# ... (see inference examples in main README)
```

---

## Best Practices

1. **Always run dry-run first**: Validate configuration before long training runs
2. **Monitor GPU memory**: Adjust batch size if close to limits
3. **Use WandB/TensorBoard**: Track experiments and compare hyperparameters
4. **Save experiment configs**: Document changes for reproducibility
5. **Validate data quality**: Run test_data_loading.py after any data changes
6. **Regular checkpoints**: Keep at least 2-3 checkpoints in case of issues
7. **Test on validation set**: Evaluate periodically during training

---

## Support & Resources

### Documentation
- Main README: `../README.md`
- Data Conversion Guide: `../../data_conversion/README.md`
- Qwen3-VL Official Docs: https://github.com/QwenLM/Qwen3-VL

### Scripts
- `scripts/test_data_loading.py`: Test data pipeline
- `scripts/dry_run.sh`: Quick validation run
- `scripts/monitor_training.py`: Real-time monitoring
- `scripts/sft_qwen3_4b_flare.sh`: Full training script

### Troubleshooting
- Check GPU memory: `nvidia-smi`
- Verify CUDA: `uv run python -c "import torch; print(torch.cuda.is_available())"`
- Test data loading: `uv run python scripts/test_data_loading.py`
- Validate conversion: `cd data_conversion && uv run python validate_conversion.py`

---

## Quick Command Reference

```bash
# Test data loading
uv run python scripts/test_data_loading.py

# Dry-run (5 steps)
bash scripts/dry_run.sh

# Full training
bash scripts/sft_qwen3_4b_flare.sh

# Monitor training
uv run python scripts/monitor_training.py --output_dir ./output/qwen3vl_flare25_starter

# Convert more datasets
cd ../../data_conversion
uv run python convert_flare_to_qwen3vl.py --datasets dataset_name

# Validate conversion
uv run python validate_conversion.py --input_dir ../converted_data

# Check GPU status
nvidia-smi

# View logs
tail -f ./output/qwen3vl_flare25_starter/*.log
```

---

**Happy Training! 🚀**
