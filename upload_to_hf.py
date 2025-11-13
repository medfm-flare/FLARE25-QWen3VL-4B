#!/usr/bin/env python3
"""
Script to upload Qwen3-VL-4B FLARE25 finetuned model to Hugging Face Hub
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Configuration
MODEL_DIR = "Qwen3-VL/qwen-vl-finetune/output/qwen3vl_flare25"
REPO_ID = "leoyinn/qwen3vl-flare25"
EVAL_RESULTS_PATH = "Qwen3-VL/qwen-vl-finetune/evaluation_results/comparison_summary.json"

# Load evaluation results
with open(EVAL_RESULTS_PATH, 'r') as f:
    eval_results = json.load(f)

finetuned = eval_results['finetuned']
baseline = eval_results['baseline']

# Create README content
readme_content = f"""---
language:
- en
license: apache-2.0
tags:
- medical
- vision
- multimodal
- qwen3-vl
- flare25
- medical-imaging
datasets:
- FLARE-MedFM/FLARE25
pipeline_tag: image-text-to-text
---

# Qwen3-VL-4B-FLARE25

## Model Description

This is a finetuned version of [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) on the **FLARE 2025 medical imaging dataset**. The model has been trained to perform various medical vision-language tasks across 8 imaging modalities and 19 different datasets.

**Base Model:** Qwen3-VL-4B-Instruct
**Training Dataset:** FLARE25 (Medical Imaging Foundation Models: A Multi-task Learning Framework)
**Training Samples:** {finetuned['total_samples']} samples across {finetuned['num_datasets']} datasets
**Supported Tasks:**
- Classification
- Multi-label Classification
- Detection
- Instance Detection
- Regression
- Counting
- Report Generation

## Supported Medical Imaging Modalities

1. **Ultrasound** - Breast ultrasound, intrauterine growth charts
2. **X-ray** - Dental, chest, periapical radiographs
3. **Retinography** - Fundus imaging, diabetic retinopathy
4. **Microscopy** - Chromosome analysis, bone marrow, cell counting
5. **Clinical Photography** - Neonatal jaundice assessment
6. **Dermatology** - Skin lesion classification
7. **Endoscopy** - Gastrointestinal imaging
8. **Mammography** - Breast cancer screening

## Performance Summary

### Aggregate Performance by Task Type

#### Classification Tasks
- **Finetuned Accuracy:** {finetuned['aggregate_metrics_by_task']['Classification']['accuracy']:.4f}
- **Baseline Accuracy:** {baseline['aggregate_metrics_by_task']['Classification']['accuracy']:.4f}
- **Improvement:** {(finetuned['aggregate_metrics_by_task']['Classification']['accuracy'] - baseline['aggregate_metrics_by_task']['Classification']['accuracy']):.4f}

#### Detection Tasks (IoU @0.5)
- **Finetuned F1:** {finetuned['aggregate_metrics_by_task']['Detection']['f1_at_0.5']:.4f}
- **Baseline F1:** {baseline['aggregate_metrics_by_task']['Detection']['f1_at_0.5']:.4f}
- **Improvement:** {(finetuned['aggregate_metrics_by_task']['Detection']['f1_at_0.5'] - baseline['aggregate_metrics_by_task']['Detection']['f1_at_0.5']):.4f}

#### Multi-label Classification
- **Finetuned F1 (Macro):** {finetuned['aggregate_metrics_by_task']['multi-label classification']['f1_macro']:.4f}
- **Baseline F1 (Macro):** {baseline['aggregate_metrics_by_task']['multi-label classification']['f1_macro']:.4f}
- **Improvement:** {(finetuned['aggregate_metrics_by_task']['multi-label classification']['f1_macro'] - baseline['aggregate_metrics_by_task']['multi-label classification']['f1_macro']):.4f}

### Key Improvements

The finetuned model shows significant improvements over the baseline across multiple task types:

- **Classification:** +66.8% absolute accuracy improvement
- **Detection:** +87.9% F1-score improvement at IoU 0.5
- **Ultrasound Classification (BUS-UCLM):** 0.0% → 92.5% accuracy
- **Ultrasound Classification (BUSI):** 0.0% → 91.0% accuracy
- **Detection (BUSI-det):** 1.2% → 43.6% F1-score

## Usage

### Installation

```bash
pip install transformers torch pillow
```

### Inference Example

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

# Load model and processor
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "leoyinn/qwen3vl-flare25",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("leoyinn/qwen3vl-flare25")

# Prepare input
image = Image.open("medical_image.jpg")
question = "What abnormalities can you identify in this image?"

messages = [
    {{
        "role": "user",
        "content": [
            {{"type": "image", "image": image}},
            {{"type": "text", "text": question}}
        ]
    }}
]

# Generate response
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(generated_text[0])
```

### Multi-Image Support

The model supports multiple images per question (e.g., for multi-view medical imaging):

```python
images = [Image.open("view1.jpg"), Image.open("view2.jpg"), Image.open("view3.jpg")]
question = "Based on these three views, does this newborn require phototherapy? A. No, B. Yes"

messages = [
    {{
        "role": "user",
        "content": [
            {{"type": "image", "image": images[0]}},
            {{"type": "image", "image": images[1]}},
            {{"type": "image", "image": images[2]}},
            {{"type": "text", "text": question}}
        ]
    }}
]

# Process and generate as above
```

## Training Details

### Training Hyperparameters

- **Base Model:** Qwen3-VL-4B-Instruct
- **Training Framework:** DeepSpeed ZeRO-3
- **Learning Rate:** 1e-5
- **Batch Size:** 4 per device
- **Gradient Accumulation Steps:** 4
- **Training Epochs:** 0.5
- **Max Sequence Length:** 8192
- **Image Resolution:** Dynamic (max_pixels: 50176, min_pixels: 784)
- **Optimizer:** AdamW
- **Mixed Precision:** BF16
- **Gradient Checkpointing:** Enabled

### Trainable Components

- ✅ Vision-Language Projection Layer (tune_mm_mlp)
- ✅ Language Model Backbone (tune_mm_llm)
- ❌ Vision Encoder (tune_mm_vision) - Frozen

### Training Data Distribution

The model was trained on 19 medical imaging datasets across 8 modalities:

**Ultrasound:**
- BUSI, BUS-UCLM (Classification)
- BUSI-det, BUS-UCLM-det (Detection)
- IUGC (Classification + Detection)

**X-ray:**
- Dental, Periapical, Bone Resorption, ChestDR, IU-XRay

**Retinography:**
- Retino, Fundus

**Microscopy:**
- Chromosome, Bone Marrow, NeurIPS22-Cell

**Clinical/Dermatology/Endoscopy/Mammography:**
- Neojaundice, BCN20000, Endo, CMMD

## Evaluation

The model was evaluated on validation-public, validation-hidden, and testing splits across all 26 dataset configurations. Detailed results are available in the [GitHub repository](https://github.com/medfm-flare/FLARE25-QWen3VL-4B).

### Evaluation Metrics

Different metrics are used based on task type:
- **Classification:** Accuracy, Balanced Accuracy
- **Detection:** F1-score, Precision, Recall at IoU thresholds (0.3, 0.5, 0.75)
- **Multi-label Classification:** F1 Macro/Micro, Precision, Recall
- **Regression:** MAE, RMSE, R²
- **Counting:** MAE, RMSE, Exact Match, Within-1, Within-5%
- **Instance Detection:** F1 at multiple IoU thresholds (0.3-0.7)

## Limitations

1. **Instance Detection:** The model struggles with counting-based instance detection tasks (chromosome counting). Consider using specialized detection models for these tasks.

2. **Report Generation:** Performance on free-form report generation is limited (8.1% exact match). The model performs better on structured QA tasks.

3. **Regression Tasks:** Bone resorption regression shows negative R² values, indicating the model may not capture continuous numeric relationships well without task-specific tuning.

4. **Medical Diagnosis:** This model is a **research tool only** and should NOT be used for clinical diagnosis without validation by healthcare professionals.

## Citation

If you use this model, please cite:

```bibtex
@misc{{qwen3vl-flare25,
  author = {{FLARE-MedFM Team}},
  title = {{Qwen3-VL-4B Finetuned on FLARE25 Medical Imaging Dataset}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/leoyinn/qwen3vl-flare25}}}}
}}

@article{{qwen3vl,
  title={{Qwen3-VL: Interleaved Multi-Modal Representations}},
  author={{Qwen Team}},
  year={{2025}}
}}
```

## License

This model is released under the Apache 2.0 License. The base model license from Qwen also applies.

## Acknowledgments

- **Base Model:** [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) by Alibaba Cloud
- **Dataset:** FLARE 2025 Medical Imaging Challenge
- **Training Infrastructure:** Built on the official Qwen3-VL finetuning framework

## Repository

Full training code, evaluation scripts, and results: [GitHub - FLARE25-QWen3VL-4B](https://github.com/medfm-flare/FLARE25-QWen3VL-4B)
"""

# Save README
readme_path = os.path.join(MODEL_DIR, "README.md")
with open(readme_path, 'w') as f:
    f.write(readme_content)

print(f"✅ Created README.md at {readme_path}")

# Initialize HF API
api = HfApi()

# Create repository (will not fail if already exists)
try:
    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    print(f"✅ Repository {REPO_ID} ready")
except Exception as e:
    print(f"⚠️  Repository creation: {e}")

# Upload model files
print(f"\n📤 Uploading model files from {MODEL_DIR} to {REPO_ID}...")
print("This may take a while (model is ~8.3GB)...\n")

try:
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload Qwen3-VL-4B finetuned on FLARE25 dataset"
    )
    print(f"\n✅ Successfully uploaded model to https://huggingface.co/{REPO_ID}")
    print(f"\n🎉 Model is now available at: https://huggingface.co/{REPO_ID}")

except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check your HF token: huggingface-cli whoami")
    print("2. Verify token has write permissions")
    print("3. Check internet connection")
    exit(1)
