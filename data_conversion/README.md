# FLARE25 to Qwen3-VL Data Conversion

This directory contains tools to convert FLARE25 dataset format to Qwen3-VL expected format.

## Overview

The conversion pipeline:
1. Reads FLARE25 annotations from `organized_dataset/`
2. Validates image existence and integrity
3. Converts to Qwen3-VL conversation format with `<image>` tags
4. Outputs to `converted_data/` directory
5. Generates validation reports

## Quick Start

### 1. Convert Starter Datasets (5 datasets)

```bash
cd data_conversion

# Convert all 5 starter datasets (train + validation splits)
uv run python convert_flare_to_qwen3vl.py \
    --base_dir ../organized_dataset \
    --datasets starter \
    --output_dir ../converted_data
```

This converts:
- neojaundice (Clinical, multi-image)
- retino (Retinography)
- BUSI-det (Ultrasound, detection)
- boneresorption (X-ray)
- bone_marrow (Microscopy)

### 2. Validate Converted Data

```bash
uv run python validate_conversion.py \
    --converted_dir ../converted_data \
    --save_report ../converted_data/validation_report.json
```

### 3. Check Conversion Report

```bash
cat ../converted_data/conversion_report.json
```

## Detailed Usage

### convert_flare_to_qwen3vl.py

**Convert specific datasets:**
```bash
uv run python convert_flare_to_qwen3vl.py \
    --base_dir ../organized_dataset \
    --datasets neojaundice retino \
    --output_dir ../converted_data
```

**Convert only training split:**
```bash
uv run python convert_flare_to_qwen3vl.py \
    --base_dir ../organized_dataset \
    --datasets starter \
    --splits train \
    --output_dir ../converted_data
```

**Convert all 19 datasets:**
```bash
uv run python convert_flare_to_qwen3vl.py \
    --base_dir ../organized_dataset \
    --datasets all \
    --output_dir ../converted_data
```

**Enable verbose logging:**
```bash
uv run python convert_flare_to_qwen3vl.py \
    --base_dir ../organized_dataset \
    --datasets starter \
    --verbose
```

### validate_conversion.py

**Validate all files:**
```bash
uv run python validate_conversion.py --converted_dir ../converted_data
```

**Validate specific files:**
```bash
uv run python validate_conversion.py \
    --converted_dir ../converted_data \
    --files neojaundice_train.json retino_train.json
```

**Save validation report:**
```bash
uv run python validate_conversion.py \
    --converted_dir ../converted_data \
    --save_report ../converted_data/validation_report.json
```

### dataset_configs.py

**View dataset configuration:**
```bash
uv run python dataset_configs.py
```

This displays:
- All 19 FLARE datasets
- Starter datasets (marked with ★)
- Categories and modalities
- Task types
- Multi-image flags

## Data Format

### Input (FLARE Format)

```json
{
  "TaskType": "Classification",
  "Modality": "Digital Camera",
  "ImageName": ["imagesTr/img1.jpg", "imagesTr/img2.jpg"],
  "Question": "Does this require treatment? A. No, B. Yes",
  "Answer": "A",
  "Split": "train"
}
```

### Output (Qwen3-VL Format)

```json
{
  "image": [
    "/absolute/path/to/img1.jpg",
    "/absolute/path/to/img2.jpg"
  ],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n<image>\nDoes this require treatment? A. No, B. Yes"
    },
    {
      "from": "gpt",
      "value": "A"
    }
  ]
}
```

## Output Structure

After conversion, the `converted_data/` directory will contain:

```
converted_data/
├── neojaundice_train.json        # Training data
├── neojaundice_val.json          # Validation data
├── retino_train.json
├── retino_val.json
├── BUSI-det_train.json
├── BUSI-det_val.json
├── boneresorption_train.json
├── boneresorption_val.json
├── bone_marrow_train.json
├── bone_marrow_val.json
├── conversion_report.json         # Statistics and errors
└── validation_report.json         # Quality assurance report
```

## Features

### Robust Error Handling
- Validates image existence before conversion
- Checks image integrity (can be opened by PIL)
- Skips corrupted samples with logging
- Continues processing even if some samples fail
- Generates detailed error reports

### Multi-Image Support
- Correctly handles both single-image and multi-image samples
- Generates exact number of `<image>` tags to match images
- Preserves image order from FLARE format

### Path Management
- Converts all image paths to absolute paths
- Handles both relative and absolute paths in FLARE format
- Works regardless of current working directory

### Statistics & Reporting
- Tracks conversion success rate
- Task type distribution
- Image count distribution
- Missing/corrupted image counts
- Generates JSON reports for analysis

## Validation Checks

The validation script performs these checks:

1. **Schema Validation**
   - Correct JSON structure
   - Required fields present
   - Correct data types

2. **Image Path Validation**
   - All image paths exist
   - Files are readable

3. **Image Tag Validation**
   - Number of `<image>` tags matches number of images
   - Tags appear in human message

4. **Conversation Validation**
   - Two conversations (human + gpt)
   - Correct "from" fields
   - Non-empty responses

5. **Duplicate Detection**
   - Identifies duplicate samples

6. **Statistics**
   - Samples per file
   - Images per sample distribution
   - Conversation length statistics

## Troubleshooting

### "Image not found" errors
- Check that `--base_dir` points to correct location
- Verify `organized_dataset/` directory structure
- Ensure image files haven't been moved

### "Corrupted image" errors
- Some images may be corrupted in the source dataset
- These are automatically skipped
- Check `conversion.log` for details

### Low conversion success rate
- Review errors in `conversion_report.json`
- Check that all image directories exist
- Verify JSON annotation files are valid

### Image tag mismatch errors
- This indicates a bug in conversion logic
- Report the issue with the specific sample

## Next Steps

After successful conversion and validation:

1. **Register datasets** in Qwen3-VL:
   ```bash
   # Edit: Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py
   # Add entries for converted datasets
   ```

2. **Launch training**:
   ```bash
   cd ../Qwen3-VL/qwen-vl-finetune
   bash scripts/sft_qwen3_4b_flare.sh
   ```

3. **Monitor progress**:
   ```bash
   # Check logs, GPU usage, etc.
   nvidia-smi -l 1
   ```

## Dependencies

- Python 3.8+
- Pillow (PIL): Image validation
- tqdm: Progress bars
- Standard library: json, argparse, pathlib, logging

Install dependencies using `uv` (recommended):
```bash
uv pip install Pillow tqdm
```

Or using traditional pip:
```bash
pip install Pillow tqdm
```

## Logs

- **conversion.log**: Detailed conversion logs with timestamps
- **Console output**: Real-time progress and summary
- **conversion_report.json**: Machine-readable statistics
- **validation_report.json**: Machine-readable validation results

## Tips

- Start with `--datasets starter` to test the pipeline
- Always run validation after conversion
- Check success rate - should be >95%
- Review errors if success rate is low
- Use `--verbose` for debugging issues

## Support

For issues or questions:
1. Check logs: `conversion.log`
2. Review reports: `conversion_report.json`
3. Verify dataset structure matches FLARE25 format
4. Check image paths are accessible

## Scaling to All Datasets

Once the 5 starter datasets work well:

```bash
# Convert all 19 datasets
uv run python convert_flare_to_qwen3vl.py \
    --base_dir ../organized_dataset \
    --datasets all \
    --output_dir ../converted_data

# Validate all
uv run python validate_conversion.py --converted_dir ../converted_data
```

This will take longer but follows the same process.
