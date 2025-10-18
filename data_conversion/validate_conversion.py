#!/usr/bin/env python3
"""
Validate Qwen3-VL Converted Data

Checks that converted JSON files:
- Follow correct schema
- Have valid image paths
- Correct number of <image> tags
- No duplicates
- Generate statistics
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from tqdm import tqdm


class ValidationResults:
    """Track validation results"""
    def __init__(self):
        self.total_files = 0
        self.total_samples = 0
        self.valid_samples = 0
        self.errors = []
        self.warnings = []
        self.image_tag_mismatches = 0
        self.missing_images = 0
        self.duplicate_samples = 0
        self.samples_per_file = {}
        self.images_per_sample_dist = Counter()
        self.conversation_lengths = []

    def add_error(self, file_name: str, sample_idx: int, error_msg: str):
        self.errors.append(f"{file_name}[{sample_idx}]: {error_msg}")

    def add_warning(self, file_name: str, sample_idx: int, warning_msg: str):
        self.warnings.append(f"{file_name}[{sample_idx}]: {warning_msg}")

    def print_summary(self):
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total files validated: {self.total_files}")
        print(f"Total samples: {self.total_samples}")
        print(f"Valid samples: {self.valid_samples}")
        print(f"Invalid samples: {self.total_samples - self.valid_samples}")

        if self.errors:
            print(f"\n❌ ERRORS: {len(self.errors)}")
            for i, error in enumerate(self.errors[:10]):
                print(f"  {i+1}. {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")

        if self.warnings:
            print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            for i, warning in enumerate(self.warnings[:10]):
                print(f"  {i+1}. {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")

        print(f"\nImage Tag Mismatches: {self.image_tag_mismatches}")
        print(f"Missing Images: {self.missing_images}")
        print(f"Duplicate Samples: {self.duplicate_samples}")

        print("\n" + "-" * 80)
        print("SAMPLES PER FILE:")
        print("-" * 80)
        for file_name, count in sorted(self.samples_per_file.items()):
            print(f"  {file_name}: {count} samples")

        print("\n" + "-" * 80)
        print("IMAGES PER SAMPLE DISTRIBUTION:")
        print("-" * 80)
        for num_images, count in sorted(self.images_per_sample_dist.items()):
            print(f"  {num_images} image(s): {count} samples")

        if self.conversation_lengths:
            avg_conv_len = sum(self.conversation_lengths) / len(self.conversation_lengths)
            print("\n" + "-" * 80)
            print("CONVERSATION STATISTICS:")
            print("-" * 80)
            print(f"  Average conversation length: {avg_conv_len:.1f} characters")
            print(f"  Min: {min(self.conversation_lengths)}")
            print(f"  Max: {max(self.conversation_lengths)}")

        success_rate = (self.valid_samples / self.total_samples * 100) if self.total_samples > 0 else 0
        print(f"\n✓ Validation Success Rate: {success_rate:.2f}%")

        if success_rate == 100.0 and not self.errors:
            print("\n✅ ALL CHECKS PASSED!")
        elif self.errors:
            print("\n❌ VALIDATION FAILED - Please fix errors before training")
        else:
            print("\n⚠️  VALIDATION PASSED WITH WARNINGS")

        print("=" * 80)


def count_image_tags(text: str) -> int:
    """Count number of <image> tags in text"""
    return text.count("<image>")


def validate_sample_schema(sample: Dict, file_name: str, sample_idx: int, results: ValidationResults) -> bool:
    """
    Validate that sample follows Qwen3-VL schema

    Expected format:
    {
        "image": ["/path/img1.jpg", "/path/img2.jpg"],
        "conversations": [
            {"from": "human", "value": "<image>\n<image>\nQuestion"},
            {"from": "gpt", "value": "Answer"}
        ]
    }
    """
    is_valid = True

    # Check required keys
    if "image" not in sample:
        results.add_error(file_name, sample_idx, "Missing 'image' key")
        is_valid = False

    if "conversations" not in sample:
        results.add_error(file_name, sample_idx, "Missing 'conversations' key")
        is_valid = False
        return is_valid  # Can't continue without conversations

    # Validate image field
    if "image" in sample:
        if not isinstance(sample["image"], list):
            results.add_error(file_name, sample_idx, f"'image' must be a list, got {type(sample['image'])}")
            is_valid = False
        elif len(sample["image"]) == 0:
            results.add_error(file_name, sample_idx, "'image' list is empty")
            is_valid = False
        else:
            # Check image paths exist
            for img_idx, img_path in enumerate(sample["image"]):
                if not isinstance(img_path, str):
                    results.add_error(file_name, sample_idx, f"Image path {img_idx} is not a string")
                    is_valid = False
                elif not Path(img_path).exists():
                    results.add_error(file_name, sample_idx, f"Image not found: {img_path}")
                    results.missing_images += 1
                    is_valid = False

    # Validate conversations field
    conversations = sample["conversations"]
    if not isinstance(conversations, list):
        results.add_error(file_name, sample_idx, f"'conversations' must be a list, got {type(conversations)}")
        return False

    if len(conversations) != 2:
        results.add_error(file_name, sample_idx, f"Expected 2 conversations (human+gpt), got {len(conversations)}")
        is_valid = False

    # Validate conversation structure
    if len(conversations) >= 1:
        human_conv = conversations[0]
        if not isinstance(human_conv, dict):
            results.add_error(file_name, sample_idx, "First conversation is not a dict")
            is_valid = False
        else:
            if human_conv.get("from") != "human":
                results.add_error(file_name, sample_idx, f"First conversation 'from' should be 'human', got '{human_conv.get('from')}'")
                is_valid = False

            if "value" not in human_conv:
                results.add_error(file_name, sample_idx, "Human conversation missing 'value' key")
                is_valid = False
            else:
                # Count <image> tags
                human_value = human_conv["value"]
                num_image_tags = count_image_tags(human_value)

                if "image" in sample:
                    num_images = len(sample["image"])
                    if num_image_tags != num_images:
                        results.add_error(
                            file_name,
                            sample_idx,
                            f"Image tag mismatch: {num_image_tags} tags but {num_images} images"
                        )
                        results.image_tag_mismatches += 1
                        is_valid = False

                results.conversation_lengths.append(len(human_value))

    if len(conversations) >= 2:
        gpt_conv = conversations[1]
        if not isinstance(gpt_conv, dict):
            results.add_error(file_name, sample_idx, "Second conversation is not a dict")
            is_valid = False
        else:
            if gpt_conv.get("from") != "gpt":
                results.add_error(file_name, sample_idx, f"Second conversation 'from' should be 'gpt', got '{gpt_conv.get('from')}'")
                is_valid = False

            if "value" not in gpt_conv:
                results.add_error(file_name, sample_idx, "GPT conversation missing 'value' key")
                is_valid = False
            elif not gpt_conv["value"].strip():
                results.add_warning(file_name, sample_idx, "GPT response is empty")

            if "value" in gpt_conv:
                results.conversation_lengths.append(len(gpt_conv["value"]))

    return is_valid


def validate_file(file_path: Path, results: ValidationResults) -> int:
    """
    Validate a single converted JSON file

    Returns:
        Number of valid samples in the file
    """
    file_name = file_path.name
    print(f"\nValidating: {file_name}")

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        results.add_error(file_name, -1, f"Invalid JSON: {e}")
        return 0
    except Exception as e:
        results.add_error(file_name, -1, f"Error reading file: {e}")
        return 0

    if not isinstance(data, list):
        results.add_error(file_name, -1, f"Root element must be a list, got {type(data)}")
        return 0

    num_samples = len(data)
    print(f"  Samples: {num_samples}")

    valid_count = 0
    seen_samples = set()

    for idx, sample in enumerate(tqdm(data, desc=f"  Checking {file_name}")):
        results.total_samples += 1

        # Check for duplicates (simple hash of conversation)
        if "conversations" in sample:
            sample_hash = json.dumps(sample["conversations"], sort_keys=True)
            if sample_hash in seen_samples:
                results.add_warning(file_name, idx, "Duplicate sample detected")
                results.duplicate_samples += 1
            seen_samples.add(sample_hash)

        # Validate schema
        if validate_sample_schema(sample, file_name, idx, results):
            valid_count += 1
            results.valid_samples += 1

            # Track image count distribution
            if "image" in sample:
                results.images_per_sample_dist[len(sample["image"])] += 1

    results.samples_per_file[file_name] = num_samples
    print(f"  ✓ Valid samples: {valid_count}/{num_samples}")

    return valid_count


def main():
    parser = argparse.ArgumentParser(
        description="Validate Qwen3-VL converted data files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--converted_dir",
        type=str,
        default="../converted_data",
        help="Directory containing converted JSON files"
    )

    parser.add_argument(
        "--files",
        type=str,
        nargs='+',
        help="Specific files to validate (default: all *.json files)"
    )

    parser.add_argument(
        "--save_report",
        type=str,
        help="Save validation report to JSON file"
    )

    args = parser.parse_args()

    converted_dir = Path(args.converted_dir).resolve()

    if not converted_dir.exists():
        print(f"❌ Error: Directory not found: {converted_dir}")
        return 1

    print("=" * 80)
    print("QWEN3-VL DATA VALIDATION")
    print("=" * 80)
    print(f"Directory: {converted_dir}")

    # Get files to validate
    if args.files:
        files_to_validate = [converted_dir / f for f in args.files]
    else:
        files_to_validate = list(converted_dir.glob("*.json"))
        # Exclude report files
        files_to_validate = [f for f in files_to_validate if "report" not in f.name.lower()]

    if not files_to_validate:
        print(f"❌ No JSON files found in {converted_dir}")
        return 1

    print(f"Files to validate: {len(files_to_validate)}")
    for f in files_to_validate:
        print(f"  - {f.name}")

    # Validate each file
    results = ValidationResults()
    results.total_files = len(files_to_validate)

    for file_path in files_to_validate:
        if not file_path.exists():
            results.add_error(file_path.name, -1, "File not found")
            continue

        validate_file(file_path, results)

    # Print summary
    results.print_summary()

    # Save report if requested
    if args.save_report:
        report_path = Path(args.save_report)
        report = {
            "validation_summary": {
                "total_files": results.total_files,
                "total_samples": results.total_samples,
                "valid_samples": results.valid_samples,
                "invalid_samples": results.total_samples - results.valid_samples,
                "image_tag_mismatches": results.image_tag_mismatches,
                "missing_images": results.missing_images,
                "duplicate_samples": results.duplicate_samples
            },
            "samples_per_file": results.samples_per_file,
            "images_per_sample_distribution": dict(results.images_per_sample_dist),
            "errors": results.errors[:100],  # Limit to first 100 errors
            "warnings": results.warnings[:100]
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Validation report saved to: {report_path}")

    # Return exit code
    if results.errors:
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit(main())
