#!/usr/bin/env python3
"""
FLARE25 to Qwen3-VL Data Converter

Converts FLARE25 dataset format to Qwen3-VL expected format:
- Handles single and multi-image samples
- Generates correct number of <image> tags
- Uses absolute paths for images
- Validates image existence
- Provides detailed logging and statistics
"""

import json
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from PIL import Image
from tqdm import tqdm

from dataset_configs import (
    get_starter_datasets,
    get_all_datasets,
    get_dataset_config,
    get_dataset_paths,
    FLARE_DATASETS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('conversion.log')
    ]
)
logger = logging.getLogger(__name__)


class ConversionStats:
    """Track conversion statistics"""
    def __init__(self):
        self.total_samples = 0
        self.converted_samples = 0
        self.skipped_samples = 0
        self.missing_images = 0
        self.corrupted_images = 0
        self.task_type_counts = Counter()
        self.image_count_distribution = Counter()
        self.errors = []

    def add_error(self, error_msg: str):
        self.errors.append(error_msg)
        if len(self.errors) <= 10:  # Only log first 10 errors
            logger.warning(error_msg)

    def print_summary(self):
        print("\n" + "=" * 80)
        print("CONVERSION SUMMARY")
        print("=" * 80)
        print(f"Total samples processed: {self.total_samples}")
        print(f"Successfully converted: {self.converted_samples}")
        print(f"Skipped samples: {self.skipped_samples}")
        print(f"Missing images: {self.missing_images}")
        print(f"Corrupted images: {self.corrupted_images}")

        print("\nTask Type Distribution:")
        for task_type, count in self.task_type_counts.most_common():
            print(f"  {task_type}: {count}")

        print("\nImage Count Distribution:")
        for img_count, sample_count in sorted(self.image_count_distribution.items()):
            print(f"  {img_count} image(s): {sample_count} samples")

        if self.errors:
            print(f"\nTotal errors: {len(self.errors)}")
            if len(self.errors) > 10:
                print(f"(Showing first 10 in log, {len(self.errors) - 10} more suppressed)")

        success_rate = (self.converted_samples / self.total_samples * 100) if self.total_samples > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.2f}%")
        print("=" * 80)


def validate_image(image_path: Path) -> bool:
    """
    Validate that an image exists and can be opened

    Args:
        image_path: Path to the image file

    Returns:
        True if image is valid, False otherwise
    """
    if not image_path.exists():
        return False

    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
        # Reopen after verify (verify closes the file)
        with Image.open(image_path) as img:
            img.load()  # Actually load the image data
        return True
    except Exception as e:
        logger.debug(f"Image validation failed for {image_path}: {e}")
        return False


def convert_sample_to_qwen3vl(
    flare_sample: Dict,
    image_dir: Path,
    stats: ConversionStats
) -> Optional[Dict]:
    """
    Convert a single FLARE sample to Qwen3-VL format

    FLARE format:
    {
        "TaskType": "Classification",
        "Modality": "Digital Camera",
        "ImageName": ["img1.jpg", "img2.jpg"] or "img.jpg",
        "Question": "Question text",
        "Answer": "Answer text",
        "Split": "train"
    }

    Qwen3-VL format:
    {
        "image": ["/abs/path/img1.jpg", "/abs/path/img2.jpg"],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<image>\nQuestion text"
            },
            {
                "from": "gpt",
                "value": "Answer text"
            }
        ]
    }
    """
    stats.total_samples += 1

    # Extract image names (handle both string and list)
    image_names = flare_sample.get("ImageName", [])
    if isinstance(image_names, str):
        image_names = [image_names]
    elif not isinstance(image_names, list):
        stats.add_error(f"Invalid ImageName type: {type(image_names)}")
        stats.skipped_samples += 1
        return None

    if not image_names:
        stats.add_error("Sample has no images")
        stats.skipped_samples += 1
        return None

    # Build absolute paths and validate images
    valid_image_paths = []
    for img_name in image_names:
        # Handle both absolute and relative paths
        if os.path.isabs(img_name):
            img_path = Path(img_name)
        else:
            # Extract just the filename if path includes subdirectories
            img_filename = os.path.basename(img_name)
            img_path = image_dir / img_filename

        # Validate image
        if not img_path.exists():
            stats.missing_images += 1
            stats.add_error(f"Image not found: {img_path}")
            continue

        if not validate_image(img_path):
            stats.corrupted_images += 1
            stats.add_error(f"Corrupted image: {img_path}")
            continue

        valid_image_paths.append(str(img_path.absolute()))

    # Skip sample if no valid images
    if not valid_image_paths:
        stats.skipped_samples += 1
        return None

    # Generate <image> tags (one per image)
    image_tags = "\n".join(["<image>"] * len(valid_image_paths))

    # Get question and answer (convert answer to string to handle both text and numerical answers)
    question = str(flare_sample.get("Question", "")).strip()
    answer = str(flare_sample.get("Answer", "")).strip()

    if not question or not answer or answer == "None":
        stats.add_error(f"Missing question or answer in sample")
        stats.skipped_samples += 1
        return None

    # Build human message: image tags followed by question
    human_message = f"{image_tags}\n{question}"

    # Create Qwen3-VL format
    qwen3vl_sample = {
        "image": valid_image_paths,
        "conversations": [
            {
                "from": "human",
                "value": human_message
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
    }

    # Update statistics
    stats.converted_samples += 1
    stats.task_type_counts[flare_sample.get("TaskType", "Unknown")] += 1
    stats.image_count_distribution[len(valid_image_paths)] += 1

    return qwen3vl_sample


def convert_dataset(
    base_dir: Path,
    dataset_name: str,
    split: str,
    output_dir: Path,
    stats: ConversionStats
) -> Tuple[int, str]:
    """
    Convert a complete dataset

    Args:
        base_dir: Base directory containing organized_dataset/
        dataset_name: Name of the dataset (e.g., 'neojaundice')
        split: 'train', 'val', or 'test'
        output_dir: Directory to save converted JSON
        stats: ConversionStats object to track progress

    Returns:
        Tuple of (num_converted_samples, output_file_path)
    """
    logger.info(f"Converting {dataset_name} ({split} split)...")

    # Get paths
    try:
        paths = get_dataset_paths(str(base_dir), dataset_name, split)
    except Exception as e:
        logger.error(f"Error getting paths for {dataset_name} {split}: {e}")
        return 0, ""

    annotation_file = paths["annotation_file"]
    image_dir = paths["image_dir"]

    # Check if files exist
    if not annotation_file.exists():
        logger.warning(f"Annotation file not found: {annotation_file}")
        return 0, ""

    if not image_dir.exists():
        logger.warning(f"Image directory not found: {image_dir}")
        return 0, ""

    # Load annotations
    try:
        with open(annotation_file, 'r') as f:
            flare_samples = json.load(f)
    except Exception as e:
        logger.error(f"Error loading annotations from {annotation_file}: {e}")
        return 0, ""

    logger.info(f"Loaded {len(flare_samples)} samples from {annotation_file}")

    # Convert samples
    converted_samples = []
    for flare_sample in tqdm(flare_samples, desc=f"Converting {dataset_name} {split}"):
        qwen3vl_sample = convert_sample_to_qwen3vl(flare_sample, image_dir, stats)
        if qwen3vl_sample is not None:
            converted_samples.append(qwen3vl_sample)

    # Save converted data
    if converted_samples:
        output_file = output_dir / f"{dataset_name}_{split}.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(converted_samples, f, indent=2)
            logger.info(f"Saved {len(converted_samples)} samples to {output_file}")
            return len(converted_samples), str(output_file)
        except Exception as e:
            logger.error(f"Error saving to {output_file}: {e}")
            return 0, ""
    else:
        logger.warning(f"No samples converted for {dataset_name} {split}")
        return 0, ""


def main():
    parser = argparse.ArgumentParser(
        description="Convert FLARE25 dataset to Qwen3-VL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all starter datasets (train and val)
  python convert_flare_to_qwen3vl.py --base_dir ../organized_dataset --datasets starter

  # Convert specific datasets
  python convert_flare_to_qwen3vl.py --base_dir ../organized_dataset --datasets neojaundice retino

  # Convert only training split
  python convert_flare_to_qwen3vl.py --base_dir ../organized_dataset --datasets starter --splits train

  # Convert all 19 datasets
  python convert_flare_to_qwen3vl.py --base_dir ../organized_dataset --datasets all
        """
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing organized_dataset/"
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=["starter"],
        help="Dataset names to convert. Use 'starter' for 5 starter datasets, 'all' for all 19, or specify names"
    )

    parser.add_argument(
        "--splits",
        type=str,
        nargs='+',
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Which splits to convert (default: train val)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../converted_data",
        help="Output directory for converted JSON files"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve paths
    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Determine which datasets to convert
    if "starter" in args.datasets:
        dataset_names = get_starter_datasets()
        logger.info(f"Converting {len(dataset_names)} starter datasets")
    elif "all" in args.datasets:
        dataset_names = get_all_datasets()
        logger.info(f"Converting all {len(dataset_names)} datasets")
    else:
        dataset_names = args.datasets
        logger.info(f"Converting {len(dataset_names)} specified datasets")

    # Print dataset info
    logger.info(f"Datasets to convert: {', '.join(dataset_names)}")
    logger.info(f"Splits to convert: {', '.join(args.splits)}")

    # Track overall statistics
    overall_stats = ConversionStats()
    conversion_results = []

    # Convert each dataset
    for dataset_name in dataset_names:
        try:
            config = get_dataset_config(dataset_name)
            logger.info(f"\n{'='*80}")
            logger.info(f"Dataset: {dataset_name}")
            logger.info(f"Category: {config.category}")
            logger.info(f"Multi-image: {config.has_multi_image}")
            logger.info(f"{'='*80}")

            dataset_stats = ConversionStats()

            for split in args.splits:
                num_converted, output_file = convert_dataset(
                    base_dir,
                    dataset_name,
                    split,
                    output_dir,
                    dataset_stats
                )

                if num_converted > 0:
                    conversion_results.append({
                        "dataset": dataset_name,
                        "split": split,
                        "samples": num_converted,
                        "output_file": output_file
                    })

            # Merge dataset stats into overall stats
            overall_stats.total_samples += dataset_stats.total_samples
            overall_stats.converted_samples += dataset_stats.converted_samples
            overall_stats.skipped_samples += dataset_stats.skipped_samples
            overall_stats.missing_images += dataset_stats.missing_images
            overall_stats.corrupted_images += dataset_stats.corrupted_images
            overall_stats.task_type_counts.update(dataset_stats.task_type_counts)
            overall_stats.image_count_distribution.update(dataset_stats.image_count_distribution)
            overall_stats.errors.extend(dataset_stats.errors)

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}", exc_info=True)
            continue

    # Print overall summary
    overall_stats.print_summary()

    # Save conversion report
    report_file = output_dir / "conversion_report.json"
    report = {
        "conversion_results": conversion_results,
        "statistics": {
            "total_samples": overall_stats.total_samples,
            "converted_samples": overall_stats.converted_samples,
            "skipped_samples": overall_stats.skipped_samples,
            "missing_images": overall_stats.missing_images,
            "corrupted_images": overall_stats.corrupted_images,
            "task_type_distribution": dict(overall_stats.task_type_counts),
            "image_count_distribution": dict(overall_stats.image_count_distribution),
            "total_errors": len(overall_stats.errors)
        }
    }

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nConversion report saved to: {report_file}")

    # Print file list
    print("\n" + "=" * 80)
    print("CONVERTED FILES:")
    print("=" * 80)
    for result in conversion_results:
        print(f"{result['dataset']} ({result['split']}): {result['samples']} samples")
        print(f"  → {result['output_file']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
