#!/usr/bin/env python3
"""
Test script for validating FLARE25 data loading pipeline.
Tests data loading, tokenization, and image preprocessing before training.
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict
from PIL import Image

# Add parent directory to path to import qwenvl modules
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from qwenvl.data import data_list


def test_annotation_files(dataset_names: List[str]) -> Dict[str, any]:
    """Test that all annotation files exist and are valid JSON."""
    print("=" * 80)
    print("Testing Annotation Files")
    print("=" * 80)

    results = {
        "success": True,
        "datasets": {},
        "total_samples": 0
    }

    configs = data_list(dataset_names)

    for config in configs:
        dataset_name = Path(config["annotation_path"]).stem.replace("_train", "")
        print(f"\n📁 Dataset: {dataset_name}")
        print(f"   Annotation: {config['annotation_path']}")

        # Check file exists
        if not os.path.exists(config["annotation_path"]):
            print(f"   ❌ ERROR: Annotation file not found!")
            results["success"] = False
            results["datasets"][dataset_name] = {"status": "missing", "samples": 0}
            continue

        # Load and validate JSON
        try:
            with open(config["annotation_path"], "r") as f:
                data = json.load(f)

            num_samples = len(data)
            results["total_samples"] += num_samples

            # Validate first sample structure
            if num_samples > 0:
                sample = data[0]
                required_keys = ["image", "conversations"]
                missing_keys = [key for key in required_keys if key not in sample]

                if missing_keys:
                    print(f"   ⚠️  WARNING: Missing keys: {missing_keys}")
                    results["success"] = False
                else:
                    print(f"   ✅ Valid JSON structure")

                # Check image field type
                if isinstance(sample["image"], list):
                    num_images = len(sample["image"])
                    print(f"   📸 Multi-image sample: {num_images} images")
                else:
                    print(f"   📸 Single-image sample")

                # Check conversations structure
                if len(sample["conversations"]) == 2:
                    human_msg = sample["conversations"][0]
                    gpt_msg = sample["conversations"][1]

                    # Count <image> tags
                    image_tag_count = human_msg["value"].count("<image>")
                    expected_count = num_images if isinstance(sample["image"], list) else 1

                    if image_tag_count == expected_count:
                        print(f"   ✅ Image tags match image count: {image_tag_count}")
                    else:
                        print(f"   ❌ ERROR: Image tag mismatch! Tags: {image_tag_count}, Images: {expected_count}")
                        results["success"] = False

            print(f"   📊 Samples: {num_samples}")
            results["datasets"][dataset_name] = {
                "status": "valid",
                "samples": num_samples,
                "annotation_path": config["annotation_path"]
            }

        except json.JSONDecodeError as e:
            print(f"   ❌ ERROR: Invalid JSON - {e}")
            results["success"] = False
            results["datasets"][dataset_name] = {"status": "invalid_json", "samples": 0}
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results["success"] = False
            results["datasets"][dataset_name] = {"status": "error", "samples": 0}

    return results


def test_image_files(dataset_names: List[str], max_samples_per_dataset: int = 5) -> Dict[str, any]:
    """Test that image files exist and can be loaded."""
    print("\n" + "=" * 80)
    print("Testing Image Files")
    print("=" * 80)

    results = {
        "success": True,
        "total_images_tested": 0,
        "missing_images": [],
        "corrupted_images": [],
        "image_stats": {
            "min_size": None,
            "max_size": None,
            "formats": set()
        }
    }

    configs = data_list(dataset_names)

    for config in configs:
        dataset_name = Path(config["annotation_path"]).stem.replace("_train", "")
        print(f"\n📁 Dataset: {dataset_name}")

        with open(config["annotation_path"], "r") as f:
            data = json.load(f)

        # Test first N samples
        samples_to_test = min(max_samples_per_dataset, len(data))
        print(f"   Testing {samples_to_test} samples...")

        for i, sample in enumerate(data[:samples_to_test]):
            image_paths = sample["image"] if isinstance(sample["image"], list) else [sample["image"]]

            for img_path in image_paths:
                results["total_images_tested"] += 1

                # Check file exists
                if not os.path.exists(img_path):
                    print(f"   ❌ Missing: {img_path}")
                    results["missing_images"].append(img_path)
                    results["success"] = False
                    continue

                # Try to load image
                try:
                    img = Image.open(img_path)
                    img.verify()  # Verify it's a valid image

                    # Re-open for stats (verify closes the file)
                    img = Image.open(img_path)

                    # Update stats
                    img_size = img.size[0] * img.size[1]  # width * height
                    if results["image_stats"]["min_size"] is None or img_size < results["image_stats"]["min_size"]:
                        results["image_stats"]["min_size"] = img_size
                    if results["image_stats"]["max_size"] is None or img_size > results["image_stats"]["max_size"]:
                        results["image_stats"]["max_size"] = img_size

                    results["image_stats"]["formats"].add(img.format)

                except Exception as e:
                    print(f"   ❌ Corrupted: {img_path} - {e}")
                    results["corrupted_images"].append((img_path, str(e)))
                    results["success"] = False

        if samples_to_test > 0:
            print(f"   ✅ Tested {samples_to_test} samples")

    # Print image stats
    print("\n📊 Image Statistics:")
    print(f"   Total images tested: {results['total_images_tested']}")
    print(f"   Min pixels: {results['image_stats']['min_size']}")
    print(f"   Max pixels: {results['image_stats']['max_size']}")
    print(f"   Formats: {', '.join(results['image_stats']['formats'])}")

    return results


def test_data_loading_with_model(dataset_names: List[str]) -> Dict[str, any]:
    """Test data loading with actual model components."""
    print("\n" + "=" * 80)
    print("Testing Data Loading with Model Components")
    print("=" * 80)

    results = {
        "success": True,
        "errors": []
    }

    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model_name = "Qwen/Qwen3-VL-4B-Instruct"
        print(f"\n🔧 Loading processor from {model_name}...")

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            min_pixels=784,
            max_pixels=50176
        )
        print("   ✅ Processor loaded successfully")

        # Test processing a sample
        configs = data_list(dataset_names)
        if configs:
            config = configs[0]
            with open(config["annotation_path"], "r") as f:
                data = json.load(f)

            if data:
                sample = data[0]
                print(f"\n🧪 Testing sample processing...")
                print(f"   Sample ID: {sample.get('id', 'N/A')}")

                # Extract messages
                conversations = sample["conversations"]
                messages = [
                    {"role": "user", "content": conversations[0]["value"]},
                    {"role": "assistant", "content": conversations[1]["value"]}
                ]

                # Load images
                image_paths = sample["image"] if isinstance(sample["image"], list) else [sample["image"]]
                images = []
                for img_path in image_paths:
                    img = Image.open(img_path)
                    images.append(img)

                print(f"   Images: {len(images)}")
                print(f"   Question length: {len(conversations[0]['value'])} chars")
                print(f"   Answer length: {len(conversations[1]['value'])} chars")

                # Apply chat template
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                # Process inputs
                inputs = processor(
                    text=[text],
                    images=images,
                    padding=True,
                    return_tensors="pt"
                )

                print(f"\n📦 Processed inputs:")
                print(f"   Input IDs shape: {inputs['input_ids'].shape}")
                print(f"   Attention mask shape: {inputs['attention_mask'].shape}")
                if 'pixel_values' in inputs:
                    print(f"   Pixel values shape: {inputs['pixel_values'].shape}")
                if 'image_grid_thw' in inputs:
                    print(f"   Image grid thw shape: {inputs['image_grid_thw'].shape}")

                print("   ✅ Sample processed successfully")

    except ImportError as e:
        print(f"   ⚠️  WARNING: Could not import transformers/model - {e}")
        print(f"   This test requires the model to be available.")
        results["errors"].append(str(e))
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        results["success"] = False
        results["errors"].append(str(e))

    return results


def print_summary(annotation_results, image_results, model_results):
    """Print final test summary."""
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_success = (
        annotation_results["success"] and
        image_results["success"] and
        model_results["success"]
    )

    print(f"\n📋 Annotation Tests: {'✅ PASSED' if annotation_results['success'] else '❌ FAILED'}")
    print(f"   Total samples: {annotation_results['total_samples']}")
    print(f"   Datasets: {len(annotation_results['datasets'])}")

    print(f"\n📸 Image Tests: {'✅ PASSED' if image_results['success'] else '❌ FAILED'}")
    print(f"   Images tested: {image_results['total_images_tested']}")
    print(f"   Missing: {len(image_results['missing_images'])}")
    print(f"   Corrupted: {len(image_results['corrupted_images'])}")

    print(f"\n🔧 Model Tests: {'✅ PASSED' if model_results['success'] else '⚠️  WARNINGS'}")
    if model_results["errors"]:
        print(f"   Errors/Warnings: {len(model_results['errors'])}")

    print("\n" + "=" * 80)
    if all_success:
        print("🎉 ALL TESTS PASSED - Ready for training!")
    else:
        print("⚠️  SOME TESTS FAILED - Please fix issues before training")
    print("=" * 80)

    return all_success


def main():
    """Main test function."""
    # Test with FLARE25 starter datasets
    dataset_names = [
        "flare_neojaundice%100",
        "flare_retino%100",
        "flare_busi_det%100",
        "flare_boneresorption%100",
        "flare_bone_marrow%100"
    ]

    print("FLARE25 Data Loading Test")
    print("=" * 80)
    print(f"Testing {len(dataset_names)} datasets:")
    for ds in dataset_names:
        print(f"  - {ds}")
    print("=" * 80)

    # Run tests
    annotation_results = test_annotation_files(dataset_names)
    image_results = test_image_files(dataset_names, max_samples_per_dataset=5)
    model_results = test_data_loading_with_model(dataset_names)

    # Print summary
    all_passed = print_summary(annotation_results, image_results, model_results)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
