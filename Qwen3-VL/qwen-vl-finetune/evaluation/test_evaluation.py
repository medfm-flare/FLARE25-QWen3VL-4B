#!/usr/bin/env python3
"""
Quick test script to verify evaluation pipeline
Tests on a small subset of data before full evaluation
"""

import sys
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add evaluation to path
sys.path.insert(0, str(Path(__file__).parent))

def test_metrics():
    """Test all metric functions"""
    from metrics import (
        calculate_classification_metrics,
        calculate_multilabel_metrics,
        calculate_detection_metrics,
        calculate_counting_metrics,
        calculate_regression_metrics,
        calculate_report_generation_metrics
    )

    logger.info("Testing metric functions...")

    # Test classification
    pred = ['A', 'B', 'C', 'A', 'B']
    ref = ['A', 'B', 'C', 'B', 'B']
    metrics = calculate_classification_metrics(pred, ref)
    logger.info(f"✓ Classification metrics: Accuracy = {metrics['accuracy']:.3f}")

    # Test multi-label
    pred = ['A,B', 'C', 'A,B,C']
    ref = ['A,B,C', 'C,D', 'A,B']
    metrics = calculate_multilabel_metrics(pred, ref)
    logger.info(f"✓ Multi-label metrics: F1 Macro = {metrics['f1_macro']:.3f}")

    # Test detection
    pred = ['[10,20,30,40]', '[50,60,70,80]']
    ref = ['[10,20,35,45]', '[50,60,70,80]']
    metrics = calculate_detection_metrics(pred, ref)
    logger.info(f"✓ Detection metrics: F1@0.5 = {metrics['f1_at_0.5']:.3f}")

    # Test counting
    pred = ['5', '10', '3']
    ref = ['5', '12', '3']
    metrics = calculate_counting_metrics(pred, ref)
    logger.info(f"✓ Counting metrics: MAE = {metrics['mae']:.3f}")

    # Test regression
    pred = ['1.5', '2.3', '3.1']
    ref = ['1.6', '2.1', '3.0']
    metrics = calculate_regression_metrics(pred, ref)
    logger.info(f"✓ Regression metrics: MAE = {metrics['mae']:.3f}")

    # Test report generation
    pred = ['This is a test report', 'Another test']
    ref = ['This is a reference report', 'Another reference']
    metrics = calculate_report_generation_metrics(pred, ref)
    logger.info(f"✓ Report generation metrics: BLEU = {metrics['bleu']:.3f}")

    logger.info("\nAll metric functions working correctly!")
    return True


def test_dataset_discovery():
    """Test finding validation datasets"""
    from evaluate_flare25 import find_validation_datasets

    logger.info("\nTesting dataset discovery...")

    # Try to find datasets
    dataset_path = Path(__file__).parent.parent.parent.parent / "organized_dataset"

    if not dataset_path.exists():
        logger.warning(f"Dataset path not found: {dataset_path}")
        logger.info("Skipping dataset discovery test")
        return False

    datasets = find_validation_datasets(dataset_path)

    if datasets:
        logger.info(f"✓ Found {len(datasets)} validation datasets:")
        for dataset_dir, questions_file, images_dir in datasets[:5]:  # Show first 5
            logger.info(f"  - {dataset_dir.name}")
        if len(datasets) > 5:
            logger.info(f"  ... and {len(datasets) - 5} more")
        return True
    else:
        logger.warning("No datasets found")
        return False


def test_model_loading():
    """Test model loading"""
    logger.info("\nTesting model loading...")

    model_path = Path(__file__).parent.parent / "output" / "qwen3vl_flare25"

    if not model_path.exists():
        logger.warning(f"Model path not found: {model_path}")
        logger.info("Skipping model loading test")
        logger.info("This is expected if you haven't trained a model yet")
        return False

    try:
        from evaluate_flare25 import FLARE25Evaluator
        import torch

        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            device = "cpu"
        else:
            device = "cuda"

        logger.info(f"Loading model from {model_path} on {device}...")
        evaluator = FLARE25Evaluator(
            model_path=str(model_path),
            device=device,
            max_new_tokens=128
        )
        logger.info("✓ Model loaded successfully!")

        # Clean up
        del evaluator
        if device == "cuda":
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("FLARE25 Evaluation Pipeline Test Suite")
    logger.info("="*60)

    results = {}

    # Test 1: Metrics
    try:
        results['metrics'] = test_metrics()
    except Exception as e:
        logger.error(f"Metrics test failed: {e}")
        results['metrics'] = False

    # Test 2: Dataset discovery
    try:
        results['dataset_discovery'] = test_dataset_discovery()
    except Exception as e:
        logger.error(f"Dataset discovery test failed: {e}")
        results['dataset_discovery'] = False

    # Test 3: Model loading
    try:
        results['model_loading'] = test_model_loading()
    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        results['model_loading'] = False

    # Summary
    logger.info("\n" + "="*60)
    logger.info("Test Summary:")
    logger.info("="*60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:20s}: {status}")

    logger.info("="*60)

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✓ All tests passed! Evaluation pipeline is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Run full evaluation: bash evaluation/run_evaluation.sh")
        logger.info("2. Check results in: evaluation_results/")
    else:
        logger.info("\n⚠ Some tests failed. Please review errors above.")
        if not results.get('model_loading', True):
            logger.info("\nNote: Model loading failure is expected if you haven't")
            logger.info("      completed training yet. The evaluation pipeline")
            logger.info("      itself is ready to use once training is complete.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
