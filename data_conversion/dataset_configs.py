"""
FLARE25 Dataset Configuration
Defines metadata for all 19 FLARE datasets including paths, categories, and task types.
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class FLAREDatasetConfig:
    """Configuration for a single FLARE dataset"""
    name: str
    category: str  # Medical imaging modality category
    modality: str  # Specific modality type
    task_types: List[str]  # Classification, Detection, etc.
    has_multi_image: bool  # Whether dataset contains multi-image samples
    is_starter: bool  # Part of initial 5-dataset subset
    description: str


# Dataset configurations for all 19 FLARE datasets
FLARE_DATASETS = {
    # ===== STARTER DATASETS (5) =====
    "neojaundice": FLAREDatasetConfig(
        name="neojaundice",
        category="Clinical",
        modality="Digital Camera",
        task_types=["Classification"],
        has_multi_image=True,  # 3 views per case
        is_starter=True,
        description="Neonatal jaundice assessment from clinical photos (3 views)"
    ),

    "retino": FLAREDatasetConfig(
        name="retino",
        category="Retinography",
        modality="Fundus Photography",
        task_types=["Classification"],
        has_multi_image=False,
        is_starter=True,
        description="Retinal imaging for diabetic retinopathy and other conditions"
    ),

    "BUSI-det": FLAREDatasetConfig(
        name="BUSI-det",
        category="Ultrasound",
        modality="Breast Ultrasound",
        task_types=["Detection"],
        has_multi_image=False,
        is_starter=True,
        description="Breast ultrasound lesion detection"
    ),

    "boneresorption": FLAREDatasetConfig(
        name="boneresorption",
        category="Xray",
        modality="Dental X-ray",
        task_types=["Classification", "Regression"],
        has_multi_image=False,
        is_starter=True,
        description="Bone resorption assessment in dental X-rays"
    ),

    "bone_marrow": FLAREDatasetConfig(
        name="bone_marrow",
        category="Microscopy",
        modality="Microscopy",
        task_types=["Classification"],
        has_multi_image=False,
        is_starter=True,
        description="Bone marrow cell classification"
    ),

    # ===== ADDITIONAL DATASETS (14) =====
    "fundus": FLAREDatasetConfig(
        name="fundus",
        category="Retinography",
        modality="Fundus Photography",
        task_types=["Classification"],
        has_multi_image=False,
        is_starter=False,
        description="Fundus imaging for multiple retinal conditions"
    ),

    "BUS-UCLM-det": FLAREDatasetConfig(
        name="BUS-UCLM-det",
        category="Ultrasound",
        modality="Breast Ultrasound",
        task_types=["Detection"],
        has_multi_image=False,
        is_starter=False,
        description="Breast ultrasound lesion detection (UCLM dataset)"
    ),

    "BUSI": FLAREDatasetConfig(
        name="BUSI",
        category="Ultrasound",
        modality="Breast Ultrasound",
        task_types=["Classification"],
        has_multi_image=False,
        is_starter=False,
        description="Breast ultrasound classification"
    ),

    "BUS-UCLM": FLAREDatasetConfig(
        name="BUS-UCLM",
        category="Ultrasound",
        modality="Breast Ultrasound",
        task_types=["Classification"],
        has_multi_image=False,
        is_starter=False,
        description="Breast ultrasound classification (UCLM dataset)"
    ),

    "iugc": FLAREDatasetConfig(
        name="iugc",
        category="Ultrasound",
        modality="Obstetric Ultrasound",
        task_types=["Classification", "Detection", "Regression"],
        has_multi_image=False,
        is_starter=False,
        description="Intrauterine growth curve assessment"
    ),

    "dental": FLAREDatasetConfig(
        name="dental",
        category="Xray",
        modality="Dental X-ray",
        task_types=["Classification"],
        has_multi_image=False,
        is_starter=False,
        description="Dental condition classification"
    ),

    "periapical": FLAREDatasetConfig(
        name="periapical",
        category="Xray",
        modality="Dental X-ray",
        task_types=["Classification", "Multi-label Classification"],
        has_multi_image=False,
        is_starter=False,
        description="Periapical lesion detection and classification"
    ),

    "IU_XRay": FLAREDatasetConfig(
        name="IU_XRay",
        category="Xray",
        modality="Chest X-ray",
        task_types=["Report Generation"],
        has_multi_image=False,
        is_starter=False,
        description="Chest X-ray report generation"
    ),

    "chestdr": FLAREDatasetConfig(
        name="chestdr",
        category="Xray",
        modality="Chest X-ray",
        task_types=["Classification", "Multi-label Classification"],
        has_multi_image=False,
        is_starter=False,
        description="Chest X-ray disease classification"
    ),

    "chromosome": FLAREDatasetConfig(
        name="chromosome",
        category="Microscopy",
        modality="Microscopy",
        task_types=["Instance Detection", "Counting"],
        has_multi_image=False,
        is_starter=False,
        description="Chromosome detection and counting"
    ),

    "neurips22cell": FLAREDatasetConfig(
        name="neurips22cell",
        category="Microscopy",
        modality="Microscopy",
        task_types=["Counting"],
        has_multi_image=False,
        is_starter=False,
        description="Cell counting in microscopy images"
    ),

    "endo": FLAREDatasetConfig(
        name="endo",
        category="Endoscopy",
        modality="Endoscopy",
        task_types=["Classification"],
        has_multi_image=False,
        is_starter=False,
        description="Endoscopic image classification"
    ),

    "bcn20000": FLAREDatasetConfig(
        name="bcn20000",
        category="Dermatology",
        modality="Dermatoscopy",
        task_types=["Classification"],
        has_multi_image=False,
        is_starter=False,
        description="Skin lesion classification"
    ),

    "CMMD": FLAREDatasetConfig(
        name="CMMD",
        category="Mammography",
        modality="Mammography",
        task_types=["Classification"],
        has_multi_image=False,
        is_starter=False,
        description="Mammography classification"
    ),
}


def get_starter_datasets() -> List[str]:
    """Get list of starter dataset names"""
    return [name for name, config in FLARE_DATASETS.items() if config.is_starter]


def get_all_datasets() -> List[str]:
    """Get list of all dataset names"""
    return list(FLARE_DATASETS.keys())


def get_datasets_by_category(category: str) -> List[str]:
    """Get datasets filtered by category"""
    return [name for name, config in FLARE_DATASETS.items() if config.category == category]


def get_dataset_config(name: str) -> FLAREDatasetConfig:
    """Get configuration for a specific dataset"""
    if name not in FLARE_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {', '.join(FLARE_DATASETS.keys())}")
    return FLARE_DATASETS[name]


def get_dataset_paths(base_dir: str, dataset_name: str, split: str) -> Dict[str, Path]:
    """
    Get paths for a dataset's images and annotations

    Args:
        base_dir: Base directory containing organized_dataset/
        dataset_name: Name of the dataset
        split: 'train', 'val', or 'test'

    Returns:
        Dictionary with 'image_dir' and 'annotation_file' paths
    """
    base_path = Path(base_dir)
    config = get_dataset_config(dataset_name)

    if split == "train":
        split_dir = "training"
        image_subdir = "imagesTr"
        annotation_file = f"{dataset_name}_questions_train.json"
    elif split == "val":
        # Try validation-hidden first, then validation-public
        hidden_path = base_path / "validation-hidden" / config.category / dataset_name
        if hidden_path.exists():
            split_dir = "validation-hidden"
        else:
            split_dir = "validation-public"
        image_subdir = "imagesVal"
        # Try with GT suffix first
        annotation_file = f"{dataset_name}_questions_val_withGT.json"
        # Will fall back to without GT if needed
    elif split == "test":
        split_dir = "testing"
        image_subdir = "imagesTs"
        annotation_file = f"{dataset_name}_questions_test.json"
    else:
        raise ValueError(f"Unknown split: {split}")

    dataset_path = base_path / split_dir / config.category / dataset_name
    image_dir = dataset_path / image_subdir
    annotation_path = dataset_path / annotation_file

    # For validation, check if GT file exists, otherwise use non-GT
    if split == "val" and not annotation_path.exists():
        annotation_path = dataset_path / f"{dataset_name}_questions_val.json"

    return {
        "image_dir": image_dir,
        "annotation_file": annotation_path,
        "category": config.category
    }


def print_dataset_summary():
    """Print summary of all datasets"""
    print("=" * 80)
    print("FLARE25 Dataset Configuration Summary")
    print("=" * 80)

    print(f"\nTotal Datasets: {len(FLARE_DATASETS)}")
    print(f"Starter Datasets: {len(get_starter_datasets())}")

    print("\n" + "-" * 80)
    print("STARTER DATASETS (5):")
    print("-" * 80)
    for name in get_starter_datasets():
        config = FLARE_DATASETS[name]
        print(f"\n{name}:")
        print(f"  Category: {config.category}")
        print(f"  Modality: {config.modality}")
        print(f"  Task Types: {', '.join(config.task_types)}")
        print(f"  Multi-image: {'Yes' if config.has_multi_image else 'No'}")
        print(f"  Description: {config.description}")

    print("\n" + "-" * 80)
    print("DATASETS BY CATEGORY:")
    print("-" * 80)
    categories = set(config.category for config in FLARE_DATASETS.values())
    for category in sorted(categories):
        datasets = get_datasets_by_category(category)
        print(f"\n{category} ({len(datasets)} datasets):")
        for ds in datasets:
            marker = "★" if FLARE_DATASETS[ds].is_starter else " "
            print(f"  {marker} {ds}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_dataset_summary()
