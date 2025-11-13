"""Shared helpers for FLARE task type normalization."""

from __future__ import annotations

import re
from typing import Any

TASK_TYPE_CANONICAL_MAP = {
    "classification": "Classification",
    "binary classification": "Classification",
    "multi label classification": "Multi-label Classification",
    "multi-label classification": "Multi-label Classification",
    "multi label": "Multi-label Classification",
    "detection": "Detection",
    "object detection": "Detection",
    "lesion detection": "Detection",
    "instance detection": "Instance Detection",
    "instance segmentation": "Instance Detection",
    "counting": "Counting",
    "cell counting": "Counting",
    "regression": "Regression",
    "report generation": "Report Generation",
    "report generation task": "Report Generation",
    "reporting": "Report Generation",
}


def canonical_task_type(task_type: Any) -> str:
    """Normalize diverse task type labels to a canonical string."""
    if task_type is None:
        return "Unknown"

    normalized = re.sub(
        r"\s+",
        " ",
        str(task_type).replace("_", " ").replace("-", " ").strip().lower(),
    )

    if not normalized:
        return "Unknown"

    return TASK_TYPE_CANONICAL_MAP.get(normalized, normalized.title())
