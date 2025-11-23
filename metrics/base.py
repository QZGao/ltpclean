from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch


@dataclass
class EvaluationContext:
    """Container for shared resources needed by all evaluators."""

    model: torch.nn.Module
    vae: torch.nn.Module
    device: torch.device
    dataset_path: Path
    prediction_lengths: Sequence[int]
    sample_steps: Sequence[int]
    seeds: Sequence[int]
    batch_size: int
    output_dir: Path
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(ABC):
    """Common interface for every evaluation component."""

    name: str = "base"
    description: str = ""

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}(name={self.name})"

    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> Dict[str, Any]:
        """Run the metric and return structured results."""

    def required_prediction_lengths(self) -> Iterable[int]:
        return []

    def is_applicable(self, context: EvaluationContext) -> bool:
        del context
        return True
