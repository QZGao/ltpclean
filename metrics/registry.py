from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .base import BaseEvaluator, EvaluationContext


class MetricRegistry:
    """Keeps track of all metric evaluators and dispatches them."""

    def __init__(self) -> None:
        self._evaluators: MutableMapping[str, BaseEvaluator] = {}

    def register(self, evaluator: BaseEvaluator) -> None:
        name = evaluator.name.lower()
        if name in self._evaluators:
            raise ValueError(f"Duplicate evaluator name: {name}")
        self._evaluators[name] = evaluator

    def registered_names(self) -> List[str]:
        return sorted(self._evaluators.keys())

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._evaluators

    def get(self, name: str) -> BaseEvaluator:
        key = name.lower()
        if key not in self._evaluators:
            raise KeyError(f"Unknown evaluator '{name}'. Available: {self.registered_names()}")
        return self._evaluators[key]

    def run(self, names: Sequence[str], context: EvaluationContext) -> Dict[str, Mapping]:
        if not names:
            names = self.registered_names()
        results = {}
        for name in names:
            evaluator = self.get(name)
            if not evaluator.is_applicable(context):
                results[name] = {
                    "status": "skipped",
                    "reason": "not_applicable",
                }
                continue
            results[name] = evaluator.evaluate(context)
        return results
