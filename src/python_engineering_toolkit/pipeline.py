from typing import Protocol, Any, runtime_checkable
import time
from dataclasses import dataclass, field
from collections.abc import Iterator


@runtime_checkable
class Transformer(Protocol):
    """
    Duck-typed interface for any transformer.
    Any class with transform() and validate() satisfies this protocol —
    no inheritance required.
    """

    def transform(self, data: Any) -> Any:
        """Transform the input data."""
        ...

    def validate(self, data: Any) -> bool:
        """Validate the input data."""
        ...


@dataclass
class StepResult:
    """Tracks the result and timing of a single pipeline step."""
    step_name: str
    duration_seconds: float
    input_data: Any
    output_data: Any
    success: bool
    error: str | None = None


class FeaturePipeline:
    """
    Strategy Pattern: accepts any Transformer strategy.
    The pipeline doesn't know what transformer it's using — it just calls it.
    """

    def __init__(self, transformer: Transformer, name: str = "unnamed") -> None:
        self._transformer = transformer  # the "strategy"
        self.name = name
        self._results: list[StepResult] = []

    def run(self, data: Any) -> Any:
        """Run the transformer strategy on the data, tracking timing."""
        start = time.perf_counter()
        try:
            if not self._transformer.validate(data):
                raise ValueError(f"Validation failed for input: {data!r}")
            result = self._transformer.transform(data)
            duration = time.perf_counter() - start
            self._results.append(StepResult(
                step_name=type(self._transformer).__name__,
                duration_seconds=duration,
                input_data=data,
                output_data=result,
                success=True,
            ))
            return result
        except Exception as e:
            duration = time.perf_counter() - start
            self._results.append(StepResult(
                step_name=type(self._transformer).__name__,
                duration_seconds=duration,
                input_data=data,
                output_data=None,
                success=False,
                error=str(e),
            ))
            raise

    @property
    def results(self) -> list[StepResult]:
        return self._results.copy()

    def __repr__(self) -> str:
        return f"FeaturePipeline(transformer={self._transformer!r}, name={self.name!r})"
    
from python_engineering_toolkit.preprocessors import (
    BasePreprocessor,
    NumericPreprocessor,
    CategoricalPreprocessor,
    DatePreprocessor,
)


class PreprocessorFactory:
    """
    Factory Pattern: one place that knows how to create preprocessors.
    Callers never instantiate preprocessors directly.
    """

    _registry: dict[str, type[BasePreprocessor]] = {
        "numeric": NumericPreprocessor,
        "categorical": CategoricalPreprocessor,
        "date": DatePreprocessor,
    }

    @classmethod
    def create(cls, type: str) -> BasePreprocessor:
        """
        Create a preprocessor by type name.

        Args:
            type: One of 'numeric', 'categorical', 'date'

        Returns:
            The appropriate BasePreprocessor subclass instance

        Raises:
            ValueError: If type is not registered
        """
        if type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown preprocessor type: {type!r}. "
                f"Available: {available}"
            )
        return cls._registry[type]()

    @classmethod
    def register(cls, name: str, preprocessor_class: type[BasePreprocessor]) -> None:
        """
        Register a new preprocessor type at runtime.
        This is what makes the factory extensible without modifying it.
        """
        cls._registry[name] = preprocessor_class

    @classmethod
    def available_types(cls) -> list[str]:
        return list(cls._registry.keys())



class Pipeline:
    """
    Chains multiple FeaturePipelines in sequence.
    Implements __iter__ and __len__ to make it iterable like a list.
    Tracks step timing for every stage.
    """

    def __init__(self, name: str = "pipeline") -> None:
        self.name = name
        self._steps: list[FeaturePipeline] = []
        self._step_times: list[float] = []

    def add_step(self, transformer: Transformer, step_name: str | None = None) -> "Pipeline":
        """Add a transformer step. Returns self so you can chain .add_step() calls."""
        name = step_name or type(transformer).__name__
        self._steps.append(FeaturePipeline(transformer, name=name))
        return self  # enables: pipeline.add_step(...).add_step(...)

    def run(self, data: Any) -> Any:
        """
        Pass data through every step in sequence.
        Each step's output becomes the next step's input.
        """
        self._step_times = []
        result = data
        for step in self._steps:
            start = time.perf_counter()
            result = step.run(result)
            self._step_times.append(time.perf_counter() - start)
        return result

    def timing_report(self) -> dict[str, float]:
        """Returns a dict of step_name -> duration_seconds for every step."""
        return {
            step.name: duration
            for step, duration in zip(self._steps, self._step_times)
        }

    # --- these two make Pipeline iterable ---

    def __iter__(self) -> Iterator[FeaturePipeline]:
        """Iterate over steps: for step in pipeline."""
        return iter(self._steps)

    def __len__(self) -> int:
        """len(pipeline) returns number of steps."""
        return len(self._steps)

    def __repr__(self) -> str:
        steps = " -> ".join(s.name for s in self._steps)
        return f"Pipeline(name={self.name!r}, steps=[{steps}])"