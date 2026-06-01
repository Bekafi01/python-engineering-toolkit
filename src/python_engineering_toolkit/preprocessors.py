from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors."""

    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Transform the input data. Must be implemented by subclass."""
        ...

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate the input data. Must be implemented by subclasses."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return f"Preprocessor: {self.__class__.__name__}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)


class NumericPreprocessor(BasePreprocessor):
    """Handles numeric data: validates type, transformes to float"""
    __slots__ = ()  # No instance attributes, so empty tuple

    def validate(self, data: Any) -> bool:
        return isinstance(data, (int, float))

    def transform(self, data: Any) -> float:
        if not self.validate(data):
            raise TypeError(f"Expected numeric, got {type(data).__name__}")
        return float(data)

    def __repr__(self) -> str:
        return "NumericPreprocessor()"

    def __str__(self) -> str:
        return "Preprocessor: NumericPreprocessor"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NumericPreprocessor)

    def __hash__(self) -> int:
        return hash("NumericPreprocessor")


class CategoricalPreprocessor(BasePreprocessor):
    """Handles categorical data: validates type, transforms to lowercase string."""
    __slots__ = ()

    def validate(self, data: Any) -> bool:
        return isinstance(data, str)

    def transform(self, data: Any) -> str:
        if not self.validate(data):
            raise TypeError(f"Expected string, got {type(data).__name__}")
        return data.strip().lower()

    def __repr__(self) -> str:
        return "CategoricalPreprocessor()"

    def __str__(self) -> str:
        return "Preprocessor: CategoricalPreprocessor"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CategoricalPreprocessor)

    def __hash__(self) -> int:
        return hash("CategoricalPreprocessor")


class DatePreprocessor(BasePreprocessor):
    """Handles date strings: validates format, transforms to ISO 8601."""
    __slots__ = ()

    def validate(self, data: Any) -> bool:
        if not isinstance(data, str):
            return False
        try:
            from datetime import datetime

            datetime.strptime(data, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def transform(self, data: Any) -> str:
        if not self.validate(data):
            raise ValueError(f"Expected YYYY-MM-DD date string, got: {data!r}")
        return data  # already ISO format

    def __repr__(self) -> str:
        return "DatePreprocessor()"

    def __str__(self) -> str:
        return "Preprocessor: DatePreprocessor"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DatePreprocessor)

    def __hash__(self) -> int:
        return hash("DatePreprocessor")


@dataclass
class PreprocessorConfig:
    """Configuration for a preprocessor pipeline."""

    name: str
    max_features: int = 100
    drop_nulls: bool = True
    allowed_types: list[str] = field(
        default_factory=lambda: ["numeric", "categorical", "date"]
    )

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not self.name:
            raise ValueError("name cannot be empty")
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")
