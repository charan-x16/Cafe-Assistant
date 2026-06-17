from __future__ import annotations

from typing import Any

from sqlalchemy import JSON
from sqlalchemy.types import TypeDecorator, UserDefinedType


class _PostgresVector(UserDefinedType):
    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def get_col_spec(self, **kw: Any) -> str:
        return f"vector({self.dimensions})"


class EmbeddingVector(TypeDecorator[list[float]]):
    """Use pgvector on Postgres and JSON elsewhere for local tests."""

    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.dimensions = dimensions

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(_PostgresVector(self.dimensions))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: list[float] | None, dialect: Any) -> Any:
        if value is None:
            return None
        vector = [float(component) for component in value]
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Expected embedding with {self.dimensions} dimensions, got {len(vector)}."
            )
        if dialect.name == "postgresql":
            return "[" + ",".join(str(component) for component in vector) + "]"
        return vector

    def process_result_value(self, value: Any, dialect: Any) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip("[]")
            if not stripped:
                return []
            return [float(component) for component in stripped.split(",")]
        return [float(component) for component in value]
