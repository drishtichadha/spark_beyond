"""General data classes related to KPI Analytics."""

from datetime import date
from typing import List, Optional, Union

from enum import StrEnum
from pydantic import BaseModel


class FilterOperator(StrEnum):
    """Filter operators to filter dataframe"""

    EQ = "equal_to"
    NEQ = "not_equal_to"
    EMPTY = "empty"
    NON_EMPTY = "non_empty"
    GRT = "greater_than"
    LWT = "lower_than"
    GEQ = "greater_than_equal_to"
    LEQ = "lower_than_equal_to"
    LIKE = "like"


class Filter(BaseModel):
    """Filter to apply on a coloumn."""

    column: str
    operator: FilterOperator
    values: Optional[List[Union[str, float, bool, date]]] = None

