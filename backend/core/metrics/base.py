import json
from abc import ABC, abstractmethod
from datetime import date
from time import time
from typing import List, Tuple, Optional, Union

# Core PySpark imports
from pyspark.sql import SparkSession, DataFrame, Column
from pyspark.sql import functions as psf
from pyspark.errors import PySparkAttributeError # Correct error handling for newer versions

# Note: Assuming these custom modules exist in your environment
from backend.core.discovery import Problem, ProblemType
# from backend.core.general import Filter, FilterOperator

DATE_COLUMN = "order_date_time"
PERIOD_WINDOW_SIZE = {"daily": 6, "weekly": 4, "monthly": 3, "quarterly": 4}

class Metrics(ABC):
    """Parent class of all Metrics classes."""

    def __init__(self, dataframe: DataFrame, problem: Problem) -> None:
        self.dataframe = dataframe
        self.problem = problem
        self.dataframe_len = dataframe.count()

    def get_aggregation(self) -> Column:
        """Define the primary aggregation logic."""
        # Fixed: using psf.col() to reference the problem column
        # if self.problem.type == ProblemType.classification:
        return (psf.mean(psf.col(self.problem.target).cast("double"))).alias(f"{self.problem.target}_pct")
        # elif self.problem.type == ProblemType.regression:
        #     return psf.mean(psf.col(self.problem.target)).alias(f"{self.problem.target}_mean")
        # else:
        #     raise Exception("Problem Type should be either classification or regression.")

    def calculate(self, 
                  dimensions: Optional[List[str]] = None,
                  filters: Optional[List] = None,
                  date_range: Optional[Tuple[date, date]] = None,
                  date_column: Optional[str] = None):
        
        tic = time()
        self.apply_filters(
            filters=filters, date_column=date_column, date_range=date_range
        )
        print(f"Time taken to apply filter on data (s): {time() - tic}")

        # Update datagram_len
        self.dataframe_len = self.dataframe.count()

        agg = self.get_aggregation()
        result = self.apply_dimension_agg(dimensions, agg)

        return result

    def build_filter_exp(self, filters: List):
        """Build Spark SQL filter expression from custom Filter objects."""
        filter_expr = None
        for _filter in filters:
            _col = _filter.column
            _operator = _filter.operator
            _values = _filter.values

            # Map FilterOperators to PySpark Column expressions
            if _operator == "EQ":
                expr = psf.col(_col).isin(_values)
            elif _operator == "NEQ":
                expr = ~psf.col(_col).isin(_values)
            elif _operator == "EMPTY":
                expr = (psf.col(_col).isNull()) | (psf.col(_col) == "")
            elif _operator == "NON_EMPTY":
                expr = (psf.col(_col).isNotNull()) & (psf.col(_col) != "")
            elif _operator == "GRT":
                expr = psf.col(_col) > _values[0]
            elif _operator == "GEQ":
                expr = psf.col(_col) >= _values[0]
            elif _operator == "LWT":
                expr = psf.col(_col) < _values[0]
            elif _operator == "LEQ":
                expr = psf.col(_col) <= _values[0]
            elif _operator == "LIKE":
                expr = psf.col(_col).like(_values[0])
            else:
                continue

            filter_expr = expr if filter_expr is None else (filter_expr & expr)

        return filter_expr

    def build_date_filter(self, date_column: str, date_range: Tuple[date, date]):
        return (psf.col(date_column) >= date_range[0]) & (psf.col(date_column) <= date_range[1])

    def apply_filters(self, filters: List, date_column: str, date_range: Tuple[date, date]):
        if date_column and date_range:
            self.dataframe = self.dataframe.filter(
                self.build_date_filter(date_column=date_column, date_range=date_range)
            )
        if filters:
            self.dataframe = self.dataframe.filter(
                self.build_filter_exp(filters=filters)
            )

    def apply_dimension_agg(self, dimensions: List[str], aggregation: Union[Column, tuple]):
        """Executes GroupBy and Aggregation, returning results as a JSON list."""
        df = self.dataframe
        
        # Handle GroupBy logic
        if dimensions:
            df_grouped = df.groupBy(*dimensions)
        else:
            df_grouped = df

        # Handle Aggregation logic
        if aggregation is None:
            result_df = df
        elif isinstance(aggregation, tuple):
            result_df = df_grouped.agg(*aggregation)
        else:
            result_df = df_grouped.agg(aggregation)

        # Convert result to JSON
        try:
            # toJSON().collect() returns a list of JSON strings
            return result_df#[json.loads(row) for row in result_df.toJSON().collect()]
        except Exception as e:
            print(f"PySpark collect failed, falling back to Pandas: {e}")
            return result_df.toPandas().to_dict(orient="records")