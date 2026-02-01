from pydantic import BaseModel
from enum import StrEnum, Enum
from typing import Literal, Union, Optional
import pyspark.sql.functions as psf
from pyspark.sql import DataFrame
import logging

logger = logging.getLogger(__name__)


# PROBLEM_TYPE = ["regression", "classification"]

class ColumnTypes(Enum):
    categorical = ["string", "char", "varchar"]
    numerical = ["byte", "short", "int", "long", "float", "double", "decimal"]
    boolean = ["bool"]
    datetime = ["date", "timestamp", "timestamp_ntz"]

class ProblemType(StrEnum):
    regression = "regression"
    classification = "classification"

class Problem(BaseModel):
    target: str
    type: ProblemType
    desired_result: Optional[Union[str, int]] = None
    date_column: Optional[str] = None

class ClassificationProblem(Problem):
    @staticmethod
    def is_allowed_dtypes(dtype: str):
        _allowed_types = ColumnTypes.categorical.value + ColumnTypes.boolean.value + ["int"]
        return dtype in _allowed_types
        
    
    def _get_col_unique_count(self, dataframe, col_name: str):
        return dataframe.select(psf.count_distinct(col_name)).collect()[0][0]
    
    def _get_unique_target_values(self, dataframe):
        return dataframe.select(self.problem.target).distinct().collect()
    
    def target_check(self, max_classes: int = 10, dataframe: DataFrame = None):
        _target_col_type = self.get_col_type(self.problem.target)
        _nunique = self._get_col_unique_count(dataframe, col_name=self.target)
        _distinct_values = self._get_unique_target_values(dataframe)

        
        if not _target_col_type in ClassificationProblem.is_allowed_dtypes():
            raise ValueError("Target column {self.problem.target} dtype {_target_col_type} is not supported for classification. "
                                f"Supported dtypes are integers, bool, string and enum.")
        
        if _nunique < 2:
            raise ValueError('Classification problem must have at least 2 classes.')

        if _nunique > max_classes:
            raise ValueError(f'Target column has more than {max_classes} classes')
        
        if (self.desired_result is not None) and (self.desired_result not in _distinct_values):
            raise ValueError(f'Desired outcome class {self.target_desired_outcome} not in list of classes: {self.classes}')
        

        
class RegressionProblem(Problem):
    @staticmethod
    def is_allowed_dtypes(dtype: str):
        _allowed_types = ColumnTypes.numerical.value
        # Exclude int from allowed types for regression
        _allowed_types = [t for t in _allowed_types if t != "int"]
        return dtype in _allowed_types
        

class SchemaChecks:
    def __init__(self, dataframe: DataFrame, problem=Problem):
        self.dataframe = self.rename_columns(dataframe=dataframe)
        self.problem = problem

    def rename_columns(self, dataframe):
        return dataframe.toDF(*[c.replace(".", "_") for c in dataframe.columns])

    def get_typed_col(self, col_type: Literal["categorical", "numerical", "boolean", "datetime"]):
        col_types = ColumnTypes[col_type].value
        return [col for col, _ctype in self.dataframe.dtypes if _ctype in col_types]

    def get_col_type(self, col_name: str):
        return dict(self.dataframe.dtypes)[col_name]

    @property
    def datetime_cols(self):
        """Return list of datetime columns"""
        return self.get_typed_col("datetime")

    @property
    def categorical_cols(self):
        """Return list of categorical columns"""
        return self.get_typed_col("categorical")

    @property
    def numerical_cols(self):
        """Return list of numerical columns"""
        return self.get_typed_col("numerical")

    @property
    def boolean_cols(self):
        """Return list of boolean columns"""
        return self.get_typed_col("boolean")
    

    def categorical_checks(self):
        categorical_columns = self.get_typed_col(col_type="categorical")
        details = []

        for _col in categorical_columns:
            details.append({"col_name": _col})
            
            details[-1]["description"] = self.dataframe.select(psf.count_distinct(_col).alias("count_distinct"),
                psf.count(_col).alias("count"),
                psf.count(psf.when(psf.col(_col).isNull(), 1)).alias("null_count")
                ).first().asDict()
            
            grouped_stats = self.dataframe.groupBy(_col).agg(
                psf.count(_col).alias("count"),
                psf.count(psf.when(psf.col(_col).isNull(), 1)).alias("null_count")

                ).collect()
            
            details[-1]["value_descriptions"] = [row.asDict() for row in grouped_stats]

        return details
        


    def numerical_checks(self):
        numerical_columns = self.get_typed_col("numerical")
        details = []

        for _col in numerical_columns:
            details.append({"col_name": _col})

            details[-1]["description"] = self.dataframe.select(psf.count_distinct(_col).alias("count_distinct"),
                psf.count(_col).alias("count"),
                psf.count(psf.when(psf.col(_col).isNull(), 1)).alias("null_count"),
                psf.mean(_col).alias("mean"),
                psf.std(_col).alias("std"),
                psf.min(_col).alias("min"),
                psf.median(_col).alias("median"),
                psf.max(_col).alias("max"),
                ).first().asDict()
        return details

    def datetime_checks(self):
        # Skip datetime checks if no date column is specified
        if self.problem.date_column is None:
            return

        if self.problem.date_column not in self.dataframe.columns:
            available = ", ".join(self.dataframe.columns[:5])
            available_suffix = "..." if len(self.dataframe.columns) > 5 else ""
            raise ValueError(
                f"Date column '{self.problem.date_column}' not found. "
                f"Available columns: {available}{available_suffix}"
            )

        date_col_type = self.get_col_type(self.problem.date_column)
        if date_col_type not in ColumnTypes.datetime.value:
            expected_types = ", ".join(ColumnTypes.datetime.value)
            raise ValueError(
                f"Date column '{self.problem.date_column}' must have a temporal dtype, "
                f"found '{date_col_type}'. Expected one of: {expected_types}"
            )
        

    def bool_checks(self):
        categorical_columns = self.get_typed_col(col_type="boolean")
        details = []

        for _col in categorical_columns:
            details.append({"col_name": _col})
            
            details[-1]["description"] = self.dataframe.select(psf.count_distinct(_col).alias("count_distinct"),
                psf.count(_col).alias("count"),
                psf.count(psf.when(psf.col(_col).isNull(), 1)).alias("null_count")
                ).first().asDict()
            
            grouped_stats = self.dataframe.groupBy(_col).agg(
                psf.count(_col).alias("count"),
                psf.count(psf.when(psf.col(_col).isNull(), 1)).alias("null_count")

                ).collect()
            
            details[-1]["value_descriptions"] = [row.asDict() for row in grouped_stats]
        return details

    def target_checks(self, max_classes: int = 10):
        logger.debug(f"Validating target column: {self.problem.target}")

        if self.problem.target not in self.dataframe.columns:
            available = ", ".join(self.dataframe.columns[:5])
            available_suffix = "..." if len(self.dataframe.columns) > 5 else ""
            raise ValueError(
                f"Target column '{self.problem.target}' not found. "
                f"Available columns: {available}{available_suffix}"
            )

        target_type = self.get_col_type(self.problem.target)
        logger.debug(f"Target column type: {target_type}")

        if self.problem.type == ProblemType.classification:
            if not ClassificationProblem.is_allowed_dtypes(target_type):
                allowed_types = "categorical (string, char, varchar), boolean, or integer"
                raise ValueError(
                    f"Target column '{self.problem.target}' has type '{target_type}' which is not valid "
                    f"for classification. Expected: {allowed_types}"
                )

            # Additional validation for classification
            unique_count = self.dataframe.select(psf.count_distinct(self.problem.target)).first()[0]
            logger.debug(f"Target unique values: {unique_count}")

            if unique_count < 2:
                raise ValueError(
                    f"Classification target '{self.problem.target}' must have at least 2 distinct classes, "
                    f"found {unique_count}"
                )

            if unique_count > max_classes:
                raise ValueError(
                    f"Target column '{self.problem.target}' has {unique_count} distinct classes, "
                    f"which exceeds the maximum of {max_classes}. Consider using regression or reducing classes."
                )

        elif self.problem.type == ProblemType.regression:
            if not RegressionProblem.is_allowed_dtypes(target_type):
                allowed_types = "numerical (float, double, decimal, long, short, byte) but not integer"
                raise ValueError(
                    f"Target column '{self.problem.target}' has type '{target_type}' which is not valid "
                    f"for regression. Expected: {allowed_types}"
                )
            

    def check(self, max_classes: int = 10):
        """Run all data QC checks."""
        try:
            self.target_checks()
        except Exception as e:
            raise ValueError(f"Target check failed with: {e}")
        
        try:
            self.datetime_checks()
        except Exception as e:
            raise ValueError(f"Datetime check failed with: {e}")
        
        
        categorical_details = self.categorical_checks()
        numerical_details = self.numerical_checks()
        bool_details = self.bool_checks()
        comb_details = {
            "categorical": categorical_details, 
            "numerical": numerical_details, 
            "boolean": bool_details
            }
        return comb_details



            
        

            
        
