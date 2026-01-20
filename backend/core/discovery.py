from pydantic import BaseModel
from enum import StrEnum, Enum
from typing import Literal, Union, Optional
import pyspark.sql.functions as psf
from pyspark.sql import DataFrame


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
        _allowed_types = getattr(ColumnTypes, "categorical").value + getattr(ColumnTypes, "boolean").value + ["int"]
        print(dtype, _allowed_types)
        if dtype in _allowed_types:
            return True
        else:
            return False
        
    
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
        _allowed_types = getattr(ColumnTypes, "numerical")
        _allowed_types.remove("int")
        if dtype in _allowed_types:
            return True
        else:
            return False
        

class SchemaChecks:
    def __init__(self, dataframe: DataFrame, problem=Problem):
        self.dataframe = self.rename_columns(dataframe=dataframe)
        self.problem = problem

    def rename_columns(self, dataframe):
        return dataframe.toDF(*[c.replace(".", "_") for c in dataframe.columns])

    def get_typed_col(self, col_type: Literal["categorical", "numerical", "boolean", "datetime"]):
        col_types = getattr(ColumnTypes, col_type).value
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
            raise ValueError(f"Date column {self.problem.date_column} not found.")

        date_col_type = self.get_col_type(self.problem.date_column)
        if date_col_type not in getattr(ColumnTypes, "datetime").value:
            raise ValueError(f'Date column {self.problem.date_column} must have a temporal dtype (date, time or timestamp), found {date_col_type}')
        

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
        if self.problem.target not in self.dataframe.columns:
            print(1)
            raise ValueError(f"Target column {self.problem.target} not found.")
        
        _target_col_type = self.get_col_type(self.problem.target)

        if self.problem.type == getattr(ProblemType, "classification").value:    
            print(2)
            if not ClassificationProblem.is_allowed_dtypes(_target_col_type):
                print(3)
                raise ValueError(f"Dtype {_target_col_type} not allowed as classification target type.")
        print(4)
        if self.problem.type == getattr(ProblemType, "regression").value:
            print(5)
            if not RegressionProblem.is_allowed_dtypes(_target_col_type):
                print(6)
                raise ValueError(f"Dtype {_target_col_type} not allowed as regression target type.")
            

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



            
        

            
        
