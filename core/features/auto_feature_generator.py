"""
Automatic Feature Generation for PySpark
This module provides automated feature engineering capabilities for PySpark DataFrames
"""

from pyspark.sql import DataFrame, SparkSession
from core.discovery import SchemaChecks, Problem
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, DateType, TimestampType
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Bucketizer
from typing import List, Dict, Optional, Tuple
import numpy as np


class AutoFeatureGenerator:
    """
    Automated feature generation for PySpark DataFrames
    
    Features:
    - Numerical transformations (log, sqrt, polynomial)
    - Interaction features
    - Aggregation features (group-by statistics)
    - Binning/discretization
    - Date/time features
    - String features (length, word count, etc.)
    - Encoding (one-hot, label encoding)
    """
    
    def __init__(self, schema_checks: SchemaChecks, problem: Problem):
        
        self.schema_checks = schema_checks
        self.problem = problem
        self.dataframe = schema_checks.dataframe
        self.generated_features = []
        
    def generate_numerical_features(
        self,
        columns: Optional[List[str]] = None,
        transformations: List[str] = ['log', 'sqrt', 'square', 'cube']
    ) -> DataFrame:
        """
        Generate numerical transformations for specified columns
        
        Args:
            columns: List of columns to transform (None = all numeric columns)
            transformations: List of transformations to apply
        
        Returns:
            DataFrame with additional transformed features
        """
        if columns is None:
            columns = [f.name for f in self.dataframe.schema.fields 
                      if isinstance(f.dataType, NumericType)]
        
        
        for col in columns:
            if 'log' in transformations:
                # Log transformation (handling zeros and negatives)
                self.dataframe = self.dataframe.withColumn(
                    f"{col}_log",
                    F.log1p(F.when(F.col(col) > 0, F.col(col)).otherwise(0))
                )
                self.generated_features.append(f"{col}_log")
            
            if 'sqrt' in transformations:
                # Square root (handling negatives)
                self.dataframe = self.dataframe.withColumn(
                    f"{col}_sqrt",
                    F.sqrt(F.when(F.col(col) >= 0, F.col(col)).otherwise(0))
                )
                self.generated_features.append(f"{col}_sqrt")
            
            if 'square' in transformations:
                self.dataframe = self.dataframe.withColumn(
                    f"{col}_square",
                    F.pow(F.col(col), 2)
                )
                self.generated_features.append(f"{col}_square")
            
            if 'cube' in transformations:
                self.dataframe = self.dataframe.withColumn(
                    f"{col}_cube",
                    F.pow(F.col(col), 3)
                )
                self.generated_features.append(f"{col}_cube")
        
        return self.dataframe
    
    def generate_interaction_features(
        self,
        columns: Optional[List[str]] = None,
        max_interactions: int = 2
    ) -> DataFrame:
        """
        Generate interaction features (multiplication, division, addition)
        
        Args:
            columns: List of columns to use for interactions
            max_interactions: Maximum number of columns to interact
        
        Returns:
            DataFrame with interaction features
        """
        if columns is None:
            columns = [f.name for f in self.dataframe.schema.fields 
                      if isinstance(f.dataType, NumericType)]
        
        
        # Pairwise interactions
        if max_interactions >= 2:
            for i, col1 in enumerate(columns):
                for col2 in columns[i+1:]:
                    # Multiplication
                    self.dataframe = self.dataframe.withColumn(
                        f"{col1}_mult_{col2}",
                        F.col(col1) * F.col(col2)
                    )
                    self.generated_features.append(f"{col1}_mult_{col2}")
                    
                    # Division (handling division by zero)
                    self.dataframe = self.dataframe.withColumn(
                        f"{col1}_div_{col2}",
                        F.when(F.col(col2) != 0, F.col(col1) / F.col(col2))
                        .otherwise(0)
                    )
                    self.generated_features.append(f"{col1}_div_{col2}")
                    
                    # Addition
                    self.dataframe = self.dataframe.withColumn(
                        f"{col1}_add_{col2}",
                        F.col(col1) + F.col(col2)
                    )
                    self.generated_features.append(f"{col1}_add_{col2}")
                    
                    # Subtraction
                    self.dataframe = self.dataframe.withColumn(
                        f"{col1}_sub_{col2}",
                        F.col(col1) - F.col(col2)
                    )
                    self.generated_features.append(f"{col1}_sub_{col2}")
        
        return self.dataframe
    
    def generate_aggregation_features(
        self,
        group_by_cols: List[str],
        agg_cols: List[str],
        agg_functions: List[str] = ['mean', 'sum', 'min', 'max', 'stddev', 'count']
    ) -> DataFrame:
        """
        Generate aggregation features based on grouping
        
        Args:
            group_by_cols: Columns to group by
            agg_cols: Columns to aggregate
            agg_functions: Aggregation functions to apply
        
        Returns:
            DataFrame with aggregation features joined back
        """
        # Create aggregations
        agg_exprs = []
        feature_names = []
        
        for agg_col in agg_cols:
            for func in agg_functions:
                if func == 'mean':
                    agg_exprs.append(F.mean(agg_col).alias(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}"))
                elif func == 'sum':
                    agg_exprs.append(F.sum(agg_col).alias(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}"))
                elif func == 'min':
                    agg_exprs.append(F.min(agg_col).alias(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}"))
                elif func == 'max':
                    agg_exprs.append(F.max(agg_col).alias(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}"))
                elif func == 'stddev':
                    agg_exprs.append(F.stddev(agg_col).alias(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}"))
                elif func == 'count':
                    agg_exprs.append(F.count(agg_col).alias(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}"))
                
                feature_names.append(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}")
        
        # Perform aggregation
        agg_dataframe = self.dataframe.groupBy(group_by_cols).agg(*agg_exprs)
        
        # Join back to original DataFrame
        self.dataframe = self.dataframe.join(agg_dataframe, on=group_by_cols, how='left')
        
        self.generated_features.extend(feature_names)
        return self.dataframe
    
    def generate_binning_features(
        self,
        columns: Optional[List[str]] = None,
        n_bins: int = 5
    ) -> DataFrame:
        """
        Generate binned/discretized features
        
        Args:
            columns: Columns to bin
            n_bins: Number of bins
        
        Returns:
            DataFrame with binned features
        """
        if columns is None:
            columns = [f.name for f in self.dataframe.schema.fields 
                      if isinstance(f.dataType, NumericType)]
        
        
        for col in columns:
            # Calculate quantiles for binning
            quantiles = self.dataframe.approxQuantile(col, 
                [i/n_bins for i in range(n_bins + 1)], 0.01)
            
            # Ensure unique splits
            quantiles = sorted(list(set(quantiles)))
            
            if len(quantiles) > 2:
                # Create bucketizer
                bucketizer = Bucketizer(
                    splits=quantiles,
                    inputCol=col,
                    outputCol=f"{col}_binned"
                )
                
                self.dataframe = bucketizer.transform(self.dataframe)
                self.generated_features.append(f"{col}_binned")
        
        return self.dataframe
    
    def generate_datetime_features(
        self,
        columns: Optional[List[str]] = None
    ) -> DataFrame:
        """
        Generate date/time features
        
        Args:
            columns: Date/timestamp columns
        
        Returns:
            DataFrame with datetime features
        """
        if columns is None:
            columns = [f.name for f in self.dataframe.schema.fields 
                      if isinstance(f.dataType, (DateType, TimestampType))]
        
        
        for col in columns:
            # Extract components
            self.dataframe = self.dataframe.withColumn(f"{col}_year", F.year(col))
            self.dataframe = self.dataframe.withColumn(f"{col}_month", F.month(col))
            self.dataframe = self.dataframe.withColumn(f"{col}_day", F.dayofmonth(col))
            self.dataframe = self.dataframe.withColumn(f"{col}_dayofweek", F.dayofweek(col))
            self.dataframe = self.dataframe.withColumn(f"{col}_dayofyear", F.dayofyear(col))
            self.dataframe = self.dataframe.withColumn(f"{col}_weekofyear", F.weekofyear(col))
            self.dataframe = self.dataframe.withColumn(f"{col}_quarter", F.quarter(col))
            
            # Is weekend
            self.dataframe = self.dataframe.withColumn(
                f"{col}_is_weekend",
                F.when(F.dayofweek(col).isin([1, 7]), 1).otherwise(0)
            )
            
            self.generated_features.extend([
                f"{col}_year", f"{col}_month", f"{col}_day",
                f"{col}_dayofweek", f"{col}_dayofyear",
                f"{col}_weekofyear", f"{col}_quarter", f"{col}_is_weekend"
            ])
        
        return self.dataframe
    
    def generate_string_features(
        self,
        columns: Optional[List[str]] = None
    ) -> DataFrame:
        """
        Generate features from string columns
        
        Args:
            columns: String columns
        
        Returns:
            DataFrame with string-based features
        """
        if columns is None:
            columns = [f.name for f in self.dataframe.schema.fields 
                      if isinstance(f.dataType, StringType)]
        
        
        for col in columns:
            # String length
            self.dataframe = self.dataframe.withColumn(
                f"{col}_length",
                F.length(F.col(col))
            )
            
            # Word count
            self.dataframe = self.dataframe.withColumn(
                f"{col}_word_count",
                F.size(F.split(F.col(col), " "))
            )
            
            # Number of unique characters
            self.dataframe = self.dataframe.withColumn(
                f"{col}_unique_chars",
                F.size(F.array_distinct(F.split(F.col(col), "")))
            )
            
            self.generated_features.extend([
                f"{col}_length", f"{col}_word_count", f"{col}_unique_chars"
            ])
        
        return self.dataframe
    
    def generate_all_features(
        self,
        include_numerical: bool = True,
        include_interactions: bool = False,
        include_binning: bool = True,
        include_datetime: bool = True,
        include_string: bool = True,
        numerical_columns: list = None,
        categorical_columns: list = None,
        datetime_columns: list = None
    ) -> DataFrame:
        """
        Generate all features automatically
        
        Args:
            include_numerical: Whether to include numerical transformations
            include_interactions: Whether to include interaction features
            include_binning: Whether to include binning features
            include_datetime: Whether to include datetime features
            include_string: Whether to include string features
        
        Returns:
            DataFrame with all generated features
        """
        
        if not numerical_columns:
            numerical_columns = self.schema_checks.get_typed_col("numerical")
        
        if not categorical_columns:
            categorical_columns = self.schema_checks.get_typed_col(col_type="categorical")
            categorical_columns.remove(self.problem.target)

        if not datetime_columns:
            datetime_columns = self.schema_checks.get_typed_col(col_type="datetime")



        
        if include_numerical:
            print("Generating numerical features...")
            self.dataframe = self.generate_numerical_features(numerical_columns)
        
        if include_interactions:
            print("Generating interaction features...")
            self.dataframe = self.generate_interaction_features(numerical_columns)
        
        if include_binning:
            print("Generating binning features...")
            self.dataframe = self.generate_binning_features(numerical_columns)
        
        if include_datetime:
            print("Generating datetime features...")
            self.dataframe = self.generate_datetime_features(datetime_columns)
        
        if include_string:
            print("Generating string features...")
            self.dataframe = self.generate_string_features(categorical_columns)
        
        print(f"Total features generated: {len(self.generated_features)}")
        return self.dataframe
    
    def get_generated_features(self) -> List[str]:
        """Return list of all generated feature names"""
        return self.generated_features


# Example usage
if __name__ == "__main__":
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("AutoFeatureGeneration") \
        .getOrCreate()
    
    # Create sample data
    data = [
        (1, 100.0, 50.0, "2023-01-15", "Category A", "Sample text here"),
        (2, 200.0, 75.0, "2023-02-20", "Category B", "Another sample"),
        (3, 150.0, 60.0, "2023-03-10", "Category A", "More text data"),
        (4, 300.0, 90.0, "2023-04-05", "Category C", "Final example text"),
    ]
    
    dataframe = spark.createDataFrame(
        data, 
        ["id", "value1", "value2", "date_col", "category", "text_col"]
    )
    
    # Convert date column
    dataframe = dataframe.withColumn("date_col", F.to_date("date_col"))
    
    # Initialize feature generator
    feature_gen = AutoFeatureGenerator(spark)
    
    # Generate all features
    dataframe_with_features = feature_gen.generate_all_features(
        dataframe,
        include_numerical=True,
        include_interactions=False,  # Set to True for interactions
        include_binning=True,
        include_datetime=True,
        include_string=True
    )
    
    # Show results
    print("\nOriginal columns:", dataframe.columns)
    print("\nNew columns:", dataframe_with_features.columns)
    print("\nGenerated features:", feature_gen.get_generated_features())
    
    dataframe_with_features.show(5, truncate=False)
    
    spark.stop()