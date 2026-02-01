"""
Featuretools integration for automated feature discovery

This module wraps featuretools to automatically discover and generate
features from relational data structures.

Key capabilities:
- Automatic entity relationship detection
- Deep Feature Synthesis (DFS)
- Temporal feature generation with cutoff times
- Feature primitive selection
"""

from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from backend.core.utils.spark_pandas_bridge import spark_to_pandas_safe, pandas_to_spark
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Lazy imports for featuretools (heavy dependency)
_featuretools = None
_woodwork = None


def _get_featuretools():
    """Lazy load featuretools"""
    global _featuretools
    if _featuretools is None:
        try:
            import featuretools as ft
            _featuretools = ft
        except ImportError:
            raise ImportError(
                "featuretools is not installed. "
                "Install it with: pip install featuretools"
            )
    return _featuretools


def _get_woodwork():
    """Lazy load woodwork for semantic typing"""
    global _woodwork
    if _woodwork is None:
        try:
            import woodwork as ww
            _woodwork = ww
        except ImportError:
            raise ImportError(
                "woodwork is not installed. "
                "Install it with: pip install woodwork"
            )
    return _woodwork


@dataclass
class EntityConfig:
    """
    Configuration for a featuretools entity

    Attributes:
        name: Entity name
        index: Primary key column
        time_index: Optional time column for temporal features
        logical_types: Optional dict mapping column names to Woodwork logical types
    """
    name: str
    index: str
    time_index: Optional[str] = None
    logical_types: Optional[Dict[str, str]] = None


@dataclass
class RelationshipConfig:
    """
    Configuration for entity relationships

    Attributes:
        parent_entity: Parent entity name
        parent_column: Parent column for join
        child_entity: Child entity name
        child_column: Child column for join
    """
    parent_entity: str
    parent_column: str
    child_entity: str
    child_column: str


@dataclass
class FeaturetoolsResult:
    """
    Results from featuretools feature generation

    Attributes:
        feature_matrix: DataFrame with generated features
        feature_definitions: List of feature definition objects
        feature_names: List of generated feature names
        entity_set: The featuretools EntitySet used
        generation_time: Time taken to generate features
    """
    feature_matrix: pd.DataFrame
    feature_definitions: List[Any]
    feature_names: List[str]
    entity_set: Any
    generation_time: float


class FeaturetoolsEngine:
    """
    Featuretools-based automated feature discovery engine

    This class handles:
    - Conversion from PySpark to Pandas for featuretools processing
    - Automatic entity set creation
    - Deep Feature Synthesis execution
    - Conversion back to PySpark

    Example:
        # Single table feature generation
        engine = FeaturetoolsEngine(spark)
        result = engine.run_dfs_single_table(
            spark_df,
            index_col='customer_id',
            target_entity='customers',
            max_depth=2
        )
        spark_df_with_features = engine.to_spark(result.feature_matrix)

        # Multi-table with relationships
        engine = FeaturetoolsEngine(spark)
        entities = {
            'customers': EntityConfig('customers', 'customer_id'),
            'transactions': EntityConfig('transactions', 'transaction_id', time_index='date')
        }
        relationships = [
            RelationshipConfig('customers', 'customer_id', 'transactions', 'customer_id')
        ]
        result = engine.run_dfs_multi_table(
            dataframes={'customers': customers_df, 'transactions': transactions_df},
            entities=entities,
            relationships=relationships,
            target_entity='customers'
        )
    """

    def __init__(
        self,
        spark: SparkSession,
        max_rows_for_pandas: int = 500000,
        verbose: bool = True
    ):
        """
        Initialize the FeaturetoolsEngine

        Args:
            spark: SparkSession instance
            max_rows_for_pandas: Maximum rows to convert to Pandas
            verbose: Whether to print progress information
        """
        self.spark = spark
        self.max_rows = max_rows_for_pandas
        self.verbose = verbose
        self._entity_set = None
        self._feature_defs = None

    def run_dfs_single_table(
        self,
        spark_df: SparkDataFrame,
        index_col: str,
        target_entity: str = "data",
        time_index: Optional[str] = None,
        max_depth: int = 2,
        agg_primitives: Optional[List[str]] = None,
        trans_primitives: Optional[List[str]] = None,
        max_features: int = 100,
        cutoff_time: Optional[pd.DataFrame] = None,
        training_window: Optional[str] = None
    ) -> FeaturetoolsResult:
        """
        Run Deep Feature Synthesis on a single table

        Args:
            spark_df: PySpark DataFrame
            index_col: Primary key column name
            target_entity: Name for the entity
            time_index: Optional time column for temporal features
            max_depth: Maximum depth of feature generation
            agg_primitives: Aggregation primitives to use
            trans_primitives: Transform primitives to use
            max_features: Maximum number of features to generate
            cutoff_time: Optional cutoff times for temporal features
            training_window: Optional training window (e.g., "30 days")

        Returns:
            FeaturetoolsResult with generated features
        """
        import time
        start_time = time.time()

        ft = _get_featuretools()

        if self.verbose:
            logger.info(f"Converting PySpark DataFrame to Pandas (max {self.max_rows} rows)")

        # Convert to Pandas
        pdf = spark_to_pandas_safe(spark_df, max_rows=self.max_rows, sample=True)

        if self.verbose:
            logger.info(f"Converted {len(pdf)} rows to Pandas")

        # Create EntitySet
        es = ft.EntitySet(id=f"{target_entity}_es")

        # Add entity with automatic type inference
        es = es.add_dataframe(
            dataframe_name=target_entity,
            dataframe=pdf,
            index=index_col,
            time_index=time_index
        )

        self._entity_set = es

        # Default primitives if not specified
        if agg_primitives is None:
            agg_primitives = ["mean", "sum", "min", "max", "std", "count"]

        if trans_primitives is None:
            trans_primitives = [
                "add_numeric", "subtract_numeric", "multiply_numeric",
                "divide_numeric", "percentile", "cum_sum", "cum_mean"
            ]

        if self.verbose:
            logger.info(f"Running DFS with max_depth={max_depth}")

        # Run Deep Feature Synthesis
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name=target_entity,
            agg_primitives=agg_primitives,
            trans_primitives=trans_primitives,
            max_depth=max_depth,
            max_features=max_features,
            cutoff_time=cutoff_time,
            training_window=training_window,
            verbose=self.verbose
        )

        self._feature_defs = feature_defs
        generation_time = time.time() - start_time

        if self.verbose:
            logger.info(f"Generated {len(feature_defs)} features in {generation_time:.2f}s")

        return FeaturetoolsResult(
            feature_matrix=feature_matrix,
            feature_definitions=feature_defs,
            feature_names=list(feature_matrix.columns),
            entity_set=es,
            generation_time=generation_time
        )

    def run_dfs_multi_table(
        self,
        dataframes: Dict[str, SparkDataFrame],
        entities: Dict[str, EntityConfig],
        relationships: List[RelationshipConfig],
        target_entity: str,
        max_depth: int = 2,
        agg_primitives: Optional[List[str]] = None,
        trans_primitives: Optional[List[str]] = None,
        max_features: int = 100,
        cutoff_time: Optional[pd.DataFrame] = None
    ) -> FeaturetoolsResult:
        """
        Run Deep Feature Synthesis on multiple related tables

        Args:
            dataframes: Dictionary of entity_name -> SparkDataFrame
            entities: Dictionary of entity_name -> EntityConfig
            relationships: List of relationships between entities
            target_entity: Target entity for feature generation
            max_depth: Maximum depth of feature generation
            agg_primitives: Aggregation primitives to use
            trans_primitives: Transform primitives to use
            max_features: Maximum number of features to generate
            cutoff_time: Optional cutoff times for temporal features

        Returns:
            FeaturetoolsResult with generated features
        """
        import time
        start_time = time.time()

        ft = _get_featuretools()

        # Create EntitySet
        es = ft.EntitySet(id="multi_table_es")

        # Convert and add each dataframe
        pandas_dfs = {}
        for name, spark_df in dataframes.items():
            if self.verbose:
                logger.info(f"Converting {name} to Pandas")

            pdf = spark_to_pandas_safe(spark_df, max_rows=self.max_rows, sample=True)
            pandas_dfs[name] = pdf

            entity_config = entities[name]
            es = es.add_dataframe(
                dataframe_name=name,
                dataframe=pdf,
                index=entity_config.index,
                time_index=entity_config.time_index,
                logical_types=entity_config.logical_types
            )

        # Add relationships
        for rel in relationships:
            es = es.add_relationship(
                rel.parent_entity,
                rel.parent_column,
                rel.child_entity,
                rel.child_column
            )

        self._entity_set = es

        # Default primitives
        if agg_primitives is None:
            agg_primitives = ["mean", "sum", "min", "max", "std", "count", "num_unique"]

        if trans_primitives is None:
            trans_primitives = ["add_numeric", "subtract_numeric", "multiply_numeric"]

        if self.verbose:
            logger.info(f"Running DFS on {len(dataframes)} tables")

        # Run DFS
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name=target_entity,
            agg_primitives=agg_primitives,
            trans_primitives=trans_primitives,
            max_depth=max_depth,
            max_features=max_features,
            cutoff_time=cutoff_time,
            verbose=self.verbose
        )

        self._feature_defs = feature_defs
        generation_time = time.time() - start_time

        if self.verbose:
            logger.info(f"Generated {len(feature_defs)} features in {generation_time:.2f}s")

        return FeaturetoolsResult(
            feature_matrix=feature_matrix,
            feature_definitions=feature_defs,
            feature_names=list(feature_matrix.columns),
            entity_set=es,
            generation_time=generation_time
        )

    def to_spark(
        self,
        feature_matrix: pd.DataFrame,
        original_spark_df: Optional[SparkDataFrame] = None,
        join_column: Optional[str] = None
    ) -> SparkDataFrame:
        """
        Convert feature matrix back to PySpark DataFrame

        Args:
            feature_matrix: Pandas DataFrame from featuretools
            original_spark_df: Optional original PySpark DataFrame to join with
            join_column: Column to join on (usually the index column)

        Returns:
            PySpark DataFrame with features
        """
        # Reset index to make it a column
        feature_df = feature_matrix.reset_index()

        # Convert to Spark
        spark_features = pandas_to_spark(feature_df, self.spark)

        if original_spark_df is not None and join_column:
            # Join with original (to get back full dataset if sampled)
            feature_cols = [c for c in spark_features.columns if c != join_column]
            spark_features = original_spark_df.join(
                spark_features.select([join_column] + feature_cols),
                on=join_column,
                how="left"
            )

        return spark_features

    def get_feature_descriptions(self) -> List[Dict[str, str]]:
        """
        Get human-readable descriptions of generated features

        Returns:
            List of dictionaries with feature info
        """
        if self._feature_defs is None:
            return []

        descriptions = []
        for feature in self._feature_defs:
            descriptions.append({
                'name': feature.get_name(),
                'description': feature.get_description() if hasattr(feature, 'get_description') else str(feature),
                'type': str(type(feature).__name__)
            })

        return descriptions

    def get_primitive_options(self) -> Dict[str, List[str]]:
        """
        Get available featuretools primitives

        Returns:
            Dictionary with aggregation and transform primitives
        """
        ft = _get_featuretools()

        agg_primitives = ft.primitives.get_aggregation_primitives()
        trans_primitives = ft.primitives.get_transform_primitives()

        return {
            'aggregation': list(agg_primitives.keys()),
            'transform': list(trans_primitives.keys())
        }

    def save_features(self, filepath: str) -> None:
        """
        Save feature definitions to file for later use

        Args:
            filepath: Path to save feature definitions
        """
        ft = _get_featuretools()

        if self._feature_defs is None:
            raise ValueError("No features have been generated yet")

        ft.save_features(self._feature_defs, filepath)
        logger.info(f"Saved {len(self._feature_defs)} feature definitions to {filepath}")

    def load_features(self, filepath: str) -> List:
        """
        Load feature definitions from file

        Args:
            filepath: Path to load feature definitions from

        Returns:
            List of feature definitions
        """
        ft = _get_featuretools()

        self._feature_defs = ft.load_features(filepath)
        logger.info(f"Loaded {len(self._feature_defs)} feature definitions from {filepath}")

        return self._feature_defs

    def calculate_features(
        self,
        spark_df: SparkDataFrame,
        index_col: str,
        cutoff_time: Optional[pd.DataFrame] = None
    ) -> SparkDataFrame:
        """
        Calculate features using saved feature definitions

        Args:
            spark_df: New data to calculate features for
            index_col: Index column name
            cutoff_time: Optional cutoff times

        Returns:
            PySpark DataFrame with calculated features
        """
        ft = _get_featuretools()

        if self._feature_defs is None:
            raise ValueError("No feature definitions loaded. Call run_dfs_* or load_features first.")

        if self._entity_set is None:
            raise ValueError("No entity set available. Call run_dfs_* first.")

        # Convert to Pandas
        pdf = spark_to_pandas_safe(spark_df, max_rows=self.max_rows, sample=True)

        # Calculate features
        feature_matrix = ft.calculate_feature_matrix(
            features=self._feature_defs,
            entityset=self._entity_set,
            cutoff_time=cutoff_time
        )

        # Convert back to Spark
        return self.to_spark(feature_matrix, spark_df, index_col)


def quick_dfs(
    spark_df: SparkDataFrame,
    spark: SparkSession,
    index_col: str,
    max_depth: int = 2,
    max_features: int = 50
) -> Tuple[SparkDataFrame, List[str]]:
    """
    Quick Deep Feature Synthesis for a single table

    Args:
        spark_df: Input PySpark DataFrame
        spark: SparkSession
        index_col: Primary key column
        max_depth: Feature generation depth
        max_features: Maximum features to generate

    Returns:
        Tuple of (DataFrame with features, list of feature names)
    """
    engine = FeaturetoolsEngine(spark, verbose=False)
    result = engine.run_dfs_single_table(
        spark_df,
        index_col=index_col,
        max_depth=max_depth,
        max_features=max_features
    )

    spark_with_features = engine.to_spark(
        result.feature_matrix,
        spark_df,
        index_col
    )

    return spark_with_features, result.feature_names
