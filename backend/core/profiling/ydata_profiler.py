"""
YData Profiling integration for comprehensive data analysis

This module wraps ydata-profiling to generate interactive data quality reports
from PySpark DataFrames with automatic sampling for large datasets.
"""

from pyspark.sql import DataFrame as SparkDataFrame
from backend.core.utils.spark_pandas_bridge import spark_to_pandas_safe, estimate_memory_usage
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import pandas as pd
import tempfile
import logging
import os

logger = logging.getLogger(__name__)

# Lazy import for ydata-profiling (heavy dependency)
_ydata_profiling = None


def _get_ydata_profiling():
    """Lazy load ydata-profiling to avoid startup overhead"""
    global _ydata_profiling
    if _ydata_profiling is None:
        try:
            from ydata_profiling import ProfileReport as YDataProfileReport
            _ydata_profiling = YDataProfileReport
        except ImportError:
            raise ImportError(
                "ydata-profiling is not installed. "
                "Install it with: pip install ydata-profiling"
            )
    return _ydata_profiling


@dataclass
class ProfileReport:
    """
    Container for profiling results

    Attributes:
        html_report: HTML string of the full report
        summary: Dictionary with key statistics
        alerts: List of data quality alerts
        correlations: Correlation matrix as dict
        missing_values: Missing value analysis
        sample_info: Information about sampling applied
    """
    html_report: str
    summary: Dict[str, Any]
    alerts: List[Dict[str, str]]
    correlations: Optional[Dict[str, Any]]
    missing_values: Dict[str, float]
    sample_info: Dict[str, Any]
    variables_info: Dict[str, Dict[str, Any]]


class DataProfiler:
    """
    Data profiling using ydata-profiling with PySpark support

    This class handles:
    - Automatic conversion from PySpark to Pandas with sampling
    - Generation of comprehensive data profiles
    - Extraction of key insights and alerts
    - Caching of reports for performance

    Example:
        profiler = DataProfiler(spark_df, title="Bank Marketing Dataset")
        report = profiler.generate_report()

        # Display in Streamlit
        st.components.v1.html(report.html_report, height=800, scrolling=True)

        # Access summary stats
        print(report.summary)
    """

    def __init__(
        self,
        df: SparkDataFrame,
        title: str = "Data Profile Report",
        max_rows: int = 100000,
        minimal: bool = False,
        explorative: bool = False
    ):
        """
        Initialize the DataProfiler

        Args:
            df: PySpark DataFrame to profile
            title: Title for the report
            max_rows: Maximum rows to sample for profiling
            minimal: If True, generate minimal report (faster)
            explorative: If True, generate explorative report (more detailed but slower)
        """
        self.spark_df = df
        self.title = title
        self.max_rows = max_rows
        self.minimal = minimal
        self.explorative = explorative
        self._pandas_df: Optional[pd.DataFrame] = None
        self._profile = None
        self._sample_info: Dict[str, Any] = {}

    def _convert_to_pandas(self) -> pd.DataFrame:
        """Convert PySpark DataFrame to Pandas with sampling"""
        if self._pandas_df is not None:
            return self._pandas_df

        # Estimate memory and determine conversion strategy
        memory_info = estimate_memory_usage(self.spark_df)
        total_rows = memory_info.get('row_count', 0)

        if total_rows > self.max_rows:
            logger.info(f"Sampling {self.max_rows:,} rows from {total_rows:,} for profiling")
            self._sample_info = {
                'sampled': True,
                'original_rows': total_rows,
                'sampled_rows': self.max_rows,
                'sample_ratio': self.max_rows / total_rows
            }
            self._pandas_df = spark_to_pandas_safe(
                self.spark_df,
                max_rows=self.max_rows,
                sample=True
            )
        else:
            self._sample_info = {
                'sampled': False,
                'original_rows': total_rows,
                'sampled_rows': total_rows,
                'sample_ratio': 1.0
            }
            self._pandas_df = spark_to_pandas_safe(self.spark_df)

        logger.info(f"Converted to Pandas DataFrame: {self._pandas_df.shape}")
        return self._pandas_df

    def generate_report(
        self,
        correlations: bool = True,
        interactions: bool = False,
        missing_diagrams: bool = True,
        samples: int = 10
    ) -> ProfileReport:
        """
        Generate comprehensive data profile report

        Args:
            correlations: Include correlation analysis
            interactions: Include feature interaction analysis (slow for large datasets)
            missing_diagrams: Include missing value diagrams
            samples: Number of sample rows to include

        Returns:
            ProfileReport object with all profiling results
        """
        YDataProfileReport = _get_ydata_profiling()
        pdf = self._convert_to_pandas()

        logger.info(f"Generating profile report for {pdf.shape[0]:,} rows, {pdf.shape[1]} columns")

        # Configure report settings
        config = {
            'title': self.title,
            'samples': {'head': samples, 'tail': samples},
            'correlations': {
                'auto': correlations,
                'pearson': correlations,
                'spearman': correlations,
                'kendall': False,  # Skip Kendall (slow)
                'phi_k': False,    # Skip phi_k (slow)
            },
            'interactions': {
                'continuous': interactions,
                'targets': []
            },
            'missing_diagrams': {
                'bar': missing_diagrams,
                'matrix': missing_diagrams and pdf.shape[1] <= 50,  # Limit for performance
                'heatmap': missing_diagrams and pdf.shape[1] <= 30,
            },
            'duplicates': {
                'head': 10
            }
        }

        # Generate profile
        if self.minimal:
            profile = YDataProfileReport(pdf, minimal=True, title=self.title)
        elif self.explorative:
            profile = YDataProfileReport(pdf, explorative=True, title=self.title)
        else:
            profile = YDataProfileReport(pdf, **config)

        self._profile = profile

        # Extract key information
        return self._extract_report_data(profile)

    def _extract_report_data(self, profile) -> ProfileReport:
        """Extract structured data from ydata-profiling report"""
        # Get HTML report
        html_report = profile.to_html()

        # Get description/summary
        description = profile.get_description()

        # Extract summary statistics
        summary = self._extract_summary(description)

        # Extract alerts
        alerts = self._extract_alerts(description)

        # Extract correlations
        correlations = self._extract_correlations(description)

        # Extract missing values
        missing_values = self._extract_missing_values(description)

        # Extract variable info
        variables_info = self._extract_variables_info(description)

        return ProfileReport(
            html_report=html_report,
            summary=summary,
            alerts=alerts,
            correlations=correlations,
            missing_values=missing_values,
            sample_info=self._sample_info,
            variables_info=variables_info
        )

    def _get_description_dict(self, description) -> Dict[str, Any]:
        """Convert BaseDescription to dict, handling different ydata-profiling versions"""
        # Try to convert to dict if possible
        if hasattr(description, 'to_dict'):
            return description.to_dict()
        elif hasattr(description, '__dict__'):
            return vars(description)
        elif isinstance(description, dict):
            return description
        else:
            # Access attributes directly for BaseDescription objects
            result = {}
            for attr in ['table', 'variables', 'alerts', 'correlations', 'missing', 'duplicates']:
                if hasattr(description, attr):
                    val = getattr(description, attr)
                    if hasattr(val, 'to_dict'):
                        result[attr] = val.to_dict()
                    elif hasattr(val, '__dict__'):
                        result[attr] = vars(val)
                    else:
                        result[attr] = val
            return result

    def _get_variables_dict(self, description) -> Dict[str, Any]:
        """Extract variables as a dict from description"""
        # Try direct attribute access first (for BaseDescription)
        if hasattr(description, 'variables'):
            variables = description.variables
            if isinstance(variables, dict):
                return variables
            elif hasattr(variables, 'to_dict'):
                return variables.to_dict()
            elif hasattr(variables, '__iter__'):
                # It might be iterable with items
                result = {}
                try:
                    for key in variables:
                        val = variables[key]
                        if hasattr(val, 'to_dict'):
                            result[key] = val.to_dict()
                        elif hasattr(val, '__dict__'):
                            result[key] = vars(val)
                        else:
                            result[key] = val
                    return result
                except Exception:
                    pass
        # Fallback to get method
        if hasattr(description, 'get'):
            return description.get('variables', {})
        return {}

    def _extract_summary(self, description) -> Dict[str, Any]:
        """Extract summary statistics from profile description"""
        try:
            # Get table info - try attribute access first, then dict access
            if hasattr(description, 'table'):
                table = description.table
                if hasattr(table, 'to_dict'):
                    table = table.to_dict()
                elif hasattr(table, '__dict__'):
                    table = vars(table)
            elif hasattr(description, 'get'):
                table = description.get('table', {})
            else:
                table = {}

            if not isinstance(table, dict):
                table = {}

            return {
                'n_rows': table.get('n', 0),
                'n_columns': table.get('n_var', 0),
                'missing_cells': table.get('n_cells_missing', 0),
                'missing_cells_pct': table.get('p_cells_missing', 0) * 100,
                'duplicate_rows': table.get('n_duplicates', 0),
                'duplicate_rows_pct': table.get('p_duplicates', 0) * 100 if table.get('p_duplicates') else 0,
                'memory_size': table.get('memory_size', 0),
                'types': table.get('types', {}),
            }
        except Exception as e:
            logger.warning(f"Error extracting summary: {e}")
            return {}

    def _extract_alerts(self, description) -> List[Dict[str, str]]:
        """Extract data quality alerts from profile"""
        alerts = []
        try:
            # Get alerts - try attribute access first, then dict access
            if hasattr(description, 'alerts'):
                raw_alerts = description.alerts
            elif hasattr(description, 'get'):
                raw_alerts = description.get('alerts', [])
            else:
                raw_alerts = []

            if raw_alerts is None:
                raw_alerts = []

            for alert in raw_alerts:
                alerts.append({
                    'column': str(alert.column_name) if hasattr(alert, 'column_name') else 'Dataset',
                    'type': str(alert.alert_type.name) if hasattr(alert.alert_type, 'name') else str(alert.alert_type),
                    'message': str(alert)
                })
        except Exception as e:
            logger.warning(f"Error extracting alerts: {e}")
        return alerts

    def _extract_correlations(self, description) -> Optional[Dict[str, Any]]:
        """Extract correlation matrix from profile"""
        try:
            # Get correlations - try attribute access first, then dict access
            if hasattr(description, 'correlations'):
                correlations = description.correlations
            elif hasattr(description, 'get'):
                correlations = description.get('correlations', {})
            else:
                correlations = {}

            if correlations:
                # Convert to serializable format
                result = {}
                if isinstance(correlations, dict):
                    items = correlations.items()
                elif hasattr(correlations, 'items'):
                    items = correlations.items()
                else:
                    return None

                for corr_type, corr_data in items:
                    if hasattr(corr_data, 'to_dict'):
                        result[corr_type] = corr_data.to_dict()
                    else:
                        result[corr_type] = corr_data
                return result
        except Exception as e:
            logger.warning(f"Error extracting correlations: {e}")
        return None

    def _extract_missing_values(self, description) -> Dict[str, float]:
        """Extract missing value percentages per column"""
        missing = {}
        try:
            variables = self._get_variables_dict(description)
            for var_name, var_info in variables.items():
                if isinstance(var_info, dict):
                    missing_pct = var_info.get('p_missing', 0)
                    if missing_pct and missing_pct > 0:
                        missing[var_name] = missing_pct * 100
        except Exception as e:
            logger.warning(f"Error extracting missing values: {e}")
        return missing

    def _extract_variables_info(self, description) -> Dict[str, Dict[str, Any]]:
        """Extract detailed variable information"""
        variables_info = {}
        try:
            variables = self._get_variables_dict(description)
            for var_name, var_info in variables.items():
                if isinstance(var_info, dict):
                    p_missing = var_info.get('p_missing', 0) or 0
                    variables_info[var_name] = {
                        'type': var_info.get('type', 'Unknown'),
                        'n_missing': var_info.get('n_missing', 0),
                        'p_missing': p_missing * 100,
                        'n_distinct': var_info.get('n_distinct', 0),
                        'is_unique': var_info.get('is_unique', False),
                        'n_unique': var_info.get('n_unique', 0),
                    }
                    # Add numeric stats if available
                    if 'mean' in var_info:
                        variables_info[var_name].update({
                            'mean': var_info.get('mean'),
                            'std': var_info.get('std'),
                            'min': var_info.get('min'),
                            'max': var_info.get('max'),
                            'median': var_info.get('median'),
                        })
        except Exception as e:
            logger.warning(f"Error extracting variables info: {e}")
        return variables_info

    def save_report(self, filepath: str) -> str:
        """
        Save the HTML report to a file

        Args:
            filepath: Path to save the HTML report

        Returns:
            Path to the saved file
        """
        if self._profile is None:
            self.generate_report()

        self._profile.to_file(filepath)
        logger.info(f"Report saved to {filepath}")
        return filepath

    def get_recommendations(self) -> List[Dict[str, str]]:
        """
        Get preprocessing recommendations based on profile analysis

        Returns:
            List of recommendations with column, issue, and suggested action
        """
        if self._profile is None:
            self.generate_report()

        recommendations = []
        description = self._profile.get_description()
        variables = self._get_variables_dict(description)

        for var_name, var_info in variables.items():
            if not isinstance(var_info, dict):
                continue

            # High missing values
            p_missing = var_info.get('p_missing', 0) or 0
            if p_missing > 0.3:
                recommendations.append({
                    'column': var_name,
                    'issue': f'High missing values ({p_missing*100:.1f}%)',
                    'action': 'Consider dropping or imputing with median/mode',
                    'priority': 'high' if p_missing > 0.5 else 'medium'
                })
            elif p_missing > 0.05:
                recommendations.append({
                    'column': var_name,
                    'issue': f'Missing values ({p_missing*100:.1f}%)',
                    'action': 'Impute with mean/median for numeric, mode for categorical',
                    'priority': 'low'
                })

            # High cardinality categorical
            var_type = var_info.get('type', '')
            n_distinct = var_info.get('n_distinct', 0) or 0
            if 'Categorical' in str(var_type) and n_distinct > 50:
                recommendations.append({
                    'column': var_name,
                    'issue': f'High cardinality ({n_distinct} unique values)',
                    'action': 'Consider grouping rare categories or using target encoding',
                    'priority': 'medium'
                })

            # Constant columns
            if var_info.get('is_unique') == False and n_distinct == 1:
                recommendations.append({
                    'column': var_name,
                    'issue': 'Constant column (single value)',
                    'action': 'Remove column - provides no information',
                    'priority': 'high'
                })

            # Highly skewed numeric
            skewness = var_info.get('skewness', 0) or 0
            if abs(skewness) > 2:
                recommendations.append({
                    'column': var_name,
                    'issue': f'Highly skewed distribution (skewness: {skewness:.2f})',
                    'action': 'Consider log transformation or Box-Cox',
                    'priority': 'medium'
                })

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))

        return recommendations


def quick_profile(
    df: SparkDataFrame,
    max_rows: int = 50000
) -> Dict[str, Any]:
    """
    Generate a quick summary profile without full HTML report

    Args:
        df: PySpark DataFrame to profile
        max_rows: Maximum rows to sample

    Returns:
        Dictionary with key statistics
    """
    profiler = DataProfiler(df, max_rows=max_rows, minimal=True)
    report = profiler.generate_report()

    return {
        'summary': report.summary,
        'missing_values': report.missing_values,
        'alerts': report.alerts,
        'sample_info': report.sample_info,
        'recommendations': profiler.get_recommendations()
    }
