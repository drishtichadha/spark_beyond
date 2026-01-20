# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Spark Beyond** (recently rebranded to "Spark Tune") is a professional ML feature discovery platform built as a Streamlit web application. It automates the entire ML workflow: data validation → feature engineering → model training → results analysis.

**Tech Stack:**
- PySpark 4.1.0+ (distributed data processing)
- XGBoost 3.1.3+ (gradient boosting models via Spark integration)
- Streamlit 1.53.0+ (web UI framework)
- Pydantic 2.12.5+ (schema validation)
- Python 3.12

**Advanced Feature Engineering & AutoML Libraries:**
- ydata-profiling 4.10.0+ (comprehensive data profiling)
- featuretools 1.31.0+ (automated feature discovery)
- tsfresh 0.20.2+ (time-series feature extraction)
- composeml 0.10.1+ (prediction engineering for time-series)
- evalml 0.112.0+ (AutoML with model comparison)
- woodwork 0.31.0+ (semantic type inference for featuretools)

**Default Dataset:** `data/bank-additional-full.csv` (bank marketing classification data)

## Development Commands

### Dependency Management

**Preferred: UV (modern Python package manager)**
```bash
uv sync                          # Install all dependencies from uv.lock
uv run streamlit run app.py      # Run app with UV
```

**Alternative: pip**
```bash
pip install -e .                 # Install in development mode
streamlit run app.py             # Run app
```

### Running the Application

```bash
# Default (opens on port 8501)
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8502

# With UV
uv run streamlit run app.py
```

**Note:** There are no formal tests, linting, or build commands. Testing is exploratory via Jupyter notebooks (`example.ipynb`, `feature_discovery.ipynb`, `pyspark_xgboost.ipynb`).

## Architecture & Core Modules

### High-Level Architecture

The application follows a **page-based navigation flow** using Streamlit session state to maintain data across pages:

```
1. Data Overview      → Load data, define problem, validate schema, data profiling
2. Feature Engineering → Generate automated features, advanced preprocessing
3. Model Training     → Train XGBoost with hyperparameters
4. Model Comparison   → Baseline models, AutoML, impact analysis
5. Results & Insights → Visualize metrics and feature importance
```

### Core Module Responsibilities

**`core/discovery.py` (211 lines)**
- Defines ML problem types (`ProblemType.regression`, `ProblemType.classification`)
- Validates schemas with `SchemaChecks` class
- Uses Pydantic models: `Problem`, `ClassificationProblem`, `RegressionProblem`
- Performs data quality checks: `target_checks()`, `categorical_checks()`, `numerical_checks()`, `datetime_checks()`

**`core/features/auto_feature_generator.py` (426 lines)**
- `AutoFeatureGenerator` class orchestrates feature creation
- Generates numerical transformations: log, sqrt, square, cube
- Creates interaction features: multiplication, division, addition, subtraction
- Handles binning, datetime extraction, string features
- Returns PySpark DataFrame with new columns

**`core/features/process.py` (91 lines)**
- `PreProcessVariables` handles encoding pipeline
- **Pipeline stages:**
  1. `StringIndexer` for categorical variables
  2. `OneHotEncoder` for indexed categoricals
  3. `VectorAssembler` to combine all features into single vector column
- `target_processing()` encodes classification targets to 0/1

**`core/features/feature_selector.py` (232 lines)**
- `FeatureSelector` trains XGBoost models using `SparkXGBClassifier` or `SparkXGBRegressor`
- Default hyperparameters: `num_round=100`, `max_depth=4`, `learning_rate=0.1`
- Extracts feature importance scores
- Generates matplotlib visualizations

**`core/metrics/base.py` (127 lines)**
- Abstract `Metrics` class for analytics
- Supports filtering with `FilterOperator` enum (EQ, NEQ, GRT, LWT, LIKE, etc.)
- Methods: `get_aggregation()`, `calculate()`, `build_filter_exp()`

**`core/metrics/segment_analysis.py` (215 lines)**
- `SegmentAnalysis` for dimension-based comparisons
- **Note:** References external `kpi_analytics` module not present in this project

### New Enhanced Modules (v0.2.0)

**`core/utils/spark_pandas_bridge.py`**
- Safe conversion between PySpark and Pandas DataFrames
- Automatic sampling for large datasets to avoid memory issues
- Memory estimation and conversion strategy recommendations
- Key functions: `spark_to_pandas_safe()`, `pandas_to_spark()`, `auto_convert_to_pandas()`

**`core/utils/time_series_detector.py`**
- Auto-detect time-series structure in datasets
- Identify entity columns, time columns, and frequency
- Recommend appropriate time-series features
- Returns `TimeSeriesInfo` dataclass with detection results

**`core/profiling/ydata_profiler.py`**
- Comprehensive data profiling using ydata-profiling
- `DataProfiler` class wraps ydata-profiling for PySpark DataFrames
- Generates interactive HTML reports with statistics
- Extracts quality alerts and preprocessing recommendations
- `quick_profile()` function for fast summary statistics

**`core/profiling/data_quality.py`**
- PySpark-native data quality checks (no Pandas conversion)
- `DataQualityChecker` for large dataset analysis
- Checks: missing values, duplicates, outliers, cardinality
- Calculates overall quality score (0-100)
- Generates preprocessing recommendations

**`core/features/preprocessing_enhanced.py`**
- Feature-engine inspired preprocessing for PySpark
- `EnhancedPreprocessor` with configurable pipeline
- Imputation strategies: mean, median, mode, constant
- Outlier handling: IQR capping, z-score removal
- Feature scaling: standard, minmax, robust
- Rare category grouping and cyclical encoding

**`core/features/featuretools_engine.py`**
- Featuretools integration for automated feature discovery
- `FeaturetoolsEngine` handles PySpark ↔ Pandas conversion
- Deep Feature Synthesis (DFS) for single and multi-table data
- Entity relationship detection and feature primitive selection
- Supports temporal features with cutoff times

**`core/features/tsfresh_engine.py`**
- TSFresh integration for time-series feature extraction
- `TSFreshEngine` with configurable extraction modes
- Modes: minimal (~10 features), efficient (~100), comprehensive (~750)
- Automatic feature relevance filtering based on target
- `TSFreshPySparkNative` for pure PySpark implementation

**`core/features/composeml_engine.py`**
- ComposeML integration for prediction engineering
- Define prediction problems for time-series data
- Automatic label generation with cutoff time management
- Prevents data leakage in temporal ML problems
- `PredictionProblemLibrary` with common templates (churn, LTV, etc.)

**`core/models/baseline_models.py`**
- Simple baseline models for comparison
- `BaselineModels` trains: Naive, Decision Tree, Logistic Regression, Random Forest
- PySpark ML implementations for classification and regression
- Quick training with automatic metric calculation
- Establishes performance baselines for impact measurement

**`core/models/evalml_runner.py`**
- EvalML AutoML integration for automated model selection
- `EvalMLRunner` handles full AutoML search
- Automatic model selection, hyperparameter tuning, ensembling
- Supports classification (binary/multiclass) and regression
- Extracts feature importance and cross-validation scores

**`core/models/model_comparison.py`**
- Framework for comparing models and measuring impact
- `ModelComparison` tracks multiple experiments
- Calculates improvements: feature engineering impact, AutoML impact
- Generates comparison tables and summary reports
- `ComparisonResult` with rankings and best model identification

### PySpark Integration

**Spark Session Initialization** ([app.py:56-64](app.py#L56-L64))
```python
SparkSession.builder \
    .master("local[*]") \           # Use all available cores
    .appName("Spark Beyond ML App") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")  # Suppress verbose logs
```

**Data Loading Pattern** ([app.py:66-74](app.py#L66-L74))
```python
df = spark.read.options(
    header=True,
    inferSchema='True',  # Auto-detect column types
    delimiter=','
).csv(file_path)
```

### Streamlit Session State Management

**Critical session state variables** (defined in [app.py:43-54](app.py#L43-L54)):
- `st.session_state.spark` - SparkSession instance (singleton)
- `st.session_state.df` - Original PySpark DataFrame
- `st.session_state.schema_checker` - SchemaChecks instance
- `st.session_state.feature_selector` - FeatureSelector instance
- `st.session_state.df_with_features` - DataFrame after feature engineering
- `st.session_state.metrics` - Dictionary of model performance metrics

**Data Flow:**
1. Page 1 loads data → `st.session_state.df`
2. Page 2 generates features → `st.session_state.df_with_features`
3. Page 3 trains model → stores in `st.session_state.feature_selector`
4. Page 4 displays results from `st.session_state.metrics`

## ML Pipeline Flow

### 1. Problem Definition
- User selects target column and problem type (classification/regression)
- `Problem` Pydantic model validates configuration
- `SchemaChecks.check()` runs data quality validation

### 2. Feature Engineering
- `AutoFeatureGenerator` creates new columns based on user selections
- Original DataFrame is transformed, not copied (PySpark lazy evaluation)
- Generated features are named with prefixes: `log_`, `sqrt_`, `interaction_`, `binned_`, etc.

### 3. Preprocessing
- `PreProcessVariables.process()` builds Spark ML Pipeline
- Categorical features → StringIndexer → OneHotEncoder
- All features → VectorAssembler → single `features` column
- Target encoding for classification (maps desired result to 1, others to 0)

### 4. Model Training
- `FeatureSelector.train_model()` fits SparkXGBClassifier/Regressor
- **Default config:** num_round=100, max_depth=4, eta=0.1
- Evaluation metric: `logloss` (classification) or `squarederror` (regression)
- Returns trained model with feature importance

### 5. Results Analysis
- Extract feature importance from XGBoost model
- Calculate metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- Visualize with Plotly and Matplotlib
- Export predictions to CSV

## Important Notes

### Spark Configuration
- Always runs in **local mode** (`local[*]`) - not a distributed cluster
- Log level set to ERROR to reduce console noise
- For memory issues with large datasets, modify: `.config("spark.driver.memory", "4g")`

### XGBoost Model Customization
- Models use Spark-integrated XGBoost (`xgboost.spark` package)
- Hyperparameters are configurable via Streamlit UI sliders
- Feature importance is extracted using `get_feature_importances()` method

### Known Limitations
- No formal test suite (use Jupyter notebooks for experimentation)
- `core/exceptions.py` is empty (placeholder)
- `core/metrics/segment_analysis.py` references external `kpi_analytics` module not in repo
- Branding inconsistency: "Spark Beyond" in code vs "Spark Tune" in recent UI changes

### File Paths
- Logo path is hardcoded: `/Users/aays/Documents/aays/spark_beyond/data/Aideticlogo.png` ([app.py:82](app.py#L82))
- Default dataset path: `data/bank-additional-full.csv`

### Troubleshooting
- **Spark hostname warnings:** Safe to ignore, or set `SPARK_LOCAL_IP` environment variable
- **Port conflicts:** Use `--server.port` flag to change from default 8501
- **Memory errors:** Reduce feature generation options or increase Spark driver memory

## PySpark ↔ Pandas Conversion Strategy

Since many advanced libraries (featuretools, tsfresh, evalml, ydata-profiling) require Pandas DataFrames, the platform uses a **PySpark-first approach** with automatic conversion:

**Strategy:**
1. Keep all data processing in PySpark for scalability
2. Convert to Pandas only when necessary for advanced libraries
3. Use sampling for large datasets (>500K rows) to avoid memory issues
4. Convert back to PySpark after Pandas operations

**Key utility:** `core/utils/spark_pandas_bridge.py`
```python
from backend.core.utils.spark_pandas_bridge import spark_to_pandas_safe, auto_convert_to_pandas

# Safe conversion with automatic sampling
pdf = spark_to_pandas_safe(spark_df, max_rows=100000, sample=True)

# Auto-detect optimal strategy
pdf, strategy = auto_convert_to_pandas(spark_df, max_memory_gb=2.0)
```

## Enhanced Feature Engineering

### AutoFeatureGenerator Enhancements

The enhanced `AutoFeatureGenerator` now includes:
- `generate_polynomial_features()` - Higher-order polynomial terms
- `generate_ratio_features()` - Domain-specific ratio calculations
- `generate_cyclical_features()` - Sin/cos encoding for periodic features
- `generate_target_encoding()` - Target-based categorical encoding
- `generate_lag_features()` - Time-series lag features
- `generate_rolling_features()` - Rolling window statistics
- `generate_diff_features()` - Difference and percentage change
- `generate_statistical_features()` - Row-wise statistics
- `get_feature_summary()` - Categorized feature breakdown

### Time-Series Detection

Automatic detection of time-series structure:
```python
from backend.core.utils.time_series_detector import detect_time_series_structure

ts_info = detect_time_series_structure(df, schema_checks)
if ts_info.is_time_series:
    print(f"Time column: {ts_info.time_column}")
    print(f"Frequency: {ts_info.frequency}")
    print(f"Recommended features: {ts_info.recommended_features}")
```

## Model Comparison Framework

Track experiments and measure impact:
```python
from backend.core.models.model_comparison import ModelComparison

comparison = ModelComparison(primary_metric='accuracy')

# Add experiments
comparison.add_experiment(
    name="Baseline",
    model_name="Decision Tree",
    feature_set="original",
    metrics={'accuracy': 0.75},
    training_time=5.2
)

# Get comparison with improvements
result = comparison.get_comparison()
print(result.improvements)  # Shows feature engineering impact
```
