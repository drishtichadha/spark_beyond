# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Spark Tune** is a professional ML feature discovery platform built as a modern SPA with a FastAPI backend and React TypeScript frontend. It automates the entire ML workflow: data validation → feature engineering → model training → results analysis.

**Tech Stack:**
- **Backend:** FastAPI (REST API + WebSocket)
- **Frontend:** React 19 + TypeScript + Vite
- **ML Processing:** PySpark 4.1.0+ (distributed data processing)
- **Models:** XGBoost (gradient boosting via Spark integration)
- **Validation:** Pydantic 2.12.5+ (schema validation)
- **Session Management:** Redis with file storage fallback
- **Python:** 3.12

**Advanced Feature Engineering & AutoML Libraries:**
- ydata-profiling 4.10.0+ (comprehensive data profiling)
- featuretools 1.31.0+ (automated feature discovery)
- tsfresh 0.20.2+ (time-series feature extraction)
- composeml 0.10.1+ (prediction engineering for time-series)
- lightautoml 0.4.2+ (AutoML framework)
- woodwork 0.31.0+ (semantic type inference for featuretools)

**Default Dataset:** `data/bank-additional-full.csv` (bank marketing classification data)

## Development Commands

### Dependency Management

**Preferred: UV (modern Python package manager)**
```bash
uv sync                          # Install all dependencies from uv.lock
```

**Alternative: pip**
```bash
pip install -e .                 # Install in development mode
```

### Running the Application

**Backend (FastAPI on port 8000):**
```bash
# With uvicorn directly
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Or via Python module
python -m backend.main
```

**Frontend (React on port 5173):**
```bash
cd frontend
npm install                      # Install dependencies (first time)
npm run dev                      # Start development server
```

**Note:** There are no formal tests, linting, or build commands. Testing is exploratory via Jupyter notebooks (`example.ipynb`, `feature_discovery.ipynb`, `pyspark_xgboost.ipynb`).

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + TypeScript)             │
│                                                               │
│  /frontend/src/                                              │
│  ├── pages/                   (6 page components)            │
│  │   ├── Dashboard.tsx                                       │
│  │   ├── DataOverview.tsx                                    │
│  │   ├── FeatureEngineering.tsx                              │
│  │   ├── ModelTraining.tsx                                   │
│  │   ├── ModelComparison.tsx                                 │
│  │   └── Insights.tsx                                        │
│  ├── components/              (UI components)                │
│  │   ├── layout/              (Layout, Sidebar)              │
│  │   └── ui/                  (Radix UI wrappers)            │
│  ├── contexts/                (SessionContext)               │
│  └── lib/                     (API client, hooks)            │
└─────────────────────────────────────────────────────────────┘
              ↓ HTTP/WebSocket (Axios + TanStack Query)
┌─────────────────────────────────────────────────────────────┐
│              BACKEND API (FastAPI) - Port 8000               │
│                                                               │
│  /backend/                                                   │
│  ├── main.py                  (FastAPI app entry point)     │
│  ├── routes/                  (4 routers)                    │
│  │   ├── data.py              (Data loading, validation)     │
│  │   ├── features.py          (Feature engineering)          │
│  │   ├── models.py            (Model training)               │
│  │   └── insights.py          (Results & analysis)           │
│  ├── schemas/                 (Pydantic API models)          │
│  ├── services/                                               │
│  │   └── spark_service.py     (ML/Spark operations)          │
│  ├── core/                    (35+ Python modules)           │
│  ├── middleware/              (Session middleware)           │
│  └── config/                  (Configuration)                │
└─────────────────────────────────────────────────────────────┘
              ↓ Redis (Sessions) + File Storage
┌─────────────────────────────────────────────────────────────┐
│                   DATA & STORAGE LAYER                       │
│  - Redis: Session management                                 │
│  - File Storage: ./data/sessions/ (fallback)                 │
│  - Default Dataset: data/bank-additional-full.csv            │
└─────────────────────────────────────────────────────────────┘
```

### Application Flow

The application follows a **6-page navigation flow**:

```
1. Dashboard         → Overview and session management
2. Data Overview     → Load data, define problem, validate schema, data profiling
3. Feature Engineering → Generate automated features, advanced preprocessing
4. Model Training    → Train XGBoost with hyperparameters
5. Model Comparison  → Baseline models, AutoML, impact analysis
6. Insights          → Visualize metrics and feature importance
```

### Core Module Responsibilities

**`backend/main.py`**
- FastAPI application entry point
- WebSocket endpoints for real-time updates
- Session middleware configuration
- CORS and security settings

**`backend/routes/data.py`**
- Data loading and validation endpoints
- Schema checks and problem definition
- Data quality analysis

**`backend/routes/features.py`**
- Feature generation endpoints
- Preprocessing pipeline configuration
- Feature statistics and summaries

**`backend/routes/models.py`**
- Model training endpoints
- Hyperparameter configuration
- Training progress WebSocket

**`backend/routes/insights.py`**
- Feature importance analysis
- Model metrics and evaluation
- Results export

**`backend/services/spark_service.py`**
- Core ML service orchestrating all Spark operations
- Session-specific Spark context management
- DataFrame caching and persistence

### Backend Core Modules

**`core/discovery.py`**
- Defines ML problem types (`ProblemType.regression`, `ProblemType.classification`)
- Validates schemas with `SchemaChecks` class
- Uses Pydantic models: `Problem`, `ClassificationProblem`, `RegressionProblem`
- Performs data quality checks: `target_checks()`, `categorical_checks()`, `numerical_checks()`, `datetime_checks()`

**`core/features/auto_feature_generator.py`**
- `AutoFeatureGenerator` class orchestrates feature creation
- Generates numerical transformations: log, sqrt, square, cube
- Creates interaction features: multiplication, division, addition, subtraction
- Handles binning, datetime extraction, string features
- Returns PySpark DataFrame with new columns

**`core/features/process.py`**
- `PreProcessVariables` handles encoding pipeline
- **Pipeline stages:**
  1. `StringIndexer` for categorical variables
  2. `OneHotEncoder` for indexed categoricals
  3. `VectorAssembler` to combine all features into single vector column
- `target_processing()` encodes classification targets to 0/1

**`core/features/feature_selector.py`**
- `FeatureSelector` trains XGBoost models using `SparkXGBClassifier` or `SparkXGBRegressor`
- Default hyperparameters: `num_round=100`, `max_depth=4`, `learning_rate=0.1`
- Extracts feature importance scores

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

**`core/profiling/data_quality.py`**
- PySpark-native data quality checks (no Pandas conversion)
- `DataQualityChecker` for large dataset analysis
- Checks: missing values, duplicates, outliers, cardinality
- Calculates overall quality score (0-100)

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

**`core/features/tsfresh_engine.py`**
- TSFresh integration for time-series feature extraction
- `TSFreshEngine` with configurable extraction modes
- Modes: minimal (~10 features), efficient (~100), comprehensive (~750)

**`core/features/composeml_engine.py`**
- ComposeML integration for prediction engineering
- Define prediction problems for time-series data
- Automatic label generation with cutoff time management

**`core/models/baseline_models.py`**
- Simple baseline models for comparison
- `BaselineModels` trains: Naive, Decision Tree, Logistic Regression, Random Forest
- PySpark ML implementations for classification and regression

**`core/models/evalml_runner.py`**
- AutoML integration for automated model selection
- `AutoMLRunner` handles full AutoML search with LightAutoML
- Automatic model selection, hyperparameter tuning, ensembling

**`core/models/model_comparison.py`**
- Framework for comparing models and measuring impact
- `ModelComparison` tracks multiple experiments
- Calculates improvements: feature engineering impact, AutoML impact

### Frontend Structure

```
frontend/
├── package.json                (React 19, TypeScript, Vite)
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
├── src/
│   ├── main.tsx               (React entry point)
│   ├── App.tsx                (Router configuration)
│   ├── pages/                 (Page components)
│   ├── components/
│   │   ├── layout/            (Layout, Sidebar)
│   │   └── ui/                (Radix UI wrappers)
│   ├── contexts/              (SessionContext)
│   └── lib/                   (API client, hooks)
```

### API Communication

**Session Management:**
- Frontend generates session ID stored in localStorage
- Every API request includes `X-Session-ID` header via Axios interceptor
- Backend validates and routes to session-specific Spark service

**API Response Format:**
```typescript
interface APIResponse<T> {
  success: boolean;
  message: string;
  data?: T;
}
```

**WebSocket Endpoints:**
- `/ws` - General echo WebSocket
- `/ws/microsegments` - Streaming microsegment discovery with progress

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
- Export predictions to CSV

## PySpark ↔ Pandas Conversion Strategy

Since many advanced libraries (featuretools, tsfresh, ydata-profiling) require Pandas DataFrames, the platform uses a **PySpark-first approach** with automatic conversion:

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

### AutoFeatureGenerator Features

The `AutoFeatureGenerator` includes:
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

## Important Notes

### Spark Configuration
- Always runs in **local mode** (`local[*]`) - not a distributed cluster
- Log level set to ERROR to reduce console noise
- For memory issues with large datasets, modify: `.config("spark.driver.memory", "4g")`

### Resource Tiers
The system supports three deployment tiers with configurable resource limits:
- `production` - Standard limits
- `development` - Relaxed limits for testing
- `enterprise` - Higher limits

Set via environment variable: `RESOURCE_TIER=production`

### Session Management
- Sessions are managed via Redis with file storage fallback
- Session data stored in `./backend/data/sessions/`
- Session ID passed via `X-Session-ID` header

### Security
- File path validation to prevent traversal attacks
- CORS configured via environment variables
- Session validation middleware
- Resource limits enforced at multiple layers

### Known Limitations
- No formal test suite (use Jupyter notebooks for experimentation)
- `core/exceptions.py` is empty (placeholder)
- `core/metrics/segment_analysis.py` references external `kpi_analytics` module not in repo

### File Paths
- Default dataset path: `data/bank-additional-full.csv`
- Session data: `backend/data/sessions/`

### Troubleshooting
- **Spark hostname warnings:** Safe to ignore, or set `SPARK_LOCAL_IP` environment variable
- **CORS issues:** Configure `CORS_ORIGINS` environment variable
- **Memory errors:** Reduce feature generation options or increase Spark driver memory
- **Redis connection:** Falls back to file storage if Redis unavailable
