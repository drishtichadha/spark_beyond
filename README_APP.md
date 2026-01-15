# Spark Beyond - ML Feature Discovery Platform

A professional Streamlit web application for automated feature discovery, engineering, and machine learning model training using PySpark and XGBoost.

## Features

- **📊 Data Overview**: Load datasets, validate schemas, and explore data statistics
- **🔧 Feature Engineering**: Automated feature generation with configurable options
  - Numerical transformations (log, sqrt, square, cube)
  - Feature interactions (multiplication, division, addition, subtraction)
  - Binning
  - Datetime feature extraction
- **🎯 Model Training**: Train XGBoost models with configurable hyperparameters
- **📈 Results & Insights**:
  - Feature importance analysis
  - Probability impact visualization
  - Performance metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
  - Export results to CSV

## Installation

### Option 1: Using UV (Recommended)

```bash
# Install dependencies
uv sync

# Run the Streamlit app
uv run streamlit run app.py
```

### Option 2: Using pip

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Run the Streamlit app
streamlit run app.py
```

## Quick Start

1. **Start the application**:
   ```bash
   uv run streamlit run app.py
   ```

2. **Navigate through the workflow**:
   - **Data Overview**: Load the default bank dataset or upload your own CSV
   - **Feature Engineering**: Configure and generate automated features
   - **Model Training**: Train an XGBoost model with your preferred settings
   - **Results & Insights**: Explore feature importance and model performance

## Default Dataset

The application comes with a default dataset: `data/bank-additional-full.csv`

This is a bank marketing dataset with:
- **Target**: `y` (whether client subscribed to term deposit)
- **Type**: Classification
- **Features**: 20 input features (demographic, campaign-related, economic indicators)

## Application Structure

```
├── app.py                          # Main Streamlit application
├── core/
│   ├── discovery.py               # Schema validation and problem definition
│   ├── features/
│   │   ├── auto_feature_generator.py  # Automated feature generation
│   │   ├── process.py             # Feature preprocessing
│   │   └── feature_selector.py    # XGBoost model and feature selection
│   └── metrics/                   # Model evaluation metrics
├── data/
│   └── bank-additional-full.csv   # Default dataset
└── pyproject.toml                 # Dependencies
```

## Usage Guide

### 1. Data Overview Page

- Choose between default dataset or upload custom CSV
- Define your ML problem:
  - Select target column
  - Choose problem type (classification/regression)
  - Specify desired result (for classification)
- Validate schema and explore data statistics

### 2. Feature Engineering Page

Configure feature generation options:
- ✅ **Numerical Transformations**: Log, square root, square, cube
- ✅ **Feature Interactions**: Cross-feature operations
- ✅ **Binning**: Discretize continuous features
- ✅ **Datetime Features**: Extract time components

The system will automatically generate and preprocess features.

### 3. Model Training Page

Configure training parameters:
- **Training Split**: Percentage of data for training (default: 80%)
- **Max Depth**: Maximum tree depth (default: 4)
- **Learning Rate**: Step size shrinkage (default: 0.1)

Train the model and view real-time training progress.

### 4. Results & Insights Page

Explore model results:
- **Feature Importance**: Top contributing features
- **Probability Impact**: How feature thresholds affect predictions
- **Performance Metrics**: Train/test comparison
- **Export Options**: Download results as CSV

## Configuration

### Spark Configuration

The app initializes Spark with default settings:
```python
SparkSession.builder \
    .master("local[*]") \
    .appName("Spark Beyond ML App") \
    .getOrCreate()
```

For custom configurations, modify the `init_spark()` function in `app.py`.

### Model Hyperparameters

Default XGBoost settings:
- `num_round`: 100
- `max_depth`: 4 (configurable in UI)
- `learning_rate`: 0.1 (configurable in UI)
- `eval_metric`: logloss (classification) / squarederror (regression)

## Troubleshooting

### Common Issues

1. **Spark warnings about hostname**:
   - These are warnings and won't affect functionality
   - Set `SPARK_LOCAL_IP` environment variable if needed

2. **Memory issues with large datasets**:
   - Reduce feature generation options
   - Increase Spark memory: `.config("spark.driver.memory", "4g")`

3. **Port already in use**:
   ```bash
   streamlit run app.py --server.port 8502
   ```

## Development

To contribute or modify the application:

1. The main app logic is in `app.py`
2. Core ML functionality is in `core/` modules
3. Use session state to maintain data between page navigations
4. Follow the existing page structure for new features

## Client Presentation Tips

- Start with the Data Overview to show data quality
- Demonstrate automated feature engineering capabilities
- Highlight the model performance metrics
- Use the feature importance charts to explain model behavior
- Export results for inclusion in reports

## Support

For issues or questions, please refer to the project documentation or contact the development team.

## License

[Add your license information here]
