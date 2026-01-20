# Spark Tune - ML Feature Discovery Platform

A modern ML feature discovery platform with a React frontend and FastAPI backend.

## Architecture

```
spark_beyond/
├── backend/                    # FastAPI backend
│   ├── main.py                # FastAPI app entry point
│   ├── routes/                # API route handlers
│   │   ├── data.py            # Data loading, quality checks
│   │   ├── features.py        # Feature engineering
│   │   ├── models.py          # Model training
│   │   └── insights.py        # Results and insights
│   ├── services/
│   │   └── spark_service.py   # Spark session management
│   ├── schemas/
│   │   └── api_models.py      # Pydantic request/response models
│   └── core/                  # ML modules
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/        # UI components (shadcn/ui)
│   │   ├── pages/             # Page components
│   │   ├── lib/               # API client
│   │   └── App.tsx
│   └── vite.config.ts
└── data/                       # Dataset files
```

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- UV (recommended) or pip

### 1. Install Backend Dependencies

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Start the Backend

```bash
# From project root
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 4. Start the Frontend

```bash
# From project root
cd frontend
npm run dev
```

The app will be available at `http://localhost:5173`

## Tech Stack

### Backend
- **FastAPI** - Modern async Python web framework
- **PySpark 4.1+** - Distributed data processing
- **XGBoost** - Gradient boosting via Spark integration
- **Pydantic** - Data validation
- **SHAP** - Model interpretability

### Frontend
- **React 18** + TypeScript
- **Vite** - Build tooling
- **shadcn/ui** - Component library
- **TailwindCSS** - Styling
- **React Query** - Data fetching
- **React Router** - Navigation
- **Recharts** - Visualizations

## API Endpoints

### Data Routes (`/api/data`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/state` | Get current pipeline state |
| POST | `/load` | Load dataset from file path |
| POST | `/problem` | Set problem definition |
| GET | `/schema` | Get schema information |
| POST | `/quality-check` | Run data quality checks |
| GET | `/columns` | Get column information |
| POST | `/reset` | Reset pipeline state |

### Feature Routes (`/api/features`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate` | Generate features |
| POST | `/preprocess` | Preprocess features |

### Model Routes (`/api/models`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train` | Train XGBoost model |
| POST | `/train-baselines` | Train baseline models |
| POST | `/automl` | Run AutoML search |

### Insights Routes (`/api/insights`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/feature-importance` | Get feature importance |
| GET | `/lift-support` | Get lift/support analysis |
| GET | `/shap` | Get SHAP analysis |

## Pages

1. **Dashboard** (`/`) - Overview of pipeline progress and quick actions
2. **Data Overview** (`/data`) - Load data, define problem, run quality checks
3. **Feature Engineering** (`/features`) - Generate and preprocess features
4. **Model Training** (`/training`) - Configure and train XGBoost model
5. **Model Comparison** (`/comparison`) - Compare baselines and run AutoML
6. **Insights** (`/insights`) - View feature importance, SHAP, lift/support analysis

## Development

### Backend Development

```bash
# Run with auto-reload
cd backend
uvicorn main:app --reload

# Run tests (if available)
pytest
```

### Frontend Development

```bash
cd frontend

# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Environment Variables

### Backend
- `SPARK_LOCAL_IP` - Override Spark local IP (optional)

### Frontend
- Proxy configuration in `vite.config.ts` points to `http://localhost:8000`

## Default Dataset

The platform includes a sample dataset at `data/bank-additional-full.csv` (bank marketing classification data).

## License

MIT
