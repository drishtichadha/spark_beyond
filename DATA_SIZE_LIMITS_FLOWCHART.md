# Data Size Limits - Request Flow

## Data Loading Flow (POST /api/data/load)

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Request                          │
│  POST /api/data/load                                        │
│  {                                                           │
│    "file_path": "/path/to/data.csv",                       │
│    "max_rows": 500000,         // Optional, default: 1M    │
│    "max_file_size_mb": 300     // Optional, default: 500MB │
│  }                                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 1: Path Validation                        │
│  ✓ Check path is within ALLOWED_DATA_DIRS                  │
│  ✓ Prevent path traversal attacks (../)                    │
│  ✗ Reject if path is outside allowed directories           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Step 2: File Existence Check                      │
│  ✓ Verify file exists at validated path                    │
│  ✗ Return 404 if file not found                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Step 3: File Size Check (NEW)                     │
│  1. Get file size: os.path.getsize(path)                   │
│  2. Convert to MB: size_mb = bytes / (1024 * 1024)         │
│  3. Get tier limit: limits.max_file_size_mb                │
│  4. Enforce minimum:                                         │
│     max_size = min(request.max_file_size_mb, tier_limit)   │
│  5. Validate:                                                │
│     if size_mb > max_size:                                  │
│       return HTTP 413 "File too large"                      │
│                                                              │
│  Limits by tier:                                             │
│    Production:   500MB                                       │
│    Development:  1000MB                                      │
│    Enterprise:   2000MB                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Step 4: Row Count Limit (NEW)                     │
│  1. Get tier limit: limits.default_max_rows                 │
│  2. Enforce minimum:                                         │
│     max_rows = min(request.max_rows, tier_limit)           │
│  3. Enforce absolute hard limit:                             │
│     max_rows = min(max_rows, 10_000_000)                   │
│                                                              │
│  Limits by tier:                                             │
│    Production:   1,000,000 rows                             │
│    Development:  5,000,000 rows                             │
│    Enterprise:   5,000,000 rows                             │
│    Absolute max: 10,000,000 rows                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         Step 5: Load Data with Spark (NEW)                  │
│  1. Read CSV with Spark:                                     │
│     df = spark.read.csv(file_path)                          │
│                                                              │
│  2. Count actual rows:                                       │
│     actual_count = df.count()                               │
│                                                              │
│  3. Apply limit if needed:                                   │
│     if actual_count > max_rows:                             │
│       df = df.limit(max_rows)                               │
│       truncated = True                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Step 6: Return Response                    │
│  Success (200 OK):                                           │
│  {                                                           │
│    "success": true,                                          │
│    "message": "Dataset loaded: 1,000,000 rows x 50 cols",  │
│    "data": {                                                 │
│      "rows": 1000000,                                        │
│      "columns": 50,                                          │
│      "truncated": true,       // NEW                        │
│      "max_rows": 1000000      // NEW                        │
│    }                                                         │
│  }                                                           │
│                                                              │
│  Error responses:                                            │
│  - 403: Path validation failed (security)                   │
│  - 404: File not found                                       │
│  - 413: File too large (NEW)                                │
│  - 500: Internal error                                       │
└─────────────────────────────────────────────────────────────┘
```

## Resource Limits Query Flow (GET /api/data/limits)

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Request                          │
│  GET /api/data/limits                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Get Current Tier Configuration                   │
│  1. Read RESOURCE_TIER environment variable                 │
│     - production (default)                                   │
│     - development                                            │
│     - enterprise                                             │
│                                                              │
│  2. Return corresponding limits                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Return Limits                            │
│  {                                                           │
│    "success": true,                                          │
│    "data": {                                                 │
│      "max_file_size_mb": 500,                               │
│      "default_max_rows": 1000000,                           │
│      "absolute_max_rows": 10000000,                         │
│      "max_profile_rows": 100000,                            │
│      "max_generated_features": 1000,                        │
│      "max_training_time_seconds": 3600,                     │
│      "max_automl_time_seconds": 600                         │
│    }                                                         │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

## Data Profiling Flow (POST /api/data/profile)

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Request                          │
│  POST /api/data/profile?max_rows=50000                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Enforce Profiling Row Limit (NEW)                 │
│  max_rows = min(max(max_rows, 1), 100_000)                 │
│                                                              │
│  Clamps to range: [1, 100,000]                              │
│  - Minimum: 1 row                                            │
│  - Maximum: 100,000 rows                                     │
│  - Prevents excessive memory usage                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Run Data Profiling                          │
│  Sample dataset if needed and generate profile              │
└─────────────────────────────────────────────────────────────┘
```

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Request Validation Layer                     │ │
│  │  - Pydantic schemas with Field validators             │ │
│  │  - LoadDataRequest(max_rows, max_file_size_mb)        │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                       │                                      │
│                       ▼                                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Data Route Layer                             │ │
│  │  - Path validation (security)                          │ │
│  │  - File size check (resource protection)              │ │
│  │  - Row limit enforcement (memory protection)          │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                       │                                      │
│                       ▼                                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Resource Limits Config                       │ │
│  │  - Three tiers: production/dev/enterprise              │ │
│  │  - Configurable via RESOURCE_TIER env var              │ │
│  │  - Global DEFAULT_LIMITS instance                      │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                       │                                      │
│                       ▼                                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Spark Service Layer                          │ │
│  │  - load_data(file_path, max_rows)                      │ │
│  │  - Enforces row limit via df.limit()                   │ │
│  │  - Returns truncation status                            │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Limit Enforcement Matrix

| Operation | Check Point | Limit Type | Error Code | Message |
|-----------|-------------|------------|------------|---------|
| File Upload | Before read | File size | 413 | "File too large: X MB exceeds max Y MB" |
| Data Load | During load | Row count | 200* | Returns with truncated=true |
| Data Profile | API param | Row count | 200 | Clamped to 1-100K silently |
| Feature Gen | Runtime | Feature count | 400 | "Too many features generated" |
| Model Train | Runtime | Training time | 408 | "Training timeout exceeded" |

*Note: Row truncation returns 200 OK but sets `truncated: true` in response

## Environment Configuration

```bash
# .env file
RESOURCE_TIER=production

# Options:
# - production: Default, conservative limits (500MB, 1M rows)
# - development: Relaxed for testing (1GB, 5M rows)
# - enterprise: Higher limits for paid users (2GB, 5M rows)

# Per-environment Spark config
SPARK_DRIVER_MEMORY=4g  # Adjust based on tier
```

## Frontend Integration Example

```typescript
// Check limits before upload
async function uploadDataset(file: File) {
  // 1. Get current limits
  const limits = await fetch('/api/data/limits')
    .then(r => r.json());

  // 2. Validate file size
  const fileSizeMB = file.size / (1024 * 1024);
  if (fileSizeMB > limits.data.max_file_size_mb) {
    throw new Error(
      `File too large: ${fileSizeMB.toFixed(1)}MB. ` +
      `Maximum: ${limits.data.max_file_size_mb}MB`
    );
  }

  // 3. Upload with optional limits
  const response = await fetch('/api/data/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      file_path: file.path,
      max_rows: 500000,  // Optional: custom limit
    })
  });

  // 4. Check for truncation
  const result = await response.json();
  if (result.data.truncated) {
    showWarning(
      `Dataset truncated to ${result.data.max_rows.toLocaleString()} rows`
    );
  }
}
```

## Security Benefits Summary

| Threat | Mitigation | Implementation |
|--------|-----------|----------------|
| Large file upload DoS | File size check before read | `os.path.getsize()` before Spark load |
| Memory exhaustion | Row count limits | `df.limit(max_rows)` |
| Shared resource monopoly | Per-tier limits | ResourceLimits class with 3 tiers |
| Feature explosion | Generated feature limits | Max 1000 features (configurable) |
| Runaway training | Training time limits | Max 1 hour (configurable) |
| Concurrent overload | Session limits | Max 100 concurrent sessions |

## Monitoring Recommendations

```python
# Add to monitoring/alerting
- Alert: File size > 80% of limit (frequent)
- Alert: Row count truncation > 50% of uploads
- Alert: Concurrent sessions > 80% of max
- Metric: Average file size per tier
- Metric: Average row count per tier
- Metric: Truncation rate per endpoint
```
