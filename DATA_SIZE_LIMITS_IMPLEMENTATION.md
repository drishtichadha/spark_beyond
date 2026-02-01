# Data Size Limits Implementation

## Summary

Implemented comprehensive data size limits at multiple layers to prevent resource exhaustion attacks and ensure fair usage across users. The system now enforces limits on file size, row count, and processing resources.

## Changes Made

### 1. API Request Schema (`backend/schemas/api_models.py`)

**Added to `LoadDataRequest`:**
```python
max_rows: Optional[int] = 1_000_000  # Default: 1 million rows
max_file_size_mb: Optional[int] = 500  # Default: 500MB
```

**Validation:**
- Row count: 1 to 10 million (hard limit)
- File size: 1MB to 2GB (depending on tier)

### 2. Resource Limits Configuration (`backend/config/resource_limits.py`)

**New module with three tiers:**

#### Production Limits (Default)
- Max file size: 500MB
- Default max rows: 1M
- Absolute max rows: 10M
- Max profile rows: 100K
- Max generated features: 1000
- Max training time: 1 hour
- Spark driver memory: 4GB
- Max concurrent sessions: 100

#### Development Limits
- Max file size: 1GB
- Default max rows: 5M
- Max profile rows: 200K
- Max generated features: 2000
- Max training time: 2 hours
- Spark driver memory: 8GB

#### Enterprise Limits
- Max file size: 2GB
- Default max rows: 5M
- Max profile rows: 500K
- Max generated features: 5000
- Max training time: 4 hours
- Spark driver memory: 16GB

**Usage:**
```python
from backend.config import get_resource_limits

limits = get_resource_limits("production")  # or "development", "enterprise"
```

**Environment variable:** Set `RESOURCE_TIER=production|development|enterprise`

### 3. Data Loading Endpoint (`backend/routes/data.py`)

**Enhanced `/api/data/load` endpoint:**

#### Security Layers:
1. **File size check** (before reading)
   ```python
   file_size_mb = os.path.getsize(validated_path) / (1024 * 1024)
   if file_size_mb > max_size_mb:
       raise HTTPException(413, "File too large")
   ```

2. **Row count limit** (during loading)
   ```python
   max_rows = min(request.max_rows, limits.absolute_max_rows)
   info = svc.load_data(validated_path, max_rows=max_rows)
   ```

3. **Tier-based enforcement**
   - User-provided limits cannot exceed tier limits
   - Absolute hard limits are enforced at all tiers

#### Response includes:
- `truncated: boolean` - Indicates if data was limited
- `max_rows: int` - Maximum rows applied

### 4. Spark Service (`backend/services/spark_service.py`)

**Updated `load_data()` method:**
```python
def load_data(self, file_path: str, max_rows: Optional[int] = None):
    df = spark.read.csv(file_path)

    if max_rows:
        actual_count = df.count()
        if actual_count > max_rows:
            df = df.limit(max_rows)

    self._df = df
```

**Features:**
- Counts rows before limiting (for accurate reporting)
- Logs truncation warnings
- Maintains backward compatibility (max_rows is optional)

### 5. Data Profiling (`backend/routes/data.py`)

**Enhanced `/api/data/profile` endpoint:**
```python
max_rows = min(max(max_rows, 1), 100_000)  # Clamp to 1-100K
```

**Enforcement:**
- Minimum: 1 row
- Maximum: 100,000 rows
- Prevents users from requesting unreasonable profiling

### 6. New API Endpoint: Resource Limits

**GET `/api/data/limits`**

Returns current tier limits:
```json
{
  "success": true,
  "data": {
    "max_file_size_mb": 500,
    "default_max_rows": 1000000,
    "absolute_max_rows": 10000000,
    "max_profile_rows": 100000,
    "max_generated_features": 1000,
    "max_training_time_seconds": 3600,
    "max_automl_time_seconds": 600
  }
}
```

**Usage:** Frontend can query limits and display to users before upload.

### 7. Environment Configuration (`.env.example`)

**Added configuration options:**
```bash
# Resource Tier
RESOURCE_TIER=production  # production|development|enterprise

# Override specific limits (optional)
# MAX_FILE_SIZE_MB=500
# MAX_ROWS=1000000
# MAX_TRAINING_TIME=3600
```

## Security Benefits

### 1. Prevents Resource Exhaustion
- ❌ **Before:** Users could upload arbitrarily large files
- ✅ **After:** File size checked before loading (HTTP 413 if too large)

### 2. Prevents Memory Exhaustion
- ❌ **Before:** No row count limits, 4GB shared across ALL users
- ✅ **After:** Row limits enforced per-dataset, configurable by tier

### 3. Fair Resource Allocation
- Concurrent session limits prevent one user from monopolizing resources
- Training time limits prevent runaway jobs
- Feature generation limits prevent exponential feature explosion

### 4. Transparent Limits
- Users can query current limits via API
- Clear error messages when limits are exceeded
- Truncation is clearly indicated in responses

## Testing the Implementation

### Test 1: File Size Limit
```bash
# Create large file (>500MB)
dd if=/dev/zero of=large.csv bs=1M count=600

# Try to load
curl -X POST http://localhost:8000/api/data/load \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/large.csv"}'

# Expected: HTTP 413 - File too large
```

### Test 2: Row Count Limit
```bash
# Try to load with custom limit
curl -X POST http://localhost:8000/api/data/load \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/data.csv",
    "max_rows": 500000
  }'

# Expected: Success, with truncated=true if file has >500K rows
```

### Test 3: Query Limits
```bash
curl http://localhost:8000/api/data/limits

# Expected: JSON with current tier limits
```

### Test 4: Tier Configuration
```bash
# Set development tier
export RESOURCE_TIER=development

# Restart server and check limits
curl http://localhost:8000/api/data/limits

# Expected: Higher limits (1GB files, 5M rows)
```

## Migration Guide

### For Existing Users

**No breaking changes** - all limits have sensible defaults:
- Default max_rows: 1 million (sufficient for most datasets)
- Default file size: 500MB
- Existing code continues to work

### For Frontend Integration

**1. Query limits before upload:**
```typescript
const limits = await fetch('/api/data/limits').then(r => r.json());
console.log(`Max file size: ${limits.data.max_file_size_mb}MB`);
```

**2. Check for truncation:**
```typescript
const response = await loadData(filePath);
if (response.data.truncated) {
  showWarning(`Data truncated to ${response.data.max_rows} rows`);
}
```

**3. Provide clear feedback:**
```typescript
if (fileSize > limits.max_file_size_mb * 1024 * 1024) {
  showError(`File too large. Maximum: ${limits.max_file_size_mb}MB`);
}
```

## Production Deployment Checklist

- [x] Set `RESOURCE_TIER=production` in environment
- [ ] Configure `ALLOWED_DATA_DIRS` for file access control
- [ ] Set `CORS_ORIGINS` to production domains only
- [ ] Monitor Spark driver memory usage (default 4GB)
- [ ] Set up alerts for resource limit violations
- [ ] Document limits for users (API docs, UI tooltips)
- [ ] Consider implementing rate limiting per user session
- [ ] Set up monitoring for concurrent sessions

## Future Enhancements

### Per-User Quotas
```python
@dataclass
class UserQuota:
    user_id: str
    max_file_size_mb: int
    max_rows: int
    max_datasets: int
    max_concurrent_trainings: int
```

### Usage Tracking
- Track total data processed per user
- Track total training time per user
- Implement monthly quotas for fair usage

### Dynamic Scaling
- Auto-adjust limits based on system load
- Queue requests when system is overloaded
- Prioritize paid users during peak times

### Rate Limiting
- Add rate limiting middleware
- Limit requests per session/user
- Prevent API abuse

## Related Security Fixes

This implementation addresses **Issue #10: No Data Size Limits** from the security audit.

**Related fixes:**
- [x] #9: Path traversal protection (`path_validator.py`)
- [x] #10: Data size limits (this implementation)
- [ ] #11: Input validation (separate PR)
- [ ] #12: Rate limiting (future enhancement)

## References

- Resource limits config: `backend/config/resource_limits.py`
- API schema: `backend/schemas/api_models.py`
- Data route: `backend/routes/data.py`
- Spark service: `backend/services/spark_service.py`
- Environment config: `.env.example`
