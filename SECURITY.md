# Security Documentation

This document outlines the security measures implemented in Spark Tune and best practices for deployment.

## Table of Contents

- [Security Vulnerabilities Fixed](#security-vulnerabilities-fixed)
- [Security Features](#security-features)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Security Checklist](#security-checklist)

## Security Vulnerabilities Fixed

### 1. Path Traversal Vulnerability (CRITICAL - Fixed)

**Issue:** User-controlled file paths in the `/api/data/load` endpoint could allow attackers to access arbitrary files on the server.

**Attack Example:**
```bash
# Could read sensitive files
curl -X POST http://api/data/load \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/etc/passwd"}'

# Could access files outside intended directory
curl -X POST http://api/data/load \
  -d '{"file_path": "../../../sensitive_data.csv"}'
```

**Fix Implemented:**
- Created `PathValidator` class ([backend/core/utils/path_validator.py](backend/core/utils/path_validator.py))
- All file paths are validated against allowed directories before access
- Symlinks are resolved to prevent bypass attempts
- Returns 403 Forbidden for unauthorized path access
- Returns 404 Not Found for non-existent files

**Usage:**
```python
from backend.core.utils.path_validator import validate_file_path

# Validate before loading
validated_path = validate_file_path(user_provided_path)
```

### 2. CORS Configuration Too Permissive (HIGH - Fixed)

**Issue:** CORS middleware allowed all HTTP methods and headers using wildcards (`*`), which could enable unauthorized cross-origin requests.

**Previous Configuration:**
```python
allow_methods=["*"],  # Allows ALL HTTP methods
allow_headers=["*"],  # Allows ALL headers
```

**Fix Implemented:**
- Explicit allow-list of HTTP methods: `GET`, `POST`, `PUT`, `DELETE`
- Explicit allow-list of headers: `Content-Type`, `Authorization`, `X-Session-ID`
- CORS origins now configurable via `CORS_ORIGINS` environment variable
- Added `max_age` for preflight request caching
- Hardcoded origins only used as fallback for local development

**Configuration:**
```bash
# In production, set environment variable
export CORS_ORIGINS="https://app.sparktune.com,https://www.sparktune.com"
```

## Security Features

### Path Validation System

The path validation system provides multiple layers of protection:

1. **Whitelist-based Access Control**
   - Only files within configured `ALLOWED_DATA_DIRS` can be accessed
   - Default: `./data` and `./backend/data`

2. **Path Canonicalization**
   - Resolves `..`, `.`, and symlinks to prevent traversal
   - Converts relative paths to absolute paths

3. **File Type Verification**
   - Ensures paths point to regular files, not directories or special files
   - Validates file existence before access

4. **Defense in Depth**
   - Validation at route handler level
   - Additional logging in service layer
   - Multiple validation checkpoints

**Configuration:**
```bash
# Set allowed directories (comma-separated absolute paths)
export ALLOWED_DATA_DIRS="/app/data/uploads,/app/data/public"
```

### CORS Security

CORS is configured to prevent unauthorized cross-origin access:

- **Origins:** Environment-based whitelist (no wildcards)
- **Methods:** Explicit allow-list (GET, POST, PUT, DELETE)
- **Headers:** Explicit allow-list (Content-Type, Authorization, X-Session-ID)
- **Credentials:** Enabled for authenticated requests
- **Preflight Caching:** 1 hour to reduce overhead

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

**Critical Security Settings:**

```bash
# CORS origins (REQUIRED in production)
CORS_ORIGINS=https://yourdomain.com

# Allowed data directories (REQUIRED in production)
ALLOWED_DATA_DIRS=/app/data/uploads,/app/data/public

# Redis password (if using Redis)
REDIS_PASSWORD=your_secure_password

# Disable debug mode in production
DEBUG=false
```

### Production Deployment

**DO NOT:**
- Use wildcard (`*`) in CORS configuration
- Allow access to system directories in `ALLOWED_DATA_DIRS`
- Run with `DEBUG=true` in production
- Use default passwords or empty `REDIS_PASSWORD`
- Expose internal paths in error messages

**DO:**
- Set explicit CORS origins for your frontend domain(s)
- Restrict `ALLOWED_DATA_DIRS` to dedicated upload directories
- Use environment variables for all sensitive configuration
- Enable HTTPS/TLS for all production traffic
- Implement rate limiting at the reverse proxy level
- Monitor logs for path validation failures (potential attacks)
- Keep dependencies updated for security patches

## Best Practices

### File Upload Security

If implementing file uploads:

```python
# 1. Validate file extensions
ALLOWED_EXTENSIONS = {'.csv', '.parquet'}

# 2. Generate secure filenames (prevent directory traversal)
import uuid
secure_filename = f"{uuid.uuid4()}.csv"

# 3. Store in dedicated upload directory
upload_dir = validate_directory_path(UPLOAD_DIR)
file_path = upload_dir / secure_filename

# 4. Scan for malware if handling user uploads
# (integrate with antivirus service)

# 5. Limit file sizes
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
```

### API Security

1. **Rate Limiting**
   ```python
   # Use slowapi or similar
   from slowapi import Limiter, _rate_limit_exceeded_handler

   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   ```

2. **Authentication**
   ```python
   # Implement JWT or OAuth2
   from fastapi.security import OAuth2PasswordBearer

   oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
   ```

3. **Input Validation**
   - Use Pydantic models for all API inputs
   - Validate ranges, formats, and types
   - Sanitize user inputs before processing

4. **Logging and Monitoring**
   - Log all path validation failures
   - Monitor for repeated failed attempts
   - Alert on suspicious patterns

### Data Security

1. **Sensitive Data**
   - Never commit `.env` files to version control
   - Use secret management services (AWS Secrets Manager, HashiCorp Vault)
   - Encrypt data at rest and in transit

2. **Session Security**
   - Use secure session IDs (UUID v4)
   - Implement session expiration
   - Clear sensitive data from memory after use

3. **Error Handling**
   - Don't expose internal paths in error messages
   - Use generic error messages for authentication failures
   - Log detailed errors server-side only

## Security Checklist

### Pre-Production Checklist

- [ ] Set `CORS_ORIGINS` to production domain(s)
- [ ] Configure `ALLOWED_DATA_DIRS` for restricted access
- [ ] Set `DEBUG=false`
- [ ] Enable HTTPS/TLS
- [ ] Set strong `REDIS_PASSWORD`
- [ ] Review and restrict file permissions on data directories
- [ ] Implement rate limiting
- [ ] Add authentication/authorization
- [ ] Set up security monitoring and alerting
- [ ] Review all environment variables
- [ ] Run security scanning tools (bandit, safety)
- [ ] Perform penetration testing
- [ ] Set up automated dependency updates (Dependabot)

### Monitoring

Monitor these indicators for potential security issues:

1. **Path Validation Failures**
   - Log entries: `"Path traversal attempt blocked"`
   - High frequency may indicate attack

2. **CORS Violations**
   - Browser console errors about CORS
   - May indicate misconfiguration or attack

3. **Session Anomalies**
   - Rapid session creation from single IP
   - Unusual session access patterns

4. **File Access Patterns**
   - Attempts to access common sensitive files (`/etc/passwd`, `config.json`)
   - Rapid file scanning attempts

### Incident Response

If you suspect a security breach:

1. **Immediate Actions**
   - Review logs for suspicious activity
   - Check file access logs
   - Verify no unauthorized data access occurred
   - Rotate credentials if compromised

2. **Investigation**
   - Identify attack vector
   - Assess scope of potential data exposure
   - Document timeline of events

3. **Remediation**
   - Patch vulnerabilities
   - Update security configurations
   - Notify affected users if required

## Reporting Security Issues

If you discover a security vulnerability:

1. **DO NOT** open a public GitHub issue
2. Email security concerns to: [security@yourdomain.com]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [FastAPI Security Guide](https://fastapi.tiangolo.com/tutorial/security/)
- [PySpark Security](https://spark.apache.org/docs/latest/security.html)

## Changelog

### 2026-01-21
- Added path validation to prevent path traversal attacks
- Fixed CORS configuration (removed wildcards)
- Created environment-based security configuration
- Added comprehensive security documentation
