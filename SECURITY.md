# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously in Mouse Locomotor Tracker. If you discover a security vulnerability, please follow these steps:

### 1. Do NOT Create a Public Issue

Security vulnerabilities should not be reported through public GitHub issues.

### 2. Report Privately

Send an email to: **security@stridelabs.cl**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### 3. Response Timeline

| Phase | Timeline |
|-------|----------|
| Initial response | Within 48 hours |
| Vulnerability assessment | Within 7 days |
| Fix development | Within 30 days |
| Public disclosure | After fix is released |

### 4. What to Expect

- Acknowledgment of your report within 48 hours
- Regular updates on the progress
- Credit in the security advisory (if desired)
- We will NOT take legal action against researchers who follow responsible disclosure

## Security Measures

### Code Security

- **Static Analysis**: Bandit security linter in CI/CD
- **Dependency Scanning**: Regular audits with `pip-audit`
- **Type Checking**: MyPy for type safety
- **Pre-commit Hooks**: Automated security checks

### Docker Security

- Non-root user execution
- Multi-stage builds (minimal attack surface)
- No secrets in images
- Read-only file systems where possible

### Data Handling

- No personal data collection
- Video files processed locally only
- No network calls during analysis
- Output files contain only tracking data

## Known Security Considerations

### Input Validation

All video inputs are validated for:
- File format compatibility
- Maximum file size (configurable)
- Codec safety checks

### Output Sanitization

- File paths are sanitized
- No arbitrary code execution in outputs
- JSON/CSV exports are properly escaped

## Dependencies

We regularly monitor and update dependencies:

```bash
# Check for known vulnerabilities
pip-audit

# Update dependencies
pip install --upgrade -r requirements.txt
```

### Key Dependencies

| Package | Security Notes |
|---------|----------------|
| OpenCV | Official releases only |
| NumPy | No known vulnerabilities |
| Pandas | No known vulnerabilities |
| h5py | Safe for local files only |

## Best Practices for Users

1. **Keep Updated**: Always use the latest version
2. **Verify Downloads**: Check SHA256 checksums
3. **Docker**: Use official images only
4. **Input Files**: Only process trusted video files
5. **Output Directory**: Use dedicated output directories

## Security Badges

- Pre-commit: Bandit security linter
- CI/CD: Automated security scanning
- Dependencies: Regular audits

---

Thank you for helping keep Mouse Locomotor Tracker secure!
