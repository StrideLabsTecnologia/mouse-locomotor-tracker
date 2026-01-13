# Contributing to Mouse Locomotor Tracker

Thank you for your interest in contributing to MLT! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment.

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Python version
   - OS and version
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages/stack traces

### Suggesting Features

1. Check existing issues/discussions
2. Use the feature request template
3. Describe:
   - Use case
   - Proposed solution
   - Alternatives considered

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Run linters: `pre-commit run --all-files`
6. Commit with conventional commits: `git commit -m 'feat: add feature'`
7. Push: `git push origin feature/your-feature`
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mouse-locomotor-tracker.git
cd mouse-locomotor-tracker

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

- **Python**: Follow PEP 8, enforced by Black and Ruff
- **Imports**: Sorted with isort
- **Types**: Add type hints where possible
- **Docstrings**: Google style

### Example

```python
def calculate_velocity(
    positions: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """
    Calculate instantaneous velocity from position data.

    Args:
        positions: Array of (x, y) coordinates, shape (N, 2).
        timestamps: Array of timestamps in seconds, shape (N,).

    Returns:
        Array of velocities in cm/s, shape (N-1,).

    Raises:
        ValueError: If arrays have mismatched lengths.
    """
    if len(positions) != len(timestamps):
        raise ValueError("Mismatched array lengths")

    dx = np.diff(positions[:, 0])
    dy = np.diff(positions[:, 1])
    dt = np.diff(timestamps)

    return np.sqrt(dx**2 + dy**2) / dt
```

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Code style (no logic change)
- `refactor:` Code refactoring
- `test:` Tests
- `chore:` Maintenance

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=analysis --cov=tracking --cov-report=html

# Specific test
pytest tests/test_velocity.py -v
```

## Documentation

- Update docstrings for any API changes
- Update README.md if adding features
- Add examples for new functionality

## Questions?

Open a discussion or issue. We're happy to help!

---

Thank you for contributing! 🎉
