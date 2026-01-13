"""
Mouse Locomotor Tracker - Test Suite

This package contains comprehensive tests for the mouse locomotor tracking system.

Test Modules:
    - test_velocity: Tests for VelocityAnalyzer class
    - test_coordination: Tests for CircularCoordinationAnalyzer
    - test_gait_cycles: Tests for GaitCycleDetector
    - test_integration: End-to-end integration tests

Coverage Targets:
    - Domain/Business Logic: 90%+
    - Data Processing: 80%+
    - Integration: 70%+

Usage:
    pytest tests/                    # Run all tests
    pytest tests/ -v                 # Verbose output
    pytest tests/ --cov=.            # With coverage
    pytest tests/ -k "velocity"      # Run only velocity tests

Author: Stride Labs
License: MIT
"""

__all__ = [
    "conftest",
    "test_velocity",
    "test_coordination",
    "test_gait_cycles",
    "test_integration",
]
