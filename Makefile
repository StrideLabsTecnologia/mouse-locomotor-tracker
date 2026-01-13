# =============================================================================
# Mouse Locomotor Tracker - Makefile
# =============================================================================
# Common commands for development, testing, and deployment
# =============================================================================

.PHONY: help install install-dev install-all clean test lint format typecheck
.PHONY: build docker-build docker-run docker-test docs serve-docs
.PHONY: release pre-commit security coverage

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
DOCKER := docker
DOCKER_COMPOSE := docker-compose
DOCKER_IMAGE := mlt
DOCKER_TAG := latest

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo ""
	@echo "  Mouse Locomotor Tracker - Development Commands"
	@echo "  ==============================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Installation
# =============================================================================

install: ## Install core dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install

install-all: ## Install all dependencies including optional
	$(PIP) install -e ".[all]"
	pre-commit install

install-cli: ## Install CLI dependencies
	$(PIP) install -e ".[cli]"

# =============================================================================
# Cleaning
# =============================================================================

clean: ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

clean-all: clean ## Deep clean including output files
	rm -rf output/*.mp4
	rm -rf output/*.csv
	rm -rf output/*.json
	rm -rf output/*.png
	rm -rf output/*.jpg

# =============================================================================
# Testing
# =============================================================================

test: ## Run tests
	$(PYTEST) tests/ -v

test-fast: ## Run tests without coverage (faster)
	$(PYTEST) tests/ -v --no-cov

test-verbose: ## Run tests with detailed output
	$(PYTEST) tests/ -v -s --tb=long

coverage: ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=analysis --cov=tracking --cov=visualization \
		--cov-report=html --cov-report=term-missing
	@echo ""
	@echo "Coverage report: htmlcov/index.html"

coverage-xml: ## Generate XML coverage report (for CI)
	$(PYTEST) tests/ --cov=analysis --cov=tracking --cov-report=xml

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linters (ruff)
	ruff check .
	ruff format --check .

lint-fix: ## Fix linting issues automatically
	ruff check --fix .
	ruff format .

format: ## Format code with black and isort
	black .
	isort .

typecheck: ## Run type checking with mypy
	mypy analysis tracking visualization export --ignore-missing-imports

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

security: ## Run security checks with bandit
	bandit -r analysis tracking visualization export -ll

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker image
	$(DOCKER) build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-build-dev: ## Build Docker development image
	$(DOCKER_COMPOSE) build mlt-dev

docker-run: ## Run Docker container (interactive)
	$(DOCKER) run -it -v $(PWD)/input:/app/input -v $(PWD)/output:/app/output \
		$(DOCKER_IMAGE):$(DOCKER_TAG) /bin/bash

docker-test: ## Run tests in Docker
	$(DOCKER_COMPOSE) run --rm mlt-test

docker-jupyter: ## Start Jupyter notebook server
	$(DOCKER_COMPOSE) up mlt-jupyter

docker-clean: ## Remove Docker images and containers
	$(DOCKER_COMPOSE) down --rmi local --volumes

# =============================================================================
# Documentation
# =============================================================================

docs: ## Build documentation
	cd docs && make html

serve-docs: ## Serve documentation locally
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# =============================================================================
# CLI
# =============================================================================

cli-help: ## Show CLI help
	$(PYTHON) cli.py --help

cli-process: ## Run CLI process command (example)
	@echo "Usage: make cli-process VIDEO=path/to/video.mp4"
	@if [ -n "$(VIDEO)" ]; then \
		$(PYTHON) cli.py process $(VIDEO) --csv --json; \
	fi

cli-info: ## Show CLI info command
	$(PYTHON) cli.py info

# =============================================================================
# Release
# =============================================================================

version: ## Show current version
	@$(PYTHON) -c "import toml; print(toml.load('pyproject.toml')['project']['version'])" 2>/dev/null || \
		grep -m1 'version' pyproject.toml | cut -d'"' -f2

build: clean ## Build distribution packages
	$(PYTHON) -m build

release-check: ## Check release readiness
	@echo "Checking release readiness..."
	@$(MAKE) lint
	@$(MAKE) typecheck
	@$(MAKE) test
	@$(MAKE) security
	@echo ""
	@echo "All checks passed!"

# =============================================================================
# Development Workflow
# =============================================================================

dev: install-dev ## Setup development environment
	@echo ""
	@echo "Development environment ready!"
	@echo "Run 'make help' to see available commands."

check: lint typecheck test ## Run all checks (lint, typecheck, test)

ci: ## Run CI pipeline locally
	@$(MAKE) lint
	@$(MAKE) typecheck
	@$(MAKE) test
	@$(MAKE) security

# =============================================================================
# Shortcuts
# =============================================================================

t: test ## Alias for test
c: clean ## Alias for clean
l: lint ## Alias for lint
f: format ## Alias for format
