.PHONY: help install install-dev test test-unit test-integration test-cov lint format type-check clean run setup

# Default target
help:
	@echo "DeepSeek-OCR Mac CLI - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Run automated setup script"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format        - Format code with Black"
	@echo "  make lint          - Lint code with Ruff"
	@echo "  make type-check    - Run type checking with MyPy"
	@echo "  make quality       - Run all quality checks"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         - Remove generated files"
	@echo "  make run FILE=     - Run CLI on a file"
	@echo ""

# Setup
setup:
	@bash setup.sh

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# Testing
test:
	pytest -v

test-unit:
	pytest tests/test_unit.py -v

test-integration:
	pytest tests/test_integration.py -v

test-cov:
	pytest --cov=. --cov-report=html --cov-report=term-missing

# Code quality
format:
	black .

lint:
	ruff check .

lint-fix:
	ruff check --fix .

type-check:
	mypy deepseek_ocr_mac.py --ignore-missing-imports

quality: format lint type-check test

# Security
security:
	bandit -r . -f json -o bandit-report.json || true
	@echo "Security report saved to bandit-report.json"

# Utilities
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf outputs/ 2>/dev/null || true
	rm -f bandit-report.json 2>/dev/null || true
	@echo "Cleaned generated files"

run:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make run FILE=path/to/file.pdf"; \
		exit 1; \
	fi
	./deepseek_ocr_mac.py $(FILE)

# Development
dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "Run 'source .venv/bin/activate' to activate virtual environment"

# CI simulation
ci: quality
	@echo "All CI checks passed!"
