.PHONY: help install install-dev install-test test test-cov test-modules test-training test-kernels lint

help: ## Show available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install molfun in editable mode
	pip install -e .

install-dev: ## Install molfun with dev dependencies
	pip install -e ".[dev]"

install-test: ## Install test dependencies
	pip install -r requirements-test.txt

test: ## Run full test suite
	@echo "Running full test suite..."
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	pytest tests/ --cov=molfun --cov-report=html --cov-report=term

test-modules: ## Run modular architecture tests (attention, blocks, builder, swapper)
	@echo "Running module tests..."
	pytest tests/modules/ -v

test-training: ## Run training strategy tests
	@echo "Running training tests..."
	pytest tests/training/ tests/models/ -v

test-kernels: ## Run GPU kernel tests
	@echo "Running kernel tests..."
	pytest tests/kernels/ -v

lint: ## Run linter checks
	@echo "Running linters..."
	python -m py_compile molfun/__init__.py
	python -m pytest tests/ --collect-only -q 2>/dev/null | tail -1
