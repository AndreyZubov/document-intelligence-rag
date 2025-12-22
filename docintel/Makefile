.PHONY: help install dev-install test lint format clean docker-build docker-up docker-down

help:
	@echo "DocIntel - Available Commands"
	@echo "=============================="
	@echo "install        Install production dependencies"
	@echo "dev-install    Install development dependencies"
	@echo "test           Run tests"
	@echo "test-cov       Run tests with coverage"
	@echo "lint           Run linters (ruff, mypy)"
	@echo "format         Format code (black, ruff)"
	@echo "clean          Remove build artifacts"
	@echo "docker-build   Build Docker image"
	@echo "docker-up      Start services with docker-compose"
	@echo "docker-down    Stop services"
	@echo "docker-logs    View docker logs"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=docintel --cov-report=html --cov-report=term

lint:
	ruff check src/
	mypy src/docintel

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/ .mypy_cache/

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-restart: docker-down docker-up

serve:
	docintel serve

qdrant:
	docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
