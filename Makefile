.PHONY: install dev data train predict test lint format clean docker_build docker_run docs

# Note: 'uv' is a faster alternative to pip. Install with: pip install uv
# Then replace 'pip install' with 'uv pip install' in the commands below.

install:
	pip install -U pip
	pip install -r requirements.txt
	pip install -e .

dev: install
	pip install -r requirements_dev.txt
	pre-commit install

data:
	python -m food_on_the_fly.data.make_dataset

train:
	python -m food_on_the_fly.train_model

predict:
	python -m food_on_the_fly.predict_model

test:
	pytest tests/

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name dist -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name build -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

docker_build:
	docker build -t food_on_the_fly -f dockerfiles/Dockerfile .

docker_run_data:                                                                                                        
	docker run --rm -v $$(PWD)/data:/app/data -v ~/.config/gcloud:/root/.config/gcloud:ro -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json -e GOOGLE_CLOUD_PROJECT=project-9aed1f8e-f40e-4a15-858 food_on_the_fly:latest python -m food_on_the_fly.data.make_dataset

docker_run_train:                                                                                                       
	docker run --rm -v $$(PWD)/data:/app/data -v $$(PWD)/models:/app/models -v $$(PWD)/mlruns:/app/mlruns -v ~/.config/gcloud:/root/.config/gcloud:ro -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json -e GOOGLE_CLOUD_PROJECT=project-9aed1f8e-f40e-4a15-858 food_on_the_fly:latest python -m food_on_the_fly.train_model

docker_shell:                                                                                                                                    
	docker run --rm -it -v $$(PWD)/data:/app/data -v $$(PWD)/models:/app/models -v ~/.config/gcloud:/root/.config/gcloud:ro -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json food_on_the_fly:latest /bin/bash

docs:
	mkdocs serve
