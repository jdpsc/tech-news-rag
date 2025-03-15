.PHONY: help install export_requirements train_local train_beam infer_local infer_beam

### Code Quality ###

pre-commit:
	pre-commit run --all-files


### Local ###

install:
	@echo "Install dependencies"
	poetry lock
	poetry install


export_requirements:
	@echo "Exporting requirements..."

	poetry lock
	if [ -f requirements.txt ]; then rm requirements.txt; fi
	poetry export --without-hashes --format=requirements.txt | sed -E 's/; (python_version|platform_system|platform_machine).+"//g' > requirements.txt


# === Training ===

train_local: install
	@echo "Running training pipeline locally using the config..."

	poetry run python -m tools.train_run --config_file configs/training_config.yaml --output_dir ./output --dataset_dir ./dataset --model_cache_dir ./model_cache

train_beam: export_requirements
	@echo "Running training pipeline on Beam using the config..."

	BEAM_IGNORE_IMPORTS_OFF=true poetry run python -m tools.train_beam


# === Inference ===

infer_local: install
	@echo "Running inference pipeline locally using the config..."

	poetry run python -m tools.infer_run --config_file configs/inference_config.yaml --output_dir ./output-inference --dataset_dir ./dataset --model_cache_dir ./model_cache

infer_beam: export_requirements
	@echo "Running inference pipeline on Beam using the config..."

	BEAM_IGNORE_IMPORTS_OFF=true poetry run python -m tools.inference_beam
