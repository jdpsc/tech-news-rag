# Export env variables from .env file if it exists.
ifneq (,$(wildcard .env))
include .env
export
endif


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
	poetry export --without-hashes --format=requirements.txt | sed -E '/;.*platform_system == "Windows"/d' | sed -E 's/; (python_version|platform_system|platform_machine).+"//g' > requirements.txt

# === Run ===

run:
	@echo "Running AI news bot..."

	poetry run python -m tools.bot \
		--question "Can you give me examples of RAG retrieval and post-retrieval optimization techniques?"


run_ui:
	@echo "Running  AI news bot UI..."

	poetry run python -m tools.ui


# === Beam ===

deploy_beam: export_requirements
	@echo "Deploying AI news bot on Beam..."

	BEAM_IGNORE_IMPORTS_OFF=true beam deploy tools/bot.py:run_beam --name ai-news-bot


serve_beam: export_requirements
	@echo "Serving AI news bot on Beam..."

	BEAM_IGNORE_IMPORTS_OFF=true beam serve tools/bot.py:run_beam


call_restful_api:
	curl -X POST  \
	--compressed 'https://${BEAM_DEPLOYMENT_ID}.app.beam.cloud' \
	-H 'Authorization: Bearer ${BEAM_AUTH_TOKEN}' \
	-H 'Connection: keep-alive' \
	-H 'Content-Type: application/json' \
	-d '{"question": "Can you give me examples of RAG retrieval and post-retrieval optimization techniques?"}'
