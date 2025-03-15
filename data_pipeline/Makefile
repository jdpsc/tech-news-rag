# Export env variables from .env file if it exists.
ifneq (,$(wildcard .env))
include .env
export
endif

### Code Quality ###

pre-commit:
	pre-commit run --all-files


### Run Local ###

install:
	@echo "Install dependencies"
	poetry install

build:
	@echo "Build docker image"

	docker build -t data_pipeline:latest -f deploy/Dockerfile .

run_docker_local:
	@echo "Run docker image"

	docker run -p 3000:3000 --rm \
		--env-file .env \
		--name data_pipeline \
		data_pipeline:latest


### Deploy AWS ###

push_ecr:
	chmod +x deploy/to_ecr.sh
	bash deploy/to_ecr.sh


deploy_ecs:
	chmod +x deploy/deploy_ecs.sh
	bash deploy/deploy_ecs.sh

deploy:
	@echo "Deploy to AWS"
	make push_ecr
	make deploy_ecs
