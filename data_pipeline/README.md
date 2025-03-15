# Data Pipeline

Data pipeline that:
- Ingests AI topics from multiple sources (right now, only papers from [Arxiv](https://arxiv.org)) on a schedule using [Dagster](https://dagster.io)
- Cleans and transforms the documents into embeddings using [Hugging Face](https://huggingface.co) sentence transformers
- Stores the embeddings in the [Qdrant Vector DB](https://qdrant.tech)


# Usage

## Dependencies

Main dependencies you have to install yourself:
* Python 3.10
* Poetry 1.5.1
* GNU Make 4.3
* AWS CLI 2.11.22

Install other dependencies:
```shell
make install
```

Prepare credentials:
```shell
cp .env.example .env
```


## Local Testing

Build local Docker image:
```shell
make build
```

Run local Docker image:
```shell
make run_docker_local
```

## AWS Deployment

Build and push Docker image to ECR:
```shell
make push_ecr
```

Deploy in ECS (Fargate):
```shell
make deploy_ecs
```
