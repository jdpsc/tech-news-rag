# AI News Bot

Chatbot application orchestrated with [Langchain](https://www.langchain.com) that:
- Cleans and embeds the user question
- Queries [Qdrant](https://qdrant.tech) for the top K (default 1) closest matches
- Uses the registered model from [Comet-ML](https://www.comet.com) to generate a response
- Can be easily deployed as an endpoint

# Usage

## Dependencies

Main dependencies you have to install yourself:
* Python 3.10
* Poetry 1.5.1
* GNU Make 4.3
* Beam CLI 0.2.148

Install other dependencies:
```shell
make install
```

Prepare credentials:
```shell
cp .env.example .env
```

You also need to create a [Beam](https://www.beam.cloud) account and run the following with your API key:
```shell
beam config create
```

## Deploying and Testing the App

Run locally (assuming you have a powerful enough GPU):
```shell
make run
```

Deploy on Beam as an endpoint:
```shell
make deploy_beam
```

Test the deployed endpoint:
```shell
make call_restful_api
```
