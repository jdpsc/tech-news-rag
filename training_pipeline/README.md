# Training / Fine-tuning Pipeline


Finetuning pipeline that:
- Loads a pre-trained LLM (in this case Falcon-7B-Instruct)
- Efficently finetunes it using [Hugging Face](https://huggingface.co) libraries
- Logs the model experiments on [Comet-ML](https://www.comet.com), so that the best model can be registered and accessed later
- Easily allows for training in the cloud (in case GPU access is needed)

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

## Runing the training pipeline

Training locally (assuming you have a powerful enough GPU):
```shell
make train_local
```

Training on Beam:
```shell
make train_beam
```

Infer locally (assuming you have a powerful enough GPU):
```shell
make infer_local
```

Infer on Beam:
```shell
make infer_beam
```

After the training, you need to register the model in Comet. You can follow the following [guide](https://www.comet.com/docs/v2/guides/model-registry/using-model-registry/#register-a-model).
