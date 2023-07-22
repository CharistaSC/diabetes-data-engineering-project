---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: diabetic-retinopathy-224-procnorm-vit
  results: []
datasets:
- martinezomg/diabetic-retinopathy
pipeline_tag: image-classification
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# diabetic-retinopathy-224-procnorm-vit

This model is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on the [diabetic retinopathy](https://huggingface.co/datasets/martinezomg/diabetic-retinopathy) dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7578
- Accuracy: 0.7431

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 4e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.8619        | 1.0   | 50   | 0.8907          | 0.7143   |
| 0.7831        | 2.0   | 100  | 0.7858          | 0.7393   |
| 0.6906        | 3.0   | 150  | 0.7412          | 0.7531   |
| 0.5934        | 4.0   | 200  | 0.7528          | 0.7393   |
| 0.5276        | 5.0   | 250  | 0.7578          | 0.7431   |


### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0
- Datasets 2.12.0
- Tokenizers 0.13.3