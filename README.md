# RecGPT: Generative  Pre-training  for  Text-based  Recommendation

We present the first domain-adapted and fully-trained large language model,  RecGPT-7B, and its instruction-following variant, RecGPT-7B-Instruct, for  text-based  recommendation. Experimental results on rating prediction and  sequential recommendation tasks show that our model, RecGPT-7B-Instruct,  outperforms previous strong baselines. The general architecture and experimental results of RecGPT can be found in our [paper](http://arxiv.org/abs/2405.12715):

```
@inproceedings{RecGPT,
title     = {{RecGPT: Generative Pre-training for Text-based Recommendation}},
author    = {Hoang Ngo and Dat Quoc Nguyen},
booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
year      = {2024}
}
```

We publicly release the RecGPT models along with their pre-training and fine-tuning datasets. Please cite our paper whenever RecGPT or the datasets are used to help produce published results or are incorporated into other software.

### Model and dataset download

Model/Dataset | Type | Note
---|--|---
[`vinai/RecGPT-7B`](https://huggingface.co/vinai/RecGPT-7B) | Base pre-trained model | 
[`vinai/RecGPT-7B-Instruct`](https://huggingface.co/vinai/RecGPT-7B-Instruct) | Instruction following model | `PROMPT_TEMPLATE ="### Instruction:\n{instruction}\n\n### Response:"` See our paper for details of the instruction.
[`vinai/RecGPT-datasets`](https://huggingface.co/datasets/vinai/RecGPT-datasets)| Dataset | Pre-training and fine-tuning datasets

## Fine-tuning the model <a name="finetuning"></a>

RecGPT is pre-trained and fine-tuned using the [llm-foundry](https://github.com/mosaicml/llm-foundry) library. See [llm-foundry docs](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#llmfinetuning) for details. To fully fine-tune RecGPT, users can find an example of model finetuning YAML configuration in [`fine-tuning.yaml`](https://github.com/VinAIResearch/RecGPT/blob/main/fine-tuning.yaml). Users can also find the `sample_instruction_following_dataset` folder as an example of an instruction-following dataset.

- To install `llm-foundry`, see Section "Installation" in [https://github.com/mosaicml/llm-foundry](https://github.com/mosaicml/llm-foundry).
- Run: `cd llm-foundry/scripts/train/` and then `composer --world_size <number_of_GPUs> train.py <path_to_yaml_configuration_file>` (e.g. `composer --world_size 1 train.py fine-tuning.yaml`). 

Other fine-tuning options may include the use of [transformers](https://github.com/huggingface/transformers)'s Trainer (e.g. see [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) as an example), [lit-gpt](https://github.com/Lightning-AI/litgpt) or [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## License

```
Copyright (c) 2024 VinAI

Licensed under the Creative Commons Attribution Non Commercial 4.0 International.
You may obtain a copy of the License at

    https://creativecommons.org/licenses/by-nc/4.0/
```
