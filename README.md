# How Large Language Models Are Transforming Machine Paraphrase Generation

[![arXiv](https://img.shields.io/badge/arXiv-2210.03568-b31b1b.svg)](https://arxiv.org/abs/2210.03568)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗-Datasets-ffce1c.svg)](https://huggingface.co/datasets/jpwahle/autoregressive-paraphrase-dataset)

## Quick Start

### Install

```bash
poetry install
```

### Run

To generate paraphrases using T5, run the following command:

> Note: T5 benefits from more few shot examples as it actually performs some gradient steps. However, to make it comparable to GPT-3, we don't recommend exceeding 50 examples.

```bash
poetry run python -m paraphrase.generate --model_name_or_path t5-11b --input_file input.txt --output_file output.txt --prompts prompts.txt
```

For generating paraphrases using GPT-3, run the following command:

> Warning: Using GPT-3 requires a paid account and can quickly run up a bill if you don't have credits.
> Reducing the number of prompts and/or the number of samples can help reduce costs.

```bash
OPENAI_API_KEY={YOUR_KEY} poetry run python paraphrase.generate --model_name_or_path gpt3 --input_file input.txt --output_file output.txt --prompts prompts.txt
```

For help, run the following command:

```bash
poetry run python -m paraphrase.generate --help
```

## Dataset

The dataset generated for our study is available on [🤗 Hugging Face Datasets](https://huggingface.co/datasets/jpwahle/autoregressive-paraphrase-dataset).

## Detection

For the detection code, please refer to this [repository](https://github.com/jpwahle/iconf22-paraphrase) and [paper](https://link.springer.com/chapter/10.1007/978-3-030-96957-8_34).

For all models except GPT-3 and T5, we used the trained versions on MPC. For PlagScan, we embedded the text in the same way as in the paper above.
 
## Citation
```bib
@inproceedings{Wahle2022d,
  title        = {How Large Language Models are Transforming Machine-Paraphrased Plagiarism},
  author       = {Jan Philip Wahle and Terry Ruas and Frederic Kirstein and Bela Gipp},
  year         = 2022,
  month        = {Dec.},
  publisher    = {Association for Computational Linguistics},
  booktitle    = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  doi          = {10.48550/arXiv.2210.03568},
  url          = {https://arxiv.org/abs/2210.03568},
}
```
## License
This repository is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
Use the code for any of your research projects, but be nice and give credit where credit is due.
Any illegal use for plagiarism or other purposes is prohibited.