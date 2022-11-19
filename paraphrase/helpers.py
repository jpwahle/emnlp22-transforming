import evaluate
import pandas as pd
import torch
from accelerate import Accelerator
from tqdm import tqdm


def to_disk(dataset, path):
    # Save dataset
    pd.DataFrame(dataset).to_csv(
        path,
        index=False,
        sep="\t",
    )


def create_metrics():
    # Define metrics
    compute_bertscore = evaluate.load("bertscore")
    compute_rouge_L = evaluate.load("rouge")
    compute_bleu = evaluate.load("bleu")
    # List metrics as uniform lambdas
    metrics = [
        lambda original, paraphrased: compute_bertscore.compute(  # + maximize BARTScore
            # Maximize BARTScore
            references=[original],
            predictions=[paraphrased],
            model_type="facebook/bart-large",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )[
            "f1"
        ][
            0
        ],
        lambda original, paraphrased: compute_bertscore.compute(  # + maximize BERTScore
            # Maximize BERTScore
            references=[original],
            predictions=[paraphrased],
            model_type="bert-large-uncased",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )[
            "f1"
        ][
            0
        ],
        lambda original, paraphrased: -compute_rouge_L.compute(  # - minimize ROUGE_L
            # Minimize rouge
            references=[original],
            predictions=[paraphrased],
        )[
            "rougeL"
        ],
        lambda original, paraphrased: -compute_bleu.compute(  # - minimize BLEU
            # Minimize bleu
            references=[original],
            predictions=[paraphrased],
        )["bleu"],
    ]
    return metrics


def pre_prediction(
    few_shot_examples,
    model,
    model_name,
    tokenizer,
    num_examples=100,
    weight_decay=0.0,
    learning_rate=5e-5,
):
    if "gpt" not in model_name:
        model.train()
        # Define the accelerator
        accelerator = Accelerator()
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=learning_rate
        )

        for original, paraphrased in tqdm(
            zip(
                few_shot_examples["sentence1"][:num_examples],
                few_shot_examples["sentence2"][:num_examples],
            )
        ):
            task_prefix = "Paraphrase: "
            input_ids = tokenizer(
                task_prefix + original,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
            labels = tokenizer(paraphrased, return_tensors="pt").input_ids.to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            # the forward function automatically creates the correct decoder_input_ids
            loss = model(input_ids=input_ids, labels=labels).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        # Back to eval mode
        model.eval()
