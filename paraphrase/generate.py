import os

import numpy as np
import openai
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer

from paraphrase.args import parse_args
from paraphrase.candidates import choose_pareto_optimal_candidate
from paraphrase.helpers import create_metrics, pre_prediction, to_disk
from paraphrase.prompts import construct_few_shot_prompt


def rephrase_paragraph(
    inputs,
    model,
    max_pred_tokens,
    tokenizer,
    max_total_tokens=512,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    if model == "gpt3":
        # Take care if you are using large models it might become expensive to use
        paraphrase = openai.Completion.create(
            # engine="text-davinci-001",
            engine="text-curie-001",
            prompt=inputs,
            max_tokens=max_pred_tokens,
        )
        # OpenAI returns a list of candidates which may be unsorted so we sort them by index
        paraphrase["choices"].sort(key=lambda x: x["index"])
        # Discard other metadata then text and return the list of candidates
        paraphrases = [el["text"] for el in paraphrase["choices"]]
    else:
        # Can potentially use any huggingface model as argument (here we just tried T5)
        input_ids = tokenizer(
            inputs,
            return_tensors="pt",
            max_length=max_total_tokens,
            padding=True,
            truncation=True,
        ).input_ids.to(device)
        with torch.no_grad():
            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                max_length=max_pred_tokens,
                num_beams=3,
                temperature=0.7,
            )
        # Decode the outputs
        paraphrases = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Clean paraphrases if they contain leasing spaces or newlines
    paraphrases = [
        paraphrase.strip().strip("\n").strip()
        for paraphrase in paraphrases
        if paraphrase.strip()
    ]
    return paraphrases


def rephrase_paragraphs(
    paragraphs,
    few_shot_examples,
    prompts,
    model,
    model_name,
    metrics,
    tokenizer,
    start_index,
    num_examples,
    num_prompts,
    batch_size,
    split="train",
):
    # Define lists
    model_inputs = []
    references = []
    source_datasets = []
    model_inputs_batched = []
    # Check if the inputs are valid
    assert (
        batch_size % num_prompts == 0
    ), "The batch size must be a multiple of the number of prompts"
    assert num_examples == -1 or start_index + num_examples <= len(
        paragraphs[split]
    ), "num_examples must be less than or equal to the number of paragraphs"
    assert (
        batch_size <= num_examples
    ), "batch size must be less than or equal to the number of examples"
    # Before prediction t5 needs to see few shot examples
    pre_prediction(few_shot_examples, model, model_name, tokenizer)
    # Iterate over the dataset and construct model inputs
    for paragraph in tqdm(
        paragraphs[split].select(
            range(start_index, start_index + num_examples)
        )
    ):
        # Clean the paragraph
        cleaned = paragraph["text"].strip().strip("\n").strip()
        # Add the current reference
        references.append(cleaned)
        # Add the current source
        source_datasets.append(paragraph["dataset"])
        # Choose a random subset of prompts
        prompts = np.random.choice(prompts, num_prompts, replace=False)
        # Construct the paraphrases
        for prompt in prompts:
            # Generate few shot prompt
            few_shot_prompt = construct_few_shot_prompt(
                few_shot_examples, cleaned, model_name, prompt
            )
            # Fill the batch
            if len(model_inputs) < batch_size:
                model_inputs.append(few_shot_prompt)
            # Add the batch to the model inputs
            else:
                # Add to the model generating inputs
                model_inputs_batched.append(model_inputs)
                # Reset the batches
                model_inputs = []
                # Add the current prompt
                model_inputs.append(few_shot_prompt)
    # Add the last batch
    if len(model_inputs) > 0:
        model_inputs_batched.append(model_inputs)
    output_dataset = []
    # Predict
    for i, inp in enumerate(tqdm(model_inputs_batched)):
        # Get the maximum number of tokens that we can generate
        max_pred_tokens = max([len(el) for el in tokenizer(inp).input_ids])
        # Rephrase the paragraphs
        paraphrases = rephrase_paragraph(
            inputs=inp,
            model=model,
            max_pred_tokens=int(max_pred_tokens * 0.9),
            tokenizer=tokenizer,
        )
        # Split candidates
        candidates = [
            paraphrases[j : j + num_prompts]
            for j in range(0, len(paraphrases), num_prompts)
        ]
        # Choose the pareto optimal candidates
        for candidates, reference, source_dataset in zip(
            candidates,
            references[
                int(i * (batch_size / num_prompts)) : int(
                    (i + 1) * (batch_size / num_prompts)
                )
            ],
            source_datasets[
                int(i * (batch_size / num_prompts)) : int(
                    (i + 1) * (batch_size / num_prompts)
                )
            ],
        ):
            # Choose the pareto optimal candidate
            pareto_optimal_candidate = choose_pareto_optimal_candidate(
                reference, candidates, metrics
            )
            # Add it to the output dataset
            output_dataset.append(
                {
                    "original": reference,
                    "paraphrased": str(pareto_optimal_candidate),
                    "model": model_name,
                    "dataset": source_dataset,
                }
            )

    return output_dataset


def main():
    # Get openai api key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # Parse the arguments
    args = parse_args()
    # Load prompts from autoprompt
    prompts = open("prompts.txt", "r").read().splitlines()
    # Define models (only GPT-3 needs an API key).
    model_name = args.model_name
    if model_name == "gpt3":
        model = "gpt3"
        tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2"
        )  # Use gpt2 tokenizer to estimate the number of tokens
    else:
        # Easily extendable to other models in hugginface
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        # Parallelize the model over multiple GPUs
        # If you want to use T5--3b or T5-11b you need to have 8 A100 GPUs
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model.parallelize()
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    # Load original dataset
    originals = load_dataset("jpwahle/autoencoder-paraphrase-dataset").filter(
        lambda x: x["method"] == "original"
    )  # JW: Updated to the new huggingface dataset which replaces the deprecated zenodo loader scripts
    # Sample randomly pairs of original and paraphrased examples
    paraphrase_examples = (
        load_dataset(args.paraphrase_dataset)
        .filter(lambda x: x["label"] == 1)
        .shuffle()[:100]
    )
    # Load the metrics
    metrics = create_metrics()
    # Generate paraphrases
    dataset = rephrase_paragraphs(
        originals,
        paraphrase_examples,
        prompts=prompts,
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        metrics=metrics,
        num_prompts=args.num_prompts,
        start_index=args.start_index,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
    )
    # Save the dataset
    to_disk(
        dataset,
        f"dataset_{args.model_name}_from_{args.start_index}_to_{args.start_index + args.num_examples}.tsv",
    )


if __name__ == "__main__":
    main()
