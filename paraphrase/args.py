import argparse


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--num_prompts",
        type=int,
        default=4,
        help="Number of prompts to use for each paragraph",
    )
    argparser.add_argument(
        "--num_examples",
        type=int,
        default=32,
        help="Number of examples to process",
    )
    argparser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Index of the first example to process",
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for the model",
    )
    argparser.add_argument(
        "--model_name",
        type=str,
        default="t5-small",
        help="Name of the model to use",
    )
    argparser.add_argument(
        "--paraphrase_dataset",
        type=str,
        default="mrpc",  # JW: Use mrpc for now until PPDB and P4P are on huggingface with the new implementation
        help="Name of the dataset to use for paraphrase few-shot examples",
    )
    return argparser.parse_args()
