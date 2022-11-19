def construct_few_shot_prompt(
    few_shot_examples,
    original_paragraph,
    model_name,
    prompt="Rephrase the following paragraph",
    num_examples=3,
):
    if "gpt" in model_name:
        # Newline after the prompt
        prompt += "\n"
        # Construct `num_examples` few shot examples
        for original, paraphrased in zip(
            few_shot_examples["sentence1"][:num_examples],
            few_shot_examples["sentence2"][:num_examples],
        ):
            # Add the original sentence
            prompt += "Original: "
            prompt += original
            prompt += "\n"
            # Add the paraphrased sentence
            prompt += "Paraphrased: "
            prompt += paraphrased
            prompt += "\n"
            prompt += (  # Seperator (optional) but helps the model to understand the boundaries
                "###"
            )
            prompt += "\n"
        # Add the original paragraph that we want to paraphrase
        prompt += "Original: "
        prompt += original_paragraph
        prompt += "\n"
        prompt += "Paraphrased:"
    elif "t5" in model_name:
        # Only use the prompt as few shot examples are provided before
        prompt = "Paraphrase: " + original_paragraph
    # Return the prompt
    return prompt
