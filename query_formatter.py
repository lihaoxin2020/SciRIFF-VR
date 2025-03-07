from datasets import load_dataset
import random

CONTEXT_ARGS = {
    "lab-bench": {
        "description": 'Answer the following multiple-choice question by giving the correct answer letter in parentheses. Provide CONCISE reasoning for the answer, and make sure to finish the response with "ANSWER: X" where X is one of A, B, C, D.\n\n',
    }
}

def make_mcq_prompt(
    question,
    choices,
    question_prefix="Question: ",
    choices_prefix="",
    answer_prefix="Answer:",
    label_prefix=" ",
    label_format=None,
):
    choice_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_format = label_format or label_prefix + "A."
    if "A" not in label_format:
        raise ValueError(
            "Label format must contain 'A' to indicate where to place the answer label"
        )
    # Use prefix space before each answer label to avoid ambiguity in tokenization
    choices_text = "\n".join(
        [
            f"{label_format.replace('A', label)} {text}"
            for label, text in zip(choice_labels, choices)
        ]
    )
    prompt = f"{question_prefix}{question}\n{choices_prefix}{choices_text}\n{answer_prefix}"
    return prompt


if __name__ == "__main__":
    random.seed(42)
    ds = load_dataset("futurehouse/lab-bench")['train']
    
    instance = ds[0]
    task_name = "lab-bench"
    question = instance["question"]
    options = instance['distractors'] + instance['ideal']
    random.shuffle(options)

    query = make_mcq_prompt(question, options, question_prefix=CONTEXT_ARGS[task_name]['description'])
    query += " Let's think step-by-step."
    
