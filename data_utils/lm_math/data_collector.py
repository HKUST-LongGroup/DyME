import os
import json
from typing import List, Dict, Any
from config import CONFIG

# Define the prompt template as a constant for clarity and easy modification.
PROMPT_TEMPLATE = (
    "Your task is to answer the question below. "
    "Give step by step thinking before you answer, and when you're ready to answer, "
    "please use the format \"Answer: ..\"\n\n"
    "Question:\n\n{question}"
)
ANSWER_TEMPLATE = CONFIG['rl']['answer_flag'] + " " +  "{answer}"

def prepare_math_lm_rl_data(json_path: str) -> List[Dict[str, Any]]:
    """
    Processes a JSON file of chart data for Reinforcement Learning.

    This function reads a JSON file, filters out entries marked as 'machine-generated',
    cleans the 'answer' field, and constructs a formatted 'prompt'.

    Args:
        json_path: The file path to the input JSON data.

    Returns:
        A list of processed dictionaries, each with a new 'prompt' key.

    Raises:
        FileNotFoundError: If the json_path does not exist.
    """
    # Use a clear check for file existence and raise a specific error.
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Error: The file '{json_path}' was not found.")

    # Use 'with open' for safe file handling.
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed_data = []
    # Use a single, clear loop to both filter and process the data.
    for entry in raw_data:
        # Filter condition: Keep if the key is missing or its value is 0 (human).
        if entry.get('human_or_machine', 0) == 0:
            # Create a new dictionary to avoid modifying the original list in place.
            new_entry = entry.copy()

            # Clean up the answer text.
            if 'answer' in new_entry:
                new_entry['answer'] = ANSWER_TEMPLATE.format(answer=new_entry['answer'].strip())

            # Format the prompt using an f-string.
            new_entry['prompt'] = PROMPT_TEMPLATE.format(question=new_entry['question'])
            new_entry['question_wo_prompt'] = new_entry['question']
            new_entry.pop('question', None)

            processed_data.append(new_entry)

    return processed_data


