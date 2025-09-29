import concurrent.futures
from typing import List, Dict, Any
from  checker import RewardCalculator

def calculate_rewards_in_parallel(
    checker: RewardCalculator,
    batch_data: Dict[str, Any],
    gpu_id: int,
    num_threads: int = 8,
    task='chart'
) -> List[float]:
    """
    Calculates accuracy rewards for a batch of data in parallel using a thread pool.

    Args:
        batch_data: A dictionary containing lists of data, including 'response',
                    'prompt', 'image', 'answer', and an optional 'tp' (answer_type).
        gpu_id: The ID of the GPU to be used for processing.
        num_threads: The number of parallel threads to use.

    Returns:
        A list of calculated reward scores for each item in the batch.
    """
    # Extract lists of data from the input dictionary
    responses = batch_data['response']
    questions = batch_data['prompt']
    answers = batch_data['answer']
    hints = batch_data['hint'] if 'hint' in batch_data else [""] * len(responses)
    num_samples = len(responses)

    # Safely get 'answer_types', providing a list of Nones as a default
    # This fixes a bug in the original code.
    answer_types = batch_data.get('tp', [None] * num_samples)

    # Prepare the arguments for each task by zipping the data together.
    # This creates an iterator of tuples, where each tuple contains all args for one call.
    task_answer_args = zip(
        responses,
        answers,
        [task] * num_samples,
        [gpu_id] * num_samples,
        answer_types,
        hints
    )
    task_thinking_args = zip(
        responses,
        questions,
        answers,
        hints,
        [task] * num_samples
    )

    # Use a ThreadPoolExecutor to process the data in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Instead of a separate function, use a lambda to unpack the arguments.
        # The '*' operator unpacks each tuple from task_args into positional arguments
        # for the get_acc_reward function.

        format_rewards = list(executor.map(lambda args: checker.get_format_reward(args[0]), responses))
        answer_rewards = list(executor.map(lambda args: checker.get_answer_reward(*args), task_answer_args))
        thinking_rewards = list(executor.map(
            lambda args: checker.get_thinking_reward_prompt(*args), task_thinking_args
        ))

        rewards = [0 if f == 0 else f + a + t for f, a, t in zip(format_rewards, answer_rewards, thinking_rewards)]

    return rewards