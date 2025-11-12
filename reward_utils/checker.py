import re
from typing import Optional

from client_utils.openai_api import OpenAIClient
from data_utils.chart.evaluator import eval_one_chart
from data_utils.commom_util import prompt_ic
from data_utils.chart.prompts import prompt_thinking_reward

# ----------------------------------------------------


class RewardCalculator:
    """
    A class to calculate various rewards for a model's response.
    Encapsulates logic for answer correctness, format adherence, and thinking quality.
    """

    def __init__(self, RL_CONFIG, CLIENT_CONFIG, gpu_id=0):
        """
        Initializes the RewardCalculator.

        Args:
            answer_flag (str): The keyword that separates reasoning from the final answer.
        """
        answer_flag = RL_CONFIG["answer_flag"]
        self.answer_flag = answer_flag.lower()
        self.count_pattern = re.compile(f'(?i){re.escape(self.answer_flag)}')
        if CLIENT_CONFIG['client_type'] == 'openai':
            if CLIENT_CONFIG['init_port'] is not None:
                num_server = int(CLIENT_CONFIG['num_server'])
                server_id = gpu_id % num_server
                CLIENT_CONFIG['api_base'] = CLIENT_CONFIG['api_base'] % str(CLIENT_CONFIG['init_port'] + server_id)
            self.client = OpenAIClient(config=CLIENT_CONFIG)
        else:
            raise ValueError(f"Client type '{CLIENT_CONFIG['client_type']}' not supported.")

    def get_answer_reward(self, response: str, reference_answer: str, task: str, gpu_id=None, answer_type=None) -> float:
        """
        Calculates the correctness reward for the answer.
        Returns 1.0 if correct, 0.0 otherwise.
        """
        try:
            if 'chart' in task:
                # Assuming eval_one_chart returns a float (e.g., 1.0 for correct, 0.0 for incorrect)
                reference_answer = reference_answer.lower().replace('answer:', '').strip()
                reward = eval_one_chart(response, reference_answer, 0, answer_flag=self.answer_flag)
                return float(reward)
            else:
                raise ValueError(f"Task '{task}' not supported for answer reward.")
        except Exception as e:
            # Catch specific exceptions and log them for better debugging.
            print(f"An error occurred during answer reward calculation: {e}")
            return 0.0

    def get_format_reward(self, response: str, min_thinking_length: int = 0) -> float:
        """
        Calculates the format reward based on two criteria:
        1. The 'answer:' flag must appear exactly once.
        2. The preceding 'thinking' text must meet a minimum length.

        Returns 1.0 if the format is correct, 0.0 otherwise.
        """
        # 1. Check if the answer flag appears exactly once.
        if len(self.count_pattern.findall(response)) != 1:
            return 0.0

        # 2. Check if the 'thinking' part has sufficient length.
        thinking = response.lower().split(self.answer_flag)[0]
        if len(thinking.strip()) < min_thinking_length:
            return 0.0

        return 1.0

    def get_thinking_reward_prompt(self, response: str, question: str, answer: str, hint: str, task: str):
        """
        Generates a prompt for an LLM to evaluate the quality of the 'thinking' process.

        This function prepares the input; an external LLM call would be needed to get a score.

        Returns:
            A formatted prompt string, or None if the task is unsupported.
        """

        def get_score(level_string):
            if "low" in level_string:
                return 0
            elif "medium" in level_string:
                return 0.5
            elif "high" in level_string:
                return 1
            else:
                # 处理未知输入
                return 0  # 或者可以返回 -1, 或者抛出一个错误

        system_prompt = None
        if 'medical' in task:
            system_prompt = 'You are a seasoned professional in the field of medical image analysis, demonstrating exceptional expertise and insight into complex medical imaging data. Your output should be only judgement, without any additional text or explanation.'
        elif 'math' in task:
            system_prompt = 'You are a seasoned professional in the field of mathematics, demonstrating exceptional expertise and insight into complex mathematical problems. Your output should be only judgement, without any additional text or explanation.'
        elif 'chart' in task:
            system_prompt = 'You are a seasoned professional in the field of chart analysis, demonstrating exceptional expertise and insight into complex chart data. Your output should be only judgement, without any additional text or explanation.'
        else:
            Exception('Unknown expert task')

        try:
            thinking = response.lower().split(self.answer_flag)[0].strip()
            in_context_example = self.client.get_completion(prompt_ic % hint, system_prompt=system_prompt, max_tokens=5000)

            if 'chart' in task:
                # Construct the final prompt for the evaluator model.
                evaluation_prompt = prompt_thinking_reward % (in_context_example, question, answer, thinking)
                output = self.client.get_completion(evaluation_prompt, system_prompt=system_prompt, max_tokens=10)
                reward = get_score(output)
                return reward
            else:
                raise ValueError(f"Task '{task}' not supported for thinking reward.")
        except Exception as e:
            print(f"An error occurred during thinking reward prompt generation: {e}")
            return None

