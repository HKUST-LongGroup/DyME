import re
from typing import Optional

from client_utils.openai_api import OpenAIClient
from data_utils.chart.evaluator import eval_one_chart
from data_utils.commom_util import prompt_ic

import os
import time
from filelock import FileLock
TEMPLATE_FILE = "best_template.txt"
LOCK_FILE = "best_template.txt.lock"


TEMPLATE_REFRESH_INTERVAL = 60  # Interval (in seconds) to refresh template from file


class ContextRefiner:
    """
    A class to refine hints / reasoning with an external LLM.
    Encapsulates logic for template management and refinement calls.
    """

    def __init__(self, RL_CONFIG, CLIENT_CONFIG, gpu_id=0):
        """
        Initializes the ContextRefiner.

        Args:
            RL_CONFIG: RL-related configuration dict.
            CLIENT_CONFIG: LLM client configuration dict.
            gpu_id: process / GPU id used to select API server.
        """
        self.refine_templetes = ["""Goal: [State the user's objective, e.g., Find the year with the highest sales]
Observation: [List key data points from the chart, e.g., 2020: 150, 2021: 200, 2022: 180]
Reasoning: [State the logical step, e.g., Compare the values. 200 is the maximum.]
Conclusion: [Draw the conclusion, e.g., The year with the highest sales was 2021.]
"""]
        self.template_lock = FileLock(LOCK_FILE)
        # Set to 0 so that the first call will force a refresh from file
        self.last_template_check_time = 0
        if CLIENT_CONFIG['client_type'] == 'openai':
            if CLIENT_CONFIG['init_port'] is not None:
                num_server = int(CLIENT_CONFIG['num_server'])
                server_id = gpu_id % num_server
                CLIENT_CONFIG['api_base'] = CLIENT_CONFIG['api_base'] % str(CLIENT_CONFIG['init_port'] + server_id)
            self.client = OpenAIClient(config=CLIENT_CONFIG)
        else:
            raise ValueError(f"Client type '{CLIENT_CONFIG['client_type']}' not supported.")

    def _check_and_update_template(self):
        """
        (Private method) Check whether we need to refresh the template from file.
        This operation is process-safe.
        """
        current_time = time.time()

        # 1. Check whether the refresh interval has passed
        if (current_time - self.last_template_check_time) < TEMPLATE_REFRESH_INTERVAL:
            return  # Not yet time, keep using cached template

        # 2. Try to acquire the lock and read (short timeout since reading should be fast)
        try:
            # print(f"[Process {os.getpid()}] Checking for template update...")  # Uncomment for debugging
            with self.template_lock.acquire(timeout=5):

                # --- Lock acquired, safe to read ---
                if not os.path.exists(TEMPLATE_FILE):
                    # File does not exist, keep default template
                    self.last_template_check_time = current_time
                    return

                with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
                    new_template = f.read().strip()

                # If file content is valid and different, update in-memory template
                if new_template and new_template != self.refine_templetes[0]:
                    self.refine_templetes = [new_template]
                    print(f"[Process {os.getpid()}] Refiner template updated from file.")

            # Regardless of success, update last check time to avoid frequent retries
            self.last_template_check_time = current_time

        except TimeoutError:
            # Failed to acquire lock (another process is likely writing)
            # Do not block; skip and try again next time
            print(f"[Process {os.getpid()}] Failed to acquire lock for template read, using cached version.")
            # Update time to avoid immediate retry
            self.last_template_check_time = current_time

        except Exception as e:
            print(f"[Process {os.getpid()}] Error reading template file: {e}")
            self.last_template_check_time = current_time

    def refine_hint(self, question, hint: str, reference_answer: str, task: str, gpu_id=None):
        if hint == "":
            return hint

        self._check_and_update_template()
        system_prompt = None
        if 'medical' in task:
            system_prompt = 'You are a seasoned professional in the field of medical image analysis, demonstrating exceptional expertise and insight into complex medical imaging data. Your output should be only judgement, without any additional text or explanation.'
        elif 'math' in task:
            system_prompt = 'You are a seasoned professional in the field of mathematics, demonstrating exceptional expertise and insight into complex mathematical problems. Your output should be only judgement, without any additional text or explanation.'
        elif 'chart' in task:
            system_prompt = 'You are a seasoned professional in the field of chart analysis, demonstrating exceptional expertise and insight into complex chart data. Your output should be only judgement, without any additional text or explanation.'
        elif 'world' in task:
            system_prompt = 'You are a seasoned professional in the field of world knowledge and image analysis, demonstrating exceptional expertise and insight into complex real-world scenarios. Your output should be only judgement, without any additional text or explanation.'
        else:
            Exception('Unknown expert task')

        try:
            in_context_example = self.client.get_completion(
                prompt_ic % hint,
                system_prompt=system_prompt,
                max_tokens=5000
            )

            if 'chart' in task or 'world' in task:
                if 'chart' in task:
                    from data_utils.chart.prompts import prompt_thinking_reward, prompt_refine
                else:
                    from data_utils.aokvqa.prompts import prompt_thinking_reward, prompt_refine
                # Construct the final prompt for the evaluator model.
                evaluation_prompt = prompt_refine % (
                    in_context_example,
                    question,
                    reference_answer,
                    self.refine_templetes[0]
                )
                output = self.client.get_completion(
                    evaluation_prompt,
                    system_prompt=system_prompt,
                    max_tokens=1000
                )
                return output
            else:
                raise ValueError(f"Task '{task}' not supported for thinking reward.")
        except Exception as e:
            print(f"An error occurred during thinking reward prompt generation: {e}")
            return None


class ContextRefinerLocal:
    """
    A local (non-LLM) refiner that simply returns the original hint.
    Used when remote refinement is disabled or not desired.
    """

    def __init__(self, RL_CONFIG, CLIENT_CONFIG, gpu_id=0):
        """
        Initializes the local ContextRefiner.

        Args:
            RL_CONFIG: RL-related configuration dict.
            CLIENT_CONFIG: client configuration dict (unused here).
            gpu_id: process / GPU id (unused here).
        """
        # Do nothing; local refiner is a no-op.
        pass

    def refine_hint(self, question, hint: str, reference_answer: str, task: str, gpu_id=None):
        return hint
