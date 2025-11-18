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

# --- 新增：每隔多少秒检查一次文件更新 ---
TEMPLATE_REFRESH_INTERVAL = 60  # 60 秒
# ----------------------------------------------------


class ContextRefiner:
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
        self.refine_templetes = ["""Goal: [State the user's objective, e.g., Find the year with the highest sales]
Observation: [List key data points from the chart, e.g., 2020: 150, 2021: 200, 2022: 180]
Reasoning: [State the logical step, e.g., Compare the values. 200 is the maximum.]
Conclusion: [Draw the conclusion, e.g., The year with the highest sales was 2021.]
"""]
        self.template_lock = FileLock(LOCK_FILE)
        # 设置为 0，以便在第一次调用时强制触发一次读取
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
        （私有方法）检查是否需要从文件刷新模板。
        这个操作是进程安全的。
        """
        current_time = time.time()

        # 1. 检查是否达到了刷新间隔
        if (current_time - self.last_template_check_time) < TEMPLATE_REFRESH_INTERVAL:
            return  # 未到时间，使用缓存

        # 2. 尝试获取锁并读取（使用短超时，因为读取应该很快）
        try:
            # print(f"[Process {os.getpid()}] Checking for template update...") # 调试时取消注释
            with self.template_lock.acquire(timeout=5):

                # --- 已获取锁，安全读取 ---
                if not os.path.exists(TEMPLATE_FILE):
                    # 文件不存在，使用默认值
                    self.last_template_check_time = current_time
                    return

                with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
                    new_template = f.read().strip()

                # 如果文件有效且内容有变化，则更新内存中的模板
                if new_template and new_template != self.refine_templetes[0]:
                    self.refine_templetes = [new_template]
                    print(f"[Process {os.getpid()}] Refiner template updated from file.")

            # 无论成功与否，都更新检查时间，避免频繁重试
            self.last_template_check_time = current_time

        except TimeoutError:
            # 获取锁超时（意味着另一个进程正在写入）
            # 我们不等待，直接跳过，下次再试
            print(f"[Process {os.getpid()}] Failed to acquire lock for template read, using cached version.")
            # 更新时间，防止立即重试
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
            in_context_example = self.client.get_completion(prompt_ic % hint, system_prompt=system_prompt,
                                                            max_tokens=5000)

            if 'chart' in task or 'world' in task:
                if 'chart' in task:
                    from data_utils.chart.prompts import prompt_thinking_reward, prompt_refine
                else:
                    from data_utils.aokvqa.prompts import prompt_thinking_reward, prompt_refine
                # Construct the final prompt for the evaluator model.
                evaluation_prompt = prompt_refine % (in_context_example, question, reference_answer, self.refine_templetes[0])
                output = self.client.get_completion(evaluation_prompt, system_prompt=system_prompt, max_tokens=1000)
                return output
            else:
                raise ValueError(f"Task '{task}' not supported for thinking reward.")
        except Exception as e:
            print(f"An error occurred during thinking reward prompt generation: {e}")
            return None


class ContextRefinerLocal:
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
        # do nothing.

    def refine_hint(self, question, hint: str, reference_answer: str, task: str, gpu_id=None):
        return hint