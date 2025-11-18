import re
from typing import Optional

from client_utils.openai_api import OpenAIClient
from data_utils.aokvqa.evaluator import eval_aokvqa_direct
from data_utils.chart.evaluator import eval_one_chart
from data_utils.commom_util import prompt_ic

import math
import os
from filelock import FileLock
TEMPLATE_FILE = "best_template.txt"
LOCK_FILE = "best_template.txt.lock"
# ----------------------------------------------------

def _get_llm_comparison(client, system_prompt, current_template, new_template) -> bool:
    comparison_prompt = f"""You are an expert in AI prompt engineering. Your task is to compare two reasoning templates. You must decide if the 'New Template' should replace the 'Current Template' as the single 'best' template.

My goal is to keep only the *best*, *clearest*, and *most general* template.

---
**Current Template:** {current_template}
---
**New Template:** {new_template}
---

**Instructions:**
1.  **Check for Novelty:** Is the 'New Template' *semantically different*?
2.  **Check for Quality:** If different, is the 'New Template' *objectively better* or *more general*?
3.  **Decision:** Should the 'New Template' **replace** the 'Current Template'?

Respond with **only** the word "YES" or "NO".

**Decision:**"""

    try:
        response = client.get_completion(comparison_prompt, system_prompt=system_prompt, max_tokens=30)
        decision = response.strip().upper()
        return decision == "YES"
    except Exception as e:
        return False 


def _read_current_template(lock: FileLock) -> str:
    """在锁保护下安全地读取文件内容（操作很快）"""
    try:
        with lock.acquire(timeout=5):
            if not os.path.exists(TEMPLATE_FILE):
                return ""
            with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception as e:
        print(f"[Process {os.getpid()}] Failed to read template: {e}")
        return ""  # 出错时返回空字符串


def _optimistic_write_template(lock: FileLock, new_template: str, original_template: str) -> bool:
    """
    执行“比较并交换”（Compare-and-Swap）的写入操作。
    只有当文件内容仍等于 original_template 时，才写入 new_template。
    """
    try:
        with lock.acquire(timeout=10):
            # 步骤 4：再次读取
            current_template_on_disk = ""
            if os.path.exists(TEMPLATE_FILE):
                with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
                    current_template_on_disk = f.read().strip()

            # 步骤 4.1：检查冲突
            # 比较磁盘上的模板和我们用于 LLM 比较的“原始”模板
            if current_template_on_disk != original_template:
                # 情况 2（冲突）：另一个进程已经修改了文件
                print(f"[Process {os.getpid()}] Write aborted. Template was modified by another process.")
                return False

            # 情况 1（成功）：文件未变，安全写入
            with open(TEMPLATE_FILE, "w", encoding="utf-8") as f:
                f.write(new_template)
            print(f"[Process {os.getpid()}] New template successfully written.")
            return True

    except Exception as e:
        print(f"[Process {os.getpid()}] Failed to write template: {e}")
        return False

def update_best_template_if_different(client, system_prompt, new_template: str):
    """
    协调整个“乐观锁”流程：
    1.（无锁）读取
    2.（无锁）慢速 LLM 比较
    3.（有锁）比较并交换（CAS）写入
    """
    lock = FileLock(LOCK_FILE)
    clean_new_template = new_template.strip()
    if not clean_new_template:
        return

    # 步骤 1：（有锁，但极快）读取当前模板
    original_template = _read_current_template(lock)

    # 如果模板完全一样，跳过昂贵的 LLM 调用
    if original_template == clean_new_template:
        return

    # 步骤 2：（无锁，慢速）执行 LLM 比较
    is_better = _get_llm_comparison(client, system_prompt, original_template, clean_new_template)

    # 步骤 3：（有锁，快速）尝试“乐观写入”
    if is_better:
        _optimistic_write_template(lock, clean_new_template, original_template)

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
            elif 'math_lm' in task:
                reference_answer = reference_answer.lower().replace('answer:', '').strip()
                reward = eval_one_chart(response, reference_answer, 0, answer_flag=self.answer_flag)
                return float(reward)
            elif 'world' in task:
                reward = eval_aokvqa_direct(response, reference_answer)
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
        elif 'world' in task:
            system_prompt = 'You are a seasoned professional in the field of world knowledge and image analysis, demonstrating exceptional expertise and insight into complex real-world scenarios. Your output should be only judgement, without any additional text or explanation.'
        else:
            Exception('Unknown expert task')

        try:
            thinking = response.lower().split(self.answer_flag)[0].strip()
            in_context_example = self.client.get_completion(prompt_ic % hint, system_prompt=system_prompt, max_tokens=5000)

            if 'chart' in task or 'world' in task:
                if 'chart' in task:
                    from data_utils.chart.prompts import prompt_thinking_reward, prompt_template
                else:
                    from data_utils.aokvqa.prompts import prompt_thinking_reward, prompt_template
                # Construct the final prompt for the evaluator model.
                evaluation_prompt = prompt_thinking_reward % (in_context_example, question, answer, thinking)
                output = self.client.get_completion(evaluation_prompt, system_prompt=system_prompt, max_tokens=10)
                reward = get_score(output)

                if reward == 1:
                    template_prompt = prompt_template % thinking
                    ext_template = self.client.get_completion(template_prompt, system_prompt=system_prompt, max_tokens=512)
                    if "none" not in ext_template.strip().lower():
                        update_best_template_if_different(self.client, system_prompt, ext_template)
                return reward
            else:
                raise ValueError(f"Task '{task}' not supported for thinking reward.")
        except Exception as e:
            print(f"An error occurred during thinking reward prompt generation: {e}")
            return None


import spacy
import string
import re


class RewardCalculatorLocal:
    def __init__(self, RL_CONFIG, CLIENT_CONFIG, gpu_id=0):
        # ... 其他初始化代码 ...
        self.answer_flag = RL_CONFIG["answer_flag"].lower()
        self.count_pattern = re.compile(f'(?i){re.escape(self.answer_flag)}')

        # 加载 spaCy 的小型英文模型
        # 我们可以在初始化时加载一次，避免重复加载
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # 定义我们认为“重要”的词性标签
        # NOUN (名词), PROPN (专有名词), VERB (动词), ADJ (形容词), NUM (数字)
        # 您可以根据需求调整这个列表
        self.important_pos_tags = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'}

    def _preprocess_text_pos(self, text: str) -> set[str]:
        """
        使用词性标注来提取关键词
        """
        doc = self.nlp(text.lower())
        keywords = set()
        for token in doc:
            # 只保留重要词性的词，并且确保它不是停用词或标点符号
            if token.pos_ in self.important_pos_tags and not token.is_stop and not token.is_punct:
                # 使用 .lemma_ 来获取词的原形，例如 'sales' -> 'sale'
                keywords.add(token.lemma_)
        return keywords

    def get_thinking_reward_prompt(self, response: str, question: str, answer: str, hint: str, task: str):
        try:
            thinking_part = response.lower().split(self.answer_flag)[0].strip()
            if not thinking_part:
                return 0.0

            # 使用基于词性的新方法
            thinking_tokens = self._preprocess_text_pos(thinking_part)
            reference_tokens = self._preprocess_text_pos(hint)

            # 计算交集
            common_tokens = thinking_tokens.intersection(reference_tokens)

            # 计算精确率 (Precision)
            # 在模型生成的所有词中，有多少是正确的（在hint中出现）
            precision = len(common_tokens) / (len(thinking_tokens) + 1e-6)

            # 计算召回率 (Recall)
            # 在所有正确的词（hint）中，有多少被模型找到了
            recall = len(common_tokens) / (len(reference_tokens) + 1e-6)

            # 计算 F1-Score
            if precision + recall == 0:
                return 0.0

            f1_score = 2 * (precision * recall) / (precision + recall)

            return f1_score

        except Exception as e:
            print(f"在本地计算思考奖励时发生错误: {e}")
            return 0.0

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
            elif 'math_lm' in task:
                reference_answer = reference_answer.lower().replace('answer:', '').strip()
                reward = eval_one_chart(response, reference_answer, 0, answer_flag=self.answer_flag)
                return float(reward)
            elif 'world' in task:
                reward = eval_aokvqa_direct(response, reference_answer)
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