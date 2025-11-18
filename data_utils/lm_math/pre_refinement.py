# process_json_distributed.py

import json
import multiprocessing
import os
from copy import deepcopy
from tqdm import tqdm
import sys
sys.path.append('/apdcephfs_qy4/share_302593112/realzliu/code/DyME')
from client_utils.openai_api import OpenAIClient
from data_utils.chart.prompts import prompt_refine
from data_utils.commom_util import prompt_ic


# ---------------- 您提供的 ContextRefiner 类 (保持不变) ----------------
class ContextRefiner:
    """
    这个类基本保持不变，但初始化逻辑现在将在每个工作进程中被正确调用。
    """

    def __init__(self, CLIENT_CONFIG, gpu_id=0):  # 添加了 gpu_id 参数以匹配原始代码
        self.refine_templetes = ["""Goal: [State the main question to be answered in one simple sentence.]
Observation: [List the key numbers and the relationships between them from the problem statement.]
Reasoning: [Show the step-by-step calculation process. Each step should be a clear mathematical operation.]
Conclusion: [State the final answer clearly.]
"""]
        if CLIENT_CONFIG['client_type'] == 'openai':
            # 关键逻辑：这个端口计算现在在 worker_initializer 中完成
            self.client = OpenAIClient(config=CLIENT_CONFIG)
        else:
            raise ValueError(f"Client type '{CLIENT_CONFIG['client_type']}' not supported.")

    def refine_hint(self, question: str, hint: str, reference_answer: str, task: str):
        # ... 此处方法体与之前完全相同，省略以保持简洁 ...
        if not hint:
            return hint
        system_prompt = None
        if 'chart' in task:
            system_prompt = 'You are a seasoned professional in the field of chart analysis...'
        elif 'math' in task:
            system_prompt = 'You are a seasoned professional in the field of mathematics, demonstrating exceptional expertise and insight into complex mathematical problems. Your output should be only judgement, without any additional text or explanation.'
        else:
            raise Exception('Unknown expert task')
        try:
            in_context_example = self.client.get_completion(prompt_ic % hint, system_prompt=system_prompt,
                                                            max_tokens=5000)
            if 'chart' in task or 'math' in task:
                evaluation_prompt = prompt_refine % (in_context_example, question, reference_answer,
                                                     self.refine_templetes[0])
                output = self.client.get_completion(evaluation_prompt, system_prompt=system_prompt, max_tokens=1000)
                return output
            else:
                raise ValueError(f"Task '{task}' not supported for thinking reward.")
        except Exception as e:
            print(f"处理 '{question}' 时发生错误: {e}")
            return hint


# ---------------- 多进程工作函数 (已修正) ----------------

refiner_instance = None


def worker_initializer(base_client_config):
    """
    每个工作进程启动时调用的初始化函数 (已修正)。
    为每个进程创建独立的、配置不同的 Refiner 实例。
    """
    global refiner_instance

    # 关键改动: 获取当前工作进程的唯一ID (从1开始)
    # 这就是我们用来模拟 gpu_id 的变量
    worker_id = multiprocessing.current_process()._identity[0] - 1

    # 创建配置的深拷贝，避免进程间互相影响
    worker_config = deepcopy(base_client_config)

    # 关键改动: 实现您代码中的端口计算逻辑
    if worker_config.get('init_port') is not None and worker_config.get('num_server') is not None:
        num_server = int(worker_config['num_server'])
        # server_id 决定使用哪个端口
        server_id = worker_id % num_server
        port = worker_config['init_port'] + server_id

        # 格式化 api_base，为该进程分配一个固定的端口
        worker_config['api_base'] = worker_config['api_base'] % str(port)

        print(f"进程 {os.getpid()} (Worker-{worker_id}) 正在初始化... 连接到 {worker_config['api_base']}")
    else:
        print(f"进程 {os.getpid()} (Worker-{worker_id}) 正在初始化... 使用默认 api_base")

    # 使用为这个特定进程定制的配置来创建实例
    refiner_instance = ContextRefiner(worker_config, gpu_id=worker_id)


def process_item_worker(item):
    """单个工作进程要执行的函数 (保持不变)"""
    global refiner_instance
    if refiner_instance is None:
        raise Exception("Refiner 未在工作进程中初始化！")

    new_hint = refiner_instance.refine_hint(
        question=item['question'],
        hint=item['hint'],
        reference_answer=item['answer'],
        task='math'
    )
    item['hint'] = new_hint
    return item


# ---------------- 主逻辑 ----------------
def main():
    # 包含了端口和服务器数量信息的配置
    from config import CLIENT_CONFIG
    input_filename = '/path/to/data/lm_math/json/train.json'
    output_filename = '/path/to/data/lm_math/json/train_refine.json'

    NUM_PROCESSES = 64
    print(f"将使用 {NUM_PROCESSES} 个进程，并向 {CLIENT_CONFIG['num_server']} 个服务器分发请求...")

    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_filename}' 未找到。")
        return

    processed_data = []

    # 关键：将基础配置传递给每个进程的 initializer
    with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=worker_initializer,
                              initargs=(CLIENT_CONFIG,)) as pool:
        with tqdm(total=len(data), desc="正在并行处理JSON") as pbar:
            for result in pool.imap_unordered(process_item_worker, data):
                processed_data.append(result)
                pbar.update(1)

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"\n处理完成！结果已保存到 '{output_filename}'。")


if __name__ == "__main__":
    main()