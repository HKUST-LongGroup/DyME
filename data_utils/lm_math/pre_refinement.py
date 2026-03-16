# process_json_distributed.py

import json
import multiprocessing
import os
from copy import deepcopy
from tqdm import tqdm
import sys
sys.path.append('/path/to/code/DyME')
from client_utils.openai_api import OpenAIClient
from data_utils.chart.prompts import prompt_refine
from data_utils.commom_util import prompt_ic


class ContextRefiner:
    """
    This class is mostly unchanged, but the initialization logic is now correctly
    invoked inside each worker process.
    """

    def __init__(self, CLIENT_CONFIG, gpu_id=0):
        self.refine_templetes = ["""Goal: [State the main question to be answered in one simple sentence.]
Observation: [List the key numbers and the relationships between them from the problem statement.]
Reasoning: [Show the step-by-step calculation process. Each step should be a clear mathematical operation.]
Conclusion: [State the final answer clearly.]
"""]
        if CLIENT_CONFIG['client_type'] == 'openai':
            # Key logic: port calculation is now done in worker_initializer
            self.client = OpenAIClient(config=CLIENT_CONFIG)
        else:
            raise ValueError(f"Client type '{CLIENT_CONFIG['client_type']}' not supported.")

    def refine_hint(self, question: str, hint: str, reference_answer: str, task: str):
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
            in_context_example = self.client.get_completion(
                prompt_ic % hint,
                system_prompt=system_prompt,
                max_tokens=5000
            )
            if 'chart' in task or 'math' in task:
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
            print(f"Error occurred while processing '{question}': {e}")
            return hint


# ---------------- Multiprocessing worker functions (fixed) ----------------

refiner_instance = None


def worker_initializer(base_client_config):
    """
    Initialization function called when each worker process starts (fixed version).
    Creates an independent, differently configured Refiner instance per process.
    """
    global refiner_instance

    # Key change: get the unique ID of the current worker process (starting from 1)
    # This is the variable we use to simulate gpu_id
    worker_id = multiprocessing.current_process()._identity[0] - 1

    # Create a deep copy of the config to avoid interference between processes
    worker_config = deepcopy(base_client_config)

    # Key change: implement your port calculation logic
    if worker_config.get('init_port') is not None and worker_config.get('num_server') is not None:
        num_server = int(worker_config['num_server'])
        # server_id decides which port to use
        server_id = worker_id % num_server
        port = worker_config['init_port'] + server_id

        # Format api_base to assign a fixed port for this process
        worker_config['api_base'] = worker_config['api_base'] % str(port)

        print(f"Process {os.getpid()} (Worker-{worker_id}) initializing... connecting to {worker_config['api_base']}")
    else:
        print(f"Process {os.getpid()} (Worker-{worker_id}) initializing... using default api_base")

    # Use the customized config for this specific process to create the instance
    refiner_instance = ContextRefiner(worker_config, gpu_id=worker_id)


def process_item_worker(item):
    """Function executed by a single worker process (unchanged)"""
    global refiner_instance
    if refiner_instance is None:
        raise Exception("Refiner has not been initialized in the worker process!")

    new_hint = refiner_instance.refine_hint(
        question=item['question'],
        hint=item['hint'],
        reference_answer=item['answer'],
        task='math'
    )
    item['hint'] = new_hint
    return item


# ---------------- Main logic ----------------
def main():
    # Configuration that contains information about ports and number of servers
    from config import CLIENT_CONFIG
    input_filename = '/path/to/data/lm_math/json/train.json'
    output_filename = '/path/to/data/lm_math/json/train_refine.json'

    NUM_PROCESSES = 64
    print(f"Using {NUM_PROCESSES} processes and distributing requests to {CLIENT_CONFIG['num_server']} servers...")

    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: input file '{input_filename}' not found.")
        return

    processed_data = []

    # Key: pass the base configuration to each process's initializer
    with multiprocessing.Pool(
        processes=NUM_PROCESSES,
        initializer=worker_initializer,
        initargs=(CLIENT_CONFIG,)
    ) as pool:
        with tqdm(total=len(data), desc="Processing JSON in parallel") as pbar:
            for result in pool.imap_unordered(process_item_worker, data):
                processed_data.append(result)
                pbar.update(1)

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"\nProcessing completed! Results saved to '{output_filename}'.")


if __name__ == "__main__":
    main()
