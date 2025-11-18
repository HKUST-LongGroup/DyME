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

    def __init__(self, CLIENT_CONFIG, gpu_id=0):
        self.refine_templetes = ["""Goal: [State the user's objective, e.g., Find the year with the highest sales]
Observation: [List key data points from the chart, e.g., 2020: 150, 2021: 200, 2022: 180]
Reasoning: [State the logical step, e.g., Compare the values. 200 is the maximum.]
Conclusion: [Draw the conclusion, e.g., The year with the highest sales was 2021.]
"""]
        if CLIENT_CONFIG['client_type'] == 'openai':

            self.client = OpenAIClient(config=CLIENT_CONFIG)
        else:
            raise ValueError(f"Client type '{CLIENT_CONFIG['client_type']}' not supported.")

    def refine_hint(self, question: str, hint: str, reference_answer: str, task: str):
        if not hint:
            return hint
        system_prompt = None
        if 'chart' in task:
            system_prompt = 'You are a seasoned professional in the field of chart analysis...'
        else:
            raise Exception('Unknown expert task')
        try:
            in_context_example = self.client.get_completion(prompt_ic % hint, system_prompt=system_prompt,
                                                            max_tokens=5000)
            if 'chart' in task:
                evaluation_prompt = prompt_refine % (in_context_example, question, reference_answer,
                                                     self.refine_templetes[0])
                output = self.client.get_completion(evaluation_prompt, system_prompt=system_prompt, max_tokens=1000)
                return output
            else:
                raise ValueError(f"Task '{task}' not supported for thinking reward.")
        except Exception as e:
            print(f"Error occurred while processing '{question}': {e}")
            return hint


refiner_instance = None


def worker_initializer(base_client_config):
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
        task='chart'
    )
    item['hint'] = new_hint
    return item


# ---------------- Main logic ----------------
def main():
    # Configuration that contains port and server count information
    from config import CLIENT_CONFIG
    input_filename = '/path/to/chartqa_output/json/train_new.json'
    output_filename = '/path/to/chartqa_output/json/train_new_prerefine.json'

    NUM_PROCESSES = 64
    print(f"Using {NUM_PROCESSES} processes and distributing requests to {CLIENT_CONFIG['num_server']} servers...")

    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: input file '{input_filename}' not found.")
        return

    processed_data = []

    # Key point: pass the base configuration to the initializer of each process
    with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=worker_initializer,
                              initargs=(CLIENT_CONFIG,)) as pool:
        with tqdm(total=len(data), desc="Processing JSON in parallel") as pbar:
            for result in pool.imap_unordered(process_item_worker, data):
                processed_data.append(result)
                pbar.update(1)

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"\nProcessing completed! Results saved to '{output_filename}'.")


if __name__ == "__main__":
    main()
