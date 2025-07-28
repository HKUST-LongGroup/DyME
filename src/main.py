# train_grpo.py
import concurrent.futures
import json
import os

import torch
import wandb
from accelerate import Accelerator
from tqdm import tqdm
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from trl import GRPOConfig, AutoModelForCausalLMWithValueHead
from PIL import Image as PILImage
from transformers import Idefics3Model
from client import get_smol_reason_reward, get_smol_acc_reward, get_smol_expert_reward, get_mllm_reward, get_format_reward, get_other_express
from datasets import Features, Image, Value, Dataset, load_dataset
from trainer import MyGRPOTrainer

wandb.login(key="a07e39e43f1a318a12a9b43a73d79d6ad4f4d2e2")

accelerator = Accelerator(mixed_precision="bf16", log_with="wandb")
DEVICE = accelerator.device

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='chart', help='task')
parser.add_argument('--train_json', type=str, default='SmolVLM_GRPO', help='model')
parser.add_argument('--image_dir', type=str, default='SmolVLM_DyME', help='image_dir')
parser.add_argument('--output_dir', type=str, default='/home/jliugj/HDD/academic/checkpoints/', help='output_dir')
parser.add_argument('--model', type=str, default='HuggingFaceTB/SmolVLM-500M-Instruct', help='model_id')

def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
      image = example["image"]
      if isinstance(image, str):
        image = PILImage.open(image)
      if image.mode != 'RGB':
        image = image.convert('RGB')
      question = example["question"]
      answer = example["answer"]
      if answer is not None:
          messages = [
              {
                  "role": "user",
                  "content": [
                      {"type": "image"},
                      {"type": "text", "text": question}
                  ]
              },
              {
                  "role": "assistant",
                  "content": [
                      {"type": "text", "text": answer}
                  ]
              }
          ]
          text = processor.apply_chat_template(messages, add_generation_prompt=False)
          texts.append(text.strip())
      else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(text.strip())

      images.append(image)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

def new_forward(self, input_embs, attention_mask, input_ids=None, images=None):
    if images is not None:
        img_features = self.vision_encoder(images, self.device)
        img_embs = self.mlp(img_features)
        return img_embs
    outputs = self.language_model(inputs_embeds=input_embs, attention_mask=attention_mask, labels=input_ids)
    return outputs

############# SLAKE

# ds = ds_new
#
# features = Features({
#     "image": Image(),  # 告诉HF这列是图像
#     "prompt": Value("string"),
#     "knowledge": Value("string"),
#     "answer": Value("string"),
#     "tp": Value("string"),
#     "only_q": Value("string")
#     # 其他需要的字段
# })

# dataset = Dataset.from_list(ds, features=features)

def deal_llava_med_slake(max_completion_length):
    # data_path = '/home/jliugj/HDD/academic/database/SLAKE/'
    image_dir = '/home/jliugj/HDD/academic/database/'
    # if not os.path.exists(image_dir):
    #     data_path = '/project/longgroup/jiazhen/dataset/moe_efficient/SLAKE'
    #     image_dir = '/project/longgroup/jiazhen/dataset/moe_efficient/SLAKE'
    # ds = json.load(open(os.path.join(data_path, 'train_all_knowledge_new.json'), 'r'))
    ds = json.load(open('/home/jliugj/HDD/code/moe_efficient/exp/AddThink/accept_train_mp_with_reason_slake_hint.json', 'r', encoding='utf-8'))
    ds_new = []
    for dd in tqdm(ds):
        assert 'slake' in dd['image']
            # continue
        if not os.path.exists(dd['image']):
            image_path = os.path.join(image_dir, dd['image'])
        else:
            image_path = dd['image']
        new_dt = {}
        new_dt['image'] = image_path
        question = dd['conversations'][0]['value'].split('<image>\n')[1]
        # question = f'Generate a context for the question "{question}" that includes the correct answer while also serving as a prompt to guide downstream models.'
        new_dt['only_q'] = question

        question = f"""{question} Think step by step and then answer the question."""

        new_dt['prompt'] = question
        new_dt['knowledge'] = dd['hint']
        new_dt['answer'] = dd['conversations'][1]['value']
        new_dt['tp'] = dd['answer_type']

        # if new_dt['tp'] != 'CLOSED':
        #     new_dt['knowledge'] = new_dt['answer']

        ds_new.append(new_dt)
    return ds_new

def deal_llava_med_alignment_data(max_completion_length):
    json_path = '/home/jliugj/HDD/academic/database/LLaVA-Med/data/train_align.json'
    ds = json.load(open(json_path, 'r'))
    ds_new = []
    for dd in tqdm(ds):
        image_path = dd['image_path']
        turns = dd['conversations']
        i = 0
        while i < len(turns):
            new_dt = {}
            new_dt['image'] = image_path
            question = turns[i]['value'].replace('<image>\n', '').replace('<image>', '').replace('\n<image>', '')
            answer = turns[i + 1]['value'][:max_completion_length]
            new_dt['only_q'] = question
            new_dt['prompt'] = question
            new_dt['knowledge'] = answer
            new_dt['answer'] = answer
            new_dt['tp'] = 'OPEN'
            i += 2

            ds_new.append(new_dt)
    return ds_new

def deal_llava_med_instruct_data(max_completion_length):
    json_path = '/home/jliugj/HDD/academic/database/LLaVA-Med/data/train_instruct_deal.json'
    ds = json.load(open(json_path, 'r'))
    ds_new = []
    for dd in tqdm(ds):
        image_path = dd['image_path']
        turns = dd['conversations']
        i = 0
        while i < len(turns):
            new_dt = {}
            new_dt['image'] = image_path

            question = turns[i]['value'].replace('<image>\n', '').replace('<image>', '').replace('\n<image>', '')
            answer = turns[i + 1]['value'][:max_completion_length]
            new_dt['only_q'] = question
            new_dt['prompt'] = question
            new_dt['knowledge'] = answer
            new_dt['answer'] = answer
            new_dt['tp'] = 'OPEN'
            i += 2

            ds_new.append(new_dt)
    return ds_new

def deal_chart_alignment_data(max_completion_length):
    json1 = '/home/jliugj/HDD/academic/database/chart/chart2text/train.json'
    json2 = '/home/jliugj/HDD/academic/database/chart/dvqa/train.json'
    json3 = '/home/jliugj/HDD/academic/database/chart/MMC_instruct/train.json'
    json4 = '/home/jliugj/HDD/academic/database/chart/SciGraphQA/train.json'

    data1 = json.load(open(json1, 'r'))
    data2 = json.load(open(json2, 'r'))
    data3 = json.load(open(json3, 'r'))
    data4 = json.load(open(json4, 'r'))

    data = data1 + data2 + data3 + data4

    for i, d in enumerate(data):
        data[i]['answer'] = d['answer'][:max_completion_length]
        data[i]['answer'] = data[i]['answer'].replace('<image>', '')
        # data[i]['pro']

    return data

def deal_chart_sft_data(max_completion_length):
    json1 = '/home/jliugj/HDD/academic/database/chart/chartQA/train.json'

    data1 = json.load(open(json1, 'r'))
    data = data1
    for i in range(len(data)):
        ans = data[i]['answer'].lower()
        know = data[i]['knowledge'].lower()[:10]
        question = data[i]['only_q']
        question = f"""{question} Think step by step and then answer the question."""
        data[i]['prompt'] = question
        if 'yes' in ans or 'no' in ans:
            if 'yes' not in know and 'no' not in know:
                know = data[i]['knowledge'].split(' ')
                know[0] = know[0].lower()
                know = ' '.join(know)
                if 'yes' in ans:
                    data[i]['knowledge'] = 'Yes, ' + know
                else:
                    data[i]['knowledge'] = 'No, ' + know
                # print(data[i]['knowledge'])
        # data[i]['answer'] = data[i]['knowledge']
    return data

def deal_chart_rl_data(max_completion_length):
    json1 = '/home/jliugj/HDD/academic/database/chart/chartQA/train.json'
    # json1 = '/home/jliugj/HDD/academic/database/chart/chartQA/val.json'
    json1 = '/home/jliugj/HDD/academic/database/chart/chartQA/accept_train_mp.json'
    need_change_path = False
    new_dir = None
    if not os.path.exists(json1):
        # new_dir = '/project/longgroup/jiazhen/dataset/moe_efficient/chartQA'
        json1 = os.path.join(new_dir, 'accept_train_mp.json')
        # json1 = os.path.join(new_dir, 'val.json')
        need_change_path = True
    data1 = json.load(open(json1, 'r'))
    data = []
    for d in data1:
        if d['human_or_machine'] == 0:
            del d['human_or_machine']
            data.append(d)

    for i in range(len(data)):
        data[i]['answer'] = data[i]['answer'].strip()
        ans = data[i]['answer'].lower()
        know = data[i]['knowledge'].lower()[:10]
        question = data[i]['only_q']

        prompt = (
                "For the question below, follow the following instructions:\n"
                + "-First output your step-by-step reasoning, then provide your answer beginning with “Answer:”.\n"
                + "-The answer should contain as few words as possible.\n"
                + "-Don't paraphrase or reformat the text you see in the image.\n"
                + "-Answer a binary question with Yes or No.\n"
                + "-When asked to give a numerical value, provide a number like 2 instead of Two.\n"
                + "-If the final answer has two or more items, provide it in the list format like [1, 2].\n"
                + "-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.\n"
                + "-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.\n"
                + "-Don't include any units in the answer.\n"
                + "-Do not include any full stops at the end of the answer.\n"
                + "-Try to include the full label from the graph when asked about an entity.\n"
                + "Question: "
        )

        question = prompt + question
        data[i]['prompt'] = question
        if need_change_path:
            image_name = os.path.basename(data[i]['image'])
            data[i]['image'] = os.path.join(new_dir, 'images',image_name)

        data[i]['knowledge'] = None
    return data

def deal_math_rl_data(max_completion_length):
    json1 = '/home/jliugj/HDD/academic/database/math/geo170k/train.json'
    need_change_path = False
    new_dir = None
    if not os.path.exists(json1):
        new_dir = '/project/longgroup/jiazhen/dataset/moe_efficient/Geometry3K'
        json1 = os.path.join(new_dir, 'train.json')
        need_change_path = True
    data1 = json.load(open(json1, 'r'))
    data = data1
    for i in range(len(data)):
        data[i]['answer'] = data[i]['answer'].strip()
        question = data[i]['only_q']
        data[i]['prompt'] = f"""{question} Think step by step and then answer the question."""
        if need_change_path:
            image_name = os.path.basename(data[i]['image'])
            data[i]['image'] = os.path.join(new_dir, 'images', image_name)

    return data

def deal_math_sft_data(max_completion_length):
    json1 = '/home/jliugj/HDD/academic/database/math/math360/train.json'
    need_change_path = False
    new_dir = None
    if not os.path.exists(json1):
        new_dir = '/project/longgroup/jiazhen/dataset/moe_efficient/math360'
        json1 = os.path.join(new_dir, 'train.json')
        need_change_path = True
    data1 = json.load(open(json1, 'r'))
    data = data1
    if need_change_path:
        for i in range(len(data)):
            data[i]['image'] = data[i]['image'].replace('/home/jliugj/HDD/academic/database/math/math360', new_dir)

    json2 = '/home/jliugj/HDD/academic/database/math/MetaMathQA/train.json'
    data2 = json.load(open(json2, 'r'))

    data = data + data2
    for i in range(len(data)):
        data[i]['answer'] = data[i]['answer'][:max_completion_length]
        data[i]['knowledge'] = data[i]['knowledge'][:max_completion_length]

    return data


def define_task_data_func(task):
    if 'medical' in task:
        if '_alignment' in task:
            return deal_llava_med_alignment_data
        elif '_instruct' in task:
            return deal_llava_med_instruct_data
        elif '_rl' in task:
            return deal_llava_med_slake
    elif 'chart' in task:
        if '_alignment' in task:
            return deal_chart_alignment_data
        elif '_rl' in task:
            return deal_chart_rl_data
        elif '_sft' in task:
            return deal_chart_sft_data
    elif 'math' in task:
        if '_rl' in task:
            return deal_math_rl_data
        elif 'sft' in task:
            return deal_math_sft_data

# def reward_format_mllm(data_dt, gpu_id, add_know=True, num_threads=8):
#     response = data_dt['response']
#     question = data_dt['prompt']
#     image_path = data_dt['image']
#     answer = data_dt['answer']
#     standard_answer = data_dt['standard_answer']
#     answer_tp = data_dt['tp']
#     add_knowledge = True
#     # 包装函数用于参数解包
#     def process_item(args):
#         r, q, a, i, gpu_id, _, a_tp = args
#         return get_reward(r, q, a, i, gpu_id, add_knowledge, tp=a_tp)
#
#     # 创建参数元组列表
#     task_args = zip(response, question, standard_answer, image_path, [gpu_id]*len(response), [add_know]*len(response), answer_tp)
#
#     # 使用线程池并行处理
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
#         rewards = list(executor.map(process_item, task_args))
#
#     add_knowledge2 = False
#     task_args = zip(response, question, standard_answer, image_path, [gpu_id] * len(response),
#                     [add_know] * len(response), answer_tp)
#
#     # 包装函数用于参数解包
#     def process_item2(args):
#         r, q, a, i, gpu_id, _, a_tp = args
#         return get_reward(r, q, a, i, gpu_id, add_knowledge2, tp=a_tp)
#
#     # 使用线程池并行处理
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
#         rewards2 = list(executor.map(process_item2, task_args))
#
#     # reward2是大模型的原始回答
#     thr = 0.8
#     for i in range(len(rewards)):
#         if rewards2[i] >= thr and rewards[i] >= thr:
#             rewards[i] = 0
#         elif rewards2[i] < thr and rewards[i] < thr:
#             rewards[i] = -1
#         elif rewards2[i] >= thr and rewards[i] < thr:
#             rewards[i] = -2
#         else:
#             rewards[i] = 1
#
#     return rewards

def reward_smol_acc(data_dt, gpu_id, num_threads=8):
    response = data_dt['response']
    question = data_dt['prompt']
    image_path = data_dt['image']
    answer = data_dt['answer']
    standard_answer = data_dt['standard_answer']
    answer_tp = data_dt['tp'] if 'tp' in data_dt else 'None'
    task = data_dt['task']
    # 包装函数用于参数解包
    def process_item(args):
        r, q, a, i, gpu_id, a_tp, task = args
        return get_smol_acc_reward(r, q, a, i, gpu_id=gpu_id, tp=a_tp, task=task)

    # 创建参数元组列表
    task_args = zip(response, question, standard_answer, image_path,
                    [gpu_id]*len(response), answer_tp, [task]*len(response))

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        rewards = list(executor.map(process_item, task_args))

    return rewards

def reward_smol_final_acc(data_dt, gpu_id, num_threads=8):
    response = data_dt['response']
    question = data_dt['prompt']
    image_path = data_dt['image']
    hints = data_dt['hints']
    standard_answer = data_dt['standard_answer']
    answer_tp = data_dt['tp'] if 'tp' in data_dt else ['None'] * len(response)
    task_list = data_dt['task']
    # 包装函数用于参数解包
    def process_item(args):
        r, q, a, i, gpu_id, a_tp, task = args
        return get_smol_acc_reward(r, q, a, i, gpu_id=gpu_id, tp=a_tp, task=task)

    # 创建参数元组列表
    task_args = zip(response, question, standard_answer, image_path,
                    [gpu_id]*len(response), answer_tp, task_list)

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        rewards = list(executor.map(process_item, task_args))

    return rewards


def reward_smol_reason(data_dt, gpu_id, num_threads=8):
    response = data_dt['response']
    question = data_dt['prompt']
    image_path = data_dt['image']
    hints = data_dt['hints']
    standard_answer = data_dt['standard_answer']
    tasks = data_dt['task']
    # 包装函数用于参数解包
    def process_item(args):
        r, q, a, i, h, t, gpu_id = args
        return get_smol_reason_reward(r, q, a, i, h, gpu_id=gpu_id, expert_task=t)

    # 创建参数元组列表
    task_args = zip(response, question, standard_answer, image_path, hints, tasks,
                    [gpu_id]*len(response))

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        rewards = list(executor.map(process_item, task_args))

    return rewards

def reward_expert(data_dt, gpu_id, num_threads=8):
    response = data_dt['response']
    question = data_dt['prompt']
    image_path = data_dt['image']
    hints = data_dt['hints']
    standard_answer = data_dt['standard_answer']
    answer_tp = data_dt['tp']
    task = data_dt['task']
    # 包装函数用于参数解包
    def process_item(args):
        r, q, a, i, gpu_id, a_tp, task = args
        return get_smol_expert_reward(r, q, a, i, task, gpu_id=gpu_id)

    # 创建参数元组列表
    task_args = zip(response, question, standard_answer, image_path,
                    [gpu_id]*len(response), answer_tp, [task]*len(response))

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        rewards = list(executor.map(process_item, task_args))

    return rewards

def reward_mllm(data_dt, gpu_id, num_threads=8):
    response = data_dt['response']
    question = data_dt['prompt']
    image_path = data_dt['image']
    hints = data_dt['hints']
    standard_answer = data_dt['standard_answer']
    answer_tp = data_dt['tp']
    task = data_dt['task']
    # 包装函数用于参数解包
    def process_item(args):
        r, q, a, i, gpu_id, a_tp, task = args
        return get_mllm_reward(r, q, a, i, gpu_id=gpu_id)

    # 创建参数元组列表
    task_args = zip(response, question, standard_answer, image_path,
                    [gpu_id]*len(response), answer_tp, [task]*len(response))

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        rewards = list(executor.map(process_item, task_args))

    return rewards

def reward_format(data_dt, gpu_id, num_threads=8):
    response = data_dt['response']
    question = data_dt['prompt']
    image_path = data_dt['image']
    hints = data_dt['hints']
    standard_answer = data_dt['standard_answer']
    # answer_tp = data_dt['tp']
    task = data_dt['task']
    # 包装函数用于参数解包
    def process_item(args):
        r, gpu_id = args
        return get_format_reward(r, gpu_id=gpu_id)

    # 创建参数元组列表
    task_args = zip(response, [gpu_id]*len(response))

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        rewards = list(executor.map(process_item, task_args))

    return rewards



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['connector', 'vision_model', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def create_features():
    """创建数据集所需的特征描述"""
    return

def create_dataset(data_func, features, **kwargs):
    """调用数据处理函数并构建数据集"""
    ds = data_func(**kwargs)
    return Dataset.from_list(ds, features=features)

def create_trainer(model, dataset, processor, training_args, is_sft, start_template, end_template, task, eval_dataset=None):
    """创建MyGRPOTrainer实例"""
    return MyGRPOTrainer(
        model=model,
        reward_funcs=[reward_format, reward_smol_reason, reward_smol_final_acc],
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        attn_implementation='flash_attention_2',
        processing_func=collate_fn,  # image, question, answer,
        end_template=end_template,
        start_template=start_template,
        prompt_template=None,
        is_sft=is_sft,
        task_name=task,
        eval_dataset=eval_dataset,
    )

def define_task(task):
    """定义任务"""
    if task == 'medical':
        return ['medical_alignment', 'medical_instruct', 'medical_rl']
    elif task == 'chart':
        return [None, 'chart_sft', 'chart_rl']
        return ['chart_alignment', 'chart_rl', 'chart_rl']
    elif task == 'math':
        return [None, 'math_sft', 'math_rl']

model_args = {"trust_remote_code": True, 'ignore_mismatched_sizes': True,
            "torch_dtype":torch.bfloat16}

output_pos = parser.add_argument_group('output position')
need_path_change = False
if not os.path.exists(output_pos):
    output_pos = 'checkpoints/'
    need_path_change = True
model = None
raw_model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
# raw_model_id = "HuggingFaceTB/SmolVLM-500M-Base"
processor = AutoProcessor.from_pretrained(raw_model_id)
processor.tokenizer.padding_side = "left"
processor.image_processor.size['longest_edge'] = 512 * 4
image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")]

task = parser.parse_args().task
tasks = define_task(task)

max_completion_length = 512
model_id = parser.parse_args().model_id

is_sft = False
features = create_features()
if is_sft:
    is_pretrain = True

    # 定义输出格式模板
    end_template = None # 只有在RL阶段才需要
    task = tasks[0]

    # 模型与处理器初始化
    if task is not None:
        output_dir = os.path.join(output_pos, task + "_long_answer_base")

        deal_data = define_task_data_func(task)
        model = Idefics3ForConditionalGeneration.from_pretrained(model_id, **model_args)
        # 第一阶段：预训练
        pretrain_args = GRPOConfig(
            output_dir=output_dir,
            logging_steps=1,
            num_generations=1,
            max_completion_length=max_completion_length,
            per_device_train_batch_size=6,
            gradient_accumulation_steps=12,
            num_train_epochs=1,
            learning_rate=5e-5,
            ddp_find_unused_parameters=True,
            save_steps=200
        )
        dataset1 = create_dataset(deal_data, features, max_completion_length=max_completion_length)
        trainer = create_trainer(model, dataset1, processor, pretrain_args, is_sft, end_template, task)
        trainer.train(resume_from_checkpoint=True)

    # 第二阶段：监督微调（SFT）
    task = tasks[1]
    if model is None:
        if 'chart' in task:
            model_id = '/home/jliugj/HDD/academic/checkpoints/chart_alignment_long_answer_instruct/checkpoint-7200'  # chart
            if need_path_change:
                model_id = 'checkpoints/chart_alignment_long_answer_instruct/checkpoint-7200'
        print(model_id)
        model = Idefics3ForConditionalGeneration.from_pretrained(model_id, **model_args)

    deal_data = define_task_data_func(task)
    output_dir = os.path.join(output_pos, task + "_long_answer_instruct")
    sft_args = GRPOConfig(
        output_dir=output_dir,
        logging_steps=1,
        num_generations=1,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=6,
        gradient_accumulation_steps=10,
        num_train_epochs=1,
        learning_rate=5e-5,
        ddp_find_unused_parameters=True,
        save_steps=100,
    )
    dataset2 = create_dataset(deal_data, features, max_completion_length=max_completion_length)
    # dataset2数据集合并到dataset_target

    trainer = create_trainer(model, dataset2, processor, sft_args, is_sft, end_template, task)
    trainer.train(resume_from_checkpoint=True)

#######################################################################################

task = tasks[2]
output_dir = os.path.join(output_pos, task + "_long_answer_rl-512")
max_completion_length = 256

from peft import LoraConfig, get_peft_model



if model is None:
    if 'medical' in task:
        # model_id = '/home/jliugj/HDD/academic/checkpoints/medical_long_answer_instruct/checkpoint-316/'
        model_id = raw_model_id
        if need_path_change:
            model_id = 'checkpoints/medical_long_answer_instruct/checkpoint-316'
            # model_id = 'checkpoints/medical_long_answer_base/checkpoint-1622'
    elif 'chart' in task:
        model_id = '/export/jliugj/academic/checkpoints/chart_sft_long_answer_instruct/checkpoint-177'
        # model_id = '/export/jliugj/academic/checkpoints/chart_alignment_long_answer_instruct/checkpoint-7200'
        if need_path_change:
            model_id = 'checkpoints/chart_alignment_long_answer_instruct/checkpoint-7200'
        model_id = raw_model_id
        # model_id = '/home/jliugj/HDD/academic/checkpoints/chart_rl_long_answer_rl/checkpoint-200'
    elif 'math' in task:
        pass
        # model_id = '/export/jliugj/academic/checkpoints/math_sft_long_answer_instruct/checkpoint-400'
    print(model_id)
    model = Idefics3ForConditionalGeneration.from_pretrained(model_id,)
    # model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id, peft_config=lora_config,)
model.model.vision_model.requires_grad_(False)
# lora_config = LoraConfig(
#             r=64,
#             lora_alpha=128,
#             target_modules=find_all_linear_names(model),
#             lora_dropout=0.05,
#             bias='none',
#             task_type="CAUSAL_LM",
#         )
# model = get_peft_model(model, peft_config=lora_config)

# 定义输出格式模板
start_template = """Answer: %s. Further check whether it's correct: <end_of_utterance>"""
# end_template = """The above reasoning and information in the image yields the following answer: %s<end_of_utterance>"""
# end_template = """Answer: %s<end_of_utterance>"""
end_template = """Answer: %s<end_of_utterance>"""
# end_template = """Therefore, answer: %s<end_of_utterance>"""



# end_template = """%s<end_of_utterance>"""

# end_template = """The answer is based on the following reasoning: Therefore, the final answer is: %s<end_of_utterance>"""

rl_args = GRPOConfig(
    output_dir=output_dir,
    logging_steps=1,
    num_generations=4,
    max_completion_length=max_completion_length,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    bf16=True,
    learning_rate=5e-5,
    gradient_checkpointing=False,
    ddp_find_unused_parameters=False,
    max_grad_norm=1.0,
    save_steps=100,
    weight_decay=0.01,
    warmup_steps=0,
    eval_strategy="steps",   # 可选 "steps" 或 "epoch"
    eval_steps=100,                # 每 100 步评估一次
    beta=0,
    num_iterations=1,
    loss_type='grpo'
)

deal_data = define_task_data_func(task)
dataset_rl = create_dataset(deal_data, features, max_completion_length=max_completion_length)
if 'chart' in task:
    eval_dataset = load_dataset("HuggingFaceM4/ChartQA")['test']
elif 'math' in task:
    eval_dataset = load_dataset('CaraJ/MathVerse-lmmseval', 'testmini')['testmini']
    # eval_dataset = load_dataset('AI4Math/MathVista', 'testmini')['testmini']
elif 'medical' in task:
    full_dataset = load_dataset("BoKelvin/SLAKE")['test']
    eval_dataset = []
    for d in full_dataset:
        if d['q_lang'] == 'zh':
            continue
        # idd = random.randint(0, len(ds))
        image_path = os.path.join('/home/jliugj/HDD/academic/code/LLaVA/playground/data/SLAKE/imgs', d['img_name'])
        question = d['question']
        prompt = question

        answer = d['answer']
        tp = d['answer_type']

        eval_dataset.append(
            {'image': image_path, 'prompt': prompt, 'answer': answer, 'tp': tp, 'question': question, 'answer_type': tp, })

# eval_dataset = eval_dataset.select(range(1000, 1100))
trainer = create_trainer(model, dataset_rl, processor, rl_args, False, start_template, end_template, task, eval_dataset)
# trainer.train(resume_from_checkpoint='/home/jliugj/HDD/academic/checkpoints/chart_rl_long_answer_rl/checkpoint-1680')
trainer.train(resume_from_checkpoint=False)

