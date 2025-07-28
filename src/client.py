import base64
import re

import openai
import requests

from eval.chart.util import eval_one_chart
from eval.eval_metric import evaluate_single_sample

import spacy

gpt_base_server_url = 'http://locpu2.cse.ust.hk'
expert_base_server_url = 'http://locpu2.cse.ust.hk'
mllm_base_server_url = 'http://locpu4.cse.ust.hk'
# #
# gpt_base_server_url = 'http://127.0.0.1'
# expert_base_server_url = 'http://127.0.0.1'
# mllm_base_server_url = 'http://127.0.0.1'

# 加载英文的小模型
nlp = spacy.load("en_core_web_sm")

from openai import OpenAI

# OpenAI 配置信息
args_dict = {
    'api_key': 'sk-VcoCz5NCv9U6CiSR881cD8909a2e47AaAbEaFa37395c6b2d',
        # 'api_base': 'https://api.chatanywhere.tech/v1',
    'api_base': 'https://api.linkapi.org/v1',

}

openai.api_key = args_dict['api_key']
openai.api_base = args_dict['api_base']
open_client = OpenAI(
    api_key=args_dict['api_key'],  # 必填
    base_url=args_dict['api_base']  # 仅第三方服务需设置
)

def get_openai_output(text: str, max_tokens: int = 1024):
    """
    调用OpenAI ChatCompletion接口并返回结果
    """
    while True:
        try:
            response = open_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {
                        'role': 'user',
                        'content': text,
                    }
                ],
                # temperature=0.2,
                max_tokens=max_tokens,
            )
            # print(response)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            # time.sleep(1)  # 暂停一秒后重试，可根据需要自行调整
            return None

# Function to encode the image as Base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_response(knowledge, question, image_path, gpu_id=0, add_knowledge=True, add_condition=True):
    answer_pred = None
    expert_task = "chart"
    if add_knowledge:
        prompt = f'''For the question “{question}” there is a reference answer: “{knowledge}” 

This reference answer serves only as a guide; you are free to refine or improve upon it if necessary. Please produce your best possible answer, using both the information from the reference answer and the image. The question is: “{question}”'''
        #ls = knowledge.split('.')
        #answer_pred = ls[0]
        # context_pred = '.'.join(ls[1:])
        #prompt = f"""For the question “{question}”, a reference answer is provided: “{answer_pred}”. 
#Below is the hint for this answer: {context_pred}

#Based on the given image, please judge whether the reference answer is correct for the question. 

#If the reference answer is correct, please output “correct” directly. Otherwise, provide an improved answer.
#Your output:
#"""
        # prompt = "Please answer my question based on both the provided image and the following context: " + knowledge + "\n\n" + question
        # prompt = f"""For the question “{question}”,
        # Below is the hint for answering: {context_pred}
        #
        # Answer the question based on the hint. Your output:
        # """
    else:
        prompt = f'{question}'
    add_p = "\nAnswer the question using a single word or phrase."
    if 'chart' not in expert_task:
        add_p = "\nAnswer the question shortly."
    prompt = prompt + add_p if add_condition else prompt
    client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{mllm_base_server_url}:{5000 + gpu_id}/v1')
    # model_name = client.models.list().data[0].id

    base64_image = encode_image_to_base64(image_path)

    response = client.chat.completions.create(
        model='llava-hf/llava-1.5-7b-hf',
        messages=[
            {'role': 'system', 'content': "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."},
            {
                'role': 'user',
                'content': [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {'type': 'text', 'text': prompt},
                ],
            }],
        temperature=0.

    )
    r = response.choices[0].message.content.strip()
    if answer_pred is not None and 'correct' in r.lower():
        r = answer_pred
    # if gpu_id == 0:
    #     print(prompt + '\n\n$$$$$\n' + r)
    return r

def judge_answer(question, answer, pred_answer, gpu_id=0):
    client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{gpt_base_server_url}:{3000 + gpu_id}/v1')
    model_name = client.models.list().data[0].id

    prompt = 'Imagine you are an intelligent teacher. Thoroughly read the question, reference answer and the prediction answer to ensure a clear understanding of the information provided. Assess the correctness of the predictions. '
    prompt += 'If the prediction answer does not conflict with the reference answer, please generate “correct”. If the prediction answer conflict with the reference answer, please generate “incorrect”. \n\nQuestion: '
    prompt += question
    prompt += '\nReference answer: '
    prompt += answer
    prompt += '\nPrediction: '
    prompt += pred_answer
    prompt += '\nDo not output any explanation and description. Your Output:'

    # print(model_name)
    # base64_image = encode_image_to_base64(image_path)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.
    )
    response = response.choices[0].message.content.strip()
    if 'incorrect' in response.lower():
        reward = 0
    elif 'correct' in response.lower():
        reward = 1
    else:
        reward = 0
    reward = float(reward)



    return reward

def get_other_express(question, answer, check, image_path, hint, gpu_id, expert_task='chart'):
    DEFAULT_TIMEOUT = 30
    # return ""
    if not check:
        return ""

    try:
        if hint is not None:
            data_figure = hint
            # return data_figure
        else:
            sentences = [""]
            # print(sentences)
            url = f"{expert_base_server_url}:8000/predict"
            headers = {"Content-Type": "application/json"}
            payload = {
                "gpu_id": gpu_id,
                "texts": sentences,
                "image_path": image_path
            }

            response = requests.post(url, headers=headers, json=payload)
            # 解析返回的 JSON 数据
            data = response.json()

            # 提取分数
            scores = data.get("scores", {})
            data_figure = data.get("resp", None)

    except:
        expert_score = 0
        data_figure = ""

    client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{gpt_base_server_url}:{3000 + gpu_id}/v1')
    model_name = client.models.list().data[0].id

    if 'chart' in expert_task:

#         prompt = f"""请生成一段进行分析的英文，用于承接问题和答案，揭示问题Question“{question}”的答案Answer为{answer}。要用英文表达。
#     注意，回答这个问题需要分析图像中的数据，具体的数据内容Data_figure包括“{data_figure}”。你需要从中选择合适的数据，用于构造这段分析。注意如果数据与给定的答案不一致，以答案为准，直接根据给定的答案修正数据，不要进行有关修正的说明。
#     这里的分析要体现问题、答案的特点，以及与图像的关系，反映图像中的具体数据与内容。请输出内容具有逻辑的段落，步骤清晰，结构完整。
#     格式需要统一。
#     以下是一个例子供你参考，你可以按这个逻辑去构造你的分析：
#     -----
#     Example:
#
# Question: What is the difference between the values of 2015 and 2016?
# Answer: 133,915.
# Data_figure: 2015: 162,915 | 2016: 29,000 | 2017: 36,700 | 2018: 40,000 | 2019: 46,000 | 2020: 46,000
# **Your response:**
# To find the difference between the values of 2015 and 2016, we need to subtract the number of asylum applications in 2016 from the number in 2015.
#
# From the chart:
# - The number of asylum applications in 2015 is 162,915.
# - The number of asylum applications in 2016 is 29,000.
#
# Subtracting these values:
# \[ 162,915 - 29,000 = 133,915 \]
#
# Conclusion: The difference between the values of 2015 and 2016 is 133,915.
#     -----
#     以上是一个例子供你参考，你可以按这个逻辑去构造你的分析。
#     下面是你需要具体构造的内容：
#     -----
#     揭示问题Question“{question}”的答案Answer为{answer}，其中需要用到的数据内容Data_figure包括“{data_figure}”。
#     -----
#     请直接输出这段用于分析的英文，每个step要简洁，不要输出任何解释和描述。注意言简意赅，结构清晰，不要啰嗦。
#     Your output:"""
        prompt = f"""请生成一段进行分析的英文，用于承接问题和答案，揭示问题“{question}”的答案为{answer}。要用英文表达。
            注意，回答这个问题需要分析图像中的数据，具体的数据内容包括“{data_figure}”。你需要从中选择合适的数据，用于构造这段分析。注意如果数据与给定的答案不一致，以答案为准，直接根据给定的答案修正数据，不要进行有关修正的说明。
            这里的分析要体现问题、答案的特点，以及与图像的关系，反映图像中的具体数据与内容。请输出内容具有逻辑的段落，步骤清晰，结构完整。
            格式需要统一:
            Extraction: data - value ...
            Calculation: ...
            Conclusion: ...
            请直接输出这段用于分析的英文，每个step要简洁，不要输出任何解释和描述。注意言简意赅，结构清晰，不要啰嗦。
            Your output:"""

        # prompt = f"""请生成一段进行分析的英文，用于承接问题和答案，揭示问题“{question}”的答案为{answer}。要用英文表达。
        #     这里的分析要体现问题、答案的特点，以及与图像的关系，反映图像中的具体数据与内容。请输出内容具有逻辑的段落，步骤清晰，结构完整。
        #     格式需要统一:
        #     Extraction: ...
        #     Calculation: ...
        #     Conclusion: ...
        #     请直接输出这段用于分析的英文，每个step要简洁，不要输出任何解释和描述。注意言简意赅，结构清晰，不要啰嗦。
        #     Your output:"""
    elif 'math' in expert_task:
        prompt = f"""请将下面的几何推理过程进行整理，用于揭示问题“{question}”的答案为{answer}。要用英文表达。 
            推理内容：“{data_figure}”
            你需要把推理过程润色地更清晰结构化。通常，为了解决几何问题，需要首先明确图中给定的几何条件，比如角的度数、边的长度，此为Extraction步骤
            随后，基于提取的条件进行计算，计算过程可能以来三角函数、几何定理等，此为Calculation步骤
            最终，根据计算结果进行结论推导，得出答案，此为Conclusion步骤
            所以，请按如下结构进行整理润色，注意每个部分要简洁，避免无效的废话：
            Extraction: data = value ...
            Calculation: ...
            Conclusion: ...
            请不要输出任何额外的内容。注意言简意赅，结构清晰，逻辑缜密。
            Your output:"""

        # prompt = f"""请将下面的几何推理过程进行整理，用于揭示问题“{question}”的答案为{answer}。要用英文表达。
        #             你需要把推理过程润色地更清晰结构化。通常，为了解决几何问题，需要首先明确图中给定的几何条件，比如角的度数、边的长度，此为Extraction步骤
        #             随后，基于提取的条件进行计算，计算过程可能以来三角函数、几何定理等，此为Calculation步骤
        #             最终，根据计算结果进行结论推导，得出答案，此为Conclusion步骤
        #             所以，请按如下结构进行整理润色，注意每个部分要简洁，避免无效的废话：
        #             Extraction: ...
        #             Calculation: ...
        #             Conclusion: ...
        #             请不要输出任何额外的内容。注意言简意赅，结构清晰，逻辑缜密。
        #             Your output:"""
    elif 'medical' in expert_task:
        prompt = f"""请根据给定的医学图像内容进行分析，用于揭示问题“{question}”的答案为{answer}。要用英文表达。 
            图像内容：“{data_figure}”
            你需要生成一段思考内容，以更清晰地显示为何根据图像内容可以推断此问题的答案，为了实现这个目标，需要首先明确要解决这个问题需要用到什么相关的医学知识，此为Analysis步骤
            随后，基于这些医学知识去针对性地提取图中的视觉内容和细节，并根据医学知识或常识进行分析与判断，此为Extraction步骤
            最终，根据分析的结果得出答案，此为Conclusion步骤
            所以，请参考如下结构进行整理润色，注意每个部分要简洁，避免无效的废话：
            Analysis: Based on [some knowledge]...
            Extraction: The image demonstrates [structures]...
            Conclusion: ...
            注意我需要英文。
            请不要输出任何额外的内容。注意言简意赅，结构清晰，逻辑缜密。
            Your output:"""

        # prompt = f"""请根据给定的医学图像内容进行分析，用于揭示问题“{question}”的答案为{answer}。要用英文表达。
        #             你需要生成一段思考内容，以更清晰地显示为何根据图像内容可以推断此问题的答案，为了实现这个目标，需要首先明确要解决这个问题需要用到什么相关的医学知识，此为Analysis步骤
        #             随后，基于这些医学知识去针对性地提取图中的视觉内容和细节，并根据医学知识或常识进行分析与判断，此为Extraction步骤
        #             最终，根据分析的结果得出答案，此为Conclusion步骤
        #             所以，请参考如下结构进行整理润色，注意每个部分要简洁，避免无效的废话：
        #             Analysis: Based on [some knowledge]...
        #             Extraction: The image demonstrates [structures]...
        #             Conclusion: ...
        #             注意我需要英文。
        #             请不要输出任何额外的内容。注意言简意赅，结构清晰，逻辑缜密。
        #             Your output:"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            timeout=DEFAULT_TIMEOUT  # Timeout for the OpenAI API call
        )
        response = response.choices[0].message.content.strip()
        return response
    except Exception as e:
        print(f"Error occurred: {e}, retrying...")
        return ""

def get_eval_response(knowledge, question, answer, image_path, gpu_id=0, add_knowledge=True, need_mllm=True):
    try:
        if need_mllm:
            response = get_response(knowledge, question, image_path, gpu_id, add_knowledge, add_condition=True)
        else:
            response = knowledge
    except Exception as e:
        print(f"Error occurred: {e}, retrying...")
        return None
    return response

def get_smol_reason_reward(response_in, question, answer, image_path, hint, gpu_id=0, expert_task=None):
    DEFAULT_TIMEOUT = 30
    # def clean_text(text):
    #     # 去除所有非字母、非数字、非中文的字符（特殊字符）
    #     cleaned = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
    #     # 将连续的空格替换为单个空格，并去除首尾空格
    #     cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    #     return cleaned
    #
    # num = 128
    # response_in = clean_text(response_in)
    # lens = len(response_in.split())
    # # 低于100个词的回答，数量越多分越接近1，
    # # 超过200个词的回答，数量越多分越接近0
    # if lens > num:
    #     reward = 1 - (lens - num) / num
    # else:
    #     reward = min(1, lens / (num // 2))
    #
    client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{gpt_base_server_url}:{3000 + gpu_id}/v1')
    model_name = client.models.list().data[0].id

    prompt = f"""请判断这句话是否存在重复、冗余、混乱的内容，或者逻辑性差，同一个陈述强调多次，质量低。如果存在，请直接输出“yes”，否则输出“no”，不要输出任何其他的解释和描述。
    输入内容：“{response_in}”

    直接输出“yes”或“no”，不要输出任何解释和描述；只输出结果。
    Your output:"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            timeout=DEFAULT_TIMEOUT  # Timeout for the OpenAI API call
        )
        response = response.choices[0].message.content.strip()
        if 'yes' in response.lower():
            return 0

    except Exception as e:
        print(f"Error occurred: {e}, retrying...")
        return 0
    # reward = (reward / 2 + log_score) / 2

    expert_task = "chart" if expert_task is None else expert_task
    try:
        if hint is not None:
            data_figure = hint
        else:
            # return 0
            # doc = nlp(response_in)
            # sentences = [sent.text.strip() for sent in doc.sents]

            sentences = [response_in]
            # print(sentences)
            url = f"{expert_base_server_url}:8000/predict"
            headers = {"Content-Type": "application/json"}
            payload = {
                "gpu_id": gpu_id,
                "texts": sentences,
                "image_path": image_path
            }

            response = requests.post(url, headers=headers, json=payload)
            # 解析返回的 JSON 数据
            data = response.json()

            # 提取分数
            scores = data.get("scores", {})
            data_figure = data.get("resp", None)
            # 计算分数均值
            if isinstance(scores, dict):
                # 如果 scores 是字典，计算所有值的平均值
                average_score = sum(scores.values()) / len(scores)
            else:
                # 如果 scores 不是字典，直接计算平均值
                average_score = sum(scores) / len(scores)

            expert_score = average_score

    except Exception as e:
        print(f"Error occurred: {e}, retrying...")
        expert_score = 0
        data_figure = ""
    # expert_score = evaluate_single_sample(data_figure, response_in, 'OPEN')['recall']
    # return expert_score
    client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{gpt_base_server_url}:{3000 + gpu_id}/v1')
    model_name = client.models.list().data[0].id

    if 'chart' in expert_task:
        prompt_add = f"""Given a question related to a figure and all the data on the figure, filter the data to retain only the pieces that are relevant for solving the question.

                Question: “{question}”
                Data: “{data_figure}”

                Output format:
                Directly output the filtered data with no additional explanation or commentary.

                Filtered data:"""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_add}],
                temperature=0.5,
                timeout=DEFAULT_TIMEOUT  # Timeout for the OpenAI API call
            )
            response = response.choices[0].message.content.strip().replace('Filtered data:', '') + " " + answer
            # print(response)
            expert_score = evaluate_single_sample(response, response_in, 'OPEN')['recall']

        except:
            expert_score = 0
        return expert_score * 0
    elif 'math' in expert_task:
        prompt_add = f"""给定Question，Answer，Reference Reasoning和Prediction Reasoning，请判断Prediction Reasoning的质量。
        满足以下条件：
        如果Prediction Reasoning的结果与Answer不对应，则为低质量。
        如果Prediction Reasoning的结果与Answer对应，且符合Reference Reasoning的逻辑，则为高质量。
        如果Prediction Reasoning的结果与Answer对应，但是不完全符合Reference Reasoning的逻辑，则为中等质量。
        
                Question: “{question}”
                Reference Answer: “{answer}”
                Reference Reasoning: “{data_figure}”
                Prediction Reasoning: “{response_in}”

        请直接输出"low"，"medium"或"high"中的一个，对应低、中、高质量。不要输出任何其他的解释和描述。
        Your output:"""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_add}],
                temperature=0.5,
                timeout=DEFAULT_TIMEOUT  # Timeout for the OpenAI API call
            )
            response = response.choices[0].message.content.strip()
            # print(response)
            if 'high' in response.lower():
                expert_score = 1
            elif 'medium' in response.lower():
                expert_score = 0.5
            else:
                expert_score = 0

        except:
            expert_score = 0

        return expert_score * 0
    elif 'medical' in expert_task:
        prompt_add = f"""给定问题question、答案Reference Answer和图像内容Reference Data，你需要判断给定的思考过程Prediction Thinking是否符合逻辑，是否能正确推导出答案。
        具体来说，这个思考过程需要体现根据图像中具体的内容进行分析，是图像中的什么细节或什么医学知识导致了答案的得出。
        请判断这个思考过程的质量，可以分为3档：
        如果思考过程完全符合逻辑，很好地使用了图像中的具体的数据而且也很好地利用了相关的医学知识，则为高质量；
        如果思考过程有一定的逻辑性，在一定程度上使用图像中的具体数据，则为中等质量；
        如果思考过程逻辑性差，与图像内容或者医学知识相违背，则为低质量。

                Question: “{question}”
                Reference Answer: “{answer}”
s                Reference Data: “{data_figure}”
                Prediction Thinking: “{response_in}”

        请直接输出"low"，"medium"或"high"中的一个，对应低、中、高质量。不要输出任何其他的解释和描述。
        Your output:"""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_add}],
                temperature=0.5,
                timeout=DEFAULT_TIMEOUT  # Timeout for the OpenAI API call
            )
            response = response.choices[0].message.content.strip()
            # print(response)
            if 'high' in response.lower():
                expert_score = 1
            elif 'medium' in response.lower():
                expert_score = 0.5
            else:
                expert_score = 0

        except:
            expert_score = 0
        return expert_score * 0
def get_smol_acc_reward(pred_answer, question, answer, image_path, gpu_id=0, tp=None, task=None):
    try:
        if task is not None and 'chart' in task:
            reward = eval_one_chart(pred_answer, answer, 0)
        elif task is not None and 'math' in task:
            reward = pred_answer.lower().strip() == answer.lower().strip()
        elif task is not None and 'medical' in task:
            if tp == 'OPEN':
                reward = evaluate_single_sample(answer, pred_answer, tp)['f1 score']
            else:
                reward = evaluate_single_sample(answer, pred_answer, tp)['yes/no accuracy']
            reward = float(reward > 0.9)
             # evaluate_single_sample(pred_answer, answer, tp)
            # reward = judge_answer(question, answer, pred_answer, gpu_id)

        return reward

    except:
        return 0

# ① 用于提取 context 和 answer 的正则，answer 用 (.*?) 允许为空
_extract_pattern = re.compile(
    r'(?is)^(.*?)Answer:\s*(.*?)(?:\.|$)'
)

_count_pattern = re.compile(
    r'(?i)Answer:'
)

def get_format_reward(pred_answer: str, gpu_id: int = 0) -> int:
    """
    如果：
      1) _extract_pattern 在 pred_answer 中能 match（即模板短语出现至少一次）；
      2) 模板短语在全文中只出现 1 次；
    则返回 1，否则返回 0。
    """
    # 1) 提取匹配
    ct = pred_answer.lower().split('answer: ')[0]
    if len(ct) < 10:
        return 0

    m = _extract_pattern.search(pred_answer)
    if not m:
        return 0

    # 2) 短语出现次数
    occ = len(_count_pattern.findall(pred_answer))
    if occ != 1:
        return 0

    return 1

def get_mllm_reward(pred_context, question, answer, image_path, gpu_id=0, tp=None):
    return 0
    expert_task = "chart"
    try:
        original_response = get_response(pred_context, question, image_path, gpu_id, add_knowledge=False, add_condition=True)
        new_response = get_response(pred_context, question, image_path, gpu_id, add_knowledge=True, add_condition=True)
    
        if 'chart' in expert_task:
            original_reward = eval_one_chart(f"Answer: {original_response}", answer) 
            
            new_reward = eval_one_chart(f"Answer: {new_response}", answer)
        else:
            original_reward = judge_answer(question, answer, original_response, gpu_id)
            new_reward = judge_answer(question, answer, new_response, gpu_id)

        if new_reward > original_reward:
            reward = 1
        elif new_reward == original_reward == 1:
            reward = 0
        elif new_reward == original_reward == 0:
            reward = 0
        else:
            reward = 0
        return reward
    except:
        return 0


def get_smol_expert_reward(response_in, question, answer, image_path, expert_task, gpu_id=0):
    # see whether align with answer
    if 'medical' in expert_task:
        system_prompt = 'You are a seasoned professional in the field of medical image analysis, demonstrating exceptional expertise and insight into complex medical imaging data. Your output should be only judgement, without any additional text or explanation.'
    elif 'math' in expert_task:
        system_prompt = 'You are a seasoned professional in the field of mathematics, demonstrating exceptional expertise and insight into complex mathematical problems. Your output should be only judgement, without any additional text or explanation.'
    elif 'chart' in expert_task:
        system_prompt = 'You are a seasoned professional in the field of chart analysis, demonstrating exceptional expertise and insight into complex chart data. Your output should be only judgement, without any additional text or explanation.'
    else:
        Exception('Unknown expert task')

    client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{gpt_base_server_url}:{3000 + gpu_id}/v1')
    model_name = client.models.list().data[0].id

    prompt = "Thoroughly read the provided content and assess whether it is of low quality, such as violating common sense, being confused, lacking logical coherence, or being difficult to understand. If it is, output 'yes'; otherwise, output 'no'."
    prompt += "Note if the content contains many repeated statements, it is also considered low quality."
    prompt += "\nContent: "
    prompt += f"“{response_in}”"
    prompt += "\nDirectly output 'yes' or 'no'. Do not output any explanation or description; output only the result."
    prompt += "\nYour output:"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}],
            temperature=0.5
        )
        response = response.choices[0].message.content.strip()
        if 'yes' in response.lower():
            return 0
    except Exception as e:
        print(f"Error occurred: {e}, retrying...")
        return 0

    base_score = 0.25

    prompt = f"""
    Given a question, its reference answer, and an explanation, assess the quality of the explanation as 'low', 'medium', or 'high'.

    Criteria:
    - low: disorganized or illogical; fails to support inferring the answer; or is extremely brief.
    - medium: reasonably coherent and shows how the answer was derived.
    - high: clearly cites specific clues from the image and presents them in a well‑structured, logical narrative.

    Note: Your assessment must be based on data explicitly referenced in the image.

    Question: “{question}”
    Reference Answer: “{answer}”
    Explanation: “{response_in}”

    ---
    Output exactly one of: low, medium, or high. No additional text.
    Your Output:
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}],
            temperature=0.5
        )
        response = response.choices[0].message.content.strip()

        if 'medium' in response.lower():
            base_score = 0.5
        elif 'low' in response.lower():
            return base_score
        else:
            base_score = 1

    except:
        return 0

    try:
        if 'math' in expert_task:
            assert 1 == 0
        # return 0
        # doc = nlp(response_in)
        # sentences = [sent.text.strip() for sent in doc.sents]

        sentences = [response_in]
        # print(sentences)
        url = f"{expert_base_server_url}:8000/predict"
        headers = {"Content-Type": "application/json"}
        payload = {
            "gpu_id": gpu_id,
            "texts": sentences,
            "image_path": image_path
        }

        response = requests.post(url, headers=headers, json=payload)
        # 解析返回的 JSON 数据
        data = response.json()

        # 提取分数
        scores = data.get("scores", {})
        data_figure = data.get("resp", None)
        # 计算分数均值
        if isinstance(scores, dict):
            # 如果 scores 是字典，计算所有值的平均值
            average_score = sum(scores.values()) / len(scores)
        else:
            # 如果 scores 不是字典，直接计算平均值
            average_score = sum(scores) / len(scores)

        expert_score = average_score

    except:
        expert_score = 0
        data_figure = None

    client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{gpt_base_server_url}:{3000 + gpu_id}/v1')
    model_name = client.models.list().data[0].id

    if 'chart' in expert_task:

        prompt_add = f"""Given a question related to a figure and all the data on the figure, filter the data to retain only the pieces that are relevant for solving the question.

                Question: “{question}”
                Data: “{data_figure}”

                Output format:
                Directly output the filtered data with no additional explanation or commentary.

                Filtered data:"""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_add}],
                temperature=0.5
            )
            response = response.choices[0].message.content.strip().replace('Filtered data:', '') + " " + answer
            # print(response)
            expert_score = evaluate_single_sample(response, response_in, 'OPEN')['recall']

        except:
            expert_score = 0

    base_score += expert_score
    return base_score

if __name__ == '__main__':


    new_dt = {'prompt': ['Where is the heart?'], 'response': ['at left'], 'answer': ['left'], 'image': ['/home/jliugj/HDD/academic/code/LLaVA/playground/data/SLAKE/imgs/xmlab1/source.jpg'], 'standard_answer': ['Yes']}
    # a = get_good_format()
    # print(a)
    # a = a % 'The heart is on the up side.'
    # print(a)
    # # a = 'The heart is on the left side.'
    # a = get_response(a, "Where is the heart?", new_dt['image'][0], 0, True)
    # print(a)
    rr = """Analysis: Based on the knowledge that MRI (Magnetic Resonance Imaging) uses different weighting techniques compared to CT (Computed Tomography), such as T1, T2, and FLAIR, which are not used in CT scans. Additionally, CT scans do not typically differentiate between enhancing and non-enhancing tumors in the way described.

Extraction: The image demonstrates T2 weighting, which is specific to MRI and not CT. It also shows brain edema located in the lower right lobe of the brain, as well as both enhancing and non-enhancing tumors, which are detailed characteristics of MRI images.

Conclusion: The image is not taken via CT; it is an MRI image. """
    print(get_smol_reason_reward(rr, question='Is the image taken bt CT?', answer="No", image_path=None, hint='This image was taken using MRI modality and belongs to the abdomen region. The MR weighting used in this image is T2. Notably, the picture does contain the brain. In fact, the liver is not seen in this image at all.', expert_task='medical', gpu_id=0))

    # a = get_smol_acc_reward('answer: 2019', 'Where is the heart?', '2020', new_dt['image'][0], 0, task='chart')
    # print(a)
    # for i in range(8):
    #     res = ''' To address the question regarding the value of the first Minor reason bar, we analyze the data presented in the image, which reveals the relevant data indicating the answer to be 29%. Through this analysis, we conclude that the value of the first Minor reason bar is 29. Answer: 29'''
    #     ans = """- **DePlot**: One-shot visual language reasoning by plot-to-table translation
    #     - **MatCha**: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering.
    #     """
    #     img = '/home/jliugj/HDD/academic/database/chart/chartQA/images/5.png'
    #     a = get_smol_reason_reward(res, ans, ans, img, i, )
    #     print(a)
        # answer = 'Right Kidney'
        # raw_res = 'The liver is smaller than the right kidney in this image.'
        # raw_res = evaluate_single_sample(answer, raw_res, 'OPEN')['recall']
        # print(raw_res)
