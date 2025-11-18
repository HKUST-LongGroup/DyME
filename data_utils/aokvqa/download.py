import os
import json
import base64
import io
import multiprocessing
from functools import partial
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI, BadRequestError
import time

# 确保 PIL 能够处理大型图像
Image.MAX_IMAGE_PIXELS = None

# --- API 和 MLLM 提示词配置 ---

# 1. 定义您的8个API端口
API_PORTS = list(range(23333, 23333 + 8))
API_URL_TEMPLATE = "http://127.0.0.1:{port}/v1/"

# 2. 定义用于获取 visual_fact 的提示词
VISUAL_FACT_SYSTEM_PROMPT = """
You are a helpful assistant that analyzes images and provides visual facts.
Your response MUST be a single, valid JSON object.
The JSON object should contain:
1. "description": A detailed and accurate description of the image.
2. "objects": A list of key objects, including their name, attributes, and approximate position in the image.

Example format:
{
  "description": "A person riding a bicycle on a city street.... (detailed description here)",
  "objects": [
    {"name": "person", "attributes": ["wearing helmet", "blue shirt"], "position": "center"},
    {"name": "bicycle", "attributes": ["red", "mountain bike"], "position": "center"},
    {"name": "street", "attributes": ["asphalt", "wet"], "position": "bottom"}
  ]
}
"""

VISUAL_FACT_USER_PROMPT = """
Analyze the attached image and provide the visual facts in the required JSON format.
For context, the user will be asked this question about the image (do not answer the question, just use it for context):
"{question}"
"""


def encode_image_to_base64(pil_image):
    """将 PIL 图像对象转换为 Base64 字符串"""
    buffered = io.BytesIO()
    if pil_image.mode == "RGBA" or "transparency" in pil_image.info:
        pil_image.save(buffered, format="PNG")
        mime_type = "image/png"
    else:
        pil_image.save(buffered, format="JPEG")
        mime_type = "image/jpeg"

    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:{mime_type};base64,{img_str}"


def get_visual_fact(api_url, pil_image, question):
    """
    调用外部 MLLM API 获取 visual_fact。
    """
    try:
        client = OpenAI(base_url=api_url, api_key="DUMMY_KEY")
        image_url = encode_image_to_base64(pil_image)

        messages = [
            {"role": "system", "content": VISUAL_FACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url,}},
                    {"type": "text", "text": VISUAL_FACT_USER_PROMPT.format(question=question)}
                ]
            }
        ]

        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            response_content = response.choices[0].message.content
            return response_content

        except (BadRequestError, Exception):
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            response_content = response.choices[0].message.content

            if response_content.startswith("```json"):
                response_content = response_content[7:].strip("` \n")

            return response_content

    except json.JSONDecodeError as e:
        print(f"!! JSON 解析失败: {e}. API: {api_url}. 原始响应: {response_content[:100]}...")
        return {"error": "Failed to parse JSON response", "raw_response": response_content}
    except Exception as e:
        print(f"!! API 调用失败: {e}. API: {api_url}")
        time.sleep(1)
        return {"error": f"API call failed: {str(e)}"}


# --- (这是被修改的函数) ---
def process_example_worker(example_with_index, split, image_output_dir, api_ports_list, fetch_visual_facts):
    """
    多进程的工作函数。
    处理单个数据样本：保存图像，（如果需要）调用 API。
    """
    i, example = example_with_index

    try:
        # 1. 提取元数据
        question = example["question"]
        pil_image = example["image"]

        # --- (A) 提取新字段 ---
        choices_list = example.get("choices")
        correct_idx = example.get("correct_choice_idx")
        direct_answers_list = example.get("direct_answers")
        rationales_list = example.get("rationales")  # A-OKVQA 有 rationales 字段

        # --- (B) 确定 "answer" (按您的要求优先使用 choice) ---
        answer = None
        if choices_list and correct_idx is not None and 0 <= correct_idx < len(choices_list):
            answer = choices_list[correct_idx]
        elif direct_answers_list:
            answer = direct_answers_list[0]  # 回退到 direct_answers

        # --- (C) 确定 "hint" (最长的 rationale) ---
        hint = None
        if rationales_list:
            try:
                # 确保 rationales_list 是一个非空字符串列表
                if isinstance(rationales_list, list) and len(rationales_list) > 0:
                    string_rationales = [r for r in rationales_list if isinstance(r, str)]
                    if string_rationales:
                        hint = max(string_rationales, key=len)  # 选择最长的字符串
            except Exception as e:
                print(f"!! 警告: 计算 'hint' 失败 (索引 {i}). 错误: {e}")
                pass  # hint 保持为 None

        # --- (D) 结束提取 ---

        # 2. 生成唯一的、绝对的图像路径
        generated_filename = f"{split}_{i:07d}.png"
        image_save_path = os.path.join(image_output_dir, generated_filename)

        # 3. 保存图像 (I/O 操作)
        if not os.path.exists(image_save_path):
            pil_image.save(image_save_path)

        # 4. (关键步骤) 获取 Visual Fact
        visual_fact_data = None
        if fetch_visual_facts:
            # 根据索引 'i' 轮询使用 API 端口
            port_to_use = api_ports_list[i % len(api_ports_list)]
            api_url = API_URL_TEMPLATE.format(port=port_to_use)

            visual_fact_data = get_visual_fact(api_url, pil_image, question)

        # 5. 构建元数据字典 (使用您指定的新格式)
        metadata = {
            "question": question,
            "question_wo_prompt": question,
            "answer": answer,
            "choices": choices_list,  # 存储完整的选项列表
            "image": image_save_path,
            "visual_fact": visual_fact_data,
            "hint": hint
        }

        return metadata

    except Exception as e:
        print(f"!! 工作进程中出现严重错误 (索引 {i}): {e}")
        return None  # 主进程将过滤掉 None


def save_aokvqa_with_facts(base_output_dir="/aokvqa_output"):
    """
    加载 A-OKVQA，使用多进程保存图像并为训练集获取 visual_facts。
    """

    # 1. 定义输出目录
    base_dir_abs = os.path.abspath(base_output_dir)
    image_output_dir = os.path.join(base_dir_abs, "images")
    json_output_dir = os.path.join(base_dir_abs, "json")

    # 2. 创建目录
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    print(f"图像将保存到 (绝对路径): {image_output_dir}")
    print(f"JSON 将保存到 (绝对路径): {json_output_dir}")
    print(f"将使用 {len(API_PORTS)} 个 API 端口: {API_PORTS}")

    # 3. 加载数据集
    print("正在加载 HuggingFaceM4/A-OKVQA 数据集...")
    try:
        dataset = load_dataset("HuggingFaceM4/A-OKVQA")
        # 取一小部分测试
        # dataset['train'] = dataset['train'].select(range(100))
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    print(f"数据集中包含的 Splits: {list(dataset.keys())}")

    # 4. 定义工作进程数
    num_workers = 64
    print(f"启动 {num_workers} 个工作进程...")

    # 5. 遍历每个 split
    for split in dataset.keys():

        fetch_facts = (split == 'train')

        if fetch_facts:
            print(f"\n--- 正在处理 {split} split (将调用 MLLM API) ---")
        else:
            print(f"\n--- ------------------- ---")
            print(f"--- 正在处理 {split} split (仅保存图像, visual_fact=None) ---")
            print(f"--- ------------------- ---")

        metadata_list = []

        worker_func = partial(
            process_example_worker,
            split=split,
            image_output_dir=image_output_dir,
            api_ports_list=API_PORTS,
            fetch_visual_facts=fetch_facts
        )

        tasks = list(enumerate(dataset[split]))
        total_count = len(tasks)

        # 6. 使用多进程池
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap_unordered(worker_func, tasks), total=total_count, desc=f"处理 {split}"):
                if result:
                    metadata_list.append(result)

        print(f"  {split} split 处理完成。成功 {len(metadata_list)} / {total_count} 条。")

        # 7. (可选) 排序
        metadata_list.sort(key=lambda x: x['image'])

        # 8. 写入 JSON 文件
        json_filename = os.path.join(json_output_dir, f"{split}.json")
        print(f"正在将 {len(metadata_list)} 条元数据保存到 {json_filename}...")

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)

    print(f"\n--- 全部处理完成！ ---")
    print(f"所有图像文件已保存在: '{image_output_dir}'")
    print(f"所有 JSON 文件已保存在: '{json_output_dir}'")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # 更改为您希望的输出目录
    save_aokvqa_with_facts(base_output_dir="/apdcephfs_nj4/share_300377003/realzliu/data/aokvqa")