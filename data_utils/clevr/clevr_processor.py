#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- 关键修复 1: PytorchGELUTanh 猴子补丁 (主进程) ---
# (保持不变)
try:
    from transformers import activations

    activations.PytorchGELUTanh = activations.GELUTanh
except ImportError:
    print("注意: 无法应用 PytorchGELUTanh 补丁。如果遇到 ImportError，请检查 transformers 版本。")
# --- 补丁结束 ---

import os
import shutil
import json
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator

# (qwen_vl_utils 导入和回退保持不变)
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("警告: 无法导入 'qwen_vl_utils.process_vision_info'。")


    def process_vision_info(messages):
        images = []
        for msg in messages:
            if msg['role'] == 'user':
                for content in msg['content']:
                    if content['type'] == 'image':
                        images.append(content['image'])
        return images, None  # 返回 (images, videos)

Image.MAX_IMAGE_PIXELS = None

# (RexOmni 依赖和 DummyRex 保持不变)
try:
    from rex_omni import RexOmniWrapper
except ImportError:
    print("警告: 'from rex_omni import RexOmniWrapper' 失败。")
    print("将使用一个虚拟的 RexOmniWrapper (DummyRex) 仅供测试。")


    class DummyRex:
        def __init__(self, *args, **kwargs):
            print("INFO: DUMMY: 正在使用 DummyRex 检测器。")

        def inference(self, images, task, categories, **kwargs):
            print("INFO: DUMMY: DummyRex 正在返回一个伪造的中心框。")
            if isinstance(images, Image.Image):
                w, h = images.size
            else:
                w, h = 800, 600
            x0, y0 = w * 0.25, h * 0.25
            x1, y1 = w * 0.75, h * 0.75
            return [{"extracted_predictions": {"anything": [{"type": "box", "coords": [x0, y0, x1, y1]}]}}]


    RexOmniWrapper = DummyRex


def _strip_tags(text, tag_name):
    # (保持不变)
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(rf'<{tag_name}>', '', text, flags=re.IGNORECASE)
    text = re.sub(rf'</{tag_name}>', '', text, flags=re.IGNORECASE)
    return text.strip()


# --- 核心 VQA 辅助函数 (移至全局) ---

def _crop_and_expand_box(image, box, padding_pixels=20):
    # (保持不变)
    x0, y0, x1, y1 = [int(c) for c in box]
    img_w, img_h = image.size
    x0_new = max(0, x0 - padding_pixels)
    y0_new = max(0, y0 - padding_pixels)
    x1_new = min(img_w, x1 + padding_pixels)
    y1_new = min(img_h, y1 + padding_pixels)
    return image.crop((x0_new, y0_new, x1_new, y1_new))


# --- ★★★ 优化点 1: 将 VQA 查询改为批处理 ★★★ ---
def _query_qwen_vl_BATCH(crop_images_list, model, processor, accelerator):
    """
    使用 Qwen-VL 批量查询裁剪后的图像块，并返回 JSON 字符串列表。
    """
    if not crop_images_list:
        return []

    prompt = """This is an object from a CLEVR scene. Analyze the primary object in the image.
Respond *strictly* with a JSON list (containing one dictionary) in the following format:
[
    {"object": "object_name", "attributes": ["attr1", "attr2"]}
]
- "object": The shape of the object (e.g., "sphere", "cube", "cylinder", "cone").
- "attributes": A list of visual attributes (e.g., ["blue", "large", "metal", "shiny", "rubber"]).
Provide only the JSON list:"""

    # 1. 为批处理中的每个图像创建消息
    # 我们需要获取模板化的文本提示
    template_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder.jpg"},  # 占位符，用于生成模板
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        # 2. 生成一次聊天提示文本
        chat_prompt_text = processor.apply_chat_template(
            template_messages, tokenize=False, add_generation_prompt=True
        )

        num_crops = len(crop_images_list)
        batch_text = [chat_prompt_text] * num_crops
        batch_images = crop_images_list

        unwrapped_model = accelerator.unwrap_model(model)

        # 3. 使用文本列表和图像列表进行批处理
        inputs = processor(
            text=batch_text,
            images=batch_images,
            padding=True,  # 关键：启用填充以处理批处理
            return_tensors="pt",
        ).to(unwrapped_model.device)

        # 4. 批量生成
        generated_ids = unwrapped_model.generate(**inputs, max_new_tokens=256, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 5. 批量解码
        output_texts_list = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_texts_list

    except Exception as e:
        print(f"Qwen-VL 批量推理失败: {e}")
        # 返回一个空列表，其长度与输入批次相同，以防止 zip 错误
        return ["[]"] * len(crop_images_list)


def _parse_qwen_json(response_text):
    # (保持不变)
    try:
        match = re.search(r'```json\s*(\[.*\])\s*```', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        match = re.search(r'(\[.*\])', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        return []
    except json.JSONDecodeError:
        print(f"JSON 解析失败: {response_text}")
        return []
    except Exception as e:
        print(f"JSON 解析时发生未知错误: {e}")
        return []


def _load_and_preprocess_data(base_output_dir, image_output_dir):
    # (保持不变)
    print("正在加载 'MMInstruction/Clevr_CoGenT_TrainA_R1'...")
    try:
        dataset = load_dataset("MMInstruction/Clevr_CoGenT_TrainA_R1", split='train')
    except Exception as e:
        print(f"加载数据集 'MMInstruction/Clevr_CoGenT_TrainA_R1' 失败: {e}")
        return []

    print(f"已加载 {len(dataset)} 个样本。")

    job_list = []
    print("正在预处理数据（保存图像并解析文本）...")
    for i, example in enumerate(tqdm(dataset, desc="预处理进度")):
        prompt = example['problem']
        hint = _strip_tags(example['thinking'], 'think')
        answer = _strip_tags(example['solution'], 'answer')

        image = example['image']
        if not isinstance(image, Image.Image):
            print(f"警告: 样本 {i} 不是一个 PIL 图像，已跳过。")
            continue

        image_filename = f"clevr_cogent_trainA_r1_{i:07d}.jpg"
        destination_image_path = os.path.join(image_output_dir, image_filename)

        try:
            os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
            if not os.path.exists(destination_image_path):
                image.convert("RGB").save(destination_image_path, "JPEG")
        except Exception as e:
            print(f"警告: 保存样本 {i} 的图像失败。已跳过。错误: {e}")
            continue

        job_list.append({
            "prompt": prompt,
            "answer": answer,
            "hint": hint,
            "destination_image_path": destination_image_path
        })

    print(f"成功预处理 {len(job_list)} 个项目。")

    job_list_path = os.path.join(base_output_dir, "job_list.json")
    with open(job_list_path, 'w', encoding='utf-8') as f:
        json.dump(job_list, f)

    print(f"作业列表已保存到: {job_list_path}")
    return job_list


def main():
    # (1. 初始化 Accelerator - 保持不变)
    accelerator = Accelerator()

    # (0. 定义配置 - 保持不变)
    MODEL_CONFIGS = {
        "rex_path": "IDEA-Research/Rex-Omni",
        "qwen_path": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
    }
    OUTPUT_DIR = "/apdcephfs_nj4/share_300377003/realzliu/data/clevr_cogent_output"
    IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
    JSON_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "json")

    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

    # (2. 预处理 (仅在主进程上运行) - 保持不变)
    job_list_path = os.path.join(OUTPUT_DIR, "job_list.json")
    if accelerator.is_main_process:
        print("主进程 [Pre-processing]: 正在加载和预处理数据...")
        _load_and_preprocess_data(OUTPUT_DIR, IMAGE_OUTPUT_DIR)

    # (3. 同步 - 保持不变)
    accelerator.wait_for_everyone()

    # (4. 加载并分发作业 - 保持不变)
    if not accelerator.is_main_process:
        print(f"进程 {accelerator.process_index}: 正在加载 job_list.json...")
    try:
        with open(job_list_path, 'r', encoding='utf-8') as f:
            all_jobs = json.load(f)
    except Exception as e:
        print(f"进程 {accelerator.process_index} 加载 job_list.json 失败: {e}")
        return
    total_jobs = len(all_jobs)
    num_processes = accelerator.num_processes
    jobs_per_process = total_jobs // num_processes
    start_index = accelerator.process_index * jobs_per_process
    end_index = (accelerator.process_index + 1) * jobs_per_process
    if accelerator.is_last_process:
        end_index = total_jobs
    my_jobs = all_jobs[start_index:end_index]
    print(f"[Process {accelerator.process_index}]:"
          f" 已分配 {len(my_jobs)} 个作业 (索引从 {start_index} 到 {end_index})。")

    # (5. 加载模型 (每个进程加载自己的副本) - 保持不变)
    try:
        try:
            from transformers import activations
            activations.PytorchGELUTanh = activations.GELUTanh
        except ImportError:
            pass

        print(f"[Process {accelerator.process_index}]: 正在加载 RexOmni...")
        rex_model = RexOmniWrapper(
            model_path=MODEL_CONFIGS['rex_path'],
            backend="transformers",
            max_tokens=2048,
            temperature=0.0,
        )

        print(f"[Process {accelerator.process_index}]: 正在加载 Qwen-VL...")
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_CONFIGS['qwen_path'],
            torch_dtype="float16",
            device_map="cuda",
            attn_implementation="flash_attention_2"
        )
        qwen_processor = AutoProcessor.from_pretrained(MODEL_CONFIGS['qwen_path'])
        qwen_model, rex_model = accelerator.prepare(qwen_model, rex_model)
        print(f"[Process {accelerator.process_index}]: 模型加载完毕。")
    except Exception as e:
        print(f"[Process {accelerator.process_index}]: 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- ★★★ 优化点 2: 修改主循环以使用批处理 ★★★ ---
    processed_metadata_list = []
    for job in tqdm(my_jobs,
                    desc=f"Worker {accelerator.process_index} 进度",
                    disable=not accelerator.is_main_process):
        try:
            destination_image_path = job['destination_image_path']
            image = Image.open(destination_image_path).convert("RGB")

            # 1. RexOmni 推理 (这通常已经是批处理的，但在这里是单张)
            rex_results = rex_model.inference(
                images=image,
                task="detection",
                categories=["anything"]
            )
            predictions = rex_results[0]["extracted_predictions"]
            detected_boxes = predictions.get("anything", [])

            visual_facts = []
            crops_to_process = []
            box_coords_list = []

            # 2. 收集所有需要处理的裁剪图
            for annotation in detected_boxes:
                if annotation.get("type") == "box" and len(annotation.get("coords", [])) == 4:
                    coords = annotation["coords"]
                    crop_image = _crop_and_expand_box(image, coords)
                    crops_to_process.append(crop_image)
                    box_coords_list.append(coords)

            # 3. 如果有需要处理的裁剪图，则进行批量 VQA 推理
            if crops_to_process:
                json_str_list = _query_qwen_vl_BATCH(
                    crops_to_process, qwen_model, qwen_processor, accelerator
                )

                # 4. 遍历批处理结果并进行解析
                for json_str, coords in zip(json_str_list, box_coords_list):
                    json_obj_list = _parse_qwen_json(json_str)

                    if json_obj_list:
                        try:
                            # 假设每个 JSON 列表都包含一个对象
                            obj_dict = json_obj_list[0]
                            obj_dict["bounding_box"] = [round(c, 2) for c in coords]
                            visual_facts.append(obj_dict)
                        except (IndexError, TypeError, KeyError) as e:
                            print(f"[Process {accelerator.process_index}]: "
                                  f"解析批处理结果时出错: {e} | JSON: {json_str}")

            # 5. 聚合此作业的结果
            processed_metadata_list.append({
                "question": job['prompt'],
                "answer": job['answer'],
                "question_wo_prompt": job['prompt'],
                "hint": job['hint'],
                "image": destination_image_path,
                "visual_fact": visual_facts
            })

        except Exception as e:
            print(f"[Process {accelerator.process_index}]: "
                  f"处理 {job['destination_image_path']} 时出错: {e}")

    print(f"[Process {accelerator.process_index}]:"
          f" 进程完成，处理了 {len(processed_metadata_list)} 个项目。")

    # (7. 收集所有结果 - 保持不变)
    print(f"[Process {accelerator.process_index}]: 正在收集结果...")
    all_results_list_of_lists = accelerator.gather_object(processed_metadata_list)

    # (8. 保存 (仅在主进程上) - 保持不变)
    if accelerator.is_main_process:
        print("主进程 [Saving]: 正在聚合和保存所有结果...")
        final_metadata_list = [item for sublist in all_results_list_of_lists for item in sublist]
        json_filename = os.path.join(JSON_OUTPUT_DIR, "clevr_cogent_trainA_r1_processed.json")
        print(f"\n正在将 {len(final_metadata_list)} 条元数据保存到 {json_filename}...")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(final_metadata_list, f, indent=4, ensure_ascii=False)
        print(f"\n--- 处理完成！ ---")
        print(f"所有图像文件已保存在: '{IMAGE_OUTPUT_DIR}'")
        print(f"最终 JSON 文件已保存在: '{JSON_OUTPUT_DIR}'")


if __name__ == "__main__":
    main()