#!/usr/bin/env python
# -*- coding: utf-8 -*-
from accelerate.utils import gather_object

# --- 关键修复 1: PytorchGELUTanh 猴子补丁 (主进程) ---
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
            print("INFO: DUMMY: DummyRex 正在返回伪造的中心框。")

            # 支持批处理的 Dummy
            results = []

            # 确保 images 是一个列表
            if not isinstance(images, list):
                images = [images]

            for img in images:
                if isinstance(img, Image.Image):
                    w, h = img.size
                else:
                    w, h = 800, 600
                x0, y0 = w * 0.25, h * 0.25
                x1, y1 = w * 0.75, h * 0.75
                results.append({"extracted_predictions": {"anything": [{"type": "box", "coords": [x0, y0, x1, y1]}]}})
            return results


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
    template_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder.jpg"},  # 占位符
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

    # 只要前几行做测试 (此处仍为 100 条)
    # dataset = dataset.select(range(100))
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

        # 注意: RexOmniWrapper (如果不是 Dummy) 可能需要被 .to(accelerator.device)
        # 但 Qwen-VL 已经通过 device_map="cuda" 指定了设备
        # accelerator.prepare 仍然是管理模型的好方法
        qwen_model, rex_model = accelerator.prepare(qwen_model, rex_model)

        print(f"[Process {accelerator.process_index}]: 模型加载完毕。")
    except Exception as e:
        print(f"[Process {accelerator.process_index}]: 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- ★★★ 优化点 3: 引入 RexOmni 批处理 (修改主循环) ★★★ ---

    # 1. 定义一个批处理大小 (Batch Size)
    REX_BATCH_SIZE = 16  # <-- 根据你的 VRAM 调整这个值

    print(f"[Process {accelerator.process_index}]:"
          f" 开始处理 {len(my_jobs)} 个作业，Rex 批处理大小为 {REX_BATCH_SIZE}。")

    processed_metadata_list = []

    # 2. 修改主循环：按 REX_BATCH_SIZE 步长迭代
    for i in tqdm(range(0, len(my_jobs), REX_BATCH_SIZE),
                  desc=f"Worker {accelerator.process_index} 批次进度",
                  disable=not accelerator.is_main_process):

        # 3. 准备这一批的作业和图像
        batch_jobs = my_jobs[i: i + REX_BATCH_SIZE]
        batch_images = []
        batch_image_paths = []  # 用于调试

        valid_jobs_in_batch = []

        for job in batch_jobs:
            try:
                img_path = job['destination_image_path']
                batch_image_paths.append(img_path)
                batch_images.append(Image.open(img_path).convert("RGB"))
                valid_jobs_in_batch.append(job)  # 只有图像加载成功，作业才有效
            except Exception as e:
                print(f"[Process {accelerator.process_index}]:"
                      f" 加载图像 {img_path} 失败: {e}，该批次中将跳过此图。")
                # 我们不添加图像或作业，保持 batch_images 和 valid_jobs_in_batch 同步

        if not batch_images:  # 如果这个批次所有图片都加载失败
            continue

        try:
            # 4. ★ 关键：批量运行 RexOmni
            # (我们只传入成功加载的图像)
            all_rex_results = rex_model.inference(
                images=batch_images,  # 传入图像列表
                task="detection",
                categories=["anything"]
            )

            # 5. 遍历这一批的结果
            # all_rex_results 列表的长度应等于 batch_images (和 valid_jobs_in_batch)
            if len(all_rex_results) != len(valid_jobs_in_batch):
                print(f"[Process {accelerator.process_index}]: 警告: RexOmni "
                      f"返回结果数 ({len(all_rex_results)}) 与输入数 "
                      f"({len(valid_jobs_in_batch)}) 不匹配。跳过此批次。")
                continue

            for job, image, rex_result in zip(valid_jobs_in_batch, batch_images, all_rex_results):

                predictions = rex_result["extracted_predictions"]
                detected_boxes = predictions.get("anything", [])

                visual_facts = []
                crops_to_process = []
                box_coords_list = []

                # 6. 收集所有需要处理的裁剪图 (来自这张图)
                for annotation in detected_boxes:
                    if annotation.get("type") == "box" and len(annotation.get("coords", [])) == 4:
                        coords = annotation["coords"]
                        crop_image = _crop_and_expand_box(image, coords)
                        crops_to_process.append(crop_image)
                        box_coords_list.append(coords)

                # 7. 批量 VQA (逻辑不变，仍然是批处理 *这张图的* 所有裁剪图)
                if crops_to_process:
                    json_str_list = _query_qwen_vl_BATCH(
                        crops_to_process, qwen_model, qwen_processor, accelerator
                    )

                    # 8. 遍历批处理结果并进行解析 (逻辑不变)
                    for json_str, coords in zip(json_str_list, box_coords_list):
                        json_obj_list = _parse_qwen_json(json_str)
                        if json_obj_list:
                            try:
                                obj_dict = json_obj_list[0]
                                obj_dict["bounding_box"] = [round(c, 2) for c in coords]
                                visual_facts.append(obj_dict)
                            except (IndexError, TypeError, KeyError) as e:
                                print(f"[Process {accelerator.process_index}]: "
                                      f"解析批处理结果时出错: {e} | JSON: {json_str}")

                # 9. 聚合此作业的结果 (逻辑不变)
                processed_metadata_list.append({
                    "question": job['prompt'],
                    "answer": job['answer'],
                    "question_wo_prompt": job['prompt'],
                    "hint": job['hint'],
                    "image": job['destination_image_path'],
                    "visual_fact": visual_facts
                })
                # --- 循环内部逻辑结束 ---

        except Exception as e:
            print(f"[Process {accelerator.process_index}]: "
                  f"处理批次 {i // REX_BATCH_SIZE} (图像 {batch_image_paths}) 时出错: {e}")
            import traceback
            traceback.print_exc()

    # --- 循环结束 ---

    print(f"[Process {accelerator.process_index}]:"
          f" 进程完成，处理了 {len(processed_metadata_list)} 个项目。")

    # (7. 收集所有结果 - 保持不变)
    print(f"[Process {accelerator.process_index}]: 正在收集结果...")
    all_results_list_of_lists = gather_object(processed_metadata_list)

    # (8. 保存 (仅在主进程上) - ★★★ 使用修复后的 GATHER 逻辑 ★★★)
    if accelerator.is_main_process:
        print("主进程 [Saving]: 正在聚合和保存所有结果...")

        # --- 关键修复 ---
        # gather_object 已经返回了一个扁平化的字典列表 (List[dict])。
        final_metadata_list = all_results_list_of_lists
        # --- 修复结束 ---

        json_filename = os.path.join(JSON_OUTPUT_DIR, "clevr_cogent_trainA_r1_processed.json")

        # 验证一下数量
        print(f"聚合后的项目总数: {len(final_metadata_list)}")
        if len(final_metadata_list) > 0:
            print(f"第一个项目的类型: {type(final_metadata_list[0])}")

        print(f"\n正在将 {len(final_metadata_list)} 条元数据保存到 {json_filename}...")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(final_metadata_list, f, indent=4, ensure_ascii=False)

        print(f"\n--- 处理完成！ ---")
        print(f"所有图像文件已保存在: '{IMAGE_OUTPUT_DIR}'")
        print(f"最终 JSON 文件已保存在: '{JSON_OUTPUT_DIR}'")


if __name__ == "__main__":
    main()