#!/usr/bin/env python
# -*- coding: utf-8 -*-
from transformers import activations
activations.PytorchGELUTanh = activations.GELUTanh
import os
import json
from PIL import Image
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 导入模型包装器
try:
    from rex_omni import RexOmniWrapper
except ImportError:
    # 导入与 clevr_processor.py 中匹配的 DummyRex
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
        return images, None

from clevr_processor import ClevrFactExtractor, _strip_tags


def run_test(configs, paths, gpu_id=0, sample_index=0):
    """
    在单个样本上运行测试流水线。
    """
    print("--- 开始单次运行测试 (CoGenT) ---")

    # --- 1. 设置环境并加载模型 ---
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"设置 CUDA_VISIBLE_DEVICES={gpu_id}")

    try:
        print(f"正在加载 RexOmni... ({configs['rex_path']})")
        rex_model = RexOmniWrapper(
            model_path=configs['rex_path'],
            backend="transformers",
            max_tokens=2048,
            temperature=0.0,
        )

        print(f"正在加载 Qwen-VL... ({configs['qwen_path']})")
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            configs['qwen_path'],
            torch_dtype="float16",
            device_map="cuda:0",
            attn_implementation="flash_attention_2"
        )
        qwen_processor = AutoProcessor.from_pretrained(configs['qwen_path'])

        print("模型加载完毕。")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    print("正在加载数据集元数据...")
    try:
        dataset = load_dataset("MMInstruction/Clevr_CoGenT_TrainA_R1", split='train', streaming=True)
        example_iter = iter(dataset)
        for _ in range(sample_index + 1):
            example = next(example_iter)

    except Exception as e:
        print(f"加载或筛选数据集失败: {e}")
        return

    print(f"正在处理样本 {sample_index}...")

    try:
        # 1. 预处理
        prompt = example['problem']
        hint = _strip_tags(example['thinking'], 'think')
        answer = _strip_tags(example['solution'], 'answer')
        image = example['image'].convert("RGB")  # 获取 PIL 图像并转为 RGB

        # 为测试保存图像
        destination_image_path = os.path.join(paths['output_dir'], "images", f"test_sample_{sample_index}.jpg")
        os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
        image.save(destination_image_path, "JPEG")
        print(f"已加载并保存测试图像: {destination_image_path}")

        # --- 阶段 1: RexOmni 检测 ---
        print("正在运行 RexOmni 检测...")
        rex_results = rex_model.inference(images=image, task="detection", categories=["anything"])
        predictions = rex_results[0]["extracted_predictions"]
        detected_boxes = predictions.get("anything", [])
        print(f"RexOmni 检测到 {len(detected_boxes)} 个 'anything' 框。")

        visual_facts = []

        # --- 阶段 2: Qwen-VL VQA ---
        for i, annotation in enumerate(detected_boxes):
            if annotation.get("type") == "box" and len(annotation.get("coords", [])) == 4:

                coords = annotation["coords"]
                print(f"  正在处理框 {i}: {coords}")

                crop_image = ClevrFactExtractor._crop_and_expand_box(image, coords)

                # 为调试而保存裁剪的图像
                crop_filename = f"./test_crop_{sample_index}_{i}.jpg"
                crop_image.save(crop_filename)
                print(f"    -> 已保存裁剪图像以便检查: {crop_filename}")

                json_str = ClevrFactExtractor._query_qwen_vl(
                    crop_image, qwen_model, qwen_processor
                )

                json_obj_list = ClevrFactExtractor._parse_qwen_json(json_str)

                if json_obj_list:
                    obj_dict = json_obj_list[0]
                    obj_dict["bounding_box"] = [round(c, 2) for c in coords]
                    visual_facts.append(obj_dict)
                    print(f"    -> Qwen-VL 结果: {obj_dict}")
                else:
                    print(f"    -> Qwen-VL 未返回有效的 JSON。")

        # --- 4. 打印最终结果 ---
        final_result = {
            "prompt": prompt,
            "answer": answer,
            "hint": hint,
            "image": destination_image_path,
            "visual_fact": visual_facts
        }

        print("\n" + "=" * 30)
        print("--- 单次测试结果 ---")
        print(json.dumps(final_result, indent=4, ensure_ascii=False))
        print("=" * 30 + "\n")

    except Exception as e:
        print(f"处理样本 {sample_index} 时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # --- 1. 配置模型 ---
    MODEL_CONFIGS = {
        "rex_path": "IDEA-Research/Rex-Omni",
        "qwen_path": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
    }

    # --- 2. 配置数据路径 ---
    PATHS = {
        # !! 修改为你希望保存图像和 JSON 的目录 !!
        "output_dir": "./clevr_cogent_output"
    }

    # --- 3. 配置测试参数 ---
    GPU_ID_TO_USE = 0
    SAMPLE_INDEX_TO_TEST = 0  # 测试第一个 CLEVR 样本

    # --- 4. 运行测试 ---
    run_test(
        configs=MODEL_CONFIGS,
        paths=PATHS,
        gpu_id=GPU_ID_TO_USE,
        sample_index=SAMPLE_INDEX_TO_TEST
    )