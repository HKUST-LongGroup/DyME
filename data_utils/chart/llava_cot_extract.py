import shutil
from datasets import load_dataset
import json
import os
import re
from PIL import Image
from tqdm import tqdm


def parse_gpt_response(response_text):
    """
    从 GPT 的响应文本中解析出 answer 和 hint。
    """
    hint = response_text
    match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", response_text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        answer = ""
    return answer, hint


def process_and_save_llava_cot(
        source_images_root,
        base_output_dir
):
    """
    加载 Xkev/LLaVA-CoT-100k 数据集，从本地读取图像，筛选 chartqa 部分，
    并将其处理成所需的格式，同时正确保存图像和元数据。
    """
    # 1. 定义输出目录
    image_output_dir = os.path.join(base_output_dir, "images")
    json_output_dir = os.path.join(base_output_dir, "json")

    # 2. 创建目录
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    print(f"源图像根目录: {os.path.abspath(source_images_root)}")
    print(f"处理后图像将保存到: {image_output_dir}")
    print(f"处理后JSON将保存到: {json_output_dir}")

    # 3. 加载数据集元数据
    print("正在加载 Xkev/LLaVA-CoT-100k 数据集元数据...")
    try:
        dataset = load_dataset("Xkev/LLaVA-CoT-100k", split='train')
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    # 4. --- 关键修正点 #1 ---
    #    直接在 example['image'] (字符串) 中进行查找
    print("正在筛选 'chartqa/train/' 的样本...")
    chartqa_dataset = dataset.filter(lambda example: 'chartqa/train/' in example['image'])
    print(f"筛选后剩余样本数: {len(chartqa_dataset)}")

    # 5. 遍历筛选后的数据集
    metadata_list = []
    for example in tqdm(chartqa_dataset, desc="正在处理 chartqa 样本"):

        # --- 关键修正点 #2 ---
        #    example['image'] 本身就是我们需要的相对路径字符串
        relative_path = example['image']
        source_image_path = os.path.join(source_images_root, relative_path)

        if not os.path.exists(source_image_path):
            print(f"\n警告: 源图像未找到，已跳过: {source_image_path}")
            continue

        conversations = example["conversations"]
        conv_iter = iter(conversations)

        for human_conv in conv_iter:
            try:
                gpt_conv = next(conv_iter)
            except StopIteration:
                continue

            if human_conv.get("from") != "human" or gpt_conv.get("from") != "gpt":
                continue

            question = human_conv["value"]
            if ' Answer the question using a single word or phrase.' in question:
                question = question.replace(' Answer the question using a single word or phrase.', '')

            answer, hint = parse_gpt_response(gpt_conv["value"])

            if not answer:
                continue

            destination_image_path = os.path.join(image_output_dir, relative_path)
            destination_dir = os.path.dirname(destination_image_path)
            os.makedirs(destination_dir, exist_ok=True)

            if not os.path.exists(destination_image_path):
                try:
                    shutil.copy(source_image_path, destination_image_path)
                except Exception as e:
                    print(f"\n保存图像 {destination_image_path} 失败: {e}")
                    continue

            metadata_list.append({
                "question": question,
                "question_wo_prompt": question,
                "answer": answer,
                "hint": hint,
                "image": destination_image_path,
            })

    # 8. 将所有元数据写入一个 JSON 文件
    json_filename = os.path.join(json_output_dir, "chartqa_train_processed.json")
    print(f"\n正在将 {len(metadata_list)} 条元数据保存到 {json_filename}...")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=4, ensure_ascii=False)

    print(f"\n--- 处理完成！ ---")
    print(f"所有图像文件已保存在: '{image_output_dir}'")
    print(f"所有 JSON 文件已保存在: '{json_output_dir}'")


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None

    # --- 如何运行 ---
    # 1. 设置您在第一步中解压图像的文件夹的路径
    #    这个路径应该指向 unzipped_images 文件夹
    SOURCE_IMAGES_ROOT_DIR = "/path/to/chartqa_output/llavacot/LLaVA-CoT-100k/unzipped_images"

    # 2. 设置您希望保存处理后数据的输出目录
    OUTPUT_DIR = "/path/to/data/chartqa_output/llavacot"

    # 3. 调用主函数
    process_and_save_llava_cot(
        source_images_root=SOURCE_IMAGES_ROOT_DIR,
        base_output_dir=OUTPUT_DIR
    )