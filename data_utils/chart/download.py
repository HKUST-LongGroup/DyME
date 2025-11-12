from datasets import load_dataset
import json
import os
from PIL import Image


def save_chartqa_with_absolute_paths(base_output_dir="/chartqa_output"):
    """
    加载 HuggingFaceM4/ChartQA 数据集。
    为每张 PIL 图像生成一个唯一文件名，将图像保存到本地，
    并生成一个包含 *绝对图像路径* 的 JSON 元数据文件。
    """

    # 1. 定义输出目录
    # 使用 os.path.abspath() 来获取完整的绝对路径
    base_dir_abs = os.path.abspath(base_output_dir)
    image_output_dir = os.path.join(base_dir_abs, "images")
    json_output_dir = os.path.join(base_dir_abs, "json")

    # 2. 创建目录
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    print(f"图像将保存到 (绝对路径): {image_output_dir}")
    print(f"JSON 将保存到 (绝对路径): {json_output_dir}")

    # 3. 加载数据集
    print("正在加载 HuggingFaceM4/ChartQA 数据集 (可能需要一些时间)...")
    try:
        # trust_remote_code=True 是 ChartQA 所需的
        dataset = load_dataset("HuggingFaceM4/ChartQA")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    print(f"数据集中包含的 Splits: {list(dataset.keys())}")

    # 4. 遍历每个 split (例如 'train', 'test')
    for split in dataset.keys():
        print(f"\n--- 正在处理 {split} split ---")

        metadata_list = []
        total_count = len(dataset[split])

        # 5. 使用 enumerate 获取索引 (i)，用于生成唯一文件名
        for i, example in enumerate(dataset[split]):

            # 提取元数据
            question = example["query"]
            answer = example["label"][0]

            # 提取 PIL 图像对象
            pil_image = example["image"]

            # --- 关键步骤 ---
            # (A) 生成一个唯一的文件名
            #     使用6位零填充 (e.g., 000001) 确保文件排序正确
            generated_filename = f"{split}_{i:06d}.png"

            # (B) 构建 *绝对* 保存路径
            image_save_path = os.path.join(image_output_dir, generated_filename)

            # 6. 保存图像到该绝对路径
            # 检查图像是否已存在，避免重复保存
            if not os.path.exists(image_save_path):
                try:
                    # 保存 PIL 图像到本地
                    pil_image.save(image_save_path)
                except Exception as e:
                    print(f"保存图像 {image_save_path} 失败: {e}")
                    continue

                    # 7. 将包含 *绝对路径* 的元数据添加到列表中
            #    我们使用 "image_path" 作为键，以明确表示这是路径
            metadata_list.append({
                "question": question,
                "question_wo_prompt": question,
                "answer": answer,
                "image": image_save_path,
                "human_or_machine": example.get("human_or_machine", 0)
            })

            # 打印进度
            if (i + 1) % 500 == 0 or (i + 1) == total_count:
                print(f"  已处理 {split} split: {i + 1} / {total_count}")

        # 8. 将该 split 的所有元数据写入 JSON 文件
        json_filename = os.path.join(json_output_dir, f"{split}.json")
        print(f"正在将 {len(metadata_list)} 条元数据保存到 {json_filename}...")

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)

    print(f"\n--- 处理完成！ ---")
    print(f"所有图像文件已保存在: '{image_output_dir}'")
    print(f"所有 JSON 文件已保存在: '{json_output_dir}'")


if __name__ == "__main__":
    # 确保 PIL 能够处理可能的大型图像
    Image.MAX_IMAGE_PIXELS = None

    save_chartqa_with_absolute_paths()