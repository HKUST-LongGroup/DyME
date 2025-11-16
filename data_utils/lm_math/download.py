from datasets import load_dataset
import json
import os


def save_gsm8k_as_json(base_output_dir="/gsm8k_output"):
    """
    加载 openai/gsm8k 数据集。
    将 'question' 和 'answer' 字段提取出来，
    并为每个 split (train, test) 生成一个 JSON 元数据文件。
    """

    # 1. 定义输出目录
    # gsm8k 是纯文本，所以我们只需要一个 json 目录
    base_dir_abs = os.path.abspath(base_output_dir)
    json_output_dir = '/apdcephfs_nj4/share_300377003/realzliu/data/lm_math/json_new'

    # 2. 创建目录
    os.makedirs(json_output_dir, exist_ok=True)

    print(f"JSON 将保存到 (绝对路径): {json_output_dir}")

    # 3. 加载数据集
    print("正在加载 openai/gsm8k 数据集 (main config)...")
    try:
        # 'main' config 是 gsm8k 的标准配置
        dataset = load_dataset("ankner/gsm8k-CoT")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    print(f"数据集中包含的 Splits: {list(dataset.keys())}")

    # 4. 遍历每个 split (例如 'train', 'test')
    for split in dataset.keys():
        print(f"\n--- 正在处理 {split} split ---")

        metadata_list = []
        total_count = len(dataset[split])

        # 5. 遍历数据集中的每个样本
        for i, example in enumerate(dataset[split]):

            # 提取元数据
            question = example["question"]
            hint = example["response"]
            pure_answer = example["answer"]

            # 6. (无图像步骤)
            # gsm8k 不包含图像，所以我们不需要保存图像或生成路径。
            # 我们直接将文本数据添加到列表中。

            # 7. 将元数据添加到列表中
            metadata_list.append({
                "question": question,
                "answer": pure_answer,
                "hint": hint,
            })

            # 打印进度
            if (i + 1) % 1000 == 0 or (i + 1) == total_count:
                print(f"  已处理 {split} split: {i + 1} / {total_count}")

        # 8. 将该 split 的所有元数据写入 JSON 文件
        json_filename = os.path.join(json_output_dir, f"{split}.json")
        print(f"正在将 {len(metadata_list)} 条元数据保存到 {json_filename}...")

        with open(json_filename, 'w', encoding='utf-8') as f:
            # indent=2 (或 4) 使 JSON 文件更易读
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"\n--- 处理完成！ ---")
    print(f"所有 JSON 文件已保存在: '{json_output_dir}'")


if __name__ == "__main__":
    # gsm8k 是纯文本，不需要 PIL.Image.MAX_IMAGE_PIXELS
    save_gsm8k_as_json()