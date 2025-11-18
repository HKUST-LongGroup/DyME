import os
import json
from datasets import load_dataset
from tqdm import tqdm
import shutil


def patch_json_with_direct_answers(base_output_dir):
    """
    加载现有的 JSON 文件和原始的 A-OKVQA 数据集，
    将 'direct_answers' 字段添加回 JSON 文件中。

    此脚本假设您生成的 JSON 文件 (例如 train.json) 中的条目
    是按照原始数据集的索引顺序排序的（您原始脚本中
    metadata_list.sort(key=lambda x: x['image']) 确保了这一点）。
    """

    json_output_dir = os.path.join(base_output_dir, "json")
    print(f"将要修补的 JSON 目录: {json_output_dir}")

    # 1. 加载原始数据集
    print("正在加载 HuggingFaceM4/A-OKVQA 数据集 (仅元数据)...")
    try:
        # 指定一个缓存目录，避免重复下载
        cache_dir = os.path.join(base_output_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        dataset = load_dataset("HuggingFaceM4/A-OKVQA", cache_dir=cache_dir)
    except Exception as e:
        print(f"加载原始数据集失败: {e}")
        return

    print(f"数据集中包含的 Splits: {list(dataset.keys())}")

    # 2. 遍历每个 split
    for split in dataset.keys():
        json_filename = os.path.join(json_output_dir, f"{split}.json")

        if not os.path.exists(json_filename):
            print(f"!! 警告: 未找到 {json_filename}，跳过 {split} split。")
            continue

        print(f"\n--- 正在修补 {split} split ({json_filename}) ---")

        # 3. 加载现有的 (不完整的) JSON 数据
        try:
            with open(json_filename, 'r', encoding='utf-8') as f:
                generated_data_list = json.load(f)
            print(f"  已加载 {len(generated_data_list)} 条已处理的数据。")
        except Exception as e:
            print(f"  !! 错误: 加载 {json_filename} 失败: {e}")
            continue

        # 4. 加载原始 split 数据
        original_split_data = dataset[split]
        print(f"  已加载 {len(original_split_data)} 条原始数据。")

        # 5. 健全性检查 (确保数量一致)
        if len(generated_data_list) != len(original_split_data):
            print(f"  !! 严重错误: 数据数量不匹配! ")
            print(f"  JSON ({split}.json) 中有 {len(generated_data_list)} 条记录。")
            print(f"  原始数据集 ('{split}') 中有 {len(original_split_data)} 条记录。")
            print(f"  跳过此 split。")
            continue

        # 6. 核心逻辑：使用 zip 合并数据 (依赖排序)
        # 您的 'image' 字段 (例如 "train_0000001.png") 保证了
        # 'generated_data_list' 列表的顺序与 'original_split_data' 一致

        print(f"  正在合并 'direct_answers'...")
        for generated_metadata, original_example in \
                tqdm(zip(generated_data_list, original_split_data),
                     total=len(generated_data_list),
                     desc=f"合并 {split}"):
            # 添加 (或覆盖) 缺失的字段
            generated_metadata['direct_answers'] = original_example.get('direct_answers')

        # 7. 备份并写回 (覆盖)
        backup_filename = os.path.join(json_output_dir, f"{split}.backup.json")
        try:
            if not os.path.exists(backup_filename):  # 只备份一次
                shutil.copyfile(json_filename, backup_filename)
                print(f"  已备份原始文件到 {backup_filename}")
            else:
                print(f"  备份文件 {backup_filename} 已存在，将直接覆盖 {json_filename}")
        except Exception as e:
            print(f"  !! 警告: 备份失败: {e}。将直接覆盖。")

        print(f"  正在将 {len(generated_data_list)} 条更新后的元数据写回 {json_filename}...")
        with open(json_filename, 'w', encoding='utf-8') as f:
            # generated_data_list 已经在内存中被修改
            json.dump(generated_data_list, f, indent=4, ensure_ascii=False)

    print("\n--- 修补完成！ ---")


if __name__ == "__main__":
    # *** 确保此路径与您原始脚本中使用的路径一致 ***
    data_directory = "/apdcephfs_nj4/share_300377003/realzliu/data/aokvqa"

    print(f"目标根目录: {data_directory}")
    patch_json_with_direct_answers(base_output_dir=data_directory)