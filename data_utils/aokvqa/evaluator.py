# --- 新增：A-OKVQA 评估函数 ---
def eval_aokvqa_direct(prediction, ground_truth_list):
    """
    简单的 VQA 准确率：如果预测在 direct_answers 列表中，则为 1，否则为 0。
    执行不区分大小写的匹配。
    """
    if not ground_truth_list:  # 处理空的真实答案列表
        return 0.0

    # 清理预测答案
    pred_cleaned = prediction.lower().strip().strip('.').strip()

    # 创建一个清理过的真实答案集合
    gt_cleaned_set = set(ans.lower().strip().strip('.').strip() for ans in ground_truth_list)

    # 检查预测是否在集合中
    if pred_cleaned in gt_cleaned_set:
        return 1.0

    # （可选）您可以添加更复杂的 VQA 匹配逻辑，但简单的“in”检查是基础
    return 0.0