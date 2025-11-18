# --- New: A-OKVQA evaluation function ---
def eval_aokvqa_direct(prediction, ground_truth_list):
    """
    Simple VQA accuracy: returns 1 if the prediction is in the direct_answers list, otherwise 0.
    Matching is case-insensitive.
    """
    if not ground_truth_list:  # Handle empty ground truth list
        return 0.0

    # Clean predicted answer
    pred_cleaned = prediction.lower().strip().strip('.').strip()

    # Create a cleaned set of ground truth answers
    gt_cleaned_set = set(ans.lower().strip().strip('.').strip() for ans in ground_truth_list)

    # Check if prediction is in the set
    if pred_cleaned in gt_cleaned_set:
        return 1.0

    # (Optional) You can add more sophisticated VQA matching logic, but simple "in" check is the baseline
    return 0.0
