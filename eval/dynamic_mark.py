"""
LABEL SUFFICIENCY JUDGEMENT AND DYNAMIC REFINEMENT WORKFLOW
Utilizes Qwen-Max for sufficiency arbitration and triggers specific Grounding DINO list for fine-grained annotation.
"""

import json

def run_labeling_refinement(image_path, initial_labels):
    """
    STAGE 1: QWEN-MAX SUFFICIENCY CHECK
    Evaluates if current bounding boxes are dense/accurate enough for data injection tasks.
    Returns: {"signal": "INSUFFICIENT", "category": "target_class"} or {"signal": "OK"}
    """
    check_result = call_qwen_max_api(image_path, initial_labels)

    if check_result["signal"] == "OK":
        return initial_labels

    """
    STAGE 2: CATEGORY-DRIVEN MODULE ROUTING
    Directly triggers specialized fine-grained labeling modules based on the category returned by the arbiter.
    """
    target_category = check_result["category"]

    if target_category == "electronics":
        import dino_electronics_specialist
        new_labels = dino_electronics_specialist.annotate(image_path)

    elif target_category == "industrial_fasteners":
        import dino_industrial_specialist
        new_labels = dino_industrial_specialist.annotate(image_path)

    elif target_category == "mechanical_parts":
        import dino_mechanical_specialist
        new_labels = dino_mechanical_specialist.annotate(image_path)

    else:
        import dino_general_specialist
        new_labels = dino_general_specialist.annotate(image_path, target_category)

    """
    STAGE 3: RESULTS AGGREGATION
    Merges original metadata with refined fine-grained labels for final injection pipeline readiness.
    """
    return initial_labels + new_labels