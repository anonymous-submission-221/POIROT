import re
import difflib
import numpy as np
import requests
from scipy.optimize import linear_sum_assignment
from swift.rewards import ORM, orms


def calculate_iou(boxA, boxB):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


class FormatRewardFunc(ORM):
    """Format Reward: Strict matching for OTA architecture."""

    def __call__(self, completions, **kwargs):
        rewards = []
        for comp in completions:
            score = 0.0
            comp_str = str(comp)
            if not (re.search(r'<observe>', comp_str) and re.search(r'<action>', comp_str)):
                rewards.append(-1.0)
                continue

            if re.search(r'<observe>.*?</observe>', comp_str, re.DOTALL): score += 0.3
            if re.search(r'<think>.*?</think>', comp_str, re.DOTALL): score += 0.3
            if re.search(r'<action>.*?</action>', comp_str, re.DOTALL): score += 0.4
            rewards.append(score)
        return rewards


class IoURewardFunc(ORM):
    """Visual Grounding Reward (SG-GDPO R_iou): Bipartite graph Hungarian matching on boxes."""

    def __call__(self, completions, **kwargs):
        solutions = kwargs.get('solution', [])
        rewards = []

        for comp, sol in zip(completions, solutions):
            comp_str = str(comp)
            pred_boxes = []

            pattern = r'<box>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box>'
            for match in re.findall(pattern, comp_str):
                pred_boxes.append([int(match[0]), int(match[1]), int(match[2]), int(match[3])])

            true_objs = sol.get('objects', [])
            true_boxes = [obj['box'] if isinstance(obj, dict) else obj for obj in true_objs]

            if not true_boxes and not pred_boxes:
                rewards.append(1.0)
                continue
            if not true_boxes or not pred_boxes:
                rewards.append(0.0)
                continue

            M, N = len(pred_boxes), len(true_boxes)
            iou_matrix = np.zeros((M, N))

            for i, p_box in enumerate(pred_boxes):
                for j, t_box in enumerate(true_boxes):
                    iou_matrix[i, j] = calculate_iou(p_box, t_box)

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            total_iou = iou_matrix[row_ind, col_ind].sum()

            rewards.append(total_iou / N)

        return rewards


class GenerativeAccuracyRewardFunc(ORM):
    """Generative Accuracy Reward: Evaluate final text answer."""

    def __call__(self, completions, **kwargs):
        solutions = kwargs.get('solution', [])
        rewards = []

        for comp, sol in zip(completions, solutions):
            comp_str = str(comp)
            match = re.search(r'<action>(.*?)</action>', comp_str, re.DOTALL)
            pred_ans = match.group(1).strip() if match else comp_str[-100:]
            true_ans = str(sol.get('answer', '')).strip()

            if not true_ans:
                rewards.append(0.0)
                continue

            matcher = difflib.SequenceMatcher(None, true_ans.lower(), pred_ans.lower())
            rewards.append(matcher.ratio())

        return rewards


class LogicalConsistencyRewardFunc(ORM):
    """Logical Consistency Reward (Think Reward): Local PRM (Qwen2.5-1.5B) RPC call."""

    def __call__(self, completions, **kwargs):
        rewards = []
        PRM_ENDPOINT = "http://localhost:8000/v1/chat/completions"

        for comp in completions:
            comp_str = str(comp)
            observe_match = re.search(r'<observe>(.*?)</observe>', comp_str, re.DOTALL)
            think_match = re.search(r'<think>(.*?)</think>', comp_str, re.DOTALL)

            if not think_match or not observe_match:
                rewards.append(0.0)
                continue

            observe_content = observe_match.group(1).strip()
            think_content = think_match.group(1).strip()

            prompt = (
                "Verify if the logical deduction is strictly consistent with the objective observations. "
                "Output ONLY '1' if logically consistent, or '0' if there are hallucinations.\n\n"
                f"[Observation]:\n{observe_content}\n\n"
                f"[Thinking]:\n{think_content}"
            )

            try:
                response = requests.post(
                    PRM_ENDPOINT,
                    json={
                        "model": "Qwen2.5-1.5B-Instruct",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 5,
                        "temperature": 0.0
                    },
                    timeout=2.0
                )

                if response.status_code == 200:
                    prm_output = response.json()["choices"][0]["message"]["content"].strip()
                    reward = 0.8 if '1' in prm_output else 0.0
                else:
                    reward = 0.0
            except Exception as e:
                reward = 0.0

            rewards.append(reward)

        return rewards


orms['format_reward_func'] = FormatRewardFunc
orms['class_aware_iou_reward_func'] = IoURewardFunc
orms['generative_accuracy_reward_func'] = GenerativeAccuracyRewardFunc
orms['logical_consistency_reward_func'] = LogicalConsistencyRewardFunc