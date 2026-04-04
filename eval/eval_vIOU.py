import json
import re
import numpy as np
from scipy.optimize import linear_sum_assignment


def calc_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def eval_viou(file_path):
    total_iou, acc_50_count, total = 0.0, 0, 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pred_str = data.get('prediction', '')

            true_objs = data.get('solution', {}).get('objects', [])
            gt_boxes = [obj['box'] if isinstance(obj, dict) else obj for obj in true_objs]
            if not gt_boxes: continue
            total += 1

            pred_boxes = []
            for match in re.findall(r'<box>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box>', pred_str):
                pred_boxes.append([int(match[0]), int(match[1]), int(match[2]), int(match[3])])

            if not pred_boxes: continue

            M, N = len(pred_boxes), len(gt_boxes)
            iou_mat = np.zeros((M, N))
            for i, p in enumerate(pred_boxes):
                for j, g in enumerate(gt_boxes):
                    iou_mat[i, j] = calc_iou(p, g)

            row_ind, col_ind = linear_sum_assignment(-iou_mat)
            miou = iou_mat[row_ind, col_ind].sum() / N

            total_iou += miou
            if miou >= 0.5 and len(pred_boxes) >= N:
                acc_50_count += 1

    if total > 0:
        print(
            f"[vIoU Eval] Total: {total} | mIoU: {(total_iou / total) * 100:.2f}% | Acc@0.5: {(acc_50_count / total) * 100:.2f}%")


if __name__ == "__main__":
    eval_viou("eval_results.jsonl")