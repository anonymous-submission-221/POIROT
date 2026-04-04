import json
import re
import string
import difflib
from collections import Counter

def normalize_ans(s):
    if not s: return ""
    s = s.lower()
    s = ''.join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def calc_f1(pred, gt):
    pred_toks, gt_toks = normalize_ans(pred).split(), normalize_ans(gt).split()
    common = Counter(pred_toks) & Counter(gt_toks)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    p, r = num_same / len(pred_toks), num_same / len(gt_toks)
    return (2 * p * r) / (p + r)

def eval_qa(file_path):
    em, f1, seq_sim, total = 0.0, 0.0, 0.0, 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pred_str = data.get('prediction', '')
            gt = str(data.get('solution', {}).get('answer', '')).strip()
            if not gt: continue

            match = re.search(r'<action>(.*?)</action>', pred_str, re.DOTALL)
            pred = match.group(1).strip() if match else pred_str[-100:]

            em += int(normalize_ans(pred) == normalize_ans(gt))
            f1 += calc_f1(pred, gt)
            seq_sim += difflib.SequenceMatcher(None, gt.lower(), pred.lower()).ratio()
            total += 1

    if total > 0:
        print(f"[QA Eval] Total: {total} | EM: {em/total*100:.2f}% | F1: {f1/total*100:.2f}% | Sim: {seq_sim/total*100:.2f}%")

if __name__ == "__main__":
    eval_qa("eval_results.jsonl")