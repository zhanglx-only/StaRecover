# reward_model.py
import torch
import os  # Read the process environment.
import re
import Levenshtein
from transformers import AutoTokenizer, AutoModel

# Read LOCAL_RANK and select the process device. Use rank 0 for single-device runs.
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

# Select a CUDA device when one is available.
device = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"
print(f"[Reward Model] Process (LOCAL_RANK={LOCAL_RANK}) is using device: {device}")

tokenizer_model = os.environ.get("CODEBERT_MODEL_PATH")
if not tokenizer_model:
    raise RuntimeError("Set CODEBERT_MODEL_PATH to the CodeBERT model directory.")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
# Load the model on the selected device.
model = AutoModel.from_pretrained(tokenizer_model).to(device)
model.eval()


def normalize(s: str):
    return s.lower().strip()


def encode_semantic(name):
    """Encode a name as a case-insensitive semantic vector."""
    name = normalize(name)
    # Keep input tensors on the same device as the model.
    inputs = tokenizer(name, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1)
    return emb / emb.norm(dim=-1, keepdim=True)


def levenshtein_sim(a, b):
    a, b = normalize(a), normalize(b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return (max_len - Levenshtein.distance(a, b)) / max_len


def tokenize_name(name):
    name = normalize(name).replace('_', ' ')
    return re.findall(r'[a-z]+', name)


def token_overlap_sim(a, b):
    ta, tb = set(tokenize_name(a)), set(tokenize_name(b))
    return len(ta & tb) / max(1, len(tb))


def extract_values_pairs(s):
    pairs = []
    for pair in s.split(','):
        if ':' in pair:
            k, v = pair.split(':', 1)
            pairs.append((normalize(k), normalize(v)))
    return pairs


def grpo_hybrid_reward(pred, gt):
    # === Step 1: Safely parse key-value pairs. ===
    try:
        pred_pairs = extract_values_pairs(pred)
        gt_pairs = extract_values_pairs(gt)

        pred_dict = dict(pred_pairs)
        gt_dict = dict(gt_pairs)

        pred_keys = set(pred_dict.keys())
        gt_keys = set(gt_dict.keys())

    except Exception:
        # Parsing failed (for example, malformed or duplicate keys).
        return 0.0

    # === Step 2: Require exactly matching key sets. ===
    if pred_keys != gt_keys:
        return 0.0  # Invalid format: keys do not match.

    rewards = []

    for key, gt_val in gt_dict.items():
        pred_val = pred_dict.get(key, "")
        if gt_val == pred_val:
            reward = 1.0
        elif gt_val == "-" or pred_val == "-":
            if gt_val == pred_val:  # Direct equality check.
                reward = 1.0
            else:
                reward = 0.0
        else:
            r_edit = levenshtein_sim(pred_val, gt_val)
            r_token = token_overlap_sim(pred_val, gt_val)

            emb_p = encode_semantic(pred_val)
            emb_g = encode_semantic(gt_val)
            r_sem = torch.cosine_similarity(emb_p, emb_g, dim=1).item()
            # print("*************************r_sem*****************************")
            # print(r_sem)
            # r_sem = 1.0
            reward = 0.2 * r_edit + 0.6 * r_token + 0.2 * r_sem

        rewards.append(reward)

    return sum(rewards) / len(rewards) if rewards else 0.0
