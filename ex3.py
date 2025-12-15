import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ===== config =====
SPLIT = "Long"           # or "ExtraLong"
N_SAMPLES = 500
BATCH_SIZE = 8
MAX_LENGTH = 512

CANDIDATE_LABELS = ["negative", "neutral", "positive"]

# Better, more explicit neutral definition (often improves neutral recall)
HYPOTHESIS = {
    "negative": "The attitude is negative.",
    "neutral":  "The attitude is indifferent or mixed (neither positive nor negative).",
    "positive": "The attitude is positive."
}

MODELS = [
    "roberta-large-mnli",
    "facebook/bart-large-mnli",
    "microsoft/deberta-large-mnli",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
]

# ===== data =====
print("Loading OpenToM dataset...")
dataset = load_dataset("SeacowX/OpenToM", split=SPLIT)

def is_attitude(ex):
    return ex["question"]["type"] == "attitude"

attitude_ds = dataset.filter(is_attitude)
n = min(N_SAMPLES, len(attitude_ds))
attitude_ds = attitude_ds.select(range(n))

y_true = [q["answer"] for q in attitude_ds["question"]]
print(f"Split={SPLIT}, attitude samples used={n}")

# ===== helpers =====
def build_premise(narrative: str, question_text: str) -> str:
    # Keep it factual (avoid instruction-like text)
    return narrative + "\n\n" + question_text

def get_nli_label_ids(model):
    label2id = model.config.label2id
    entail_id = None
    contra_id = None
    for name, idx in label2id.items():
        low = name.lower()
        if "entail" in low:
            entail_id = idx
        if "contrad" in low:
            contra_id = idx
    # reasonable fallbacks
    if entail_id is None:
        entail_id = max(label2id.values())
    if contra_id is None:
        contra_id = min(label2id.values())
    return entail_id, contra_id

def eval_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    entail_id, contra_id = get_nli_label_ids(model)

    preds = []
    for start in tqdm(range(0, len(attitude_ds), BATCH_SIZE), desc=model_name, leave=False):
        batch = attitude_ds[start:start+BATCH_SIZE]
        narratives = batch["narrative"]
        questions = batch["question"]

        premises = [
            build_premise(narratives[i], questions[i]["question"])
            for i in range(len(narratives))
        ]

        all_premises, all_hypos = [], []
        for p in premises:
            for lbl in CANDIDATE_LABELS:
                all_premises.append(p)
                all_hypos.append(HYPOTHESIS[lbl])

        enc = tokenizer(
            all_premises,
            all_hypos,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits  # [batch*num_labels, nli_labels]

        num_examples = len(premises)
        num_labels = len(CANDIDATE_LABELS)
        logits = logits.view(num_examples, num_labels, -1)

        # stronger score: entailment - contradiction
        scores = logits[:, :, entail_id] - logits[:, :, contra_id]
        best = torch.argmax(scores, dim=-1).cpu().tolist()
        preds.extend([CANDIDATE_LABELS[i] for i in best])

    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro")
    return acc, macro_f1

# ===== run all =====
results = []
for m in MODELS:
    try:
        acc, mf1 = eval_model(m)
        results.append((m, acc, mf1))
        print(f"{m}:  acc={acc:.4f}  macroF1={mf1:.4f}")
    except Exception as e:
        print(f"{m}: FAILED -> {e}")

print("\nLeaderboard (best acc first):")
for m, acc, mf1 in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"{m:60s}  acc={acc:.4f}  macroF1={mf1:.4f}")
