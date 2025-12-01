import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

#   - "roberta-large-mnli"          (strong, heavier)
#   - "facebook/bart-large-mnli"    (a bit faster)
#   - "roberta-base-mnli"           (smaller & faster)

MODEL_NAME = "roberta-large-mnli"

N_SAMPLES = 500

BATCH_SIZE = 4

MAX_LENGTH = 512

CANDIDATE_LABELS = ["negative", "neutral", "positive"]
HYPOTHESIS_TEMPLATE = "The attitude is {}."

print("Loading OpenToM dataset...")
dataset = load_dataset("SeacowX/OpenToM", split="Long")

# Filter attitude questions
def is_attitude(example):
    return example["question"]["type"] == "attitude"

attitude_ds = dataset.filter(is_attitude)
print("Total attitude samples:", len(attitude_ds))

n = min(N_SAMPLES, len(attitude_ds))
attitude_ds = attitude_ds.select(range(n))
print("Using samples:", n)

print(f"\nLoading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Device:", device)

label2id = model.config.label2id
entailment_id = None
for name, idx in label2id.items():
    if "entail" in name.lower():
        entailment_id = idx
        break
if entailment_id is None:
    entailment_id = max(label2id.values())  

print("Model label2id:", label2id)
print("Using entailment index:", entailment_id)

def build_premise(narrative: str, question_text: str) -> str:
    """
    Build the premise string passed to the NLI model.
    You can tweak the wording if you like.
    """
    return (
        narrative
        + "\n\nQuestion: "
        + question_text
        + "\nAnswer with one of: negative, neutral, positive."
    )


def predict_zero_shot_batch(batch, candidate_labels, hypo_template):
    """
    Manual zero-shot NLI:
      - For each example and each label, create (premise, hypothesis)
      - Run through NLI model
      - Use entailment logit as score
      - Return predicted label per example
    """
    narratives = batch["narrative"]
    questions = batch["question"]

    premises = [
        build_premise(narratives[i], questions[i]["question"])
        for i in range(len(narratives))
    ]

    all_premises = []
    all_hypotheses = []

    for p in premises:
        for lbl in candidate_labels:
            all_premises.append(p)
            all_hypotheses.append(hypo_template.format(lbl))

    enc = tokenizer(
        all_premises,
        all_hypotheses,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits  

    num_examples = len(premises)
    num_labels = len(candidate_labels)

    logits = logits.view(num_examples, num_labels, -1)

    entail_logits = logits[:, :, entailment_id] 
    
    best_idx = torch.argmax(entail_logits, dim=-1).cpu().tolist()
    preds = [candidate_labels[i] for i in best_idx]
    return preds


all_preds = []
all_true = [q["answer"] for q in attitude_ds["question"]]

print("\nRunning zero-shot evaluation on attitude questions...")
for start in tqdm(range(0, len(attitude_ds), BATCH_SIZE)):
    batch = attitude_ds[start : start + BATCH_SIZE]
    preds = predict_zero_shot_batch(
        batch,
        candidate_labels=CANDIDATE_LABELS,
        hypo_template=HYPOTHESIS_TEMPLATE,
    )
    all_preds.extend(preds)

assert len(all_preds) == len(all_true)


print("\nUnique gold labels:", sorted(set(all_true)))
print("Unique predicted labels:", sorted(set(all_preds)))

acc = accuracy_score(all_true, all_preds)
macro_f1 = f1_score(all_true, all_preds, average="macro")

print(f"\nAccuracy (attitude, {n} samples): {acc:.4f}")
print(f"Macro F1:                         {macro_f1:.4f}\n")

print("Detailed classification report:\n")
print(classification_report(all_true, all_preds, digits=4))

