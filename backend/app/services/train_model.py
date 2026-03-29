"""
FairLens AI — Model Training
Uses: DistilBERT + HuggingFace Trainer API
Input: data/processed/train.csv, val.csv, test.csv
Output: saved model in saved_model/
"""

# ── BLOCK 1: Imports ───────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import evaluate                              # for accuracy, precision, recall

from transformers import (
    AutoTokenizer,                           # converts text → numbers
    AutoModelForSequenceClassification,      # DistilBERT with a classifier head
    TrainingArguments,                       # all training settings in one place
    Trainer,                                 # handles the training loop for us
    EarlyStoppingCallback                    # stops training if no improvement
)
from torch.utils.data import Dataset         # PyTorch dataset class


# ── BLOCK 2: Settings ──────────────────────────────────────────────────────
# Change these to experiment later

MODEL_NAME  = "distilbert-base-uncased"   # lightweight BERT — good for laptops
MAX_LENGTH  = 128                          # max tokens per sentence
BATCH_SIZE  = 8                            # how many samples per training step
                                           # (keep at 8 for CPU — higher = crashes)
EPOCHS      = 3                            # how many times to loop through data
LEARNING_RATE = 2e-5                       # how fast the model learns
DATA_DIR    = "data/processed"            # where your CSVs are
SAVE_DIR    = "saved_model"               # where to save the trained model


# ── BLOCK 3: Load your preprocessed CSVs ──────────────────────────────────
print("=" * 50)
print("STEP 1 — Loading preprocessed data")
print("=" * 50)

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df   = pd.read_csv(f"{DATA_DIR}/val.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/test.csv")

# Drop rows with missing text or label (safety check)
train_df = train_df.dropna(subset=["text", "label"])
val_df   = val_df.dropna(subset=["text", "label"])
test_df  = test_df.dropna(subset=["text", "label"])

print(f"  Train samples : {len(train_df)}")
print(f"  Val   samples : {len(val_df)}")
print(f"  Test  samples : {len(test_df)}")


# ── BLOCK 4: Load tokenizer ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2 — Loading tokenizer")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"  Tokenizer loaded: {MODEL_NAME}")


# ── BLOCK 5: Create a PyTorch Dataset class ────────────────────────────────
# The Trainer needs data in a special "Dataset" format.
# This class wraps our CSV data into that format.

class BiasDataset(Dataset):
    def __init__(self, dataframe):
        self.texts  = dataframe["text"].tolist()    # list of sentences
        self.labels = dataframe["label"].tolist()   # list of 0s and 1s

    def __len__(self):
        # tells PyTorch how many samples we have
        return len(self.texts)

    def __getitem__(self, idx):
        # called for each sample during training
        # tokenizes ONE sentence and returns it with its label
        text = str(self.texts[idx])

        encoding = tokenizer(
            text,
            truncation=True,       # cut if too long
            padding="max_length",  # pad if too short
            max_length=MAX_LENGTH,
            return_tensors="pt"    # return PyTorch tensors
        )

        return {
            # squeeze() removes the extra dimension tokenizer adds
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         self.labels[idx]
        }

# Create dataset objects for each split
train_dataset = BiasDataset(train_df)
val_dataset   = BiasDataset(val_df)
test_dataset  = BiasDataset(test_df)

print("\n" + "=" * 50)
print("STEP 3 — Datasets ready")
print("=" * 50)
print(f"  Train dataset : {len(train_dataset)} samples")
print(f"  Val   dataset : {len(val_dataset)} samples")
print(f"  Test  dataset : {len(test_dataset)} samples")

# Show what one sample looks like
sample = train_dataset[0]
print(f"\n  Sample keys   : {list(sample.keys())}")
print(f"  input_ids shape : {sample['input_ids'].shape}")
print(f"  label           : {sample['labels']}")


# ── BLOCK 6: Load the model ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4 — Loading DistilBERT model")
print("=" * 50)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2           # 2 classes: 0 = not biased, 1 = biased
)

# Count how many parameters the model has
total_params = sum(p.numel() for p in model.parameters())
print(f"  Model loaded: {MODEL_NAME}")
print(f"  Total parameters: {total_params:,}")
print(f"  Running on: CPU")


# ── BLOCK 7: Define metrics ────────────────────────────────────────────────
# This function is called automatically after each epoch
# to evaluate how well the model is doing

accuracy_metric  = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric    = evaluate.load("recall")
f1_metric        = evaluate.load("f1")

def compute_metrics(eval_pred):
    """
    eval_pred contains:
      - logits : raw model outputs (e.g. [-0.5, 1.2])
      - labels : actual correct answers (0 or 1)
    """
    logits, labels = eval_pred

    # Convert logits → predictions by picking the higher value
    # e.g. [-0.5, 1.2] → 1 (biased)
    predictions = np.argmax(logits, axis=1)

    acc  = accuracy_metric.compute( predictions=predictions, references=labels)
    prec = precision_metric.compute(predictions=predictions, references=labels)
    rec  = recall_metric.compute(   predictions=predictions, references=labels)
    f1   = f1_metric.compute(       predictions=predictions, references=labels)

    return {
        "accuracy" : round(acc["accuracy"],   4),
        "precision": round(prec["precision"], 4),
        "recall"   : round(rec["recall"],     4),
        "f1"       : round(f1["f1"],          4),
    }


# ── BLOCK 8: Training settings ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5 — Setting up training")
print("=" * 50)

os.makedirs(SAVE_DIR, exist_ok=True)       # create save folder

training_args = TrainingArguments(
    output_dir=SAVE_DIR,                   # save checkpoints here

    # ── Training duration
    num_train_epochs=EPOCHS,               # 3 full passes over training data
    per_device_train_batch_size=BATCH_SIZE,# 8 samples at a time (CPU-friendly)
    per_device_eval_batch_size=BATCH_SIZE,

    # ── Learning rate
    learning_rate=LEARNING_RATE,           # 2e-5 is standard for fine-tuning BERT
    weight_decay=0.01,                     # regularization (prevents overfitting)
    warmup_ratio=0.1,                      # gradually increase lr at the start

    # ── Evaluation & saving
    eval_strategy="epoch",                 # evaluate after every epoch
    save_strategy="epoch",                 # save checkpoint after every epoch
    load_best_model_at_end=True,           # keep the best checkpoint
    metric_for_best_model="f1",            # use F1 to decide "best"

    # ── Logging
    logging_dir=f"{SAVE_DIR}/logs",
    logging_steps=50,                      # print loss every 50 steps
    report_to="none",                      # don't send to wandb/tensorboard

    # ── CPU optimization
    use_cpu=True,                          # force CPU (no GPU needed)
    dataloader_num_workers=0,              # Windows needs this set to 0
    fp16=False,                            # fp16 only works on GPU
)

print(f"  Epochs         : {EPOCHS}")
print(f"  Batch size     : {BATCH_SIZE}")
print(f"  Learning rate  : {LEARNING_RATE}")
print(f"  Save directory : {SAVE_DIR}")


# ── BLOCK 9: Create Trainer and start training ─────────────────────────────
print("\n" + "=" * 50)
print("STEP 6 — Starting training")
print("  (Each epoch will take 10-30 min on CPU — this is normal!)")
print("=" * 50)

trainer = Trainer(
    model=model,                           # our DistilBERT model
    args=training_args,                    # all the settings above
    train_dataset=train_dataset,           # training data
    eval_dataset=val_dataset,              # validation data
    compute_metrics=compute_metrics,       # our metrics function
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2)
        # stops training if F1 doesn't improve for 2 epochs in a row
    ]
)

# Start training!
trainer.train()


# ── BLOCK 10: Evaluate on test set ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 7 — Final evaluation on test set")
print("=" * 50)

results = trainer.evaluate(test_dataset)

print(f"\n  Final Test Results:")
print(f"  Accuracy  : {results.get('eval_accuracy',  'N/A')}")
print(f"  Precision : {results.get('eval_precision', 'N/A')}")
print(f"  Recall    : {results.get('eval_recall',    'N/A')}")
print(f"  F1 Score  : {results.get('eval_f1',        'N/A')}")


# ── BLOCK 11: Save the final model ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 8 — Saving model")
print("=" * 50)

# Save model weights + config
model.save_pretrained(SAVE_DIR)

# Save tokenizer alongside the model (needed for inference later)
tokenizer.save_pretrained(SAVE_DIR)

print(f"  Model saved to: {SAVE_DIR}/")
print(f"  Files saved:")
print(f"    config.json          — model architecture")
print(f"    model.safetensors    — trained weights")
print(f"    tokenizer.json       — tokenizer")
print(f"    vocab.txt            — vocabulary")

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("Next step → use this model in the FastAPI backend")
print("=" * 50)