# ── BLOCK 1: Import libraries ──────────────────────────────────────────────
# These are tools we need. Think of them like apps we're opening.

import re                        # for finding/replacing patterns in text
import string                    # gives us a list of all punctuation symbols
import numpy as np               # for working with arrays of numbers
import pandas as pd              # for working with tables (like Excel)
import matplotlib.pyplot as plt  # for drawing charts
import os                        # for creating folders

from datasets import load_dataset          # downloads dataset from Hugging Face
from sklearn.model_selection import train_test_split  # splits data into train/test
from sklearn.utils import resample         # fixes imbalanced data
from transformers import AutoTokenizer    # converts text → numbers for BERT


# ── BLOCK 2: Settings ──────────────────────────────────────────────────────
# These are variables you can change to experiment later.

MODEL_NAME   = "distilbert-base-uncased"  # the AI model we'll use
MAX_LENGTH   = 128                         # max words per sentence (longer = cut off)
RANDOM_STATE = 42                          # makes random operations repeatable
OUTPUT_DIR   = "data/processed"           # folder where we save our output


# ── BLOCK 3: Load the dataset ──────────────────────────────────────────────
print("Loading dataset... (this may take a minute)")

# Download the dataset directly from Hugging Face
raw = load_dataset("allenai/social_bias_frames", trust_remote_code=True)

# Convert each split (train/val/test) from HuggingFace format to a pandas table
train_df = raw["train"].to_pandas()
val_df   = raw["validation"].to_pandas()
test_df  = raw["test"].to_pandas()

# Combine all three into one big table
df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"Total rows loaded: {len(df)}")       # should print ~147,000
print(f"Columns: {list(df.columns)}")        # shows all column names


# ── BLOCK 4: Keep only what we need ────────────────────────────────────────
# The dataset has 20 columns. We only need 2:
#   'post'        = the actual text (what someone wrote online)
#   'offensiveYN' = how offensive it is (1.0 = yes, 0.5 = maybe, 0.0 = no)

df = df[["post", "offensiveYN"]].copy()
print("\nSample rows:")
print(df.head())


# ── BLOCK 5: Turn offensiveYN into a simple 0 or 1 label ──────────────────
# Machine learning needs simple numbers, not 0.5 or 1.0

def make_label(value):
    # Try to convert the value to a decimal number
    try:
        score = float(value)
        if score >= 0.5:
            return 1   # offensive (or maybe offensive)
        else:
            return 0   # not offensive
    except:
        return None    # if the value is missing/broken, return nothing

# Apply the function to every row in the offensiveYN column
df["label"] = df["offensiveYN"].apply(make_label)

# Remove rows where we couldn't figure out the label
df = df.dropna(subset=["label"])

# Convert label from decimal to integer (1.0 → 1, 0.0 → 0)
df["label"] = df["label"].astype(int)

print(f"\nLabel counts:")
print(df["label"].value_counts())
# 0 = not offensive, 1 = offensive


# ── BLOCK 6: Remove duplicate posts ────────────────────────────────────────
# The same post was rated by multiple people, creating duplicate rows.
# We keep one row per post, using the majority vote of all raters.

df = (
    df.groupby("post", as_index=False)
      .agg(label=("label", lambda x: int(x.mean() >= 0.5)))
)

print(f"\nUnique posts after deduplication: {len(df)}")


# ── BLOCK 7: Clean the text ─────────────────────────────────────────────────
# Raw social media text is messy. We clean it before feeding it to the model.

def clean_text(text):
    if not isinstance(text, str):  # skip if not a string
        return ""

    text = text.lower()                                        # UPPERCASE → lowercase
    text = re.sub(r"http\S+|www\S+", "", text)                # remove URLs
    text = re.sub(r"@", "", text)                              # remove @ symbol
    text = re.sub(r"#", "", text)                              # remove # symbol
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove .,!? etc
    text = re.sub(r"\s+", " ", text)                           # remove extra spaces
    text = text.strip()                                        # remove leading/trailing space
    return text

# Apply the cleaning function to every post
df["clean_text"] = df["post"].apply(clean_text)

# Remove any posts that became empty after cleaning
df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

# Show a before/after example
print("\nCleaning example:")
print("BEFORE:", df["post"].iloc[0])
print("AFTER :", df["clean_text"].iloc[0])


# ── BLOCK 8: Save a chart of class distribution ────────────────────────────
# Let's see how many offensive vs non-offensive posts we have

os.makedirs(OUTPUT_DIR, exist_ok=True)   # create output folder if it doesn't exist

label_counts = df["label"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(
    ["Not Offensive (0)", "Offensive (1)"],
    [label_counts.get(0, 0), label_counts.get(1, 0)],
    color=["#4A90D9", "#E05C4B"]
)
plt.title("Class Distribution")
plt.ylabel("Number of posts")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png")
plt.show()
print("Chart saved!")


# ── BLOCK 9: Split into train / val / test ─────────────────────────────────
# Train = model learns from this (70%)
# Val   = we check progress during training (15%)
# Test  = final check after training (15%)

X = df["clean_text"].to_numpy(dtype=str)    # X = the input text (features)
y = df["label"].to_numpy(dtype=int)         # y = the label (what we want to predict)

# First: cut off 15% for test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)

# Then: cut 15% from the remaining for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.176, random_state=RANDOM_STATE, stratify=y_trainval
)

print(f"\nSplit sizes:")
print(f"  Train : {len(X_train)} samples")
print(f"  Val   : {len(X_val)} samples")
print(f"  Test  : {len(X_test)} samples")


# ── BLOCK 10: Fix class imbalance (oversampling) ───────────────────────────
# If one class has far more examples, the model gets lazy and always predicts that class.
# Solution: duplicate some examples from the smaller class until both are equal.

train_df_temp = pd.DataFrame({"text": X_train, "label": y_train})

class_0 = train_df_temp[train_df_temp["label"] == 0]   # not offensive
class_1 = train_df_temp[train_df_temp["label"] == 1]   # offensive

# Find the bigger class
bigger_size = max(len(class_0), len(class_1))

# Oversample whichever is smaller
class_0_balanced = resample(class_0, replace=True, n_samples=bigger_size, random_state=RANDOM_STATE)
class_1_balanced = resample(class_1, replace=True, n_samples=bigger_size, random_state=RANDOM_STATE)

# Combine and shuffle
train_balanced = pd.concat([class_0_balanced, class_1_balanced]).sample(frac=1, random_state=RANDOM_STATE)

X_train = train_balanced["text"].values
y_train = train_balanced["label"].values

print(f"\nAfter balancing:")
print(f"  Class 0: {sum(y_train == 0)}")
print(f"  Class 1: {sum(y_train == 1)}")


# ── BLOCK 11: Tokenization ─────────────────────────────────────────────────
# Computers don't understand words — they understand numbers.
# Tokenization converts "hello world" → [101, 7592, 2088, 102]
# This is what BERT/DistilBERT expects as input.

print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts, labels):
    # Convert list of strings → token IDs + attention masks
    encoded = tokenizer(
        list(texts),
        truncation=True,        # cut off if longer than MAX_LENGTH
        padding="max_length",   # pad with zeros if shorter than MAX_LENGTH
        max_length=MAX_LENGTH,
        return_tensors="np"     # return as numpy arrays
    )
    return {
        "input_ids":      encoded["input_ids"],       # the token numbers
        "attention_mask": encoded["attention_mask"],  # 1=real token, 0=padding
        "labels":         np.array(labels)            # 0 or 1
    }

print("Tokenizing train set...")
train_tokens = tokenize(X_train, y_train)

print("Tokenizing val set...")
val_tokens = tokenize(X_val, y_val)

print("Tokenizing test set...")
test_tokens = tokenize(X_test, y_test)

print(f"\nToken shape (train): {train_tokens['input_ids'].shape}")
print("This means:", train_tokens['input_ids'].shape[0], "posts,",
      train_tokens['input_ids'].shape[1], "tokens each")


# ── BLOCK 12: Save everything ──────────────────────────────────────────────

# Save text + labels as CSV (easy to open in Excel)
pd.DataFrame({"text": X_train, "label": y_train}).to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
pd.DataFrame({"text": X_val,   "label": y_val  }).to_csv(f"{OUTPUT_DIR}/val.csv",   index=False)
pd.DataFrame({"text": X_test,  "label": y_test }).to_csv(f"{OUTPUT_DIR}/test.csv",  index=False)

# Save tokenized arrays (for model training later)
np.save(f"{OUTPUT_DIR}/train_tokens.npy", train_tokens)
np.save(f"{OUTPUT_DIR}/val_tokens.npy",   val_tokens)
np.save(f"{OUTPUT_DIR}/test_tokens.npy",  test_tokens)

print("\n✅ Done! Files saved to:", OUTPUT_DIR)
print("   train.csv, val.csv, test.csv")
print("   train_tokens.npy, val_tokens.npy, test_tokens.npy")