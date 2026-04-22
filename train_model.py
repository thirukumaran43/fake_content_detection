import pandas as pd

# Load feedback data
df = pd.read_csv("feedback_data.csv")

# Show columns to confirm structure
print("Columns in dataset:", df.columns)

# Use only WRONG feedback samples
mistakes = df[df["feedback"] == "wrong"]

# If no mistakes available
if mistakes.empty:
    print("No wrong feedback data found. Add feedback first.")
    exit()

# Use 'result' column as training text
texts = mistakes["result"].astype(str).tolist()

# Create labels based on result
labels = []
for r in mistakes["result"]:
    if str(r).upper() in ["FAKE", "LOW RISK"]:
        labels.append("FAKE")
    else:
        labels.append("REAL")

print(f"Training on {len(texts)} corrected samples")

# Preview data
for i in range(min(5, len(texts))):
    print(f"Text: {texts[i]}  |  Label: {labels[i]}")

# ------------------------------------------
# NOTE:
# This is only data preparation.
# Real model training requires HuggingFace Trainer or similar.
# ------------------------------------------

print("\nData prepared successfully. Ready for model fine‑tuning.")
