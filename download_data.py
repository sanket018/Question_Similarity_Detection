from datasets import load_dataset
import os
import pandas as pd

print("Loading Quora Question Pairs dataset...")

dataset = load_dataset("nyu-mll/glue", "qqp")

os.makedirs("data", exist_ok=True)

df = pd.DataFrame(dataset["train"][:])

df = df.rename(columns={
    "question1": "question1",
    "question2": "question2",
    "label": "is_duplicate"
})

df.to_csv("data/quora_questions.csv", index=False)

print("Dataset saved successfully!")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())