from datasets import load_dataset
from collections import Counter
import json
import os
from pathlib import Path

dataset = load_dataset("banking77")
old_train_data = dataset['train']
old_test_data = dataset['test']

sampled_dataset = old_train_data.train_test_split(
    train_size=7000, 
    stratify_by_column="label", 
    seed=42 
)
train_data = sampled_dataset['train']

sampled_dataset = old_train_data.train_test_split(
    test_size=700, 
    stratify_by_column="label", 
    seed=42 
)
test_data = sampled_dataset['test']

original_counts = Counter(old_train_data['label'])
small_counts = Counter(train_data['label']) 

print("Label distribution before and after sampling:")
for i in range(dataset['train'].features['label'].num_classes):
    print(f"Label {i}: original {original_counts[i]} samples -> New has {small_counts[i]} samples")


current_file = Path(__file__).resolve() 
project_root = current_file.parent.parent 
data_dir = project_root / "sample_data"
data_dir.mkdir(parents=True, exist_ok=True)

train_data.to_csv(data_dir / "train.csv", index=False)
test_data.to_csv(data_dir / "test.csv", index=False)


label = dataset["train"].features["label"].names
id2label = {i: label for i, label in enumerate(label)}
with open("configs/label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(id2label, f, indent=4, ensure_ascii=False)