from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, classification_report
import torch
import argparse, os
from os import path
from transformers import set_seed
import random
set_seed(42)

parser = argparse.ArgumentParser(description='Finetune MuRIL checkpoint on English disfluency detection')
parser.add_argument('--data-dir', '-d', default='data/labeled_data/', type=str, help='Data Directory which contains data.dis & data.labels')
parser.add_argument('--splits', '-s', nargs='+', default=[80, 10], type=float, help=f"percentage of data for training, validation & test (optional) respectively. If percentage for test is not given, remaining data is used for testing")
parser.add_argument('--separate-files', '-sep', action='store_true', help="data-dir contains [SPLIT].dis & [SPLIT].labels where SPLIT in ['train', 'valid', 'test']")
parser.add_argument('--experiment-id', '-id', default="12345678", type=str, help='Experiment ID (8 alpha-numeric characters long)') 
args = parser.parse_args()


if len(args.splits) not in [2, 3]:
    parser.error("--splits should have 2 or 3 arguments")

if sum(args.splits) > 100:
    parser.error("Sum of all splits should never go beyond 100%")

if not args.separate_files and (not path.exists(f"{args.data_dir}/data.dis") or not path.exists(f"{args.data_dir}/data.labels")):
    parser.error(f"Invalid data-dir `{args.data_dir}`")

if args.separate_files:
    splits = ['train', 'valid', 'test']
    for split in splits:
        if not path.exists(f"{args.data_dir}/{split}.dis"):
            parser.error(f"Invalid file `{args.data_dir}/{split}.dis`")

        if not path.exists(f"{args.data_dir}/{split}.labels"):
            parser.error(f"Invalid file `{args.data_dir}/{split}.labels`")


# If test-set percentage is not provided, use remaining data as test-set
if len(args.splits) == 2:
    args.splits.append(100 - sum(args.splits))


task = "dc"
model_checkpoint = "google/muril-base-cased"
print("Model Name:", model_checkpoint)

batch_size = 16
label_list = ['is_fluent', 'is_disfluent'] # 0 -> isFluent , 1 -> isDisfluent


def get_dataset(path=args.data_dir):

    def save_split(data_dict, prefix):
        os.makedirs(os.path.join(path, "split_data"), exist_ok=True)
        with open(os.path.join(path, "split_data", f"{prefix}.dis"), "w") as dis_file, \
             open(os.path.join(path, "split_data", f"{prefix}.labels"), "w") as label_file:
             for dis, lab in zip(data_dict["disfluent"], data_dict["labels"]):
                dis_file.write(" ".join(dis) + "\n")
                label_file.write(" ".join(map(str, lab)) + "\n")

    if args.separate_files:
        train_dict = {'labels': [], 'disfluent': []}
        valid_dict = {'labels': [], 'disfluent': []}
        test_dict  = {'labels': [], 'disfluent': []}

        with open(f"{path}/train.dis", 'r') as dis, open(f"{path}/train.labels", 'r') as labels:
            disfluent_lines = dis.readlines()
            labels_lines = labels.readlines()

            for disfluent, labels in zip(disfluent_lines, labels_lines):

                disfluent = disfluent.strip().split()
                labels = list(map(int, labels.strip().split()))

                train_dict['disfluent'].append(disfluent)
                train_dict['labels'].append(labels)
        
        with open(f"{path}/valid.dis", 'r') as dis, open(f"{path}/valid.labels", 'r') as labels:
            disfluent_lines = dis.readlines()
            labels_lines = labels.readlines()

            for disfluent, labels in zip(disfluent_lines, labels_lines):

                disfluent = disfluent.strip().split()
                labels = list(map(int, labels.strip().split()))

                valid_dict['disfluent'].append(disfluent)
                valid_dict['labels'].append(labels)

        with open(f"{path}/test.dis", 'r') as dis, open(f"{path}/test.labels", 'r') as labels:
            disfluent_lines = dis.readlines()
            labels_lines = labels.readlines()

            for disfluent, labels in zip(disfluent_lines, labels_lines):

                disfluent = disfluent.strip().split()
                labels = list(map(int, labels.strip().split()))

                test_dict['disfluent'].append(disfluent)
                test_dict['labels'].append(labels)

        train_dataset = Dataset.from_dict(train_dict)
        valid_dataset = Dataset.from_dict(valid_dict)
        test_dataset = Dataset.from_dict(test_dict)


        return DatasetDict({'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset})

    with open(f"{path}/data.dis", 'r') as dis, open(f"{path}/data.labels", 'r') as labels:

        train_dict = {'labels': [], 'disfluent': []}
        valid_dict = {'labels': [], 'disfluent': []}
        test_dict  = {'labels': [], 'disfluent': []}

        disfluent_lines = dis.readlines()
        labels_lines = labels.readlines()

        paired_lines = list(zip(disfluent_lines, labels_lines))
        random.shuffle(paired_lines)
        disfluent_lines, labels_lines = zip(*paired_lines)

        total_size = len(disfluent_lines)

        for i, (disfluent, labels) in enumerate(zip(disfluent_lines, labels_lines)):

            disfluent = disfluent.strip().split()
            labels = list(map(int, labels.strip().split()))

            if i < round(args.splits[0] * total_size / 100) :
                train_dict['disfluent'].append(disfluent)
                train_dict['labels'].append(labels)
            elif i < round(sum(args.splits[0:2]) * total_size / 100):
                valid_dict['disfluent'].append(disfluent)
                valid_dict['labels'].append(labels)
            elif i < round(sum(args.splits) * total_size / 100):
                test_dict['disfluent'].append(disfluent)
                test_dict['labels'].append(labels)
            else:
                break

    # Save locally
    save_split(train_dict, "train")
    save_split(valid_dict, "valid")
    save_split(test_dict, "test")

    train_dataset = Dataset.from_dict(train_dict)
    valid_dataset = Dataset.from_dict(valid_dict)
    test_dataset = Dataset.from_dict(test_dict)

    return DatasetDict({'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset})


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["disfluent"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


datasets = get_dataset()
print(datasets)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
label_all_tokens = True
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"checkpoints/{args.experiment_id}",
    eval_strategy = "steps",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,#10,
    weight_decay=0.01,
    eval_steps=1000, #1000,
    logging_steps=1000, #1000,
    save_steps=1000, #1000,
    save_total_limit=5,
    load_best_model_at_end=True
)

print("Training Args", args)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 1-d prediction & true label
    true_predictions = [
        p for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100
    ]
    true_labels = [
        l for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100 
    ]

    results = precision_recall_fscore_support(true_labels, true_predictions, zero_division=0)
    return {
        'accuracy': accuracy_score(true_labels, true_predictions),
        'precision0': torch.tensor(results[0])[0],
        'precision1': torch.tensor(results[0])[1],
        'recall0': torch.tensor(results[1])[0],
        'recall1': torch.tensor(results[1])[1],
        'f1score0': torch.tensor(results[2])[0],
        'f1score1': torch.tensor(results[2])[1],
    }


data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()
trainer.evaluate()


# Evaluate on test sentences
predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=2)

# 1-d prediction & true label
true_predictions = [
    p for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100
]
true_labels = [
    l for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100 
]

results = precision_recall_fscore_support(true_labels, true_predictions, zero_division=0)
print({
    'precision': results[0],
    'recall': results[1],
    'f1score': results[2]
})
print("Confusion Matrix:")
print(confusion_matrix(true_labels, true_predictions, normalize='all'))
print(classification_report(true_labels, true_predictions, target_names=label_list, zero_division=0))
