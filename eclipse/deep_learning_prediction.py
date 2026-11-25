"""Compare RNN/LSTM deep-learning models for Eclipse bug priority."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from workflow_utils import (
    DEFAULT_DL_OUTPUT,
    DEFAULT_INPUT,
    TEXT_COLUMNS,
    create_output_dir,
    load_dataset,
    save_dataset_profile,
    save_run_summary,
    save_split,
    split_dataset,
)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        model_type: str,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        recurrent_type = nn.RNN if model_type == "rnn" else nn.LSTM
        self.recurrent = recurrent_type(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            self.embedding(tokens),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, state = self.recurrent(packed)
        hidden = state[0] if isinstance(state, tuple) else state
        return self.output(self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate RNN and LSTM priority classifiers."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DL_OUTPUT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    if min(
        args.epochs,
        args.batch_size,
        args.max_vocab_size,
        args.max_length,
        args.embedding_dim,
        args.hidden_dim,
    ) < 1:
        parser.error("epochs, sizes, and batch size must be positive")
    return args


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def join_text(data: pd.DataFrame) -> pd.Series:
    return data[TEXT_COLUMNS[0]] + " " + data[TEXT_COLUMNS[1]]


def build_vocabulary(texts: pd.Series, max_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counts.most_common(max_size - 2):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode(
    texts: pd.Series, vocabulary: dict[str, int], max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.zeros((len(texts), max_length), dtype=torch.long)
    lengths = torch.ones(len(texts), dtype=torch.long)
    for row, text in enumerate(texts):
        ids = [
            vocabulary.get(token, vocabulary[UNK_TOKEN])
            for token in tokenize(text)[:max_length]
        ]
        if ids:
            tokens[row, : len(ids)] = torch.tensor(ids)
            lengths[row] = len(ids)
        else:
            tokens[row, 0] = vocabulary[UNK_TOKEN]
    return tokens, lengths


def make_loader(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(tokens, lengths, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def train(
    model: RecurrentClassifier,
    loader: DataLoader,
    weights: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> list[float]:
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for tokens, lengths, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(tokens, lengths), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        epoch_loss = total_loss / len(loader.dataset)
        losses.append(epoch_loss)
        print(f"  Epoch {epoch + 1}/{args.epochs} - loss: {epoch_loss:.4f}")
    return losses


def predict(
    model: RecurrentClassifier, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for tokens, lengths, _ in loader:
            logits = model(tokens.to(device), lengths)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def metrics(
    name: str, labels: list[str], actual: list[str], predicted: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = classification_report(
        actual, predicted, labels=labels, output_dict=True, zero_division=0
    )
    summary = {
        "Model": name,
        "Accuracy": accuracy_score(actual, predicted),
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Weighted Precision": report["weighted avg"]["precision"],
        "Weighted Recall": report["weighted avg"]["recall"],
        "Weighted F1": report["weighted avg"]["f1-score"],
    }
    per_class = [
        {
            "Model": name,
            "Priority": label,
            "Precision": report[label]["precision"],
            "Recall": report[label]["recall"],
            "F1": report[label]["f1-score"],
            "Support": report[label]["support"],
        }
        for label in labels
    ]
    return summary, per_class


def main() -> None:
    args = parse_args()
    data, labels = load_dataset(args.input)
    train_indices, test_indices = split_dataset(data, args.test_size, args.seed)
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_y = data.loc[train_indices, "Priority"]
    test_y = data.loc[test_indices, "Priority"]
    train_ids = torch.tensor([label_to_id[label] for label in train_y])
    test_labels = test_y.tolist()
    train_text = join_text(data.loc[train_indices])
    test_text = join_text(data.loc[test_indices])

    vocabulary = build_vocabulary(train_text, args.max_vocab_size)
    train_tokens, train_lengths = encode(train_text, vocabulary, args.max_length)
    test_tokens, test_lengths = encode(test_text, vocabulary, args.max_length)
    train_loader = make_loader(
        train_tokens, train_lengths, train_ids, args.batch_size, True, args.seed
    )
    test_loader = make_loader(
        test_tokens,
        test_lengths,
        torch.tensor([label_to_id[label] for label in test_labels]),
        args.batch_size,
        False,
        args.seed,
    )
    class_counts = torch.bincount(train_ids, minlength=len(labels))
    weights = len(train_ids) / (len(labels) * class_counts.clamp_min(1)).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = create_output_dir(args.output_dir)
    save_dataset_profile(data, output_dir)
    save_split(data, train_indices, test_indices, output_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    summaries: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    test_predictions: dict[str, Any] = {
        "Bug_ID": data.loc[test_indices, "Bug_ID"].to_numpy(),
        "Actual Priority": test_labels,
    }
    for model_type in ("rnn", "lstm"):
        print(f"Training {model_type.upper()} on {device}...")
        model = RecurrentClassifier(
            len(vocabulary),
            len(labels),
            model_type,
            args.embedding_dim,
            args.hidden_dim,
            args.dropout,
        )
        epoch_losses = train(model, train_loader, weights, device, args)
        probabilities = predict(model, test_loader, device)
        predicted_ids = probabilities.argmax(axis=1)
        predicted_labels = [labels[int(index)] for index in predicted_ids]
        summary, per_class = metrics(
            model_type.upper(), labels, test_labels, predicted_labels
        )
        summaries.append(summary)
        class_rows.extend(per_class)
        test_predictions[f"{model_type.upper()} Prediction"] = predicted_labels
        pd.DataFrame(
            confusion_matrix(test_labels, predicted_labels, labels=labels),
            index=labels,
            columns=labels,
        ).rename_axis("Actual\\Predicted").to_csv(
            output_dir / f"confusion_matrix_{model_type}.csv"
        )
        torch.save(
            {
                "model_state_dict": model.cpu().state_dict(),
                "vocab_size": len(vocabulary),
                "num_classes": len(labels),
                "model_type": model_type,
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "max_length": args.max_length,
                "priority_labels": labels,
            },
            output_dir / f"{model_type}_model.pt",
        )
        (output_dir / f"{model_type}_vocabulary.json").write_text(
            json.dumps(vocabulary), encoding="utf-8"
        )
        pd.DataFrame(
            {"Model": model_type.upper(), "Epoch": range(1, args.epochs + 1), "Loss": epoch_losses}
        ).to_csv(output_dir / f"{model_type}_training_loss.csv", index=False)
        del model

    result_frame = pd.DataFrame(summaries).sort_values(
        ["Macro F1", "Weighted F1"], ascending=False
    )
    result_frame.to_csv(output_dir / "deep_learning_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(
        output_dir / "deep_learning_per_class.csv", index=False
    )
    pd.DataFrame(test_predictions).to_csv(
        output_dir / "held_out_test_predictions.csv", index=False
    )
    save_run_summary(
        data,
        labels,
        args.input,
        train_indices,
        test_indices,
        args.seed,
        args.test_size,
        {
            "workflow": "deep_learning_text",
            "models": ["RNN", "LSTM"],
            "epochs": args.epochs,
            "device": str(device),
            "graph_neural_network": "not run: no explicit record-level relationship columns",
        },
        output_dir,
    )
    print(f"\nRows: {len(data):,}; train: {len(train_indices):,}; test: {len(test_indices):,}")
    print("\nHeld-out test metrics:")
    print(result_frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
