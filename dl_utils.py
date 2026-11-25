import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from model_utils import PRIORITY_LABELS, add_common_arguments, load_and_split


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class RecurrentTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        model_type: str,
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
        self.output = nn.Linear(hidden_dim * 2, len(PRIORITY_LABELS))

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.recurrent(packed)
        hidden = state[0] if isinstance(state, tuple) else state
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.output(self.dropout(features))


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_vocabulary(texts: pd.Series, max_vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counts.most_common(max_vocab_size - len(vocabulary)):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_texts(
    texts: pd.Series, vocabulary: dict[str, int], max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = torch.zeros((len(texts), max_length), dtype=torch.long)
    lengths = torch.ones(len(texts), dtype=torch.long)
    for row, text in enumerate(texts):
        token_ids = [
            vocabulary.get(token, vocabulary[UNK_TOKEN])
            for token in tokenize(text)[:max_length]
        ]
        if token_ids:
            encoded[row, : len(token_ids)] = torch.tensor(token_ids)
            lengths[row] = len(token_ids)
        else:
            encoded[row, 0] = vocabulary[UNK_TOKEN]
    return encoded, lengths


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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for tokens, lengths, labels in train_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens, lengths), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss / len(train_loader.dataset):.4f}")


def predict_probabilities(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for tokens, lengths, batch_labels in loader:
            logits = model(tokens.to(device), lengths)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def save_evaluation(
    model_name: str,
    output_dir: Path,
    true_ids: np.ndarray,
    probabilities: np.ndarray,
    test_indices: pd.Index,
    data: pd.DataFrame,
) -> pd.DataFrame:
    predicted_ids = probabilities.argmax(axis=1)
    true_labels = [PRIORITY_LABELS[index] for index in true_ids]
    predicted_labels = [PRIORITY_LABELS[index] for index in predicted_ids]
    class_scores = classification_report(
        true_labels,
        predicted_labels,
        labels=PRIORITY_LABELS,
        output_dict=True,
        zero_division=0,
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=PRIORITY_LABELS,
        zero_division=0,
    )
    binary_labels = label_binarize(true_labels, classes=PRIORITY_LABELS)
    metrics = pd.DataFrame(
        [
            {
                "Model": model_name.upper(),
                "Accuracy": accuracy_score(true_labels, predicted_labels),
                "Macro Precision": precision.mean(),
                "Macro Recall (MAR)": recall.mean(),
                "Macro F1": f1.mean(),
                "Weighted Precision": class_scores["weighted avg"]["precision"],
                "Weighted Recall": class_scores["weighted avg"]["recall"],
                "Weighted F1": class_scores["weighted avg"]["f1-score"],
                "MAP (Macro Avg Precision, OVR)": average_precision_score(
                    binary_labels, probabilities, average="macro"
                ),
            }
        ]
    )
    metrics.to_csv(output_dir / "held_out_test_metrics.csv", index=False)
    pd.DataFrame.from_dict(
        {label: class_scores[label] for label in PRIORITY_LABELS}, orient="index"
    ).rename_axis("Priority").to_csv(output_dir / "held_out_test_per_class.csv")
    pd.DataFrame(class_scores).transpose().to_csv(
        output_dir / "held_out_test_classification_report.csv"
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=PRIORITY_LABELS)
    pd.DataFrame(matrix, index=PRIORITY_LABELS, columns=PRIORITY_LABELS).rename_axis(
        "Actual\\Predicted"
    ).to_csv(output_dir / "held_out_test_confusion_matrix.csv")
    pd.DataFrame(
        {
            "Bug ID": data.loc[test_indices, "Bug ID"].to_numpy(),
            "Actual Priority": true_labels,
            "Predicted Priority": predicted_labels,
        }
    ).to_csv(output_dir / "held_out_test_predictions.csv", index=False)
    return metrics


def run_recurrent_model(model_type: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Train and evaluate the {model_type.upper()} bug priority model."
    )
    add_common_arguments(parser)
    parser.set_defaults(output_dir=Path(__file__).with_name("deep_learning_results"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1:
        parser.error("epochs, batch-size, and max-length must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_dir / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    data, train_x, test_x, train_y, test_y = load_and_split(
        argparse.Namespace(
            input=args.input,
            folds=args.folds,
            test_size=args.test_size,
            seed=args.seed,
            output_dir=output_dir,
        )
    )
    baseline_split = Path(__file__).with_name("baseline_results") / "fixed_train_test_split.csv"
    saved_split = output_dir / "fixed_train_test_split.csv"
    if baseline_split.exists():
        expected = pd.read_csv(baseline_split).sort_values("Bug ID").reset_index(drop=True)
        actual = pd.read_csv(saved_split).sort_values("Bug ID").reset_index(drop=True)
        if not expected.equals(actual):
            raise ValueError("Generated split does not match baseline_results/fixed_train_test_split.csv")

    vocabulary = build_vocabulary(train_x, args.max_vocab_size)
    train_tokens, train_lengths = encode_texts(train_x, vocabulary, args.max_length)
    test_tokens, test_lengths = encode_texts(test_x, vocabulary, args.max_length)
    label_to_id = {label: index for index, label in enumerate(PRIORITY_LABELS)}
    train_ids = torch.tensor([label_to_id[label] for label in train_y], dtype=torch.long)
    test_ids = torch.tensor([label_to_id[label] for label in test_y], dtype=torch.long)
    counts = np.bincount(train_ids.numpy(), minlength=len(PRIORITY_LABELS))
    class_weights = torch.tensor(
        len(train_ids) / (len(PRIORITY_LABELS) * counts), dtype=torch.float32
    )
    train_loader = make_loader(
        train_tokens, train_lengths, train_ids, args.batch_size, True, args.seed
    )
    test_loader = make_loader(
        test_tokens, test_lengths, test_ids, args.batch_size, False, args.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentTextClassifier(
        len(vocabulary),
        args.embedding_dim,
        args.hidden_dim,
        model_type,
        args.dropout,
    )
    print(f"Model: {model_type.upper()} | device: {device}")
    print(f"Rows: {len(data):,}; train: {len(train_y):,}; test: {len(test_y):,}")
    print("Training class counts:", dict(zip(PRIORITY_LABELS, counts.tolist())))
    train_model(
        model,
        train_loader,
        class_weights,
        device,
        args.epochs,
        args.learning_rate,
    )
    actual_ids, probabilities = predict_probabilities(model, test_loader, device)
    metrics = save_evaluation(
        model_type, output_dir, actual_ids, probabilities, test_x.index, data
    )
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "vocab_size": len(vocabulary),
            "model_type": model_type,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "max_length": args.max_length,
            "priority_labels": PRIORITY_LABELS,
        },
        output_dir / f"{model_type}_model.pt",
    )
    (output_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary), encoding="utf-8"
    )
    config: dict[str, Any] = vars(args).copy()
    config.update({"model_type": model_type, "device": str(device)})
    (output_dir / "model_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    print("\nHeld-out test metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Outputs written to: {output_dir.resolve()}")