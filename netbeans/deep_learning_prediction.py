"""Train recurrent and graph neural networks for NetBeans bug priority."""

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_INPUT = Path(__file__).with_name("merged_bug_reports.csv")
DEFAULT_OUTPUT = Path(__file__).with_name("deep_learning_results")
TEXT_COLUMNS = ["Summary", "Keywords"]
CATEGORY_COLUMNS = ["Product", "Component", "OS", "Severity"]
REQUIRED_COLUMNS = {"Bug ID", "Priority", *TEXT_COLUMNS, *CATEGORY_COLUMNS}
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
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.recurrent(packed)
        hidden = state[0] if isinstance(state, tuple) else state
        return self.output(self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1)))


class MeanGraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim * 2, output_dim)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        sources, targets = edge_index
        neighbor_sums = torch.zeros_like(features)
        neighbor_sums.index_add_(0, targets, features[sources])
        degrees = torch.bincount(targets, minlength=len(features)).to(features.dtype)
        neighbor_means = neighbor_sums / degrees.clamp_min(1).unsqueeze(1)
        return self.linear(torch.cat((features, neighbor_means), dim=1))


class BugReportGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.first = MeanGraphConvolution(input_dim, hidden_dim)
        self.second = MeanGraphConvolution(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.first(features, edges))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = F.relu(self.second(hidden, edges))
        return self.output(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RNN, LSTM, and GNN bug-priority models."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--text-components", type=int, default=96)
    parser.add_argument("--metadata-features", type=int, default=64)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Backward-compatible text-only mode: use Summary only.",
    )
    parser.add_argument(
        "--feature-preset",
        choices=("summary_component", "with_keywords", "with_product"),
        default=None,
        help="Controlled feature ablation. The base always uses Summary and Component.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional label for the ablation folder and run summary.",
    )
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
        args.neighbors,
        args.text_components,
        args.metadata_features,
    ) < 1:
        parser.error("model sizes, epochs, batch size, and neighbors must be positive")
    return args


def create_output_dir(requested: Path) -> Path:
    if not requested.exists():
        requested.mkdir(parents=True)
        return requested
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = requested.with_name(f"{requested.name}_{timestamp}")
    suffix = 2
    while candidate.exists():
        candidate = requested.with_name(f"{requested.name}_{timestamp}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_vocabulary(texts: pd.Series, max_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counts.most_common(max_size - len(vocabulary)):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_texts(
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


def class_weight_tensor(train_ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(train_ids, minlength=num_classes)
    return len(train_ids) / (num_classes * counts.clamp_min(1)).float()


def train_recurrent(
    model: RecurrentClassifier,
    train_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
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
        print(
            f"  Epoch {epoch + 1}/{args.epochs} "
            f"- loss: {epoch_loss / len(train_loader.dataset):.4f}"
        )


def predict_recurrent(
    model: RecurrentClassifier, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for tokens, lengths, _ in loader:
            logits = model(tokens.to(device), lengths)
            batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(batches, axis=0)


def build_graph_edges(
    train_tfidf: Any,
    test_tfidf: Any,
    train_components: np.ndarray,
    test_components: np.ndarray,
    neighbors_per_node: int,
) -> torch.Tensor:
    train_groups: dict[str, list[int]] = defaultdict(list)
    for row, component in enumerate(train_components):
        train_groups[str(component)].append(row)

    edges: set[tuple[int, int]] = set()
    for rows in train_groups.values():
        if len(rows) < 2:
            continue
        count = min(neighbors_per_node + 1, len(rows))
        search = NearestNeighbors(n_neighbors=count, metric="cosine")
        search.fit(train_tfidf[rows])
        nearest = search.kneighbors(train_tfidf[rows], return_distance=False)
        for local_row, candidates in enumerate(nearest):
            source = rows[local_row]
            for candidate in candidates:
                target = rows[int(candidate)]
                if source != target:
                    edges.add((source, target))
                    edges.add((target, source))

    test_groups: dict[str, list[int]] = defaultdict(list)
    for row, component in enumerate(test_components):
        if str(component) in train_groups:
            test_groups[str(component)].append(row)
    train_count = len(train_components)
    for component, test_rows in test_groups.items():
        candidates = train_groups[component]
        count = min(neighbors_per_node, len(candidates))
        search = NearestNeighbors(n_neighbors=count, metric="cosine")
        search.fit(train_tfidf[candidates])
        nearest = search.kneighbors(test_tfidf[test_rows], return_distance=False)
        for local_row, candidate_indices in enumerate(nearest):
            test_node = train_count + test_rows[local_row]
            for candidate_index in candidate_indices:
                edges.add((candidates[int(candidate_index)], test_node))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([sources, targets], dtype=torch.long)


def evaluate(
    model_name: str,
    labels: list[str],
    true_ids: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]], list[str]]:
    predicted_ids = probabilities.argmax(axis=1)
    actual = [labels[int(index)] for index in true_ids]
    predicted = [labels[int(index)] for index in predicted_ids]
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    binary_labels = label_binarize(actual, classes=labels)
    metrics: dict[str, object] = {
        "Model": model_name,
        "Accuracy": accuracy_score(actual, predicted),
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Weighted Precision": report["weighted avg"]["precision"],
        "Weighted Recall": report["weighted avg"]["recall"],
        "Weighted F1": report["weighted avg"]["f1-score"],
        "MAP (Macro Avg Precision, OVR)": average_precision_score(
            binary_labels, probabilities, average="macro"
        ),
    }
    per_class = [
        {
            "Model": model_name,
            "Priority": label,
            "Precision": report[label]["precision"],
            "Recall": report[label]["recall"],
            "F1": report[label]["f1-score"],
            "Support": report[label]["support"],
        }
        for label in labels
    ]
    matrix = confusion_matrix(actual, predicted, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).rename_axis(
        "Actual\\Predicted"
    ).to_csv(output_dir / f"confusion_matrix_{model_name.lower()}.csv")
    return metrics, per_class, predicted


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.input}")
    source = pd.read_csv(args.input)
    missing_columns = REQUIRED_COLUMNS - set(source.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    data = source.copy()
    if args.text_only:
        args.feature_preset = "summary_only"
    elif args.feature_preset is None:
        args.feature_preset = "summary_component"
    feature_preset = args.feature_preset

    data["Priority"] = data["Priority"].astype("string").str.strip()
    data = data.loc[data["Priority"].notna() & data["Priority"].ne("")].reset_index(
        drop=True
    )
    for column in TEXT_COLUMNS:
        data[column] = data[column].fillna("").astype(str).str.strip()
    for column in CATEGORY_COLUMNS:
        data[column] = (
            data[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    labels = sorted(data["Priority"].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_indices, test_indices = train_test_split(
        np.arange(len(data)),
        test_size=args.test_size,
        stratify=data["Priority"],
        random_state=args.seed,
    )
    train_y = data.loc[train_indices, "Priority"]
    test_y = data.loc[test_indices, "Priority"]
    train_ids = torch.tensor(
        [label_to_id[label] for label in train_y], dtype=torch.long
    )
    test_ids = np.asarray([label_to_id[label] for label in test_y], dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.experiment_name:
        args.output_dir = args.output_dir / args.experiment_name
    output_dir = create_output_dir(args.output_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_text = data.loc[train_indices, "Summary"].copy()
    test_text = data.loc[test_indices, "Summary"].copy()
    if feature_preset != "summary_only":
        train_text = train_text + " " + data.loc[train_indices, "Component"].fillna("").astype(str)
        test_text = test_text + " " + data.loc[test_indices, "Component"].fillna("").astype(str)
    if feature_preset == "with_keywords":
        train_text = train_text + " " + data.loc[train_indices, "Keywords"].fillna("").astype(str)
        test_text = test_text + " " + data.loc[test_indices, "Keywords"].fillna("").astype(str)
    if feature_preset == "with_product":
        train_text = train_text + " " + data.loc[train_indices, "Product"].fillna("").astype(str)
        test_text = test_text + " " + data.loc[test_indices, "Product"].fillna("").astype(str)
    vocabulary = build_vocabulary(train_text, args.max_vocab_size)
    train_tokens, train_lengths = encode_texts(
        train_text, vocabulary, args.max_length
    )
    test_tokens, test_lengths = encode_texts(test_text, vocabulary, args.max_length)
    train_loader = make_loader(
        train_tokens, train_lengths, train_ids, args.batch_size, True, args.seed
    )
    test_loader = make_loader(
        test_tokens,
        test_lengths,
        torch.tensor(test_ids, dtype=torch.long),
        args.batch_size,
        False,
        args.seed,
    )
    class_weights = class_weight_tensor(train_ids, len(labels))
    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []

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
        train_recurrent(model, train_loader, class_weights, device, args)
        probabilities = predict_recurrent(model, test_loader, device)
        metrics, per_class, predicted = evaluate(
            model_type.upper(),
            labels,
            test_ids,
            probabilities,
            output_dir,
        )
        summary_rows.append(metrics)
        class_rows.extend(per_class)
        data.loc[test_indices, f"{model_type.upper()} Predicted Priority"] = predicted
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
        del model

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )
    train_tfidf = vectorizer.fit_transform(train_text)
    test_tfidf = vectorizer.transform(test_text)
    component_count = min(
        args.text_components, train_tfidf.shape[1] - 1, train_tfidf.shape[0] - 1
    )
    if component_count < 1:
        raise ValueError("Not enough training text features for graph-model SVD.")
    svd = TruncatedSVD(n_components=component_count, random_state=args.seed)
    train_text_features = svd.fit_transform(train_tfidf).astype(np.float32)
    test_text_features = svd.transform(test_tfidf).astype(np.float32)
    if feature_preset == "summary_only":
        graph_category_columns = []
    elif feature_preset == "with_product":
        graph_category_columns = ["Component", "Product"]
    else:
        graph_category_columns = ["Component"]
    categories = data[graph_category_columns].astype(str)
    category_dicts = [
        {f"{column}={row[column]}": 1 for column in graph_category_columns}
        for _, row in categories.iterrows()
    ]
    hasher = FeatureHasher(
        n_features=args.metadata_features,
        input_type="dict",
        alternate_sign=False,
    )
    metadata = hasher.transform(category_dicts).toarray().astype(np.float32)
    ordered_positions = np.concatenate((train_indices, test_indices))
    all_features = np.concatenate(
        (
            np.concatenate((train_text_features, test_text_features), axis=0),
            metadata[ordered_positions],
        ),
        axis=1,
    )
    edges = build_graph_edges(
        train_tfidf,
        test_tfidf,
        data.loc[train_indices, "Component"].to_numpy(),
        data.loc[test_indices, "Component"].to_numpy(),
        args.neighbors,
    )
    features_tensor = torch.tensor(all_features, dtype=torch.float32, device=device)
    edges = edges.to(device)
    gnn = BugReportGNN(
        all_features.shape[1], args.hidden_dim, len(labels), args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    train_ids_device = train_ids.to(device)
    print(
        f"Training GNN on {device} (nodes: {len(data):,}; "
        f"directed edges: {edges.shape[1]:,})..."
    )
    for epoch in range(args.epochs):
        gnn.train()
        optimizer.zero_grad(set_to_none=True)
        logits = gnn(features_tensor, edges)
        loss = criterion(logits[: len(train_indices)], train_ids_device)
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch + 1}/{args.epochs} - loss: {loss.item():.4f}")
    gnn.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(
            gnn(features_tensor, edges)[len(train_indices) :], dim=1
        ).cpu().numpy()
    metrics, per_class, predicted = evaluate(
        "GNN", labels, test_ids, probabilities, output_dir
    )
    summary_rows.append(metrics)
    class_rows.extend(per_class)
    data.loc[test_indices, "GNN Predicted Priority"] = predicted
    torch.save(
        {
            "model_state_dict": gnn.cpu().state_dict(),
            "input_dim": all_features.shape[1],
            "hidden_dim": args.hidden_dim,
            "num_classes": len(labels),
            "dropout": args.dropout,
            "priority_labels": labels,
            "graph_definition": (
                "training nodes connect to same-component training neighbors; "
                "test nodes receive messages only from training nodes"
            ),
        },
        output_dir / "gnn_model.pt",
    )
    import joblib

    joblib.dump(
        {"tfidf_vectorizer": vectorizer, "svd": svd, "feature_hasher": hasher},
        output_dir / "gnn_preprocessors.joblib",
    )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["Macro F1", "Weighted F1"], ascending=False
    )
    summary.to_csv(output_dir / "deep_learning_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(
        output_dir / "deep_learning_per_class.csv", index=False
    )
    pd.DataFrame(
        {
            "Bug ID": data.loc[test_indices, "Bug ID"].to_numpy(),
            "Actual Priority": [labels[index] for index in test_ids],
            "RNN Predicted Priority": data.loc[
                test_indices, "RNN Predicted Priority"
            ].to_numpy(),
            "LSTM Predicted Priority": data.loc[
                test_indices, "LSTM Predicted Priority"
            ].to_numpy(),
            "GNN Predicted Priority": data.loc[
                test_indices, "GNN Predicted Priority"
            ].to_numpy(),
        }
    ).to_csv(output_dir / "test_rows.csv", index=False)
    pd.DataFrame(
        {
            "Bug ID": data.loc[train_indices, "Bug ID"].to_numpy(),
            "Priority": train_y.to_numpy(),
            "Split": "train",
        }
    ).to_csv(output_dir / "train_rows.csv", index=False)
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "rows": len(data),
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "target": "Priority",
                "labels": labels,
                "seed": args.seed,
                "device": str(device),
                "feature_preset": feature_preset,
                "experiment_name": args.experiment_name,
                "feature_text": (
                    ["Summary"]
                    if feature_preset == "summary_only"
                    else ["Summary", "Component"]
                    + (["Keywords"] if feature_preset == "with_keywords" else [])
                    + (["Product"] if feature_preset == "with_product" else [])
                ),
                "graph_metadata": graph_category_columns,
                "excluded_columns": [
                    column
                    for column in source.columns
                    if column not in (["Summary"] + (["Component"] if feature_preset != "summary_only" else []) + (["Keywords"] if feature_preset == "with_keywords" else []) + (["Product"] if feature_preset == "with_product" else []) + ["Priority"])
                ],
                "graph_edges": int(edges.shape[1]),
                "models": ["RNN", "LSTM", "GNN"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "model_config.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nHeld-out test metrics:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nOutputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
