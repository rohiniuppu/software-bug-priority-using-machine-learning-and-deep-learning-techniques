"""Shared dataset preparation for the Eclipse bug-priority workflows."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_INPUT = Path(__file__).with_name("jdt_top20_500.csv")
DEFAULT_OUTPUT = Path(__file__).with_name("priority_results")
DEFAULT_DL_OUTPUT = Path(__file__).with_name("deep_learning_results")
TEXT_COLUMNS = ["Summary", "Description"]
CATEGORICAL_COLUMNS = ["Severity", "Component"]
FEATURE_COLUMNS = TEXT_COLUMNS + CATEGORICAL_COLUMNS
REQUIRED_COLUMNS = {"Bug_ID", "Priority", *FEATURE_COLUMNS}


def load_dataset(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = pd.read_csv(path, encoding="utf-8-sig")
    source_stats = {
        "missing_counts": data.isna().sum().to_dict(),
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "full_row_duplicates": int(data.duplicated().sum()),
        "row_count": len(data),
    }
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    data["Priority"] = data["Priority"].astype("string").str.strip()
    data = data.loc[data["Priority"].notna() & data["Priority"].ne("")].copy()
    for column in TEXT_COLUMNS:
        data[column] = data[column].fillna("").astype(str).str.strip()
    for column in CATEGORICAL_COLUMNS:
        data[column] = (
            data[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )
    labels = sorted(data["Priority"].unique().tolist())
    if len(labels) < 2:
        raise ValueError("Priority must contain at least two classes.")
    if data["Bug_ID"].duplicated().any():
        raise ValueError("Bug_ID must be unique to prevent leakage across splits.")
    data.attrs["source_stats"] = source_stats
    return data, labels


def split_dataset(
    data: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.Index, pd.Index]:
    train_indices, test_indices = train_test_split(
        data.index,
        test_size=test_size,
        random_state=seed,
        stratify=data["Priority"],
    )
    return pd.Index(train_indices), pd.Index(test_indices)


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


def save_dataset_profile(data: pd.DataFrame, output_dir: Path) -> None:
    source_stats = data.attrs.get("source_stats", {})
    missing_counts = source_stats.get("missing_counts", data.isna().sum().to_dict())
    dtypes = source_stats.get(
        "dtypes", {column: str(dtype) for column, dtype in data.dtypes.items()}
    )
    pd.DataFrame(
        {
            "Column": data.columns,
            "Data Type": [dtypes[column] for column in data.columns],
            "Missing Count": [missing_counts[column] for column in data.columns],
            "Missing Percent": [
                missing_counts[column] / max(source_stats.get("row_count", len(data)), 1) * 100
                for column in data.columns
            ],
            "Unique Values": data.nunique(dropna=True).to_numpy(),
        }
    ).to_csv(output_dir / "dataset_profile.csv", index=False)
    overview = pd.DataFrame(
        [
            {"Measure": "Rows", "Value": len(data)},
            {"Measure": "Columns", "Value": len(data.columns)},
            {
                "Measure": "Full-row duplicates",
                "Value": source_stats.get(
                    "full_row_duplicates", int(data.duplicated().sum())
                ),
            },
            {"Measure": "Duplicate Bug_IDs", "Value": int(data["Bug_ID"].duplicated().sum())},
            {"Measure": "Missing Description values", "Value": int(data["Description"].eq("").sum())},
            {"Measure": "Target", "Value": "Priority"},
        ]
    )
    overview.to_csv(output_dir / "dataset_overview.csv", index=False)
    counts = data["Priority"].value_counts().sort_index()
    distribution = counts.rename_axis("Priority").reset_index(name="Count")
    distribution["Percent"] = distribution["Count"] / len(data) * 100
    distribution.to_csv(output_dir / "priority_distribution.csv", index=False)


def save_split(
    data: pd.DataFrame,
    train_indices: pd.Index,
    test_indices: pd.Index,
    output_dir: Path,
) -> None:
    parts = []
    for split_name, indices in (("train", train_indices), ("test", test_indices)):
        parts.append(
            pd.DataFrame(
                {
                    "Bug_ID": data.loc[indices, "Bug_ID"].to_numpy(),
                    "Priority": data.loc[indices, "Priority"].to_numpy(),
                    "Split": split_name,
                }
            )
        )
    pd.concat(parts, ignore_index=True).to_csv(
        output_dir / "split_assignments.csv", index=False
    )


def save_run_summary(
    data: pd.DataFrame,
    labels: list[str],
    input_path: Path,
    train_indices: pd.Index,
    test_indices: pd.Index,
    seed: int,
    test_size: float,
    extra: dict[str, Any],
    output_dir: Path,
) -> None:
    summary = {
        "input": str(input_path.resolve()),
        "rows": len(data),
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "target": "Priority",
        "labels": labels,
        "seed": seed,
        "test_size": test_size,
        "features": FEATURE_COLUMNS,
        "excluded_columns": [
            column
            for column in data.columns
            if column not in FEATURE_COLUMNS + ["Priority"]
        ],
    }
    summary.update(extra)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
