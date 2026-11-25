# NetBeans bug-priority prediction

This workflow predicts `Priority` in `merged_bug_reports.csv`. It uses
TF-IDF features from `Summary` and `Keywords`, plus one-hot encoded
`Product`, `Component`, `OS`, and `Severity` fields. It compares a majority
baseline with decision tree, random forest, linear SVM, and Naive Bayes models.
Class-weighted tree/SVM models and a uniform-prior Naive Bayes model account
for class imbalance. Macro F1 is the primary model-selection metric.

`Bug ID`, `Status`, `Changed`, `Number of Comments`, `Opened`, and `Source File`
are excluded from model features. The status, change date, and comment count
can reflect events after a report is filed; IDs and source-file names are
identifiers rather than generalizable report content. The script assumes
`Severity`, `Product`, `Component`, and `OS` are available when priority is
predicted.

From the project root, run:

```powershell
.\.venv\Scripts\python.exe .\netbeans\priority_prediction.py
```

Optional arguments:

```powershell
.\.venv\Scripts\python.exe .\netbeans\priority_prediction.py `
  --input .\netbeans\merged_bug_reports.csv `
  --folds 5 --test-size 0.2 --seed 42
```

Results are written under `netbeans\priority_model_results`. If that directory
already exists, a timestamped sibling directory is created so earlier results
are not overwritten. Outputs include dataset profile and class counts,
cross-validation and held-out metrics, per-class scores, confusion matrices,
test-set predictions, split assignments, run metadata, and the best fitted
model selected by cross-validation macro F1.

## Deep-learning models

The separate deep-learning workflow trains bidirectional RNN and LSTM text
classifiers and a graph neural network using the same target, stratified split
seed, and held-out test proportion. The recurrent models use `Summary` and
`Keywords`; the GNN also uses `Product`, `Component`, `OS`, and `Severity`.
Class-weighted loss addresses target imbalance. Test nodes only receive graph
messages from training nodes.

Run it from the project root:

```powershell
.\.venv\Scripts\python.exe .\netbeans\deep_learning_prediction.py
```

By default, outputs are saved under `netbeans\deep_learning_results`; if that
folder exists, a timestamped sibling is used. Use `--epochs` to adjust training
duration. Results include aggregate and per-class test metrics, confusion
matrices, model checkpoints, and GNN text preprocessors.
