# Eclipse JDT priority prediction

This workflow predicts `Priority` (P1–P5) from `Summary`, `Description`,
`Severity`, and `Component`. It excludes `Bug_ID` as an identifier, `Assignee`
as assignment metadata that can leak process information, and `Opened` to avoid
learning time-specific patterns. The data contains no explicit record-to-record relationships. The optional GNN
is therefore experimental: it constructs an assumed graph by connecting
TF-IDF cosine-similar bug reports within the same component. Test nodes only
connect to training nodes; this graph construction is a modeling assumption,
not a relationship supplied by the source data.

The traditional ML script compares a majority baseline, decision tree, random
forest, linear SVM, and Naive Bayes. It uses stratified 5-fold training
cross-validation for model comparison and a held-out test set for final metrics.
The RNN/LSTM script trains bidirectional recurrent text classifiers with
class-weighted loss. Both scripts use the same stratified split (`seed=42`,
`test-size=0.2`) and report accuracy, macro/weighted precision, recall and F1,
per-class scores, confusion matrices, and test predictions.

Run both scripts from the project root:

```powershell
.\.venv\Scripts\python.exe .\eclipse\priority_prediction.py
.\.venv\Scripts\python.exe .\eclipse\deep_learning_prediction.py
.\.venv\Scripts\python.exe .\eclipse\gnn_prediction.py
```

By default, results are written under `eclipse\priority_results` and
`eclipse\deep_learning_results`; the experimental GNN writes to
`eclipse\gnn_results`. If a results directory already exists, each script
creates a timestamped sibling instead of overwriting it. The deep-learning
scripts support `--epochs` to adjust training duration.
