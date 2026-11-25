# Software Bug Priority Prediction

Machine-learning and deep-learning experiments for classifying software bug reports into priority levels **P1–P5**. The project includes classical text classifiers, recurrent neural networks, and an experimental graph neural network.

## Models

- **Classical:** majority-class baseline, decision tree, random forest, linear SVM, and Naive Bayes.
- **Deep learning:** bidirectional RNN and LSTM text classifiers, plus a GNN.
- **Features:** The main workflow uses bug summaries. Eclipse and NetBeans workflows also use project-specific text and metadata features.

The GNN builds edges between TF–IDF-similar reports within the same component. This is an experimental graph assumption; the datasets do not provide explicit report-to-report relationships.

## Evaluation

The default workflow uses a stratified 80/20 train/test split, seed 42, and 5-fold cross-validation. Metrics include accuracy, precision, recall, F1, per-class results, and confusion matrices.

### Example held-out results

The following scores are from the saved Eclipse run. The neural models used shortened training (1 epoch for LSTM/RNN and 3 for GNN), so treat these as experimental results.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Majority baseline | 0.825 | 0.746 |
| Decision Tree | 0.746 | 0.756 |
| Random Forest | 0.831 | 0.762 |
| SVM | 0.811 | 0.794 |
| Naive Bayes | 0.826 | 0.748 |
| LSTM | 0.501 | 0.586 |
| RNN | 0.330 | 0.428 |
| GNN | 0.141 | 0.134 |

Full cross-validation and held-out metrics are in [`eclipse/priority_cv_results_fast_20261124_2115/combined_cv_and_dl_metrics.csv`](eclipse/priority_cv_results_fast_20261124_2115/combined_cv_and_dl_metrics.csv).

## Hyperparameters

Defaults include a linear SVM (`C=1.0`), a 300-tree Random Forest, and bidirectional RNN/LSTM models with 128-dimensional embeddings and hidden states, dropout 0.3, 8 epochs, batch size 128, and AdamW (`learning_rate=0.001`). The GNN uses hidden size 128, dropout 0.4, 30 epochs, AdamW (`learning_rate=0.003`), and 5 neighbors.

## Project structure

- `baseline_experiment.py`, `model_utils.py` — classical model comparison and evaluation.
- `deep_learning_cv.py`, `dl_utils.py`, `gnn_model.py` — deep-learning training and evaluation.
- `eclipse/` — Eclipse-specific data and priority-prediction workflows.
- `netbeans/` — NetBeans-specific data and priority-prediction workflows.
- `baseline_results/`, `deep_learning_results/`, and other results folders — saved metrics and model outputs.

## Setup

Python 3. Install the dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install pandas numpy scikit-learn torch
