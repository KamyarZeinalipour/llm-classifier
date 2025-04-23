import re
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    classification_report
)
from sklearn.preprocessing import MultiLabelBinarizer

class Evaluator:
    """Compute multi-label metrics: exact-match, Jaccard, micro precision/recall/F1."""
    @staticmethod
    def _ensure_list(x) -> list[str]:
        # If x is already a list or numpy array, use it directly
        if isinstance(x, (list, np.ndarray)):
            return list(x)
        # If it's a string, split on commas/semicolons
        if isinstance(x, str):
            return [p.strip() for p in re.split(r"[,;]", x) if p.strip()]
        # If it's NaN or None, return empty list
        if pd.isna(x):
            return []
        # Otherwise, treat it as a single-item list of its string representation
        return [str(x)]

    def evaluate(self, df: pd.DataFrame, ground_truth_path: str) -> pd.DataFrame:
        gt = pd.read_csv(ground_truth_path)
        results = {}
        for col in gt.columns:
            pred_col = f"{col}_predict"
            if pred_col not in df:
                continue
            # Build true and predicted label lists
            y_true = gt[col].apply(self._ensure_list)
            y_pred = df[pred_col].apply(self._ensure_list)
            # Determine all possible labels
            labels = sorted(set(sum(y_true.tolist() + y_pred.tolist(), [])))
            mlb = MultiLabelBinarizer(classes=labels)
            Y_true = mlb.fit_transform(y_true)
            Y_pred = mlb.transform(y_pred)

            results[col] = {
                'exact_match_accuracy': accuracy_score(Y_true, Y_pred),
                'jaccard_score': jaccard_score(Y_true, Y_pred, average='samples', zero_division=0),
                'precision_micro': precision_score(Y_true, Y_pred, average='micro', zero_division=0),
                'recall_micro': recall_score(Y_true, Y_pred, average='micro', zero_division=0),
                'f1_micro': f1_score(Y_true, Y_pred, average='micro', zero_division=0),
                'classification_report': classification_report(
                    Y_true, Y_pred, target_names=mlb.classes_, zero_division=0
                )
            }
        return pd.DataFrame(results).T
