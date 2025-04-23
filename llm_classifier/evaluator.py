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
        # If x is already a list or numpy array, convert to list
        if isinstance(x, (list, np.ndarray)):
            return list(x)
        # If it's a string, split on commas/semicolons
        if isinstance(x, str):
            return [p.strip() for p in re.split(r"[,;]", x) if p.strip()]
        # If it's NaN or None
        if pd.isna(x):
            return []
        # Fallback: single-element list
        return [str(x)]

    def evaluate(self, df: pd.DataFrame, ground_truth_path: str) -> pd.DataFrame:
        """
        Reads ground truth CSV and predicted DataFrame, computes metrics per column.
        """
        gt = pd.read_csv(ground_truth_path)
        results = {}

        for col in gt.columns:
            pred_col = f"{col}_predict"
            if pred_col not in df:
                continue

            # Build true/predicted label lists
            y_true = gt[col].apply(self._ensure_list)
            y_pred = df[pred_col].apply(self._ensure_list)

            # Fit multilabel binarizer
            all_labels = sorted(set(sum(y_true.tolist() + y_pred.tolist(), [])))
            mlb = MultiLabelBinarizer(classes=all_labels)
            Y_true = mlb.fit_transform(y_true)
            Y_pred = mlb.transform(y_pred)

            # Compute metrics
            exact_acc = accuracy_score(Y_true, Y_pred)
            try:
                jacc = jaccard_score(Y_true, Y_pred, average='samples', zero_division=0)
            except ValueError:
                jacc = None
            prec = precision_score(Y_true, Y_pred, average='micro', zero_division=0)
            rec = recall_score(Y_true, Y_pred, average='micro', zero_division=0)
            f1 = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
            report = classification_report(Y_true, Y_pred, target_names=mlb.classes_, zero_division=0)

            results[col] = {
                'exact_match_accuracy': exact_acc,
                'jaccard_score': jacc,
                'precision_micro': prec,
                'recall_micro': rec,
                'f1_micro': f1,
                'classification_report': report,
            }

        return pd.DataFrame(results).T
