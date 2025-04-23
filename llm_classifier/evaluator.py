import re
import pandas as pd
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
        if isinstance(x, str):
            return [p.strip() for p in re.split(r"[,;]", x) if p.strip()]
        if pd.isna(x):
            return []
        return list(x)

    def evaluate(self, df: pd.DataFrame, ground_truth_path: str) -> pd.DataFrame:
        gt = pd.read_csv(ground_truth_path)
        results = {}
        for col in gt.columns:
            pred_col = f"{col}_predict"
            if pred_col not in df:
                continue
            y_true = gt[col].apply(self._ensure_list)
            y_pred = df[pred_col].apply(self._ensure_list)
            labels = sorted(set(sum(y_true.tolist() + y_pred.tolist(), [])))
            mlb = MultiLabelBinarizer(classes=labels)
            Y_true, Y_pred = mlb.fit_transform(y_true), mlb.transform(y_pred)

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
