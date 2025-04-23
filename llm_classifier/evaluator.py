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
    """Compute exact-match, Jaccard, micro-averaged metrics for multi-label."""
    @staticmethod
    def _ensure_list(x) -> list[str]:
        if isinstance(x, str): return [p.strip() for p in re.split(r"[,;]", x) if p.strip()]
        if pd.isna(x): return []
        return list(x)

    def evaluate(self, df: pd.DataFrame, ground_truth_path: str) -> pd.DataFrame:
        gt = pd.read_csv(ground_truth_path)
        results = {}
        for col in gt.columns:
            pred = f"{col}_predict"
            if pred not in df: continue
            y_t = gt[col].apply(self._ensure_list)
            y_p = df[pred].apply(self._ensure_list)
            labels = sorted(set(sum(y_t.tolist()+y_p.tolist(), [])))
            mlb = MultiLabelBinarizer(classes=labels)
            Y_t, Y_p = mlb.fit_transform(y_t), mlb.transform(y_p)
            results[col] = {
                'exact_match_accuracy': accuracy_score(Y_t, Y_p),
                'jaccard_score': jaccard_score(Y_t, Y_p, average='samples', zero_division=0),
                'precision_micro': precision_score(Y_t, Y_p, average='micro', zero_division=0),
                'recall_micro': recall_score(Y_t, Y_p, average='micro', zero_division=0),
                'f1_micro': f1_score(Y_t, Y_p, average='micro', zero_division=0),
                'report': classification_report(Y_t, Y_p, target_names=mlb.classes_, zero_division=0)
            }
        return pd.DataFrame(results).T
