"""
Multi-label evaluator for llm-classifier with handy debug output.

Usage
-----
from llm_classifier.evaluator import Evaluator

ev = Evaluator()
metrics_df, debug_df = ev.evaluate(
        df_pred,
        ground_truth_path="data/df_cleanv3.csv",
        return_debug=True,
)

# ➊ metrics_df – one row per label column with exact-match, Jaccard, micro P/R/F1
# ➋ debug_df   – one row per data sample & label column so you can inspect errors:
print(debug_df.query("relation_correct == False").head())
"""
from __future__ import annotations

import re, ast
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    classification_report,
)
from sklearn.preprocessing import MultiLabelBinarizer


class Evaluator:
    """Compute multilabel metrics and (optionally) row-level debug info."""

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _ensure_list(x) -> List[str]:
        """
        Normalise *anything* into a list of strings.

        Handles:
        1.  list / tuple / ndarray                → list(str)
        2.  stringified Python list "['a','b']"   → ['a', 'b']
        3.  plain string "a, b; c"                → ['a', 'b', 'c']
        4.  NaN / None                            → []
        """
        # 1️⃣ already iterable
        if isinstance(x, (list, tuple, np.ndarray)):
            return [str(i).strip() for i in x]

        # 4️⃣ NaN / None
        if pd.isna(x):
            return []

        # 2️⃣ or 3️⃣ – string input
        if isinstance(x, str):
            s = x.strip()
            # looks like "[...]" → try literal-eval first
            if s.startswith("[") and s.endswith("]"):
                try:
                    val = ast.literal_eval(s)
                    if isinstance(val, (list, tuple, np.ndarray)):
                        return [str(i).strip() for i in val]
                except Exception:
                    pass  # fall through to generic splitter

            # generic comma/semicolon splitter
            return [p.strip(" []'\"") for p in re.split(r"[;,]", s) if p.strip()]

        # fallback: single element
        return [str(x).strip()]

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    def evaluate(
        self,
        df_pred: pd.DataFrame,
        ground_truth_path: str | pd.DataFrame,
        *,
        return_debug: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """
        Parameters
        ----------
        df_pred : DataFrame
            Must contain columns named "<label>_predict".
        ground_truth_path : str | DataFrame
            CSV path or already-loaded ground-truth dataframe.
        return_debug : bool, default False
            If True, also return a per-row debug dataframe.

        Returns
        -------
        metrics_df  or  (metrics_df, debug_df)
        """
        # Load ground truth
        gt = (
            ground_truth_path
            if isinstance(ground_truth_path, pd.DataFrame)
            else pd.read_csv(ground_truth_path)
        ).copy()

        metrics: Dict[str, Dict[str, float | str]] = {}
        debug_rows: List[dict] = []

        for col in gt.columns:
            pred_col = f"{col}_predict"
            if pred_col not in df_pred:
                continue  # silently skip missing prediction columns

            # ---- clean to list-of-labels ---------------------------------
            y_true_lists = gt[col].apply(self._ensure_list)
            y_pred_lists = df_pred[pred_col].apply(self._ensure_list)

            # ---- binarise -------------------------------------------------
            label_space = sorted(
                set(sum(y_true_lists.tolist() + y_pred_lists.tolist(), []))
            )
            mlb = MultiLabelBinarizer(classes=label_space)
            Y_true = mlb.fit_transform(y_true_lists)
            Y_pred = mlb.transform(y_pred_lists)

            # ---- metrics --------------------------------------------------
            metrics[col] = {
                "exact_match_accuracy": accuracy_score(Y_true, Y_pred),
                "jaccard_score": jaccard_score(
                    Y_true, Y_pred, average="samples", zero_division=0
                ),
                "precision_micro": precision_score(
                    Y_true, Y_pred, average="micro", zero_division=0
                ),
                "recall_micro": recall_score(
                    Y_true, Y_pred, average="micro", zero_division=0
                ),
                "f1_micro": f1_score(
                    Y_true, Y_pred, average="micro", zero_division=0
                ),
                "classification_report": classification_report(
                    Y_true, Y_pred, target_names=mlb.classes_, zero_division=0
                ),
            }

            # ---- per-row debug info --------------------------------------
            if return_debug:
                for idx, (yt, yp) in enumerate(zip(y_true_lists, y_pred_lists)):
                    debug_rows.append(
                        {
                            "row_id": idx,
                            "label_col": col,
                            f"{col}_true": ", ".join(yt),
                            f"{col}_pred": ", ".join(yp),
                            f"{col}_correct": set(yt) == set(yp),
                            "jaccard": (
                                len(set(yt).intersection(yp))
                                / len(set(yt).union(yp))
                                if yt or yp
                                else 1.0
                            ),
                        }
                    )

        metrics_df = pd.DataFrame(metrics).T

        if return_debug:
            debug_df = pd.DataFrame(debug_rows)
            return metrics_df, debug_df

        return metrics_df
