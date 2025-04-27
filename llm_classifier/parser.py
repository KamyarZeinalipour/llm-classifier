"""
Robust Parser for llm-classifier outputs.

Example
-------
from llm_classifier.parser import Parser

parser = Parser(verbose=True)          # see failures while parsing
df_parsed = parser.parse_csv("results_matched/system1/output_gpt-4o-mini.csv")
"""
from __future__ import annotations

import re
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


class Parser:
    """
    Extract JSON / dict objects from LLM outputs and normalise them into columns.

    Improvements over the original version
    --------------------------------------
    * non-greedy regex (`{.*?}`) so we don't swallow the whole message
    * tries `json.loads` **then** `ast.literal_eval`
    * scans **all** brace-blocks in the cell; stops at the first that parses
    * optional `verbose` flag to print rows that cannot be parsed
    """

    BRACE_RE = re.compile(r"\{.*?\}", re.DOTALL)

    def __init__(self, *, verbose: bool = False) -> None:
        self.verbose = verbose

    # ──────────────────────────────────────────────────────────────────
    # core helpers
    # ──────────────────────────────────────────────────────────────────
    def _parse_block(self, text: str) -> dict | None:
        """Try JSON first, then Python literal-eval."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(text)
        except Exception:
            return None

    def extract_json(self, cell: Any) -> dict | None:
        """Return a dict if we can parse **any** { ... } in the cell."""
        if not isinstance(cell, str):
            return None

        for block in reversed(self.BRACE_RE.findall(cell)):  # last block first
            parsed = self._parse_block(block)
            if parsed is not None and isinstance(parsed, dict):
                return parsed

        # nothing worked
        return None

    # ──────────────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────────────
    def parse_csv(
        self,
        csv_path: str | Path,
        *,
        output_col: str = "output_json",
        keep_original: bool = True,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        csv_path : str | Path
            File that contains at least one column with model output strings.
        output_col : str, default "output_json"
            Column that holds the raw LLM answer.
        keep_original : bool, default True
            If False, drop the raw output column in the returned dataframe.

        Returns
        -------
        DataFrame  – original columns + flattened prediction columns
        """
        df = pd.read_csv(csv_path)
        if output_col not in df.columns:
            raise KeyError(f"Column '{output_col}' not found in {csv_path}")

        records = df[output_col].apply(self.extract_json)

        # Debug print for rows we could not parse
        if self.verbose:
            num_fail = records.isna().sum()
            if num_fail:
                print(
                    f"[Parser] {num_fail} / {len(df)} rows could not be parsed "
                    f"in {Path(csv_path).name}"
                )

        parsed = pd.json_normalize(records.dropna())

        # align indexes so concat works even if some rows are NaN
        parsed.index = records.dropna().index

        result = pd.concat([df, parsed], axis=1)
        if not keep_original:
            result = result.drop(columns=[output_col])
        return result
