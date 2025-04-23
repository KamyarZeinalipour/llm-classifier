import re
import json
import pandas as pd

class Parser:
    """Extract JSON objects from LLM outputs and normalize to a DataFrame."""
    @staticmethod
    def extract_json(cell: str) -> dict | None:
        if not isinstance(cell, str):
            return None
        m = re.search(r"\{.*\}", cell, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None

    def parse_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        records = df['output_json'].apply(self.extract_json)
        parsed = pd.json_normalize(records.dropna())
        return pd.concat([df, parsed], axis=1)
