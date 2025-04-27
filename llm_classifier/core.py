import os
import time
import logging
from typing import List, Optional

import pandas as pd
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

# --------------------------------------------------------------------------- #
# Annotator
# --------------------------------------------------------------------------- #
class Annotator:
    """
    Annotate a CSV of text rows with one or more LLMs.

    * Rows are read from ``input_csv``.
    * A prompt template can reference any column via ``{column_name}``.
    * One ``output_<model>.csv`` file is written per model so runs
      can be resumed safely.
    """

    DEEPSEEK_MODELS = {"deepseek-chat", "deepseek-reasoner"}

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        openai_api_key: str,
        deepseek_api_key: Optional[str] = None,
        models: Optional[List[str]] = None,
        system_message_path: Optional[str] = None,
        prompt_template_path: Optional[str] = None,
        prompt_columns: Optional[List[str]] = None,
        retry_delay: int = 300,
        debug: bool = False,
    ) -> None:
        self.openai_api_key = openai_api_key
        self.deepseek_api_key = deepseek_api_key or openai_api_key
        self.models = models or []
        self.prompt_columns = prompt_columns
        self.retry_delay = retry_delay
        self.debug = debug

        # turn on verbose logging when requested
        if self.debug:
            logger.setLevel(logging.DEBUG)

        # load system message (if provided)
        self.system_message = None
        if system_message_path:
            with open(system_message_path, "r", encoding="utf-8") as f:
                self.system_message = f.read()

        # load prompt template (if provided)
        self.prompt_template = None
        if prompt_template_path:
            with open(prompt_template_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _get_client(self, model: str) -> OpenAI:
        """Return a properly configured OpenAI client (or DeepSeek shim)."""
        if model in self.DEEPSEEK_MODELS:
            return OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com",
            )
        return OpenAI(api_key=self.openai_api_key)

    def _build_prompt(self, row: pd.Series) -> str:
        """
        Fill ``{column}`` placeholders from the CSV row.

        If *prompt_columns* is supplied, only those columns are considered.
        Otherwise every column in the row is available for substitution.
        """
        if not self.prompt_template:
            # fallback: just send the raw text (or an empty string)
            return str(row.get("text_clean", ""))

        prompt = self.prompt_template
        cols = self.prompt_columns or list(row.index)
        for col in cols:
            placeholder = f"{{{col}}}"
            prompt = prompt.replace(placeholder, str(row.get(col, "")))
        return prompt

    def _get_completion(self, client: OpenAI, prompt: str, model: str) -> str:
        """Call the LLM with an optional system message followed by the prompt."""
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content.strip()

    # --------------------------------------------------------------------- #
    # Main entry point
    # --------------------------------------------------------------------- #
    def run(self, input_csv: str, output_dir: str = ".") -> None:
        """
        Process *input_csv* with every model in ``self.models``.

        * Results are written to ``output_dir/output_<model>.csv``.
        * If a file already exists, rows that already have ``output_json``
          are **skipped**, so the run can be resumed.
        """
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_csv)
        df["output_json"] = df.get("output_json", pd.NA)

        for model in self.models:
            if self.debug:
                logger.info(f"▶ Running model “{model}”")

            out_path = os.path.join(output_dir, f"output_{model}.csv")
            start_idx = 0

            # resume logic: reuse existing outputs if the file exists
            if os.path.exists(out_path):
                df_exist = pd.read_csv(out_path)
                filled = df_exist["output_json"].notna()
                start_idx = filled.sum()
                df.loc[: start_idx - 1, "output_json"] = df_exist.loc[
                    : start_idx - 1, "output_json"
                ]

            client = self._get_client(model)

            for idx, row in df.iterrows():
                if idx < start_idx:
                    continue

                prompt = self._build_prompt(row)

                # show the FIRST prompt we feed to each model
                if self.debug and idx == start_idx:
                    logger.debug(
                        f"── Prompt example for {model} (row {idx}):\n{prompt}\n"
                    )

                try:
                    out = self._get_completion(client, prompt, model)
                except Exception as e:
                    logger.error(
                        f"Error on {model} row {idx}: {e}, retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                    out = self._get_completion(client, prompt, model)

                df.at[idx, "output_json"] = out
                df.to_csv(out_path, index=False)
