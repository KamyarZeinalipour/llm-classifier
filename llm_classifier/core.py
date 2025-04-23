import os
import time
import logging
import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

class Annotator:
    """
    Annotate a CSV of text rows via LLMs (OpenAI/DeepSeek), injecting specified columns into prompts.
    """
    DEEPSEEK_MODELS = {'deepseek-chat', 'deepseek-reasoner'}

    def __init__(
        self,
        openai_api_key: str,
        deepseek_api_key: str | None = None,
        models: list[str] = None,
        system_message_path: str | None = None,
        prompt_template_path: str | None = None,
        prompt_columns: list[str] | None = None,
        retry_delay: int = 300,
    ):
        self.openai_api_key = openai_api_key
        self.deepseek_api_key = deepseek_api_key or openai_api_key
        self.models = models or []
        self.prompt_columns = prompt_columns
        self.retry_delay = retry_delay

        # Load system message
        self.system_message = None
        if system_message_path:
            with open(system_message_path, 'r', encoding='utf-8') as f:
                self.system_message = f.read()

        # Load prompt template
        self.prompt_template = None
        if prompt_template_path:
            with open(prompt_template_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()

    def _get_client(self, model: str) -> OpenAI:
        """Return configured OpenAI client (or DeepSeek)."""
        if model in self.DEEPSEEK_MODELS:
            return OpenAI(api_key=self.deepseek_api_key, base_url='https://api.deepseek.com')
        return OpenAI(api_key=self.openai_api_key)

    def _build_prompt(self, row: pd.Series) -> str:
        """
        Fill `{column}` placeholders from the row. If `prompt_columns` is set, only those.
        """
        if not self.prompt_template:
            return str(row.get('text_clean', ''))
        prompt = self.prompt_template
        cols = self.prompt_columns or list(row.index)
        for col in cols:
            placeholder = f'{{{col}}}'
            prompt = prompt.replace(placeholder, str(row.get(col, '')))
        return prompt

    def _get_completion(self, client: OpenAI, prompt: str, model: str) -> str:
        """Call the LLM with optional system msg + user prompt."""
        messages = []
        if self.system_message:
            messages.append({'role': 'system', 'content': self.system_message})
        messages.append({'role': 'user', 'content': prompt})
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content.strip()

    def run(self, input_csv: str, output_dir: str = '.') -> None:
        """
        Process each row of `input_csv` with each model, saving JSON outputs per-model.
        """
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_csv)
        df['output_json'] = df.get('output_json', pd.NA)

        for model in self.models:
            out_path = os.path.join(output_dir, f'output_{model}.csv')
            start_idx = 0
            if os.path.exists(out_path):
                df_exist = pd.read_csv(out_path)
                filled = df_exist['output_json'].notna()
                start_idx = filled.sum()
                df.loc[:start_idx-1, 'output_json'] = df_exist.loc[:start_idx-1, 'output_json']

            client = self._get_client(model)
            for idx, row in df.iterrows():
                if idx < start_idx:
                    continue
                prompt = self._build_prompt(row)
                try:
                    out = self._get_completion(client, prompt, model)
                except Exception as e:
                    logger.error(f"Error on {model} row {idx}: {e}, retrying...")
                    time.sleep(self.retry_delay)
                    out = self._get_completion(client, prompt, model)
                df.at[idx, 'output_json'] = out
                df.to_csv(out_path, index=False)
