import io
import re
import time
import json
import argparse
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

import logging

logger = logging.getLogger(__name__)



SYSTEM_PROMPT_TEMPLATES = {
    'answer': '''You are a financial-data extraction expert. From the text provided by the user, extract the FINAL answer (the single numeric result after all calculations).

Requirements:
- Output ONLY the numeric value. Do not include units, percent signs, thousands separators, currency symbols, explanations, or any other text.
- For example, if the final answer is 9.61%, output 9.61.
- If no final answer is given in the text, output NA.
''',
    'formula': '''You are a financial-data extraction expert. From the text provided by the user, extract the indicator formula used in the calculation.

Requirements:
- Output ONLY the formula string itself, on a single line, preserving the original symbols and expression.
- Do not output explanations, derivations, units, or any other text. Do not wrap the output in a markdown code block.
- If no formula is given in the text, output NA.
''',
    'data': '''You are a financial-data extraction expert. From the text provided by the user, extract the RAW input data the calculation relies on, and format it as CSV.

Requirements:
- The first line of your output MUST be exactly the CSV header given below, character for character. Do not add, remove, or reorder columns.
- Preserve the original numeric precision; do not round.
- If a column in the CSV header given below explicitly specifies the target unit (e.g. "国内生产总值 (亿元)"), convert the value to that unit. Otherwise, if the source uses abbreviated units (e.g. 万 / 亿 for CN markets, or thousand / million / billion for US markets), convert to the base monetary unit keeping the original currency (e.g. 元 for CN markets, or $ for US markets). (e.g. 14万 -> 140000; 152,286 M -> 152286000000)
- Output ONLY the CSV text including the header. Do NOT wrap the output in a markdown code block. Do NOT add any explanation.
- If a cell cannot be found in the text, leave it empty.

CSV header:
{header}
''',
    'data_indicator': '''You are a financial-data extraction expert. From the text provided by the user, extract the per-period (e.g. per-year) INDICATOR values that were computed, and format them as CSV.

Requirements:
- The first line of your output MUST be exactly the CSV header given below, character for character. Do not add, remove, or reorder columns.
- Preserve the original numeric precision; do not round.
- If a column in the CSV header given below explicitly specifies the target unit (e.g. "国内生产总值 (亿元)"), convert the value to that unit. Otherwise, if the source uses abbreviated units (e.g. 万 / 亿 for CN markets, or thousand / million / billion for US markets), convert to the base monetary unit keeping the original currency (e.g. 元 for CN markets, or $ for US markets). (e.g. 14万 -> 140000; 152,286 M -> 152286000000)
- Output ONLY the CSV text including the header. Do NOT wrap the output in a markdown code block. Do NOT add any explanation.
- If a cell cannot be found in the text, leave it empty.

CSV header:
{header}
''',
}


USER_PROMPT_TEMPLATE = '''Here is the text:
<text>
{response}
</text>

Extract per the instructions and output only the required content.'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--extract_type",
        type=str,
        required=True,
        choices=['answer', 'data', 'data_indicator', 'formula'],
        help="Which field to extract from the response column.",
    )
    parser.add_argument("--max_concurrency", type=int, default=20)
    parser.add_argument("--chat_args", type=str, help="JSON string for chat arguments")
    args = parser.parse_args()

    args.chat_args = json.loads(args.chat_args) if args.chat_args else {}

    return args


def retry_on_exception(max_retries=3, exceptions=(Exception,), delay=0):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Retrying {retries + 1}/{max_retries}... {type(e).__name__}: {e}.")
                    retries += 1
                    if delay:
                        await asyncio.sleep(delay)
            raise Exception(f"Failed after {max_retries} retries.")
        return wrapper
    return decorator


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def get_csv_header(csv_value):
    """Return the header (first line) of a CSV string value, or None if missing."""
    if _is_missing(csv_value):
        return None
    return str(csv_value).strip().split('\n', 1)[0].strip()


def strip_code_fence(text):
    """Strip surrounding triple-backtick code fences if present."""
    if not isinstance(text, str):
        return text
    s = text.strip()
    if s.startswith('```'):
        first_break = s.find('\n')
        s = s[first_break + 1:] if first_break != -1 else ''
        if s.endswith('```'):
            s = s[:-3]
    return s.strip()


def _strip_thousands_sep(text):
    return re.sub(
        r'(?<![0-9-])\b\d{1,3}(,\d{3})+\b',
        lambda m: m.group(0).replace(',', ''),
        text,
    )


class ExtractionFormatError(ValueError):
    """The LLM returned content that does not match the requested extraction format."""


def _validate_answer(text):
    if text.upper() == 'NA':
        return text
    try:
        return float(_strip_thousands_sep(text))
    except ValueError as e:
        raise ExtractionFormatError(f'extracted answer is not a valid number: {text!r}') from e


def _validate_formula(text):
    # Empty case is already handled in validate_extracted; nothing else to check here.
    return text


_CSV_INDEX_COLS = ('year', 'corporate', 'date')


def _merge_split_rows(text):
    """Coalesce rows that share the same index key; first non-empty value wins per column."""
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except Exception:
        return text
    index_cols = [c for c in _CSV_INDEX_COLS if c in df.columns]
    if not index_cols or not df.duplicated(subset=index_cols).any():
        return text
    def _first_nonempty(s):
        v = s[s.str.strip() != '']
        return v.iloc[0] if len(v) else ''
    merged = df.groupby(index_cols, sort=False).agg(_first_nonempty).reset_index()[df.columns]
    return merged.to_csv(index=False, lineterminator='\n').strip()


def _validate_csv(text, expected_header):
    first_line = text.split('\n', 1)[0].strip()
    expected = (expected_header or '').strip()
    if first_line != expected:
        raise ExtractionFormatError(f'CSV header mismatch. expected={expected!r}, got={first_line!r}')
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except Exception as e:
        raise ExtractionFormatError(f'cannot parse extracted CSV: {e}') from e
    df = df.apply(lambda col: col.map(_strip_thousands_sep))
    return df.to_csv(index=False, lineterminator='\n').strip()


def validate_extracted(text, extract_type, header):
    s = (text or '').strip()
    if not s:
        raise ExtractionFormatError(f'extracted {extract_type} is empty')
    if extract_type == 'answer':
        s = _validate_answer(s)
    elif extract_type == 'formula':
        s = _validate_formula(s)
    elif extract_type in ('data', 'data_indicator'):
        s = _validate_csv(s, header)
    else:
        raise ValueError(f'unknown extract_type: {extract_type}')
    return s


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_concurrency: int = 5,
        chat_args: dict = {}
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.sem = asyncio.Semaphore(max_concurrency)
        self.chat_args = chat_args

    async def query_chat_completions(self, user_prompt, system_prompt=None):
        messages = []
        if system_prompt is not None:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.chat_args
        )
        # print(response, '\n\n\n')  ### debug

        response_text = response.choices[0].message.content
        if response_text:
            response_text = response_text.strip()
        else:
            response_text = ''
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning_content = response.choices[0].message.reasoning_content.strip()
        else:
            reasoning_content = ''

        return response_text, reasoning_content

    async def query_response(self, user_prompt, system_prompt=None):
        inputs = []
        if system_prompt is not None:
            inputs.append({'role': 'system', 'content': system_prompt})
        inputs.append({'role': 'user', 'content': user_prompt})
        response = await self.client.responses.create(
            model=self.model_name,
            input=inputs,
            **self.chat_args
        )
        # print(response, '\n\n\n')  ### debug
        response_text = response.output_text
        reasoning_content = ''
        return response_text, reasoning_content

    # @retry_on_exception()
    async def query(self, user_prompt, system_prompt=None, return_reasoning_content=False):
        # print(f'Querying {self.model_name}')
        if self.model_name.startswith('gpt-5.'):
            response_text, reasoning_content = await self.query_response(user_prompt, system_prompt)
        else:
            response_text, reasoning_content = await self.query_chat_completions(user_prompt, system_prompt)

        # print(f'Got response from {self.model_name}')
        ### debug
        # print(f'=== System Prompt ===\n{system_prompt}\n')
        # print(f'=== User Prompt ===\n{user_prompt}\n')
        # print(f'=== Response Text ===\n{response_text}\n')

        if return_reasoning_content:
            return response_text, reasoning_content
        return response_text

    @retry_on_exception()
    async def query_and_validate(self, user_prompt, system_prompt, extract_type, header):
        text = await self.query(user_prompt, system_prompt=system_prompt)
        text = strip_code_fence(text)
        text = validate_extracted(text, extract_type, header)
        if extract_type in ('data', 'data_indicator'):
            text = _merge_split_rows(text)
        return text

    async def async_extract(self, responses, headers, extract_type):
        system_template = SYSTEM_PROMPT_TEMPLATES[extract_type]
        needs_header = extract_type in ('data', 'data_indicator')

        async def job(response, header):
            if _is_missing(response):
                return ''
            if needs_header and not header:
                return ''
            system_prompt = system_template.format(header=header) if needs_header else system_template
            user_prompt = USER_PROMPT_TEMPLATE.format(response=str(response))
            async with self.sem:
                try:
                    return await self.query_and_validate(user_prompt, system_prompt, extract_type, header)
                except Exception as e:
                    logger.warning(f'Extraction failed for a row after retries (extract_type={extract_type}): {e}')
                    return ''

        tasks = [job(r, h) for r, h in zip(responses, headers)]
        return await tqdm_asyncio.gather(*tasks)

    async def eval(self, output_path, input_path, extract_type):
        print(f'Extracting {extract_type} from responses in {input_path}...')
        df = pd.read_csv(input_path)
        if 'response' not in df.columns:
            raise ValueError(f"Input file {input_path} has no 'response' column.")

        headers = [None] * len(df)
        if extract_type in ('data', 'data_indicator'):
            if extract_type not in df.columns:
                logger.warning(
                    f"Input file {input_path} has no '{extract_type}' column; "
                    f"skipping extraction (extract_type={extract_type} requires an expected CSV header from that column)."
                )
                return
            headers = [get_csv_header(v) for v in df[extract_type].tolist()]

        responses = df['response'].tolist()

        extracted = await self.async_extract(responses, headers, extract_type)

        out_col = f'response_{extract_type}'
        df[out_col] = extracted
        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path} (column '{out_col}').")


def main(output_path, input_path, api_key, base_url, model_name, extract_type, max_concurrency, chat_args):
    llm = LLMClient(api_key, base_url, model_name, max_concurrency, chat_args)
    asyncio.run(llm.eval(output_path, input_path, extract_type))
    print('--- All done ---')



if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
