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



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--base_url', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--max_concurrency', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--chat_args', type=str, help='JSON string for chat arguments')
    args = parser.parse_args()

    args.chat_args = json.loads(args.chat_args) if args.chat_args else {}

    return args


JUDGE_SYSTEM_PROMPT = '''
# Role
You are an expert financial-formula evaluator. You will be given a batch of items, each with an id, a ground-truth formula, and a candidate formula. Your job is to judge how well each candidate formula matches its ground truth in meaning, not in surface text. Two formulas can be considered equivalent even if they use different variable names, different ordering of commutative operations, algebraically equivalent rewrites (e.g. a/b vs a*b^-1), or different but synonymous notation, as long as they compute the same financial quantity from the same inputs.

You MUST output a single integer score in {0, 1, 2} for each item using the rubric below.

## Scoring Rubric
- 2 (Fully correct): The candidate formula is mathematically and semantically equivalent to the ground truth. It uses the same inputs (allowing synonyms / equivalent notation) and would produce the same numerical result on any valid input.
- 1 (Partially correct): The candidate formula captures the right general idea or the right inputs, but has a non-trivial error — e.g. a missing term, a wrong constant, a wrong aggregation (sum vs average), a swapped numerator/denominator, an off-by-one in the time window, or only part of the required computation.
- 0 (Incorrect): The candidate formula is wrong, unrelated, empty, or uses fundamentally different inputs / operations such that it would not produce the intended quantity.

# Input Format
| id | ground_truth | candidate_formula |
|---|---|---|
| <id> | <ground_truth> | <candidate_formula> |

# Output Format
| id | score |
|---|---|
| <id> | <score 0-2> |

# NOTE
- Provide only the output table. Do not include explanations or additional columns.
- Output one row per input id, preserving the input ids exactly.
'''.strip()


def retry_on_exception(max_retries=3, exceptions=(Exception,), delay=0):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f'Retrying {retries + 1}/{max_retries}... {type(e).__name__}: {e}.')
                    retries += 1
                    if delay:
                        await asyncio.sleep(delay)
            raise Exception(f'Failed after {max_retries} retries.')
        return wrapper
    return decorator

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
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'system', 'content': system_prompt},{'role': 'user', 'content': user_prompt}],
            **self.chat_args
        )

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
        response = await self.client.responses.create(
            model=self.model_name,
            input=[{'role': 'system', 'content': system_prompt},{'role': 'user', 'content': user_prompt}],
            **self.chat_args
        )
        response_text = response.output_text
        reasoning_content = ''
        return response_text, reasoning_content

    # @retry_on_exception()
    async def query(self, user_prompt, system_prompt=None, return_reasoning_content=False):
        if self.model_name.startswith('gpt-5.'):
            response_text, reasoning_content = await self.query_response(user_prompt, system_prompt)
        else:
            response_text, reasoning_content = await self.query_chat_completions(user_prompt, system_prompt)

        if return_reasoning_content:
            return response_text, reasoning_content
        return response_text

    @retry_on_exception()
    async def query_and_parse(self, user_prompt, system_prompt=None):
        response_text = await self.query(user_prompt, system_prompt)
        lines = response_text.split('\n')[2:]
        ids, scores = [], []
        for line in lines:
            try:
                _, id, score, __ = line.split('|')
                ids.append(int(id.strip()))
                scores.append(int(score.strip()))
            except Exception:
                raise ValueError(f'Failed to parse line: {line}')
        return ids, scores

    async def eval(self, output_path, input_path, batch_size=5):
        df = pd.read_csv(input_path).reset_index()

        missing = [c for c in ('formula', 'response_formula') if c not in df.columns]
        if missing:
            raise ValueError(f'Input CSV is missing required column(s): {missing}')

        def batched(df, batch_size):
            for i in range(0, len(df), batch_size):
                yield df.iloc[i:i + batch_size]

        async def job(batch):
            async with self.sem:
                q = '| id | ground_truth | candidate_formula |\n|---|---|---|\n'
                q += '\n'.join([f'| {row["index"]} | {row["formula"]} | {row["response_formula"]} |' for _, row in batch.iterrows()])

                try:
                    ids, scores = await self.query_and_parse(q, JUDGE_SYSTEM_PROMPT)
                except Exception as e:
                    logger.error(f'Query failed for batch starting with id {batch["index"].iloc[0]}: {type(e).__name__}: {e}')
                    ids, scores = [], []

                return ids, scores

        tasks = [job(b) for b in batched(df, batch_size)]
        results = await tqdm_asyncio.gather(*tasks)
        score_map = {id: s for ids, scores in results for id, s in zip(ids, scores)}

        df['accuracy_formula'] = df['index'].map(score_map)
        df['accuracy_formula'] /= 2  # convert to 0-1 range
        del df['index']
        df.to_csv(output_path, index=False)
        print(f'Saved results to {output_path}')



def main(output_path, input_path, api_key, base_url, model_name, max_concurrency, batch_size, chat_args):
    llm = LLMClient(api_key, base_url, model_name, max_concurrency, chat_args)
    asyncio.run(llm.eval(output_path, input_path, batch_size))
    print('--- All done ---')



if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
