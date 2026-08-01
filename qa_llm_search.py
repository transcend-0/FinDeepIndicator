import time
import os
import json
import argparse
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

from qa_agent import web_search

import logging

logger = logging.getLogger(__name__)


SEARCH_QUERY_SYSTEM_PROMPT = """
You are a search query generator. Given the user's question, output a single concise web search query that will help retrieve information needed to answer it.
Output ONLY the search query string itself: no quotes, no prefixes like "Query:", no explanations, no additional text.
""".strip()


ANSWER_SYSTEM_PROMPT = """
You are a professional financial assistant.
Answer the user's question using the provided web search results.
You may cite the result index (e.g. [1], [2]) when referring to specific entries.
If the search results are insufficient, do your best with the information available and clearly note any uncertainty.
""".strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_concurrency", type=int, default=20)
    parser.add_argument("--serper_api_key", type=str, default=None, help="Serper API key. Falls back to the SERPER_API_KEY env var.")
    parser.add_argument("--num_search_results", type=int, default=5)
    parser.add_argument("--return_conversation", action="store_true", help="Persist the full conversation trace in a `conversation` column.")
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

class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_concurrency: int = 5,
        chat_args: dict = {},
        num_search_results: int = 10,
        return_conversation: bool = False,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.sem = asyncio.Semaphore(max_concurrency)
        self.chat_args = chat_args
        self.num_search_results = num_search_results
        self.return_conversation = return_conversation

    @retry_on_exception()
    async def query(self, messages) -> str:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.chat_args,
        )
        return (response.choices[0].message.content or "").strip()

    @retry_on_exception()
    async def search(self, query: str) -> str:
        return await asyncio.to_thread(web_search, query, self.num_search_results)

    async def search_and_answer(self, question) -> tuple[str, list]:
        # Step 1: ask the LLM to generate a search query for the question
        query_messages = [
            {"role": "system", "content": SEARCH_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        search_query = await self.query(query_messages)

        # Run the search with the generated query
        search_results = await self.search(search_query)

        # Step 2: ask the LLM to answer the question using the search results
        answer_user_prompt = f"Question: {question}\n\nSearch results:\n{search_results}"
        answer_messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": answer_user_prompt},
        ]
        final_answer = await self.query(answer_messages)

        if self.return_conversation:
            conversation = [
                *query_messages,
                {"role": "assistant", "content": search_query},
                *answer_messages,
                {"role": "assistant", "content": final_answer},
            ]
        else:
            conversation = []
        return final_answer, conversation

    async def async_qa(self, questions):
        async def job(q):
            async with self.sem:
                try:
                    return await self.search_and_answer(q)
                except Exception as e:
                    logger.error(f"Query failed: {type(e).__name__}: {e}")
                    return '', []

        tasks = [job(q) for q in questions]
        results = await tqdm_asyncio.gather(*tasks)

        return results

    async def eval(self, output_path, input_path):
        df = pd.read_csv(input_path)
        # df = df.iloc[:5, :]  ### debug
        questions = df['question'].tolist()

        results = await self.async_qa(questions)
        df['response'] = [r[0] for r in results]
        if self.return_conversation:
            df['conversation'] = [json.dumps(r[1], ensure_ascii=False) for r in results]
        df.to_csv(output_path, index=False)
        print(f'Saved results to {output_path}')


    
def main(output_path, input_path, api_key, base_url, model_name, max_concurrency, chat_args, serper_api_key, num_search_results, return_conversation):
    if serper_api_key:
        os.environ["SERPER_API_KEY"] = serper_api_key
    llm = LLMClient(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        max_concurrency=max_concurrency,
        chat_args=chat_args,
        num_search_results=num_search_results,
        return_conversation=return_conversation,
    )
    asyncio.run(llm.eval(output_path, input_path))
    print('--- All done ---')



if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
