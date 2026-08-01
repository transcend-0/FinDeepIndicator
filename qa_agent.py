import os
import sys
import json
import inspect
import argparse
import subprocess
import asyncio
import requests
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

import logging

logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_concurrency", type=int, default=20)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--return_conversation", action="store_true", help="Persist the full conversation trace in a `conversation` column.")
    parser.add_argument("--search_backend", type=str, default="serper", choices=["serper", "hosted"], help="Web-search backend: 'serper' (local function tool) or 'hosted' (provider built-in).")
    parser.add_argument("--serper_api_key", type=str, default=None, help="Serper API key. Falls back to the SERPER_API_KEY env var.")
    parser.add_argument("--chat_args", type=str, help="JSON string for chat arguments")
    parser.add_argument("--retry_errors", action="store_true", help="Re-run rows whose response starts with 'Error' in output_path (in-place patch). input_path is ignored.")
    args = parser.parse_args()

    args.chat_args = json.loads(args.chat_args) if args.chat_args else {}

    return args


REACT_SYSTEM_PROMPT = '''
# Role
You are a professional financial assistant.

# Instruction
Think - Act Loop: At each turn, you MUST first write a brief reasoning paragraph summarising what you have learned and what to do next. After that, you have two options:
1) If you need more information, you can call relevant tools. You may call multiple tools and call one tool multiple times in a turn.
2) If you think it's time to end the loop, reason about the question and the information gathered, and then output your final answer.

# Guideline
- Never output with only tool_calls — always reason in text first.
- Do not repeat near-duplicate search queries. If two searches return the same results without the precise figure you need, switch strategy rather than searching again with slightly different keywords.
- `python_exec` is available for numeric calculations or for fetching/parsing data.
- Do NOT invent or guess specific numbers inside search queries to "confirm" them — that wastes turns. Read the source instead.
- Stopping rule: if after ~5 tool calls you still cannot locate an exact figure, use the most credible value you have known, clearly state your source and any assumption, and produce the final answer. Do not loop indefinitely.
'''.strip()

FORCING_FINAL_ANSWER_PROMPT = '''
You have reached the maximum number of tool-use turns. Tools are no longer available.
Based on the information you have already gathered, produce the final answer NOW. Do not call any tools.
If a figure is missing, use the most credible value you found and state the assumption explicitly.
'''.strip()


def web_search(query: str, num: int = 10) -> str:
    """Search the web via Serper (Google) and return top results.

    Args:
        query: Search query string.
        num: Maximum number of organic results to return.

    Returns:
        A formatted multi-line string with each result's title, url and snippet.
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return "web_search error: SERPER_API_KEY is not set."
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=15,
        )
        data = response.json()
    except Exception as e:
        return f"web_search error: {type(e).__name__}: {e}"

    items = []
    if isinstance(data.get("answerBox"), dict):
        ab = data["answerBox"]
        items.append(f"[AnswerBox] {ab.get('title', '')}: {ab.get('answer') or ab.get('snippet', '')}".strip())
    for r in data.get("organic", [])[:num]:
        items.append(
            f"{r.get('title', '')}\n{r.get('link', '')}\n{r.get('snippet', '')}".strip()
        )
    if not items:
        return "No results."
    return "\n\n".join(f"[{i + 1}] {it}" for i, it in enumerate(items))


def python_exec(code: str, timeout: int = 60) -> str:
    """Run Python in an isolated subprocess. STATELESS — each call spawns a FRESH Python interpreter; variables, imports, downloaded files/HTML, and any other state from previous python_exec calls DO NOT persist. To reuse data from an earlier call, re-fetch or re-compute it inside this one — do NOT reference names like `html`, `content`, `df` defined in a prior call. Use `print(...)` to surface values; 8000-character output limit; code that hangs is killed after `timeout` seconds.

    Args:
        code: Python source code to execute.

    Returns:
        Combined stdout and stderr from the subprocess.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import sys; exec(sys.stdin.read())"],
            input=code,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"(execution timed out after {timeout}s)"
    out = (proc.stdout or '') + (proc.stderr or '')
    if len(out) > 8000:
        out = out[:8000] + "\n... (truncated)"
    return out or "(no output)"


_GOOGLE_SECTIONS = {
    'Args:', 'Arguments:', 'Returns:', 'Return:', 'Yields:', 'Raises:',
    'Example:', 'Examples:', 'Note:', 'Notes:', 'Attributes:',
}


def parse_google_docstring(doc: str) -> tuple[str, dict]:
    if not doc:
        return '', {}
    lines = doc.splitlines()

    summary_lines, args_start = [], None
    for i, line in enumerate(lines):
        if line.strip() in _GOOGLE_SECTIONS:
            if line.strip() in ('Args:', 'Arguments:'):
                args_start = i + 1
            break
        summary_lines.append(line)

    summary = '\n'.join(summary_lines).strip().split('\n\n', 1)[0]
    summary = ' '.join(summary.split())

    args_desc: dict[str, str] = {}
    if args_start is None:
        return summary, args_desc

    block = []
    for line in lines[args_start:]:
        if line.strip() in _GOOGLE_SECTIONS or (line and not line.startswith((' ', '\t'))):
            break
        block.append(line)

    base_indent = None
    for bl in block:
        if bl.strip():
            base_indent = len(bl) - len(bl.lstrip())
            break
    if base_indent is None:
        return summary, args_desc

    current_name, current_parts = None, []
    for bl in block:
        if not bl.strip():
            continue
        indent = len(bl) - len(bl.lstrip())
        stripped = bl.strip()
        if indent == base_indent and ':' in stripped:
            if current_name is not None:
                args_desc[current_name] = ' '.join(current_parts).strip()
            name, _, desc = stripped.partition(':')
            current_name = name.split('(')[0].strip()
            current_parts = [desc.strip()]
        elif current_name is not None:
            current_parts.append(stripped)
    if current_name is not None:
        args_desc[current_name] = ' '.join(current_parts).strip()

    return summary, args_desc

def function_to_tool_schema(fn) -> dict:
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    sig = inspect.signature(fn)
    description, arg_descriptions = parse_google_docstring(inspect.getdoc(fn) or '')
    if not description:
        description = fn.__name__

    properties, required = {}, []
    for name, param in sig.parameters.items():
        ann = param.annotation if param.annotation is not inspect.Parameter.empty else str
        prop = {"type": type_map.get(ann, "string")}
        if name in arg_descriptions:
            prop["description"] = arg_descriptions[name]
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

def build_tools(model_name: str, search_backend: str = "serper"):
    """Build the tool list and local-tool registry.

    Args:
        model_name: Full model identifier, e.g. `openai/gpt-5.5`.
        search_backend: `serper` (default) uses a local Serper-backed
            `web_search` function for every model; `hosted` uses the
            provider's built-in search (OpenAI hosted for `openai/*`,
            `openrouter:web_search` otherwise) and OpenAI's
            `code_interpreter` for `openai/*`.

    Returns:
        `(tools, local_tools)` where `tools` is the value passed to the
        request and `local_tools` maps tool names to callables the client
        executes locally.
    """
    if search_backend == "serper":
        local_fns = [web_search, python_exec]
        tools = [function_to_tool_schema(fn) for fn in local_fns]
        return tools, {fn.__name__: fn for fn in local_fns}

    if model_name.startswith('openai/'):
        tools = [
            {"type": "web_search"},
            {"type": "code_interpreter", "container": {"type": "auto", "memory_limit": "1g"}},
        ]
        return tools, {}

    local_fns = [python_exec]
    tools = [{"type": "openrouter:web_search"}] + [function_to_tool_schema(fn) for fn in local_fns]
    return tools, {fn.__name__: fn for fn in local_fns}


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


class AgentClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_concurrency: int = 5,
        max_turns: int = 10,
        chat_args: dict = {},
        return_conversation: bool = False,
        search_backend: str = "serper",
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.max_turns = max_turns
        self.sem = asyncio.Semaphore(max_concurrency)
        self.chat_args = chat_args
        self.return_conversation = return_conversation
        self.search_backend = search_backend
        self.tools, self.local_tools = build_tools(model_name, search_backend)

    async def _call_local_tool(self, name: str, arguments: dict) -> str:
        fn = self.local_tools.get(name)
        if fn is None:
            return f"Unknown tool: {name}"
        try:
            return await asyncio.to_thread(fn, **arguments)
        except Exception as e:
            return f"Tool `{name}` execution error: {type(e).__name__}: {e}"

    async def _chat(self, messages):
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self.tools,
            **self.chat_args,
        )
        return response.choices[0].message

    @staticmethod
    def _extract_reasoning(msg) -> dict:
        """Pull thinking-trace fields off a response message, regardless of route.

        Native deepseek API exposes `reasoning_content`; OpenRouter exposes
        `reasoning` (text). We deliberately skip `reasoning_details` — echoing
        the structured form back in subsequent requests has been observed to
        make deepseek-v4-flash return totally empty responses.
        """
        extra = getattr(msg, "model_extra", None) or {}
        out = {}
        for attr in ("reasoning_content", "reasoning"):
            val = getattr(msg, attr, None)
            if val is None:
                val = extra.get(attr)
            if val:
                out[attr] = val
        return out

    async def _chat_no_tools(self, messages):
        chat_args = {k: v for k, v in self.chat_args.items() if k not in ("tool_choice", "parallel_tool_calls")}
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **chat_args,
        )
        return response.choices[0].message

    @retry_on_exception()
    async def query(self, user_prompt: str) -> tuple[str, list]:
        if self.model_name.startswith('openai/') and self.search_backend == "hosted":
            return await self._query_via_responses(user_prompt)
        return await self._query_via_chat(user_prompt)

    async def _query_via_responses(self, user_prompt: str) -> tuple[str, list]:
        """Use the Responses API so hosted tool calls surface as output items."""
        response = await self.client.responses.create(
            model=self.model_name,
            instructions=REACT_SYSTEM_PROMPT,
            input=user_prompt,
            tools=self.tools,
            **self.chat_args,
        )
        final_answer = (response.output_text or '').strip()
        if not self.return_conversation:
            return final_answer, []

        output_items = []
        for item in response.output:
            try:
                output_items.append(item.model_dump())
            except Exception:
                output_items.append({"type": getattr(item, "type", "unknown"), "repr": str(item)})

        conversation = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            *output_items,
        ]
        return final_answer, conversation

    async def _query_via_chat(self, user_prompt: str) -> tuple[str, list]:
        """Manual Think-Act loop over chat.completions with client-side function tools."""
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        final_answer = ''
        abort_reason = None
        for turn in range(self.max_turns):
            msg = await self._chat(messages)
            content = msg.content or ''
            reasoning = self._extract_reasoning(msg)

            # deepseek-v4-flash via OpenRouter occasionally emits a turn with
            # reasoning but neither content nor tool_calls — the model
            # "thought" about its next action but didn't commit to one.
            # Replaying the same context usually works (verified empirically),
            # so retry once. Note we don't require `reasoning` to be empty —
            # the failure mode that hurts us most is "reasoning present, no
            # action", which the previous three-empty check missed.
            if not content.strip() and not msg.tool_calls:
                logger.warning(
                    f'Unproductive response on turn {turn} (reasoning_len={len(reasoning.get("reasoning") or reasoning.get("reasoning_content") or "")}), retrying once.'
                )
                msg = await self._chat(messages)
                content = msg.content or ''
                reasoning = self._extract_reasoning(msg)

            # If still unproductive after retry, don't pollute history with
            # the empty turn — drop it and break to the forced-final-answer
            # path with the last productive context. Keeping it in history
            # tends to bias the model toward "I'll think some more" again.
            if not content.strip() and not msg.tool_calls:
                abort_reason = f'empty response with no tool calls on turn {turn} (after retry)'
                break

            assistant_msg = {"role": "assistant", "content": content}
            # Echo the model's chain-of-thought back into history so the next
            # turn keeps continuity (and per deepseek docs is REQUIRED when
            # the prior assistant message carries tool_calls).
            assistant_msg.update(reasoning)
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                final_answer = content.strip()
                return final_answer, (messages if self.return_conversation else [])

            async def run_call(tc):
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    return tc.id, tc.function.name, f"Invalid JSON arguments: {tc.function.arguments}"
                result = await self._call_local_tool(tc.function.name, arguments)
                return tc.id, tc.function.name, result

            results = await asyncio.gather(*[run_call(tc) for tc in msg.tool_calls])
            for tool_call_id, name, result in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "content": result,
                })
        else:
            abort_reason = f'max_turns ({self.max_turns}) reached'

        logger.warning(f'{abort_reason}, forcing final answer.')
        messages.append({
            "role": "user",
            "content": FORCING_FINAL_ANSWER_PROMPT,
        })
        # Strip reasoning fields from prior assistant turns before the
        # tool-less call. Per deepseek docs, prior reasoning shouldn't be
        # included in a request without tool_calls; empirically, leaving 20+
        # "let me search next" reasoning blocks in history makes deepseek
        # output everything into reasoning again and leave `content` empty.
        clean_messages = []
        for m in messages:
            if m.get("role") == "assistant" and (m.get("reasoning") or m.get("reasoning_content")):
                m = {k: v for k, v in m.items() if k not in ("reasoning", "reasoning_content")}
            clean_messages.append(m)

        try:
            forced_msg = await self._chat_no_tools(clean_messages)
            final_answer = (forced_msg.content or '').strip()
            if final_answer:
                messages.append({"role": "assistant", "content": final_answer})
        except Exception as e:
            logger.error(f"Forced final-answer call failed: {type(e).__name__}: {e}")

        # Last-ditch fallback: scan history for any content the model
        # actually produced. Check `content` first, then fall back to the
        # last assistant `reasoning` — for thinking-mode models the working
        # analysis often only lives in reasoning, and a structured trace is
        # still more useful than `nan`.
        if not final_answer:
            for m in reversed(messages):
                if m.get("role") == "assistant" and (m.get("content") or '').strip():
                    final_answer = m["content"].strip()
                    break
        if not final_answer:
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    r = m.get("reasoning") or m.get("reasoning_content") or ''
                    if r.strip():
                        logger.warning("Falling back to last assistant reasoning as final answer.")
                        final_answer = r.strip()
                        break
        return final_answer, (messages if self.return_conversation else [])

    async def async_qa(self, questions):
        async def job(q):
            async with self.sem:
                try:
                    return await self.query(q)
                except Exception as e:
                    logger.error(f"Query failed: {type(e).__name__}: {e}")
                    return f'Error: {e}', []

        tasks = [job(q) for q in questions]
        results = await tqdm_asyncio.gather(*tasks)

        return results

    async def retry_errors(self, output_path):
        df = pd.read_csv(output_path)
        error_mask = df['response'].astype(str).str.startswith('Error')
        error_indices = df.index[error_mask].tolist()
        if not error_indices:
            print('No error rows found.')
            return
        questions = df.loc[error_indices, 'question'].tolist()
        print(f'Retrying {len(questions)} error rows...')
        results = await self.async_qa(questions)
        df.loc[error_indices, 'response'] = [r[0] for r in results]
        if self.return_conversation:
            df.loc[error_indices, 'conversation'] = [json.dumps(r[1], ensure_ascii=False) for r in results]
        df.to_csv(output_path, index=False)
        print(f'Patched {len(questions)} rows in {output_path}')

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


def main(output_path, input_path, api_key, base_url, model_name, max_concurrency, max_turns, chat_args, return_conversation, search_backend, serper_api_key, retry_errors):
    if serper_api_key:
        os.environ["SERPER_API_KEY"] = serper_api_key
    agent_client = AgentClient(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        max_concurrency=max_concurrency,
        max_turns=max_turns,
        chat_args=chat_args,
        return_conversation=return_conversation,
        search_backend=search_backend,
    )
    if retry_errors:
        asyncio.run(agent_client.retry_errors(output_path))
    else:
        asyncio.run(agent_client.eval(output_path, input_path))
    print('--- All done ---')



if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
