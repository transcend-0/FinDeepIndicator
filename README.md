# Financial Deep Indicator (FinDeepIndicator)

<p align="center">
  <a href="https://openfinarena.com/"><b>🏠︎ Homepage</b></a> |
  <a href="https://arxiv.org/abs/2608.00764"><b>📑 Paper</b></a> |
  <a href="https://huggingface.co/datasets/OpenFinArena/FinDeepIndicator"><b>🤗 Hugging Face</b></a>
</p>

FinDeepIndicator is the first benchmark dedicated to evaluating Deep Research (DR) agents in end-to-end financial indicator construction. Specifically, FinDeepIndicator evaluates DR agents across four stages in indicator construction: formula specification, data collection, indicator calculation, and answer generation, and covers fundamental, technical, and macroeconomic indicators organized into 21 fine-grained sub-categories. It contains 3,350 curated question-answer (QA) pairs derived from both U.S. and Chinese markets, 10 years of historical financial data, and 800 listed companies. Extensive experiments on search-equipped Large Language Models (LLMs) and DR agents show that, while LLMs generally perform well in formula specification, their accuracy drops substantially during data retrieval and numerical execution. DR agents consistently outperform search-equipped LLMs, yet remain unreliable in realistic financial analysis settings. These findings provide insights for developing more capable and trustworthy DR agents in finance.

## Environment

- Python 3.10
- openai 2.36

## Benchmark Construction

FinDeepIndicator constructs a diverse benchmark by randomly instantiating question templates with sampled variable values.

If you want to reconstruct the benchmark, run the following command to generate QA sets for US market.

```bash
cd QAGenerator/US
python QAGenerator.py
```

For CN market, just change the path to `QAGenerator/CN` and run the same command.

## Evaluation

In `qa_agent.sh`, `qa_llm_compare.sh`, `qa_llm_extract.sh`, and `qa_llm_search.sh`, set `OPENROUTER_API_KEY` and `SERPER_API_KEY`, then run the following command to evaluate the performance of LLMs and DR agents:

```bash
bash script/pipeline.sh search --input_path QA/all.csv --output_dir result/all_search
bash script/pipeline.sh agent --input_path QA/all.csv --output_dir result/all_agent
```