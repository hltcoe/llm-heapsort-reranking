# LLM Setwise Reranking using Heapsort

## Setup
- Create and load a venv
- Install dependencies: `pip install -r requirements.txt`

## Preprocess datasets
- `python -m llm_heapsort_reranking.preprocess --dataset dl19 --run data/dl19/bm25.run --output data/dl19`
- `python -m llm_heapsort_reranking.preprocess --dataset dl20 --run data/dl20/bm25.run --output data/dl20`
- `python -m llm_heapsort_reranking.preprocess --dataset robust04 --run data/robust04/bm25.run --output data/robust04`

## Rerank with a LLM
LLM calls use an OpenAI-compatible API, which can be provided by vLLM running locally or by an external service (e.g., OpenAI or [together.ai](https://together.ai)). The default is to run vLLM locally with one GPU.

Example command:
```bash
python -m llm_heapsort_reranking.run --gpus 1 --collection data/dl19/first128_collection_passage.tsv --input data/dl19/bm25.run --model meta-llama/Llama-3.1-8B-Instruct --nary 9 --temperature 0 --pool 5 --prompt original --rerankN 100 --topics data/dl19/topics.tsv --topk 10 --output dl19_nary9_n100_k10_llama8b.run
```
Explanation of the command line arguments:
- `--collection <file>`: a TSV file mapping document IDs to text (*created in preprocessing*)
- `--input <file>`: a run file from first-stage retrieval in TREC format (*our bm25.run or a run provided by you*)
- `--topics <file>`: a TSV file mapping query IDs to text
- `--model <model>`: HuggingFace model to use for predictions
- `--nary <nary>`: number of children of each parent node in the tree
- `--output <file>`: output run file
- `--prompt <prompt>`: either original or cot to use the original or CoT prompt
- `--rerankN <n>`: number of documents to rerank from the input run file
- `--topk <k>`: finish reranking after finding the top k documents
- `--api-url <url>`: OpenAI-compatible API to call for ranking; vLLM will be started locally if this is not set
- `--api-key <key>`: API key to use with the API; not needed for vLLM
- `--gpus <int>`: number of GPUs to use with the local vLLM instance; ignored if `--api-url` is set

## Experiments
Run files are availables in `runs/`. See the *command* column of [runs/runs.csv](runs/runs.csv) for the arguments used to produce each run, which can be used with `python -m llm_heapsort_reranking.run` as shown above.