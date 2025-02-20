# Setup
- `uv venv venv`
- `source venv/bin/activate`
- `uv pip install -r requirements.txt`

# Experiments
Run files are availables in `runs/`. See `runs/runs.csv` for the commands used to produce each run.
You can use `run.sh` to run these commands with VLLM. For example, to test with Llama 3.1 8B:
```
bash run.sh --coll dl20/first128_collection_passage.tsv --input dl20/bm25.run --map dl20/first128_mapping.tsv --model meta-llama/Llama-3.1-8B-Instruct --nary 9 --output test-dl20-llama3 --pool 1 --prompt original --rerankN 100 --rerankP dl20-first128/ranking.trec --topics dl20/topics.jsonl --topk 10
```
Explanation of the command line arguments:
- `--coll <collection>`: a TSV file mapping passage IDs to text
- `--input <run>`: a run file from first-stage retrieval in TREC format
- `--map <mapping>`: a mapping between the original doc IDs and passage IDs
- `--model <model>`: HuggingFace model ID to use for predictions
- `--nary <nary>`: number of children of each parent node in the tree
- `--output <file>`: output run file
- `--pool <n>`: number of queries to run in parallel
- `--prompt <prompt>`: either original or cot, to use the original or CoT prompt
- `--rerankN <n>`: number of documents to rerank from the input run file
- `--rerankP <run>`: the input run file with doc IDs rewritten to passage IDs
- `--topics <file>`: a jsonl file containing topics in the NeuCLIR format
- `--topk <k>`: finish reranking after finding the top k documents

