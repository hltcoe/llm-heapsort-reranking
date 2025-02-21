import argparse
import os
import sys
from collections import defaultdict
from multiprocessing import Pool, Process

from tqdm import tqdm

from llm_heapsort_reranking.models import LLMAPI, Llama2
from llm_heapsort_reranking.sort import sort_array_passage
from llm_heapsort_reranking.utils import load_documents, load_topics, output_run
from llm_heapsort_reranking.vllm import launch_vllm


def parse_args():
    parser = argparse.ArgumentParser(description="Rerank with LLM")
    parser.add_argument("--collection", help="passage collection TSV file", required=True)
    parser.add_argument("--topics", help="topic TSV file", required=True)
    parser.add_argument("--input", help="run file in TREC format", required=True)
    #
    parser.add_argument("--output", "-o", help="new ranked file", required=True)
    parser.add_argument("--nary", help="number of children in the tree", default=4, type=int)
    parser.add_argument("--rerankN", help="depth to look in ranked list - default rerank all", type=int)
    parser.add_argument("--topk", help="Number of documents to output in list", default=20, type=int)
    parser.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Model name")
    parser.add_argument("--temperature", default=0.9, type=float, help="Temperature for LLM")
    parser.add_argument("--prompt", choices=["original", "cot"], default="original", help="Prompt to use")
    #
    parser.add_argument("--api-url", default="", help="API URL to use; vLLM will be started locally if this is blank")
    parser.add_argument("--api-key", default="", help="API key to use")
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs to use for local vLLM's tensor parallelism")
    #
    parser.add_argument("--pool", default=1, type=int, help="Number of topics to rerank in parallel")
    parser.add_argument("--shard", default=-1, type=int, help="Shard to process")
    parser.add_argument("--total-shards", default=0, type=int, help="Total number of shards (or 0 to disable)")

    return parser.parse_args()


def sort_passage(args):
    documents = load_documents(args.collection)  # <docid>_<idx> -> text
    topics = load_topics(args.topics)
    full_rank = defaultdict(list)
    with open(args.input) as fin:
        for line in fin:
            split = line.strip().split()
            full_rank[split[0]].append(split[2])

    full_rank_docs = {docid for topicid in full_rank for docid in full_rank[topicid]}
    documents = {pid: txt for pid, txt in documents.items() if pid.rsplit("_", maxsplit=1)[0] in full_rank_docs}

    best_passages = {topicid: [docid for docid in full_rank[topicid]] for topicid in full_rank}
    mapping = {fullid: fullid.rsplit("_", maxsplit=1)[0] for fullid in documents}

    output = {}
    all_worker_args = [(run, best_passages[run], topics, documents, mapping, full_rank, args) for run in sorted(best_passages)]

    # split up queries
    if args.total_shards > 0:
        assert args.shard < args.total_shards
        assert args.shard >= 0
        args.output += f"_shard{args.shard}"
        all_worker_args = [
            worker_args for idx, worker_args in enumerate(all_worker_args) if idx % args.total_shards == args.shard
        ]

    assert args.pool >= 1
    if args.pool == 1:
        for worker_args in tqdm(all_worker_args, desc="reranking topics"):
            run = worker_args[0]
            output[run] = rerank_topic(worker_args)
    else:
        with Pool(args.pool) as p:
            for worker_args, worker_output in zip(
                all_worker_args, tqdm(p.imap(rerank_topic, all_worker_args), total=len(all_worker_args), desc="reranking topics")
            ):
                run = worker_args[0]
                output[run] = worker_output

    output_run(output, args.output, "llm")

    if args.total_shards > 0:
        with open(args.output + ".done", "wt") as donef:
            print("done", file=donef)

        # if all shards are done, concat to create the original args.output
        orig_output = args.output.split("_shard")[0]
        shard_outputs = [orig_output + f"_shard{shard}" for shard in range(args.total_shards)]
        if all(os.path.exists(shard_output + ".done") for shard_output in shard_outputs):
            outf = open(orig_output, "wt", encoding="utf-8")
            for shard_output in shard_outputs:
                with open(shard_output, "rt", encoding="utf-8") as inf:
                    for line in inf:
                        print(line.strip(), file=outf)
            outf.close()


def rerank_topic(worker_args):
    run, passages, topics, documents, mapping, full_rank, args = worker_args
    if args.model == "meta-llama/Llama-2-7b-chat-hf":
        # WARNING: this loads the model for every topic
        reranker = Llama2(model=args.model, prompt=args.prompt, mapping=mapping, temperature=args.temperature)
    else:
        # this works for any hosted model
        reranker = LLMAPI(
            model=args.model,
            prompt=args.prompt,
            mapping=mapping,
            temperature=args.temperature,
            api_key=args.api_key,
            api_url=args.api_url,
        )

    output = []
    print(f"Topic {run}")
    print(f"Init Order: {passages}")
    data = {"topicid": run, "rank": [], "topics": topics, "docs": documents}

    doc2psgs = {}
    for pid, docid in mapping.items():
        doc2psgs.setdefault(docid, []).append(pid)
    for idx, docid in enumerate(passages):
        data["rank"].extend(doc2psgs[docid])
        if idx + 1 == args.rerankN:
            break

    num_unsorted = len(data["rank"]) - args.topk
    if num_unsorted < 1:
        num_unsorted = 1
    sorted = sort_array_passage(data, args.nary, reranker, (run, mapping), num_unsorted)
    print(f"Final Order: {sorted['rank']}")
    found_docids = set()
    print(len(sorted["rank"]))
    for pid in sorted["rank"][: args.topk]:
        docid = pid.rsplit("_", maxsplit=1)[0]
        if docid not in found_docids:
            found_docids.add(docid)
            output.append(docid)
    for docid in full_rank[run]:
        if docid not in found_docids:
            found_docids.add(docid)
            output.append(docid)
    return output


def main():
    args = parse_args()
    if os.path.exists(args.output):
        print("skipping existing output file:", args.output)
        return 0

    vllm_process = None
    if args.model != "meta-llama/Llama-2-7b-chat-hf" and not args.api_url:
        # start vLLM if (1) no API URL has been specified and (2) we're not running Llama 2 via hgf
        args.api_url, vllm_process = launch_vllm(args.model, args.gpus)
        args.api_key = "none"

    if vllm_process:
        retcode = 1
        try:
            p = Process(target=sort_passage, args=(args,))
            p.start()
            p.join()
            retcode = 0
        finally:
            sys.exit(retcode)
    else:
        sort_passage(args)


if __name__ == "__main__":
    main()
