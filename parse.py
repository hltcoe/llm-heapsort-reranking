import csv
import json
import os
import sys
import time

from trecrun import TRECRun


def parse_row(d):
    args = {}
    args["--topk"] = 10

    dataset = d["Document Set"].lower()
    assert dataset in ("dl19", "dl20", "robust04")

    if d["tokens"] == "1st 128":
        chunking = "first128"
    elif d["tokens"] == "450":
        chunking = "450"
    else:
        assert False, d["tokens"]

    firststage = "bm25"
    args["--input"] = f"{dataset}/bm25.run"
    args["--map"] = f"{dataset}/{chunking}_mapping.tsv"
    args["--coll"] = f"{dataset}/{chunking}_collection_passage.tsv"
    args["--rerankP"] = f"{dataset}-{chunking}/ranking.trec"
    args["--topics"] = f"{dataset}/topics.jsonl"

    args["--prompt"] = None
    if d["Prompt"] == "Theirs":
        args["--prompt"] = "original"
    elif d["Prompt"] == "CoT":
        args["--prompt"] = "cot"
    elif d["Prompt"] == "CoT2":
        args["--prompt"] = "cot2"

    args["--rerankN"] = int(d["# Docs to Rerank"])
    assert args["--rerankN"] > 0 and args["--rerankN"] <= 1000

    args["--nary"] = int(d["n-ary Heap"].split("-ary")[0])
    assert args["--nary"] in (2, 4, 9, 19)

    args["--model"] = None
    bigpool = 10 if "robust" in dataset else 30
    smallpool = 2 if "robust" in dataset else 10

    args["--pool"] = bigpool
    if d["LLM"] == "llama-3":
        args["--model"] = "meta-llama/Llama-3.3-70B-Instruct"
        args["--pool"] = smallpool
    elif d["LLM"] == "llama-2":
        args["--model"] = "meta-llama/Llama-2-7b-chat-hf"
    elif d["LLM"] == "llama-3-8b":
        args["--model"] = "meta-llama/Llama-3.1-8B-Instruct"
    elif d["LLM"] == "llama-3.2-1B":
        args["--model"] = "meta-llama/Llama-3.2-1B-Instruct"
    elif d["LLM"] == "llama-3.2-3B":
        args["--model"] = "meta-llama/Llama-3.2-3B-Instruct"
    elif d["LLM"].startswith("Qwen2.5"):
        args["--model"] = "Qwen/" + d["LLM"]
    elif d["LLM"] == "Mixtral":
        args["--model"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    elif d["LLM"] == "Qwen":
        args["--pool"] = smallpool
        args["--model"] = "Qwen/Qwen2.5-72B-Instruct"
        if args["--prompt"] == "cot":
            args["--prompt"] = "cot2"
    else:
        raise ValueError(d["LLM"])

    return args, dataset, chunking, firststage


def add_efsun_row(d):
    dataset = d["Document Set"].lower()
    assert dataset in ("dl19", "dl20", "robust04"), dataset

    if d["tokens"] == "1st 128":
        chunking = "passage-first128"
    elif d["tokens"] == "450":
        chunking = "passage-450"

    model = d["LLM"].upper()

    if d["Prompt"] == "Theirs":
        prompt = "rankgpt"
    elif d["Prompt"] == "CoT":
        prompt = "cot"
    elif d["Prompt"] == "CoT2":
        prompt = "cot2"

    depth = "depth" + d["# Docs to Rerank"]

    nary = "n" + d["n-ary Heap"].split("-ary")[0]

    if model == "QWEN":
        basedir = "/exp/ekayi/heapsort/qwen-runs"
        prompt = prompt.replace("rankgpt", "original")
    else:
        basedir = "/exp/ekayi/heapsort/runs"
    outfn = f"{basedir}/{dataset}/{model}/{prompt}/{chunking}/{depth}/bm25.run-{nary}"

    if os.path.exists(outfn):
        qrels = {}
        with open("qrels." + dataset, "rt") as qrelf:
            for line in qrelf:
                qid, _, docid, label = line.strip().split(" ")
                qrels.setdefault(qid, {})[docid] = int(label)

        run = TRECRun(outfn)
        metrics = run.evaluate(qrels)
        row["nDCG@10"] = metrics["nDCG@10"]["mean"]


#    else:
#        print("missing efsun row:", row, outfn)

with open("stats.json", "rt", encoding="utf-8") as statsf:
    stats = json.load(statsf)

assert len(sys.argv) == 2
with open(sys.argv[1], "rt", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    orig_rows = list(reader)
    fieldnames = list(reader.fieldnames)  # + ["nDCG@10"]

rows = []
last_rows = []
vlast_rows = []  # keep these after Qwen to preserve order in edited spreadsheet
for row in orig_rows:
    #    if "nDCG@10" in row:
    #        del row["nDCG@10"]

    rows.append(row)

    #    if row["LLM"] == "llama-2":
    #        newrow = {k: v for k, v in row.items()}
    #        newrow["LLM"] = "llama-3-8b"
    #        rows.append(newrow)
    #    elif row["LLM"] == "Mixtral":
    #        newrow = {k: v for k, v in row.items()}
    #        newrow["LLM"] = "Qwen"
    #        last_rows.append(newrow)
    if row["LLM"] == "llama-3":
        # run llama 1B, 3B, 8B with same configuration as 70B
        for newmodel in ["llama-3.2-1B", "llama-3.2-3B", "llama-3-8b"]:
            break
            newrow = {k: v for k, v in row.items()}
            newrow["LLM"] = newmodel
            vlast_rows.append(newrow)

        for size in ["0.5", "1.5", "3", "7", "14", "32"]:
            newmodel = f"Qwen2.5-{size}B-Instruct"
            break
            newrow = {k: v for k, v in row.items()}
            newrow["LLM"] = newmodel
            vlast_rows.append(newrow)

modelf = {}
updated = False
rows.extend(last_rows)
rows.extend(vlast_rows)
for row in rows:
    assert row["Assigned to"] in ("Andrew", "Efsun")
    if row["Assigned to"] == "Efsun":
        add_efsun_row(row)

    # if row["Assigned to"] != "Andrew":
    #    continue

    args, dataset, chunking, firststage = parse_row(row)

    if args["--prompt"] == "cot":
        print("skipping")
        continue

    main_args = {
        k: v for k, v in args.items() if k not in ("--pool", "--input", "--map", "--coll", "--rerankP", "--topics", "--model")
    }
    model = args["--model"].split("/")[1]
    outfn = "__".join([dataset, chunking, firststage, model] + [f"{k}:{v}" for k, v in sorted(main_args.items())])

    args["--output"] = "out/" + outfn
    cmd = " ".join([f"{k} {v}" for k, v in sorted(args.items())])

    # if we have the run file from this row, populate the metrics column
    if os.path.exists("out/" + outfn):
        qrels = {}
        with open("qrels." + dataset, "rt") as qrelf:
            for line in qrelf:
                qid, _, docid, label = line.strip().split(" ")
                qrels.setdefault(qid, {})[docid] = int(label)

        updated = True
        run = TRECRun("out/" + outfn)
        metrics = run.evaluate(qrels)
        row["nDCG@10"] = metrics["nDCG@10"]["mean"]

    # if we calculated stats from the log files for this run, add them
    if args["--output"] in stats and row.get("nDCG@10", ""):
        run_stats = stats[args["--output"]]
        row["Query latency"] = run_stats["seconds/query"]
        row["Query LLM calls"] = run_stats["requests/query"]
        if run_stats["0nonerel_right"] or run_stats["0nonerel_wrong"]:
            row["None relevant (>0): TN"] = run_stats["0nonerel_right"]
            row["None relevant (>0): FN"] = run_stats["0nonerel_wrong"]
            row["None relevant (>1): TN"] = run_stats["1nonerel_right"]
            row["None relevant (>1): FN"] = run_stats["1nonerel_wrong"]
        updated = True

    if model not in modelf:
        modelf[model] = open("experiments-with-" + model, "wt")

    # if the row doesn't have a populated metric column already, add it to run
    if not row.get("nDCG@10", ""):
        if "robust" in dataset:
            total_shards = 10
            for shard in range(total_shards):
                shard_cmd = cmd + f" --total-shards {total_shards} --shard {shard}"
                print(shard_cmd, file=modelf[model])
        else:
            print(cmd, file=modelf[model])
#    elif dataset in ("dl19", "dl20") and chunking == "450" and args["--prompt"] == "original" and model == "Llama-3.3-70B-Instruct":
#        print("READDING", row)
#        print(cmd, file=modelf[model])

for f in modelf.values():
    f.close()

if updated:
    with open(f"{sys.argv[1]}-updated_{time.time()}.csv", "wt", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


time.time()
