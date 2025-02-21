import argparse
from pathlib import Path

import ir_datasets as irds
from tqdm import tqdm
from transformers import AutoTokenizer
from trecrun import TRECRun


model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_dataset(name):
    if name == "robust04":
        ds = irds.load("disks45/nocr/trec-robust-2004")
    elif name == "dl19":
        ds = irds.load("msmarco-passage/trec-dl-2019/judged")
    elif name == "dl20":
        ds = irds.load("msmarco-passage/trec-dl-2020/judged")
    else:
        ds = irds.load(name)

    return ds


def get_topic(q):
    txt = q.description if hasattr(q, "description") else q.text
    return (q.query_id, txt)


def split_into_windows(text, window_size=450, stride=450):
    tokens = tokenizer(text, add_special_tokens=False, return_special_tokens_mask=False)["input_ids"]

    windows = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + window_size, len(tokens))
        window_tokens = tokens[start_idx:end_idx]

        window_text = tokenizer.decode(window_tokens).strip()
        windows.append(window_text)

        if stride == -1:
            break

        start_idx += stride

    return windows


def create_docs(docs, store, psgfn, window_size, stride):
    with psgfn.open("wt", encoding="utf-8") as psgf:
        idx = 0
        for docid in tqdm(docs, desc=f"window={window_size}"):
            text = store.get(docid).default_text().replace("\n", " ")
            for window in split_into_windows(text, window_size=window_size, stride=stride):
                print(f"{docid}_{idx}\t{window}", file=psgf)
                idx += 1


def main():
    parser = argparse.ArgumentParser(description="Prepare data to rerank")
    parser.add_argument("--dataset", help="IRDS dataset", default="dl19", required=True)
    parser.add_argument("--run", help="a run file in TREC format", default="data/dl19/bm25.run", required=True)
    parser.add_argument("--output", help="output path", default="data/dl19", required=True)

    args = parser.parse_args()
    args.output = Path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    run = TRECRun(args.run)
    ds = get_dataset(args.dataset)
    store = ds.docs_store()

    # write topics file
    topicfn = args.output / "topics.tsv"
    with topicfn.open("wt") as outf:
        for q in ds.queries:
            print("\t".join(get_topic(q)), file=outf)

    docs = sorted({docid for qid, scores in run.results.items() for docid in scores})

    create_docs(
        docs,
        store,
        args.output / "450_collection_passage.tsv",
        450,
        450,
    )

    create_docs(
        docs,
        store,
        args.output / "first128_collection_passage.tsv",
        128,
        -1,
    )


if __name__ == "__main__":
    main()
