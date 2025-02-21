from collections import defaultdict


# Output a run to a file in TREC run format
def output_run(run, filename, runid="test"):
    with open(filename, "wt") as outfile:
        for topicid in run.keys():
            ranking = run[topicid]
            for rank, docid in enumerate(ranking):
                score = len(ranking) - rank
                outfile.write(f"{topicid} Q0 {docid} {rank + 1} {score} {runid}\n")
    print(f"Wrote to file {filename}")


def load_trec_run(filename, limit=1000):
    result = defaultdict(list)
    with open(filename, "r") as infile:
        for line in infile:
            queryid, _, docid, rank, score, runid = line.strip().split()
            if len(result[queryid]) < limit:
                result[queryid].append((docid, rank))
    return result, runid


def load_documents(filename):
    result = {}
    with open(filename, "r") as infile:
        for line in infile:
            docid, text = line.strip().split("\t", maxsplit=1)
            result[docid] = text
    return result


def load_topics(filename):
    topics = {}
    with open(filename, "rt", encoding="utf-8") as infile:
        for line in infile:
            qid, txt = line.strip().split("\t", maxsplit=1)
            topics[qid] = txt
    return topics
