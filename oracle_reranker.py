import sys

from trecrun import TRECRun


assert len(sys.argv) == 3, "<qrel file> <run file>"
qrelfn, runfn = sys.argv[1:]

qrels = {}
with open(qrelfn, "rt") as qf:
    for line in qf:
        qid, _, docid, label = line.strip().split(" ")
        qrels.setdefault(qid, {})[docid] = int(label)

orig_run = TRECRun(runfn)

for topk in [10, 30, 100, 1000]:
    runk = orig_run.topk(topk)
    oracle_ranking = {
        qid: {docid: qrels[qid].get(docid, -1) for docid in docscores} for qid, docscores in runk.results.items() if qid != "672"
    }  # 672 has no relevant results in qrels
    oracle_run = TRECRun(oracle_ranking)
    oracle_run.write_trec_run(runfn + ".oracle-top%d" % topk)
