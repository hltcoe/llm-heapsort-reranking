import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from openai import OpenAI
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Rerank with gpt4/mixtral")
    parser.add_argument("--input", "-i", help="trec file", required=True)
    parser.add_argument("--map", help="map passages to documents", required=True)
    parser.add_argument("--coll", help="passage collection", required=True)
    parser.add_argument("--rerankP", help="trec file of passages (from expand_ranklist)", required=True)
    parser.add_argument("--topics", help="topic file", required=True)
    parser.add_argument("--output", "-o", help="new ranked file", required=True)
    parser.add_argument("--qrels", help="qrels")
    parser.add_argument("--nary", help="number of children in the tree", default=4, type=int)
    parser.add_argument("--rerankN", help="depth to look in ranked list - default rerank all", type=int)
    parser.add_argument("--topk", help="Number of documents to output in list", default=20, type=int)
    parser.add_argument("--API_TOKEN", help="API token for the model to be used if needed")
    parser.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Model name")
    parser.add_argument("--prompt", choices=["original", "new", "cot", "norel", "cot2"], default="original", help="Prompt to use")
    parser.add_argument("--pool", default=1, type=int, help="Number of topics to rerank in parallel")
    parser.add_argument("--shard", default=-1, type=int, help="Shard index to process")
    parser.add_argument("--total-shards", default=0, type=int, help="Total number of shards (or 0 to disable)")

    return parser.parse_args()


"""# Loaders

Load various types of file. Not all are currently used.
"""


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
            docid, text = line.strip().split("\t", 1)
            result[docid] = text
    return result


def load_jsonl_topics(filename):
    result = {}
    with open(filename, "r") as infile:
        for line in infile:
            topic = json.loads(line)
            for entry in topic["topics"]:
                if entry["lang"] == "eng" and entry["source"] == "original":
                    if "topic_description" not in entry:
                        result[topic["topic_id"]] = {"title": entry["topic_title"], "description": entry["topic_title"]}
                    else:
                        result[topic["topic_id"]] = {"title": entry["topic_title"], "description": entry["topic_description"]}
    return result


"""
Support functions for integer sorting
"""


def find_best_int(data, parent, children, count):
    best = parent
    for i in children:
        if i < len(data) and data[i] > data[best]:
            best = i
    return best, count + 1


def swap_int(data, best, index):
    tmp = data[best]
    data[best] = data[index]
    data[index] = tmp
    return data


"""
Support functions for passage sorting
"""


def swap_passage(data, best, index):
    tmp = data["rank"][best]
    data["rank"][best] = data["rank"][index]
    data["rank"][index] = tmp
    return data


"""
Heapsort functions
"""


def heapify(data, length, index, nary, count, find_best, swap):
    first = index * nary + 1
    if first < length:
        children = []
        for i in range(nary):
            children.append(first + i)
        best, count = find_best(data, index, children, count)
    else:
        best = index
    if best != index:
        data = swap(data, best, index)
        data, count = heapify(data, length, best, nary, count, find_best, swap)
    return data, count


def parallel_build_heap(data, length, nary, find_best, swap):
    total_count = [0]

    def parallel_heapify_task(cur, cur_length):
        local_result, local_count = heapify(data, cur_length, cur, nary, 0, find_best, swap)
        return local_result, local_count

    last_parent = (length - 2) // nary
    nodes_to_process = list(range(last_parent, -1, -1))

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(parallel_heapify_task, node, length) for node in nodes_to_process]
        for future in futures:
            result_data, operation_count = future.result()
            total_count[0] += operation_count

    return data, total_count[0]


def orig_build_heap(data, length, nary, find_best, swap):
    count = 0
    cur = (length - 2) // nary
    while cur >= 0:
        data, count = heapify(data, length, cur, nary, count, find_best, swap)
        cur -= 1
    return data, count


# build_heap = orig_build_heap
build_heap = parallel_build_heap


def sort_array_int(data, nary, find_best, swap, unsorted=1):
    if unsorted < 1:
        unsorted = 1
    print(data)
    length = len(data)
    data, count = build_heap(data, length, nary, find_best, swap)
    print(f"Best call to build: {count}")
    assert len(data) == length
    print(data)
    sorted = []
    while len(data) > unsorted:
        sorted.append(data[0])
        data[0] = data[-1]
        data = data[:-1]
        data, count = heapify(data, len(data), 0, nary, count, find_best, swap)
        print(data)
    sorted.extend(data)
    assert len(sorted) == length
    print(f"Total calls to best: {count}")
    return sorted


def check_heap(data, nary, args):
    topicid, qrels, mapping = args
    if not qrels or topicid not in qrels:
        return
    length = len(data["rank"])
    cur = (length - 2) // nary
    while cur >= 0:
        parent = mapping[data["rank"][cur]]
        first = cur * nary + 1
        children = []
        for i in range(nary):
            if first + i < length:
                children.append(mapping[data["rank"][first + i]])
        if parent in qrels[topicid]:
            parent_judge = qrels[topicid][parent]
        else:
            parent_judge = 0
        for cid in children:
            if cid in qrels[topicid] and qrels[topicid][cid] > parent_judge:
                print(
                    f"WARNING: head property violation - parent {parent} of {parent_judge} less than child {cid} of {qrels[topicid][cid]}",
                    file=sys.stderr,
                )
        cur -= 1


def sort_array_passage(data, nary, reranker, swap, args, unsorted=1):
    find_best = reranker.find_best_passage
    if unsorted < 1:
        unsorted = 1
    print(data["rank"])
    length = len(data["rank"])
    """
  if data['topicid'] == '253':
    data['rank'] = ['4453840', '1992477', '4452980', '4453270', '4457930', '1163038', '1830516', '1655129', '1987120', '1985749', '2100113', '1985602', '4456175', '1988516', '2009122', '2199003', '1993650', '1656178', '1831830', '1652293', '1832186', '1655075', '1841438', '1974699', '1670244', '1690981', '1989657', '1955510', '1093058', '1639107']
    count = 11
  else:
  """
    data, count = build_heap(data, length, nary, find_best, swap)
    check_heap(data, nary, args)
    print(f"Best call to build: {count}")
    assert len(data["rank"]) == length
    print(data["rank"])
    sorted = []
    while len(data["rank"]) > unsorted:
        sorted.append(data["rank"][0])
        # passage has been added to the ranking, so ignore all other passages from the same doc
        reranker.exclude_passages_from_same_doc(data["topicid"], data["rank"][0])

        data["rank"][0] = data["rank"][-1]
        data["rank"] = data["rank"][:-1]
        data, count = heapify(data, len(data["rank"]), 0, nary, count, find_best, swap)
        check_heap(data, nary, args)
        print(data["rank"])
    sorted.extend(data["rank"])
    assert len(sorted) == length
    print(f"Total calls to best: {count}")
    data["rank"] = sorted
    return data


"""
LLM functions
"""
"""# LLMs"""


class LLM:
    def __init__(self):
        pass

    def create_prompt(self, topic, sequence, docs):
        pass

    def run_prompt(self, prompt, model):
        pass

    def parse_completion(self, completion):
        pass

    def count_tokens(self, text):
        pass


"""## GPT LLM

Uses the prompting approach created by https://github.com/Parry-Parry/TDPart/tree/main
"""

PROMPTS = {
    "original": (
        "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pool of passages based on their relevance to the query.",
        'Given a query "{query}", which of the following passages is the most relevant one to the query?',
        "The output format should be [ID], e.g., [4] meaning document 4 is most relevant to the search query. Only respond with the passage number; do not say any word or explain.",
    ),
    "cot": (
        "You are an intelligent assistant that can identify the best passage based on its relevance to a query.",
        "I will provide you with {numdocs} passages in no particular order, each indicated by a numerical identifier in square brackets. Identify the best passage based on its relevance to this search query: {query}.\n",
        'Search Query: {query}.\nStep 1: Are any passages relevant to the search query? Answer yes or no.\n Step 2: If any passage is relevant, identify the best passage above based on its relevance to the search query. The output format should be [ID], e.g., [4] meaning document 4 is most relevant to the search query. Only respond with the best passage number; do not say any word or explain. If no passage is relevant, respond with "[-1]" and nothing else.',
    ),
    "new": (
        "You are an intelligent assistant that can identify the best passage based on its relevance to a query.",
        "I will provide you with {numdocs} passages in no particular order, each indicated by a numerical identifier in square brackets. Identify the best passage based on its relevance to this search query: {query}.\n",
        "Search Query: {query}\nIdentify the best passage above based on its relevance to the search query. The output format should be [ID], e.g., [4] meaning document 4 is most relevant to the search query. Only respond with the passage number; do not say any word or explain.",
    ),
    "norel": (
        "You are an intelligent assistant that can identify the best passage based on its relevance to a query.",
        "I will provide you with {numdocs} passages in no particular order, each indicated by a numerical identifier in square brackets. Identify the best passage based on its relevance to this search query: {query}.\n",
        'Search Query: {query}\nIdentify the best passage above based on its relevance to the search query. The output format should be [ID], e.g., [4] meaning document 4 is most relevant to the search query. Only respond with the passage number; do not say any word or explain. If no passage is relevant, respond with "[-1]" and nothing else.',
    ),
    "cot2": (
        "You are an intelligent assistant that can identify the best passage based on its relevance to a query.",
        "I will provide you with {numdocs} passages in no particular order, each indicated by a numerical identifier in square brackets. Identify the best passage based on its relevance to this search query: {query}.\n",
        'Search Query: {query}.\nStep 1: Decide whether there are any passages relevant to the search query. If there are none, answer "no" and do nothing else. Do not explain or give any reasoning.\nStep 2: If any passage is relevant, identify the best passage above based on its relevance to the search query. The output format should be [ID], e.g., [4] meaning document 4 is most relevant to the search query. Only respond with the best passage number; do not say any word or explain.',
    ),
}


class LLM_RANKER(LLM):
    def __init__(self, model, prompt, mapping):
        super().__init__()
        self.model = model
        self.TIME_BETWEEN_RERANKINGS = 1.0

        self.SYSTEM_MESSAGE, self.PRE, self.POST = PROMPTS[prompt]
        self.prompt = prompt

        self.psg2doc = mapping
        self.doc2psgs = {}
        for pid, docid in self.psg2doc.items():
            self.doc2psgs.setdefault(docid, []).append(pid)

        # passage IDs that should be ignored when reranking: topicid -> set()
        self.excluded_psgs = {}

        # self.encoding = tiktoken.encoding_for_model(model)

    def exclude_passages_from_same_doc(self, topicid, psgid):
        docid = self.psg2doc[psgid]
        self.excluded_psgs.setdefault(topicid, set()).update(self.doc2psgs[docid])

    # Glue together the prompt
    def create_prompt(self, topicid, passages, topics, documents):
        print("*** excluded:", self.excluded_psgs)
        numdocs = len(passages)
        sequence = passages
        topicid = topicid if topicid in topics else int(topicid)
        query = topics[topicid].get("text", topics[topicid]["description"])
        docseq = [documents[x] for x in sequence]
        doc_strings = [f"[{x + 1}] {docseq[x]}" for x in range(len(sequence))]
        input = [f"[{i}] {x}" for i, x in enumerate(sequence)]
        print(f"INPUT {input}")
        prestring = self.PRE.format(numdocs=numdocs, query=query)
        poststring = self.POST.format(numdocs=numdocs, query=query)
        system_role = {"role": "system", "content": self.SYSTEM_MESSAGE}
        user_role = {"role": "user", "content": prestring + "\n" + "\n".join(doc_strings) + "\n" + poststring}
        # print(user_role)
        return [system_role, user_role]

    # This is where the LLM is actually called
    def run_prompt(self, prompt):
        pass

    # LLM responds with a sequence like "4 > 1 > 2 > 8". Pull out the ranks and
    # convert them to numbers. Make sure all numbers are represented, and
    # eliminate duplicates
    def parse_completion(self, completion, length, topicid):
        # remove any punctuation from the end of completion
        completion = re.sub(r"[.,!? ]+$", "", completion).strip()

        # try to catch errors with the CoT output, such as returning "No" / "None" / "[]" / empty
        if self.prompt in ("cot", "cot2"):
            if (
                completion.strip().lower().endswith("no")
                or completion.strip().lower().endswith("none")
                or completion.strip().endswith("[]")
                or not completion.strip()
            ):
                completion = "[-1]"
                print("*** CoT prompt returned none relevant")

        raworder = [int(match.group(1)) - 1 for match in re.finditer(r"\[(-?\d+)\]", completion)]
        # raworder = [int(x) - 1 for x in re.sub(r'[^0-9]', ' ', completion).strip().split()]

        dedup = set()
        deduped = []
        for x in raworder:
            if x not in dedup and x < length:
                if x >= 0 or self.prompt in ("cot", "cot2"):
                    deduped.append(x)
                    dedup.add(x)
        if len(deduped) != 1:
            print(f"Topic {topicid}: Corrected {raworder} to {deduped}")
            print("LLM output was:", completion)
        print(f"OUTPUT {deduped}")

        if len(deduped) >= 1:
            filled = deduped[0]
        else:
            filled = 0
        # filled = deduped + [i for i in range(length) if i not in dedup]
        # if filled != raworder:
        #  print(f"Topic {topicid}: Corrected {raworder} to {filled}")
        return filled

    # Rerank a single topic
    def rerank_topic(self, topicid, run, numdocs, topics, documents):
        prompt = self.create_prompt(topicid, run, numdocs, topics, documents)
        completion = self.run_prompt(
            prompt,
        )
        parsed = self.parse_completion(completion, numdocs, topicid)
        reranked = self.apply_new_order(topicid, run, parsed)
        return reranked

    def find_best_passage(self, data, parent, children, count):
        passages = [(parent, data["rank"][parent])]
        for idx in children:
            if idx < len(data["rank"]):
                passages.append((idx, data["rank"][idx]))

        # filter out passages that have already been excluded for this topic
        passages = [(idx, pid) for idx, pid in passages if pid not in self.excluded_psgs.get(data["topicid"], set())]
        print(passages)

        # if all (or most) passages have been excluded, return without a LLM call
        if len(passages) == 0:
            print("*** all candidate passages excluded")
            best = parent
            return best, count + 1

        psg_idxs, psg_ids = zip(*passages)

        if len(passages) == 1:
            print("*** only one included candidate")
            best = psg_idxs[0]
        else:
            pass_loc = self.best_passage(data["topicid"], psg_ids, data["topics"], data["docs"])
            print(pass_loc)
            if self.prompt in ("cot", "cot2") and pass_loc < 0:
                print("*** returning parent because CoT decided none relevant")
                print("NONE-RELEVANT", json.dumps({"topic": data["topicid"], "psgs": list(psg_ids)}), "/NONE-RELEVANT/")
                # LLM said no passages are relevant, so exclude all the passages provided
                self.excluded_psgs.setdefault(data["topicid"], set()).update(psg_ids)
                best = parent
            else:
                best = psg_idxs[pass_loc]
        return best, count + 1

    def best_passage(self, topicid, passages, topics, documents):
        prompt = self.create_prompt(topicid, passages, topics, documents)
        # print('FIX ME random choice')
        # completion = f"[{random.randint(1, len(passages))}]"
        completion = self.run_prompt(prompt)
        parsed = self.parse_completion(completion, len(passages), topicid)
        return parsed

    # Reorder top docs and paste in the remaining ones
    def apply_new_order(self, topicid, run, neworder):
        unordered = run[topicid]
        ordered = [unordered[i] for i in neworder] + unordered[len(neworder) :]
        return ordered

    # Rerank all topics
    def rerank_run(self, run, numdocs_to_rerank, topics, documents):
        results = {}
        for topicid in run.keys():
            results[topicid] = self.rerank_topic(topicid, run, numdocs_to_rerank, topics, documents)
            # I'm not certain whether this is necessary or what the value should be
            time.sleep(self.TIME_BETWEEN_RERANKINGS)
        return results

    # For each document find the best passage
    def find_representative_passage(self, run, topics, documents):
        results = {}
        for topicid in run.keys():
            if len(run[topicid]) == 0:
                continue
            passages = []
            doc_pass = []
            rank = run[topicid][0][1]
            for pass_info in run[topicid]:
                if pass_info[1] != rank:
                    if doc_pass:
                        if len(doc_pass) == 1:
                            pass_id = doc_pass[0]
                        else:
                            pass_loc = self.best_passage(topicid, doc_pass, topics, documents)
                            pass_id = doc_pass[pass_loc]
                            time.sleep(self.TIME_BETWEEN_RERANKINGS)
                        passages.append(pass_id)
                        doc_pass = []
                        rank = pass_info[1]
                doc_pass.append(pass_info[0])
            if doc_pass:
                if len(doc_pass) == 1:
                    pass_id = doc_pass[0]
                    passages.append(pass_id)
                elif len(doc_pass) > 1:
                    pass_loc = self.best_passage(topicid, doc_pass, topics, documents)
                    pass_id = doc_pass[pass_loc]
                    time.sleep(self.TIME_BETWEEN_RERANKINGS)
                    passages.append(pass_id)
            results[topicid] = passages
        return results

    # def count_tokens(self, text):
    #   return len(self.encoding.encode(text))


# Output a run to a file in TREC run format
def output_run(run, filename, runid="test"):
    with open(filename, "wt") as outfile:
        for topicid in run.keys():
            ranking = run[topicid]
            for rank, docid in enumerate(ranking):
                score = len(ranking) - rank
                outfile.write(f"{topicid} Q0 {docid} {rank+1} {score} {runid}\n")
    print(f"Wrote to file {filename}")


class GPT(LLM_RANKER):
    def __init__(self, user_api_key, model="gpt-4o", prompt=None, mapping=None):
        super().__init__(model, prompt, mapping)
        # You'll need to store the COE OpenAI key -- click the key icon at the
        # very left of the screen
        self.api = OpenAI(api_key=user_api_key)
        # print(f'FIX ME remove time change')
        # self.TIME_BETWEEN_RERANKINGS = 0

    def run_prompt(self, prompt):
        # print(f"Prompt: {prompt}")
        # print(f"Number of tokens: {self.count_tokens(prompt[0]['content'] + prompt[1]['content'])}")

        completion = self.api.chat.completions.create(model=self.model, messages=prompt)
        return completion.choices[0].message.content


class Mixtral(LLM_RANKER):
    def __init__(self, model="mistralai/Mixtral-8x7B-Instruct-v0.1", prompt=None, mapping=None):
        super().__init__(model, prompt, mapping)
        # You'll need to store the COE OpenAI key -- click the key icon at the
        # very left of the screen

        if "LUMI_LMOD_FAMILY_COMPILER" in os.environ:
            # looks like we're running on lumi
            self.client = OpenAI(
                api_key="MY API KEY",
                base_url=f"http://0.0.0.0:{os.environ['API_PORT']}/v1",
            )
        elif "TOGETHER_KEY" in os.environ:
            # use together.ai
            self.client = OpenAI(
                api_key=os.environ["TOGETHER_KEY"],
                base_url="https://api.together.xyz/v1",
            )
        else:
            # default to APL
            self.client = OpenAI(
                api_key="MY API KEY",
                base_url="https://opal.livelab.jhuapl.edu/v2",
            )

        self.model = model
        self.TIME_BETWEEN_RERANKINGS = 0

    def _retry_prompt(self, prompt, max_retries, delay, max_delay):
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    max_tokens=150,
                    temperature=0.9,
                    stream=False,
                )

                resp = response.choices[0].message.content

                if len(resp) > 0:
                    return resp
                else:
                    raise Exception("Empty response received")
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")

                delay = min(delay * (2 ** (retry_count - 1)), max_delay)
                print(f"Attempt {retry_count} failed. Error: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)

    def run_prompt(self, prompt, max_reprompts=3, max_retries=10, delay=2, max_delay=60):
        for _ in range(max_reprompts):
            resp = self._retry_prompt(prompt, max_retries, delay, max_delay).strip()
            if "2-7b-chat" not in self.model:
                return resp

            if "2-7b-chat" in self.model:
                match = re.search(r"(\d+) is the most relevant", resp)
                if match:
                    return match.group(1)

                resp = re.sub(r"(Answer|Passage|Number)[:-]? ?#?", "", resp, flags=re.IGNORECASE).strip()

            if self.prompt != "original" or len(resp) <= 4:  # output should be no longer than: [99]
                break
            print("*** reprompting:", len(resp), resp)

        return resp


class Llama2(LLM_RANKER):
    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", prompt=None, mapping=None):
        super().__init__(model, prompt, mapping)
        assert model == "meta-llama/Llama-2-7b-chat-hf"
        self.model = model
        self.TIME_BETWEEN_RERANKINGS = 0

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.use_default_system_prompt = False
        self.llm = AutoModelForCausalLM.from_pretrained(
            model, device_map="auto", torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).eval()

    def run_prompt(self, prompt):
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt += " Passage:"

        print("PROMPT:", prompt)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        # self.total_prompt_tokens += input_ids.shape[1]

        output_ids = self.llm.generate(input_ids, do_sample=False, temperature=0.0, top_p=None, max_new_tokens=1)[0]

        # self.total_completion_tokens += output_ids.shape[0]

        output = self.tokenizer.decode(output_ids[input_ids.shape[1] :], skip_special_tokens=True).strip()

        return output


"""# Main"""


def sort_passage(args):
    # Specify reranking task.  Pull documents from ir_datasets.  I couldn't figure
    # out how to cache the dataset locally.
    documents = load_documents(args.coll)
    topics = load_jsonl_topics(args.topics)

    # if args.model == 'gpt-4o':
    #   if not args.API_TOKEN:
    #     print('ERROR: GPT requires API_TOKEN parameter', file=sys.stderr)
    #     return
    #   reranker = GPT(args.API_TOKEN, model=args.model, prompt=args.prompt, mapping=mapping)
    # else:
    #   # this works for any hosted model
    #   reranker = Mixtral(model=args.model, prompt=args.prompt, mapping=mapping)

    qrel = {}
    if args.qrels:
        with open(args.qrels) as fin:
            for line in fin:
                split = line.strip().split()
                if split[0] not in qrel:
                    qrel[split[0]] = {}
                qrel[split[0]][split[2]] = int(split[-1])
    mapping = {}
    with open(args.map) as fin:
        for line in fin:
            split = line.strip().split()
            docid = split[1].split("_")[0]
            mapping[split[0]] = docid
    full_rank = defaultdict(list)
    with open(args.input) as fin:
        for line in fin:
            split = line.strip().split()
            full_rank[split[0]].append(split[2])
    print("Data loaded")

    # Use this to test a single topic
    testrun, runid = load_trec_run(args.rerankP, limit=1000)
    # test_topicids = ['200', '253', '206']
    # testtopic = {"253":testrun['253']}
    # testtopic = {}
    # for tid in test_topicids:
    #  testtopic[tid] = testrun[tid]
    testtopic = testrun
    best_passages = {topicid: [docid for docid, _ in testrun[topicid]] for topicid in testrun}
    # best_passages = reranker.find_representative_passage(testtopic, topics, documents)
    # print('Best passages selected')

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

    if os.path.exists(args.output):
        print("skipping existing output file:", args.output)
        return 0

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

    # reranked = reranker.rerank_topic('200', testrun, 30, topics, documents)
    output_run(output, args.output, f"{runid}GPT")

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


def rerank_topic(worker_args, qrel=None):
    run, passages, topics, documents, mapping, full_rank, args = worker_args
    if args.model == "gpt-4o":
        if not args.API_TOKEN:
            print("ERROR: GPT requires API_TOKEN parameter", file=sys.stderr)
            return
        reranker = GPT(args.API_TOKEN, model=args.model, prompt=args.prompt, mapping=mapping)
    elif "Llama-2" in args.model:
        reranker = Llama2(model=args.model, prompt=args.prompt, mapping=mapping)
    else:
        # this works for any hosted model
        reranker = Mixtral(model=args.model, prompt=args.prompt, mapping=mapping)

    output = []
    print(f"Topic {run}")
    print(f"Init Order: {passages}")
    data = {"topicid": run, "rank": passages, "topics": topics, "docs": documents}

    if args.rerankN and len(data["rank"]) > args.rerankN:
        # data['rank'] = data['rank'][:args.rerankN]
        docs = set()
        psgs = []
        for pid in data["rank"]:
            psgs.append(pid)
            docs.add(
                mapping[pid].split(
                    "_",
                )[0]
            )
            if len(docs) == args.rerankN:
                break
        data["rank"] = psgs
    num_unsorted = len(data["rank"]) - args.topk
    if num_unsorted < 1:
        num_unsorted = 1
    sorted = sort_array_passage(data, args.nary, reranker, swap_passage, (run, qrel, mapping), num_unsorted)
    print(f"Final Order: {sorted['rank']}")
    found_docids = set()
    print(len(sorted["rank"]))
    for pid in sorted["rank"][: args.topk]:
        docid = mapping[pid]
        if docid not in found_docids:
            found_docids.add(docid)
            output.append(docid)
    for docid in full_rank[run]:
        if docid not in found_docids:
            found_docids.add(docid)
            output.append(docid)
    return output


def sort_ints(args):
    if not args.rerankN:
        print("ERROR: must set rerankN to sort ints", file=sys.stderr)
        return
    data = []
    for i in range(args.rerankN):
        data.append(i)
    random.shuffle(data)
    num_sort = args.rerankN - args.topk
    if num_sort < 1:
        num_sort = 1
    sorted = sort_array_int(data, args.nary, find_best_int, swap_int, num_sort)
    print(sorted)


if __name__ == "__main__":
    args = parse_args()
    # sort_ints(args)
    sort_passage(args)
