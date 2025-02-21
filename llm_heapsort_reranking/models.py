import json
import time

from openai import OpenAI

import llm_heapsort_reranking.prompts as prompts


class LLM:
    def create_prompt(self, topic, sequence, docs): ...

    def run_prompt(self, prompt, model): ...

    def parse_completion(self, completion): ...

    def count_tokens(self, text): ...


class LLM_RANKER(LLM):
    def __init__(self, model, prompt, mapping, temperature):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.parallel = False

        if prompt.lower() == "original":
            self.prompt = prompts.Standard()
        elif prompt.lower() == "cot":
            self.prompt = prompts.CoT()
        else:
            raise ValueError("unknown prompt:", prompt)

        self.psg2doc = mapping
        self.doc2psgs = {}
        for pid, docid in self.psg2doc.items():
            self.doc2psgs.setdefault(docid, []).append(pid)

        # passage IDs that should be ignored when reranking: topicid -> set()
        self.excluded_psgs = {}

    def exclude_passages_from_same_doc(self, topicid, psgid):
        docid = self.psg2doc[psgid]
        self.excluded_psgs.setdefault(topicid, set()).update(self.doc2psgs[docid])

    # Glue together the prompt
    def create_prompt(self, topicid, sequence, topics, documents):
        topicid = topicid if topicid in topics else int(topicid)
        query = topics[topicid]
        docseq = [documents[x] for x in sequence]
        return self.prompt.format(docs=docseq, query=query)

    def parse_completion(self, completion, length, topicid):
        raworder = self.prompt.parse_completion(completion)

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

        return filled

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
        completion = self.run_prompt(prompt)
        parsed = self.parse_completion(completion, len(passages), topicid)
        return parsed

    # Reorder top docs and paste in the remaining ones
    def apply_new_order(self, topicid, run, neworder):
        unordered = run[topicid]
        ordered = [unordered[i] for i in neworder] + unordered[len(neworder) :]
        return ordered


class LLMAPI(LLM_RANKER):
    def __init__(
        self,
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        prompt=None,
        mapping=None,
        temperature=None,
        api_url=None,
        api_key=None,
    ):
        super().__init__(model, prompt, mapping, temperature)
        self.parallel = True
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.model = model

    def _retry_prompt(self, prompt, max_retries, delay, max_delay):
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    max_tokens=20,
                    temperature=self.temperature,
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

    def run_prompt(self, prompt, max_retries=10, delay=2, max_delay=60):
        return self._retry_prompt(prompt, max_retries, delay, max_delay).strip()


class Llama2(LLM_RANKER):
    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", prompt=None, mapping=None, temperature=None):
        super().__init__(model, prompt, mapping, temperature)
        self.model = model

        assert prompt == "original"
        self.prompt = prompts.Llama2()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.use_default_system_prompt = False
        self.llm = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.float16).eval()

    def run_prompt(self, prompt):
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt += " Passage:"

        print("PROMPT:", prompt)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.llm.generate(input_ids, do_sample=False, temperature=self.temperature, top_p=None, max_new_tokens=20)[0]
        output = self.tokenizer.decode(output_ids[input_ids.shape[1] :], skip_special_tokens=True).strip()

        return output
