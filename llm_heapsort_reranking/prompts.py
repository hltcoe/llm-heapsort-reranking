import re


class Prompt:
    def format(self, docs: list[str], query: str) -> list[dict[str, str]]: ...

    def parse_completion(self, completion: str) -> list[int]: ...


class Llama2(Prompt):
    PRE = 'Given a query "{query}", which of the following passages is the most relevant one to the query?\n'
    POST = "\nOutput only the passage label of the most relevant passage:"

    def parse_completion(self, completion: str) -> list[int]:
        # remove any punctuation from the end of completion
        completion = re.sub(r"[.,!? ]+$", "", completion).strip()

        raworder = [int(match.group(1)) - 1 for match in re.finditer(r"(-?\d+)", completion)]
        return raworder

    def format(self, docs: list[str], query: str) -> list[dict[str, str]]:
        numdocs = len(docs)
        doc_strings = [f'Passage {i + 1}: "{doc}"\n' for i, doc in enumerate(docs)]
        prestring = self.PRE.format(numdocs=numdocs, query=query)
        poststring = self.POST.format(numdocs=numdocs, query=query)
        user_role = {"role": "user", "content": prestring + "\n" + "\n".join(doc_strings) + "\n" + poststring}
        return [user_role]


class Standard(Prompt):
    SYSTEM = "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pool of passages based on their relevance to the query."
    PRE = 'Given a query "{query}", which of the following passages is the most relevant one to the query?'
    POST = "The output format should be [ID], e.g., [4] meaning document 4 is most relevant to the search query. Only respond with the passage number; do not say any word or explain."

    def parse_completion(self, completion: str) -> list[int]:
        # remove any punctuation from the end of completion
        completion = re.sub(r"[.,!? ]+$", "", completion).strip()

        raworder = [int(match.group(1)) - 1 for match in re.finditer(r"\[(-?\d+)\]", completion)]
        return raworder

    def format(self, docs: list[str], query: str) -> list[dict[str, str]]:
        numdocs = len(docs)
        doc_strings = [f"[{i + 1}] {doc}" for i, doc in enumerate(docs)]
        prestring = self.PRE.format(numdocs=numdocs, query=query)
        poststring = self.POST.format(numdocs=numdocs, query=query)
        system_role = {"role": "system", "content": self.SYSTEM}
        user_role = {"role": "user", "content": prestring + "\n" + "\n".join(doc_strings) + "\n" + poststring}
        return [system_role, user_role]


class CoT:
    SYSTEM = "You are an intelligent assistant that can identify the best passage based on its relevance to a query."
    PRE = "I will provide you with {numdocs} passages in no particular order, each indicated by a numerical identifier in square brackets. Identify the best passage based on its relevance to this search query: {query}.\n"
    POST = 'Search Query: {query}.\nStep 1: Decide whether there are any passages relevant to the search query. If there are none, answer "no" and do nothing else. Do not explain or give any reasoning.\nStep 2: If any passage is relevant, identify the best passage above based on its relevance to the search query. The output format should be [ID], e.g., [4] meaning document 4 is most relevant to the search query. Only respond with the best passage number; do not say any word or explain.'

    def parse_completion(self, completion: str) -> list[dict[str, str]]:
        # remove any punctuation from the end of completion
        completion = re.sub(r"[.,!? ]+$", "", completion).strip()

        if (
            completion.strip().lower().endswith("no")
            or completion.strip().lower().endswith("none")
            or completion.strip().endswith("[]")
            or not completion.strip()
        ):
            completion = "[-1]"
            print("*** CoT prompt returned none relevant")
        raworder = [int(match.group(1)) - 1 for match in re.finditer(r"\[(-?\d+)\]", completion)]
        return raworder

    def format(self, docs: list[str], query: str) -> str:
        numdocs = len(docs)
        doc_strings = [f"[{i + 1}] {doc}" for i, doc in enumerate(docs)]
        prestring = self.PRE.format(numdocs=numdocs, query=query)
        poststring = self.POST.format(numdocs=numdocs, query=query)
        system_role = {"role": "system", "content": self.SYSTEM}
        user_role = {"role": "user", "content": prestring + "\n" + "\n".join(doc_strings) + "\n" + poststring}
        return [system_role, user_role]
