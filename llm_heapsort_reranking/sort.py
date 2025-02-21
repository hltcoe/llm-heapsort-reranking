import argparse
import random
from concurrent.futures import ThreadPoolExecutor


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

    def task(cur, cur_length):
        local_result, local_count = heapify(data, cur_length, cur, nary, 0, find_best, swap)
        return local_result, local_count

    last_parent = (length - 2) // nary
    nodes_to_process = list(range(last_parent, -1, -1))

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(task, node, length) for node in nodes_to_process]
        for future in futures:
            _, operation_count = future.result()
            total_count[0] += operation_count

    return data, total_count[0]


def seq_build_heap(data, length, nary, find_best, swap):
    count = 0
    cur = (length - 2) // nary
    while cur >= 0:
        data, count = heapify(data, length, cur, nary, count, find_best, swap)
        cur -= 1
    return data, count


def sort_array_passage(data, nary, reranker, args, unsorted=1):
    find_best = reranker.find_best_passage
    swap = swap_passage
    build_heap = parallel_build_heap if reranker.parallel else seq_build_heap
    if unsorted < 1:
        unsorted = 1
    print(data["rank"])
    length = len(data["rank"])

    data, count = build_heap(data, length, nary, find_best, swap)
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
        print(data["rank"])
    sorted.extend(data["rank"])
    assert len(sorted) == length
    print(f"Total calls to best: {count}")
    data["rank"] = sorted
    return data


def swap_passage(data, best, index):
    tmp = data["rank"][best]
    data["rank"][best] = data["rank"][index]
    data["rank"][index] = tmp
    return data


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


def sort_array_int(data, nary, find_best, swap, unsorted=1):
    if unsorted < 1:
        unsorted = 1
    print(data)
    length = len(data)
    data, count = parallel_build_heap(data, length, nary, find_best, swap)
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


def sort_ints(data, topk, nary):
    random.shuffle(data)
    num_sort = len(data) - topk
    if num_sort < 1:
        num_sort = 1
    sorted_ints = sort_array_int(data, nary, find_best_int, swap_int, num_sort)
    return sorted_ints


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort ints with heapsort")
    parser.add_argument("--nary", help="number of children in the tree", default=4, type=int)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--topk", help="Number of ints to output in list", default=20, type=int)

    args = parser.parse_args()
    data = list(range(args.n))
    sorted_ints = sort_ints(data, args.topk, args.nary)
    print("heapsort:", sorted_ints)
