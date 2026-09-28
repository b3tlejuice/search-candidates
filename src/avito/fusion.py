"""
Reciprocal Rank Fusion : этап N4
объединение списков по местам, а не по оценкам
"""
from collections import defaultdict


def rrf(lists, weights, k_rrf=60, topn=50, depth=1000):
    scores = defaultdict(float)
    for lst, w in zip(lists, weights):
        for r, item in enumerate(lst[:depth]):
            scores[item] += w / (k_rrf + r + 1)
    return sorted(scores, key=scores.get, reverse=True)[:topn]


def fuse_all(list_groups, weights, k_rrf=60, topn=50, depth=1000):
    #list_groups: по одной "колонке" списков на метод, в каждой — список на запрос
    return [rrf(lists, weights, k_rrf=k_rrf, topn=topn, depth=depth) for lists in zip(*list_groups)]
