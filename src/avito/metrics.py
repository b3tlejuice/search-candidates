"""метрика recall@k и проверка формата ответа."""
import numpy as np
import pandas as pd


def recall_per_query(predictions: dict, val: pd.DataFrame, k: int = 50) -> pd.Series:
    # recall@k каждого запроса. predictions: {q_key: список item_id по убыванию уверенности}
    values = []
    for q_key, rel in zip(val["q_key"], val["relevant"]):
        rel = set(rel)
        values.append(len(set(predictions.get(q_key, [])[:k]) & rel) / len(rel))
    return pd.Series(values, index=val.index)


def evaluate(predictions: dict, val: pd.DataFrame, k: int = 50) -> dict:
    # recall@k: macro (среднее по запросам) и micro (все найденные / все правильные)
    hits, n_rel = [], []
    for q_key, rel in zip(val["q_key"], val["relevant"]):
        rel = set(rel)
        hits.append(len(set(predictions.get(q_key, [])[:k]) & rel))
        n_rel.append(len(rel))
    hits, n_rel = np.array(hits), np.array(n_rel)
    return {"recall_macro": float((hits / n_rel).mean()), "recall_micro": float(hits.sum() / n_rel.sum())}


def check_answer(answer: pd.DataFrame, queries: pd.DataFrame, item_ids, k: int = 50):
    # проверяет формат answer.csv. Бросает AssertionError с понятным текстом
    item_ids = set(map(str, item_ids))
    assert list(answer.columns) == ["query_id", "answer"], "колонки должны быть query_id, answer"
    assert len(answer) == len(queries), "число ответов не совпадает с числом запросов"
    assert answer["query_id"].tolist() == queries["query_id"].astype(str).tolist(), \
        "query_id не совпадают с запросами или идут в другом порядке"
    for qid, ans in zip(answer["query_id"], answer["answer"]):
        ids = ans.split(" ")
        assert len(ids) == k, f"{qid}: {len(ids)} объявлений вместо {k}"
        assert len(set(ids)) == k, f"{qid}: есть повторы"
        assert all(i in item_ids for i in ids), f"{qid}: есть объявления не из корпуса"
