"""
оценка на валидации: recall@50/200/1000 для BM25, каждой модели эмбеддингов и объединения,
в разрезах по типам запросов.

    python scripts/evaluate.py --config config.yaml

результат и сохраняется в artifacts/metrics.json
"""
import argparse
import json

import pandas as pd

from avito.generator import CandidateGenerator
from avito.metrics import recall_per_query
from avito.utils import load_config, log


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    P = cfg["paths"]["processed_dir"]

    val = pd.read_parquet(P / "val_queries.parquet")
    val["relevant"] = val["relevant"].map(list)
    val_corpus = pd.read_parquet(P / "val_corpus.parquet")
    train_part = pd.read_parquet(P / "train_part.parquet", columns=["search_location_id", "item_id"])
    train_items = pd.read_parquet(P / "train_items.parquet",
                                  columns=["item_id", "item_location_id", "item_latitude", "item_longitude"])

    # для валидации соответствие локаций учится только на train_part: иначе утечка
    gen = CandidateGenerator(cfg, val_corpus, train_part, train_items)
    out = gen.predict(val, topn=cfg["fusion"]["depth"])

    seen_items = set(train_part["item_id"])
    groups = {
        "знакомый текст": val["seen"],
        "новый текст": ~val["seen"],
        "с фильтром вида услуги": val["search_vid"] != "",
        "без фильтра": val["search_vid"] == "",
        "ответ был в train_part": val["relevant"].map(lambda r: any(i in seen_items for i in r)),
        "ответа не было в train_part": val["relevant"].map(lambda r: not any(i in seen_items for i in r)),
    }

    rows = []
    for name, lists in out.items():
        preds = dict(zip(val["q_key"], lists))
        pq50 = recall_per_query(preds, val, 50)
        row = {"метод": name, "R@50": pq50.mean(),
               "R@200": recall_per_query(preds, val, 200).mean(),
               "R@1000": recall_per_query(preds, val, 1000).mean()}
        row.update({f"R@50 | {g}": pq50[m].mean() for g, m in groups.items()})
        rows.append(row)
    table = pd.DataFrame(rows).set_index("метод").round(4)

    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("\n", table[["R@50", "R@200", "R@1000"]], "\n")
        print(table.drop(columns=["R@200", "R@1000"]).T, "\n")
    log.info("Размеры групп: " + ", ".join(f"{g}: {int(m.sum())}" for g, m in groups.items()))

    metrics_path = cfg["paths"]["artifacts_dir"] / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(table.to_dict(orient="index"), ensure_ascii=False, indent=2),
                            encoding="utf-8")
    log.info(f"Метрики сохранены: {metrics_path}")


if __name__ == "__main__":
    main()
