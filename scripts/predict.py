"""
ответ для benchmark: 50 кандидатов на каждый запрос -> запись в answer.csv

    python scripts/predict.py --config config.yaml

соответствие локаций учится на всём train (прятать для валидации теперь не нужно)
"""
import argparse

import numpy as np
import pandas as pd

from avito.generator import CandidateGenerator
from avito.metrics import check_answer
from avito.utils import load_config, log


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    P = cfg["paths"]["processed_dir"]

    queries = pd.read_parquet(P / "queries.parquet")
    items = pd.read_parquet(P / "items.parquet")
    train_pairs = pd.read_parquet(P / "train_pairs.parquet", columns=["search_location_id", "item_id"])
    train_items = pd.read_parquet(P / "train_items.parquet",
                                  columns=["item_id", "item_location_id", "item_latitude", "item_longitude"])
    log.info(f"Запросов: {len(queries)}, корпус: {len(items)}")

    gen = CandidateGenerator(cfg, items, train_pairs, train_items)
    predictions = gen.predict(queries, topn=50)["fused"]

    answer = pd.DataFrame({
        "query_id": queries["query_id"].astype(str).tolist(),
        "answer": [" ".join(map(str, top50)) for top50 in predictions],
    })
    check_answer(answer, queries, items["item_id"], k=50)
    log.info("Формат ответа проверен ✓")

    out = cfg["paths"]["output"]
    answer.to_csv(out, index=False)
    log.info(f"Сохранено: {out}")

    titles = items.set_index("item_id")["title"]
    for i in np.random.default_rng(cfg["seed"]).choice(len(queries), 3, replace=False):
        log.info(f"«{queries['search_query'].iloc[i]}» → " +
                 " | ".join(titles[x] for x in predictions[i][:3]))


if __name__ == "__main__":
    main()
