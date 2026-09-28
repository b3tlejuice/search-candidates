"""
предобработка : этап N1

    python scripts/preprocess.py --config config.yaml

этот этап приводит числовые колонки к числам, удаляет дубли в train,
чистит тексты, разбирает параметры на вид услуги, тип услуги и адрес, собирает текст объявления

читает data/raw/{train,benchmark_queries,benchmark_items}.parquet,
пишет в data/processed/: items, train_items, train_pairs, queries (.parquet).
"""
import argparse

import pandas as pd

from avito.preprocess import (Q_COLS, dedup_train, process_items, process_queries, to_numeric)
from avito.utils import load_config, log, timer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw, out = cfg["paths"]["raw_dir"], cfg["paths"]["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    pp = cfg["preprocess"]

    with timer("Загрузка"):
        train = to_numeric(pd.read_parquet(raw / "train.parquet"))
        queries = to_numeric(pd.read_parquet(raw / "benchmark_queries.parquet"))
        items = to_numeric(pd.read_parquet(raw / "benchmark_items.parquet"))
    log.info(f"train: {train.shape}, queries: {queries.shape}, items: {items.shape}")

    before = len(train)
    train = dedup_train(train)
    log.info(f"Удалено дублей в train: {before - len(train)}")

    with timer("Объявления корпуса"):
        items_p = process_items(items, pp["desc_max_words"], pp["max_address_words"])
    with timer("Объявления train"):
        train_items_p = process_items(train.drop_duplicates("item_id"), pp["desc_max_words"],
                                      pp["max_address_words"])
    with timer("Запросы"):
        queries_p = process_queries(queries)
        pairs = process_queries(train[Q_COLS + ["item_id"]])

    log.info(f"Вид услуги найден у {(items_p['vid'] != '').mean():.3f} объявлений корпуса")
    log.info(f"Адресов, обрезанных страховкой: "
             f"{(items_p['address'].str.split().str.len() >= pp['max_address_words']).mean():.3f}")

    items_p.to_parquet(out / "items.parquet", index=False)
    train_items_p.to_parquet(out / "train_items.parquet", index=False)
    pairs.to_parquet(out / "train_pairs.parquet", index=False)
    queries_p.to_parquet(out / "queries.parquet", index=False)
    log.info(f"Сохранено в {out}")


if __name__ == "__main__":
    main()
