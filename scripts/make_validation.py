"""
валидация : этап N2

    python scripts/make_validation.py --config config.yaml

этот этап откладывает ~5000 запросов из train так, чтобы они были похожи на benchmark, и убирает их из обучающей части

пишет в data/processed/: val_queries, train_part, val_corpus (.parquet).
нужна для scripts/evaluate.py и для дообучения (scripts/finetune.py).
"""
import argparse

import pandas as pd

from avito.utils import load_config, log, set_seed, timer
from avito.validation import build_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    P = cfg["paths"]["processed_dir"]

    pairs = pd.read_parquet(P / "train_pairs.parquet")
    queries = pd.read_parquet(P / "queries.parquet")
    items = pd.read_parquet(P / "items.parquet")
    train_items = pd.read_parquet(P / "train_items.parquet")

    with timer("Сборка валидации"):
        val, train_part, val_corpus, stats = build_validation(
            pairs, queries, items, train_items, n_val=cfg["validation"]["n_queries"], seed=cfg["seed"])
    for k, v in stats.items():
        log.info(f"   {k}: {v}")

    val.to_parquet(P / "val_queries.parquet", index=False)
    train_part.to_parquet(P / "train_part.parquet", index=False)
    val_corpus.to_parquet(P / "val_corpus.parquet", index=False)
    log.info(f"Сохранено в {P}")


if __name__ == "__main__":
    main()
