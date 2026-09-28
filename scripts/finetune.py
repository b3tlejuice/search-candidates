"""
дообучение : этап N3

    python scripts/finetune.py --config config.yaml [--force]

функция потерь MultipleNegativesRankingLoss: для каждого запроса его объявление — позитив,
объявления других запросов в той же пачке — негативы. сэмплер NO_DUPLICATES не допускает
повторов в пачке (иначе модель штрафовала бы себя за правильный ответ).

нужна видеокарта: на T4 ~30 минут, на CPU — непрактично долго.
"""
import argparse
import math

import pandas as pd

from avito.embeddings import doc_texts
from avito.utils import get_device, load_config, log, set_seed, timer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true", help="переобучить, даже если модель уже есть")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ft, emb = cfg["finetune"], cfg["embeddings"]
    set_seed(cfg["seed"])
    P = cfg["paths"]["processed_dir"]
    out_dir = ft["output"]

    if (out_dir / "config.json").exists() and not args.force:
        log.info(f"Модель уже есть: {out_dir}. Для переобучения запустите с --force")
        return
    device = get_device(cfg)
    if device != "cuda":
        log.warning("Видеокарта не найдена: обучение на CPU займёт очень много времени")

    # обучающие пары
    source = "train_part.parquet" if ft["train_on"] == "train_part" else "train_pairs.parquet"
    pairs = pd.read_parquet(P / source, columns=["q_clean", "item_id"])
    pairs = (pairs.drop_duplicates().sample(frac=1, random_state=cfg["seed"])
             .groupby("q_clean").head(ft["max_per_text"]))
    if len(pairs) > ft["max_pairs"]:
        pairs = pairs.sample(ft["max_pairs"], random_state=cfg["seed"])
    pairs = pairs.reset_index(drop=True)

    mcfg = emb["models"][0]      # префиксы query:/passage: и формат текста — как при поиске
    docs = pd.read_parquet(P / "train_items.parquet", columns=["item_id", "title", "vid", "tip", "desc"])
    docs = docs[docs["item_id"].isin(set(pairs["item_id"]))]
    text_of = dict(zip(docs["item_id"], doc_texts(docs, mcfg["d_prefix"], emb["desc_words"])))
    anchors = (mcfg["q_prefix"] + pairs["q_clean"]).tolist()
    positives = pairs["item_id"].map(text_of).tolist()
    log.info(f"Обучающих пар: {len(pairs)} (источник: {source}), "
             f"запросов: {pairs['q_clean'].nunique()}, объявлений: {pairs['item_id'].nunique()}")

    # обучение
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    try:
        from sentence_transformers.sentence_transformer import losses
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError:     # версии sentence-transformers до 5.x
        from sentence_transformers import losses
        from sentence_transformers.training_args import BatchSamplers

    model = SentenceTransformer(ft["base_model"], device=device)
    model.max_seq_length = emb["max_len"]
    steps = math.ceil(len(pairs) / ft["batch_size"]) * ft["epochs"]
    ckpt_dir = out_dir.parent / f"{out_dir.name}-checkpoints"

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=ft["epochs"],
        per_device_train_batch_size=ft["batch_size"],
        learning_rate=ft["lr"],
        warmup_steps=int(ft["warmup_ratio"] * steps),
        fp16=(device == "cuda"),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=training_args,
        train_dataset=Dataset.from_dict({"anchor": anchors, "positive": positives}),
        loss=losses.MultipleNegativesRankingLoss(model),
    )
    ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    resume = str(ckpts[-1]) if ckpts and not args.force else None
    with timer(f"Обучение ({steps} шагов, продолжение с {resume or 'начала'})"):
        trainer.train(resume_from_checkpoint=resume)

    model.save(str(out_dir))
    log.info(f"Модель сохранена: {out_dir}")
    log.info("Если эмбеддинги этой модели уже кешировались раньше, удалите artifacts/embeddings/<key>")


if __name__ == "__main__":
    main()
