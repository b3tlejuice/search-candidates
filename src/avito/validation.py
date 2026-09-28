"""
валидация : этап N2

валидация пытается повторить свойства настоящего benchmark, с помощью неё мы проверим как натренировалась наша модель
"""
import numpy as np
import pandas as pd

from .preprocess import Q_COLS


def build_sessions(pairs: pd.DataFrame) -> pd.DataFrame:
    """Сессия = уникальная комбинация признаков запроса + список всех выбранных объявлений."""
    pairs = pairs.copy()
    pairs["q_key"] = pairs.groupby(Q_COLS, dropna=False).ngroup()
    relevant = pairs.groupby("q_key")["item_id"].agg(lambda s: sorted(set(s)))
    sessions = (pairs.drop_duplicates("q_key").drop(columns="item_id")
                .set_index("q_key").assign(relevant=relevant).reset_index())
    return pairs, sessions


def build_validation(pairs, queries, items, train_items, n_val=5000, seed=42):
    """Возвращает (val, train_part, val_corpus)."""
    pairs, sessions = build_sessions(pairs)
    rng = np.random.default_rng(seed)

    target_vid = (queries["search_vid"] != "").mean()
    target_seen = queries["q_clean"].isin(set(pairs["q_clean"])).mean()

    one_per_text = (sessions.assign(_rnd=rng.random(len(sessions)))
                    .sort_values("_rnd").drop_duplicates("q_clean").drop(columns="_rnd"))

    with_vid = one_per_text[one_per_text["search_vid"] != ""]
    without_vid = one_per_text[one_per_text["search_vid"] == ""]
    n_vid = int(round(n_val * target_vid))
    val = pd.concat([
        with_vid.sample(min(n_vid, len(with_vid)), random_state=seed),
        without_vid.sample(min(n_val - n_vid, len(without_vid)), random_state=seed),
    ]).reset_index(drop=True)

    sessions_per_text = sessions["q_clean"].value_counts()
    val["seen"] = val["q_clean"].map(sessions_per_text) > 1
    natural_seen = val["seen"].mean()
    if natural_seen > target_seen:
        n_make_unseen = int(round((natural_seen - target_seen) * len(val)))
        idx = val[val["seen"]].sample(n_make_unseen, random_state=seed).index
        val.loc[idx, "seen"] = False

    unseen_texts = set(val.loc[~val["seen"], "q_clean"])
    drop = pairs["q_key"].isin(set(val["q_key"])) | pairs["q_clean"].isin(unseen_texts)
    train_part = pairs[~drop].reset_index(drop=True)

    val_items = {i for rel in val["relevant"] for i in rel}
    extra = train_items[train_items["item_id"].isin(val_items)
                        & ~train_items["item_id"].isin(set(items["item_id"]))]
    val_corpus = pd.concat([items, extra], ignore_index=True)

    stats = {
        "запросов": len(val),
        "с фильтром вида услуги (цель)": round(target_vid, 3),
        "с фильтром вида услуги": round((val["search_vid"] != "").mean(), 3),
        "знакомых текстов (цель)": round(target_seen, 3),
        "знакомых текстов": round(val["q_clean"].isin(set(train_part["q_clean"])).mean(), 3),
        "обучающая часть, строк": len(train_part),
        "корпус валидации": len(val_corpus),
    }
    return val, train_part, val_corpus, stats
