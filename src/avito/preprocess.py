"""
предобработка : этап N1

ключевые решения:
    все дубли в train удаляются, чтобы не засорять данные;

    адрес, цены и график работы вырезаются из текста объявления: они создают много шума;

    лемматизация не делается: BM25 делает её сам, а эмбеддинг работает с естественным текст
"""
import re

import pandas as pd

Q_COLS = ["search_query", "search_location_id", "search_is_delivery_search",
          "search_infm_params_text", "search_category"]

NUM_COLS = ["item_latitude", "item_longitude", "item_price", "item_rating",
            "item_rating_reviews_count", "item_is_phone_hidden", "item_is_message_forbidden",
            "search_is_delivery_search"]

ITEM_COLS = ["item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
             "item_category_id", "item_microcat_id", "item_location_id",
             "item_latitude", "item_longitude", "item_price", "item_rating",
             "item_rating_reviews_count", "item_is_phone_hidden", "item_is_message_forbidden"]

# названия полей в item_infm_params_text / search_infm_params_text
# собраны по примерам и по частотам фраз с заглавной буквы
PARAM_KEYS = [
    "Вид услуги", "Тип услуги", "Тип услуги автосервиса", "Название услуги",
    "Место оказания услуг", "Место сделки",
    "Тип стоимости", "Начальная цена", "Стоимость",
    "График работы от", "График работы до",
    "Работаете с юрлицами и ИП", "Работа по договору", "Опыт работы", "Гарантия",
    "Дополнительно", "Марка", "Модель",
    "Вид товара", "Тип товара", "Вид объявления", "Вид запчасти",
    "Онлайн-запись", "Рейтинг пользователя",
    "Груз", "Тип техники", "Состояние", "Качество", "Направление", # "Доставка", # обязательно убрать доставку, тк она пересекается с "Доставка еды и продуктов" как одним из Видов Услуги
    "Вид торгового оборудования", "Чем вы занимаетесь", "Расстояние доставки",
    "Куда выезжаете", "Как вы работаете",
]
_KEY_RE = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, sorted(PARAM_KEYS, key=len, reverse=True)))
                     + r")(?!\w)")
ADDRESS_KEYS = {"Место оказания услуг", "Место сделки"}
DROP_KEYS = ADDRESS_KEYS | {"Тип стоимости", "Начальная цена", "Стоимость",
                            "График работы от", "График работы до"}


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # в parquet часть числовых колонок хранилась как object, поэтому нужно преобразовать
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def clean_text(s) -> str:
    # чистит текст, сохраняя регистр
    if not isinstance(s, str):
        return ""
    s = s.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    s = re.sub(r"[\n\r\t]", " ", s)
    s = s.replace("ё", "е").replace("Ё", "Е")
    s = re.sub(r"[^\w\s.,:;!?()/%+№-]", " ", s)
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_params(raw, max_address_words: int = 12) -> dict:
    # разбирает строку параметров на вид услуги, тип услуги, адрес и "полезный" остаток
    text = clean_text(raw)
    parts = _KEY_RE.split(text)
    head = parts[0].strip()
    pairs = [(k, v.strip()) for k, v in zip(parts[1::2], parts[2::2])]

    vid = [v for k, v in pairs if k == "Вид услуги" and v]
    tip = [v for k, v in pairs if k.startswith("Тип услуги") and v]
    addr = [v for k, v in pairs if k in ADDRESS_KEYS and v]
    addr = " ".join(addr[0].split()[:max_address_words]) if addr else ""
    useful = [head] + [f"{k} {v}" for k, v in pairs if k not in DROP_KEYS]
    return {
        "vid": vid[0].lower() if vid else "",
        "tip": " ; ".join(tip).lower(),
        "address": addr.lower(),
        "params_clean": re.sub(r"\s+", " ", " ".join(useful)).strip().lower(),
    }


def process_items(df: pd.DataFrame, desc_max_words: int = 100, max_address_words: int = 12) -> pd.DataFrame:
    out = df[ITEM_COLS].copy()
    out["title"] = out["item_title_raw"].map(clean_text).str.lower()
    out["desc"] = (out["item_description_raw"].map(clean_text).str.lower()
                   .str.split().str[:desc_max_words].str.join(" "))
    parsed = pd.DataFrame([parse_params(x, max_address_words) for x in out["item_infm_params_text"]],
                          index=out.index)
    out = pd.concat([out, parsed], axis=1)
    out["item_text"] = (out["title"] + ". " + out["title"] + ". " + out["vid"] + ". " + out["tip"]
                        + ". " + out["params_clean"] + ". " + out["desc"])
    out["item_text"] = out["item_text"].str.replace(r"(\.\s*){2,}", ". ", regex=True).str.strip(". ")
    return out.drop(columns=["item_title_raw", "item_description_raw", "item_infm_params_text"])


def process_queries(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["q_clean"] = out["search_query"].map(clean_text).str.lower()
    parsed = pd.DataFrame([parse_params(x) for x in out["search_infm_params_text"]], index=out.index)
    out["search_vid"] = parsed["vid"]
    out["search_tip"] = parsed["tip"]
    out["search_params_clean"] = parsed["params_clean"]
    return out


def dedup_train(train: pd.DataFrame) -> pd.DataFrame:
    return train.drop_duplicates(subset=Q_COLS + ["item_id"]).reset_index(drop=True)
