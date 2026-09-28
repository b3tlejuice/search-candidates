"""токенизация и нормализация слов для BM25 (лемматизация через pymorphy3)."""
import re
from functools import lru_cache

TOKEN_PATTERN = r"(?u)\b\w+\b"
TOKEN_RE = re.compile(TOKEN_PATTERN)

# короткий список служебных слов (сгенерировано нейросетью)
STOP = set("""
и в во не на с со к ко у о об обо от до по за из изо для при про под над а но да или либо же ли бы
то это этот эта эти как что чтобы так также все всё весь его ее её их мы вы я ты он она они оно мне
нам вам нас вас меня свой свою свои который которая которые кто где когда там тут есть был была
""".split())

_morph = None


def _get_morph():
    global _morph
    if _morph is None:
        import pymorphy3
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


def _base(w):
    if w in STOP or (len(w) == 1 and not w.isdigit()):
        return None
    return w


@lru_cache(maxsize=None)
def norm_lemma(w):
    # словарная форма слова; None для стоп-слов, латиница и числа не меняются
    w = _base(w)
    if w is None:
        return None
    if not re.search("[а-я]", w):
        return w
    return _get_morph().parse(w)[0].normal_form.replace("ё", "е")
