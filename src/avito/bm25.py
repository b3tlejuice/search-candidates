"""
BM25 : этап N3
лексический поиск с бонусами за локацию и вид услуги
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .text import TOKEN_PATTERN, TOKEN_RE, norm_lemma

SERVICE_CATEGORY = 114


class BM25Index:
    # заранее считает вклад каждого слова в каждое описание (разреженная матрица слово X документ)
    # поиск = сумма строк матрицы для слов запроса

    def __init__(self, texts, normalizer=norm_lemma, k1=1.2, b=0.75):
        self.normalizer = normalizer
        cv = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True, dtype=np.float32)
        X_raw = cv.fit_transform(texts)
        raw_vocab = cv.get_feature_names_out()

        # склеиваем формы одного слова в один столбец нормальной формы
        self.term_id = {}
        rows, cols = [], []
        for i, w in enumerate(raw_vocab):
            t = normalizer(w)
            if t is None:
                continue
            rows.append(i)
            cols.append(self.term_id.setdefault(t, len(self.term_id)))
        M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(len(raw_vocab), len(self.term_id)))
        X = (X_raw @ M).tocsr()
        del X_raw

        n_docs = X.shape[0]
        dl = np.asarray(X.sum(axis=1)).ravel()
        avgdl = max(dl.mean(), 1e-9)
        df = np.bincount(X.indices, minlength=X.shape[1])
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = X.data
        doc_len = np.repeat(dl, np.diff(X.indptr))
        X.data = (idf[X.indices] * tf * (k1 + 1)
                  / (tf + k1 * (1 - b + b * doc_len / avgdl))).astype(np.float32)

        self.WT = X.T.tocsr()
        self.n_docs = n_docs
        self.vocab_size = len(self.term_id)

    def query_terms(self, q):
        terms = {self.normalizer(w) for w in TOKEN_RE.findall(q.lower())}
        return [self.term_id[t] for t in terms if t is not None and t in self.term_id]

    def score(self, q):
        s = np.zeros(self.n_docs, np.float32)
        WT = self.WT
        for t in self.query_terms(q):
            a, e = WT.indptr[t], WT.indptr[t + 1]
            s[WT.indices[a:e]] += WT.data[a:e]
        return s


def top_k(scores, k):
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def bm25_search(index, queries, corpus, loc, alpha=4.0, beta=1.0, k=1000,
                penalize_non_service=True, non_service_factor=0.1):
    # объявление без общих слов остаётся с нулём): локация *(1+alpha), вид услуги *(1+beta)
    # возвращает списки item_id в порядке строк queries
    doc_ids = corpus["item_id"].to_numpy()
    corpus_vid = corpus["vid"].to_numpy()
    non_service = corpus["item_category_id"].to_numpy() != SERVICE_CATEGORY
    vid_masks, out = {}, []
    for q, s_loc, vid, cat in zip(queries["q_clean"], queries["search_location_id"],
                                  queries["search_vid"], queries["search_category"]):
        f = index.score(q)
        if alpha:
            f[loc.local_idx(s_loc)] *= 1 + alpha
        if beta and vid:
            if vid not in vid_masks:
                vid_masks[vid] = corpus_vid == vid
            f[vid_masks[vid]] *= 1 + beta
        if penalize_non_service and cat == SERVICE_CATEGORY:
            f[non_service] *= non_service_factor
        out.append(list(doc_ids[top_k(f, k)]))
    return out
