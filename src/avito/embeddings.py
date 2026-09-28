"""
семантический поиск (эмбеддинг) : этап 4

эмбеддинги документов кешируются на диске по item_id: при повторном запуске пересчитываются
только новые объявления. Кеш привязан к `key` модели в конфиге — при смене модели или текста
документа меняйте key или удаляйте папку кеша
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .bm25 import SERVICE_CATEGORY
from .utils import log


def doc_texts(df, d_prefix="passage: ", desc_words=40):
    """Текст объявления для модели. Один и тот же для корпуса и для обучения модели."""
    desc = df["desc"].fillna("").str.split().str[:desc_words].str.join(" ")
    text = df["title"] + ". " + df["vid"] + ". " + df["tip"] + ". " + desc
    return (d_prefix + text.str.replace(r"(\.\s*){2,}", ". ", regex=True)).tolist()


class EmbeddingModel:
    def __init__(self, mcfg: dict, emb_cfg: dict, cache_root: Path, device: str):
        from sentence_transformers import SentenceTransformer

        self.key = mcfg["key"]
        self.cfg = mcfg
        self.emb_cfg = emb_cfg
        self.device = device
        self.cache_dir = Path(cache_root) / "embeddings" / self.key
        self.model = SentenceTransformer(mcfg["name"], device=device)
        self.model.max_seq_length = emb_cfg["max_len"]
        if device == "cuda":
            self.model.half()     # 16-битные числа: почти в 2х быстрее, качество почти не меняется

    def encode_queries(self, texts):
        import torch
        Q = self.model.encode([self.cfg["q_prefix"] + t for t in texts], batch_size=256,
                              normalize_embeddings=True, convert_to_tensor=True)
        return Q.to(torch.float16 if self.device == "cuda" else torch.float32)

    def doc_embeddings(self, corpus: pd.DataFrame, chunk: int = 20_000):
        # эмбеддинги документов корпуса (из кеша, недостающие досчитываются), torch-тензор на device
        import torch

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ids, embs = self._load_cache()
        known = set(ids)
        missing = corpus[~corpus["item_id"].astype(str).isin(known)]
        if len(missing):
            log.info(f"[{self.key}] кодируем {len(missing)} документов (в кеше {len(known)})")
            texts = doc_texts(missing, self.cfg["d_prefix"], self.emb_cfg["desc_words"])
            miss_ids = missing["item_id"].astype(str).to_numpy()
            n_parts = len(list(self.cache_dir.glob("part_*_emb.npy")))
            for start in range(0, len(texts), chunk):
                e = self.model.encode(texts[start:start + chunk], batch_size=self.emb_cfg["batch_size"],
                                      normalize_embeddings=True, convert_to_numpy=True,
                                      show_progress_bar=True).astype(np.float16)
                np.save(self.cache_dir / f"part_{n_parts:04d}_emb.npy", e)
                np.save(self.cache_dir / f"part_{n_parts:04d}_ids.npy", miss_ids[start:start + chunk])
                n_parts += 1
                log.info(f"[{self.key}] {min(start + chunk, len(texts))} / {len(texts)}")
            ids, embs = self._load_cache()

        pos = pd.Series(np.arange(len(ids)), index=ids)
        order = pos.reindex(corpus["item_id"].astype(str).to_numpy()).to_numpy().astype(int)
        D = torch.from_numpy(embs[order]).to(self.device)
        return D if self.device == "cuda" else D.float()

    def _load_cache(self):
        parts = sorted(self.cache_dir.glob("part_*_emb.npy"))
        if not parts:
            return np.array([], dtype=str), np.zeros((0, 0), np.float16)
        embs = np.concatenate([np.load(p) for p in parts])
        ids = np.concatenate([np.load(str(p).replace("_emb.npy", "_ids.npy"), allow_pickle=True)
                              for p in parts]).astype(str)
        return ids, embs


def emb_search(Q, D, queries, corpus, loc, gamma=0.3, delta=0.2, k=1000,
               penalize_non_service=True, non_service_penalty=1.0, batch=256):
    # бонусы прибавляются как сумма к косиносному сходству
    import torch

    device = D.device
    doc_ids = corpus["item_id"].to_numpy()
    corpus_vid = corpus["vid"].to_numpy()
    non_service = torch.from_numpy(corpus["item_category_id"].to_numpy() != SERVICE_CATEGORY).to(device)
    vid_masks, loc_cache, out = {}, {}, []
    locs = queries["search_location_id"].tolist()
    vids = queries["search_vid"].tolist()
    cats = queries["search_category"].tolist()
    with torch.no_grad():
        for start in range(0, len(queries), batch):
            S = (Q[start:start + batch] @ D.T).float()
            for i in range(S.shape[0]):
                j = start + i
                if gamma:
                    if locs[j] not in loc_cache:
                        loc_cache[locs[j]] = torch.from_numpy(loc.local_idx(locs[j])).to(device)
                    S[i, loc_cache[locs[j]]] += gamma
                if delta and vids[j]:
                    if vids[j] not in vid_masks:
                        vid_masks[vids[j]] = torch.from_numpy(corpus_vid == vids[j]).to(device)
                    S[i] += delta * vid_masks[vids[j]]
                if penalize_non_service and cats[j] == SERVICE_CATEGORY:
                    S[i, non_service] -= non_service_penalty
            top = torch.topk(S, min(k, S.shape[1]), dim=1).indices.cpu().numpy()
            out.extend(list(doc_ids[row]) for row in top)
    return out
