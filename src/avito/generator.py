""" итоговый пайплайн : BM25 + модели эмбеддингов + RRF (все параметры берутся из конфига) """
import pandas as pd

from .bm25 import BM25Index, bm25_search
from .embeddings import EmbeddingModel, emb_search
from .fusion import fuse_all
from .location import LocationHelper
from .utils import check_models_exist, get_device, log, timer


class CandidateGenerator:
    """
    corpus - где ищем (обработанные объявления);
    pairs - пары «запрос -> объявление», по которым учится соответствие локаций;
    train_items - локации и координаты объявлений из train.
    """

    def __init__(self, cfg: dict, corpus: pd.DataFrame, pairs: pd.DataFrame, train_items: pd.DataFrame):
        check_models_exist(cfg)
        self.cfg = cfg
        self.corpus = corpus.reset_index(drop=True)
        self.device = get_device(cfg)
        log.info(f"Устройство: {self.device}")

        with timer("Соответствие локаций"):
            self.loc = LocationHelper(pairs, train_items, self.corpus, **cfg["location"])
        with timer("Индекс BM25"):
            self.index = BM25Index(self.corpus["item_text"], k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"])

        self.models, self.doc_emb = [], {}
        for mcfg in cfg["embeddings"]["models"]:
            with timer(f"Модель {mcfg['key']}: эмбеддинги корпуса"):
                m = EmbeddingModel(mcfg, cfg["embeddings"], cfg["paths"]["artifacts_dir"], self.device)
                self.doc_emb[m.key] = m.doc_embeddings(self.corpus)
                self.models.append(m)

    def predict(self, queries: pd.DataFrame, topn: int = 50) -> dict:
        # возвращает {"bm25": ..., <key модели>: ..., "fused": ...}: списки item_id на каждый запрос
        cfg, ns = self.cfg, self.cfg["non_service"]
        depth = cfg["fusion"]["depth"]
        out = {}
        with timer(f"BM25-поиск ({len(queries)} запросов)"):
            out["bm25"] = bm25_search(self.index, queries, self.corpus, self.loc,
                                      alpha=cfg["bm25"]["alpha_location"], beta=cfg["bm25"]["beta_vid"],
                                      k=depth, penalize_non_service=ns["penalize"],
                                      non_service_factor=ns["bm25_factor"])
        for m in self.models:
            with timer(f"Поиск {m.key}"):
                Q = m.encode_queries(queries["q_clean"].tolist())
                out[m.key] = emb_search(Q, self.doc_emb[m.key], queries, self.corpus, self.loc,
                                        gamma=m.cfg["gamma"], delta=m.cfg["delta"], k=depth,
                                        penalize_non_service=ns["penalize"],
                                        non_service_penalty=ns["emb_penalty"])
        groups = [out["bm25"]] + [out[m.key] for m in self.models]
        weights = [cfg["fusion"]["bm25_weight"]] + [m.cfg["weight"] for m in self.models]
        with timer("Объединение (RRF)"):
            out["fused"] = fuse_all(groups, weights, k_rrf=cfg["fusion"]["rrf_k"], topn=topn, depth=depth)
        return out
