"""
поиск локации : этап N2-3

точная локация — слабый фильтр: совпадает лишь в 83% пар, а у 17% benchmark-запросов
id локации региональный, и своих объявлений в корпусе у него нет. поэтому своим (местным) считаем:
    объявления из локаций, куда уходит >= min_share выборов из этой локации поиска (учим по train);
    плюс всё в радиусе radius_km от центра локации поиска.
это не фильтр, а основа для бонуса: жёсткий фильтр терял бы ~5% ответов.
"""
import numpy as np
from sklearn.neighbors import BallTree

EARTH_R = 6371.0


class LocationHelper:
    def __init__(self, pairs, train_items, corpus, min_share=0.01, radius_km=50):
        tp = pairs[["search_location_id", "item_id"]].merge(
            train_items[["item_id", "item_location_id", "item_latitude", "item_longitude"]], on="item_id")
        counts = tp.groupby(["search_location_id", "item_location_id"]).size()
        share = counts / counts.groupby(level=0).transform("sum")
        kept = share[share >= min_share].reset_index()
        self.loc_map = kept.groupby("search_location_id")["item_location_id"].agg(set).to_dict()
        self.centers = tp.groupby("search_location_id")[["item_latitude", "item_longitude"]].median()
        self.radius = radius_km / EARTH_R
        self.corpus_loc = corpus["item_location_id"].to_numpy()
        # объявления без координат получают (0, 0) и просто не попадают ни в один радиус
        coords = np.radians(corpus[["item_latitude", "item_longitude"]].fillna(0).to_numpy())
        self.tree = BallTree(coords, metric="haversine")
        self._cache = {}

    def local_idx(self, search_loc):
        # номера строк корпуса, которые считаются местными для локации поиска
        if search_loc not in self._cache:
            allowed = {search_loc} | self.loc_map.get(search_loc, set())
            mask = np.isin(self.corpus_loc, list(allowed))
            if search_loc in self.centers.index:
                c = np.radians(self.centers.loc[search_loc].to_numpy())[None, :]
                mask[self.tree.query_radius(c, r=self.radius)[0]] = True
            self._cache[search_loc] = np.flatnonzero(mask)
        return self._cache[search_loc]
