"""конфигурация, логирование и мелкие вспомогательные функции."""
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("avito")


def load_config(path) -> dict:
    # читает YAML-конфиг
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent
    cfg["_root"] = root
    cfg["paths"] = {k: _resolve(root, v) for k, v in cfg["paths"].items()}
    if "finetune" in cfg:
        cfg["finetune"]["output"] = _resolve(root, cfg["finetune"]["output"])
    for m in cfg["embeddings"]["models"]:
        # локальная (дообученная) модель задаётся через path, модель с Hugging Face — через name
        if "path" in m:
            m["path"] = _resolve(root, m["path"])
            m["name"] = str(m["path"])
    return cfg


def check_models_exist(cfg):
    # ловит ошибку если модели нет
    for m in cfg["embeddings"]["models"]:
        if "path" in m and not (m["path"] / "config.json").exists():
            raise FileNotFoundError(
                f"Модель {m['key']} не найдена в {m['path']}. Сначала запустите "
                f"scripts/finetune.py или используйте config_no_finetune.yaml")


def _resolve(root: Path, p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else root / p


def get_device(cfg) -> str:
    import torch
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


@contextmanager
def timer(name: str):
    t0 = time.time()
    log.info(f"{name}...")
    yield
    log.info(f"{name}: готово за {time.time() - t0:.0f} с")
