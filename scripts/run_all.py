"""
запуск весь пайплайн одной командой: этапы запускаются по очереди, при ошибке всё останавливается

    python scripts/run_all.py                                     # все этапы
    python scripts/run_all.py --no-eval                           # без оценки на валидации

дообучение запускается, только если в конфиге есть секция finetune,
и само пропускается, если модель уже обучена
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from avito.utils import load_config

SCRIPTS = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-eval", action="store_true", help="не запускать evaluate.py")
    args = parser.parse_args()

    finetune = "finetune" in load_config(args.config)
    stages = ["preprocess"]
    if finetune or not args.no_eval:
        stages.append("make_validation")      # нужна и для оценки, и для честного дообучения
    if finetune:
        stages.append("finetune")
    if not args.no_eval:
        stages.append("evaluate")
    stages.append("predict")
    print("Этапы:", " → ".join(stages), flush=True)

    t_all = time.time()
    for stage in stages:
        print(f"\n===== {stage} =====", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, str(SCRIPTS / f"{stage}.py"), "--config", args.config])
        if result.returncode != 0:
            sys.exit(f"\nЭтап {stage} завершился с ошибкой, дальше не идём")
        print(f"===== {stage}: {time.time() - t0:.0f} с", flush=True)
    print(f"\nГотово за {(time.time() - t_all) / 60:.1f} мин")


if __name__ == "__main__":
    main()
