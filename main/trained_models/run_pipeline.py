from pathlib import Path
import argparse
import os
import subprocess
import sys


# 1. Jeden příkaz spustí celou pipeline. Při opravě pozdějšího kroku můžeme
# pokračovat od něj, například --start-at 3 přeskočí přípravu a fine-tuning.
parser = argparse.ArgumentParser(description="EmbeddingGemma: fine-tuning a porovnání klasifikace.")
parser.add_argument("--start-at", type=int, choices=range(1, 11), default=1)
arguments = parser.parse_args()

current_dir = Path(__file__).resolve().parent
experiment_dir = current_dir / "outputs" / "embeddinggemma_random_lora"
environment = os.environ.copy()

# Pro malé klasifikátory se vyhneme nadbytečnému soupeření mnoha CPU vláken.
environment.setdefault("OMP_NUM_THREADS", "1")
environment.setdefault("OPENBLAS_NUM_THREADS", "1")
environment.setdefault("MKL_NUM_THREADS", "1")
environment["MPLCONFIGDIR"] = str(current_dir / ".cache" / "matplotlib")

# 2. Kontrola skončí dřív, než by se vytvořily nové výsledky nebo načetl model.
subprocess.run(
    [sys.executable, "-u", str(current_dir / "check_environment.py")],
    env=environment,
    check=True,
)
if arguments.start_at == 1 and experiment_dir.exists() and any(experiment_dir.iterdir()):
    raise SystemExit(
        "Složka experimentu už existuje. Pro nový běh ji nejdřív přejmenuj. "
        "Pro pokračování použij --start-at a číslo nedokončeného kroku."
    )

# 3. Každý soubor má jeden hlavní úkol. Pořadí je podstatné.
scripts = [
    "prepare_training_data.py",                         # 1: textové trojice
    "train_embeddinggemma_random.py",                   # 2: Unsloth a LoRA
    "create_embeddings_random.py",                      # 3: obě varianty embeddingů
    "classifiers/baseline/train_knn.py",                 # 4
    "classifiers/baseline/train_logistic_regression.py",  # 5
    "classifiers/baseline/train_mlp.py",                 # 6
    "classifiers/finetuned/train_knn.py",                # 7
    "classifiers/finetuned/train_logistic_regression.py", # 8
    "classifiers/finetuned/train_mlp.py",                # 9
    "compare_results.py",                               # 10: souhrn statistik a graf
]
logs_dir = experiment_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

# 4. Každý krok běží v novém procesu, takže po jeho konci uvolní GPU paměť.
for step, script in enumerate(scripts, start=1):
    if step < arguments.start_at:
        continue
    print(f"\n[{step}/{len(scripts)}] Spouštím {script}", flush=True)
    log_path = logs_dir / f"step_{step:02d}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        with subprocess.Popen(
            [sys.executable, "-u", str(current_dir / script)],
            cwd=current_dir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()

    if return_code != 0:
        raise SystemExit(f"Krok {step} selhal. Podrobnosti: {log_path}")

print("\nPipeline dokončena. Porovnání:", experiment_dir / "comparison")
