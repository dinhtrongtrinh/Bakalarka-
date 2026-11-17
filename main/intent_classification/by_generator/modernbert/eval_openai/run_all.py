from pathlib import Path
import subprocess
import sys

# Všechny tři skripty jsou ve stejné složce jako tento spouštěč.
current_dir = Path(__file__).resolve().parent
scripts = ["train_knn.py", "train_logistic_regression.py", "train_mlp.py"]

# Postupné spouštění: další model začne až po dokončení předchozího.
# Spouštěč nepoužívá import, protože každý soubor má vlastní trénovací kód.
for script in scripts:
    print(f"\nSpouštím {script}", flush=True)
    results_dir = current_dir / "results" / Path(script).stem
    results_dir.mkdir(parents=True, exist_ok=True)

    # sys.executable použije stejný Python a prostředí jako tento spouštěč.
    # -u zajistí průběžný výpis, abychom viděli postup trénování.
    # Výstup a případná varování vidíme v terminálu a současně v run.log.
    with (results_dir / "run.log").open("w", encoding="utf-8") as log_file:
        with subprocess.Popen(
            [sys.executable, "-u", str(current_dir / script)],
            cwd=current_dir,
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

    # Při chybě skončíme a její podrobnosti zůstanou uložené v run.log.
    if return_code != 0:
        raise SystemExit(f"{script} skončil chybou. Podrobnosti: {results_dir / 'run.log'}")

print("\nHotovo. Všechny tři klasifikátory dokončily běh.")
