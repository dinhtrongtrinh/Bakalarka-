import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_classification_experiments import write_csv, write_json


def main():
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Sloučení uložených statistik bez nového trénování.")
    parser.add_argument("--results-dir", type=Path, default=project_dir / "intent_classification" / "results_by_generator")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    result_paths = sorted(results_dir.glob("*/*/*/result.json"))
    if len(result_paths) != 18:
        raise ValueError(f"Nejprve dokonči všech 18 experimentů; zatím existuje {len(result_paths)} výsledků.")

    all_per_class = []
    all_trial_per_class = []
    grouped_results = defaultdict(list)
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        identity = {key: result[key] for key in ("embedding", "experiment", "classifier")}
        grouped_results[(result["embedding"], result["classifier"])].append(result)
        for split in ("train", "validation", "test"):
            for row in result[split]["per_class"]:
                all_per_class.append({**identity, "split": split, **row})
        for trial_path in sorted((result_path.parent / "trials").glob("trial_*.json")):
            trial = json.loads(trial_path.read_text(encoding="utf-8"))
            parameters = json.dumps(trial["parameters"], ensure_ascii=False)
            for split in ("train", "validation"):
                for row in trial[split]["per_class"]:
                    all_trial_per_class.append({
                        **identity, "trial_id": trial["trial_id"],
                        "parameters": parameters, "split": split, **row,
                    })

    aggregates = []
    for (embedding, classifier), results in sorted(grouped_results.items()):
        row = {"embedding": embedding, "classifier": classifier, "experiments": len(results)}
        for split in ("train", "validation", "test"):
            for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
                values = np.array([result[split][metric] for result in results])
                for statistic, value in (
                    ("mean", values.mean()), ("std", values.std(ddof=0)),
                    ("min", values.min()), ("max", values.max()),
                ):
                    row[f"{split}_{metric}_{statistic}"] = float(value)
        aggregates.append(row)

    write_csv(results_dir / "all_per_class.csv", all_per_class)
    write_csv(results_dir / "all_trial_per_class.csv", all_trial_per_class)
    write_csv(results_dir / "aggregate_metrics.csv", aggregates)
    write_json(results_dir / "statistics_schema.json", {
        "best_models.csv": "Jedna řádka pro každý z 18 modelů vybraných podle validačního macro F1.",
        "all_search_results.csv": "Všechny konfigurace: explicitní parametry, časy, iterace, varování a train/validation metriky.",
        "all_per_class.csv": "Precision, recall, F1, počty a TP/FP/FN/TN všech tříd pro vítěze na train/validation/test.",
        "all_trial_per_class.csv": "Stejné statistiky tříd pro každou vyzkoušenou konfiguraci, pouze train/validation.",
        "aggregate_metrics.csv": "Průměr, populační směrodatná odchylka, minimum a maximum přes tři vyhodnocované generátory; vždy samostatně pro embedding a klasifikátor.",
        "aggregation_caveat": "Jde o popisné statistiky tří souvisejících experimentů, nikoli intervaly spolehlivosti. OOS příklady jsou mezi variantami sdílené a modely mohou mít různá vybraná nastavení.",
        "class_accuracy": "Správnost uvnitř skutečné třídy: TP/(TP+FN), tedy recall. Není to one-vs-rest accuracy.",
        "one_vs_rest_accuracy": "(TP+TN)/N; při mnoha třídách může být vysoká i pro špatný klasifikátor.",
        "confusion_matrices": "Řádky = skutečné třídy, sloupce = předpovězené třídy. Pořadí je v poli labels v result.json.",
        "probabilities": "Předpovězené pravděpodobnosti všech tříd jsou v NPZ; pořadí sloupců je v poli classes, případně train_classes a validation_classes.",
        "predictions": "Každý pokus má trials/trial_NNN_predictions.npz s train/validation texty, štítky, předpověďmi a pravděpodobnostmi. Test je uložen pouze pro vítěze v test_predictions.npz a test_predictions.csv.",
        "model": "best_model.joblib obsahuje slovník s položkami model a trial_id; verze knihoven jsou v manifest.json.",
        "loss_curves": "Každý MLP pokus má loss_curve a warnings v trials/trial_NNN.json.",
        "timings": "Výpočty běží na CPU souběžně ve více procesech, každý s jedním vláknem BLAS. predict_ms_per_sentence je čas dávkové předpovědi dělený počtem vět, nikoli změřená latence samostatného dotazu. Časy hledání zahrnují více konfigurací.",
        "rows": {"all_per_class": len(all_per_class), "all_trial_per_class": len(all_trial_per_class), "aggregates": len(aggregates)},
    })
    print(f"Uloženo: {len(all_per_class)} řádků tříd vítězů, {len(all_trial_per_class)} řádků tříd všech pokusů a {len(aggregates)} souhrnů.")


if __name__ == "__main__":
    main()
