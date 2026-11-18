# 1. Unsloth importujeme před torch a sentence_transformers, aby mohl
# zapnout své úpravy trénování. Unsloth zatím musí být do prostředí doinstalován.
from unsloth import FastSentenceTransformer

from pathlib import Path
from importlib.metadata import version
import json
import hashlib
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import TripletLoss, TripletDistanceMetric
from transformers.trainer_utils import get_last_checkpoint
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 2. Vstupem jsou připravené TEXTOVÉ trojice, nikoli staré NPZ embeddingy.
current_dir = Path(__file__).resolve().parent
data_dir = current_dir / "data" / "random"
experiment_dir = current_dir / "outputs" / "embeddinggemma_random_lora"
output_dir = experiment_dir / "training"
last_checkpoint = None
if output_dir.exists() and any(output_dir.iterdir()):
    # Po přerušení lze pokračovat od posledního uloženého konce epochy.
    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint is None:
        raise FileExistsError("Neúplný běh bez checkpointu. Před opakováním přejmenuj složku experimentu.")
    print("Pokračuji z checkpointu:", last_checkpoint)

with (data_dir / "source_summary.json").open(encoding="utf-8") as file:
    source_summary = json.load(file)
for split in ["train", "validation", "test"]:
    source_path = current_dir.parent / "split_datasets" / "random" / f"{split}.jsonl"
    actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual_hash != source_summary[split]["sha256"]:
        raise ValueError("Původní data se změnila. Znovu spusť prepare_training_data.py.")

dataset = load_dataset("json", data_files={
    "train": str(data_dir / "train_triplets.jsonl"),
    "validation": str(data_dir / "validation_triplets.jsonl"),
})
# Pořadí textových sloupců určuje role předávané TripletLoss.
train_dataset = dataset["train"].select_columns(["anchor", "positive", "negative"])
validation_dataset = dataset["validation"].select_columns(["anchor", "positive", "negative"])
print("Počet trojic:", len(train_dataset), len(validation_dataset))

# 3. Ověříme GPU. EmbeddingGemma používá BF16, případně FP32, ne FP16.
if not torch.cuda.is_available():
    raise RuntimeError("Tento trénink vyžaduje dostupnou NVIDIA GPU přes CUDA.")
use_bf16 = torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float32
print("GPU:", torch.cuda.get_device_name(0), "Přesnost:", dtype)

# Stejný základní model jako u původních embeddingů. Kratší kontext šetří VRAM;
# věty delší než 256 tokenů včetně instrukce se při trénování oříznou.
model = FastSentenceTransformer.from_pretrained(
    model_name="google/embeddinggemma-300m",
    max_seq_length=256,
    dtype=dtype,
    load_in_4bit=False,
    full_finetuning=False,
    use_exact_model_name=True,
)

# 4. Přidáme LoRA. Učit se budou malé přidané matice, základní váhy zůstanou zmrazené.
model = FastSentenceTransformer.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    max_seq_length=256,
    random_state=42,
    task_type="FEATURE_EXTRACTION",
)

# Zmrazíme také případné projekční vrstvy mimo samotný transformer.
# Tím se v tomto experimentu opravdu učí pouze LoRA matice.
for parameter_name, parameter in model.named_parameters():
    if "lora_" in parameter_name:
        parameter.requires_grad = True
    else:
        parameter.requires_grad = False
model.max_seq_length = 256

# Počet parametrů vypíšeme i uložíme. requires_grad=True znamená,
# že se pro tento parametr počítají gradienty a může se při učení měnit.
total_parameters = 0
trainable_parameters = 0
for parameter in model.parameters():
    total_parameters += parameter.numel()
    if parameter.requires_grad:
        trainable_parameters += parameter.numel()
print("Všechny parametry:", total_parameters)
print("Trénovatelné parametry:", trainable_parameters)
if trainable_parameters == 0 or trainable_parameters == total_parameters:
    raise RuntimeError("Počet trénovatelných parametrů neodpovídá očekávanému LoRA tréninku.")

# 5. Loss je číslo popisující chybu jedné dávky. Menší je lepší.
# COSINE odpovídá porovnávání směrů vektorů v naší klasifikační pipeline.
loss = TripletLoss(model, distance_metric=TripletDistanceMetric.COSINE, triplet_margin=0.2)

# 6. Nastavení prvního malého experimentu, nikoli hledání nejlepších parametrů.
prompt = "task: classification | query: "
args = SentenceTransformerTrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=3,                 # Tři průchody trénovacími trojicemi.
    per_device_train_batch_size=2,       # Jedna dávka = dvě trojice, tedy šest vět.
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,       # Váhy se upraví po osmi malých dávkách.
    learning_rate=2e-5,                 # Velikost kroku aktualizace vah.
    warmup_ratio=0.1,
    bf16=use_bf16,
    fp16=False,
    prompts={"anchor": prompt, "positive": prompt, "negative": prompt},
    eval_strategy="epoch",             # Validation pouze měří chybu, neučí váhy.
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Nejlepší checkpoint podle validačního loss.
    greater_is_better=False,            # Nejde o accuracy ani macro F1 klasifikátoru.
    save_total_limit=2,
    logging_steps=10,
    report_to="none",                  # Bez posílání výsledků do externí služby.
    seed=42,
)

# Uložíme nastavení, abychom později věděli, jak model vznikl.
experiment_dir.mkdir(parents=True, exist_ok=True)
model_info = {
    "base_model": "google/embeddinggemma-300m",
    "base_revision": model[0].auto_model.config._commit_hash,
    "prompt": prompt,
    "max_seq_length": 256,
    "lora_rank": 16,
    "lora_alpha": 32,
    "triplet_margin": 0.2,
    "train_triplets": len(train_dataset),
    "validation_triplets": len(validation_dataset),
    "total_parameters": total_parameters,
    "trainable_parameters": trainable_parameters,
    "gpu": torch.cuda.get_device_name(0),
    "dtype": str(dtype),
    "torch_version": version("torch"),
    "unsloth_version": version("unsloth"),
    "sentence_transformers_version": version("sentence-transformers"),
    "source_splits": source_summary,
}
with (experiment_dir / "model_info.json").open("w", encoding="utf-8") as file:
    json.dump(model_info, file, ensure_ascii=False, indent=2)

torch.cuda.reset_peak_memory_stats()

# 7. Trainer propojí model, data, chybu a nastavení.
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    loss=loss,
)

# TADY začíná učení: dopředný průchod -> loss -> backward -> optimizer.step.
# Tyto operace PyTorche a opakování po dávkách provádí Trainer za nás.
result = trainer.train(resume_from_checkpoint=last_checkpoint)

# 8. Trainer na konci vrátil nejlepší checkpoint podle validační chyby.
# Adaptér = pouze přidané malé matice; bez základního modelu nestačí.
model.save_pretrained(str(experiment_dir / "best_adapter"))
trainer.save_metrics("train", result.metrics)
trainer.save_metrics("eval", trainer.evaluate())
trainer.save_state()

# Celý průběh obsahuje trénovací loss, validační loss, epochy a kroky.
with (output_dir / "training_history.json").open("w", encoding="utf-8") as file:
    json.dump(trainer.state.log_history, file, ensure_ascii=False, indent=2)

# 9. Nakreslíme vývoj chyby. Validační loss se měří po každé epoše,
# trénovací loss je průměr za několik kroků, proto křivky nemají stejné body.
train_epochs = []
train_losses = []
validation_epochs = []
validation_losses = []
for record in trainer.state.log_history:
    if "loss" in record:
        train_epochs.append(record["epoch"])
        train_losses.append(record["loss"])
    if "eval_loss" in record:
        validation_epochs.append(record["epoch"])
        validation_losses.append(record["eval_loss"])

figure, axis = plt.subplots(figsize=(10, 6))
axis.plot(train_epochs, train_losses, label="Train loss")
axis.plot(validation_epochs, validation_losses, marker="o", label="Validation loss")
axis.set_xlabel("Epocha")
axis.set_ylabel("Triplet loss — nižší je lepší")
axis.set_title("Fine-tuning EmbeddingGemma s LoRA")
axis.legend()
axis.grid(alpha=0.3)
figure.tight_layout()
figure.savefig(output_dir / "loss.png", dpi=160)
plt.close(figure)

# 10. Sloučíme základní váhy s nejlepším LoRA adaptérem.
# Tento samostatný embedding model načte v dalším skriptu SentenceTransformer.
# Sloučení může dočasně potřebovat další RAM/VRAM.
model.save_pretrained_merged(str(experiment_dir / "merged_model"))

training_summary = {
    "best_checkpoint": trainer.state.best_model_checkpoint,
    "best_validation_loss": trainer.state.best_metric,
    "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    "peak_reserved_gpu_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
}
with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
    json.dump(training_summary, file, ensure_ascii=False, indent=2)
print("Nejlepší adaptér:", experiment_dir / "best_adapter")
print("Samostatný embedding model:", experiment_dir / "merged_model")
