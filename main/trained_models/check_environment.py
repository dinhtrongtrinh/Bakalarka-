from importlib import util
from importlib.metadata import version
from pathlib import Path
import sys


# 1. Zkontrolujeme knihovny bez importu Unsloth a bez načítání modelu.
packages = {
    "unsloth": "unsloth",
    "torch": "torch",
    "torchvision": "torchvision",
    "sentence_transformers": "sentence-transformers",
    "datasets": "datasets",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "joblib": "joblib",
}
missing_packages = []
print("Použitý Python:", sys.executable)
for import_name, package_name in packages.items():
    if util.find_spec(import_name) is None:
        missing_packages.append(package_name)
    else:
        print(package_name, version(package_name))

if missing_packages:
    print("Chybějící balíčky:", ", ".join(missing_packages))
    print("Nejdřív spusť install_dependencies.sh ve složce trained_models.")
    raise SystemExit(1)

# 2. Nestačí jen číslo verze. Ověříme také načtení binárních operátorů.
# Tento malý test běží na CPU a nic netrénuje. Zachytí i chybu torchvision::nms.
import torch

try:
    import torchvision

    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    scores = torch.tensor([1.0])
    torchvision.ops.nms(boxes, scores, 0.5)
except (ImportError, RuntimeError, OSError) as error:
    print("Torchvision nemůže načíst své binární operátory:", error)
    print("Spusť install_dependencies.sh, který sjednotí CUDA sestavení s PyTorchem.")
    raise SystemExit(1) from None
print("Torchvision — skutečné sestavení:", torchvision.__version__)
print("Binární operátor torchvision::nms funguje.")

# 3. Ověříme dostupnost GPU a vypíšeme skutečně dostupnou paměť.
if not torch.cuda.is_available():
    raise SystemExit("CUDA není dostupná. Nejdřív oprav přístup PyTorche ke GPU.")
free_memory, total_memory = torch.cuda.mem_get_info()
print("GPU:", torch.cuda.get_device_name(0))
print("Celkem VRAM v GiB:", round(total_memory / (1024 ** 3), 2))
print("Volná VRAM v GiB:", round(free_memory / (1024 ** 3), 2))
print("Podpora BF16:", torch.cuda.is_bf16_supported())

# 4. Ověříme existenci původních splitů. Zatím nevytváříme žádné embeddingy.
current_dir = Path(__file__).resolve().parent
for split in ["train", "validation", "test"]:
    dataset_path = current_dir.parent / "split_datasets" / "random" / f"{split}.jsonl"
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
print("Základní kontrola prošla. Kompatibilita Unsloth se ověří až při načtení modelu.")
