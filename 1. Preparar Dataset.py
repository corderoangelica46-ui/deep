
from pathlib import Path
import shutil
import random

# Carpeta del proyecto y dataset original
PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "archive"
OUTPUT = PROJECT / "dataset_yolo"

# Orden de las clases: debe coincidir con data.yaml
CLASSES = {
    "Triangle": 0,
    "Square": 1,
    "Circle": 2,
}

# Proporciones: 70 % entrenamiento, 20 % validación, 10 % prueba
SPLITS = {
    "train": 0.70,
    "val": 0.20,
    "test": 0.10,
}

random.seed(42)

# Busca imágenes en las carpetas de clase
images = []
for class_name, class_id in CLASSES.items():
    class_folder = SOURCE / "train" / class_name
    if not class_folder.exists():
        print(f"No se encontró la carpeta: {class_folder}")
        continue

    for image_path in class_folder.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in {
            ".png", ".jpg", ".jpeg"
        }:
            images.append((image_path, class_id))

if not images:
    raise SystemExit("No se encontraron imágenes en archive/train.")

random.shuffle(images)

# Crear carpetas de salida
for split in SPLITS:
    (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

# Repartir imágenes
n = len(images)
n_train = int(n * SPLITS["train"])
n_val = int(n * SPLITS["val"])

groups = {
    "train": images[:n_train],
    "val": images[n_train:n_train + n_val],
    "test": images[n_train + n_val:],
}

for split, items in groups.items():
    for index, (source_image, class_id) in enumerate(items):
        # Nombre único para evitar sobrescribir imágenes
        filename = f"{class_id}_{index}_{source_image.name}"
        destination_image = OUTPUT / "images" / split / filename
        destination_label = OUTPUT / "labels" / split / (
            Path(filename).stem + ".txt"
        )

        shutil.copy2(source_image, destination_image)

        # Suposición inicial: figura centrada que ocupa el 80 %.
        # Debe verificarse visualmente antes de entrenar.
        destination_label.write_text(
            f"{class_id} 0.5 0.5 0.8 0.8\n",
            encoding="utf-8"
        )

print(f"Imágenes procesadas: {n}")
for split, items in groups.items():
    print(f"{split}: {len(items)} imágenes")
print(f"Dataset creado en: {OUTPUT}")
print("IMPORTANTE: revisa los cuadros delimitadores antes de entrenar.")
