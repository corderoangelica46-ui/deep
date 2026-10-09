
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    # Carpeta principal del proyecto
    PROJECT = Path(__file__).resolve().parent

    # Rutas del modelo y del nuevo dataset
    modelo_path = PROJECT / "yolo12n.pt"
    data_path = PROJECT / "Geometrical shapes.v1i.yolov12" / "data.yaml"

    # Verificar que existan los archivos
    if not modelo_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {modelo_path}"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {data_path}"
        )

    print("--- VERIFICACIÓN DE HARDWARE ---")
    cuda_disponible = torch.cuda.is_available()
    print(f"¿CUDA está disponible?: {cuda_disponible}")

    device_val = 0 if cuda_disponible else "cpu"

    if cuda_disponible:
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("Se utilizará CPU. El entrenamiento puede tardar.")

    print("\nCargando YOLO12n...")
    model = YOLO(str(modelo_path))

    print("\nIniciando entrenamiento de prueba...")
    print(f"Dataset: {data_path}")

    inicio_tiempo = time.time()

    results = model.train(
        data=str(data_path),
        epochs=30,
        imgsz=416,
        batch=8,
        patience=3,
        device=device_val,
        workers=0,
        project=str(PROJECT / "entrenamiento"),
        name="prueba_dataset399",
        save=True,
        plots=True
    )

    tiempo_total = (time.time() - inicio_tiempo) / 60

    carpeta_resultados = (
        PROJECT / "entrenamiento" / "prueba_dataset399"
    )

    print("\n====================================")
    print("¡Entrenamiento finalizado!")
    print(f"Tiempo total: {tiempo_total:.2f} minutos")
    print(f"Resultados: {carpeta_resultados}")
    print(f"Mejores pesos: {carpeta_resultados / 'weights' / 'best.pt'}")
    print("====================================")


if __name__ == "__main__":
    main()
