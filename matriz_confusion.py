
from pathlib import Path
from ultralytics import YOLO


def main():
    proyecto = Path(__file__).resolve().parent

    # Modelo entrenado
    ruta_modelo = (
        proyecto / "entrenamiento" / "prueba_dataset399-3"
        / "weights" / "best.pt"
    )

    # Archivo YAML del dataset usado para entrenar
    ruta_dataset = proyecto / "geometrical shapes" / "data.yaml"

    if not ruta_modelo.exists():
        print("No se encontro el modelo:")
        print(ruta_modelo)
        return

    if not ruta_dataset.exists():
        print("No se encontro el archivo data.yaml:")
        print(ruta_dataset)
        print("Revisa la ruta del dataset.")
        return

    print("Cargando modelo...")
    modelo = YOLO(str(ruta_modelo))

    print("Clases:", modelo.names)
    print("Evaluando el modelo...")

    resultados = modelo.val(
        data=str(ruta_dataset),
        split="test",
        plots=True,
        conf=0.25
    )

    print("\nEvaluacion terminada.")
    print("Resultados guardados en:")
    print(resultados.save_dir)

    print("\nMatriz de confusion:")
    print(Path(resultados.save_dir) / "confusion_matrix.png")


if __name__ == "__main__":
    main()
