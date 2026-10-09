

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# ==========================================
# IDENTIFICAR EL COLOR DE UNA FIGURA
# ==========================================
def identificar_color(roi):
    if roi.size == 0:
        return "desconocido"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mascara = (s > 60) & (v > 40)

    if np.count_nonzero(mascara) < 20:
        brillo = float(np.mean(v))

        if brillo < 55:
            return "negro"
        elif brillo > 190:
            return "blanco"
        return "gris/indeterminado"

    tono = float(np.median(h[mascara]))
    saturacion = float(np.median(s[mascara]))
    brillo = float(np.median(v[mascara]))

    if brillo < 55:
        return "negro"
    elif saturacion < 65:
        return "blanco/gris"
    elif tono < 10 or tono >= 170:
        return "rojo"
    elif tono < 22:
        return "naranja"
    elif tono < 35:
        return "amarillo"
    elif tono < 85:
        return "verde"
    elif tono < 100:
        return "cian"
    elif tono < 135:
        return "azul"
    elif tono < 160:
        return "morado"
    else:
        return "rojo"


# ==========================================
# PROGRAMA PRINCIPAL
# ==========================================
def main():
    proyecto = Path(__file__).resolve().parent

    # Ruta del modelo entrenado
    ruta_modelo = (
        proyecto
        / "entrenamiento"
        / "prueba_dataset399-3"
        / "weights"
        / "best.pt"
    )

    # Carpetas para construir el dataset
    carpeta_dataset = proyecto / "mi_dataset"
    carpeta_imagenes = carpeta_dataset / "images" / "train"
    carpeta_etiquetas = carpeta_dataset / "labels" / "train"

    carpeta_imagenes.mkdir(parents=True, exist_ok=True)
    carpeta_etiquetas.mkdir(parents=True, exist_ok=True)

    # Verificar que exista el modelo
    if not ruta_modelo.exists():
        print("ERROR: No se encontró el modelo.")
        print("Ruta buscada:")
        print(ruta_modelo)
        print("\nBusca dónde quedó tu archivo best.pt y ajusta ruta_modelo.")
        return

    # Cargar modelo
    print("Cargando modelo YOLO...")
    modelo = YOLO(str(ruta_modelo))

    print("Modelo cargado correctamente.")
    print("Clases del modelo:", modelo.names)

    # Abrir cámara
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        print("ERROR: No se pudo abrir la cámara.")
        return

    print("\nCámara iniciada.")
    print("-----------------------------------")
    print("S = guardar imagen y etiquetas")
    print("Q o ESC = salir")
    print("-----------------------------------")
    print("Dataset:", carpeta_dataset)

    try:
        while True:
            correcto, frame = camara.read()

            if not correcto:
                print("No se pudo leer la cámara.")
                break

            # Ejecutar detección YOLO
            resultado = modelo.predict(
                source=frame,
                conf=0.25,
                verbose=False
            )[0]

            salida = frame.copy()
            cantidad = 0

            if resultado.boxes is not None:
                for caja in resultado.boxes:
                    x1, y1, x2, y2 = map(
                        int, caja.xyxy[0].tolist()
                    )

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Analizar color dentro de la detección
                    roi = frame[y1:y2, x1:x2]
                    color = identificar_color(roi)

                    # Obtener clase y confianza
                    clase_id = int(caja.cls[0])
                    forma = modelo.names[clase_id]
                    confianza = float(caja.conf[0])

                    etiqueta = (
                        f"{forma} - {color} {confianza:.2f}"
                    )

                    # Dibujar detección
                    cv2.rectangle(
                        salida,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        salida,
                        etiqueta,
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    cantidad += 1

            # Mostrar instrucciones
            cv2.putText(
                salida,
                f"Detecciones: {cantidad} | S: guardar | Q: salir",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.imshow("YOLO - Figuras y colores", salida)

            tecla = cv2.waitKey(1) & 0xFF

            # ==========================================
            # GUARDAR IMAGEN Y ETIQUETAS AL PRESIONAR S
            # ==========================================
            if tecla == ord("s"):
                nombre = datetime.now().strftime(
                    "figura_%Y%m%d_%H%M%S_%f"
                )

                ruta_imagen = carpeta_imagenes / f"{nombre}.jpg"
                ruta_etiqueta = carpeta_etiquetas / f"{nombre}.txt"

                # Guardar la imagen original sin recuadros
                imagen_guardada = cv2.imwrite(
                    str(ruta_imagen), frame
                )

                if not imagen_guardada:
                    print("ERROR: No se pudo guardar la imagen.")
                    continue

                # Crear las etiquetas a partir de las detecciones
                lineas = []
                alto_img, ancho_img = frame.shape[:2]

                if resultado.boxes is not None:
                    for caja in resultado.boxes:
                        clase_id = int(caja.cls[0])

                        x1, y1, x2, y2 = (
                            caja.xyxy[0].tolist()
                        )

                        # Formato YOLO normalizado:
                        # clase, centro_x, centro_y, ancho, alto
                        centro_x = ((x1 + x2) / 2) / ancho_img
                        centro_y = ((y1 + y2) / 2) / alto_img
                        ancho = (x2 - x1) / ancho_img
                        alto = (y2 - y1) / alto_img

                        linea = (
                            f"{clase_id} "
                            f"{centro_x:.6f} "
                            f"{centro_y:.6f} "
                            f"{ancho:.6f} "
                            f"{alto:.6f}"
                        )

                        lineas.append(linea)

                with open(
                    ruta_etiqueta,
                    "w",
                    encoding="utf-8"
                ) as archivo:
                    archivo.write("\n".join(lineas))

                print("\nImagen guardada:", ruta_imagen.name)
                print("Etiquetas guardadas:", len(lineas))

                if len(lineas) == 0:
                    print(
                        "AVISO: No se detectaron figuras. "
                        "La etiqueta está vacía; revisa la imagen "
                        "antes de usarla para entrenar."
                    )

            # Salir
            elif tecla == ord("q") or tecla == 27:
                break

    finally:
        camara.release()
        cv2.destroyAllWindows()
        print("\nPrograma finalizado.")


if __name__ == "__main__":
    main()
