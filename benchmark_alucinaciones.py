import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

def compute_gradient_magnitude(img_gray: np.ndarray) -> np.ndarray:
    """Calcula la magnitud del gradiente espacial usando operadores Sobel."""
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return magnitude

def analizar_alucinaciones(ruta_degradada: str, ruta_restaurada: str, output_heatmap_path: str) -> dict:
    # 1. Cargar imágenes en escala de grises
    deg = cv2.imread(ruta_degradada, cv2.IMREAD_GRAYSCALE)
    res = cv2.imread(ruta_restaurada, cv2.IMREAD_GRAYSCALE)

    if deg is None or res is None:
        raise FileNotFoundError("No se pudo cargar alguna de las imágenes para el análisis.")

    h, w = res.shape[:2]
    deg_scaled = cv2.resize(deg, (w, h), interpolation=cv2.INTER_CUBIC)

    # 2. Magnitud de gradientes de alta frecuencia
    grad_deg = compute_gradient_magnitude(deg_scaled)
    grad_res = compute_gradient_magnitude(res)

    # 3. Diferencia residual entre gradientes
    diff = np.abs(grad_res - grad_deg)

    # Aislar anomalías en el percentil superior (posibles texturas artificiales)
    umbral = np.percentile(diff, 90)
    anomalias = np.where(diff > umbral, diff, 0)

    # 4. Generar mapa de calor visual (ColorMap JET)
    anomalias_norm = cv2.normalize(anomalias, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomalias_norm, cv2.COLORMAP_JET)

    # Superponer el mapa sobre la imagen restaurada
    res_bgr = cv2.imread(ruta_restaurada)
    overlay = cv2.addWeighted(res_bgr, 0.65, heatmap, 0.35, 0)
    cv2.imwrite(output_heatmap_path, overlay)

    # 5. Cálculo del Fidelity Score
    error_medio = np.mean(diff) / 255.0
    fidelity_score = max(0.0, min(1.0, 1.0 - (error_medio * 2.0)))

    # Porcentaje de píxeles marcados como alta discrepancia de gradiente
    hallucination_ratio = (np.count_nonzero(anomalias) / anomalias.size) * 100.0

    return {
        "Fidelity_Score": round(fidelity_score, 4),
        "Hallucination_Ratio_%": round(hallucination_ratio, 2),
        "Mean_Residual_Gradient": round(float(np.mean(diff)), 4)
    }

if __name__ == "__main__":
    out_dir = Path("storage/benchmark_metricas_outputs")
    rutas_deg = out_dir / "2_degradada_bicubica.png"
    rutas_res = out_dir / "3_restaurada_adaptativa.png"

    if not rutas_deg.exists() or not rutas_res.exists():
        print("Ejecuta primero benchmark_metricas.py para generar las muestras base.")
        exit()

    heatmap_out = str(out_dir / "4_mapa_alucinaciones.png")
    metricas = analizar_alucinaciones(str(rutas_deg), str(rutas_res), heatmap_out)

    df_res = pd.DataFrame([metricas])
    df_res.to_csv("storage/benchmark_alucinaciones_resultados.csv", index=False)

    print("\n" + "=" * 65)
    print("      DETECCIÓN DE ALUCINACIONES Y CONTROL DE FIDELIDAD")
    print("=" * 65)
    print(df_res.to_string(index=False))
    print("=" * 65)
    print(f"Mapa visual guardado en: {heatmap_out}")
    print("Reporte exportado a: storage/benchmark_alucinaciones_resultados.csv\n")