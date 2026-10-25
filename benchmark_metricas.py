import os
from pathlib import Path
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# Cargar modelo perceptual LPIPS preentrenado (AlexNet, modo evaluación en CPU)
print("Cargando modelo perceptual LPIPS preentrenado (AlexNet)...")
loss_fn_alex = lpips.LPIPS(net='alex', verbose=False)

def tensor_para_lpips(img_bgr: np.ndarray) -> torch.Tensor:
    """Convierte array BGR [0, 255] a tensor PyTorch RGB normalizado a [-1, 1]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1.0
    return tensor.unsqueeze(0)

def pipeline_adaptativo(img_bgr: np.ndarray) -> np.ndarray:
    """Aplica reducción de ruido bilateral + escalado + unsharp masking."""
    denoised = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    resultado = cv2.addWeighted(denoised, 1.4, gaussian, -0.4, 0)
    return resultado

if __name__ == "__main__":
    uploads = list(Path("storage/uploads").glob("*.*"))
    if not uploads:
        print("Coloca una imagen en storage/uploads/.")
        exit()

    img_ref_path = str(uploads[0])
    img_gt = cv2.imread(img_ref_path)
    h, w = img_gt.shape[:2]

    # Redimensionar a tamaño par para evitar desalineaciones
    if w % 2 != 0 or h % 2 != 0:
        img_gt = cv2.resize(img_gt, (w - (w % 2), h - (h % 2)))
        h, w = img_gt.shape[:2]

    # 1. Simular degradación: downsampling 50% con ruido gaussiano
    baja_res = cv2.resize(img_gt, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    ruido = np.random.normal(0, 15, baja_res.shape).astype(np.float32)
    baja_res_ruidosa = np.clip(baja_res.astype(np.float32) + ruido, 0, 255).astype(np.uint8)

    # 2. Restauración A: Interpolación Bicúbica tradicional
    rest_bicubica = cv2.resize(baja_res_ruidosa, (w, h), interpolation=cv2.INTER_CUBIC)

    # 3. Restauración B: Pipeline Adaptativo
    rest_intermedia = cv2.resize(baja_res_ruidosa, (w, h), interpolation=cv2.INTER_LANCZOS4)
    rest_adaptativa = pipeline_adaptativo(rest_intermedia)

    # Guardar muestras visuales
    out_dir = Path("storage/benchmark_metricas_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "1_ground_truth.png"), img_gt)
    cv2.imwrite(str(out_dir / "2_degradada_bicubica.png"), rest_bicubica)
    cv2.imwrite(str(out_dir / "3_restaurada_adaptativa.png"), rest_adaptativa)

    # 4. Cálculo de métricas
    # PSNR (dB): mayor es mejor
    psnr_bicubica = psnr(img_gt, rest_bicubica, data_range=255)
    psnr_adaptativa = psnr(img_gt, rest_adaptativa, data_range=255)

    # SSIM: mayor es mejor (escala 0 a 1)
    ssim_bicubica = ssim(img_gt, rest_bicubica, channel_axis=2, data_range=255)
    ssim_adaptativa = ssim(img_gt, rest_adaptativa, channel_axis=2, data_range=255)

    # LPIPS: menor es mejor (distancia perceptual)
    t_gt = tensor_para_lpips(img_gt)
    t_bicubica = tensor_para_lpips(rest_bicubica)
    t_adaptativa = tensor_para_lpips(rest_adaptativa)

    with torch.no_grad():
        lpips_bicubica = loss_fn_alex(t_gt, t_bicubica).item()
        lpips_adaptativa = loss_fn_alex(t_gt, t_adaptativa).item()

    # Consolidar resultados
    resultados = [
        {
            "Metodo": "Bicúbica Clásica",
            "PSNR (dB) ↑": round(psnr_bicubica, 2),
            "SSIM ↑": round(ssim_bicubica, 4),
            "LPIPS ↓": round(lpips_bicubica, 4)
        },
        {
            "Metodo": "Pipeline Adaptativo",
            "PSNR (dB) ↑": round(psnr_adaptativa, 2),
            "SSIM ↑": round(ssim_adaptativa, 4),
            "LPIPS ↓": round(lpips_adaptativa, 4)
        },
    ]

    df_res = pd.DataFrame(resultados)
    df_res.to_csv("storage/benchmark_metricas_resultados.csv", index=False)

    print("\n" + "=" * 65)
    print("        RESULTADOS DEL BENCHMARK DE CALIDAD DE IMAGEN")
    print("=" * 65)
    print(df_res.to_string(index=False))
    print("=" * 65)
    print(f"Reporte exportado a: storage/benchmark_metricas_resultados.csv\n")