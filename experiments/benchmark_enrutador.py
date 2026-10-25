import os
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

import torch
from torchvision import models, transforms
from google import genai

load_dotenv()

# ==========================================
# 1. Configuración de Modelos Preentrenados
# ==========================================

# A. Modelo Convolucional Preentrenado (MobileNetV3)
print("Cargando MobileNetV3 preentrenada (ImageNet)...")
weights = models.MobileNet_V3_Small_Weights.DEFAULT
cnn_model = models.mobilenet_v3_small(weights=weights)
cnn_model.eval()

# Transformación estándar requerida por ImageNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# B. Cliente Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ==========================================
# 2. Funciones de Clasificación
# ==========================================

def clasificar_con_cnn(ruta_imagen: str) -> tuple[str, float]:
    """Clasifica usando MobileNetV3 y mapea la categoría de ImageNet."""
    t0 = time.perf_counter()
    img = Image.open(ruta_imagen).convert("RGB")
    batch = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = cnn_model(batch)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        class_id = prob.argmax().item()
        score = prob[class_id].item()

    categoria_cruda = weights.meta["categories"][class_id].lower()
    latencia = time.perf_counter() - t0

    # Mapeo semántico simple de ImageNet a nuestras clases de estudio
    cat_normalizada = "paisaje"
    if any(k in categoria_cruda for k in ["person", "face", "suit", "wig", "groom", "jersey"]):
        cat_normalizada = "rostro"
    elif any(k in categoria_cruda for k in ["book", "menu", "envelope", "comic", "web_site", "packet"]):
        cat_normalizada = "texto"

    return cat_normalizada, latencia, f"{categoria_cruda} ({score:.2f})"


def clasificar_con_vlm(ruta_imagen: str) -> tuple[str, float]:
    """Clasifica usando Gemini zero-shot con fallback de modelo."""
    t0 = time.perf_counter()
    img = Image.open(ruta_imagen)
    prompt = "Clasifica estrictamente en: 'rostro', 'texto' o 'paisaje'. Responde solo esa palabra."
    
    cat = "desconocido"
    modelos = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    for m in modelos:
        try:
            res = client.models.generate_content(model=m, contents=[img, prompt])
            cat = res.text.strip().lower()
            break
        except Exception:
            continue

    latencia = time.perf_counter() - t0
    return cat, latencia


# ==========================================
# 3. Inyector de Degradación Sintética (Stress Testing)
# ==========================================

def degradar_imagen(ruta_origen: str, ruta_salida: str, nivel_ruido: float = 35.0):
    """Agrega desenfoque y ruido gaussiano para simular degradación severa."""
    img = cv2.imread(ruta_origen)
    if img is None:
        return
    
    # Desenfoque leve
    blurred = cv2.GaussianBlur(img, (7, 7), 2.0)
    
    # Ruido gaussiano
    h, w, c = blurred.shape
    gauss = np.random.normal(0, nivel_ruido, (h, w, c)).astype(np.float32)
    degradada = np.clip(blurred.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    
    cv2.imwrite(ruta_salida, degradada)


# ==========================================
# 4. Ejecución del Experimento
# ==========================================
if __name__ == "__main__":
    import pandas as pd

    test_dir = Path("storage/test_samples")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    uploads = list(Path("storage/uploads").glob("*.*"))
    if not uploads:
        print("Coloca al menos una imagen en storage/uploads/ para la prueba.")
        exit()

    registros = []

    for img_path in uploads:
        img_original = str(img_path)
        img_degradada = str(test_dir / f"deg_{img_path.name}")
        degradar_imagen(img_original, img_degradada, nivel_ruido=40.0)

        # Inferencia CNN
        cat_c_orig, lat_c_orig, _ = clasificar_con_cnn(img_original)
        cat_c_deg, lat_c_deg, _ = clasificar_con_cnn(img_degradada)

        # Inferencia Gemini
        cat_v_orig, lat_v_orig = clasificar_con_vlm(img_original)
        cat_v_deg, lat_v_deg = clasificar_con_vlm(img_degradada)

        registros.extend([
            {"imagen": img_path.name, "modelo": "MobileNetV3", "condicion": "limpia", "prediccion": cat_c_orig, "latencia_s": round(lat_c_orig, 4)},
            {"imagen": img_path.name, "modelo": "MobileNetV3", "condicion": "degradada", "prediccion": cat_c_deg, "latencia_s": round(lat_c_deg, 4)},
            {"imagen": img_path.name, "modelo": "Gemini-Flash", "condicion": "limpia", "prediccion": cat_v_orig, "latencia_s": round(lat_v_orig, 4)},
            {"imagen": img_path.name, "modelo": "Gemini-Flash", "condicion": "degradada", "prediccion": cat_v_deg, "latencia_s": round(lat_v_deg, 4)},
        ])

    df = pd.DataFrame(registros)
    csv_path = "storage/benchmark_enrutador_resultados.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n--- Resultados consolidados ---")
    print(df.to_string(index=False))
    print(f"\nReporte guardado para la tesis en: {csv_path}")