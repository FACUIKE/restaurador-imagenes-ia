import base64
import io
import os
import time
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google import genai

load_dotenv()

# Servir estáticos desde ../frontend
app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

def clasificar_semantica(pil_img: Image.Image) -> tuple[str, float]:
    if not client:
        return "paisaje", 0.0
    t0 = time.perf_counter()
    prompt = "Clasifica estrictamente en: 'rostro', 'texto' o 'paisaje'. Responde solo con una de esas tres palabras en minúsculas."
    cat = "paisaje"
    for m in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
        try:
            res = client.models.generate_content(model=m, contents=[pil_img, prompt])
            texto = res.text.strip().lower()
            for k in ["rostro", "texto", "paisaje"]:
                if k in texto:
                    cat = k
                    break
            break
        except Exception:
            continue
    latencia = time.perf_counter() - t0
    return cat, latencia

def restaurar_adaptativa(img_np: np.ndarray, categoria: str) -> np.ndarray:
    h, w = img_np.shape[:2]
    nueva_w, nueva_h = int(w * 2), int(h * 2)

    if categoria == "texto":
        escalada = cv2.resize(img_np, (nueva_w, nueva_h), interpolation=cv2.INTER_CUBIC)
        denoised = cv2.bilateralFilter(escalada, d=9, sigmaColor=75, sigmaSpace=75)
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        return cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

    elif categoria == "rostro":
        pre_limpia = cv2.bilateralFilter(img_np, d=5, sigmaColor=16, sigmaSpace=16)
        escalada = cv2.resize(pre_limpia, (nueva_w, nueva_h), interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(escalada, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        a_clean = cv2.GaussianBlur(a, (7, 7), 2.0)
        b_clean = cv2.GaussianBlur(b, (7, 7), 2.0)
        l_base = cv2.bilateralFilter(l, d=7, sigmaColor=18, sigmaSpace=18).astype(np.float32)

        gx = cv2.Sobel(l_base, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l_base, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)
        mascara_contorno = np.clip((grad_mag - 28.0) / 38.0, 0.0, 1.0)
        mascara_contorno = cv2.GaussianBlur(mascara_contorno, (5, 5), 1.2)

        l_antialiasing = cv2.GaussianBlur(l_base, (3, 3), 0.85)
        l_base_suave = l_base * (1.0 - mascara_contorno) + l_antialiasing * mascara_contorno

        g_fina = cv2.GaussianBlur(l_base_suave, (0, 0), sigmaX=1.1)
        g_media = cv2.GaussianBlur(l_base_suave, (0, 0), sigmaX=2.6)
        det_fino = l_base_suave - g_fina
        det_medio = g_fina - g_media

        det_fino_c = np.sign(det_fino) * np.maximum(0.0, np.abs(det_fino) - 1.2)
        det_medio_c = np.sign(det_medio) * np.maximum(0.0, np.abs(det_medio) - 1.6)

        factor_proteccion = 1.0 - (0.75 * mascara_contorno)
        detalle_total = (det_fino_c * 1.55 + det_medio_c * 1.25) * factor_proteccion
        detalle_total = np.clip(detalle_total, -14.0, 14.0)

        l_estructurada = np.clip(l_base_suave + detalle_total, 0.0, 255.0)
        l_norm = l_estructurada / 255.0
        l_curva = l_norm - 0.12 * (np.sin(2.0 * np.pi * l_norm) / (2.0 * np.pi))
        l_final = np.clip(l_curva * 255.0, 0, 255).astype(np.uint8)

        lab_final = cv2.merge((l_final, a_clean, b_clean))
        return cv2.cvtColor(lab_final, cv2.COLOR_LAB2RGB)

    else:
        escalada = cv2.resize(img_np, (nueva_w, nueva_h), interpolation=cv2.INTER_LANCZOS4)
        lab = cv2.cvtColor(escalada, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_merge = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_merge, cv2.COLOR_LAB2RGB)

def calcular_ssim_simple(img1_g: np.ndarray, img2_g: np.ndarray) -> float:
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    i1, i2 = img1_g.astype(np.float64), img2_g.astype(np.float64)
    m1 = cv2.GaussianBlur(i1, (11, 11), 1.5)
    m2 = cv2.GaussianBlur(i2, (11, 11), 1.5)
    s1 = cv2.GaussianBlur(i1 ** 2, (11, 11), 1.5) - m1 ** 2
    s2 = cv2.GaussianBlur(i2 ** 2, (11, 11), 1.5) - m2 ** 2
    s12 = cv2.GaussianBlur(i1 * i2, (11, 11), 1.5) - (m1 * m2)
    ssim = ((2 * m1 * m2 + c1) * (2 * s12 + c2)) / ((m1 ** 2 + m2 ** 2 + c1) * (s1 + s2 + c2))
    return float(np.mean(ssim))

def auditar_alucinaciones(orig_rgb: np.ndarray, rest_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    h_r, w_r = rest_rgb.shape[:2]
    deg_g = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    res_g = cv2.cvtColor(rest_rgb, cv2.COLOR_RGB2GRAY)
    deg_scaled = cv2.resize(deg_g, (w_r, h_r), interpolation=cv2.INTER_CUBIC)

    g_deg = np.sqrt(cv2.Sobel(deg_scaled, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(deg_scaled, cv2.CV_64F, 0, 1)**2)
    g_res = np.sqrt(cv2.Sobel(res_g, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(res_g, cv2.CV_64F, 0, 1)**2)

    diff = np.abs(g_res - g_deg)
    anomalias = np.where(diff > np.percentile(diff, 90), diff, 0)

    anomalias_norm = cv2.normalize(anomalias, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(anomalias_norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rest_rgb, 0.65, heatmap_rgb, 0.35, 0)

    fid_score = max(0.0, min(1.0, 1.0 - ((np.mean(diff) / 255.0) * 2.0)))
    hallucination_ratio = (np.count_nonzero(anomalias) / anomalias.size) * 100.0
    ssim_val = calcular_ssim_simple(deg_scaled, res_g)

    nit_orig = float(cv2.Laplacian(deg_g, cv2.CV_64F).var())
    nit_rest = float(cv2.Laplacian(res_g, cv2.CV_64F).var())
    delta_nitidez = ((nit_rest - nit_orig) / max(nit_orig, 1.0)) * 100.0

    return overlay, {
        "fidelity_score": round(fid_score, 4),
        "hallucination_ratio": round(hallucination_ratio, 1),
        "ssim": round(ssim_val, 4),
        "delta_nitidez": round(delta_nitidez, 1),
        "out_res": f"{w_r} x {h_r} px"
    }

def np_to_base64(img_np: np.ndarray) -> str:
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")

@app.route("/api/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400

    archivo = request.files["image"]
    pil_raw = ImageOps.exif_transpose(Image.open(archivo.stream)).convert("RGB")
    img_np = np.array(pil_raw)

    t_inicio = time.perf_counter()
    cat, lat_gemini = clasificar_semantica(pil_raw)
    rest_np = restaurar_adaptativa(img_np, cat)
    heat_np, metricas = auditar_alucinaciones(img_np, rest_np)
    lat_total = time.perf_counter() - t_inicio

    return jsonify({
        "categoria": cat.upper(),
        "latencia": round(lat_total, 2),
        "metricas": metricas,
        "img_original": np_to_base64(img_np),
        "img_restaurada": np_to_base64(rest_np),
        "img_heatmap": np_to_base64(heat_np)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)