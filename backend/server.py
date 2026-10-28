import os
import time
import base64
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI(title="VisionRestore API")

# Permitir conexiones desde el Frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

def img_to_base64(img_np):
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_img(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(b64_str), np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Modelos Pydantic para ingesta JSON estricta (Evita errores Form/Multipart)
class ImageRequest(BaseModel):
    image_b64: str

class ProcessRequest(BaseModel):
    image_b64: str
    categoria: str
    sub_modo: str
    intensidad: float = None

# ==========================================
# ENDPOINT 1: Clasificación Semántica
# ==========================================
@app.post("/api/classify")
async def classify_image(req: ImageRequest):
    img_rgb = base64_to_img(req.image_b64)
    
    from PIL import Image
    pil_img = Image.fromarray(img_rgb)
    
    t0 = time.perf_counter()
    cat = "paisaje"
    
    if client:
        prompt = "Clasifica estrictamente en: 'rostro', 'texto' o 'paisaje'. Responde solo con una de esas tres palabras."
        try:
            res = client.models.generate_content(model="gemini-3.5-flash-lite", contents=[pil_img, prompt])
            texto = res.text.strip().lower()
            if any(k in texto for k in ["rostro", "texto", "paisaje"]):
                cat = next(k for k in ["rostro", "texto", "paisaje"] if k in texto)
        except Exception:
            pass
            
    lat = time.perf_counter() - t0
    
    return {
        "categoria": cat,
        "latencia": lat,
        "original_b64": img_to_base64(img_rgb)
    }

# ==========================================
# ENDPOINT 2: Procesamiento Matemático PDI
# ==========================================
@app.post("/api/process")
async def process_image(req: ProcessRequest):
    img_np = base64_to_img(req.image_b64)
    
    h, w = img_np.shape[:2]
    nw, nh = int(w * 2), int(h * 2)
    rest_np = img_np.copy()
    
    # --- MATEMÁTICA EXACTA PDI CON CONTROL DE INTENSIDAD ---
    if req.categoria == "texto":
        escalada = cv2.resize(img_np, (nw, nh), interpolation=cv2.INTER_CUBIC)
        
        if req.sub_modo == "Iluminación Irregular (Sauvola)":
            c_val = int(req.intensidad) if req.intensidad else 10
            gris = cv2.cvtColor(escalada, cv2.COLOR_RGB2GRAY)
            binarizada = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, c_val)
            rest_np = cv2.cvtColor(binarizada, cv2.COLOR_GRAY2RGB)
            
        elif req.sub_modo == "Trazos Desgastados (Top-Hat)":
            alpha_val = req.intensidad if req.intensidad else 2.5
            gris = cv2.cvtColor(escalada, cv2.COLOR_RGB2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
            blackhat = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, kernel)
            blackhat_fuerte = cv2.convertScaleAbs(blackhat, alpha=alpha_val, beta=0)
            resultado = cv2.subtract(gris, blackhat_fuerte)
            resultado = cv2.normalize(resultado, None, 0, 255, cv2.NORM_MINMAX)
            rest_np = cv2.cvtColor(resultado, cv2.COLOR_GRAY2RGB)
            
        elif req.sub_modo == "Eliminar Muaré de Pantalla (FFT 2D)":
            radio = int(req.intensidad) if req.intensidad else 40
            gris = cv2.cvtColor(escalada, cv2.COLOR_RGB2GRAY)
            f = np.fft.fft2(gris)
            fshift = np.fft.fftshift(f)
            filas, cols = gris.shape
            crow, ccol = filas // 2 , cols // 2
            mascara = np.ones((filas, cols), np.uint8)
            mascara[crow-3:crow+3, :] = 0 
            mascara[:, ccol-3:ccol+3] = 0
            mascara[crow-radio:crow+radio, ccol-radio:ccol+radio] = 1
            fshift = fshift * mascara
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.abs(np.fft.ifft2(f_ishift))
            img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, binarizada = cv2.threshold(img_back, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            rest_np = cv2.cvtColor(binarizada, cv2.COLOR_GRAY2RGB)
            
    elif req.categoria == "rostro":
        ycbcr = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        y_up = cv2.resize(y, (nw, nh), interpolation=cv2.INTER_CUBIC)
        cr_up = cv2.resize(cr, (nw, nh), interpolation=cv2.INTER_CUBIC)
        cb_up = cv2.resize(cb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        escalada = cv2.cvtColor(cv2.merge((y_up, cr_up, cb_up)), cv2.COLOR_YCrCb2RGB)

        if req.sub_modo == "Super-Resolución (Solo YCbCr)":
            clip_val = req.intensidad if req.intensidad else 1.5
            y_up_clahe = cv2.createCLAHE(clipLimit=clip_val, tileGridSize=(8, 8)).apply(y_up)
            rest_np = cv2.cvtColor(cv2.merge((y_up_clahe, cr_up, cb_up)), cv2.COLOR_YCrCb2RGB)
            
        elif req.sub_modo == "Trepidación (Deconvolución / Sharpening)":
            fuerza = req.intensidad if req.intensidad else 0.7
            kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            resultado = cv2.filter2D(escalada, -1, kernel_sharp)
            rest_np = cv2.addWeighted(escalada, 1.0 - fuerza, resultado, fuerza, 0)
            
        elif req.sub_modo == "Ruido ISO Alto (No-Local Means)":
            h_val = req.intensidad if req.intensidad else 10.0
            rest_np = cv2.fastNlMeansDenoisingColored(escalada, None, h_val, h_val, 7, 21)
            
        elif req.sub_modo == "Iluminación Nocturna (MSRCR / Retinex)":
            alpha_val = req.intensidad if req.intensidad else 1.2
            img_float = np.float32(escalada) + 1.0
            retinex = np.zeros_like(img_float)
            for sigma in [15, 80, 250]:
                blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
                retinex += np.log10(img_float) - np.log10(blur)
            retinex = (retinex / 3.0) * 255.0
            resultado = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
            resultado = cv2.convertScaleAbs(resultado, alpha=alpha_val, beta=10)
            rest_np = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
            
        elif req.sub_modo == "Suavizado de Piel (Filtro Guiado)":
            sigma_s_val = int(req.intensidad) if req.intensidad else 60
            rest_np = cv2.edgePreservingFilter(escalada, flags=cv2.RECURS_FILTER, sigma_s=sigma_s_val, sigma_r=0.2)

    # --- Auditoría de Gradientes y Métricas ---
    h_r, w_r = rest_np.shape[:2]
    deg_scaled = cv2.resize(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), (w_r, h_r), interpolation=cv2.INTER_CUBIC)
    res_g = cv2.cvtColor(rest_np, cv2.COLOR_RGB2GRAY)
    
    g_deg = np.sqrt(cv2.Sobel(deg_scaled, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(deg_scaled, cv2.CV_64F, 0, 1)**2)
    g_res = np.sqrt(cv2.Sobel(res_g, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(res_g, cv2.CV_64F, 0, 1)**2)
    
    diff = np.abs(g_res - g_deg)
    anomalias = np.where(diff > np.percentile(diff, 90), diff, 0)
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(cv2.normalize(anomalias, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    
    n_orig = float(cv2.Laplacian(deg_scaled, cv2.CV_64F).var())
    n_rest = float(cv2.Laplacian(res_g, cv2.CV_64F).var())
    
    return {
        "restaurada_b64": img_to_base64(rest_np),
        "heatmap_b64": img_to_base64(cv2.addWeighted(rest_np, 0.65, heatmap_rgb, 0.35, 0)),
        "metricas": {
            "fid": round(max(0.0, min(1.0, 1.0 - ((np.mean(diff) / 255.0) * 2.0))), 3),
            "hal": round((np.count_nonzero(anomalias) / anomalias.size) * 100.0, 1),
            "delta": round(((n_rest - n_orig) / max(n_orig, 1.0)) * 100.0, 1),
            "res": f"{w_r} x {h_r} px"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)