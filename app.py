import os
import io
import time
from pathlib import Path
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image, ImageOps

import streamlit as st
from streamlit_image_comparison import image_comparison
from google import genai

load_dotenv()

# ==========================================
# Configuración General y Estilos Dark Glass
# ==========================================
st.set_page_config(
    page_title="VisionRestore AI - Auditoría & Super-Resolución",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Tarjetas de telemetría superior */
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    .metric-label {
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.70rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Tarjetas laterales de la columna izquierda */
    .side-metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 8px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: border-color 0.2s ease;
    }
    .side-metric-card:hover {
        border-color: #374151;
    }
    .side-metric-title {
        font-size: 0.70rem;
        text-transform: uppercase;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 0.05em;
    }
    .side-metric-num {
        font-size: 1.35rem;
        font-weight: 800;
        margin-top: 2px;
    }
    
    /* Botón rojo de acción */
    .stButton > button {
        width: 100%;
        background-color: #ef4444;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 12px 20px;
        border-radius: 10px;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #dc2626;
        color: #ffffff;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
    }
    
    /* Botón de descarga */
    .stDownloadButton > button {
        width: 100%;
        background-color: #1f2937;
        color: #f8fafc;
        border: 1px solid #374151;
        font-weight: 600;
        padding: 10px 16px;
        border-radius: 10px;
    }
    .stDownloadButton > button:hover {
        background-color: #374151;
        border-color: #4b5563;
        color: #ffffff;
    }

    .viewer-header {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Inicialización del Cliente Gemini
# ==========================================
api_key = os.getenv("GEMINI_API_KEY")
client = None
if api_key:
    client = genai.Client(api_key=api_key)

# ==========================================
# Funciones de Visión, Restauración y Auditoría
# ==========================================
def clasificar_con_gemini(pil_image: Image.Image) -> tuple[str, float]:
    """Clasifica la imagen en: rostro, texto o paisaje."""
    if not client:
        return "paisaje", 0.0
    
    t0 = time.perf_counter()
    prompt = "Clasifica estrictamente en: 'rostro', 'texto' o 'paisaje'. Responde solo con una de esas tres palabras en minúsculas."
    modelos = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    
    cat = "paisaje"
    for m in modelos:
        try:
            res = client.models.generate_content(model=m, contents=[pil_image, prompt])
            cat_limpia = res.text.strip().lower()
            if any(k in cat_limpia for k in ["rostro", "texto", "paisaje"]):
                for k in ["rostro", "texto", "paisaje"]:
                    if k in cat_limpia:
                        cat = k
                        break
                break
        except Exception:
            continue
            
    latencia = time.perf_counter() - t0
    return cat, latencia


def restaurar_imagen_adaptativa(img_np: np.ndarray, categoria: str) -> np.ndarray:
    """Super-resolución 2x con Anti-Aliasing de contornos y separación de frecuencias."""
    h, w = img_np.shape[:2]
    nueva_w, nueva_h = int(w * 2), int(h * 2)

    if categoria == "texto":
        escalada = cv2.resize(img_np, (nueva_w, nueva_h), interpolation=cv2.INTER_CUBIC)
        denoised = cv2.bilateralFilter(escalada, d=9, sigmaColor=75, sigmaSpace=75)
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        resultado = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        return resultado

    elif categoria == "rostro":
        # 1. DE-BLOCKING PRE-ESCALADO (En resolución nativa)
        # Disuelve los bloques cuadrados del JPEG antes de agrandarlos
        pre_limpia = cv2.bilateralFilter(img_np, d=5, sigmaColor=16, sigmaSpace=16)

        # 2. Escalado 2x con INTER_CUBIC (evita el pixelado de rebote/ringing de Lanczos en bordes)
        escalada = cv2.resize(pre_limpia, (nueva_w, nueva_h), interpolation=cv2.INTER_CUBIC)

        # 3. Conversión al espacio perceptual CIE LAB
        lab = cv2.cvtColor(escalada, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # 4. Limpieza profunda de ruido cromático (elimina manchas violetas/azules en bordes nocturnos)
        a_clean = cv2.GaussianBlur(a, (7, 7), 2.0)
        b_clean = cv2.GaussianBlur(b, (7, 7), 2.0)

        # 5. Base de luminancia suavizada preservando bordes
        l_base = cv2.bilateralFilter(l, d=7, sigmaColor=18, sigmaSpace=18).astype(np.float32)

        # 6. DETECCIÓN DE CONTORNOS EXTERIORES PARA ANTI-ALIASING
        # Calculamos la fuerza del borde con operadores de Scharr/Sobel
        gx = cv2.Sobel(l_base, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l_base, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)

        # Máscara de contorno fuerte (bordes de alto contraste como perfil de cara contra fondo)
        # Donde grad_mag > 30, empieza a proteger el borde para no pixelarlo
        mascara_contorno = np.clip((grad_mag - 28.0) / 38.0, 0.0, 1.0)
        mascara_contorno = cv2.GaussianBlur(mascara_contorno, (5, 5), 1.2)

        # Suavizado sub-píxel exclusivo para alisar el efecto escalera en los contornos
        l_antialiasing = cv2.GaussianBlur(l_base, (3, 3), 0.85)
        l_base_suave = l_base * (1.0 - mascara_contorno) + l_antialiasing * mascara_contorno

        # 7. Descomposición Multiescala de Facciones Internas
        g_fina = cv2.GaussianBlur(l_base_suave, (0, 0), sigmaX=1.1)
        g_media = cv2.GaussianBlur(l_base_suave, (0, 0), sigmaX=2.6)

        detalle_fino = l_base_suave - g_fina
        detalle_medio = g_fina - g_media

        # Soft-Coring para no levantar ruido en zonas lisas de la piel
        detalle_fino_cored = np.sign(detalle_fino) * np.maximum(0.0, np.abs(detalle_fino) - 1.2)
        detalle_medio_cored = np.sign(detalle_medio) * np.maximum(0.0, np.abs(detalle_medio) - 1.6)

        # Atenuamos el enfoque sobre el contorno exterior (1.0 - mascara_contorno)
        # Así enfocamos ojos, cejas y pelo, pero NO aserramos el borde de la cara
        factor_proteccion = 1.0 - (0.75 * mascara_contorno)
        detalle_total = (detalle_fino_cored * 1.55 + detalle_medio_cored * 1.25) * factor_proteccion
        detalle_total = np.clip(detalle_total, -14.0, 14.0)

        l_estructurada = np.clip(l_base_suave + detalle_total, 0.0, 255.0)

        # 8. Curva Tonal S-Sinusoidal suave para dar volumen a las sombras
        l_norm = l_estructurada / 255.0
        alpha_contraste = 0.12
        l_curva = l_norm - alpha_contraste * (np.sin(2.0 * np.pi * l_norm) / (2.0 * np.pi))
        l_final = np.clip(l_curva * 255.0, 0, 255).astype(np.uint8)

        # 9. Recomposición final
        lab_final = cv2.merge((l_final, a_clean, b_clean))
        resultado = cv2.cvtColor(lab_final, cv2.COLOR_LAB2RGB)
        return resultado

    else:  # paisaje / general
        escalada = cv2.resize(img_np, (nueva_w, nueva_h), interpolation=cv2.INTER_LANCZOS4)
        lab = cv2.cvtColor(escalada, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_merge = cv2.merge((l_clahe, a, b))
        resultado = cv2.cvtColor(lab_merge, cv2.COLOR_LAB2RGB)
        return resultado


def calcular_nitidez_laplaciana(img_gray: np.ndarray) -> float:
    """Calcula la varianza del Laplaciano como proxy de nitidez."""
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def calcular_ssim_simple(img1_gray: np.ndarray, img2_gray: np.ndarray) -> float:
    """Cálculo de SSIM global determinista."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    img1 = img1_gray.astype(np.float64)
    img2 = img2_gray.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.mean(ssim_map))


def auditar_alucinaciones(img_orig_rgb: np.ndarray, img_rest_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Auditoría de residuos de gradiente y métricas forenses."""
    h_rest, w_rest = img_rest_rgb.shape[:2]
    
    deg_gray = cv2.cvtColor(img_orig_rgb, cv2.COLOR_RGB2GRAY)
    res_gray = cv2.cvtColor(img_rest_rgb, cv2.COLOR_RGB2GRAY)
    
    deg_scaled = cv2.resize(deg_gray, (w_rest, h_rest), interpolation=cv2.INTER_CUBIC)
    
    grad_deg_x = cv2.Sobel(deg_scaled, cv2.CV_64F, 1, 0, ksize=3)
    grad_deg_y = cv2.Sobel(deg_scaled, cv2.CV_64F, 0, 1, ksize=3)
    grad_deg = np.sqrt(grad_deg_x**2 + grad_deg_y**2)
    
    grad_res_x = cv2.Sobel(res_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_res_y = cv2.Sobel(res_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_res = np.sqrt(grad_res_x**2 + grad_res_y**2)
    
    diff = np.abs(grad_res - grad_deg)
    
    umbral = np.percentile(diff, 90)
    anomalias = np.where(diff > umbral, diff, 0)
    
    anomalias_norm = cv2.normalize(anomalias, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_jet = cv2.applyColorMap(anomalias_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img_rest_rgb, 0.65, heatmap_rgb, 0.35, 0)
    
    error_medio = np.mean(diff) / 255.0
    fidelity_score = max(0.0, min(1.0, 1.0 - (error_medio * 2.0)))
    hallucination_ratio = (np.count_nonzero(anomalias) / anomalias.size) * 100.0
    ssim_val = calcular_ssim_simple(deg_scaled, res_gray)
    
    # Delta de nitidez
    nitidez_orig = calcular_nitidez_laplaciana(deg_gray)
    nitidez_rest = calcular_nitidez_laplaciana(res_gray)
    delta_nitidez = ((nitidez_rest - nitidez_orig) / max(nitidez_orig, 1.0)) * 100.0
    
    metricas = {
        "fidelity_score": fidelity_score,
        "hallucination_ratio": hallucination_ratio,
        "ssim": ssim_val,
        "delta_nitidez": delta_nitidez,
        "out_res": f"{w_rest} x {h_rest} px"
    }
    
    return overlay, metricas


# ==========================================
# Inicialización de Estado de Sesión
# ==========================================
if "categoria" not in st.session_state:
    st.session_state.categoria = "PENDIENTE"
if "latencia" not in st.session_state:
    st.session_state.latencia = 0.0
if "fidelity" not in st.session_state:
    st.session_state.fidelity = 0.0
if "hallucination_rate" not in st.session_state:
    st.session_state.hallucination_rate = 0.0
if "ssim_score" not in st.session_state:
    st.session_state.ssim_score = 0.0
if "delta_nitidez" not in st.session_state:
    st.session_state.delta_nitidez = 0.0
if "out_res" not in st.session_state:
    st.session_state.out_res = "0 x 0 px"
if "img_restaurada" not in st.session_state:
    st.session_state.img_restaurada = None
if "img_heatmap" not in st.session_state:
    st.session_state.img_heatmap = None
if "img_original" not in st.session_state:
    st.session_state.img_original = None

# ==========================================
# Telemetría Superior (Fila de 4 Métricas Globales)
# ==========================================
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🏷️ Clasificación Visual</div>
        <div class="metric-value" style="color: #60a5fa;">{st.session_state.categoria.upper()}</div>
        <div class="metric-sub">Enrutador semántico multimodal</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⏱️ Tiempo de Inferencia</div>
        <div class="metric-value" style="color: #34d399;">{st.session_state.latencia:.2f}s</div>
        <div class="metric-sub">Latencia total de pipeline</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    color_fid = "#38bdf8" if st.session_state.fidelity >= 0.7 else "#fbbf24"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🛡️ Fidelity Score</div>
        <div class="metric-value" style="color: {color_fid};">{st.session_state.fidelity:.3f}</div>
        <div class="metric-sub">Confiabilidad estructural [0-1]</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⚠️ Tasa de Alucinación</div>
        <div class="metric-value" style="color: #fbbf24;">{st.session_state.hallucination_rate:.1f}%</div>
        <div class="metric-sub">Discrepancia de alta frecuencia</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# Cuerpo Principal: Columna Izquierda + Visores Paralelos
# ==========================================
col_izq, col_centro, col_der = st.columns([1.0, 1.35, 1.35], gap="small")

# --- COLUMNA 1: INGESTA, CONTROL Y CARACTERÍSTICAS DEL PROCESAMIENTO ---
with col_izq:
    st.markdown("""
    <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 10px;">
        🕹️ Ingesta y Control
    </div>
    """, unsafe_allow_html=True)

    archivo_subido = st.file_uploader(
        "Seleccionar imagen a restaurar",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed"
    )

    if archivo_subido is not None:
        pil_raw = Image.open(archivo_subido)
        # Corrección automática de rotación EXIF de móviles
        pil_raw = ImageOps.exif_transpose(pil_raw).convert("RGB")
        w_orig, h_orig = pil_raw.size

        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 10px 14px; margin: 10px 0; font-size: 0.8rem; color: #94a3b8;">
            📄 <b>Archivo:</b> {archivo_subido.name}<br>
            📐 <b>Resolución Original:</b> {w_orig} x {h_orig} px
        </div>
        """, unsafe_allow_html=True)

        if st.button("⚡ Ejecutar Restauración + Auditoría"):
            with st.spinner("Procesando en memoria RAM..."):
                img_np = np.array(pil_raw)

                cat, lat = clasificar_con_gemini(pil_raw)
                rest_np = restaurar_imagen_adaptativa(img_np, cat)
                heatmap_np, metricas = auditar_alucinaciones(img_np, rest_np)

                st.session_state.img_original = pil_raw
                st.session_state.img_restaurada = Image.fromarray(rest_np)
                st.session_state.img_heatmap = Image.fromarray(heatmap_np)
                st.session_state.categoria = cat
                st.session_state.latencia = lat
                st.session_state.fidelity = metricas["fidelity_score"]
                st.session_state.hallucination_rate = metricas["hallucination_ratio"]
                st.session_state.ssim_score = metricas["ssim"]
                st.session_state.delta_nitidez = metricas["delta_nitidez"]
                st.session_state.out_res = metricas["out_res"]

                st.rerun()

        # NUEVAS CARACTERÍSTICAS SIN REPETICIÓN DEBO DEL BOTÓN ROJO
        if st.session_state.img_restaurada is not None:
            st.markdown("""
            <div style="margin-top: 18px; margin-bottom: 6px; font-size: 0.70rem; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.08em;">
                📊 Características del Procesamiento
            </div>
            """, unsafe_allow_html=True)

            # Tarjeta 1: SSIM Estructural
            st.markdown(f"""
            <div class="side-metric-card">
                <div class="side-metric-title">SSIM Estructural</div>
                <div class="side-metric-num" style="color: #34d399;">{st.session_state.ssim_score:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

            # Tarjeta 2: Ganancia de Nitidez (Delta Laplaciano)
            st.markdown(f"""
            <div class="side-metric-card">
                <div class="side-metric-title">Ganancia de Nitidez (Δ)</div>
                <div class="side-metric-num" style="color: #a78bfa;">+{st.session_state.delta_nitidez:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Tarjeta 3: Resolución Escalada
            st.markdown(f"""
            <div class="side-metric-card">
                <div class="side-metric-title">Resolución de Salida (2x)</div>
                <div class="side-metric-num" style="font-size: 1.05rem; color: #f8fafc; margin-top: 5px;">{st.session_state.out_res}</div>
            </div>
            """, unsafe_allow_html=True)

            # Botón de Descarga
            buffer_dl = io.BytesIO()
            st.session_state.img_restaurada.save(buffer_dl, format="PNG")
            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Descargar Imagen Restaurada",
                data=buffer_dl.getvalue(),
                file_name="imagen_restaurada.png",
                mime="image/png"
            )

    else:
        st.info("Sube una imagen para comenzar.")


# --- PREPARACIÓN DE IMÁGENES ESCALADAS PARA VISUALIZACIÓN COMPLETA ---
TARGET_PREVIEW_HEIGHT = 700

img_orig_display = None
img_rest_display = None
img_heat_display = None

if st.session_state.img_restaurada is not None and st.session_state.img_original is not None:
    w_rest_raw, h_rest_raw = st.session_state.img_restaurada.size
    
    scale_factor = max(1.0, TARGET_PREVIEW_HEIGHT / float(h_rest_raw))
    display_w = int(w_rest_raw * scale_factor)
    display_h = int(h_rest_raw * scale_factor)

    img_orig_display = st.session_state.img_original.resize((display_w, display_h), Image.Resampling.NEAREST)
    img_rest_display = st.session_state.img_restaurada.resize((display_w, display_h), Image.Resampling.LANCZOS)
    img_heat_display = st.session_state.img_heatmap.resize((display_w, display_h), Image.Resampling.LANCZOS)


# --- COLUMNA 2 (CENTRAL): COMPARADOR ANTES / DESPUÉS ---
with col_centro:
    st.markdown("""
    <div class="viewer-header">
        🖼️ Comparativa: Original vs Restaurada
    </div>
    """, unsafe_allow_html=True)

    if img_rest_display is not None and img_orig_display is not None:
        image_comparison(
            img1=img_orig_display,
            img2=img_rest_display,
            label1="Original (Baja Res)",
            label2=f"Restaurada ({st.session_state.categoria.capitalize()})",
            width=display_w,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )
    else:
        st.markdown("""
        <div style="background: #111827; border: 2px dashed #1f2937; border-radius: 12px; height: 480px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #64748b; text-align: center; padding: 20px;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">🖼️</div>
            <b>Comparador Deslizante</b><br>
            Aquí aparecerá la cortina interactiva entre la imagen original y la restaurada.
        </div>
        """, unsafe_allow_html=True)


# --- COLUMNA 3 (DERECHA): MAPA DE CALOR DE ALUCINACIONES ---
with col_der:
    st.markdown("""
    <div class="viewer-header">
        🔬 Auditoría de Gradientes (Mapa JET)
    </div>
    """, unsafe_allow_html=True)

    if img_heat_display is not None:
        st.image(
            img_heat_display,
            caption="Residuos de Gradiente de Sobolev: Rojo/Amarillo = Modificaciones | Azul = Zona Lisa",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div style="background: #111827; border: 2px dashed #1f2937; border-radius: 12px; height: 480px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #64748b; text-align: center; padding: 20px;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">🔬</div>
            <b>Mapa de Incertidumbre y Alucinaciones</b><br>
            Aquí se proyectará el mapa de calor con los gradientes espaciales de la inferencia.
        </div>
        """, unsafe_allow_html=True)