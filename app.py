import os
import time
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import streamlit as st
from streamlit_image_comparison import image_comparison
from google import genai

# Cargar variables de entorno locales (.env)
load_dotenv()

# Configuración de página panorámica
st.set_page_config(
    page_title="Aura AI Studio | Neural Restoration",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos visuales Dark Glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0d1527 0%, #05070d 100%);
        color: #f1f5f9;
    }

    .saas-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 12px 24px;
        margin-bottom: 20px;
    }

    .telemetry-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 14px 18px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease;
    }
    .telemetry-card:hover {
        transform: translateY(-2px);
    }
    .telemetry-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
    }
    .telemetry-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    .cockpit-panel {
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 16px 40px -15px rgba(0, 0, 0, 0.6);
    }

    .badge-chip {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .badge-cyan { background: rgba(6, 182, 212, 0.12); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.3); }
    .badge-green { background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }
</style>
""", unsafe_allow_html=True)

# Obtener clave API (soporta .env local o secrets de Streamlit Cloud)
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY and "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]

# Inicializar Estado de Sesión
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "categoria" not in st.session_state:
    st.session_state.categoria = "EN ESPERA"
if "latency" not in st.session_state:
    st.session_state.latency = 0.0
if "orig_dims" not in st.session_state:
    st.session_state.orig_dims = (0, 0)
if "out_dims" not in st.session_state:
    st.session_state.out_dims = (0, 0)
if "img_input" not in st.session_state:
    st.session_state.img_input = None
if "img_output" not in st.session_state:
    st.session_state.img_output = None

# Barra superior
st.markdown("""
<div class="saas-navbar">
    <div style="display:flex; align-items:center; gap:12px;">
        <div style="width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #06b6d4, #6366f1); display:flex; align-items:center; justify-content:center; font-size:1.1rem;">⚡</div>
        <div>
            <span style="font-weight:800; font-size:1.15rem; color:#ffffff;">AURA RESTORE <span style="font-size:0.75rem; color:#06b6d4; font-weight:600; padding:2px 6px; border:1px solid rgba(6,182,212,0.3); border-radius:6px; margin-left:6px;">CLOUD EDITION</span></span>
            <div style="font-size:0.75rem; color:#64748b;">Clasificación Multimodal Gemini + Restauración Adaptativa</div>
        </div>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
        <span class="badge-chip badge-cyan">Serverless Cloud</span>
        <span class="badge-chip badge-green">Zero-Timeout Engine</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 4 Métricas de Telemetría
c1, c2, c3, c4 = st.columns(4)
scale_txt = f"{round(st.session_state.out_dims[0] / st.session_state.orig_dims[0], 1)}x" if st.session_state.orig_dims[0] > 0 else "1x"
res_txt = f"{st.session_state.orig_dims[0]}x{st.session_state.orig_dims[1]} → {st.session_state.out_dims[0]}x{st.session_state.out_dims[1]}" if st.session_state.out_dims[0] > 0 else "Entrada / Salida"

with c1:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">🏷️ Clasificación Visual</div>
        <div class="telemetry-value" style="color: #38bdf8;">{st.session_state.categoria.upper()}</div>
        <span style="font-size:0.7rem; color:#64748b;">Gemini 3.8 Flash Vision</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">⏱️ Tiempo de Inferencia</div>
        <div class="telemetry-value" style="color: #34d399;">{st.session_state.latency:.2f}s</div>
        <span style="font-size:0.7rem; color:#64748b;">Latencia total Cloud</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">🔬 Escalado & Resolución</div>
        <div class="telemetry-value" style="color: #fbbf24;">{scale_txt}</div>
        <span style="font-size:0.7rem; color:#64748b;">{res_txt}</span>
    </div>
    """, unsafe_allow_html=True)

with c4:
    estado_api = "Conectada" if API_KEY else "Falta API Key"
    color_api = "#34d399" if API_KEY else "#f87171"
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">🔑 Estado API Gemini</div>
        <div class="telemetry-value" style="color: {color_api}; font-size:1.15rem;">{estado_api}</div>
        <span style="font-size:0.7rem; color:#64748b;">Free Tier Serverless</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Funciones de Soporte
def clasificar_con_gemini(pil_image: Image.Image, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    prompt = """
    Analiza esta imagen y clasifícala estrictamente en una de estas 3 opciones:
    - rostro
    - texto
    - paisaje
    
    Responde ÚNICAMENTE con una sola palabra en minúsculas: 'rostro', 'texto' o 'paisaje'.
    """
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[pil_image, prompt]
    )
    cat = response.text.strip().lower()
    if cat not in ["rostro", "texto", "paisaje"]:
        return "paisaje"
    return cat

def restaurar_imagen(pil_img: Image.Image, categoria: str) -> Image.Image:
    # Conversión PIL -> OpenCV BGR
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    if categoria == "texto":
        denoised = cv2.bilateralFilter(img_cv, d=9, sigmaColor=75, sigmaSpace=75)
        upscaled = cv2.resize(denoised, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
        resultado = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)
    elif categoria == "rostro":
        denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, 6, 6, 7, 21)
        upscaled = cv2.resize(denoised, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]])
        resultado = cv2.filter2D(upscaled, -1, kernel)
    else:  # paisaje / general
        upscaled = cv2.resize(img_cv, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        resultado = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    # Conversión OpenCV BGR -> PIL RGB
    return Image.fromarray(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))

# Workspace: Panel Izquierdo (Ingesta) y Panel Derecho (Visor Interactivo)
col_left, col_right = st.columns([1, 2.3], gap="medium")

with col_left:
    st.markdown('<div class="cockpit-panel">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0; color:#f8fafc; font-size:1rem; font-weight:700;'>🕹️ Ingesta y Control</h4>", unsafe_allow_html=True)

    if not API_KEY:
        st.warning("⚠️ Ingresa tu GEMINI_API_KEY en tu archivo `.env` o en los Secrets de Streamlit.")

    uploaded = st.file_uploader("Seleccionar imagen a restaurar", type=["jpg", "jpeg", "png", "webp"])

    if uploaded is not None:
        pil_in = Image.open(uploaded)
        st.session_state.img_input = pil_in
        st.session_state.orig_dims = (pil_in.width, pil_in.height)

        st.markdown(f"""
        <div style="margin: 10px 0; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); font-size: 0.8rem; color:#94a3b8;">
            📁 <b>Archivo:</b> {uploaded.name}<br>
            📏 <b>Resolución Original:</b> {pil_in.width} x {pil_in.height} px
        </div>
        """, unsafe_allow_html=True)

        if st.button("⚡ Ejecutar Restauración Cloud", type="primary", use_container_width=True, disabled=not API_KEY):
            t0 = time.time()
            with st.spinner("1. Clasificando con Gemini Vision API..."):
                cat = clasificar_con_gemini(pil_in, API_KEY)
                st.session_state.categoria = cat

            with st.spinner(f"2. Aplicando restauración adaptativa para [{cat.upper()}]..."):
                pil_out = restaurar_imagen(pil_in, cat)
                st.session_state.img_output = pil_out
                st.session_state.out_dims = (pil_out.width, pil_out.height)

            st.session_state.latency = time.time() - t0
            st.session_state.analyzed = True
            st.rerun()
    else:
        st.markdown("""
        <div style="border: 2px dashed rgba(255,255,255,0.08); border-radius: 10px; padding: 30px 15px; text-align: center; color: #64748b; font-size: 0.85rem;">
            Arrastra una fotografía aquí para comenzar
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="cockpit-panel">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0; color:#f8fafc; font-size:1rem; font-weight:700;'>🖼️ Inspección Interactiva Antes / Después</h4>", unsafe_allow_html=True)

    if st.session_state.analyzed and st.session_state.img_input and st.session_state.img_output:
        image_comparison(
            img1=st.session_state.img_input,
            img2=st.session_state.img_output,
            label1="Original (Input)",
            label2="Restaurada IA (Cloud 2x) ✨",
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )

        # Botón de Descarga directa en memoria
        buf = BytesIO()
        st.session_state.img_output.save(buf, format="PNG")
        st.download_button(
            label="📥 Descargar Imagen Restaurada (PNG)",
            data=buf.getvalue(),
            file_name="restaurada_master.png",
            mime="image/png",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:450px; border:2px dashed rgba(255,255,255,0.06); border-radius:12px; color:#475569; text-align:center; padding:24px;">
            <div style="font-size:3rem; margin-bottom:12px; opacity:0.6;">⚡</div>
            <div style="font-size:1.05rem; font-weight:600; color:#94a3b8;">Visor Panorámico en Espera</div>
            <div style="font-size:0.82rem; color:#64748b; max-width:400px; margin-top:6px;">
                Carga una imagen en el panel de control y presiona <b>Ejecutar Restauración Cloud</b> para ver el deslizador comparativo.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

