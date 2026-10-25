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
    page_title="Aura AI Studio | Neural Restoration & Audit",
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
    .badge-amber { background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }

    /* Pestañas estilizadas */
    button[data-baseweb="tab"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
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
if "engine_used" not in st.session_state:
    st.session_state.engine_used = "Gemini Flash Vision"
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
if "img_heatmap" not in st.session_state:
    st.session_state.img_heatmap = None
if "fidelity_score" not in st.session_state:
    st.session_state.fidelity_score = 0.0
if "hallucination_ratio" not in st.session_state:
    st.session_state.hallucination_ratio = 0.0
if "ssim_score" not in st.session_state:
    st.session_state.ssim_score = 0.0

# Barra superior
st.markdown("""
<div class="saas-navbar">
    <div style="display:flex; align-items:center; gap:12px;">
        <div style="width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #06b6d4, #6366f1); display:flex; align-items:center; justify-content:center; font-size:1.1rem;">⚡</div>
        <div>
            <span style="font-weight:800; font-size:1.15rem; color:#ffffff;">AURA RESTORE <span style="font-size:0.75rem; color:#06b6d4; font-weight:600; padding:2px 6px; border:1px solid rgba(6,182,212,0.3); border-radius:6px; margin-left:6px;">RESEARCH EDITION</span></span>
            <div style="font-size:0.75rem; color:#64748b;">Enrutador Semántico + Restauración Adaptativa + Auditoría de Alucinaciones</div>
        </div>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
        <span class="badge-chip badge-cyan">Serverless Cloud</span>
        <span class="badge-chip badge-amber">Hallucination Audit</span>
        <span class="badge-chip badge-green">Pipeline Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 4 Métricas de Telemetría Dinámicas
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">🏷️ Clasificación Visual</div>
        <div class="telemetry-value" style="color: #38bdf8;">{st.session_state.categoria.upper()}</div>
        <span style="font-size:0.7rem; color:#64748b;">{st.session_state.engine_used}</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">⏱️ Tiempo de Inferencia</div>
        <div class="telemetry-value" style="color: #34d399;">{st.session_state.latency:.2f}s</div>
        <span style="font-size:0.7rem; color:#64748b;">Latencia de ejecución</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    fid_val = f"{st.session_state.fidelity_score:.3f}" if st.session_state.analyzed else "--"
    fid_color = "#34d399" if st.session_state.fidelity_score >= 0.70 else "#f59e0b"
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">🛡️ Fidelity Score</div>
        <div class="telemetry-value" style="color: {fid_color};">{fid_val}</div>
        <span style="font-size:0.7rem; color:#64748b;">Confiabilidad estructural [0-1]</span>
    </div>
    """, unsafe_allow_html=True)

with c4:
    hal_val = f"{st.session_state.hallucination_ratio:.1f}%" if st.session_state.analyzed else "--"
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">🔬 Tasa de Alucinación</div>
        <div class="telemetry-value" style="color: #fbbf24;">{hal_val}</div>
        <span style="font-size:0.7rem; color:#64748b;">Discrepancia de alta frecuencia</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Función de Clasificación con Cascada de Modelos y Reintentos
def clasificar_con_gemini(pil_image: Image.Image, api_key: str):
    client = genai.Client(api_key=api_key)
    prompt = """
    Analiza esta imagen y clasifícala estrictamente en una de estas 3 opciones:
    - rostro
    - texto
    - paisaje
    
    Responde ÚNICAMENTE con una sola palabra en minúsculas: 'rostro', 'texto' o 'paisaje'.
    """
    
    modelos_candidatos = [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ]

    for model_name in modelos_candidatos:
        for _ in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[pil_image, prompt]
                )
                if response and response.text:
                    cat = response.text.strip().lower()
                    for valida in ["rostro", "texto", "paisaje"]:
                        if valida in cat:
                            return valida, f"Gemini ({model_name})"
            except Exception:
                time.sleep(0.8)
                continue

    return "paisaje", "Heurístico Local (Fallback)"

def restaurar_imagen(pil_img: Image.Image, categoria: str) -> Image.Image:
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

    return Image.fromarray(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))

def auditar_alucinaciones(pil_orig: Image.Image, pil_rest: Image.Image):
    """
    Computa los gradientes espaciales de Sobel, genera el mapa de calor de incertidumbre
    y extrae el Fidelity Score y Hallucination Ratio en memoria.
    """
    arr_orig = np.array(pil_orig.convert("L"))
    arr_rest = np.array(pil_rest.convert("L"))

    h, w = arr_rest.shape[:2]
    # Reescalar imagen original al tamaño final para comparar mapas espaciales
    arr_orig_scaled = cv2.resize(arr_orig, (w, h), interpolation=cv2.INTER_CUBIC)

    # 1. Gradientes de Sobel
    sobel_orig_x = cv2.Sobel(arr_orig_scaled, cv2.CV_64F, 1, 0, ksize=3)
    sobel_orig_y = cv2.Sobel(arr_orig_scaled, cv2.CV_64F, 0, 1, ksize=3)
    grad_orig = np.sqrt(sobel_orig_x**2 + sobel_orig_y**2)

    sobel_rest_x = cv2.Sobel(arr_rest, cv2.CV_64F, 1, 0, ksize=3)
    sobel_rest_y = cv2.Sobel(arr_rest, cv2.CV_64F, 0, 1, ksize=3)
    grad_rest = np.sqrt(sobel_rest_x**2 + sobel_rest_y**2)

    # 2. Residuo diferencial
    diff = np.abs(grad_rest - grad_orig)

    # 3. Identificación de píxeles anómalos (percentil 90)
    umbral = np.percentile(diff, 90)
    anomalias = np.where(diff > umbral, diff, 0)

    # 4. Mapa de Calor (JET Colormap) superpuesto
    anomalias_norm = cv2.normalize(anomalias, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(anomalias_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    rest_rgb = np.array(pil_rest)
    overlay = cv2.addWeighted(rest_rgb, 0.65, heatmap_rgb, 0.35, 0)
    pil_heatmap = Image.fromarray(overlay)

    # 5. Métricas numéricas
    error_medio = np.mean(diff) / 255.0
    fidelity_score = max(0.0, min(1.0, 1.0 - (error_medio * 2.0)))
    hallucination_ratio = (np.count_nonzero(anomalias) / anomalias.size) * 100.0

    # Similitud estructural simplificada (SSIM proxy sobre luminancia)
    c1_c = (0.01 * 255)**2
    c2_c = (0.03 * 255)**2
    mu1 = np.mean(arr_orig_scaled)
    mu2 = np.mean(arr_rest)
    sigma1_sq = np.var(arr_orig_scaled)
    sigma2_sq = np.var(arr_rest)
    sigma12 = np.mean((arr_orig_scaled - mu1) * (arr_rest - mu2))
    ssim_val = ((2 * mu1 * mu2 + c1_c) * (2 * sigma12 + c2_c)) / ((mu1**2 + mu2**2 + c1_c) * (sigma1_sq + sigma2_sq + c2_c))

    return pil_heatmap, round(fidelity_score, 4), round(hallucination_ratio, 2), round(float(ssim_val), 4)

# Layout de dos columnas
col_left, col_right = st.columns([1, 2.3], gap="medium")

with col_left:
    st.markdown('<div class="cockpit-panel">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0; color:#f8fafc; font-size:1rem; font-weight:700;'>🕹️ Ingesta y Control</h4>", unsafe_allow_html=True)

    if not API_KEY:
        st.warning("⚠️ Configura GEMINI_API_KEY en tu archivo `.env` o en Secrets de Streamlit.")

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

        if st.button("⚡ Ejecutar Restauración + Auditoría", type="primary", use_container_width=True, disabled=not API_KEY):
            t0 = time.time()
            
            with st.spinner("1. Clasificando semántica con Gemini Vision API..."):
                cat, engine_name = clasificar_con_gemini(pil_in, API_KEY)
                st.session_state.categoria = cat
                st.session_state.engine_used = engine_name

            with st.spinner(f"2. Restauración adaptativa 2x [{cat.upper()}]..."):
                pil_out = restaurar_imagen(pil_in, cat)
                st.session_state.img_output = pil_out
                st.session_state.out_dims = (pil_out.width, pil_out.height)

            with st.spinner("3. Auditando alucinaciones y gradientes de Sobolev..."):
                heatmap, fid, hal, ssim_v = auditar_alucinaciones(pil_in, pil_out)
                st.session_state.img_heatmap = heatmap
                st.session_state.fidelity_score = fid
                st.session_state.hallucination_ratio = hal
                st.session_state.ssim_score = ssim_v

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
    
    if st.session_state.analyzed and st.session_state.img_input and st.session_state.img_output:
        tab_restore, tab_forensic = st.tabs([
            "🖼️ Restauración (Antes / Después)",
            "🔬 Auditoría de Alucinaciones (Heatmap)"
        ])

        with tab_restore:
            image_comparison(
                img1=st.session_state.img_input,
                img2=st.session_state.img_output,
                label1="Original (Input)",
                label2=f"Restaurada IA (2x {st.session_state.categoria.title()}) ✨",
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )

            buf = BytesIO()
            st.session_state.img_output.save(buf, format="PNG")
            st.download_button(
                label="📥 Descargar Master Restaurada (PNG)",
                data=buf.getvalue(),
                file_name="restaurada_master.png",
                mime="image/png",
                use_container_width=True
            )

        with tab_forensic:
            st.markdown("""
            <div style="margin-bottom: 12px; font-size: 0.82rem; color: #94a3b8;">
                El mapa de calor superpone en <b>rojo y amarillo</b> las regiones con mayor gradiente residual (nuevas texturas introducidas por la máscara de enfoque), mientras que los tonos <b>azules</b> señalan residuo nulo (fidelidad pura a zonas lisas).
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.img_heatmap:
                image_comparison(
                    img1=st.session_state.img_output,
                    img2=st.session_state.img_heatmap,
                    label1="Restaurada IA",
                    label2="Mapa de Calor Residual (JET)",
                    starting_position=50,
                    show_labels=True,
                    make_responsive=True,
                    in_memory=True
                )

                sub_c1, sub_c2, sub_c3 = st.columns(3)
                with sub_c1:
                    st.metric("Fidelity Score", f"{st.session_state.fidelity_score:.4f}", help="Cercanía a 1.0 indica alta fidelidad estructural")
                with sub_c2:
                    st.metric("Ratio de Alucinaciones", f"{st.session_state.hallucination_ratio:.1f}%", help="Porcentaje de píxeles con variaciones de alta frecuencia")
                with sub_c3:
                    st.metric("SSIM Estructural", f"{st.session_state.ssim_score:.4f}", help="Similitud estructural respecto al original")
    else:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:450px; border:2px dashed rgba(255,255,255,0.06); border-radius:12px; color:#475569; text-align:center; padding:24px;">
            <div style="font-size:3rem; margin-bottom:12px; opacity:0.6;">⚡</div>
            <div style="font-size:1.05rem; font-weight:600; color:#94a3b8;">Visor Panorámico en Espera</div>
            <div style="font-size:0.82rem; color:#64748b; max-width:400px; margin-top:6px;">
                Carga una imagen en el panel de control y presiona <b>Ejecutar Restauración + Auditoría</b> para ver tanto la restauración como la inspección forense de alucinaciones.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)