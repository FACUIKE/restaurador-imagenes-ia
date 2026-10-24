import base64
import sys
import time
from pathlib import Path
import requests
from PIL import Image

# Añadir la raíz del proyecto al sys.path para compatibilidad de módulos internos
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from streamlit_image_comparison import image_comparison

from config import settings
from agent.decision_agent import DecisionAgent
from pipeline.orchestrator import RestorationPipeline

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Aura AI Studio | Neural Restoration Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. INYECCIÓN CSS GLASSMORPHISM DARK SAAS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    code, pre, .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Fondo general estilo Obsidian / Modern SaaS */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #07090e 100%);
        color: #f1f5f9;
    }

    /* Tarjetas Glassmorphism con bordes brillantes */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
    }

    .glass-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 12px;
    }

    /* Badges de Hardware y Estado */
    .badge-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    .badge-gpu-active {
        background: rgba(6, 182, 212, 0.12);
        color: #22d3ee;
        border: 1px solid rgba(6, 182, 212, 0.35);
    }

    .badge-status-online {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .badge-meta {
        background: rgba(255, 255, 255, 0.05);
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Ajustes sutiles a componentes de Streamlit */
    div[data-testid="stSidebar"] {
        background-color: rgba(10, 15, 26, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. ESTADO DE SESIÓN ---
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "decisions" not in st.session_state:
    st.session_state.decisions = None
if "result_pipeline" not in st.session_state:
    st.session_state.result_pipeline = None
if "input_path" not in st.session_state:
    st.session_state.input_path = None
if "latency" not in st.session_state:
    st.session_state.latency = 0.0

# --- 4. BARRA LATERAL (TELEMETRÍA Y CONTROLES AVANZADOS) ---
with st.sidebar:
    st.markdown("### 🎛️ **Engine Control**")
    st.caption("Configuración de inferencia y orquestación")
    
    usar_n8n = st.toggle("Modo Orquestador n8n", value=True, help="Deriva el tráfico por el flujo de trabajo de n8n.")
    
    st.divider()
    st.markdown("#### ⚡ Hardware Telemetry")
    st.markdown("""
        <div style="margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                <span style="color:#94a3b8;">GPU Device</span>
                <span style="color:#f8fafc; font-weight:600;">RTX 3050 (4GB)</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                <span style="color:#94a3b8;">VRAM Usage (Est.)</span>
                <span style="color:#22d3ee; font-weight:600;">2.8 GB / 4.0 GB</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.progress(0.70)

    st.divider()
    st.markdown("#### 🧪 Model Pipelines")
    st.markdown(f"- **Clasificador:** `CLIP ViT-Base-32`")
    st.markdown(f"- **Rostros:** `GFPGAN / CodeFormer`")
    st.markdown(f"- **Texto:** `Real-ESRGAN + Sharpen`")
    st.markdown(f"- **General:** `NAFNet Deblur`")
    
    if st.session_state.latency > 0:
        st.divider()
        st.markdown(f"⏱️ **Última latencia:** `{st.session_state.latency:.2f}s`")

# --- 5. TOP NAVBAR / HEADER ---
nav_col1, nav_col2 = st.columns([3, 2])
with nav_col1:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <div style="width: 36px; height: 36px; border-radius: 8px; background: linear-gradient(135deg, #06b6d4, #3b82f6); display: flex; align-items: center; justify-content: center; font-size: 1.2rem;">⚡</div>
            <div>
                <h2 style="margin: 0; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;">AURA RESTORE <span style="font-weight: 400; color: #64748b; font-size: 1rem;">v2.1 PRO</span></h2>
                <p style="margin: 0; color: #94a3b8; font-size: 0.85rem;">Motor Autónomo de Clasificación Zero-Shot y Super-Resolución Neuronal</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

with nav_col2:
    st.markdown("""
        <div style="display: flex; justify-content: flex-end; gap: 8px; align-items: center; height: 100%;">
            <span class="badge-chip badge-status-online">● Backend Online</span>
            <span class="badge-chip badge-gpu-active">CUDA Core Ready</span>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# --- 6. WORKSPACE PRINCIPAL (GRID 2 COLUMNAS) ---
col_ops, col_view = st.columns([1, 1], gap="large")

# COLUMNA IZQUIERDA: Ingesta de Datos y Disparador
with col_ops:
    st.markdown("""
        <div class="glass-card">
            <div class="glass-card-header">
                <span style="font-weight: 700; font-size: 1rem; color: #f8fafc;">1. Ingesta de Imagen</span>
                <span class="badge-chip badge-meta">JPG, PNG, JPEG</span>
            </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona el archivo a restaurar", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        input_path = settings.UPLOADS_DIR / uploaded_file.name
        st.session_state.input_path = input_path
        
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        img_preview = Image.open(input_path)
        
        st.image(img_preview, caption=f"Entrada cargada: {uploaded_file.name} ({img_preview.width}x{img_preview.height})", use_container_width=True)
        
        btn_action = st.button("⚡ Ejecutar Pipeline Neural", type="primary", use_container_width=True)
        
        if btn_action:
            start_ts = time.time()
            if usar_n8n:
                with st.spinner("Enviando memoria RAM a n8n y evaluando con CLIP..."):
                    try:
                        with open(input_path, "rb") as f_img:
                            b64_img = base64.b64encode(f_img.read()).decode('utf-8')

                        n8n_url = "http://127.0.0.1:5678/webhook/restaurar-imagen"
                        payload = {
                            "image_path": str(input_path),
                            "base64": b64_img
                        }
                        
                        resp = requests.post(n8n_url, json=payload, timeout=300)
                        resp.raise_for_status()
                        res_data = resp.json()
                        if isinstance(res_data, list) and len(res_data) > 0:
                            res_data = res_data[0]

                        st.session_state.result_pipeline = {
                            "status": res_data.get("status"),
                            "output_path": res_data.get("output_path"),
                            "applied_steps": res_data.get("applied_steps", []),
                            "duration_seconds": res_data.get("duration_seconds", round(time.time() - start_ts, 2))
                        }

                        class RestoredDecision:
                            def __init__(self, diag, cat):
                                self.motivo_diagnostico = diag
                                self.categoria = cat

                        st.session_state.decisions = RestoredDecision(
                            res_data.get("diagnostic", "Pipeline autónomo n8n"),
                            res_data.get("categoria", "general")
                        )
                        st.session_state.latency = time.time() - start_ts
                        st.session_state.analyzed = True

                    except Exception as err:
                        st.error(f"Fallo en la comunicación con n8n/Flask: {err}")
                        st.stop()
            else:
                agent = DecisionAgent()
                pipeline = RestorationPipeline()

                with st.spinner("Inspección de CLIP local en curso..."):
                    try:
                        decisions = agent.analyze_image(input_path)
                        st.session_state.decisions = decisions
                        st.session_state.analyzed = True
                    except Exception as err:
                        st.error(f"Error en DecisionAgent: {err}")
                        st.stop()

                with st.spinner("Procesando tensores en GPU..."):
                    try:
                        result = pipeline.run(input_path, decisions)
                        st.session_state.result_pipeline = result
                        st.session_state.latency = time.time() - start_ts
                    except Exception as err:
                        st.error(f"Error en RestorationPipeline: {err}")
                        st.stop()

    st.markdown("</div>", unsafe_allow_html=True)

# COLUMNA DERECHA: Telemetría de Decisión y Comparador Before/After
with col_view:
    st.markdown("""
        <div class="glass-card">
            <div class="glass-card-header">
                <span style="font-weight: 700; font-size: 1rem; color: #f8fafc;">2. Inspección Visual y Métricas</span>
                <span class="badge-chip badge-meta">Split-Screen View</span>
            </div>
    """, unsafe_allow_html=True)

    if st.session_state.analyzed and st.session_state.decisions is not None:
        decisions = st.session_state.decisions
        result = st.session_state.result_pipeline
        cat = getattr(decisions, "categoria", "general").upper()

        # Telemetría en tarjeta compacta
        st.markdown(f"""
            <div style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 14px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 16px;">
                <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
                    <span style="color:#94a3b8; font-size:0.85rem;">Categoría CLIP:</span>
                    <span style="color:#38bdf8; font-weight:700; font-size:0.85rem;">{cat}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#94a3b8; font-size:0.85rem;">Tiempo de Proceso:</span>
                    <span style="color:#34d399; font-weight:700; font-size:0.85rem;">{result.get('duration_seconds', 0.0)}s</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if result and "applied_steps" in result and result["applied_steps"]:
            tags_html = " ".join([f"<span class='badge-chip badge-meta'>{step}</span>" for step in result["applied_steps"]])
            st.markdown(f"<div style='margin-bottom: 14px;'>{tags_html}</div>", unsafe_allow_html=True)

        # Visor Interactivo
        out_path = result.get("output_path")
        if out_path and Path(out_path).exists():
            img_before = Image.open(st.session_state.input_path)
            img_after = Image.open(out_path)

            image_comparison(
                img1=img_before,
                img2=img_after,
                label1="Original",
                label2="Restaurada IA ✨",
                width=700,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )

            # Botón de Descarga
            with open(out_path, "rb") as f_out:
                st.download_button(
                    label="📥 Exportar Master en Alta Resolución (PNG)",
                    data=f_out.read(),
                    file_name=Path(out_path).name,
                    mime="image/png",
                    use_container_width=True
                )
        else:
            st.warning("No se localizó el archivo de salida restaurado.")
    else:
        st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 320px; border: 2px dashed rgba(255,255,255,0.08); border-radius: 12px; color: #64748b; text-align: center; padding: 20px;">
                <span style="font-size: 2.2rem; margin-bottom: 10px;">⚡</span>
                <span style="font-weight: 500; font-size: 0.95rem;">Área de Comparación en Espera</span>
                <span style="font-size: 0.8rem; color: #475569; margin-top: 4px;">Sube una imagen y ejecuta el pipeline para habilitar el deslizador interactivo.</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)