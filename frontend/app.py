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

# --- 1. CONFIGURACIÓN DE PÁGINA (WIDE COCKPIT FOR PC) ---
st.set_page_config(
    page_title="Aura AI Studio | Neural Restoration Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. INYECCIÓN CSS GLASSMORPHISM DARK SAAS (DESKTOP OPTIMIZED) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    code, pre, .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Fondo Obsidian Deep Space */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0d1527 0%, #05070d 100%);
        color: #f1f5f9;
    }

    /* Estilo de la Barra Superior */
    .saas-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 12px 24px;
        margin-bottom: 20px;
    }

    /* Tarjetas Métricas KPI / Telemetría */
    .telemetry-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 14px 18px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    
    .telemetry-card:hover {
        border-color: rgba(6, 182, 212, 0.35);
        transform: translateY(-2px);
    }

    .telemetry-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .telemetry-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    .telemetry-sub {
        font-size: 0.72rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Contenedor Principal Glass */
    .cockpit-panel {
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 16px 40px -15px rgba(0, 0, 0, 0.6);
    }

    .badge-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    .badge-cyan {
        background: rgba(6, 182, 212, 0.12);
        color: #22d3ee;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }

    .badge-green {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-purple {
        background: rgba(168, 85, 247, 0.12);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }

    /* Ocultar padding excesivo superior de Streamlit */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }

    div[data-testid="stSidebar"] {
        background-color: rgba(10, 15, 26, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
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
if "orig_dims" not in st.session_state:
    st.session_state.orig_dims = (0, 0)
if "out_dims" not in st.session_state:
    st.session_state.out_dims = (0, 0)

# --- 4. TOP NAVBAR ---
st.markdown("""
<div class="saas-navbar">
    <div style="display:flex; align-items:center; gap:12px;">
        <div style="width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #06b6d4, #6366f1); display:flex; align-items:center; justify-content:center; font-size:1.1rem; box-shadow: 0 4px 14px rgba(6, 182, 212, 0.35);">⚡</div>
        <div>
            <span style="font-weight:800; font-size:1.15rem; letter-spacing:-0.02em; color:#ffffff;">AURA RESTORE <span style="font-size:0.75rem; color:#06b6d4; font-weight:600; padding:2px 6px; border:1px solid rgba(6,182,212,0.3); border-radius:6px; margin-left:6px;">PRO DESKTOP</span></span>
            <div style="font-size:0.75rem; color:#64748b; margin-top:-2px;">Motor Autónomo CLIP Zero-Shot + PyTorch GPU Acceleration</div>
        </div>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
        <span class="badge-chip badge-cyan">RTX 3050 (4GB CUDA)</span>
        <span class="badge-chip badge-green">Pipeline Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. TARJETAS DE TELEMETRÍA (4-METRICS ROW) ---
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

active_cat = "EN ESPERA"
if st.session_state.decisions:
    active_cat = getattr(st.session_state.decisions, "categoria", "GENERAL").upper()

applied_cnt = 0
if st.session_state.result_pipeline:
    applied_cnt = len(st.session_state.result_pipeline.get("applied_steps", []))

scale_factor = "1x"
if st.session_state.orig_dims[0] > 0 and st.session_state.out_dims[0] > 0:
    ratio = round(st.session_state.out_dims[0] / st.session_state.orig_dims[0], 1)
    scale_factor = f"{ratio}x"

with m_col1:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">🏷️ Clasificación CLIP</div>
        <div class="telemetry-value" style="color: #38bdf8;">{active_cat}</div>
        <div class="telemetry-sub">Zero-Shot Vision ViT-32</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    lat_val = f"{st.session_state.latency:.2f}s" if st.session_state.latency > 0 else "--"
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">⏱️ Tiempo de Inferencia</div>
        <div class="telemetry-value" style="color: #34d399;">{lat_val}</div>
        <div class="telemetry-sub">Latencia total de ejecución</div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">🔬 Modelos Aplicados</div>
        <div class="telemetry-value" style="color: #c084fc;">{applied_cnt} <span style="font-size:0.85rem; color:#94a3b8; font-weight:400;">etapas</span></div>
        <div class="telemetry-sub">Filtros & Super-Resolución</div>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    res_sub = f"{st.session_state.orig_dims[0]}x{st.session_state.orig_dims[1]} → {st.session_state.out_dims[0]}x{st.session_state.out_dims[1]}" if st.session_state.out_dims[0] > 0 else "Entrada / Salida"
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">📐 Escalado & Resolución</div>
        <div class="telemetry-value" style="color: #fbbf24;">{scale_factor}</div>
        <div class="telemetry-sub">{res_sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# --- 6. WORKSPACE DESKTOP: CONTROL PANEL (IZQ) & VISOR ÚNICO INTERACTIVO (DER) ---
panel_left, panel_right = st.columns([1, 2.4], gap="medium")

# PANEL IZQUIERDO: Configuración, Carga y Despacho
with panel_left:
    st.markdown('<div class="cockpit-panel">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0; color:#f8fafc; font-size:1rem; font-weight:700;'>🕹️ Panel de Ingesta y Control</h4>", unsafe_allow_html=True)
    
    usar_n8n = st.toggle("Orquestar vía n8n Webhook", value=True, help="Deriva el procesamiento a través del workflow de n8n.")
    
    uploaded_file = st.file_uploader(
        "Cargar imagen a restaurar", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        input_path = settings.UPLOADS_DIR / uploaded_file.name
        st.session_state.input_path = input_path
        
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pil_in = Image.open(input_path)
        st.session_state.orig_dims = (pil_in.width, pil_in.height)
        
        st.markdown(f"""
        <div style="margin: 10px 0; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); font-size: 0.8rem; color:#94a3b8;">
            📁 <b>Archivo:</b> {uploaded_file.name}<br>
            📏 <b>Resolución:</b> {pil_in.width} x {pil_in.height} px
        </div>
        """, unsafe_allow_html=True)

        btn_run = st.button("⚡ Ejecutar Pipeline Neural", type="primary", use_container_width=True)

        if btn_run:
            start_ts = time.time()
            if usar_n8n:
                with st.spinner("Enviando imagen a n8n y ejecutando restauración intensiva en GPU (esto puede tardar si la imagen tiene alta resolución)..."):
                    try:
                        with open(input_path, "rb") as f_img:
                            b64_img = base64.b64encode(f_img.read()).decode('utf-8')

                        n8n_url = "http://127.0.0.1:5678/webhook/restaurar-imagen"
                        payload = {
                            "image_path": str(input_path),
                            "base64": b64_img
                        }
                        
                        # Aumentado el timeout a 1200 segundos (20 min) para procesar secuencias de alta resolución sin abortar
                        resp = requests.post(n8n_url, json=payload, timeout=1200)
                        resp.raise_for_status()
                        res_data = resp.json()
                        if isinstance(res_data, list) and len(res_data) > 0:
                            res_data = res_data[0]

                        out_p = res_data.get("output_path")
                        if out_p and Path(out_p).exists():
                            p_out = Image.open(out_p)
                            st.session_state.out_dims = (p_out.width, p_out.height)

                        st.session_state.result_pipeline = {
                            "status": res_data.get("status"),
                            "output_path": out_p,
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
                        st.rerun()

                    except Exception as err:
                        st.error(f"Error de enlace con n8n/Flask: {err}")
                        st.stop()
            else:
                agent = DecisionAgent()
                pipeline = RestorationPipeline()

                with st.spinner("CLIP clasificando imagen localmente..."):
                    try:
                        decisions = agent.analyze_image(input_path)
                        st.session_state.decisions = decisions
                        st.session_state.analyzed = True
                    except Exception as err:
                        st.error(f"Error en DecisionAgent: {err}")
                        st.stop()

                with st.spinner("Procesando en tensores de GPU..."):
                    try:
                        result = pipeline.run(input_path, decisions)
                        out_p = result.get("output_path")
                        if out_p and Path(out_p).exists():
                            p_out = Image.open(out_p)
                            st.session_state.out_dims = (p_out.width, p_out.height)

                        st.session_state.result_pipeline = result
                        st.session_state.latency = time.time() - start_ts
                        st.rerun()
                    except Exception as err:
                        st.error(f"Error en RestorationPipeline: {err}")
                        st.stop()
    else:
        st.markdown("""
        <div style="border: 2px dashed rgba(255,255,255,0.08); border-radius: 10px; padding: 30px 15px; text-align: center; color: #64748b; font-size: 0.85rem;">
            Arrastra una fotografía aquí para comenzar
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.result_pipeline and st.session_state.result_pipeline.get("applied_steps"):
        st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.06); margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:0.75rem; font-weight:600; color:#94a3b8; text-transform:uppercase;'>Cadena de Inferencia:</span>", unsafe_allow_html=True)
        chips_html = "".join([f"<span class='badge-chip badge-purple' style='margin: 3px 2px;'>{s}</span>" for s in st.session_state.result_pipeline["applied_steps"]])
        st.markdown(f"<div style='margin-top:6px;'>{chips_html}</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# PANEL DERECHO: Visor Único Interactivo (Before/After Slider)
with panel_right:
    st.markdown('<div class="cockpit-panel">', unsafe_allow_html=True)
    
    # Barra de estado sobre el visor
    hdr_col1, hdr_col2 = st.columns([2, 1])
    with hdr_col1:
        st.markdown("<h4 style='margin-top:0; color:#f8fafc; font-size:1rem; font-weight:700;'>🖼️ Inspección Interactiva Antes / Después</h4>", unsafe_allow_html=True)
    with hdr_col2:
        if st.session_state.analyzed:
            st.markdown("<div align='right'><span class='badge-chip badge-green'>Comparador Activo</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div align='right'><span class='badge-chip' style='background:rgba(255,255,255,0.05); color:#64748b;'>En Espera</span></div>", unsafe_allow_html=True)

    # Renderizado del Comparador Único
    if st.session_state.analyzed and st.session_state.result_pipeline:
        out_path = st.session_state.result_pipeline.get("output_path")
        if out_path and Path(out_path).exists() and st.session_state.input_path:
            img_original = Image.open(st.session_state.input_path)
            img_restaurada = Image.open(out_path)

            # Visor interactivo único en pantalla panorámica
            image_comparison(
                img1=img_original,
                img2=img_restaurada,
                label1="Original (Input)",
                label2="Restaurada IA (RTX 3050) ✨",
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )

            # Barra de Acciones de Exportación
            exp_col1, exp_col2 = st.columns([3, 1])
            with exp_col1:
                st.caption(f"Salida Master generada: `{Path(out_path).name}`")
            with exp_col2:
                with open(out_path, "rb") as f_out:
                    st.download_button(
                        label="📥 Descargar Master PNG",
                        data=f_out.read(),
                        file_name=Path(out_path).name,
                        mime="image/png",
                        use_container_width=True
                    )
        else:
            st.warning("No se localizó el archivo restaurado en disco.")
    else:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:450px; border:2px dashed rgba(255,255,255,0.06); border-radius:12px; color:#475569; text-align:center; padding:24px;">
            <div style="font-size:3rem; margin-bottom:12px; opacity:0.6;">⚡</div>
            <div style="font-size:1.05rem; font-weight:600; color:#94a3b8;">Visor Panorámico en Espera</div>
            <div style="font-size:0.82rem; color:#64748b; max-width:400px; margin-top:6px;">
                Carga una imagen en el panel izquierdo y haz clic en <b>Ejecutar Pipeline Neural</b> para activar el deslizador comparativo a pantalla completa.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)