import os
import io
import time
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from streamlit_image_comparison import image_comparison
from google import genai

load_dotenv()

# ==========================================
# Configuración General
# ==========================================
st.set_page_config(
    page_title="VisionRestore AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def cargar_css(ruta_archivo):
    with open(ruta_archivo) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Cargamos el CSS de la carpeta externa
cargar_css("css/style.css")

# ==========================================
# Funciones de Visión y Restauración PDI
# ==========================================
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

def clasificar_semantica(pil_img: Image.Image) -> tuple[str, float]:
    if not client: return "paisaje", 0.0
    t0 = time.perf_counter()
    prompt = "Clasifica estrictamente en: 'rostro', 'texto' o 'paisaje'. Responde solo con una de esas tres palabras en minúsculas."
    cat = "paisaje"
    for m in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
        try:
            res = client.models.generate_content(model=m, contents=[pil_img, prompt])
            texto = res.text.strip().lower()
            if any(k in texto for k in ["rostro", "texto", "paisaje"]):
                cat = next(k for k in ["rostro", "texto", "paisaje"] if k in texto)
                break
        except Exception:
            continue
    return cat, time.perf_counter() - t0

def restaurar_adaptativa(img_np: np.ndarray, categoria: str, sub_modo: str = "") -> np.ndarray:
    # Si no hay técnica seleccionada (Toggles apagados), devolvemos la imagen original
    if not sub_modo: return img_np 
    
    h, w = img_np.shape[:2]
    nw, nh = int(w * 2), int(h * 2)
    
    if categoria == "texto":
        escalada = cv2.resize(img_np, (nw, nh), interpolation=cv2.INTER_CUBIC)
        
        if sub_modo == "Trazos Desgastados (Top-Hat)":
            gris = cv2.cvtColor(escalada, cv2.COLOR_RGB2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
            blackhat = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, kernel)
            blackhat_fuerte = cv2.convertScaleAbs(blackhat, alpha=2.5, beta=0)
            resultado = cv2.subtract(gris, blackhat_fuerte)
            resultado = cv2.normalize(resultado, None, 0, 255, cv2.NORM_MINMAX)
            return cv2.cvtColor(resultado, cv2.COLOR_GRAY2RGB)
            
        elif sub_modo == "Eliminar Muaré de Pantalla (FFT 2D)":
            gris = cv2.cvtColor(escalada, cv2.COLOR_RGB2GRAY)
            f = np.fft.fft2(gris)
            fshift = np.fft.fftshift(f)
            filas, cols = gris.shape
            crow, ccol = filas // 2 , cols // 2
            mascara = np.ones((filas, cols), np.uint8)
            mascara[crow-3:crow+3, :] = 0 
            mascara[:, ccol-3:ccol+3] = 0
            mascara[crow-40:crow+40, ccol-40:ccol+40] = 1
            fshift = fshift * mascara
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.abs(np.fft.ifft2(f_ishift))
            img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, binarizada = cv2.threshold(img_back, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.cvtColor(binarizada, cv2.COLOR_GRAY2RGB)
            
        elif sub_modo == "Iluminación Irregular (Sauvola)":
            gris = cv2.cvtColor(escalada, cv2.COLOR_RGB2GRAY)
            binarizada = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
            return cv2.cvtColor(binarizada, cv2.COLOR_GRAY2RGB)
            
    elif categoria == "rostro":
        ycbcr = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        y_up = cv2.resize(y, (nw, nh), interpolation=cv2.INTER_CUBIC)
        cr_up = cv2.resize(cr, (nw, nh), interpolation=cv2.INTER_CUBIC)
        cb_up = cv2.resize(cb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        escalada = cv2.cvtColor(cv2.merge((y_up, cr_up, cb_up)), cv2.COLOR_YCrCb2RGB)

        if sub_modo == "Trepidación (Deconvolución / Sharpening)":
            kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            resultado = cv2.filter2D(escalada, -1, kernel_sharp)
            return cv2.addWeighted(escalada, 0.3, resultado, 0.7, 0)

        elif sub_modo == "Ruido ISO Alto (No-Local Means)":
            return cv2.fastNlMeansDenoisingColored(escalada, None, 10, 10, 7, 21)

        elif sub_modo == "Iluminación Nocturna (MSRCR / Retinex)":
            img_float = np.float32(escalada) + 1.0
            retinex = np.zeros_like(img_float)
            for sigma in [15, 80, 250]:
                blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
                retinex += np.log10(img_float) - np.log10(blur)
            retinex = (retinex / 3.0) * 255.0
            resultado = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
            resultado = cv2.convertScaleAbs(resultado, alpha=1.2, beta=10)
            return cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)

        elif sub_modo == "Suavizado de Piel (Filtro Guiado)":
            return cv2.edgePreservingFilter(escalada, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.2)

        elif sub_modo == "Super-Resolución (Solo YCbCr)":
            y_up_clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(y_up)
            return cv2.cvtColor(cv2.merge((y_up_clahe, cr_up, cb_up)), cv2.COLOR_YCrCb2RGB)
            
    return img_np

def auditar_alucinaciones(orig_rgb: np.ndarray, rest_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    h_r, w_r = rest_rgb.shape[:2]
    deg_scaled = cv2.resize(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY), (w_r, h_r), interpolation=cv2.INTER_CUBIC)
    res_g = cv2.cvtColor(rest_rgb, cv2.COLOR_RGB2GRAY)
    g_deg = np.sqrt(cv2.Sobel(deg_scaled, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(deg_scaled, cv2.CV_64F, 0, 1)**2)
    g_res = np.sqrt(cv2.Sobel(res_g, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(res_g, cv2.CV_64F, 0, 1)**2)
    diff = np.abs(g_res - g_deg)
    anomalias = np.where(diff > np.percentile(diff, 90), diff, 0)
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(cv2.normalize(anomalias, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    n_orig, n_rest = float(cv2.Laplacian(deg_scaled, cv2.CV_64F).var()), float(cv2.Laplacian(res_g, cv2.CV_64F).var())
    
    return cv2.addWeighted(rest_rgb, 0.65, heatmap_rgb, 0.35, 0), {
        "fid": max(0.0, min(1.0, 1.0 - ((np.mean(diff) / 255.0) * 2.0))),
        "hal": (np.count_nonzero(anomalias) / anomalias.size) * 100.0,
        "delta": ((n_rest - n_orig) / max(n_orig, 1.0)) * 100.0,
        "res": f"{w_r} x {h_r} px"
    }

# ==========================================
# Lógica de Exclusión Mutua para los Toggles
# ==========================================
def exclusion_mutua(clave_activa, lista_claves):
    if st.session_state[clave_activa]:
        for clave in lista_claves:
            if clave != clave_activa:
                st.session_state[clave] = False

def draw_tool_card(title, desc, key, group_keys):
    st.markdown("""<hr style="margin: 8px 0; border-color: #1f2937;">""", unsafe_allow_html=True)
    c1, c2 = st.columns([3.5, 1], vertical_alignment="center")
    with c1:
        st.markdown(f"<strong style='color:#e2e8f0;'>{title}</strong><br><span style='font-size:0.75rem;color:#94a3b8;line-height:1.2;display:block;'>{desc}</span>", unsafe_allow_html=True)
    with c2:
        st.toggle("Activar", key=key, label_visibility="collapsed", on_change=exclusion_mutua, args=(key, group_keys))

# ==========================================
# Inicialización de Estados
# ==========================================
CLAVES_TEXTO = ["t_sau", "t_top", "t_fft"]
CLAVES_ROSTRO = ["r_sr", "r_trep", "r_iso", "r_noc", "r_piel"]
estados_iniciales = {
    "cat": "PENDIENTE", "lat": 0.0, "fid": 0.0, "hal": 0.0, "delta": 0.0, "res": "-", 
    "img_orig": None, "img_rest": None, "img_heat": None, "procesar": False, "descargado": False
}
for k in CLAVES_TEXTO + CLAVES_ROSTRO:
    estados_iniciales[k] = False

for k, v in estados_iniciales.items():
    if k not in st.session_state: 
        st.session_state[k] = v

# ==========================================
# Interfaz: Telemetría Superior
# ==========================================
st.markdown(f"""
<div class="telemetry-row">
    <div class="metric-card"><span class="metric-label">🏷️ Clasificación Visual</span><span class="metric-value" style="color: #60a5fa;">{st.session_state.cat.upper()}</span></div>
    <div class="metric-card"><span class="metric-label">⏱️ Tiempo de Inferencia</span><span class="metric-value" style="color: #34d399;">{st.session_state.lat:.2f}s</span></div>
    <div class="metric-card"><span class="metric-label">🛡️ Fidelity Score</span><span class="metric-value" style="color: #38bdf8;">{st.session_state.fid:.3f}</span></div>
    <div class="metric-card"><span class="metric-label">⚠️ Tasa de Alucinación</span><span class="metric-value" style="color: #fbbf24;">{st.session_state.hal:.1f}%</span></div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# Interfaz: 3 Columnas Estructuradas
# ==========================================
col_izq, col_centro, col_der = st.columns([1.2, 1.3, 1.3], gap="large")

# COLUMNA 1: Controles 
with col_izq:
    with st.container():
        st.markdown('<div class="col-header">🕹️ Ingesta y Control</div>', unsafe_allow_html=True)
        archivo = st.file_uploader("Seleccionar imagen a restaurar", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
        
        if archivo:
            pil_raw = ImageOps.exif_transpose(Image.open(archivo)).convert("RGB")
            
            if st.button("⚡ Analizar Fotografía", key="btn_ejecutar_principal"):
                with st.spinner("Clasificando..."):
                    cat, lat = clasificar_semantica(pil_raw)
                    st.session_state.update({"img_orig": pil_raw, "cat": cat, "lat": lat, "procesar": True})
                    # Apagamos todos los toggles por si el usuario carga una nueva imagen
                    for k in CLAVES_TEXTO + CLAVES_ROSTRO:
                        st.session_state[k] = False
                    st.rerun()

        # Despliegue de Menús Dinámicos
        if st.session_state.procesar and st.session_state.img_orig is not None:
            sub_modo = None
            
            if st.session_state.cat == "texto":
                st.markdown('<div class="col-header" style="margin-top:24px; color:#60a5fa;">📄 Herramientas de Documento</div>', unsafe_allow_html=True)
                
                draw_tool_card("💡 Aclarar Fondo y Sombras", "Separa el texto del fondo, logrando legibilidad incluso con iluminación irregular.", "t_sau", CLAVES_TEXTO)
                draw_tool_card("✍️ Remarcar Letras Borrosas", "Rescata trazos finos empastados o gastados por el tiempo mediante morfología.", "t_top", CLAVES_TEXTO)
                draw_tool_card("📺 Quitar Rayas de Pantalla", "Elimina el efecto muaré y patrones ondulados generados por monitores.", "t_fft", CLAVES_TEXTO)
                
                if st.session_state.t_sau: sub_modo = "Iluminación Irregular (Sauvola)"
                elif st.session_state.t_top: sub_modo = "Trazos Desgastados (Top-Hat)"
                elif st.session_state.t_fft: sub_modo = "Eliminar Muaré de Pantalla (FFT 2D)"

            elif st.session_state.cat == "rostro":
                st.markdown('<div class="col-header" style="margin-top:24px; color:#f472b6;">👤 Herramientas de Retrato</div>', unsafe_allow_html=True)
                
                draw_tool_card("🔍 Mejorar Definición (HD)", "Amplía la captura mejorando definición sin afectar los colores reales.", "r_sr", CLAVES_ROSTRO)
                draw_tool_card("📳 Arreglar Foto Movida", "Corrige ligero desenfoque por movimiento de mano (trepidación).", "r_trep", CLAVES_ROSTRO)
                draw_tool_card("🌌 Limpiar Lluvia o Grano", "Elimina grano (ISO alto) nocturno sin que la piel parezca plástico.", "r_iso", CLAVES_ROSTRO)
                draw_tool_card("💡 Iluminar Foto Oscura", "Equilibra luz irregular y aclara sombras pronunciadas en la noche.", "r_noc", CLAVES_ROSTRO)
                draw_tool_card("✨ Suavizado Mágico Piel", "Suavizado estético que protege bordes críticos como los anteojos.", "r_piel", CLAVES_ROSTRO)
                
                if st.session_state.r_sr: sub_modo = "Super-Resolución (Solo YCbCr)"
                elif st.session_state.r_trep: sub_modo = "Trepidación (Deconvolución / Sharpening)"
                elif st.session_state.r_iso: sub_modo = "Ruido ISO Alto (No-Local Means)"
                elif st.session_state.r_noc: sub_modo = "Iluminación Nocturna (MSRCR / Retinex)"
                elif st.session_state.r_piel: sub_modo = "Suavizado de Piel (Filtro Guiado)"

            # Procesamiento Matemático Central
            img_np = np.array(st.session_state.img_orig)
            rest_np = restaurar_adaptativa(img_np, st.session_state.cat, sub_modo=sub_modo)
            heat_np, metrics = auditar_alucinaciones(img_np, rest_np)
            
            st.session_state.img_rest = Image.fromarray(rest_np)
            st.session_state.img_heat = Image.fromarray(heat_np)
            st.session_state.fid = metrics["fid"]
            st.session_state.hal = metrics["hal"]
            st.session_state.delta = metrics["delta"]
            st.session_state.res = metrics["res"]
            
            # --- Animación del Botón de Descarga ---
            buf = io.BytesIO()
            st.session_state.img_rest.save(buf, format="PNG")
            
            btn_container = st.empty()
            
            def marcar_descarga():
                st.session_state.descargado = True
                
            if st.session_state.descargado:
                with btn_container.container():
                    st.markdown('<div class="download-success">', unsafe_allow_html=True)
                    st.download_button(
                        label="Descargado", 
                        data=buf.getvalue(), 
                        file_name="restaurada.png", 
                        mime="image/png", 
                        key="btn_descarga_done"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                time.sleep(3)
                st.session_state.descargado = False
                st.rerun()
            else:
                with btn_container.container():
                    st.download_button(
                        label="📥 Descargar Imagen", 
                        data=buf.getvalue(), 
                        file_name="restaurada.png", 
                        mime="image/png", 
                        key="btn_descarga",
                        on_click=marcar_descarga
                    )
            
            # Tarjetas de Datos Debajo del Botón
            st.markdown(f'''
            <div class="col-header" style="margin-top:32px; font-size:0.75rem;">📊 Características del Procesamiento</div>
            <div class="side-card"><span class="side-title">Ganancia de Nitidez (Δ)</span><span class="side-val" style="color:#a78bfa;">{"+" if st.session_state.delta > 0 else ""}{st.session_state.delta:.1f}%</span></div>
            <div class="side-card"><span class="side-title">Resolución de Salida (2x)</span><span class="side-val" style="color:#f8fafc;">{st.session_state.res}</span></div>
            ''', unsafe_allow_html=True)

# COLUMNA 2: Comparador
with col_centro:
    with st.container():
        st.markdown('<div class="col-header">🖼️ Comparativa: Original vs Restaurada</div>', unsafe_allow_html=True)
        if st.session_state.img_rest and st.session_state.img_orig:
            w, h = st.session_state.img_rest.size
            scale = max(1.0, 520 / float(h))
            dw, dh = int(w * scale), int(h * scale)
            image_comparison(
                img1=st.session_state.img_orig.resize((dw, dh), Image.NEAREST),
                img2=st.session_state.img_rest.resize((dw, dh), Image.LANCZOS),
                label1="Original", label2="Restaurada",
                width=dw, starting_position=50, show_labels=True, make_responsive=True, in_memory=True
            )
        else:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-icon">🖼️</span>
                <p class="empty-text">Aquí aparecerá la cortina interactiva entre la imagen original y la restaurada.</p>
            </div>
            """, unsafe_allow_html=True)

# COLUMNA 3: Heatmap
with col_der:
    with st.container():
        st.markdown('<div class="col-header">🔬 Auditoría de Gradientes (Mapa JET)</div>', unsafe_allow_html=True)
        if st.session_state.img_heat:
            st.image(st.session_state.img_heat, caption="Rojo/Amarillo = Modificaciones | Azul = Fidelidad Pura", width="stretch")
        else:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-icon">🔬</span>
                <p class="empty-text">Aquí se proyectará el mapa de calor con los gradientes residuales de Sobel.</p>
            </div>
            """, unsafe_allow_html=True)