import base64
import sys
from pathlib import Path
import requests

# Añadir la raíz del proyecto (un nivel arriba de frontend/) a sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Importaciones de Streamlit y librerías base
import streamlit as st
from PIL import Image
import time
from streamlit_image_comparison import image_comparison

from config import settings
from agent.decision_agent import DecisionAgent
from pipeline.orchestrator import RestorationPipeline


# Configuración visual de la página
st.set_page_config(
    page_title="Restaurador de Imágenes con Agente IA",
    page_icon="🖼️",
    layout="wide"
)

# Inicializar sesión y estados
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "decisions" not in st.session_state:
    st.session_state.decisions = None
if "result_pipeline" not in st.session_state:
    st.session_state.result_pipeline = None
if "input_path" not in st.session_state:
    st.session_state.input_path = None

st.title("Restaurador Inteligente de Imágenes con Agente IA")
st.markdown(
    """
    Sube una fotografía antigua, un documento o una imagen borrosa. El **Agente Decisor** 
    inspeccionará visualmente la naturaleza de la imagen y activará las rutas especializadas de restauración.
    """
)

# Barra lateral: Configuración y estado del sistema
with st.sidebar:
    st.header("⚙️ Configuración del Sistema")
    st.info(f"**Proveedor del Agente:** `{settings.AGENT_PROVIDER}`")
    
    if settings.AGENT_PROVIDER == "ollama":
        st.write(f"**Modelo Local:** `{settings.OLLAMA_MODEL}`")
        st.write(f"**Host:** `{settings.OLLAMA_BASE_URL}`")
    else:
        st.write(f"**Modelo en Nube:** `{settings.GEMINI_MODEL}`")
    
    st.divider()
    st.write(f"**Backend de Herramientas:** `{settings.TOOLS_BACKEND}`")

    st.divider()
    st.subheader("🔌 Orquestación externa")
    usar_n8n = st.checkbox("Procesar mediante n8n (Webhook)", value=False, help="Requiere tener activo api_server.py y n8n local.")

# Contenedor principal: Carga y Procesamiento
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Subir Imagen Original")
    uploaded_file = st.file_uploader(
        "Selecciona un archivo JPG, PNG o JPEG", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Guardar archivo en storage/uploads y en el session_state
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        input_path = settings.UPLOADS_DIR / uploaded_file.name
        st.session_state.input_path = input_path
        
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image = Image.open(input_path)
        st.image(image, caption="Fotografía Original", use_container_width=True)

        # Botón para iniciar el análisis
        if st.button("🔍 Analizar y Restaurar Imagen", type="primary", use_container_width=True):
            
            if usar_n8n:
                # MODO N8N (Envío seguro con Base64 desde Python)
                with st.spinner("🚀 Codificando imagen y enviando a n8n..."):
                    try:
                        # Leer y convertir la imagen a Base64 en Python para evitar bloqueos de disco en n8n
                        with open(input_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

                        n8n_url = "http://127.0.0.1:5678/webhook/restaurar-imagen"
                        payload = {
                            "image_path": str(input_path),
                            "base64": encoded_string
                        }
                        
                        response = requests.post(
                            n8n_url,
                            json=payload,
                            timeout=300
                        )
                        response.raise_for_status()
                        res_json = response.json()
                        # Si n8n devuelve una lista [ {...} ], tomamos el primer elemento
                        if isinstance(res_json, list) and len(res_json) > 0:    
                            res_json = res_json[0]

                        # Adaptar la respuesta de n8n al formato del session_state
                        st.session_state.result_pipeline = {
                            "status": res_json.get("status"),
                            "output_path": res_json.get("output_path"),
                            "applied_steps": res_json.get("applied_steps", []),
                            "duration_seconds": res_json.get("duration_seconds", 0.0)
                        }
                        
                        # Crear un objeto de decisiones simulado para la interfaz
                        class DummyDecision:
                            def __init__(self, diag, cat):
                                self.motivo_diagnostico = diag
                                self.categoria = cat
                                self.reducir_ruido = "denoise" in res_json.get("applied_steps", [])
                                self.colorizar = "colorize" in res_json.get("applied_steps", [])
                                self.restaurar_rostros = "face_restoration" in res_json.get("applied_steps", [])
                                self.super_resolucion = True

                        st.session_state.decisions = DummyDecision(
                            res_json.get("diagnostic", "Procesado vía n8n"),
                            res_json.get("categoria", "general")
                        )
                        st.session_state.analyzed = True

                    except Exception as e:
                        st.error(f"Error al conectar con n8n o la API local (¿Están encendidos api_server.py y n8n?): {e}")
                        st.stop()
            else:
                # MODO LOCAL PYTHON PURO (Con enrutador inteligente)
                agent = DecisionAgent()
                pipeline = RestorationPipeline()

                with st.spinner("🧠 El Agente Decisor está inspeccionando la fotografía..."):
                    try:
                        decisions = agent.analyze_image(input_path)
                        st.session_state.decisions = decisions
                        st.session_state.analyzed = True
                    except Exception as e:
                        st.error(f"Error al analizar con el agente: {e}")
                        st.stop()

                with st.spinner("⚙️ Ejecutando pipeline especializado..."):
                    try:
                        result = pipeline.run(input_path, decisions)
                        st.session_state.result_pipeline = result
                    except Exception as e:
                        st.error(f"Error durante el procesamiento del pipeline: {e}")
                        st.stop()

with col_right:
    st.subheader("2. Veredicto del Agente y Resultado")

    if st.session_state.analyzed and st.session_state.decisions is not None:
        decisions = st.session_state.decisions
        result = st.session_state.result_pipeline

        # Diagnóstico del Agente
        st.success("✅ Diagnóstico y procesamiento completados")
        categoria_actual = getattr(decisions, "categoria", "general").upper()
        st.markdown(f"**Categoría Detectada:** `{categoria_actual}`")
        st.markdown(f"**Diagnóstico técnico:** *{decisions.motivo_diagnostico}*")

        # Visualización de herramientas seleccionadas
        st.markdown("**Pasos ejecutados en la ruta:**")
        if result and "applied_steps" in result:
            steps_str = ", ".join(result["applied_steps"]) if result["applied_steps"] else "Ninguno"
            st.code(steps_str, language="text")

        st.divider()

        # Mostrar resultado final con comparador interactivo
        if result and result.get("output_path") and Path(result["output_path"]).exists():
            st.markdown(f"**Resultado Final (Procesado en {result['duration_seconds']}s)**")
            
            # Cargar la imagen original y la restaurada
            img_original = Image.open(st.session_state.input_path)
            img_restaurada = Image.open(result["output_path"])
            
            # Renderizar el componente comparador
            image_comparison(
                img1=img_original,
                img2=img_restaurada,
                label1="Original",
                label2="Restaurada ✨",
                width=700,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )

            # Botón de descarga debajo del comparador
            with open(result["output_path"], "rb") as f:
                st.download_button(
                    label="📥 Descargar Imagen Restaurada",
                    data=f.read(),
                    file_name=Path(result["output_path"]).name,
                    mime="image/png",
                    use_container_width=True
                )
        else:
            st.warning(f"⚠️ No se generó la imagen. Respuesta recibida de n8n o Flask: {result}")
    else:
        st.info("Sube una imagen y presiona 'Analizar y Restaurar Imagen' para ver la clasificación del agente y el resultado.")