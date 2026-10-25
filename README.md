# Agente IA de Restauración Automática de Imágenes

Aplicación basada en una **Arquitectura de Agente con Uso de Herramientas (Tool Calling / Function Calling)** para la restauración inteligente y automática de fotografías dañadas o antiguas.

El agente analiza visualmente la imagen mediante un Modelo de Lenguaje Multimodal (VLM), diagnostica sus defectos y activa de forma secuencial únicamente las herramientas de visión computacional necesarias.

---

## 🏛 Arquitectura del Sistema

El proyecto está dividido en tres capas principales:

1. **Frontend (Interfaz de Usuario):**
   - Construida con **Streamlit** para permitir la carga de imágenes, previsualización de diagnóstico del agente y descarga del resultado final.

2. **Agente Decisor (Cerebro):**
   - Basado en **Google Gemini** con capacidades de visión.
   - Evalúa defectos (falta de nitidez, ruido de sensor, decoloración, daños en rostros).
   - Genera una orden estructurada en JSON (`true`/`false`) para activar o descartar herramientas.

3. **Herramientas de Procesamiento (Músculo Técnico):**
   - Modelos especializados de visión computacional consumidos vía API o contenedores:
     - **Real-ESRGAN:** Aumento de resolución y superresolución.
     - **GFPGAN / CodeFormer:** Restauración y reconstrucción de rostros.
     - **DeOldify:** Colorización de fotografías en blanco y negro o sepia.
     - **Denoise Filters:** Reducción de grano y ruido digital.

---

## 📁 Estructura del Repositorio

```text
restaurador-imagenes-ia/
├── config/             # Configuración del sistema y prompts
├── frontend/           # Interfaz web con Streamlit
├── agent/              # Lógica y esquemas del agente decisor
├── tools/              # Integraciones con modelos de restauración
├── pipeline/           # Orquestador del flujo de herramientas
├── storage/            # Almacenamiento temporal de imágenes
├── .env.example        # Plantilla de variables de entorno
├── requirements.txt    # Dependencias de Python
└── README.md           # Documentación del proyecto
```

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el repositorio y crear el entorno virtual
```bash
python -m venv venv
# En Linux/macOS:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

python.exe -m pip install --upgrade pip

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno
Copia el archivo de ejemplo y completa tus credenciales:
```bash
cp .env.example .env
```

Define en `.env`:
- `GEMINI_API_KEY`: Clave de Google AI Studio.
- `REPLICATE_API_TOKEN`: Token de Replicate (para los modelos de restauración).

### 4. Ejecutar la aplicación
```bash
streamlit run frontend/app.py
```

tesis-experimentos/
│
├── Punto 1: Enrutador Semántico (VLM vs CNN Preentrenada)  
├── Punto 2: Benchmark y Correlación de Métricas (PSNR, SSIM, LPIPS, NIQE)
└── Punto 3: Detección y Cuantificación de Alucinaciones