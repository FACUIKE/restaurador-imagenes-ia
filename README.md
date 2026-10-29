🖼️ VisionRestore AI - Plataforma de Restauración PDI

VisionRestore AI es una aplicación web interactiva y 100% offline diseñada para la restauración matemática de imágenes digitales (documentos y retratos).

Fue desarrollada como proyecto de tesis para la Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial. Su mayor innovación es que no requiere internet ni tarjetas gráficas costosas (GPU). Utiliza Procesamiento Digital de Imágenes (PDI) clásico optimizado, lo que permite que funcione con fluidez incluso en notebooks de bajos recursos (ej. procesadores Celeron con 8GB de RAM).

🚀 Características Principales

100% Privado y Local: Las fotos nunca salen de tu computadora. No se usan APIs externas ni servicios en la nube.

Optimización Extrema: Comprime las imágenes internamente a formato WebP y reduce su tamaño para vistas previas instantáneas, aplicando la calidad original solo al momento de descargar.

Auditoría de Alucinaciones: Cuenta con un "Mapa JET" que compara los gradientes (bordes) matemáticos de la foto original y la restaurada, demostrando que la aplicación no inventa ni alucina detalles (crítico para uso forense o documental).

Modo Acumulativo: Permite encadenar múltiples filtros (ej. iluminar + remarcar texto + quitar ruido) de forma visual.

📁 Estructura del Proyecto

El proyecto sigue una estricta Arquitectura Limpia (Clean Architecture), separando la interfaz visual (Frontend) del motor matemático (Backend).

restaurador-imagenes-ia/
├── frontend/                   # Interfaz visual de usuario (Lo que ves en el navegador)
│   ├── index.html              # Estructura de la página web
│   ├── style.css               # Diseño, colores y animaciones
│   └── app.js                  # Lógica interactiva y conexión con el motor
│
├── backend/                    # Motor principal y servidor (El "cerebro" oculto)
│   ├── requirements.txt        # Lista de librerías necesarias para funcionar
│   ├── app/                    # El servidor web
│   │   ├── main.py             # Archivo que enciende el servidor (FastAPI)
│   │   ├── schemas.py          # Reglas de cómo viajan los datos
│   │   └── routers/process.py  # Enrutador que recibe las órdenes del frontend
│   │
│   ├── vision/                 # Núcleo de Matemática y Procesamiento
│   │   ├── imageio.py          # Lee y convierte imágenes a matrices de datos
│   │   └── filters.py          # Fórmulas de OpenCV (Sauvola, Retinex, CLAHE, etc.)
│   │
│   ├── experiments/            # (Fines Académicos) Registros del método científico
│   ├── tests/                  # (Fines Académicos) Pruebas automatizadas de calidad
│   ├── tools/                  # (Fines Académicos) Herramientas modulares extra
│   └── pipeline/               # (Fines Académicos) Tuberías de automatización
│
└── README.md                   # Este manual de instrucciones


🔍 ¿Por qué hay carpetas que no se usan directamente en el programa?

Notarás que dentro de backend/ hay carpetas como experiments/, tests/, tools/ y pipeline/ que no son llamadas por la aplicación principal al restaurar una foto.
Estas carpetas se mantienen por el rigor académico de la tesis:

experiments/: Contiene los scripts de validación (benchmarks) donde se comprobó qué fórmulas matemáticas eran más rápidas o eficientes.

tests/: Contiene pruebas de código (pytest). Sirve para que los programadores comprueben que las fórmulas no fallan al hacer modificaciones.

tools/ y pipeline/: Son carpetas de "arquitectura expansible". Si en el futuro el proyecto crece y se vuelve automático en lugar de manual, el código se organizaría aquí usando Programación Orientada a Objetos. Mantenerlas demuestra madurez en la Ingeniería de Software.

💻 Guía de Instalación (Para principiantes)

Sigue estos pasos al pie de la letra para usar el programa en tu computadora con Windows, incluso si no sabes programar.

Requisitos Previos

Descarga e instala Python desde la página oficial (marca la casilla "Add Python to PATH" durante la instalación).

Opcional pero recomendado: Descarga un editor de texto como Visual Studio Code (VS Code).

Paso 1: Preparar el Motor (Backend)

Abre tu carpeta restaurador-imagenes-ia en Visual Studio Code (o abre una terminal en esa carpeta).

Ve a la barra superior, haz clic en Terminal -> New Terminal.

Escribe los siguientes comandos uno por uno y presiona Enter después de cada uno:

# Entrar a la carpeta del motor
cd backend

# Crear un "entorno virtual" (una burbuja segura para instalar cosas)
python -m venv venv

# Activar la burbuja (En Windows)
.\venv\Scripts\activate

# Instalar los motores matemáticos (OpenCV, FastAPI, etc.)
pip install -r requirements.txt


(Nota: Solo debes hacer este Paso 1 la primera vez que instalas el programa).

Paso 2: Encender el Servidor

Cada vez que quieras usar la aplicación, debes encender el motor. En la misma terminal (con el (venv) verde activado), escribe:

uvicorn app.main:app --reload


Verás un mensaje que dice Uvicorn running on http://127.0.0.1:8000. ¡No cierres esta ventana negra! Déjala minimizada; es el cerebro trabajando.

Paso 3: Abrir la Aplicación Visual

Ve a la carpeta frontend/ en tu computadora (usando el Explorador de Archivos de Windows normal).

Haz doble clic en el archivo index.html.

¡Listo! Se abrirá en tu navegador web (Chrome, Edge, etc.) y podrás empezar a usarla.

🎨 ¿Cómo usar la aplicación?

Subir Foto: Arrastra una imagen o haz clic en "Upload".

Seleccionar Categoría: Haz clic en 📄 Documento (si es un ticket, libro o apunte) o en 👤 Retrato (si es una foto de personas o paisajes). Esto desplegará las herramientas correctas.

Aplicar Filtros: Enciende los interruptores de la izquierda. El programa procesará una versión rápida para que veas el resultado al instante sin trabar tu PC. Puedes usar la barra deslizante central para comparar el "Antes y Después".

Auditar (Opcional): Si quieres comprobar científicamente los cambios, haz clic en "Auditar Gradientes".

Descargar: Haz clic en el botón de descarga para guardar tu foto restaurada en alta resolución.