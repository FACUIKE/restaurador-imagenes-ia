Materia: Técnicas de Procesamiento Digital de Imágenes
Año y cuatrimestre de cursada: 3 año segundo cuatrimestre
Su nombre y apellido: IKEMIYA FACUNDO 
Docente: BONINI JUAN
Que los motivo a crear lo que crearon:


Que problemáticas encontraron y como las resolvieron:
1 - Rama incorrecta en Vercel: Vercel intentaba desplegar la rama main por defecto en lugar de la rama donde tenías el vercel.json y los cambios. Solución: Cambie la configuración para que tomara tu rama de trabajo y lograste el primer despliegue exitoso.
2 - Error de conexión del frontend (127.0.0.1:8000): Al probar la app en la nube, los botones no hacían nada porque el navegador seguía buscando el servidor en tu PC local. Solución: Cambie las rutas en frontend/app.js a /api/process, subi el cambio a GitHub y actualice el despliegue.
3 - Error 500 en el backend (No module named 'app'): Vercel ejecuta el código desde la carpeta raíz y Python no encontraba las carpetas internas de backend/. Solución: Agregue las líneas de sys.path.append al inicio de backend/app/main.py para indicarle a Vercel la ruta exacta de los módulos.
4 - Despliegue atrapado en "Preview": La última actualización con el arreglo de main.py quedó en modo vista previa en lugar de reemplazar la versión principal. Solución: La promovi manualmente a Production desde el menú de los tres puntitos en Vercel.
5 - Se quito el uso de n8n y la IA
    A - Sin IA generativa (ni GPU): Se optó por Procesamiento Digital de Imágenes (PDI) clásico optimizado en lugar de modelos pesados de IA para que la aplicación no requiera internet ni tarjeta gráfica, logrando que funcione fluida incluso en notebooks de bajos recursos. Además, el enfoque matemático garantiza que el programa no invente ni "alucine" detalles, lo cual es crítico para restaurar documentos.
    B - Sin n8n ni servicios externos: Se descartaron orquestadores o APIs en la nube para que el sistema sea 100% privado y local, asegurando que las fotos nunca salgan de la computadora del usuario. Como la herramienta es interactiva (el usuario prende y apaga filtros al instante), conectar el frontend directamente a FastAPI evita capas innecesarias y reduce la latencia.
6 - Por qué están las carpetas que no se usan directamente, aunque la aplicación principal no las llama al procesar una foto,se amantienen por buenas prácticas y para poder escalar el proyecto a futuro.
    A - experiments/: Guarda los scripts de validación (benchmarks) donde se comprobó qué fórmulas matemáticas eran más rápidas y eficientes.
    B - tests/: Contiene las pruebas automatizadas (pytest) que sirven para verificar que las fórmulas no fallen al hacer modificaciones en el código.
    C - tools/ y pipeline/: Funcionan como una base de arquitectura expansible; si más adelante el proyecto crece y pasa a ser automático en lugar de manual, el código nuevo se organizaría ahí usando Programación Orientada a Objetos.

🖼️ VisionRestore AI - Plataforma de Restauración 

VisionRestore AI es una aplicación web interactiva y 100% offline diseñada para la restauración matemática de imágenes digitales (documentos y retratos).

Fue desarrollada como proyecto para la Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial. Su mayor innovación es que no requiere internet ni tarjetas gráficas (GPU). Utiliza Procesamiento Digital de Imágenes (PDI) clásico optimizado, lo que permite que funcione con fluidez incluso en notebooks de bajos recursos.

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
├── README.md                   # Este manual de instrucciones
└── Manual_README.html          # Manual web para su uso


🔍 ¿Por qué hay carpetas que no se usan directamente en el programa?

Notarás que dentro de backend/ hay carpetas como experiments/, tests/, tools/ y pipeline/ que no son llamadas por la aplicación principal al restaurar una foto.
Estas carpetas se mantienen para poder expandir las herramientas en un futuro:

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

# Activar la burbuja (En Linux)
source venv/bin/activate

# Instalar los motores matemáticos (OpenCV, FastAPI, etc.)
pip install -r requirements.txt


(Nota: Solo debes hacer este Paso 1 la primera vez que instalas el programa).

Paso 2: Encender el Servidor

Cada vez que quieras usar la aplicación, debes encender el motor. En la misma terminal (con el (venv) verde activado), escribe:

uvicorn app.main:app --reload


Verás un mensaje que dice Uvicorn running on http://127.0.0.1:8000. ¡No cierres esta ventana negra! Déjala minimizada; es el cerebro trabajando.

Paso 3: Abrir la Aplicación Visual

Abre una segunda terminal y entra en el frontend

cd frontend

Ejecuta el siguiente comando

python -m http.server 3000

Ingresa el siguiente link http://localhost:3000 en el navegador

¡Listo! Se abrirá en tu navegador web (Chrome, Edge, etc.) y podrás empezar a usarla.

🎨 ¿Cómo usar la aplicación?

Subir Foto: Arrastra una imagen o haz clic en "Upload".

Seleccionar Categoría: Haz clic en 📄 Documento (si es un ticket, libro o apunte) o en 👤 Retrato (si es una foto de personas o paisajes). Esto desplegará las herramientas correctas.

Aplicar Filtros: Enciende los interruptores de la izquierda. El programa procesará una versión rápida para que veas el resultado al instante sin trabar tu PC. Puedes usar la barra deslizante central para comparar el "Antes y Después".

Auditar (Opcional): Si quieres comprobar científicamente los cambios, haz clic en "Auditar Gradientes".

Descargar: Haz clic en el botón de descarga para guardar tu foto restaurada en alta resolución.