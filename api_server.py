import time
import shutil
import base64
import os
import torch
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from pathlib import Path
from tools.classifier import ImageClassifierTool

# Optimizaciones de rendimiento para la RTX 3050
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from config import settings
from tools import (
    DenoiseTool,
    SuperResolutionTool,
    FaceRestorationTool,
    ColorizerTool,
    NAFNetDeblurTool,
    TextSharpenTool
)



# Inicializamos el clasificador para que viva en la VRAM de la RTX 3050
vision_classifier = ImageClassifierTool()

app = Flask(__name__)

# Diccionario de herramientas cargadas bajo demanda
TOOLS = {
    "denoise": DenoiseTool(),
    "colorize": ColorizerTool(),
    "deblur": NAFNetDeblurTool(),
    "text_sharpen": TextSharpenTool(),
    "face_restoration": FaceRestorationTool(),
    "super_resolution": SuperResolutionTool()
}

@app.route('/encode-image', methods=['POST'])
def encode_image():
    data = request.get_json(force=True, silent=True) or {}
    image_path = data.get("image_path")

    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": f"No se encontró la imagen en: {image_path}"}), 400

    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode('utf-8')

    return jsonify({
        "status": "ok",
        "image_path": image_path,
        "base64": b64_string
    })

@app.route('/classify-image', methods=['POST'])
def classify_image():
    try:
        data = request.get_json()
        base64_str = data.get("base64_image")
        image_path = data.get("image_path") # Solo lo rebotamos, no lo usamos
        
        if not base64_str:
            return jsonify({"error": "No se recibió la imagen en base64"}), 400
            
        # Decodificamos el Base64 directamente en la memoria RAM
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Clasificamos la imagen sin tocar el disco duro
        categoria = vision_classifier.classify(image)
        
        return jsonify({
            "categoria": categoria,
            "image_path": image_path  # Lo devolvemos para que el Switch de n8n funcione
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/execute-tools", methods=["POST"])
def execute_tools():
    """Ejecuta la secuencia exacta de herramientas decidida por el Switch de n8n."""
    start_time = time.time()
    data = request.json
    image_path = Path(data.get("image_path", ""))
    steps = data.get("steps", [])
    categoria = data.get("categoria", "general")
    diagnostico = data.get("motivo_diagnostico", "Procesado vía n8n")

    if not image_path.exists():
        return jsonify({"status": "error", "message": f"Archivo no encontrado: {image_path}"}), 404

    settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    current_path = image_path
    applied_steps = []

    for step_name in steps:
        if step_name in TOOLS:
            next_path = TOOLS[step_name].process(current_path)
            if next_path and Path(next_path).exists():
                current_path = Path(next_path)
                applied_steps.append(step_name)

    timestamp = int(time.time())
    final_output = settings.OUTPUTS_DIR / f"restored_{timestamp}_{image_path.name}"
    shutil.copy(str(current_path), str(final_output))

    return jsonify({
        "status": "success",
        "categoria": categoria,
        "input_path": str(image_path),
        "output_path": str(final_output),
        "applied_steps": applied_steps,
        "diagnostic": diagnostico,
        "duration_seconds": round(time.time() - start_time, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)