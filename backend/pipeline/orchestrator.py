import logging
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List

from backend.agent.schemas import RestorationDecision
from backend.config import settings
from tools import (
    DenoiseTool, 
    SuperResolutionTool, 
    FaceRestorationTool, 
    ColorizerTool, 
    NAFNetDeblurTool,
    TextSharpenTool
)

logger = logging.getLogger(__name__)

class RestorationPipeline:
    """
    Orquestador con enrutamiento inteligente según el tipo de imagen:
    - Ruta 'texto': Limpieza suave de artefactos JPEG + RealESRGAN x2 fiel a tipografías.
    - Ruta 'rostro': NAFNet (mejora cuerpo/ropa) + Colorizer (opcional) + GFPGAN (rostro).
    - Ruta 'paisaje': Denoise + NAFNet + SuperResolution general.
    """

    def __init__(self):
        self.tools = {
            "denoise": DenoiseTool(),
            "colorize": ColorizerTool(),
            "deblur": NAFNetDeblurTool(),
            "text_sharpen": TextSharpenTool(),
            "face_restoration": FaceRestorationTool(),
            "super_resolution": SuperResolutionTool()
        }

    def _select_route(self, decisions: RestorationDecision) -> List[str]:
        """Determina la secuencia exacta de herramientas sin mezclar filtros incompatibles."""
        
        # Inferencia de respaldo por si el modelo omitió la categoría explícita
        categoria = getattr(decisions, "categoria", "paisaje")
        if decisions.restaurar_rostros:
            categoria = "rostro"

        steps = []

        if categoria == "texto":
            # RUTA LETRAS / DOCUMENTOS: Prohibido usar denoise agresivo o GFPGAN
            steps.append("text_sharpen")

        elif categoria == "rostro":
            # RUTA PERSONAS / RETRATOS: Mejorar cuerpo primero y luego rostro
            if decisions.reducir_ruido:
                steps.append("denoise")
            steps.append("deblur")  # Enfoca campera, pelo y fondo
            if decisions.colorizar:
                steps.append("colorize")
            steps.append("face_restoration")  # Reconstruye rostros + bg_upsampler

        else:
            # RUTA PAISAJES / OBJETOS GENERALES
            if decisions.reducir_ruido:
                steps.append("denoise")
            steps.append("deblur")
            if decisions.colorizar:
                steps.append("colorize")
            if decisions.super_resolucion:
                steps.append("super_resolution")

        return steps

    def run(self, image_path: Path, decisions: RestorationDecision) -> Dict[str, Any]:
        start_time = time.time()
        current_image_path = Path(image_path)
        applied_steps = []

        settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        output_filename = f"restored_{timestamp}_{current_image_path.name}"
        final_output_path = settings.OUTPUTS_DIR / output_filename

        # Obtener la ruta específica según el tipo de imagen
        route_steps = self._select_route(decisions)
        logger.info(f"Categoría detectada: {getattr(decisions, 'categoria', 'auto')} -> Ruta: {route_steps}")

        for tool_name in route_steps:
            logger.info(f"-> Ejecutando herramienta: {tool_name}")
            tool_instance = self.tools[tool_name]
            
            next_image_path = tool_instance.process(current_image_path)
            
            if next_image_path and Path(next_image_path).exists():
                current_image_path = Path(next_image_path)
                applied_steps.append(tool_name)
            else:
                logger.warning(f"La herramienta {tool_name} no generó {next_image_path}. Manteniendo paso previo.")

        shutil.copy(str(current_image_path), str(final_output_path))
        total_duration = round(time.time() - start_time, 2)

        return {
            "status": "success",
            "input_path": str(image_path),
            "output_path": str(final_output_path),
            "applied_steps": applied_steps,
            "diagnostic": f"[{getattr(decisions, 'categoria', 'general').upper()}] {decisions.motivo_diagnostico}",
            "duration_seconds": total_duration
        }