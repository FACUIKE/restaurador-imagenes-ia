import logging
import shutil
from pathlib import Path
from tools.base_tool import BaseRestorationTool
from backend.config import settings

logger = logging.getLogger(__name__)

class ColorizerTool(BaseRestorationTool):
    """
    Herramienta para colorear fotografías monocromáticas o en sepia (DeOldify).
    """

    def __init__(self, render_factor: int = 35):
        super().__init__(name="colorizer")
        self.render_factor = render_factor

    def process(self, input_path: Path) -> Path:
        input_path = Path(input_path)
        output_path = settings.OUTPUTS_DIR / f"step_colorized_{input_path.name}"

        # 1. Asegurar que la carpeta de destino exista
        settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        # 2. Si no existe el archivo de entrada, no podemos continuar
        if not input_path.exists():
            raise FileNotFoundError(f"El archivo de entrada no existe: {input_path}")

        # 3. Intentar procesar con Replicate solo si el token está presente
        token = getattr(settings, "REPLICATE_API_TOKEN", "")
        if settings.TOOLS_BACKEND == "replicate" and token:
            try:
                import replicate
                import urllib.request

                output = replicate.run(
                    "ariel415el/deoldify:0da600582269d0364c12574c3e80ff369ff700586e2469446f7e449a0d8ff06c",
                    input={"image": open(input_path, "rb"), "render_factor": self.render_factor}
                )
                urllib.request.urlretrieve(str(output), str(output_path))
                if output_path.exists():
                    return output_path
            except Exception as e:
                logger.warning(f"Fallo en Replicate Colorizer: {e}. Pasando a copia directa.")

        # 4. Fallback seguro: copiar el archivo original al destino
        shutil.copyfile(str(input_path), str(output_path))
        return output_path