import cv2
from pathlib import Path
from tools.base_tool import BaseRestorationTool
from backend.config import settings

class DenoiseTool(BaseRestorationTool):
    """
    Herramienta para reducción de grano y ruido digital.
    Implementación local ligera y eficiente con OpenCV (Fast Non-Local Means).
    """

    def __init__(self, h_luminance: float = 10, h_color: float = 10):
        super().__init__(name="denoise")
        self.h_luminance = h_luminance
        self.h_color = h_color

    def process(self, input_path: Path) -> Path:
        img = cv2.imread(str(input_path))
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {input_path}")

        # Reducción de ruido no local preservando bordes
        denoised = cv2.fastNlMeansDenoisingColored(
            img,
            None,
            h=self.h_luminance,
            hColor=self.h_color,
            templateWindowSize=7,
            searchWindowSize=21
        )

        output_path = settings.OUTPUTS_DIR / f"step_denoise_{input_path.name}"
        cv2.imwrite(str(output_path), denoised)
        return output_path