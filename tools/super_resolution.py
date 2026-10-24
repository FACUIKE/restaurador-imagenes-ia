from pathlib import Path
import cv2
import torch
from tools.base_tool import BaseRestorationTool
from config import settings

class SuperResolutionTool(BaseRestorationTool):
    """
    Super-resolución neuronal local mediante Real-ESRGAN (x2 o x4).
    """
    def __init__(self, scale: int = 2):
        super().__init__(name="super_resolution")
        self.scale = scale
        self.upsampler = None
        self._load_model()  # Carga el modelo en la VRAM al arrancar el servidor

    def _load_model(self):
        if self.upsampler is None:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale)
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.upsampler = RealESRGANer(
                scale=self.scale,
                model_path=model_url,
                model=model,
                tile=400,  # Procesa por bloques para no sobrepasar la memoria de la GPU
                tile_pad=10,
                pre_pad=0,
                half=True if device == "cuda" else False, # FP16 para máxima velocidad
                device=device
            )

    def process(self, input_path: Path) -> Path:
        self._load_model()
        output_path = settings.OUTPUTS_DIR / f"step_super_res_{input_path.name}"

        img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        output, _ = self.upsampler.enhance(img, outscale=self.scale)

        cv2.imwrite(str(output_path), output)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_path