import sys
from pathlib import Path
import cv2
import torch

# Parche de compatibilidad torchvision / basicsr
try:
    import torchvision.transforms.functional as F
    import torchvision.transforms as transforms
    transforms.functional_tensor = F
    sys.modules["torchvision.transforms.functional_tensor"] = F
except Exception:
    pass

from tools.base_tool import BaseRestorationTool
from backend.config import settings

class FaceRestorationTool(BaseRestorationTool):
    """
    Restauración integral: mejora la escena completa con Real-ESRGAN
    y reconstruye rostros con GFPGAN.
    """
    def __init__(self):
        super().__init__(name="face_restoration")
        self.restorer = None
        self._load_model()  # Carga el modelo en la VRAM al arrancar el servidor

    def _load_model(self):
        """Carga los modelos en memoria solo una vez."""
        if self.restorer is None:
            from gfpgan import GFPGANer
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            device = "cuda" if torch.cuda.is_available() else "cpu"
            use_half = True if device == "cuda" else False

            # 1. Configuramos Real-ESRGAN para cuerpo, ropa y fondo
            bg_model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, 
                num_block=23, num_grow_ch=32, scale=2
            )
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                model=bg_model,
                tile=400,        # Mantiene el consumo de VRAM bajo control
                tile_pad=10,
                pre_pad=0,
                half=use_half,
                device=device
            )

            # 2. Inicializamos GFPGAN vinculando el bg_upsampler
            self.restorer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=bg_upsampler,  # Activa la mejora de toda la imagen
                device=device
            )

    def process(self, input_path: Path) -> Path:
        """Aplica la restauración sobre la imagen."""
        self._load_model()
        output_path = settings.OUTPUTS_DIR / f"step_face_{input_path.name}"
        
        img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No se pudo abrir {input_path}")

        # Inferencia conjunta: weight entre 0.5 y 0.7 evita el efecto de 'cara pegada'
        _, _, restored_img = self.restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.55  # 0.55 permite que la textura natural del rostro coincida con la textura recuperada por NAFNet
        )

        cv2.imwrite(str(output_path), restored_img)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_path