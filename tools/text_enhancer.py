import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from tools.base_tool import BaseRestorationTool
from config import settings

# Parche de compatibilidad torchvision / basicsr
try:
    import torchvision.transforms.functional as F
    import torchvision.transforms as transforms
    transforms.functional_tensor = F
    sys.modules["torchvision.transforms.functional_tensor"] = F
except Exception:
    pass


class TextSharpenTool(BaseRestorationTool):
    """
    Reconstruye y escala texto de baja resolución preservando la geometría
    de las letras y el anti-aliasing natural sin deformar ni rellenar huecos.
    """
    def __init__(self):
        super().__init__(name="text_sharpen")
        self.upsampler = None

    def _load_model(self):
        if self.upsampler is None:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            device = "cuda" if torch.cuda.is_available() else "cpu"
            use_half = True if device == "cuda" else False

            # Usamos RealESRGAN_x2plus (23 bloques): respeta fielmente la geometría
            # tipográfica sin inventar trazos de dibujo animado ni derretir letras pequeñas.
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2
            )
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

            self.upsampler = RealESRGANer(
                scale=2,
                model_path=model_url,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=use_half,
                device=device
            )

    def process(self, input_path: Path) -> Path:
        self._load_model()
        output_path = settings.OUTPUTS_DIR / f"step_text_{input_path.name}"

        img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No se pudo abrir {input_path}")

        # 1. Filtro bilateral ligero: limpia la suciedad JPEG alrededor de las letras
        # sin borrar ni engrosar los bordes finos de las filas inferiores.
        clean_img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)

        # 2. Super-resolución neuronal fiel (x2)
        enhanced, _ = self.upsampler.enhance(clean_img, outscale=2)

        # 3. Micro-contraste sutil (Unsharp Mask muy suave en alta resolución)
        # Solo define el borde sin rellenar los centros de las letras O, P, D, E.
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
        final_img = cv2.addWeighted(enhanced, 1.3, gaussian, -0.3, 0)

        cv2.imwrite(str(output_path), final_img)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_path