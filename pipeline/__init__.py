from tools import (
    DenoiseTool, 
    SuperResolutionTool, 
    FaceRestorationTool, 
    ColorizerTool, 
    NAFNetDeblurTool
)

class RestorationPipeline:
    def __init__(self):
        self.tools = {
            "denoise": DenoiseTool(),
            "colorize": ColorizerTool(),
            "deblur": NAFNetDeblurTool(),
            "face_restoration": FaceRestorationTool(),
            "super_resolution": SuperResolutionTool()
        }

        # Secuencia ordenada de ejecución:
        # 1. Denoise (elimina ruido digital)
        # 2. Deblur con NAFNet (enfoca cuerpo, campera, pelo y fondo)
        # 3. Colorize (si corresponde)
        # 4. Face Restoration (perfecciona el rostro ya integrado)
        self.execution_order = [
            ("reducir_ruido", "denoise"),
            ("super_resolucion", "deblur"),             # NAFNet corrige el movimiento general
            ("colorizar", "colorize"),
            ("restaurar_rostros", "face_restoration")   # GFPGAN refina la cara
        ]