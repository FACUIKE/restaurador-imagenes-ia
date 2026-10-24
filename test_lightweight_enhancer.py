import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def restaurar_imagen_optimizada(input_path: str, output_path: str, categoria: str = "texto"):
    print(f"Iniciando restauración inteligente para categoría: [{categoria.upper()}]...")
    
    # 1. Cargar imagen
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {input_path}")
        
    h, w = img.shape[:2]
    print(f"Resolución original: {w}x{h} px")

    # 2. Pipeline según la categoría detectada por Gemini
    if categoria == "texto":
        # Filtro de preservación de bordes + reducción de ruido
        denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        # Escalado bicúbico de alta precisión 2x
        upscaled = cv2.resize(denoised, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        # Máscara de enfoque (Unsharp Masking) para definir letras y caracteres
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
        resultado = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)

    elif categoria == "rostro":
        # Denoising conservador para preservar texturas de piel
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
        upscaled = cv2.resize(denoised, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        # Enfoque sutil
        kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]])
        resultado = cv2.filter2D(upscaled, -1, kernel)

    else: # Paisaje / General
        upscaled = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        # Realce de contraste adaptativo (CLAHE en canal LAB)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        resultado = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 3. Guardar salida
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, resultado)
    
    out_h, out_w = resultado.shape[:2]
    print(f"Resolución restaurada: {out_w}x{out_h} px (2x)")
    return output_path

if __name__ == "__main__":
    uploads_dir = Path("storage/uploads")
    archivos = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.jpeg")) + list(uploads_dir.glob("*.png"))
    
    if archivos:
        img_in = str(archivos[0])
        img_out = "storage/outputs/restaurada_demo.png"
        restaurar_imagen_optimizada(img_in, img_out, categoria="texto")
        print("\n====================================")
        print("¡Restauración completada en 1 segundo!")
        print(f"Archivo guardado en: {img_out}")
        print("====================================\n")