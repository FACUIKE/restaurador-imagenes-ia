import cv2
from fastapi import APIRouter
from app.schemas import ProcessRequest
from vision.imageio import base64_to_img, img_to_base64
from vision.filters import procesar_adaptativa, calcular_metricas

router = APIRouter()

@router.post("/process")
async def process_image(req: ProcessRequest):
    img_np = base64_to_img(req.image_b64)
    
    if not req.sub_modo:
        h, w = img_np.shape[:2]
        return {
            "restaurada_b64": req.image_b64,
            "heatmap_b64": req.image_b64,
            "metricas": { "fid": 1.0, "hal": 0.0, "delta": 0.0, "res": f"{w} x {h} px" }
        }

    rest_np = procesar_adaptativa(img_np, req.categoria, req.sub_modo, req.intensidad)
    heatmap_rgb, metricas = calcular_metricas(img_np, rest_np)
    
    mapa_fusionado = cv2.addWeighted(rest_np, 0.65, heatmap_rgb, 0.35, 0)
    
    return {
        "restaurada_b64": img_to_base64(rest_np),
        "heatmap_b64": img_to_base64(mapa_fusionado),
        "metricas": metricas
    }