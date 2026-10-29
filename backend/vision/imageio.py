import base64
import cv2
import numpy as np

def img_to_base64(img_np):
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_img(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(b64_str), np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)