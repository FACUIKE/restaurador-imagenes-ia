import cv2
import numpy as np

def procesar_adaptativa(img_np: np.ndarray, categoria: str, sub_modo: str, intensidad: float) -> np.ndarray:
    h, w = img_np.shape[:2]
    nw, nh = int(w * 2), int(h * 2)
    escalada = cv2.resize(img_np, (nw, nh), interpolation=cv2.INTER_CUBIC)
    
    if categoria == "texto":
        if sub_modo == "Iluminación Irregular (Sauvola)":
            c_val = int(intensidad) if intensidad else 10
            gris = cv2.cvtColor(escalada, cv2.COLOR_RGB2GRAY)
            return cv2.cvtColor(cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, c_val), cv2.COLOR_GRAY2RGB)
            
        elif sub_modo == "Trazos Desgastados (Top-Hat)":
            alpha_val = intensidad if intensidad else 2.5
            gris = cv2.cvtColor(escalada, cv2.COLOR_RGB2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
            blackhat = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, kernel)
            res = cv2.subtract(gris, cv2.convertScaleAbs(blackhat, alpha=alpha_val, beta=0))
            return cv2.cvtColor(cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2RGB)
            
        elif sub_modo == "Eliminar Muaré de Pantalla (FFT 2D)":
            radio = int(intensidad) if intensidad else 40
            gris = cv2.cvtColor(escalada, cv2.COLOR_RGB2GRAY)
            fshift = np.fft.fftshift(np.fft.fft2(gris))
            crow, ccol = gris.shape[0] // 2 , gris.shape[1] // 2
            mascara = np.ones(gris.shape, np.uint8)
            mascara[crow-3:crow+3, :], mascara[:, ccol-3:ccol+3] = 0, 0
            mascara[crow-radio:crow+radio, ccol-radio:ccol+radio] = 1
            f_ishift = np.fft.ifftshift(fshift * mascara)
            img_back = np.abs(np.fft.ifft2(f_ishift))
            img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, binarizada = cv2.threshold(img_back, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.cvtColor(binarizada, cv2.COLOR_GRAY2RGB)
            
    elif categoria == "rostro":
        ycbcr = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        y_up = cv2.resize(y, (nw, nh), interpolation=cv2.INTER_CUBIC)
        cr_up = cv2.resize(cr, (nw, nh), interpolation=cv2.INTER_CUBIC)
        cb_up = cv2.resize(cb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        escalada = cv2.cvtColor(cv2.merge((y_up, cr_up, cb_up)), cv2.COLOR_YCrCb2RGB)

        if sub_modo == "Super-Resolución (Solo YCbCr)":
            y_up_clahe = cv2.createCLAHE(clipLimit=(intensidad or 1.5), tileGridSize=(8, 8)).apply(y_up)
            return cv2.cvtColor(cv2.merge((y_up_clahe, cr_up, cb_up)), cv2.COLOR_YCrCb2RGB)
            
        elif sub_modo == "Trepidación (Deconvolución / Sharpening)":
            fuerza = intensidad if intensidad else 0.7
            res = cv2.filter2D(escalada, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
            return cv2.addWeighted(escalada, 1.0 - fuerza, res, fuerza, 0)
            
        elif sub_modo == "Ruido ISO Alto (No-Local Means)":
            h_val = intensidad if intensidad else 10.0
            return cv2.fastNlMeansDenoisingColored(escalada, None, h_val, h_val, 7, 21)
            
        elif sub_modo == "Iluminación Nocturna (MSRCR / Retinex)":
            img_float = np.float32(escalada) + 1.0
            retinex = sum(np.log10(img_float) - np.log10(cv2.GaussianBlur(img_float, (0, 0), s)) for s in [15, 80, 250])
            res = cv2.normalize((retinex / 3.0) * 255.0, None, 0, 255, cv2.NORM_MINMAX)
            return cv2.cvtColor(cv2.convertScaleAbs(res, alpha=(intensidad or 1.2), beta=10), cv2.COLOR_BGR2RGB)
            
        elif sub_modo == "Suavizado de Piel (Filtro Guiado)":
            return cv2.edgePreservingFilter(escalada, flags=cv2.RECURS_FILTER, sigma_s=(int(intensidad) if intensidad else 60), sigma_r=0.2)

    return img_np.copy()

def calcular_metricas(img_orig: np.ndarray, rest_np: np.ndarray):
    h_r, w_r = rest_np.shape[:2]
    deg_scaled = cv2.resize(cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY), (w_r, h_r), interpolation=cv2.INTER_CUBIC)
    res_g = cv2.cvtColor(rest_np, cv2.COLOR_RGB2GRAY)
    
    g_deg = np.sqrt(cv2.Sobel(deg_scaled, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(deg_scaled, cv2.CV_64F, 0, 1)**2)
    g_res = np.sqrt(cv2.Sobel(res_g, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(res_g, cv2.CV_64F, 0, 1)**2)
    
    diff = np.abs(g_res - g_deg)
    anomalias = np.where(diff > np.percentile(diff, 90), diff, 0)
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(cv2.normalize(anomalias, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    
    n_orig = float(cv2.Laplacian(deg_scaled, cv2.CV_64F).var())
    n_rest = float(cv2.Laplacian(res_g, cv2.CV_64F).var())
    
    metricas = {
        "fid": round(max(0.0, min(1.0, 1.0 - ((np.mean(diff) / 255.0) * 2.0))), 3),
        "hal": round((np.count_nonzero(anomalias) / max(anomalias.size, 1)) * 100.0, 1),
        "delta": round(((n_rest - n_orig) / max(n_orig, 1.0)) * 100.0, 1),
        "res": f"{w_r} x {h_r} px"
    }
    return heatmap_rgb, metricas