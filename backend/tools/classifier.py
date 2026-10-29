import torch
from transformers import pipeline
from PIL import Image
from pathlib import Path

class ImageClassifierTool:
    def __init__(self):
        self.classifier = None

    def _load_model(self):
        """Carga el modelo CLIP en la GPU (RTX 3050) solo una vez."""
        if self.classifier is None:
            device = 0 if torch.cuda.is_available() else -1
            self.classifier = pipeline(
                "zero-shot-image-classification",
                model="openai/clip-vit-base-patch32",
                device=device
            )

    def classify(self, image: Image.Image) -> str:
        """Recibe la imagen directo desde la memoria RAM y la clasifica."""
        self._load_model()
        
        candidate_labels = [
          "un documento, cartel, o imagen con texto y letras", 
          "un retrato, selfie, o foto donde el foco principal es una persona o rostro", 
          "un paisaje natural, urbano, o fondo sin personas destacadas"
        ]
        
        results = self.classifier(image, candidate_labels=candidate_labels)
        best_match = results[0]["label"]
        
        if "texto" in best_match:
            return "texto"
        elif "persona" in best_match:
            return "rostro"
        else:
            return "paisaje"