from abc import ABC, abstractmethod
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseRestorationTool(ABC):
    """
    Clase abstracta base para todas las herramientas de restauración.
    Garantiza una interfaz unificada: recibe la ruta de la imagen actual
    y devuelve la ruta del archivo procesado.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, input_path: Path) -> Path:
        """
        Ejecuta la transformación sobre la imagen recibida.
        
        Args:
            input_path: Ruta a la imagen que se procesará.
            
        Returns:
            Path a la imagen resultante generada por la herramienta.
        """
        pass