from pydantic import BaseModel, Field
from typing import Literal

class RestorationDecision(BaseModel):
    categoria: Literal["rostro", "texto", "paisaje"] = Field(
        default="paisaje",
        description="Clasificación principal de la imagen: 'rostro' si hay personas, 'texto' si son letras/carteles/documentos, 'paisaje' para escenas generales u objetos."
    )
    reducir_ruido: bool = Field(
        default=False,
        description="True si la imagen presenta grano severo o ruido ISO."
    )
    colorizar: bool = Field(
        default=False,
        description="True únicamente si la imagen está en blanco y negro o sepia."
    )
    restaurar_rostros: bool = Field(
        default=False,
        description="True si hay rostros humanos visibles que requieren mejora."
    )
    super_resolucion: bool = Field(
        default=True,
        description="True si la imagen está desenfocada, pixelada o requiere mayor nitidez."
    )
    motivo_diagnostico: str = Field(
        default="Análisis completado.",
        description="Breve explicación técnica de los defectos detectados y la categoría elegida."
    )