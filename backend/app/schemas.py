from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    image_b64: str
    categoria: str
    sub_modo: str
    intensidad: Optional[float] = None