from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import process

app = FastAPI(title="VisionRestore API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conectar el router modular de procesamiento (IA eliminada)
app.include_router(process.router, prefix="/api", tags=["Procesamiento PDI"])