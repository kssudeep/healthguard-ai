from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_PROJECT: str = "healthguard-ai"
    VISION_MODEL_WEIGHTS_PATH: str = "ml/vision/weights/densenet121_chestxray14.pth"
    FAISS_INDEX_PATH: str = "ml/rag/indexes/faiss_medical"
    BM25_CORPUS_PATH: str = "ml/rag/indexes/bm25_corpus.json"
    GRADCAM_OUTPUT_DIR: str = "/tmp/healthguard/gradcam"
    UPLOAD_DIR: str = "/tmp/healthguard/uploads"
    REDIS_URL: str = "redis://localhost:6379"
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    VISION_CONFIDENCE_THRESHOLD: float = 0.5
    RAG_TOP_K: int = 5
    MAX_REFLECTION_LOOPS: int = 3
    CRITIC_CONFIDENCE_THRESHOLD: float = 0.65

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
