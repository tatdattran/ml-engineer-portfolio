import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.detector import YOLODetector

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


detector = YOLODetector(
    model_path=settings.model_path,
    conf_threshold=settings.conf_threshold,
    iou_threshold=settings.iou_threshold,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        detector.load()
    except Exception as exc:
        logger.exception("Failed to load model during startup: %s", exc)
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Production-ready YOLO object detection service",
    lifespan=lifespan,
)
app.include_router(router)
