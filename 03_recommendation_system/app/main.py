from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.config import get_settings
from app.services.recommender import SeniorHybridRecommender

settings = get_settings()
recommender: SeniorHybridRecommender | None = None
templates = Jinja2Templates(directory="app/templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    recommender = SeniorHybridRecommender(model_path=settings.model_path)
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.1.0",
    description="Production-style e-commerce recommendation system service",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": settings.app_name,
        },
    )
