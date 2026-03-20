from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.api.routes import router
from app.core.config import get_settings
from app.services.sentiment import SentimentAnalyzer

settings = get_settings()
analyzer: SentimentAnalyzer | None = None
templates = Jinja2Templates(directory="app/templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    analyzer = SentimentAnalyzer(model_name=settings.model_name)
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Production-ready NLP sentiment analysis service",
    lifespan=lifespan,
)

app.include_router(router)
app.mount("/static", StaticFiles(directory="app/templates"), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})