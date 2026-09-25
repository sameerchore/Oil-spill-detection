"""
TideTrace Main FastAPI Application.
Mounts static HUD console assets, data fixtures, CORS middleware, exception handlers,
and REST API routes.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import BASE_DIR, DATA_DIR, AIS_DB_PATH, AIS_DIR
from app.ais.store import AISStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tidetrace")

STATIC_DIR = BASE_DIR / "backend" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TideTrace Naval Tactical Console starting up...")
    if not AIS_DB_PATH.exists():
        csv_path = AIS_DIR / "tidetrace_ais.csv"
        if csv_path.exists():
            logger.info("Initializing AIS SQLite database from %s", csv_path)
            with AISStore(AIS_DB_PATH) as store:
                count = store.load_csv(csv_path)
                logger.info("Loaded %d AIS records into %s", count, AIS_DB_PATH)
    yield
    logger.info("TideTrace Naval Tactical Console shutting down...")


app = FastAPI(
    title="TideTrace Naval Forensic Attribution Console",
    version="1.0.0",
    description="Air-Gapped Satellite SAR, Lagrangian Metocean Drift, and AIS Forensic Attribution Console for NTRO.",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# 1. CORS Middleware (Permissive for offline air-gapped web console)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Exception Handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled server error at %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "path": request.url.path,
        },
    )


# 3. Include REST API Routes
app.include_router(router)

# 4. Mount Data Directory (for previews and GeoTIFFs)
if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# 5. Mount Static HUD Console Assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"status": "online", "message": "TideTrace Tactical Backend Ready"})

    @app.get("/favicon.ico", include_in_schema=False)
    async def serve_favicon():
        svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="6" fill="#0b1320"/><text x="16" y="22" fill="#00f0ff" font-family="monospace" font-weight="bold" font-size="16" text-anchor="middle">TT</text></svg>'
        return Response(content=svg, media_type="image/svg+xml")
