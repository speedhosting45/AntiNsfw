import io
import time
import asyncio
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import aiofiles
import magic

from .core.config import settings
from .models.schemas import (
    ScanRequest, ScanResponse, BatchScanRequest, 
    BatchScanResponse, HealthCheck, NSFWCategory
)
from .services.scanner import scanner
from .services.cache import cache
from .utils.image_processor import validate_image, get_image_hash

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
a
# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Startup events
@app.on_event("startup")
async def startup_event():
    await cache.connect()
    await scanner.load_models()

# Rate limiting middleware (simplified)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main UI"""
    async with aiofiles.open('app/static/index.html', 'r') as f:
        return await f.read()

@app.post(f"{settings.API_V1_STR}/scan", response_model=ScanResponse)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def scan_image(
    request: Request,                                # 👈 Add this line
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    return_heatmap: bool = False,
    detailed_analysis: bool = False
):
    """Scan single image for NSFW content"""
    try:
        # Validate file
        await validate_image(file)
        
        # Read file content
        content = await file.read()
        
        # Check cache
        image_hash = get_image_hash(content)
        cached_result = await cache.get(f"scan:{image_hash}")
        if cached_result:
            return cached_result
        
        # Process image
        result = await scanner.scan_image(
            content, 
            detailed=detailed_analysis, 
            heatmap=return_heatmap
        )
        
        # Cache result
        await cache.set(f"scan:{image_hash}", result)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"{settings.API_V1_STR}/scan/batch", response_model=BatchScanResponse)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def scan_batch_images(
    request: BatchScanRequest
):
    """Scan multiple images in batch"""
    start_time = time.time()
    results = []
    failed_count = 0
    
    # Process images
    if request.parallel_processing:
        # Parallel processing
        tasks = []
        for image_request in request.images:
            # Implementation for batch processing
            pass
    else:
        # Sequential processing
        for image_request in request.images:
            try:
                # Process each image
                pass
            except Exception as e:
                failed_count += 1
                results.append(ScanResponse(
                    is_nsfw=False,
                    confidence=0.0,
                    categories={},
                    processing_time=0.0,
                    warnings=[str(e)]
                ))
    
    total_time = time.time() - start_time
    
    return BatchScanResponse(
        results=results,
        total_processed=len(results) - failed_count,
        failed_count=failed_count,
        total_time=total_time
    )

@app.get(f"{settings.API_V1_STR}/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    import psutil
    import os
    
    return HealthCheck(
        status="healthy",
        model_loaded=scanner.is_loaded,
        cache_connected=cache.is_connected,
        uptime=psutil.Process(os.getpid()).create_time(),
        total_scans=scanner.total_scans
    )

@app.get(f"{settings.API_V1_STR}/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "total_scans": scanner.total_scans,
        "model_loaded": scanner.is_loaded,
        "cache_connected": cache.is_connected,
        "rate_limit": settings.RATE_LIMIT_PER_MINUTE
    }

@app.get("/heatmaps/{heatmap_id}")
async def get_heatmap(heatmap_id: str):
    """Serve generated heatmaps"""
    heatmap_data = await cache.get(f"heatmap:{heatmap_id}")
    if not heatmap_data:
        raise HTTPException(status_code=404, detail="Heatmap not found")
    
    return StreamingResponse(
        io.BytesIO(heatmap_data), 
        media_type="image/png"
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
