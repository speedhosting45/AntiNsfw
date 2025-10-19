from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class NSFWCategory(str, Enum):
    SAFE = "safe"
    EXPLICIT = "explicit"
    SUGGESTIVE = "suggestive"
    VIOLENCE = "violence"
    DISTURBING = "disturbing"

class ScanRequest(BaseModel):
    image_url: Optional[str] = None
    return_heatmap: bool = False
    detailed_analysis: bool = False
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)

class ScanResponse(BaseModel):
    is_nsfw: bool
    confidence: float
    categories: Dict[NSFWCategory, float]
    processing_time: float
    image_hash: Optional[str] = None
    heatmap_url: Optional[str] = None
    detailed_analysis: Optional[Dict[str, Any]] = None
    warnings: List[str] = []

class BatchScanRequest(BaseModel):
    images: List[ScanRequest]
    parallel_processing: bool = True

class BatchScanResponse(BaseModel):
    results: List[ScanResponse]
    total_processed: int
    failed_count: int
    total_time: float

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    cache_connected: bool
    uptime: float
    total_scans: int
