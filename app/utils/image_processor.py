import hashlib
import io
from fastapi import HTTPException, UploadFile
import magic
from PIL import Image
from ..core.config import settings


async def validate_image(file: UploadFile):
    """Validate uploaded image file"""
    # Check file size
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large"
        )

    # Check file type
    mime = magic.from_buffer(content, mime=True)
    if not mime.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # Check extension
    file_extension = file.filename.split(".")[-1].lower()
    if f".{file_extension}" not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="File type not allowed"
        )

    # Validate image can be opened
    try:
        Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image content"
        )

    # Reset file pointer
    await file.seek(0)


def get_image_hash(content: bytes) -> str:
    """Generate hash for image content"""
    return hashlib.md5(content).hexdigest()
