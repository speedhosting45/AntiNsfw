import redis
import json
import pickle
from typing import Optional, Any
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.redis_client = None
        self.is_connected = False
        
    async def connect(self):
        """Connect to Redis cache"""
        try:
            self.redis_client = redis.Redis.from_url(
                settings.REDIS_URL, 
                decode_responses=True
            )
            self.redis_client.ping()
            self.is_connected = True
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.is_connected = False

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.is_connected:
            return None
            
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value.encode('latin1'))
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
        return None

    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        if not self.is_connected:
            return False
            
        try:
            if ttl is None:
                ttl = settings.CACHE_TTL
                
            serialized_value = pickle.dumps(value).decode('latin1')
            return self.redis_client.setex(key, ttl, serialized_value)
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.is_connected:
            return False
            
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            return False

# Global cache instance
cache = CacheService()
