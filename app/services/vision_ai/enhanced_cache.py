# enhanced_cache.py
import os
import time
import pickle
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from dataclasses import dataclass, field
from threading import Lock

# Fix: Import from app.core.config instead of .config
from app.core.config import CONFIG

# Add model imports that might be needed
from app.models.schemas import (
    VisualIntelligenceResult,
    ValidationResult,
    TrustMetrics
)

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Cache statistics for monitoring"""
    hits: int = 0
    misses: int = 0
    memory_size: int = 0
    disk_size_mb: float = 0.0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class EnhancedCache:
    """
    Enhanced production cache with disk persistence and intelligent management
    
    ENHANCED for unlimited thumbnail loading:
    - Increased memory limits for large documents
    - Separate thumbnail TTL configuration
    - Better batch caching support
    - Smarter eviction for thumbnails
    
    BACKWARD COMPATIBLE: Original tuple format preserved in _memory_cache
    Cache types stored separately in _cache_types dict
    """
    
    def __init__(self):
        # Cache configuration - ENHANCED for unlimited loading
        self.config = {
            "max_memory_items": 2000,  # Increased from 500 for large documents
            "max_thumbnail_items": 500,  # Separate limit for thumbnails
            "max_disk_size_mb": CONFIG.get("max_cache_size_mb", 2000),
            "cache_dir": Path(CONFIG.get("cache_dir", "/tmp/blueprint_cache")),
            "enable_disk_cache": CONFIG.get("enable_disk_cache", True),
            "aggressive_caching": CONFIG.get("aggressive_caching", True),
            "max_memory_item_size_mb": 20  # Increased from 10MB for batch operations
        }
        
        # TTL configuration by cache type - ENHANCED
        self.ttl_config = {
            "metadata": CONFIG.get("metadata_cache_ttl", 3600),      # 1 hour
            "image": CONFIG.get("image_cache_ttl", 7200),           # 2 hours
            "thumbnail": CONFIG.get("thumbnail_cache_ttl", 10800),   # 3 hours - longer for thumbnails
            "analysis": CONFIG.get("analysis_cache_ttl", 14400),     # 4 hours
            "thumbnail_batch": CONFIG.get("thumbnail_cache_ttl", 10800) * 1.5,  # 4.5 hours for batches
            "default": CONFIG.get("cache_ttl_seconds", 7200)        # 2 hours
        }
        
        # Memory cache: key -> (value, timestamp, size_estimate)
        self._memory_cache: Dict[str, Tuple[Any, float, int]] = {}
        self._access_order: Dict[str, float] = {}  # For LRU tracking
        
        # NEW: Separate cache type tracking to avoid breaking changes
        self._cache_types: Dict[str, str] = {}  # key -> cache_type
        
        # Separate tracking for thumbnails to prevent them from dominating cache
        self._thumbnail_count = 0
        
        # Thread safety
        self._lock = Lock()
        
        # Statistics
        self.stats = CacheStats()
        
        # Initialize disk cache
        self._init_disk_cache()
        
        # Initialize thumbnail count
        self._recalculate_thumbnail_count()
    
    def _init_disk_cache(self):
        """Initialize disk cache directory and cleanup"""
        if self.config["enable_disk_cache"]:
            try:
                self.config["cache_dir"].mkdir(parents=True, exist_ok=True)
                self._cleanup_old_disk_cache()
                logger.info(f"✅ Disk cache initialized at {self.config['cache_dir']}")
            except Exception as e:
                logger.error(f"Failed to initialize disk cache: {e}")
                self.config["enable_disk_cache"] = False
    
    def get(self, key: str, cache_type: str = "default") -> Optional[Any]:
        """
        Get value from cache with type-specific TTL
        Checks memory first, then disk
        """
        with self._lock:
            # Check memory cache
            if key in self._memory_cache:
                value, timestamp, size = self._memory_cache[key]
                # Get the stored cache type if available
                stored_type = self._cache_types.get(key, cache_type)
                ttl = self._get_ttl(stored_type)
                
                if time.time() - timestamp <= ttl:
                    # IMPORTANT: Validate the cached value
                    if value is None or (isinstance(value, (list, dict, str)) and not value):
                        # Empty or None value - remove from cache and miss
                        logger.warning(f"⚠️ Cache contained empty data for key: {key[:20]}...")
                        self._remove_from_memory(key)
                        self.stats.misses += 1
                        return None
                        
                    # Cache hit - update access time
                    self._access_order[key] = time.time()
                    self.stats.hits += 1
                    logger.debug(f"✅ Memory cache hit: {key[:20]}... ({stored_type})")
                    return value
                else:
                    # Expired - remove from cache
                    self._remove_from_memory(key)
        
        # Check disk cache if enabled
        if self.config["enable_disk_cache"]:
            disk_value = self._get_from_disk(key, cache_type)
            if disk_value is not None:
                # Validate disk value too
                if isinstance(disk_value, (list, dict, str)) and not disk_value:
                    logger.warning(f"⚠️ Disk cache contained empty data for key: {key[:20]}...")
                    # Remove the bad cache file
                    disk_path = self._get_disk_path(key)
                    try:
                        disk_path.unlink()
                    except:
                        pass
                    self.stats.misses += 1
                    return None
                    
                # Valid data - promote to memory cache if not too large
                size_estimate = self._estimate_size(disk_value)
                if size_estimate < self.config["max_memory_item_size_mb"] * 1024 * 1024:
                    self.set(key, disk_value, cache_type, persist_only=False)
                self.stats.hits += 1
                return disk_value
        
        # Cache miss
        self.stats.misses += 1
        logger.debug(f"❌ Cache miss: {key[:20]}... ({cache_type})")
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        cache_type: str = "default",
        persist_only: bool = False
    ) -> None:
        """
        Set value in cache
        persist_only: If True, only save to disk (for large items)
        
        ENHANCED: Better handling of thumbnail batches
        """
        timestamp = time.time()
        size_estimate = self._estimate_size(value)
        
        # Determine if this is a thumbnail
        is_thumbnail = cache_type in ["thumbnail", "thumbnail_batch"] or "thumb" in key.lower()
        
        # Memory cache (unless persist_only or too large)
        max_size = self.config["max_memory_item_size_mb"] * 1024 * 1024
        if not persist_only and size_estimate < max_size:
            with self._lock:
                # Check limits based on type
                if is_thumbnail:
                    # Check thumbnail-specific limit
                    if self._thumbnail_count >= self.config["max_thumbnail_items"]:
                        self._evict_thumbnails()
                else:
                    # Check general limit
                    non_thumbnail_count = len(self._memory_cache) - self._thumbnail_count
                    if non_thumbnail_count >= (self.config["max_memory_items"] - self.config["max_thumbnail_items"]):
                        self._evict_lru(exclude_thumbnails=True)
                
                # Store with original 3-tuple format for backward compatibility
                self._memory_cache[key] = (value, timestamp, size_estimate)
                self._cache_types[key] = cache_type  # Store type separately
                self._access_order[key] = timestamp
                self.stats.memory_size += size_estimate
                
                if is_thumbnail:
                    self._thumbnail_count += 1
                
                logger.debug(f"💾 Cached to memory: {key[:20]}... (size: {size_estimate/1024:.1f}KB, type: {cache_type})")
        
        # Disk cache if enabled
        if self.config["enable_disk_cache"]:
            self._save_to_disk(key, value, timestamp, cache_type)
    
    def set_batch(self, items: List[Tuple[str, Any, str]]) -> None:
        """
        NEW: Set multiple items in cache efficiently
        items: List of (key, value, cache_type) tuples
        """
        batch_size = sum(self._estimate_size(item[1]) for item in items)
        
        # If batch is large, use disk cache
        if batch_size > self.config["max_memory_item_size_mb"] * 1024 * 1024:
            logger.info(f"📦 Large batch ({batch_size/1024/1024:.1f}MB) - using disk cache")
            for key, value, cache_type in items:
                self.set(key, value, cache_type, persist_only=True)
        else:
            # Add to memory cache
            for key, value, cache_type in items:
                self.set(key, value, cache_type)
    
    def get_batch(self, keys: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        NEW: Get multiple items from cache efficiently
        keys: List of (key, cache_type) tuples
        Returns: Dict of key -> value for found items
        """
        results = {}
        for key, cache_type in keys:
            value = self.get(key, cache_type)
            if value is not None:
                results[key] = value
        return results
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries
        pattern: If provided, invalidate keys containing this pattern
        Returns: Number of entries invalidated
        """
        count = 0
        
        with self._lock:
            if pattern:
                # Pattern-based invalidation
                keys_to_remove = [
                    k for k in self._memory_cache.keys()
                    if pattern in k
                ]
                
                for key in keys_to_remove:
                    self._remove_from_memory(key)
                    count += 1
                
                logger.info(f"🗑️ Invalidated {count} memory cache entries matching '{pattern}'")
                
                # Also invalidate disk cache
                if self.config["enable_disk_cache"]:
                    disk_count = self._invalidate_disk_pattern(pattern)
                    count += disk_count
            else:
                # Clear all caches
                count = len(self._memory_cache)
                self._memory_cache.clear()
                self._access_order.clear()
                self._cache_types.clear()  # Also clear type tracking
                self.stats.memory_size = 0
                self._thumbnail_count = 0
                
                if self.config["enable_disk_cache"]:
                    disk_count = self._clear_disk_cache()
                    count += disk_count
                
                logger.info(f"🗑️ Cleared all cache entries ({count} total)")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics - ENHANCED"""
        with self._lock:
            memory_count = len(self._memory_cache)
            
        disk_count = 0
        if self.config["enable_disk_cache"]:
            disk_count = len(list(self.config["cache_dir"].glob("*.cache")))
        
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": f"{self.stats.hit_rate * 100:.1f}%",
            "memory_items": memory_count,
            "memory_thumbnails": self._thumbnail_count,  # NEW
            "memory_size_mb": self.stats.memory_size / (1024 * 1024),
            "disk_items": disk_count,
            "disk_size_mb": self._calculate_disk_size(),
            "evictions": self.stats.evictions,
            "limits": {  # NEW
                "max_memory_items": self.config["max_memory_items"],
                "max_thumbnail_items": self.config["max_thumbnail_items"],
                "max_disk_size_mb": self.config["max_disk_size_mb"]
            }
        }
    
    # Private methods
    
    def _get_ttl(self, cache_type: str) -> float:
        """Get TTL for cache type"""
        return self.ttl_config.get(cache_type, self.ttl_config["default"])
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of object in bytes"""
        try:
            # For lists/dicts of images, estimate more accurately
            if isinstance(value, list) and value and isinstance(value[0], dict) and "url" in value[0]:
                # Thumbnail batch - estimate based on base64 data
                total_size = 0
                for item in value:
                    if "url" in item and item["url"].startswith("data:"):
                        # Base64 data URL
                        total_size += len(item["url"])
                    else:
                        total_size += len(str(item))
                return total_size
            else:
                # Simple estimation - serialize and measure
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            # Fallback estimation
            return 1024  # 1KB default
    
    def _evict_lru(self, exclude_thumbnails: bool = False) -> None:
        """
        Evict least recently used item from memory cache
        ENHANCED: Can exclude thumbnails from eviction
        """
        if not self._access_order:
            return
        
        # Build list of candidates
        candidates = []
        for key, access_time in self._access_order.items():
            if key in self._memory_cache:
                cache_type = self._cache_types.get(key, "default")
                is_thumbnail = cache_type in ["thumbnail", "thumbnail_batch"] or "thumb" in key.lower()
                
                if not exclude_thumbnails or not is_thumbnail:
                    candidates.append((key, access_time))
        
        if not candidates:
            # If all are thumbnails and we're excluding them, evict oldest thumbnail
            candidates = list(self._access_order.items())
        
        if candidates:
            # Find oldest accessed key
            lru_key = min(candidates, key=lambda x: x[1])[0]
            
            # Remove it
            self._remove_from_memory(lru_key)
            self.stats.evictions += 1
            
            logger.debug(f"♻️ Evicted LRU item: {lru_key[:20]}...")
    
    def _evict_thumbnails(self) -> None:
        """
        NEW: Evict oldest thumbnails when thumbnail limit reached
        """
        # Find all thumbnail keys
        thumbnail_keys = []
        for key, (_, timestamp, _) in self._memory_cache.items():
            cache_type = self._cache_types.get(key, "default")
            if cache_type in ["thumbnail", "thumbnail_batch"] or "thumb" in key.lower():
                thumbnail_keys.append((key, timestamp))
        
        # Sort by timestamp (oldest first)
        thumbnail_keys.sort(key=lambda x: x[1])
        
        # Remove oldest 10% of thumbnails
        to_remove = max(1, len(thumbnail_keys) // 10)
        for key, _ in thumbnail_keys[:to_remove]:
            self._remove_from_memory(key)
            self.stats.evictions += 1
        
        logger.debug(f"♻️ Evicted {to_remove} old thumbnails")
    
    def _remove_from_memory(self, key: str) -> None:
        """Remove item from memory cache - ENHANCED"""
        if key in self._memory_cache:
            _, _, size = self._memory_cache[key]
            cache_type = self._cache_types.get(key, "default")
            del self._memory_cache[key]
            self.stats.memory_size -= size
            
            # Update thumbnail count
            if cache_type in ["thumbnail", "thumbnail_batch"] or "thumb" in key.lower():
                self._thumbnail_count = max(0, self._thumbnail_count - 1)
        
        if key in self._access_order:
            del self._access_order[key]
        
        if key in self._cache_types:
            del self._cache_types[key]
    
    def _get_disk_path(self, key: str) -> Path:
        """Get disk cache file path for key"""
        # Use MD5 hash for filename to avoid filesystem issues
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.config["cache_dir"] / f"{safe_key}.cache"
    
    def _get_from_disk(self, key: str, cache_type: str) -> Optional[Any]:
        """Get value from disk cache"""
        disk_path = self._get_disk_path(key)
        
        if not disk_path.exists():
            return None
        
        try:
            with open(disk_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check TTL
            stored_cache_type = data.get('cache_type', cache_type)
            ttl = self._get_ttl(stored_cache_type)
            
            if time.time() - data['timestamp'] <= ttl:
                logger.debug(f"💾 Disk cache hit: {key[:20]}...")
                return data['value']
            else:
                # Expired - remove file
                disk_path.unlink()
                logger.debug(f"🕐 Disk cache expired: {key[:20]}...")
        except Exception as e:
            logger.warning(f"Disk cache read error for {key[:20]}...: {e}")
            try:
                disk_path.unlink()
            except:
                pass
        
        return None
    
    def _save_to_disk(self, key: str, value: Any, timestamp: float, cache_type: str) -> None:
        """Save value to disk cache"""
        if not self.config["enable_disk_cache"]:
            return
        
        disk_path = self._get_disk_path(key)
        
        try:
            # Check disk space before saving
            if self._calculate_disk_size() >= self.config["max_disk_size_mb"]:
                self._cleanup_old_disk_cache()
            
            # Save to disk
            data = {
                'value': value,
                'timestamp': timestamp,
                'cache_type': cache_type
            }
            
            # Write to temporary file first (atomic operation)
            temp_path = disk_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Move to final location
            temp_path.replace(disk_path)
            
            logger.debug(f"💾 Cached to disk: {key[:20]}... ({cache_type})")
            
        except Exception as e:
            logger.warning(f"Disk cache write error for {key[:20]}...: {e}")
            # Clean up temp file if exists
            try:
                temp_path.unlink()
            except:
                pass
    
    def _cleanup_old_disk_cache(self) -> None:
        """Clean up old cache files based on size limit and age - ENHANCED"""
        try:
            cache_files = list(self.config["cache_dir"].glob("*.cache"))
            if not cache_files:
                return
            
            # Group files by age
            current_time = time.time()
            files_by_age = []
            
            for cache_file in cache_files:
                try:
                    stat = cache_file.stat()
                    age_hours = (current_time - stat.st_mtime) / 3600
                    size_mb = stat.st_size / (1024 * 1024)
                    
                    # Read cache type if possible
                    cache_type = "unknown"
                    try:
                        with open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                            cache_type = data.get('cache_type', 'unknown')
                    except:
                        pass
                    
                    files_by_age.append((cache_file, age_hours, size_mb, cache_type))
                except:
                    pass
            
            # Calculate total size
            total_size_mb = sum(f[2] for f in files_by_age)
            
            # Remove files based on priority
            if total_size_mb > self.config["max_disk_size_mb"]:
                # Sort by priority: age + type (thumbnails are lower priority)
                def priority_key(item):
                    file, age, size, ctype = item
                    # Thumbnails can be removed first, then by age
                    type_priority = 1.0 if ctype in ["thumbnail", "thumbnail_batch"] else 2.0
                    return age / type_priority
                
                files_by_age.sort(key=priority_key, reverse=True)
                
                # Remove oldest/lowest priority files until under 80% of limit
                target_size_mb = self.config["max_disk_size_mb"] * 0.8
                
                for cache_file, age, size, ctype in files_by_age:
                    if total_size_mb <= target_size_mb:
                        break
                    
                    cache_file.unlink()
                    total_size_mb -= size
                    logger.debug(f"🗑️ Removed cache file: {cache_file.name} (age: {age:.1f}h, type: {ctype})")
            
            # Remove very old files
            max_age_hours = {
                "thumbnail": 24,
                "thumbnail_batch": 24,
                "image": 48,
                "analysis": 72,
                "metadata": 72
            }
            
            for cache_file, age, size, ctype in files_by_age:
                if cache_file.exists():  # May have been deleted above
                    max_age = max_age_hours.get(ctype, 48)
                    if age > max_age:
                        cache_file.unlink()
                        logger.debug(f"🗑️ Removed expired cache file: {cache_file.name} (age: {age:.1f}h)")
                        
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")
    
    def _calculate_disk_size(self) -> float:
        """Calculate total disk cache size in MB"""
        if not self.config["enable_disk_cache"]:
            return 0.0
        
        try:
            cache_files = self.config["cache_dir"].glob("*.cache")
            total_bytes = sum(f.stat().st_size for f in cache_files)
            return total_bytes / (1024 * 1024)
        except:
            return 0.0
    
    def _invalidate_disk_pattern(self, pattern: str) -> int:
        """Invalidate disk cache entries matching pattern"""
        count = 0
        
        # We can't easily match patterns with hashed filenames
        # So we'll need to check each file
        try:
            for cache_file in self.config["cache_dir"].glob("*.cache"):
                # For now, just clear all disk cache when pattern invalidation is requested
                # In production, you might store a mapping of keys to hashes
                cache_file.unlink()
                count += 1
        except Exception as e:
            logger.warning(f"Disk invalidation error: {e}")
        
        return count
    
    def _clear_disk_cache(self) -> int:
        """Clear all disk cache files"""
        count = 0
        
        try:
            for cache_file in self.config["cache_dir"].glob("*.cache"):
                cache_file.unlink()
                count += 1
        except Exception as e:
            logger.warning(f"Disk cache clear error: {e}")
        
        return count
    
    def _recalculate_thumbnail_count(self) -> None:
        """Recalculate thumbnail count from current cache"""
        count = 0
        for key in self._memory_cache:
            cache_type = self._cache_types.get(key, "default")
            if cache_type in ["thumbnail", "thumbnail_batch"] or "thumb" in key.lower():
                count += 1
        self._thumbnail_count = count
    
    def warmup(self, keys: List[Tuple[str, str]]) -> None:
        """
        Warmup cache by preloading common keys from disk
        keys: List of (key, cache_type) tuples
        """
        if not self.config["enable_disk_cache"]:
            return
        
        loaded = 0
        for key, cache_type in keys:
            if self.get(key, cache_type) is not None:
                loaded += 1
        
        logger.info(f"🔥 Cache warmup: loaded {loaded}/{len(keys)} items")
    
    def persist_important(self, keys: List[str]) -> None:
        """
        Ensure important keys are persisted to disk
        Useful before shutdown or for critical data
        """
        if not self.config["enable_disk_cache"]:
            return
        
        persisted = 0
        with self._lock:
            for key in keys:
                if key in self._memory_cache:
                    value, timestamp, _ = self._memory_cache[key]
                    cache_type = self._cache_types.get(key, "analysis")
                    self._save_to_disk(key, value, timestamp, cache_type)
                    persisted += 1
        
        logger.info(f"💾 Persisted {persisted} important cache entries to disk")
    
    def optimize_for_thumbnails(self, document_id: str, expected_pages: int) -> None:
        """
        NEW: Optimize cache for loading many thumbnails
        """
        # Pre-allocate space by clearing old thumbnails if needed
        if expected_pages > 50:
            logger.info(f"🔧 Optimizing cache for {expected_pages} thumbnails")
            
            # Clear old thumbnails from other documents
            pattern = "thumb"
            if document_id not in pattern:
                self.invalidate(pattern)
            
            # Adjust limits temporarily if needed
            if expected_pages > self.config["max_thumbnail_items"]:
                self.config["max_thumbnail_items"] = min(expected_pages * 1.2, 1000)
                logger.info(f"📈 Temporarily increased thumbnail limit to {self.config['max_thumbnail_items']}")

# Convenience class for backward compatibility
@dataclass
class ProductionCache:
    """Lightweight wrapper for backward compatibility"""
    _cache: EnhancedCache = field(default_factory=EnhancedCache)
    max_size: int = 200  # Ignored, for compatibility
    ttl_seconds: int = 3600  # Ignored, for compatibility
    
    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        self._cache.set(key, value)
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        return self._cache.invalidate(pattern)
    
    def get_stats(self) -> Dict[str, Any]:
        return self._cache.get_stats()
