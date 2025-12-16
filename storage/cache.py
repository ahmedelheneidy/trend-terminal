"""
Cache Manager

High-level caching layer on top of the database.
Provides:
- Expiry-based caching
- Memory caching for hot data
- Cache invalidation
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import logging

from storage.database import DatabaseManager
from storage.models import OHLCVRecord, FundamentalsRecord, NewsRecord

logger = logging.getLogger(__name__)


class CacheManager:
    """
    High-level cache manager with expiry logic.
    
    Features:
    - SQLite-backed persistent cache
    - In-memory LRU cache for hot data
    - Configurable expiry times
    - Automatic cache invalidation
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        ohlcv_expiry_hours: float = 1.0,
        fundamentals_expiry_hours: float = 24.0,
        news_expiry_hours: float = 2.0,
    ):
        """
        Initialize cache manager.
        
        Args:
            db_path: Path to SQLite database
            ohlcv_expiry_hours: OHLCV cache expiry in hours
            fundamentals_expiry_hours: Fundamentals cache expiry in hours
            news_expiry_hours: News cache expiry in hours
        """
        # Use default path if not provided
        if db_path is None:
            db_path = "data/trend_terminal.db"
        self.db = DatabaseManager(db_path)
        
        self.ohlcv_expiry = timedelta(hours=ohlcv_expiry_hours)
        self.fundamentals_expiry = timedelta(hours=fundamentals_expiry_hours)
        self.news_expiry = timedelta(hours=news_expiry_hours)
        
        # In-memory cache for frequently accessed data
        self._memory_cache: Dict[str, Any] = {}
        self._memory_timestamps: Dict[str, datetime] = {}
        self._memory_expiry = timedelta(minutes=5)
        
        # Track last fetch times for update checks
        self._last_fetch: Dict[str, datetime] = {}
    
    def _memory_get(self, key: str) -> Optional[Any]:
        """Get from memory cache if not expired."""
        if key not in self._memory_cache:
            return None
        
        timestamp = self._memory_timestamps.get(key)
        if timestamp and datetime.now() - timestamp > self._memory_expiry:
            # Expired
            del self._memory_cache[key]
            del self._memory_timestamps[key]
            return None
        
        return self._memory_cache[key]
    
    def _memory_set(self, key: str, value: Any) -> None:
        """Set in memory cache."""
        self._memory_cache[key] = value
        self._memory_timestamps[key] = datetime.now()
    
    # ========================
    # OHLCV Cache Methods
    # ========================
    
    def needs_ohlcv_update(self, ticker: str) -> bool:
        """Check if OHLCV data needs to be refreshed."""
        cache_key = f"ohlcv_fetch:{ticker}"
        last_fetch = self._last_fetch.get(cache_key)
        
        if last_fetch is None:
            return True
        
        return datetime.now() - last_fetch > self.ohlcv_expiry
    
    def get_ohlcv(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[List[OHLCVRecord]]:
        """
        Get OHLCV data from cache.
        
        Args:
            ticker: Stock ticker
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            List of OHLCVRecord or None if not cached
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        # Check memory cache first
        cache_key = f"ohlcv:{ticker}"
        memory_data = self._memory_get(cache_key)
        if memory_data is not None:
            return memory_data
        
        # Check database
        records = self.db.get_ohlcv(ticker, start_date, end_date)
        
        if records:
            self._memory_set(cache_key, records)
            self._last_fetch[f"ohlcv_fetch:{ticker}"] = datetime.now()
            return records
        
        return None
    
    def save_ohlcv(self, records: List[OHLCVRecord]) -> None:
        """Store OHLCV data in cache."""
        if not records:
            return
        
        self.db.save_ohlcv(records)
        
        # Update last fetch time
        ticker = records[0].ticker
        self._last_fetch[f"ohlcv_fetch:{ticker}"] = datetime.now()
        
        # Invalidate memory cache
        mem_key = f"ohlcv:{ticker}"
        if mem_key in self._memory_cache:
            del self._memory_cache[mem_key]
            self._memory_timestamps.pop(mem_key, None)
    
    # ========================
    # Fundamentals Cache Methods
    # ========================
    
    def needs_fundamentals_update(self, ticker: str) -> bool:
        """Check if fundamentals data needs to be refreshed."""
        cache_key = f"fundamentals_fetch:{ticker}"
        last_fetch = self._last_fetch.get(cache_key)
        
        if last_fetch is None:
            # Check database for existing record
            record = self.db.get_fundamentals(ticker)
            if record and record.updated_at:
                self._last_fetch[cache_key] = record.updated_at
                return datetime.now() - record.updated_at > self.fundamentals_expiry
            return True
        
        return datetime.now() - last_fetch > self.fundamentals_expiry
    
    def get_fundamentals(self, ticker: str) -> Optional[FundamentalsRecord]:
        """
        Get fundamentals from cache.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            FundamentalsRecord or None if not cached/expired
        """
        # Check memory cache
        cache_key = f"fundamentals:{ticker}"
        memory_data = self._memory_get(cache_key)
        if memory_data is not None:
            return memory_data
        
        # Check database
        record = self.db.get_fundamentals(ticker)
        
        if not record:
            return None
        
        # Cache in memory
        self._memory_set(cache_key, record)
        self._last_fetch[f"fundamentals_fetch:{ticker}"] = record.updated_at or datetime.now()
        
        return record
    
    def save_fundamentals(self, record: FundamentalsRecord) -> None:
        """Store fundamentals in cache."""
        record.updated_at = datetime.now()
        self.db.save_fundamentals(record)
        
        # Update tracking
        self._last_fetch[f"fundamentals_fetch:{record.ticker}"] = datetime.now()
        
        # Update memory cache
        self._memory_set(f"fundamentals:{record.ticker}", record)
    
    def set_fundamentals(self, record: FundamentalsRecord) -> None:
        """Alias for save_fundamentals."""
        self.save_fundamentals(record)
    
    def is_fundamentals_fresh(self, ticker: str) -> bool:
        """Check if fundamentals cache is fresh."""
        return not self.needs_fundamentals_update(ticker)
    
    # ========================
    # News Cache Methods
    # ========================
    
    def needs_news_update(self, ticker: str) -> bool:
        """Check if news data needs to be refreshed."""
        cache_key = f"news_fetch:{ticker}"
        last_fetch = self._last_fetch.get(cache_key)
        
        if last_fetch is None:
            return True
        
        return datetime.now() - last_fetch > self.news_expiry
    
    def get_news(
        self,
        ticker: str,
        limit: int = 20,
    ) -> List[NewsRecord]:
        """
        Get news from cache.
        
        Args:
            ticker: Stock ticker
            limit: Maximum number of news items
            
        Returns:
            List of NewsRecord objects
        """
        # Check memory cache
        cache_key = f"news:{ticker}:{limit}"
        memory_data = self._memory_get(cache_key)
        if memory_data is not None:
            return memory_data
        
        # Check database
        records = self.db.get_news(ticker, hours_back=48, limit=limit)
        
        if records:
            self._memory_set(cache_key, records)
            return records
        
        return []
    
    def save_news(self, records: List[NewsRecord]) -> None:
        """Store news in cache."""
        if not records:
            return
        
        now = datetime.now()
        for record in records:
            record.fetched_at = now
        
        self.db.save_news(records)
        
        # Update tracking for each ticker
        tickers = set(r.ticker for r in records)
        for ticker in tickers:
            self._last_fetch[f"news_fetch:{ticker}"] = now
        
        # Invalidate memory cache
        keys_to_remove = [
            k for k in self._memory_cache 
            if any(k.startswith(f"news:{t}:") for t in tickers)
        ]
        for key in keys_to_remove:
            del self._memory_cache[key]
            self._memory_timestamps.pop(key, None)
    
    def set_news(self, records: List[NewsRecord]) -> None:
        """Alias for save_news."""
        self.save_news(records)
    
    def is_news_fresh(self, ticker: str) -> bool:
        """Check if news cache is fresh."""
        return not self.needs_news_update(ticker)
    
    # ========================
    # Settings Methods
    # ========================
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        value = self.db.get_setting(key)
        return value if value is not None else default
    
    def set_setting(self, key: str, value: Any) -> None:
        """Set a setting value."""
        self.db.save_setting(key, str(value))
    
    # ========================
    # Cache Management
    # ========================
    
    def clear_memory_cache(self) -> None:
        """Clear the in-memory cache."""
        self._memory_cache.clear()
        self._memory_timestamps.clear()
        logger.info("Memory cache cleared")
    
    def clear_ohlcv_cache(self, ticker: Optional[str] = None) -> None:
        """Clear OHLCV cache, optionally for specific ticker."""
        with self.db.get_connection() as conn:
            if ticker:
                conn.execute("DELETE FROM ohlcv_cache WHERE ticker = ?", (ticker,))
            else:
                conn.execute("DELETE FROM ohlcv_cache")
        
        # Clear memory cache
        if ticker:
            keys = [k for k in self._memory_cache if k.startswith(f"ohlcv:{ticker}")]
        else:
            keys = [k for k in self._memory_cache if k.startswith("ohlcv:")]
        
        for key in keys:
            del self._memory_cache[key]
            self._memory_timestamps.pop(key, None)
        
        logger.info(f"OHLCV cache cleared" + (f" for {ticker}" if ticker else ""))
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        with self.db.get_connection() as conn:
            conn.execute("DELETE FROM ohlcv_cache")
            conn.execute("DELETE FROM fundamentals_cache")
            conn.execute("DELETE FROM news_cache")
        self.clear_memory_cache()
        logger.info("All caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count records in each table
            ohlcv_count = cursor.execute("SELECT COUNT(*) FROM ohlcv_cache").fetchone()[0]
            fundamentals_count = cursor.execute("SELECT COUNT(*) FROM fundamentals_cache").fetchone()[0]
            news_count = cursor.execute("SELECT COUNT(*) FROM news_cache").fetchone()[0]
            
            # Get unique tickers
            ohlcv_tickers = cursor.execute("SELECT COUNT(DISTINCT ticker) FROM ohlcv_cache").fetchone()[0]
        
        return {
            "ohlcv_records": ohlcv_count,
            "ohlcv_tickers": ohlcv_tickers,
            "fundamentals_records": fundamentals_count,
            "news_records": news_count,
            "memory_cache_items": len(self._memory_cache),
            "expiry_settings": {
                "ohlcv_hours": self.ohlcv_expiry.total_seconds() / 3600,
                "fundamentals_hours": self.fundamentals_expiry.total_seconds() / 3600,
                "news_hours": self.news_expiry.total_seconds() / 3600,
            }
        }


def get_cache_manager() -> CacheManager:
    """Get a configured CacheManager instance."""
    return CacheManager()
