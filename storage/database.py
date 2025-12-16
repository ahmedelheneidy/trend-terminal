"""
Database Manager for Trend Terminal.
Handles SQLite database operations with proper schema management.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

from storage.models import (
    OHLCVRecord,
    FundamentalsRecord,
    NewsRecord,
    AlertRecord,
    BacktestRun,
    NewsCategory,
    NewsSentiment,
    SignalDirection,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    SQLite database manager with connection pooling and schema management.
    """
    
    # Schema version for migrations
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: str = "data/trend_terminal.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Schema version table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            
            # OHLCV cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    adj_close REAL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date 
                ON ohlcv_cache(ticker, date)
            """)
            
            # Fundamentals cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fundamentals_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT UNIQUE NOT NULL,
                    market_cap REAL,
                    sector TEXT,
                    industry TEXT,
                    company_name TEXT,
                    currency TEXT,
                    exchange TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # News cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    source TEXT,
                    url TEXT,
                    summary TEXT,
                    category TEXT DEFAULT 'other',
                    sentiment INTEGER DEFAULT 0,
                    sentiment_score REAL DEFAULT 0.0,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, headline, published_at)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_ticker_published 
                ON news_cache(ticker, published_at)
            """)
            
            # Settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Alert history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    reasons TEXT,
                    news_links TEXT,
                    alerted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notification_method TEXT DEFAULT 'telegram',
                    success INTEGER DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alert_ticker_date 
                ON alert_history(ticker, alerted_at)
            """)
            
            # Backtest runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    tickers TEXT NOT NULL,
                    start_date DATE,
                    end_date DATE,
                    holding_period INTEGER DEFAULT 5,
                    exit_strategy TEXT DEFAULT 'fixed',
                    commission REAL DEFAULT 0.0,
                    slippage_pct REAL DEFAULT 0.001,
                    include_news INTEGER DEFAULT 1,
                    params_json TEXT,
                    results_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Set schema version
            cursor.execute("""
                INSERT OR REPLACE INTO schema_version (version) VALUES (?)
            """, (self.SCHEMA_VERSION,))
            
            logger.info(f"Database initialized at {self.db_path}")
    
    # ==================== OHLCV Operations ====================
    
    def save_ohlcv(self, records: List[OHLCVRecord]) -> int:
        """
        Save OHLCV records to database.
        
        Args:
            records: List of OHLCVRecord objects
            
        Returns:
            Number of records saved
        """
        if not records:
            return 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO ohlcv_cache 
                (ticker, date, open, high, low, close, volume, adj_close, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (r.ticker, r.date, r.open, r.high, r.low, r.close, 
                 r.volume, r.adj_close, r.fetched_at or datetime.now())
                for r in records
            ])
            return cursor.rowcount
    
    def get_ohlcv(
        self, 
        ticker: str, 
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[OHLCVRecord]:
        """
        Get OHLCV records for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            List of OHLCVRecord objects
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM ohlcv_cache WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date ASC"
            cursor.execute(query, params)
            
            return [
                OHLCVRecord(
                    ticker=row["ticker"],
                    date=datetime.strptime(row["date"], "%Y-%m-%d").date() if isinstance(row["date"], str) else row["date"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    adj_close=row["adj_close"],
                    fetched_at=row["fetched_at"],
                )
                for row in cursor.fetchall()
            ]
    
    def get_ohlcv_last_date(self, ticker: str) -> Optional[date]:
        """Get the last date we have OHLCV data for a ticker."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(date) as last_date FROM ohlcv_cache WHERE ticker = ?
            """, (ticker,))
            row = cursor.fetchone()
            if row and row["last_date"]:
                if isinstance(row["last_date"], str):
                    return datetime.strptime(row["last_date"], "%Y-%m-%d").date()
                return row["last_date"]
            return None
    
    def is_ohlcv_fresh(self, ticker: str, max_age_hours: int = 24) -> bool:
        """Check if OHLCV data is fresh enough."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(fetched_at) as last_fetch FROM ohlcv_cache WHERE ticker = ?
            """, (ticker,))
            row = cursor.fetchone()
            if row and row["last_fetch"]:
                last_fetch = row["last_fetch"]
                if isinstance(last_fetch, str):
                    last_fetch = datetime.fromisoformat(last_fetch)
                return (datetime.now() - last_fetch).total_seconds() < max_age_hours * 3600
            return False
    
    # ==================== Fundamentals Operations ====================
    
    def save_fundamentals(self, record: FundamentalsRecord) -> bool:
        """Save fundamentals record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO fundamentals_cache 
                (ticker, market_cap, sector, industry, company_name, currency, exchange, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.ticker, record.market_cap, record.sector, record.industry,
                record.company_name, record.currency, record.exchange,
                record.updated_at or datetime.now()
            ))
            return cursor.rowcount > 0
    
    def get_fundamentals(self, ticker: str) -> Optional[FundamentalsRecord]:
        """Get fundamentals for a ticker."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM fundamentals_cache WHERE ticker = ?
            """, (ticker,))
            row = cursor.fetchone()
            if row:
                return FundamentalsRecord(
                    ticker=row["ticker"],
                    market_cap=row["market_cap"],
                    sector=row["sector"],
                    industry=row["industry"],
                    company_name=row["company_name"],
                    currency=row["currency"],
                    exchange=row["exchange"],
                    updated_at=row["updated_at"],
                )
            return None
    
    def is_fundamentals_fresh(self, ticker: str, max_age_hours: int = 24) -> bool:
        """Check if fundamentals data is fresh enough."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT updated_at FROM fundamentals_cache WHERE ticker = ?
            """, (ticker,))
            row = cursor.fetchone()
            if row and row["updated_at"]:
                updated_at = row["updated_at"]
                if isinstance(updated_at, str):
                    updated_at = datetime.fromisoformat(updated_at)
                return (datetime.now() - updated_at).total_seconds() < max_age_hours * 3600
            return False
    
    # ==================== News Operations ====================
    
    def save_news(self, records: List[NewsRecord]) -> int:
        """Save news records."""
        if not records:
            return 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR IGNORE INTO news_cache 
                (ticker, headline, published_at, source, url, summary, category, sentiment, sentiment_score, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    r.ticker, r.headline, r.published_at, r.source, r.url, r.summary,
                    r.category.value if isinstance(r.category, NewsCategory) else r.category,
                    r.sentiment.value if isinstance(r.sentiment, NewsSentiment) else r.sentiment,
                    r.sentiment_score,
                    r.fetched_at or datetime.now()
                )
                for r in records
            ])
            return cursor.rowcount
    
    def get_news(
        self, 
        ticker: str, 
        hours_back: int = 48,
        limit: int = 50
    ) -> List[NewsRecord]:
        """Get recent news for a ticker."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cutoff = datetime.now() - timedelta(hours=hours_back)
            
            cursor.execute("""
                SELECT * FROM news_cache 
                WHERE ticker = ? AND published_at >= ?
                ORDER BY published_at DESC
                LIMIT ?
            """, (ticker, cutoff, limit))
            
            results = []
            for row in cursor.fetchall():
                try:
                    category = NewsCategory(row["category"]) if row["category"] else NewsCategory.OTHER
                except ValueError:
                    category = NewsCategory.OTHER
                
                try:
                    sentiment = NewsSentiment(row["sentiment"]) if row["sentiment"] is not None else NewsSentiment.NEUTRAL
                except ValueError:
                    sentiment = NewsSentiment.NEUTRAL
                
                published_at = row["published_at"]
                if isinstance(published_at, str):
                    published_at = datetime.fromisoformat(published_at)
                
                results.append(NewsRecord(
                    ticker=row["ticker"],
                    headline=row["headline"],
                    published_at=published_at,
                    source=row["source"],
                    url=row["url"],
                    summary=row["summary"],
                    category=category,
                    sentiment=sentiment,
                    sentiment_score=row["sentiment_score"] or 0.0,
                    fetched_at=row["fetched_at"],
                ))
            return results
    
    def is_news_fresh(self, ticker: str, max_age_hours: int = 1) -> bool:
        """Check if news data is fresh enough."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(fetched_at) as last_fetch FROM news_cache WHERE ticker = ?
            """, (ticker,))
            row = cursor.fetchone()
            if row and row["last_fetch"]:
                last_fetch = row["last_fetch"]
                if isinstance(last_fetch, str):
                    last_fetch = datetime.fromisoformat(last_fetch)
                return (datetime.now() - last_fetch).total_seconds() < max_age_hours * 3600
            return False
    
    # ==================== Settings Operations ====================
    
    def save_setting(self, key: str, value: Any) -> bool:
        """Save a setting."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, json.dumps(value), datetime.now()))
            return cursor.rowcount > 0
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row["value"])
                except json.JSONDecodeError:
                    return row["value"]
            return default
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM settings")
            settings = {}
            for row in cursor.fetchall():
                try:
                    settings[row["key"]] = json.loads(row["value"])
                except json.JSONDecodeError:
                    settings[row["key"]] = row["value"]
            return settings
    
    # ==================== Alert Operations ====================
    
    def save_alert(self, alert: AlertRecord) -> int:
        """Save an alert record and return its ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alert_history 
                (ticker, direction, score, confidence, reasons, news_links, alerted_at, notification_method, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.ticker,
                alert.direction.value if isinstance(alert.direction, SignalDirection) else alert.direction,
                alert.score,
                alert.confidence,
                json.dumps(alert.reasons),
                json.dumps(alert.news_links),
                alert.alerted_at or datetime.now(),
                alert.notification_method,
                1 if alert.success else 0,
            ))
            return cursor.lastrowid
    
    def was_alerted_today(self, ticker: str) -> bool:
        """Check if ticker was already alerted today."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cursor.execute("""
                SELECT COUNT(*) as count FROM alert_history 
                WHERE ticker = ? AND alerted_at >= ? AND success = 1
            """, (ticker, today_start))
            row = cursor.fetchone()
            return row["count"] > 0 if row else False
    
    def get_alerts_today_count(self) -> int:
        """Get count of alerts sent today."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cursor.execute("""
                SELECT COUNT(*) as count FROM alert_history 
                WHERE alerted_at >= ? AND success = 1
            """, (today_start,))
            row = cursor.fetchone()
            return row["count"] if row else 0
    
    def get_recent_alerts(self, days: int = 7, limit: int = 100) -> List[AlertRecord]:
        """Get recent alerts."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cutoff = datetime.now() - timedelta(days=days)
            cursor.execute("""
                SELECT * FROM alert_history 
                WHERE alerted_at >= ?
                ORDER BY alerted_at DESC
                LIMIT ?
            """, (cutoff, limit))
            
            results = []
            for row in cursor.fetchall():
                try:
                    direction = SignalDirection(row["direction"])
                except ValueError:
                    direction = SignalDirection.NEUTRAL
                
                results.append(AlertRecord(
                    id=row["id"],
                    ticker=row["ticker"],
                    direction=direction,
                    score=row["score"],
                    confidence=row["confidence"],
                    reasons=json.loads(row["reasons"]) if row["reasons"] else [],
                    news_links=json.loads(row["news_links"]) if row["news_links"] else [],
                    alerted_at=row["alerted_at"],
                    notification_method=row["notification_method"],
                    success=bool(row["success"]),
                ))
            return results
    
    # ==================== Backtest Operations ====================
    
    def save_backtest_run(self, run: BacktestRun) -> int:
        """Save a backtest run and return its ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO backtest_runs 
                (name, tickers, start_date, end_date, holding_period, exit_strategy, 
                 commission, slippage_pct, include_news, params_json, results_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.name,
                json.dumps(run.tickers),
                run.start_date,
                run.end_date,
                run.holding_period,
                run.exit_strategy,
                run.commission,
                run.slippage_pct,
                1 if run.include_news else 0,
                json.dumps({"equity_curve": run.equity_curve}),
                json.dumps(run.to_dict()),
                run.created_at or datetime.now(),
            ))
            return cursor.lastrowid
    
    def get_backtest_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent backtest runs (summary only)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, tickers, start_date, end_date, holding_period, 
                       exit_strategy, include_news, created_at
                FROM backtest_runs 
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "tickers": json.loads(row["tickers"]) if row["tickers"] else [],
                    "start_date": row["start_date"],
                    "end_date": row["end_date"],
                    "holding_period": row["holding_period"],
                    "exit_strategy": row["exit_strategy"],
                    "include_news": bool(row["include_news"]),
                    "created_at": row["created_at"],
                }
                for row in cursor.fetchall()
            ]
    
    def get_backtest_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get full backtest run details."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM backtest_runs WHERE id = ?
            """, (run_id,))
            row = cursor.fetchone()
            if row:
                results = json.loads(row["results_json"]) if row["results_json"] else {}
                return results
            return None
    
    # ==================== Utility Methods ====================
    
    def clear_cache(self, table: str = None):
        """Clear cache tables."""
        tables = ["ohlcv_cache", "fundamentals_cache", "news_cache"]
        if table and table in tables:
            tables = [table]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for t in tables:
                cursor.execute(f"DELETE FROM {t}")
                logger.info(f"Cleared {t}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            stats = {}
            
            cursor.execute("SELECT COUNT(*) as count FROM ohlcv_cache")
            stats["ohlcv_records"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(DISTINCT ticker) as count FROM ohlcv_cache")
            stats["ohlcv_tickers"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM fundamentals_cache")
            stats["fundamentals_records"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM news_cache")
            stats["news_records"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM alert_history")
            stats["alerts"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM backtest_runs")
            stats["backtest_runs"] = cursor.fetchone()["count"]
            
            return stats
