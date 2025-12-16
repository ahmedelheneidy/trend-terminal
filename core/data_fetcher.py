"""
Data Fetcher for Trend Terminal.
Handles fetching OHLCV data, fundamentals, and news from various sources.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple
import time

import yfinance as yf
import pandas as pd
import feedparser
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from storage.database import DatabaseManager
from storage.cache import CacheManager
from storage.models import (
    OHLCVRecord,
    FundamentalsRecord,
    NewsRecord,
    NewsCategory,
    NewsSentiment,
)

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches market data from yfinance and news from various sources.
    Implements caching to avoid rate limits.
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        """
        Initialize data fetcher.
        
        Args:
            cache_manager: CacheManager instance
            db_manager: DatabaseManager instance (used if cache_manager not provided)
        """
        if cache_manager:
            self.cache = cache_manager
        else:
            self.cache = CacheManager(db_manager or DatabaseManager())
        
        self._rate_limit_delay = 0.1  # Seconds between API calls
        self._last_api_call = 0
    
    def _rate_limit(self):
        """Implement rate limiting."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_api_call = time.time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_ohlcv(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> List[OHLCVRecord]:
        """
        Fetch OHLCV data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            force_refresh: Force fetch even if cached
            
        Returns:
            List of OHLCVRecord objects
        """
        # Check cache first
        if not force_refresh and not self.cache.needs_ohlcv_update(ticker):
            cached = self.cache.get_ohlcv(ticker)
            if cached:
                logger.debug(f"Using cached OHLCV data for {ticker}")
                return cached
        
        logger.info(f"Fetching OHLCV data for {ticker}")
        self._rate_limit()
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No OHLCV data returned for {ticker}")
                return []
            
            records = []
            now = datetime.now()
            
            for idx, row in df.iterrows():
                # Handle timezone-aware datetime
                if hasattr(idx, 'tz') and idx.tz is not None:
                    record_date = idx.tz_localize(None).date()
                else:
                    record_date = idx.date() if hasattr(idx, 'date') else idx
                
                records.append(OHLCVRecord(
                    ticker=ticker.upper(),
                    date=record_date,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                    adj_close=float(row.get('Adj Close', row['Close'])),
                    fetched_at=now,
                ))
            
            # Save to cache
            if records:
                self.cache.save_ohlcv(records)
                logger.info(f"Cached {len(records)} OHLCV records for {ticker}")
            
            return records
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {ticker}: {e}")
            # Return cached data if available
            cached = self.cache.get_ohlcv(ticker)
            if cached:
                logger.info(f"Returning stale cached data for {ticker}")
                return cached
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_fundamentals(
        self,
        ticker: str,
        force_refresh: bool = False,
    ) -> Optional[FundamentalsRecord]:
        """
        Fetch fundamental data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            force_refresh: Force fetch even if cached
            
        Returns:
            FundamentalsRecord or None
        """
        # Check cache first
        if not force_refresh and not self.cache.needs_fundamentals_update(ticker):
            cached = self.cache.get_fundamentals(ticker)
            if cached:
                logger.debug(f"Using cached fundamentals for {ticker}")
                return cached
        
        logger.info(f"Fetching fundamentals for {ticker}")
        self._rate_limit()
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                logger.warning(f"No fundamentals data returned for {ticker}")
                return None
            
            record = FundamentalsRecord(
                ticker=ticker.upper(),
                market_cap=info.get('marketCap'),
                sector=info.get('sector'),
                industry=info.get('industry'),
                company_name=info.get('longName') or info.get('shortName'),
                currency=info.get('currency'),
                exchange=info.get('exchange'),
                updated_at=datetime.now(),
            )
            
            # Save to cache
            self.cache.save_fundamentals(record)
            logger.info(f"Cached fundamentals for {ticker}")
            
            return record
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            # Return cached data if available
            cached = self.cache.get_fundamentals(ticker)
            if cached:
                logger.info(f"Returning stale cached fundamentals for {ticker}")
                return cached
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_news_yfinance(
        self,
        ticker: str,
        force_refresh: bool = False,
    ) -> List[NewsRecord]:
        """
        Fetch news from yfinance.
        
        Args:
            ticker: Stock ticker symbol
            force_refresh: Force fetch even if cached
            
        Returns:
            List of NewsRecord objects
        """
        # Check cache first
        if not force_refresh and not self.cache.needs_news_update(ticker):
            cached = self.cache.get_news(ticker)
            if cached:
                logger.debug(f"Using cached news for {ticker}")
                return cached
        
        logger.info(f"Fetching news for {ticker} from yfinance")
        self._rate_limit()
        
        records = []
        now = datetime.now()
        
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if news:
                for item in news:
                    # Parse publish time
                    pub_time = item.get('providerPublishTime')
                    if pub_time:
                        published_at = datetime.fromtimestamp(pub_time)
                    else:
                        published_at = now
                    
                    # Skip old news
                    if (now - published_at).days > 7:
                        continue
                    
                    records.append(NewsRecord(
                        ticker=ticker.upper(),
                        headline=item.get('title', ''),
                        published_at=published_at,
                        source=item.get('publisher'),
                        url=item.get('link'),
                        summary=item.get('summary'),
                        category=NewsCategory.OTHER,  # Will be classified later
                        sentiment=NewsSentiment.NEUTRAL,
                        sentiment_score=0.0,
                        fetched_at=now,
                    ))
            
        except Exception as e:
            logger.error(f"Error fetching yfinance news for {ticker}: {e}")
        
        return records
    
    def fetch_news_google_rss(
        self,
        ticker: str,
        company_name: Optional[str] = None,
    ) -> List[NewsRecord]:
        """
        Fetch news from Google News RSS feed.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Company name for better search results
            
        Returns:
            List of NewsRecord objects
        """
        logger.info(f"Fetching Google News for {ticker}")
        
        records = []
        now = datetime.now()
        
        # Build search query
        search_term = f"{ticker} stock"
        if company_name:
            search_term = f"{company_name} OR {ticker} stock"
        
        # Encode for URL
        search_term = requests.utils.quote(search_term)
        
        try:
            url = f"https://news.google.com/rss/search?q={search_term}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:20]:  # Limit to 20 articles
                # Parse publish time
                pub_time = entry.get('published_parsed')
                if pub_time:
                    published_at = datetime(*pub_time[:6])
                else:
                    published_at = now
                
                # Skip old news
                if (now - published_at).days > 7:
                    continue
                
                # Extract source from title (Google News format: "Title - Source")
                title = entry.get('title', '')
                source = None
                if ' - ' in title:
                    parts = title.rsplit(' - ', 1)
                    title = parts[0]
                    source = parts[1] if len(parts) > 1 else None
                
                records.append(NewsRecord(
                    ticker=ticker.upper(),
                    headline=title,
                    published_at=published_at,
                    source=source,
                    url=entry.get('link'),
                    summary=entry.get('summary'),
                    category=NewsCategory.OTHER,
                    sentiment=NewsSentiment.NEUTRAL,
                    sentiment_score=0.0,
                    fetched_at=now,
                ))
            
        except Exception as e:
            logger.error(f"Error fetching Google News for {ticker}: {e}")
        
        return records
    
    def fetch_news(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        force_refresh: bool = False,
        include_google: bool = True,
    ) -> List[NewsRecord]:
        """
        Fetch news from all available sources.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Company name for better search
            force_refresh: Force fetch even if cached
            include_google: Include Google News RSS
            
        Returns:
            Combined list of NewsRecord objects
        """
        # Check cache first
        if not force_refresh and not self.cache.needs_news_update(ticker):
            cached = self.cache.get_news(ticker)
            if cached:
                logger.debug(f"Using cached news for {ticker}")
                return cached
        
        all_news = []
        
        # Fetch from yfinance
        yf_news = self.fetch_news_yfinance(ticker, force_refresh=True)
        all_news.extend(yf_news)
        
        # Fetch from Google News if enabled and yfinance didn't return much
        if include_google and len(yf_news) < 5:
            google_news = self.fetch_news_google_rss(ticker, company_name)
            all_news.extend(google_news)
        
        # Deduplicate by headline similarity
        seen_headlines = set()
        unique_news = []
        for news in all_news:
            headline_key = news.headline.lower()[:50]  # First 50 chars
            if headline_key not in seen_headlines:
                seen_headlines.add(headline_key)
                unique_news.append(news)
        
        # Save to cache
        if unique_news:
            self.cache.save_news(unique_news)
            logger.info(f"Cached {len(unique_news)} news items for {ticker}")
        
        return unique_news
    
    def fetch_batch_ohlcv(
        self,
        tickers: List[str],
        period: str = "1y",
        force_refresh: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, List[OHLCVRecord]]:
        """
        Fetch OHLCV data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            period: Data period
            force_refresh: Force fetch even if cached
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            Dict mapping ticker to list of OHLCVRecord
        """
        results = {}
        total = len(tickers)
        
        for i, ticker in enumerate(tickers):
            try:
                results[ticker] = self.fetch_ohlcv(ticker, period, force_refresh=force_refresh)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                results[ticker] = []
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def fetch_batch_fundamentals(
        self,
        tickers: List[str],
        force_refresh: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Optional[FundamentalsRecord]]:
        """
        Fetch fundamentals for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            force_refresh: Force fetch
            progress_callback: Optional progress callback
            
        Returns:
            Dict mapping ticker to FundamentalsRecord
        """
        results = {}
        total = len(tickers)
        
        for i, ticker in enumerate(tickers):
            try:
                results[ticker] = self.fetch_fundamentals(ticker, force_refresh=force_refresh)
            except Exception as e:
                logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
                results[ticker] = None
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get the latest close price for a ticker."""
        ohlcv = self.fetch_ohlcv(ticker, period="5d")
        if ohlcv:
            return ohlcv[-1].close
        return None
    
    def get_price_dataframe(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Get OHLCV data as a pandas DataFrame.
        
        Args:
            ticker: Stock ticker
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with OHLCV data
        """
        records = self.fetch_ohlcv(ticker)
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame([r.to_dict() for r in records])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Apply date filters
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        return df
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status.
        
        Returns:
            Dict with market status info
        """
        now = datetime.now()
        
        # Simple market hours check (US Eastern)
        # This is simplified - real implementation would use proper timezone handling
        is_weekday = now.weekday() < 5
        
        # Approximate market hours (9:30 AM - 4:00 PM ET)
        # Assuming local time is close to ET for simplicity
        is_market_hours = 9 <= now.hour < 16
        
        return {
            "timestamp": now.isoformat(),
            "is_market_open": is_weekday and is_market_hours,
            "is_weekday": is_weekday,
            "note": "Market hours are approximate. Use proper market calendar for production.",
        }
