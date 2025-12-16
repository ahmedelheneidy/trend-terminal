"""
Data Models for Trend Terminal.
Dataclasses representing database records and domain objects.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from enum import Enum


class SignalDirection(Enum):
    """Signal direction enum."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class NewsCategory(Enum):
    """News category enum."""
    EARNINGS = "earnings"
    UPGRADE_DOWNGRADE = "upgrade_downgrade"
    MA = "ma"  # Mergers & Acquisitions
    REGULATORY = "regulatory"
    LEGAL = "legal"
    PRODUCT = "product"
    OTHER = "other"


class NewsSentiment(Enum):
    """News sentiment enum."""
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


@dataclass
class OHLCVRecord:
    """OHLCV price data record."""
    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None
    fetched_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date.isoformat() if isinstance(self.date, date) else self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "adj_close": self.adj_close,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
        }


@dataclass
class FundamentalsRecord:
    """Company fundamentals record."""
    ticker: str
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    company_name: Optional[str] = None
    currency: Optional[str] = None
    exchange: Optional[str] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "market_cap": self.market_cap,
            "sector": self.sector,
            "industry": self.industry,
            "company_name": self.company_name,
            "currency": self.currency,
            "exchange": self.exchange,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class NewsRecord:
    """News article record."""
    ticker: str
    headline: str
    published_at: datetime
    source: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    category: NewsCategory = NewsCategory.OTHER
    sentiment: NewsSentiment = NewsSentiment.NEUTRAL
    sentiment_score: float = 0.0
    fetched_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "headline": self.headline,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "source": self.source,
            "url": self.url,
            "summary": self.summary,
            "category": self.category.value if isinstance(self.category, NewsCategory) else self.category,
            "sentiment": self.sentiment.value if isinstance(self.sentiment, NewsSentiment) else self.sentiment,
            "sentiment_score": self.sentiment_score,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
        }


@dataclass
class IndicatorValues:
    """Computed indicator values for a ticker."""
    ticker: str
    date: date
    close: float
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    rsi: Optional[float] = None
    volume: Optional[int] = None
    avg_volume_30: Optional[float] = None
    volume_spike: Optional[float] = None
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_20d: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date.isoformat() if isinstance(self.date, date) else self.date,
            "close": self.close,
            "ema20": self.ema20,
            "ema50": self.ema50,
            "rsi": self.rsi,
            "volume": self.volume,
            "avg_volume_30": self.avg_volume_30,
            "volume_spike": self.volume_spike,
            "return_1d": self.return_1d,
            "return_5d": self.return_5d,
            "return_20d": self.return_20d,
        }


@dataclass
class SignalRecord:
    """Trading signal record."""
    ticker: str
    date: date
    direction: SignalDirection
    trend_score: float  # 0-100
    volume_score: float  # 0-100
    news_score: float  # 0-100 (normalized)
    final_score: float  # 0-100 (weighted combination)
    confidence: float  # 0-1
    is_trending: bool
    reasons: List[str] = field(default_factory=list)
    indicators: Optional[IndicatorValues] = None
    news_items: List[NewsRecord] = field(default_factory=list)
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date.isoformat() if isinstance(self.date, date) else self.date,
            "direction": self.direction.value if isinstance(self.direction, SignalDirection) else self.direction,
            "trend_score": self.trend_score,
            "volume_score": self.volume_score,
            "news_score": self.news_score,
            "final_score": self.final_score,
            "confidence": self.confidence,
            "is_trending": self.is_trending,
            "reasons": self.reasons,
            "indicators": self.indicators.to_dict() if self.indicators else None,
            "news_items": [n.to_dict() for n in self.news_items],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class AlertRecord:
    """Alert history record."""
    id: Optional[int] = None
    ticker: str = ""
    direction: SignalDirection = SignalDirection.NEUTRAL
    score: float = 0.0
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    news_links: List[str] = field(default_factory=list)
    alerted_at: Optional[datetime] = None
    notification_method: str = "telegram"
    success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ticker": self.ticker,
            "direction": self.direction.value if isinstance(self.direction, SignalDirection) else self.direction,
            "score": self.score,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "news_links": self.news_links,
            "alerted_at": self.alerted_at.isoformat() if self.alerted_at else None,
            "notification_method": self.notification_method,
            "success": self.success,
        }


@dataclass
class BacktestTrade:
    """Individual trade in a backtest."""
    ticker: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    direction: SignalDirection
    signal_score: float
    return_pct: float
    return_after_costs: float
    holding_days: int
    exit_reason: str  # "holding_period", "signal_flip", "end_of_data"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "entry_date": self.entry_date.isoformat() if isinstance(self.entry_date, date) else self.entry_date,
            "entry_price": self.entry_price,
            "exit_date": self.exit_date.isoformat() if isinstance(self.exit_date, date) else self.exit_date,
            "exit_price": self.exit_price,
            "direction": self.direction.value if isinstance(self.direction, SignalDirection) else self.direction,
            "signal_score": self.signal_score,
            "return_pct": self.return_pct,
            "return_after_costs": self.return_after_costs,
            "holding_days": self.holding_days,
            "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    cumulative_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_holding_days: float = 0.0
    turnover: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "avg_winner": self.avg_winner,
            "avg_loser": self.avg_loser,
            "cumulative_return": self.cumulative_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "avg_holding_days": self.avg_holding_days,
            "turnover": self.turnover,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
        }


@dataclass
class BacktestRun:
    """Complete backtest run record."""
    id: Optional[int] = None
    name: Optional[str] = None
    tickers: List[str] = field(default_factory=list)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    holding_period: int = 5
    exit_strategy: str = "fixed"  # "fixed" or "signal_flip"
    commission: float = 0.0
    slippage_pct: float = 0.001
    include_news: bool = True
    trades: List[BacktestTrade] = field(default_factory=list)
    metrics: Optional[BacktestMetrics] = None
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tickers": self.tickers,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "holding_period": self.holding_period,
            "exit_strategy": self.exit_strategy,
            "commission": self.commission,
            "slippage_pct": self.slippage_pct,
            "include_news": self.include_news,
            "trades": [t.to_dict() for t in self.trades],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "equity_curve": self.equity_curve,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class MarketOverview:
    """Market overview summary."""
    index_ticker: str
    index_name: str
    last_price: float
    change_1d: float
    change_1d_pct: float
    change_5d_pct: float
    volume: int
    avg_volume: float
    trend: SignalDirection
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_ticker": self.index_ticker,
            "index_name": self.index_name,
            "last_price": self.last_price,
            "change_1d": self.change_1d,
            "change_1d_pct": self.change_1d_pct,
            "change_5d_pct": self.change_5d_pct,
            "volume": self.volume,
            "avg_volume": self.avg_volume,
            "trend": self.trend.value if isinstance(self.trend, SignalDirection) else self.trend,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class ScanResult:
    """Result of a stock scan."""
    ticker: str
    company_name: Optional[str]
    sector: Optional[str]
    last_price: float
    change_1d_pct: float
    change_5d_pct: float
    volume: int
    volume_spike: float
    market_cap: Optional[float]
    trend_score: float
    news_score: float
    final_score: float
    direction: SignalDirection
    confidence: float
    reasons: List[str]
    has_recent_news: bool
    news_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "last_price": self.last_price,
            "change_1d_pct": self.change_1d_pct,
            "change_5d_pct": self.change_5d_pct,
            "volume": self.volume,
            "volume_spike": self.volume_spike,
            "market_cap": self.market_cap,
            "trend_score": self.trend_score,
            "news_score": self.news_score,
            "final_score": self.final_score,
            "direction": self.direction.value if isinstance(self.direction, SignalDirection) else self.direction,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "has_recent_news": self.has_recent_news,
            "news_count": self.news_count,
        }
