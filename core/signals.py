"""
Signal Generator for Trend Terminal.
Generates trading signals based on technical indicators and news.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass

from storage.models import (
    SignalRecord,
    SignalDirection,
    IndicatorValues,
    NewsRecord,
    ScanResult,
    FundamentalsRecord,
)
from core.indicators import IndicatorCalculator, calculate_trend_direction, is_trending
from core.news_classifier import NewsClassifier

logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    # Weights
    technical_weight: float = 0.60
    volume_weight: float = 0.20
    news_weight: float = 0.20
    
    # Trend score points
    price_above_ema20_points: int = 20
    price_above_ema50_points: int = 20
    ema20_above_ema50_points: int = 20
    rsi_in_band_points: int = 20
    volume_spike_points: int = 20
    
    # RSI bands
    rsi_bullish_min: float = 55
    rsi_bullish_max: float = 70
    rsi_bearish_min: float = 30
    rsi_bearish_max: float = 45
    
    # Trending gate
    min_price_change_1d: float = 0.02  # 2%
    min_volume_spike: float = 1.5
    
    # Confidence thresholds
    high_confidence_threshold: float = 75
    medium_confidence_threshold: float = 50
    
    # Minimum score
    min_signal_score: float = 40
    
    # Trading style
    trading_style: str = "swing"  # scalp, swing, position, long_term


# Pre-configured signal configs for different trading styles
SCALP_CONFIG = SignalConfig(
    technical_weight=0.70,
    volume_weight=0.25,
    news_weight=0.05,
    rsi_bullish_min=60,
    rsi_bullish_max=80,
    rsi_bearish_min=20,
    rsi_bearish_max=40,
    min_price_change_1d=0.01,  # 1% - more sensitive
    min_volume_spike=2.0,      # Higher volume requirement
    min_signal_score=50,
    trading_style="scalp",
)

SWING_CONFIG = SignalConfig(
    technical_weight=0.55,
    volume_weight=0.25,
    news_weight=0.20,
    rsi_bullish_min=50,
    rsi_bullish_max=70,
    rsi_bearish_min=30,
    rsi_bearish_max=50,
    min_price_change_1d=0.02,  # 2%
    min_volume_spike=1.5,
    min_signal_score=45,
    trading_style="swing",
)

POSITION_CONFIG = SignalConfig(
    technical_weight=0.50,
    volume_weight=0.20,
    news_weight=0.30,
    rsi_bullish_min=45,
    rsi_bullish_max=65,
    rsi_bearish_min=35,
    rsi_bearish_max=55,
    min_price_change_1d=0.03,  # 3%
    min_volume_spike=1.2,
    min_signal_score=40,
    trading_style="position",
)

LONG_TERM_CONFIG = SignalConfig(
    technical_weight=0.40,
    volume_weight=0.15,
    news_weight=0.45,
    rsi_bullish_min=40,
    rsi_bullish_max=60,
    rsi_bearish_min=40,
    rsi_bearish_max=60,
    min_price_change_1d=0.05,  # 5%
    min_volume_spike=1.0,
    min_signal_score=35,
    trading_style="long_term",
)


def get_config_for_style(style: str) -> SignalConfig:
    """Get signal config for a trading style."""
    style = style.lower()
    if style in ("scalp", "scalping", "day", "daytrading"):
        return SCALP_CONFIG
    elif style in ("swing", "swing_trade"):
        return SWING_CONFIG
    elif style in ("position", "position_trade", "medium"):
        return POSITION_CONFIG
    elif style in ("long_term", "longterm", "investment", "invest"):
        return LONG_TERM_CONFIG
    else:
        return SWING_CONFIG  # Default


class SignalGenerator:
    """
    Generates trading signals from indicators and news.
    """
    
    def __init__(
        self,
        config: Optional[SignalConfig] = None,
        indicator_calculator: Optional[IndicatorCalculator] = None,
        news_classifier: Optional[NewsClassifier] = None,
        trading_style: Optional[str] = None,
    ):
        """
        Initialize signal generator.
        
        Args:
            config: Signal configuration
            indicator_calculator: Indicator calculator instance
            news_classifier: News classifier instance
            trading_style: Trading style ('scalp', 'swing', 'position', 'long_term')
        """
        # If trading_style provided, use that config instead
        if trading_style and not config:
            self.config = get_config_for_style(trading_style)
        else:
            self.config = config or SignalConfig()
        
        self.indicators = indicator_calculator or IndicatorCalculator()
        self.news_classifier = news_classifier or NewsClassifier()
    
    def set_trading_style(self, style: str) -> None:
        """Change the trading style configuration."""
        self.config = get_config_for_style(style)
    
    def calculate_trend_score(
        self,
        indicators: IndicatorValues,
    ) -> Tuple[float, List[str]]:
        """
        Calculate trend score from indicators.
        
        Args:
            indicators: IndicatorValues object
            
        Returns:
            Tuple of (score 0-100, list of reasons)
        """
        score = 0.0
        reasons = []
        
        close = indicators.close
        ema20 = indicators.ema20
        ema50 = indicators.ema50
        rsi = indicators.rsi
        volume_spike = indicators.volume_spike
        
        # Price above EMA20
        if ema20 is not None and close > ema20:
            score += self.config.price_above_ema20_points
            pct_above = ((close - ema20) / ema20) * 100
            reasons.append(f"Price {pct_above:.1f}% above 20-day EMA")
        elif ema20 is not None and close < ema20:
            pct_below = ((ema20 - close) / ema20) * 100
            reasons.append(f"Price {pct_below:.1f}% below 20-day EMA")
        
        # Price above EMA50
        if ema50 is not None and close > ema50:
            score += self.config.price_above_ema50_points
            pct_above = ((close - ema50) / ema50) * 100
            reasons.append(f"Price {pct_above:.1f}% above 50-day EMA")
        elif ema50 is not None and close < ema50:
            pct_below = ((ema50 - close) / ema50) * 100
            reasons.append(f"Price {pct_below:.1f}% below 50-day EMA")
        
        # EMA20 above EMA50 (trend confirmation)
        if ema20 is not None and ema50 is not None:
            if ema20 > ema50:
                score += self.config.ema20_above_ema50_points
                reasons.append("Short-term trend above long-term (bullish alignment)")
            else:
                reasons.append("Short-term trend below long-term (bearish alignment)")
        
        # RSI in bullish or bearish band
        if rsi is not None:
            if self.config.rsi_bullish_min <= rsi <= self.config.rsi_bullish_max:
                score += self.config.rsi_in_band_points
                reasons.append(f"RSI at {rsi:.1f} indicates bullish momentum")
            elif self.config.rsi_bearish_min <= rsi <= self.config.rsi_bearish_max:
                # For bearish signals, also count this as "in band"
                score += self.config.rsi_in_band_points
                reasons.append(f"RSI at {rsi:.1f} indicates bearish momentum")
            elif rsi > 70:
                reasons.append(f"RSI at {rsi:.1f} suggests overbought conditions")
            elif rsi < 30:
                reasons.append(f"RSI at {rsi:.1f} suggests oversold conditions")
            else:
                reasons.append(f"RSI at {rsi:.1f} (neutral)")
        
        # Volume spike
        if volume_spike is not None and volume_spike >= self.config.min_volume_spike:
            score += self.config.volume_spike_points
            reasons.append(f"Volume {volume_spike:.1f}x above 30-day average (high interest)")
        elif volume_spike is not None:
            reasons.append(f"Volume at {volume_spike:.1f}x average")
        
        return score, reasons
    
    def calculate_volume_score(
        self,
        indicators: IndicatorValues,
    ) -> Tuple[float, List[str]]:
        """
        Calculate volume score.
        
        Args:
            indicators: IndicatorValues object
            
        Returns:
            Tuple of (score 0-100, list of reasons)
        """
        reasons = []
        volume_spike = indicators.volume_spike
        
        if volume_spike is None:
            return 50.0, ["Volume data unavailable"]
        
        # Scale volume spike to score
        # 0.5x -> 0, 1x -> 50, 2x -> 100, capped
        if volume_spike < 0.5:
            score = 0.0
            reasons.append(f"Very low volume ({volume_spike:.1f}x average)")
        elif volume_spike < 1.0:
            score = (volume_spike - 0.5) * 100  # 0-50
            reasons.append(f"Below average volume ({volume_spike:.1f}x)")
        elif volume_spike < 2.0:
            score = 50 + (volume_spike - 1.0) * 50  # 50-100
            reasons.append(f"Above average volume ({volume_spike:.1f}x)")
        else:
            score = 100.0
            reasons.append(f"Significant volume surge ({volume_spike:.1f}x average)")
        
        return min(score, 100.0), reasons
    
    def determine_direction(
        self,
        indicators: IndicatorValues,
    ) -> SignalDirection:
        """
        Determine signal direction.
        
        Args:
            indicators: IndicatorValues object
            
        Returns:
            SignalDirection enum
        """
        close = indicators.close
        ema20 = indicators.ema20
        ema50 = indicators.ema50
        rsi = indicators.rsi
        
        # Need all indicators for direction
        if ema20 is None or ema50 is None:
            return SignalDirection.NEUTRAL
        
        # Bullish conditions
        bullish = (
            close > ema20 and
            close > ema50 and
            ema20 > ema50 and
            (rsi is None or rsi >= self.config.rsi_bullish_min)
        )
        
        # Bearish conditions
        bearish = (
            close < ema20 and
            close < ema50 and
            ema20 < ema50 and
            (rsi is None or rsi <= self.config.rsi_bearish_max)
        )
        
        if bullish:
            return SignalDirection.BULLISH
        elif bearish:
            return SignalDirection.BEARISH
        else:
            return SignalDirection.NEUTRAL
    
    def calculate_confidence(
        self,
        final_score: float,
        direction: SignalDirection,
        is_trending: bool,
    ) -> float:
        """
        Calculate confidence level.
        
        Args:
            final_score: Combined final score
            direction: Signal direction
            is_trending: Whether the stock is trending
            
        Returns:
            Confidence value 0-1
        """
        if direction == SignalDirection.NEUTRAL:
            return 0.3
        
        # Base confidence from score
        base_confidence = final_score / 100
        
        # Boost if trending
        if is_trending:
            base_confidence = min(base_confidence * 1.2, 1.0)
        
        return base_confidence
    
    def generate_signal(
        self,
        ticker: str,
        indicators: IndicatorValues,
        news_list: Optional[List[NewsRecord]] = None,
        fundamentals: Optional[FundamentalsRecord] = None,
        as_of_date: Optional[date] = None,
    ) -> SignalRecord:
        """
        Generate a complete signal for a ticker.
        
        Args:
            ticker: Stock ticker
            indicators: Computed indicator values
            news_list: Optional list of news records
            fundamentals: Optional fundamentals record
            as_of_date: Date for signal (for backtesting)
            
        Returns:
            SignalRecord object
        """
        # Calculate component scores
        trend_score, trend_reasons = self.calculate_trend_score(indicators)
        volume_score, volume_reasons = self.calculate_volume_score(indicators)
        
        # News score
        if news_list:
            # Classify news if not already
            classified_news = self.news_classifier.classify_batch(news_list)
            reference_time = datetime.combine(as_of_date, datetime.max.time()) if as_of_date else None
            news_score, top_news = self.news_classifier.calculate_news_score(
                classified_news, reference_time
            )
            news_reasons = []
            for n in top_news[:2]:  # Top 2 news
                sentiment_emoji = "🟢" if n.sentiment.value > 0 else "🔴" if n.sentiment.value < 0 else "⚪"
                news_reasons.append(f"{sentiment_emoji} {n.headline[:80]}...")
        else:
            news_score = 50.0  # Neutral
            top_news = []
            news_reasons = ["No recent news available"]
        
        # Calculate final score
        final_score = (
            trend_score * self.config.technical_weight +
            volume_score * self.config.volume_weight +
            news_score * self.config.news_weight
        )
        
        # Determine direction
        direction = self.determine_direction(indicators)
        
        # Check if trending
        stock_is_trending = is_trending(
            indicators.return_1d,
            indicators.volume_spike,
            self.config.min_price_change_1d,
            self.config.min_volume_spike,
        )
        
        # Calculate confidence
        confidence = self.calculate_confidence(final_score, direction, stock_is_trending)
        
        # Compile reasons
        all_reasons = []
        
        # Add price movement reason
        if indicators.return_1d is not None:
            pct = indicators.return_1d * 100
            if pct > 0:
                all_reasons.append(f"📈 Up {pct:.1f}% today")
            else:
                all_reasons.append(f"📉 Down {abs(pct):.1f}% today")
        
        # Add selected trend reasons (most important)
        all_reasons.extend(trend_reasons[:3])
        
        # Add volume reason
        if volume_reasons:
            all_reasons.append(volume_reasons[0])
        
        # Add news reasons
        all_reasons.extend(news_reasons[:2])
        
        # Ensure 3-7 reasons
        all_reasons = all_reasons[:7]
        if len(all_reasons) < 3:
            all_reasons.append(f"Signal strength: {final_score:.0f}/100")
        
        return SignalRecord(
            ticker=ticker,
            date=as_of_date or indicators.date,
            direction=direction,
            trend_score=trend_score,
            volume_score=volume_score,
            news_score=news_score,
            final_score=final_score,
            confidence=confidence,
            is_trending=stock_is_trending,
            reasons=all_reasons,
            indicators=indicators,
            news_items=top_news if news_list else [],
            created_at=datetime.now(),
        )
    
    def generate_scan_result(
        self,
        signal: SignalRecord,
        fundamentals: Optional[FundamentalsRecord] = None,
    ) -> ScanResult:
        """
        Convert a signal to a scan result for display.
        
        Args:
            signal: SignalRecord object
            fundamentals: Optional fundamentals
            
        Returns:
            ScanResult object
        """
        indicators = signal.indicators
        
        return ScanResult(
            ticker=signal.ticker,
            company_name=fundamentals.company_name if fundamentals else None,
            sector=fundamentals.sector if fundamentals else None,
            last_price=indicators.close if indicators else 0.0,
            change_1d_pct=(indicators.return_1d or 0) * 100,
            change_5d_pct=(indicators.return_5d or 0) * 100,
            volume=indicators.volume if indicators else 0,
            volume_spike=indicators.volume_spike or 0,
            market_cap=fundamentals.market_cap if fundamentals else None,
            trend_score=signal.trend_score,
            news_score=signal.news_score,
            final_score=signal.final_score,
            direction=signal.direction,
            confidence=signal.confidence,
            reasons=signal.reasons,
            has_recent_news=len(signal.news_items) > 0,
            news_count=len(signal.news_items),
        )


def format_signal_for_display(signal: SignalRecord) -> Dict:
    """
    Format signal for UI display.
    
    Args:
        signal: SignalRecord object
        
    Returns:
        Dict with formatted values
    """
    direction_emoji = {
        SignalDirection.BULLISH: "🟢 Bullish",
        SignalDirection.BEARISH: "🔴 Bearish",
        SignalDirection.NEUTRAL: "⚪ Neutral",
    }
    
    confidence_label = "High" if signal.confidence >= 0.7 else "Medium" if signal.confidence >= 0.5 else "Low"
    
    return {
        "ticker": signal.ticker,
        "direction": direction_emoji.get(signal.direction, "⚪ Neutral"),
        "score": f"{signal.final_score:.0f}/100",
        "confidence": f"{confidence_label} ({signal.confidence:.0%})",
        "trend_score": f"{signal.trend_score:.0f}",
        "volume_score": f"{signal.volume_score:.0f}",
        "news_score": f"{signal.news_score:.0f}",
        "is_trending": "Yes" if signal.is_trending else "No",
        "reasons": signal.reasons,
    }


def get_signal_summary(signals: List[SignalRecord]) -> Dict:
    """
    Get summary statistics for a list of signals.
    
    Args:
        signals: List of SignalRecord objects
        
    Returns:
        Summary dict
    """
    if not signals:
        return {
            "total": 0,
            "bullish": 0,
            "bearish": 0,
            "neutral": 0,
            "avg_score": 0,
            "trending_count": 0,
        }
    
    bullish = sum(1 for s in signals if s.direction == SignalDirection.BULLISH)
    bearish = sum(1 for s in signals if s.direction == SignalDirection.BEARISH)
    neutral = len(signals) - bullish - bearish
    trending = sum(1 for s in signals if s.is_trending)
    
    return {
        "total": len(signals),
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "avg_score": sum(s.final_score for s in signals) / len(signals),
        "trending_count": trending,
    }
