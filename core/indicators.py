"""
Technical Indicators Calculator for Trend Terminal.
Computes EMA, RSI, volume metrics, and returns.
"""

import logging
from typing import Optional, List, Dict, Tuple
from datetime import date

import pandas as pd
import numpy as np

from storage.models import OHLCVRecord, IndicatorValues

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """
    Calculates technical indicators from OHLCV data.
    All calculations are vectorized using pandas for efficiency.
    """
    
    def __init__(
        self,
        ema_short: int = 20,
        ema_long: int = 50,
        rsi_period: int = 14,
        volume_avg_period: int = 30,
    ):
        """
        Initialize indicator calculator.
        
        Args:
            ema_short: Short EMA period (default 20)
            ema_long: Long EMA period (default 50)
            rsi_period: RSI period (default 14)
            volume_avg_period: Volume average period (default 30)
        """
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.rsi_period = rsi_period
        self.volume_avg_period = volume_avg_period
    
    def records_to_dataframe(self, records: List[OHLCVRecord]) -> pd.DataFrame:
        """Convert OHLCV records to DataFrame."""
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame([r.to_dict() for r in records])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            series: Price series
            period: EMA period
            
        Returns:
            EMA series
        """
        return series.ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            series: Price series
            period: SMA period
            
        Returns:
            SMA series
        """
        return series.rolling(window=period).mean()
    
    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Uses the Wilder smoothing method (exponential moving average).
        
        Args:
            series: Price series
            period: RSI period (default 14)
            
        Returns:
            RSI series (0-100)
        """
        # Calculate price changes
        delta = series.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)
        
        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi = rsi.replace([np.inf, -np.inf], np.nan)
        
        return rsi
    
    def calculate_volume_spike(
        self,
        volume: pd.Series,
        avg_period: int = 30,
    ) -> pd.Series:
        """
        Calculate volume spike ratio.
        
        Args:
            volume: Volume series
            avg_period: Period for average volume calculation
            
        Returns:
            Volume spike ratio series
        """
        avg_volume = volume.rolling(window=avg_period).mean()
        spike = volume / avg_volume
        return spike
    
    def calculate_returns(
        self,
        close: pd.Series,
        periods: List[int] = None,
    ) -> Dict[int, pd.Series]:
        """
        Calculate returns for multiple periods.
        
        Args:
            close: Close price series
            periods: List of periods (default [1, 5, 20])
            
        Returns:
            Dict mapping period to returns series
        """
        if periods is None:
            periods = [1, 5, 20]
        
        returns = {}
        for period in periods:
            returns[period] = close.pct_change(periods=period)
        
        return returns
    
    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period
            
        Returns:
            ATR series
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
    
    def calculate_bollinger_bands(
        self,
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            close: Close price series
            period: Period for moving average
            std_dev: Number of standard deviations
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def calculate_macd(
        self,
        close: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD.
        
        Args:
            close: Close price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        fast_ema = self.calculate_ema(close, fast_period)
        slow_ema = self.calculate_ema(close, slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = self.calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_all_indicators(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate all indicators for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added indicator columns
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # EMAs
        df['ema20'] = self.calculate_ema(df['close'], self.ema_short)
        df['ema50'] = self.calculate_ema(df['close'], self.ema_long)
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        
        # Volume metrics
        df['avg_volume_30'] = df['volume'].rolling(window=self.volume_avg_period).mean()
        df['volume_spike'] = df['volume'] / df['avg_volume_30']
        
        # Returns
        returns = self.calculate_returns(df['close'], [1, 5, 20])
        df['return_1d'] = returns[1]
        df['return_5d'] = returns[5]
        df['return_20d'] = returns[20]
        
        # ATR for volatility
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])
        
        return df
    
    def get_latest_indicators(
        self,
        records: List[OHLCVRecord],
        as_of_date: Optional[date] = None,
    ) -> Optional[IndicatorValues]:
        """
        Get the latest indicator values for a list of OHLCV records.
        
        Args:
            records: List of OHLCVRecord
            as_of_date: Optional date to get indicators as of (for backtesting)
            
        Returns:
            IndicatorValues object or None
        """
        if not records:
            return None
        
        df = self.records_to_dataframe(records)
        
        # Filter by date if specified (for backtesting - no look-ahead)
        if as_of_date:
            df = df[df.index.date <= as_of_date]
        
        if df.empty:
            return None
        
        # Calculate indicators
        df = self.calculate_all_indicators(df)
        
        # Get latest row
        latest = df.iloc[-1]
        
        return IndicatorValues(
            ticker=records[0].ticker,
            date=latest.name.date() if hasattr(latest.name, 'date') else latest.name,
            close=float(latest['close']),
            ema20=float(latest['ema20']) if pd.notna(latest['ema20']) else None,
            ema50=float(latest['ema50']) if pd.notna(latest['ema50']) else None,
            rsi=float(latest['rsi']) if pd.notna(latest['rsi']) else None,
            volume=int(latest['volume']),
            avg_volume_30=float(latest['avg_volume_30']) if pd.notna(latest['avg_volume_30']) else None,
            volume_spike=float(latest['volume_spike']) if pd.notna(latest['volume_spike']) else None,
            return_1d=float(latest['return_1d']) if pd.notna(latest['return_1d']) else None,
            return_5d=float(latest['return_5d']) if pd.notna(latest['return_5d']) else None,
            return_20d=float(latest['return_20d']) if pd.notna(latest['return_20d']) else None,
        )
    
    def get_indicator_history(
        self,
        records: List[OHLCVRecord],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Get indicator history as DataFrame.
        
        Args:
            records: List of OHLCVRecord
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with all indicators
        """
        if not records:
            return pd.DataFrame()
        
        df = self.records_to_dataframe(records)
        df = self.calculate_all_indicators(df)
        
        # Apply date filters
        if start_date:
            df = df[df.index.date >= start_date]
        if end_date:
            df = df[df.index.date <= end_date]
        
        return df


def calculate_trend_direction(
    close: float,
    ema20: Optional[float],
    ema50: Optional[float],
    rsi: Optional[float],
) -> str:
    """
    Determine trend direction based on indicators.
    
    Args:
        close: Current close price
        ema20: 20-day EMA
        ema50: 50-day EMA
        rsi: RSI value
        
    Returns:
        "bullish", "bearish", or "neutral"
    """
    if ema20 is None or ema50 is None:
        return "neutral"
    
    # Bullish conditions
    if close > ema20 and close > ema50 and ema20 > ema50:
        if rsi is not None and rsi >= 55:
            return "bullish"
    
    # Bearish conditions
    if close < ema20 and close < ema50 and ema20 < ema50:
        if rsi is not None and rsi <= 45:
            return "bearish"
    
    return "neutral"


def is_trending(
    return_1d: Optional[float],
    volume_spike: Optional[float],
    min_price_change: float = 0.02,
    min_volume_spike: float = 1.5,
) -> bool:
    """
    Check if stock is trending (significant move with volume).
    
    Args:
        return_1d: 1-day return
        volume_spike: Volume spike ratio
        min_price_change: Minimum absolute price change
        min_volume_spike: Minimum volume spike ratio
        
    Returns:
        True if trending
    """
    if return_1d is None or volume_spike is None:
        return False
    
    return abs(return_1d) >= min_price_change and volume_spike >= min_volume_spike
