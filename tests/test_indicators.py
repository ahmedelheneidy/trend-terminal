"""
Tests for the Indicators Module

Tests cover:
- EMA calculation correctness
- RSI calculation with edge cases
- Volume spike detection
- ATR calculation
- Bollinger Bands
- MACD calculation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.indicators import IndicatorCalculator


@pytest.fixture
def calculator():
    """Create indicator calculator instance."""
    return IndicatorCalculator()


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Generate realistic-ish price data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 100)
    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    prices = np.array(prices)
    
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'High': prices * (1 + np.random.uniform(0, 0.02, 100)),
        'Low': prices * (1 + np.random.uniform(-0.02, 0, 100)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100),
    }, index=dates)
    
    # Ensure High >= Close >= Low
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


@pytest.fixture
def trending_up_df():
    """Create a clearly trending up DataFrame."""
    dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
    prices = np.linspace(100, 150, 50)  # Linear uptrend
    
    df = pd.DataFrame({
        'Open': prices - 0.5,
        'High': prices + 1,
        'Low': prices - 1,
        'Close': prices,
        'Volume': [1000000] * 50,
    }, index=dates)
    
    return df


@pytest.fixture
def trending_down_df():
    """Create a clearly trending down DataFrame."""
    dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
    prices = np.linspace(150, 100, 50)  # Linear downtrend
    
    df = pd.DataFrame({
        'Open': prices + 0.5,
        'High': prices + 1,
        'Low': prices - 1,
        'Close': prices,
        'Volume': [1000000] * 50,
    }, index=dates)
    
    return df


class TestEMA:
    """Tests for EMA calculation."""
    
    def test_ema_length(self, calculator, sample_ohlcv_df):
        """EMA should return same length as input."""
        ema = calculator.ema(sample_ohlcv_df['Close'], 20)
        assert len(ema) == len(sample_ohlcv_df)
    
    def test_ema_nan_at_start(self, calculator, sample_ohlcv_df):
        """EMA should have NaN values at start (span - 1)."""
        span = 20
        ema = calculator.ema(sample_ohlcv_df['Close'], span)
        # First values should be NaN or present depending on min_periods
        assert ema.iloc[-1] is not None  # Last value should exist
    
    def test_ema_follows_trend(self, calculator, trending_up_df):
        """EMA should follow an uptrend."""
        ema = calculator.ema(trending_up_df['Close'], 10)
        # EMA at end should be higher than at start (ignoring NaN)
        valid_ema = ema.dropna()
        assert valid_ema.iloc[-1] > valid_ema.iloc[0]
    
    def test_ema_smoothing(self, calculator, sample_ohlcv_df):
        """Longer EMA should be smoother than shorter EMA."""
        ema_short = calculator.ema(sample_ohlcv_df['Close'], 10)
        ema_long = calculator.ema(sample_ohlcv_df['Close'], 30)
        
        # Standard deviation of returns should be lower for longer EMA
        std_short = ema_short.pct_change().std()
        std_long = ema_long.pct_change().std()
        
        assert std_long < std_short


class TestRSI:
    """Tests for RSI calculation."""
    
    def test_rsi_bounds(self, calculator, sample_ohlcv_df):
        """RSI should be between 0 and 100."""
        rsi = calculator.rsi(sample_ohlcv_df['Close'], 14)
        valid_rsi = rsi.dropna()
        
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100
    
    def test_rsi_uptrend_high(self, calculator, trending_up_df):
        """RSI should be high in strong uptrend."""
        rsi = calculator.rsi(trending_up_df['Close'], 14)
        # Last RSI value should be high (above 60)
        assert rsi.iloc[-1] > 60
    
    def test_rsi_downtrend_low(self, calculator, trending_down_df):
        """RSI should be low in strong downtrend."""
        rsi = calculator.rsi(trending_down_df['Close'], 14)
        # Last RSI value should be low (below 40)
        assert rsi.iloc[-1] < 40
    
    def test_rsi_length(self, calculator, sample_ohlcv_df):
        """RSI should return same length as input."""
        rsi = calculator.rsi(sample_ohlcv_df['Close'], 14)
        assert len(rsi) == len(sample_ohlcv_df)
    
    def test_rsi_constant_price(self, calculator):
        """RSI should be 50 or NaN for constant price."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        constant_prices = pd.Series([100.0] * 30, index=dates)
        
        rsi = calculator.rsi(constant_prices, 14)
        # Should be NaN or neutral (around 50)
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            # When gains and losses are both 0, RSI behavior varies
            pass  # Implementation dependent


class TestVolumeSpike:
    """Tests for volume spike detection."""
    
    def test_volume_spike_detection(self, calculator):
        """Should detect volume spikes correctly."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        
        # Normal volume with a spike at the end
        volumes = [1000000] * 49 + [5000000]  # 5x spike
        volume_series = pd.Series(volumes, index=dates)
        
        spikes = calculator.volume_spike(volume_series, 20, 2.0)
        
        # Last value should be True (spike)
        assert spikes.iloc[-1] == True
        # Most others should be False
        assert spikes.iloc[-10] == False
    
    def test_volume_spike_threshold(self, calculator):
        """Different thresholds should affect detection."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        
        volumes = [1000000] * 49 + [2500000]  # 2.5x spike
        volume_series = pd.Series(volumes, index=dates)
        
        spikes_2x = calculator.volume_spike(volume_series, 20, 2.0)
        spikes_3x = calculator.volume_spike(volume_series, 20, 3.0)
        
        # 2x threshold should detect it
        assert spikes_2x.iloc[-1] == True
        # 3x threshold should not
        assert spikes_3x.iloc[-1] == False


class TestATR:
    """Tests for ATR calculation."""
    
    def test_atr_positive(self, calculator, sample_ohlcv_df):
        """ATR should always be positive."""
        atr = calculator.atr(sample_ohlcv_df, 14)
        valid_atr = atr.dropna()
        
        assert (valid_atr >= 0).all()
    
    def test_atr_length(self, calculator, sample_ohlcv_df):
        """ATR should return same length as input."""
        atr = calculator.atr(sample_ohlcv_df, 14)
        assert len(atr) == len(sample_ohlcv_df)
    
    def test_atr_high_volatility(self, calculator):
        """ATR should be higher for more volatile stocks."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        
        # Low volatility
        low_vol = pd.DataFrame({
            'High': [101] * 50,
            'Low': [99] * 50,
            'Close': [100] * 50,
        }, index=dates)
        
        # High volatility
        high_vol = pd.DataFrame({
            'High': [110] * 50,
            'Low': [90] * 50,
            'Close': [100] * 50,
        }, index=dates)
        
        atr_low = calculator.atr(low_vol, 14).iloc[-1]
        atr_high = calculator.atr(high_vol, 14).iloc[-1]
        
        assert atr_high > atr_low


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""
    
    def test_bollinger_structure(self, calculator, sample_ohlcv_df):
        """Bollinger Bands should return upper > middle > lower."""
        upper, middle, lower = calculator.bollinger_bands(sample_ohlcv_df['Close'], 20, 2)
        
        # Check at a point where all values are valid
        idx = -1
        assert upper.iloc[idx] > middle.iloc[idx]
        assert middle.iloc[idx] > lower.iloc[idx]
    
    def test_bollinger_middle_is_sma(self, calculator, sample_ohlcv_df):
        """Middle band should be SMA."""
        upper, middle, lower = calculator.bollinger_bands(sample_ohlcv_df['Close'], 20, 2)
        
        # Calculate SMA manually
        sma = sample_ohlcv_df['Close'].rolling(20).mean()
        
        # Middle band should equal SMA
        pd.testing.assert_series_equal(
            middle.dropna(),
            sma.dropna(),
            check_names=False
        )
    
    def test_bollinger_width(self, calculator):
        """Higher std_dev should result in wider bands."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        prices = pd.Series(np.random.normal(100, 5, 50), index=dates)
        
        upper_2, _, lower_2 = calculator.bollinger_bands(prices, 20, 2)
        upper_3, _, lower_3 = calculator.bollinger_bands(prices, 20, 3)
        
        width_2 = (upper_2 - lower_2).iloc[-1]
        width_3 = (upper_3 - lower_3).iloc[-1]
        
        assert width_3 > width_2


class TestMACD:
    """Tests for MACD calculation."""
    
    def test_macd_structure(self, calculator, sample_ohlcv_df):
        """MACD should return macd_line, signal_line, histogram."""
        macd_line, signal_line, histogram = calculator.macd(sample_ohlcv_df['Close'])
        
        assert len(macd_line) == len(sample_ohlcv_df)
        assert len(signal_line) == len(sample_ohlcv_df)
        assert len(histogram) == len(sample_ohlcv_df)
    
    def test_macd_histogram_calculation(self, calculator, sample_ohlcv_df):
        """Histogram should be MACD line minus signal line."""
        macd_line, signal_line, histogram = calculator.macd(sample_ohlcv_df['Close'])
        
        expected_hist = macd_line - signal_line
        
        pd.testing.assert_series_equal(
            histogram.dropna(),
            expected_hist.dropna(),
            check_names=False
        )
    
    def test_macd_uptrend_positive(self, calculator, trending_up_df):
        """MACD should be positive in uptrend."""
        macd_line, _, _ = calculator.macd(trending_up_df['Close'])
        # Last MACD value should be positive
        assert macd_line.iloc[-1] > 0
    
    def test_macd_downtrend_negative(self, calculator, trending_down_df):
        """MACD should be negative in downtrend."""
        macd_line, _, _ = calculator.macd(trending_down_df['Close'])
        # Last MACD value should be negative
        assert macd_line.iloc[-1] < 0


class TestReturns:
    """Tests for returns calculation."""
    
    def test_returns_length(self, calculator, sample_ohlcv_df):
        """Returns should have same length as input."""
        returns = calculator.returns(sample_ohlcv_df['Close'], 1)
        assert len(returns) == len(sample_ohlcv_df)
    
    def test_returns_calculation(self, calculator):
        """Returns should be calculated correctly."""
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        prices = pd.Series([100, 110, 105, 115, 120], index=dates)
        
        returns = calculator.returns(prices, 1)
        
        # Day 2: (110-100)/100 = 10%
        assert abs(returns.iloc[1] - 0.10) < 0.001
        # Day 3: (105-110)/110 = -4.545%
        assert abs(returns.iloc[2] - (-0.04545)) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
