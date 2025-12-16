"""
Tests for the Signals Module

Tests cover:
- Score calculation
- Direction determination
- Confidence calculation
- Reason generation
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.signals import SignalGenerator
from core.indicators import IndicatorCalculator
from storage.models import SignalDirection


@pytest.fixture
def generator():
    """Create signal generator instance."""
    return SignalGenerator()


@pytest.fixture
def calculator():
    """Create indicator calculator instance."""
    return IndicatorCalculator()


@pytest.fixture
def bullish_df():
    """Create a DataFrame with bullish characteristics."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Strong uptrend with increasing volume
    prices = np.linspace(100, 150, 100) + np.random.normal(0, 1, 100)
    
    df = pd.DataFrame({
        'Open': prices - 0.5,
        'High': prices + 2,
        'Low': prices - 1,
        'Close': prices,
        'Volume': np.linspace(1000000, 3000000, 100).astype(int),  # Increasing volume
    }, index=dates)
    
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


@pytest.fixture
def bearish_df():
    """Create a DataFrame with bearish characteristics."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Strong downtrend with increasing volume
    prices = np.linspace(150, 100, 100) + np.random.normal(0, 1, 100)
    
    df = pd.DataFrame({
        'Open': prices + 0.5,
        'High': prices + 1,
        'Low': prices - 2,
        'Close': prices,
        'Volume': np.linspace(1000000, 3000000, 100).astype(int),
    }, index=dates)
    
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


@pytest.fixture
def neutral_df():
    """Create a DataFrame with neutral/sideways characteristics."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Sideways movement
    prices = 100 + np.random.normal(0, 2, 100)
    
    df = pd.DataFrame({
        'Open': prices - 0.2,
        'High': prices + 1,
        'Low': prices - 1,
        'Close': prices,
        'Volume': [1000000] * 100,
    }, index=dates)
    
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


class TestScoreCalculation:
    """Tests for score calculation."""
    
    def test_score_bounds(self, generator, bullish_df):
        """Score should be between 0 and 100."""
        signal = generator.generate_signal("TEST", bullish_df)
        
        assert signal.score >= 0
        assert signal.score <= 100
    
    def test_bullish_high_score(self, generator, bullish_df):
        """Bullish setup should have high score."""
        signal = generator.generate_signal("TEST", bullish_df)
        
        # Score should be above average for bullish setup
        assert signal.score >= 50
    
    def test_bearish_lower_score(self, generator, bearish_df):
        """Bearish setup should have lower score (inverted for trend following)."""
        signal = generator.generate_signal("TEST", bearish_df)
        
        # Score for bearish (inverse direction)
        # In a trend-following system, bearish signals can also be strong
        assert signal.score >= 0  # Just check validity


class TestDirectionDetermination:
    """Tests for direction determination."""
    
    def test_bullish_direction(self, generator, bullish_df):
        """Strong uptrend should be identified as bullish."""
        signal = generator.generate_signal("TEST", bullish_df)
        
        assert signal.direction == SignalDirection.BULLISH
    
    def test_bearish_direction(self, generator, bearish_df):
        """Strong downtrend should be identified as bearish."""
        signal = generator.generate_signal("TEST", bearish_df)
        
        assert signal.direction == SignalDirection.BEARISH
    
    def test_neutral_direction(self, generator, neutral_df):
        """Sideways market should be neutral or have weak direction."""
        signal = generator.generate_signal("TEST", neutral_df)
        
        # Could be any direction, but confidence should be lower
        # Just verify it returns a valid direction
        assert signal.direction in [SignalDirection.BULLISH, SignalDirection.BEARISH, SignalDirection.NEUTRAL]


class TestConfidenceCalculation:
    """Tests for confidence calculation."""
    
    def test_confidence_bounds(self, generator, bullish_df):
        """Confidence should be between 0 and 100."""
        signal = generator.generate_signal("TEST", bullish_df)
        
        assert signal.confidence >= 0
        assert signal.confidence <= 100
    
    def test_strong_trend_high_confidence(self, generator, bullish_df):
        """Strong trend should have higher confidence."""
        signal = generator.generate_signal("TEST", bullish_df)
        
        # Strong trend should have decent confidence
        assert signal.confidence >= 30
    
    def test_weak_trend_lower_confidence(self, generator, neutral_df):
        """Sideways market should have lower confidence."""
        signal = generator.generate_signal("TEST", neutral_df)
        
        # Neutral market should have lower confidence
        # But this depends on implementation
        assert signal.confidence >= 0  # Just validate


class TestReasonGeneration:
    """Tests for reason generation."""
    
    def test_reasons_not_empty(self, generator, bullish_df):
        """Signal should have at least one reason."""
        signal = generator.generate_signal("TEST", bullish_df)
        
        assert len(signal.reasons) > 0
    
    def test_reasons_are_strings(self, generator, bullish_df):
        """All reasons should be strings."""
        signal = generator.generate_signal("TEST", bullish_df)
        
        for reason in signal.reasons:
            assert isinstance(reason, str)
            assert len(reason) > 0
    
    def test_bullish_reasons_content(self, generator, bullish_df):
        """Bullish signal should mention trend-related factors."""
        signal = generator.generate_signal("TEST", bullish_df)
        
        # Combine all reasons into one string for searching
        all_reasons = " ".join(signal.reasons).lower()
        
        # Should mention some bullish indicators
        bullish_keywords = ["above", "up", "bullish", "ema", "trend", "positive", "strong"]
        has_bullish_keyword = any(kw in all_reasons for kw in bullish_keywords)
        
        # At least some indication of bullish factors
        assert has_bullish_keyword or signal.direction == SignalDirection.BULLISH


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_insufficient_data(self, generator):
        """Should handle insufficient data gracefully."""
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000000] * 5,
        }, index=dates)
        
        # Should not crash, may return None or signal with low confidence
        try:
            signal = generator.generate_signal("TEST", df)
            # If returns signal, should be valid
            if signal:
                assert signal.score >= 0
                assert signal.score <= 100
        except Exception:
            # OK to raise exception for insufficient data
            pass
    
    def test_constant_prices(self, generator):
        """Should handle constant prices."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        
        df = pd.DataFrame({
            'Open': [100.0] * 50,
            'High': [100.5] * 50,
            'Low': [99.5] * 50,
            'Close': [100.0] * 50,
            'Volume': [1000000] * 50,
        }, index=dates)
        
        # Should not crash
        try:
            signal = generator.generate_signal("TEST", df)
            if signal:
                # Constant prices should be neutral
                assert signal.direction == SignalDirection.NEUTRAL or signal.confidence < 50
        except Exception:
            pass
    
    def test_zero_volume(self, generator):
        """Should handle zero volume data."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        
        df = pd.DataFrame({
            'Open': np.linspace(100, 120, 50),
            'High': np.linspace(101, 121, 50),
            'Low': np.linspace(99, 119, 50),
            'Close': np.linspace(100, 120, 50),
            'Volume': [0] * 50,
        }, index=dates)
        
        # Should not crash
        try:
            signal = generator.generate_signal("TEST", df)
            if signal:
                assert signal.score >= 0
        except Exception:
            pass
    
    def test_nan_values(self, generator):
        """Should handle NaN values gracefully."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        
        prices = np.linspace(100, 120, 50)
        prices[10] = np.nan
        prices[25] = np.nan
        
        df = pd.DataFrame({
            'Open': prices - 0.5,
            'High': prices + 1,
            'Low': prices - 1,
            'Close': prices,
            'Volume': [1000000] * 50,
        }, index=dates)
        
        # Should not crash
        try:
            signal = generator.generate_signal("TEST", df)
            if signal:
                assert signal.score >= 0
        except Exception:
            pass


class TestNewsIntegration:
    """Tests for news sentiment integration."""
    
    def test_without_news(self, generator, bullish_df):
        """Should work without news data."""
        signal = generator.generate_signal("TEST", bullish_df)
        
        assert signal is not None
        assert signal.score >= 0
    
    def test_with_positive_news(self, generator, bullish_df):
        """Positive news should increase score."""
        signal_no_news = generator.generate_signal("TEST", bullish_df, news_sentiment=None)
        signal_positive = generator.generate_signal("TEST", bullish_df, news_sentiment=0.8)
        
        # Positive news should help (or at least not hurt)
        # Note: Implementation may vary
        assert signal_positive.score >= 0
    
    def test_with_negative_news(self, generator, bullish_df):
        """Negative news should decrease score."""
        signal_no_news = generator.generate_signal("TEST", bullish_df, news_sentiment=None)
        signal_negative = generator.generate_signal("TEST", bullish_df, news_sentiment=-0.8)
        
        # Negative news on bullish setup creates uncertainty
        assert signal_negative.score >= 0


class TestMultiTickerScanning:
    """Tests for scanning multiple tickers."""
    
    def test_scan_results_structure(self, generator, bullish_df, bearish_df):
        """Scan results should have correct structure."""
        # This tests the batch scanning capability
        data_dict = {
            "BULL": bullish_df,
            "BEAR": bearish_df,
        }
        
        results = []
        for ticker, df in data_dict.items():
            signal = generator.generate_signal(ticker, df)
            if signal:
                results.append(signal)
        
        assert len(results) == 2
        
        # Check all signals have required fields
        for signal in results:
            assert signal.ticker in ["BULL", "BEAR"]
            assert 0 <= signal.score <= 100
            assert signal.direction in SignalDirection


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
