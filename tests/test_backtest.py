"""
Tests for the Backtest Module

Tests cover:
- Basic backtest execution
- No look-ahead bias
- Metric calculations
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

from backtest.engine import BacktestEngine, BacktestConfig
from backtest.metrics import calculate_metrics, calculate_equity_curve, calculate_monthly_returns


@pytest.fixture
def engine():
    """Create backtest engine with default config."""
    return BacktestEngine()


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Generate trending up then down pattern
    mid = n // 2
    prices_up = np.linspace(100, 150, mid)
    prices_down = np.linspace(150, 110, n - mid)
    prices = np.concatenate([prices_up, prices_down])
    
    # Add some noise
    prices = prices + np.random.normal(0, 2, n)
    
    df = pd.DataFrame({
        'Open': prices - np.random.uniform(0, 1, n),
        'High': prices + np.random.uniform(1, 3, n),
        'Low': prices - np.random.uniform(1, 3, n),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, n),
    }, index=dates)
    
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


@pytest.fixture
def trending_up_ohlcv():
    """Create consistently trending up OHLCV data."""
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    prices = np.linspace(100, 200, 200)
    
    df = pd.DataFrame({
        'Open': prices - 0.5,
        'High': prices + 1,
        'Low': prices - 1,
        'Close': prices,
        'Volume': [2000000] * 200,
    }, index=dates)
    
    return df


@pytest.fixture
def sample_signals(sample_ohlcv):
    """Create sample signals for testing."""
    signals = pd.DataFrame(index=sample_ohlcv.index)
    signals['score'] = np.random.uniform(40, 90, len(sample_ohlcv))
    signals['direction'] = np.where(signals['score'] > 65, 'bullish', 'neutral')
    signals['confidence'] = np.random.uniform(30, 80, len(sample_ohlcv))
    return signals


class TestBacktestExecution:
    """Tests for basic backtest execution."""
    
    def test_backtest_runs(self, engine, sample_ohlcv, sample_signals):
        """Backtest should run without errors."""
        result = engine.run_backtest("TEST", sample_ohlcv, sample_signals)
        
        assert result is not None
        assert "metrics" in result
        assert "trades" in result
        assert "equity_curve" in result
    
    def test_backtest_metrics_structure(self, engine, sample_ohlcv, sample_signals):
        """Backtest metrics should have expected structure."""
        result = engine.run_backtest("TEST", sample_ohlcv, sample_signals)
        metrics = result["metrics"]
        
        required_keys = [
            "total_return",
            "total_trades",
            "win_rate",
            "max_drawdown",
        ]
        
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"
    
    def test_backtest_with_config(self, sample_ohlcv, sample_signals):
        """Should respect configuration parameters."""
        config = BacktestConfig(
            entry_threshold=70,
            exit_strategy="fixed_days",
            hold_days=10,
            transaction_cost=0.002,
            slippage=0.001,
        )
        
        engine = BacktestEngine(config)
        result = engine.run_backtest("TEST", sample_ohlcv, sample_signals)
        
        assert result is not None


class TestNoLookAheadBias:
    """Tests to verify no look-ahead bias."""
    
    def test_signals_before_trades(self, engine, sample_ohlcv, sample_signals):
        """Trades should only happen after signals, not before."""
        result = engine.run_backtest("TEST", sample_ohlcv, sample_signals)
        trades = result["trades"]
        
        for trade in trades:
            signal_date = trade.get("signal_date")
            entry_date = trade.get("entry_date")
            
            if signal_date and entry_date:
                # Entry should be on or after signal date
                assert entry_date >= signal_date
    
    def test_no_future_data_in_signals(self, engine, sample_ohlcv):
        """Signals should only use data available at signal time."""
        # Create signals that only look at past data
        signals = pd.DataFrame(index=sample_ohlcv.index)
        
        # Calculate a 10-day moving average (only uses past data)
        signals['score'] = sample_ohlcv['Close'].rolling(10).mean().shift(1)  # Shift to avoid same-day
        signals['score'] = (signals['score'] / sample_ohlcv['Close'] * 100).fillna(50)
        signals['direction'] = np.where(signals['score'] > 50, 'bullish', 'bearish')
        signals['confidence'] = 60
        
        result = engine.run_backtest("TEST", sample_ohlcv, signals)
        
        # Should complete without errors
        assert result is not None
    
    def test_entry_price_at_entry_date(self, engine, sample_ohlcv, sample_signals):
        """Entry price should match the actual price on entry date."""
        result = engine.run_backtest("TEST", sample_ohlcv, sample_signals)
        trades = result["trades"]
        
        for trade in trades:
            entry_date = trade.get("entry_date")
            entry_price = trade.get("entry_price")
            
            if entry_date and entry_price:
                # Price should be close to actual OHLC on that date
                if entry_date in sample_ohlcv.index:
                    actual_close = sample_ohlcv.loc[entry_date, 'Close']
                    actual_open = sample_ohlcv.loc[entry_date, 'Open']
                    
                    # Entry should be between low and high of that day
                    assert entry_price >= sample_ohlcv.loc[entry_date, 'Low'] * 0.99
                    assert entry_price <= sample_ohlcv.loc[entry_date, 'High'] * 1.01


class TestMetricCalculations:
    """Tests for metric calculation accuracy."""
    
    def test_win_rate_bounds(self, engine, sample_ohlcv, sample_signals):
        """Win rate should be between 0 and 100."""
        result = engine.run_backtest("TEST", sample_ohlcv, sample_signals)
        
        win_rate = result["metrics"]["win_rate"]
        assert 0 <= win_rate <= 100
    
    def test_max_drawdown_bounds(self, engine, sample_ohlcv, sample_signals):
        """Max drawdown should be between 0 and 100."""
        result = engine.run_backtest("TEST", sample_ohlcv, sample_signals)
        
        max_dd = result["metrics"]["max_drawdown"]
        assert 0 <= max_dd <= 100
    
    def test_profit_factor_calculation(self):
        """Profit factor should be correctly calculated."""
        trades = [
            {"pnl_pct": 10},  # Win
            {"pnl_pct": 5},   # Win
            {"pnl_pct": -3},  # Loss
            {"pnl_pct": -2},  # Loss
        ]
        
        metrics = calculate_metrics(trades, 10000)
        
        # Profit factor = gross profit / gross loss = 15 / 5 = 3.0
        assert abs(metrics["profit_factor"] - 3.0) < 0.01
    
    def test_sharpe_ratio_calculation(self):
        """Sharpe ratio should be reasonable."""
        # Create trades with known returns
        np.random.seed(42)
        trades = [{"pnl_pct": r} for r in np.random.normal(1, 2, 50)]
        
        metrics = calculate_metrics(trades, 10000)
        
        # Sharpe should be finite
        assert np.isfinite(metrics.get("sharpe_ratio", 0))


class TestEquityCurve:
    """Tests for equity curve calculation."""
    
    def test_equity_curve_length(self):
        """Equity curve should match trade count + 1."""
        trades = [
            {"entry_date": datetime(2023, 1, 1), "exit_date": datetime(2023, 1, 10), "pnl_pct": 5},
            {"entry_date": datetime(2023, 1, 15), "exit_date": datetime(2023, 1, 25), "pnl_pct": -2},
            {"entry_date": datetime(2023, 2, 1), "exit_date": datetime(2023, 2, 10), "pnl_pct": 3},
        ]
        
        curve = calculate_equity_curve(trades, 10000)
        
        # Should have initial point + one point per trade
        assert len(curve) == len(trades) + 1
    
    def test_equity_curve_starts_at_initial(self):
        """Equity curve should start at initial capital."""
        initial = 10000
        trades = [
            {"entry_date": datetime(2023, 1, 1), "exit_date": datetime(2023, 1, 10), "pnl_pct": 5},
        ]
        
        curve = calculate_equity_curve(trades, initial)
        
        assert curve[0]["equity"] == initial
    
    def test_equity_curve_reflects_trades(self):
        """Equity curve should reflect trade outcomes."""
        initial = 10000
        trades = [
            {"entry_date": datetime(2023, 1, 1), "exit_date": datetime(2023, 1, 10), "pnl_pct": 10},
        ]
        
        curve = calculate_equity_curve(trades, initial)
        
        # After 10% gain, equity should be 11000
        assert abs(curve[-1]["equity"] - 11000) < 1


class TestMonthlyReturns:
    """Tests for monthly returns calculation."""
    
    def test_monthly_returns_structure(self):
        """Monthly returns should have proper structure."""
        trades = [
            {"exit_date": datetime(2023, 1, 15), "pnl_pct": 5},
            {"exit_date": datetime(2023, 1, 25), "pnl_pct": 3},
            {"exit_date": datetime(2023, 2, 10), "pnl_pct": -2},
            {"exit_date": datetime(2023, 3, 5), "pnl_pct": 4},
        ]
        
        monthly = calculate_monthly_returns(trades)
        
        assert "2023-01" in monthly or len(monthly) > 0
    
    def test_monthly_returns_aggregation(self):
        """Monthly returns should aggregate correctly."""
        trades = [
            {"exit_date": datetime(2023, 1, 15), "pnl_pct": 10},
            {"exit_date": datetime(2023, 1, 25), "pnl_pct": 5},
        ]
        
        monthly = calculate_monthly_returns(trades)
        
        # January should have combined return of trades
        jan_return = monthly.get("2023-01", 0)
        # Should be approximately 15.5% (compounded: 1.10 * 1.05 = 1.155)
        assert abs(jan_return - 15.5) < 1


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_no_signals_above_threshold(self, engine, sample_ohlcv):
        """Should handle case with no qualifying signals."""
        signals = pd.DataFrame(index=sample_ohlcv.index)
        signals['score'] = 30  # All below typical threshold
        signals['direction'] = 'neutral'
        signals['confidence'] = 20
        
        result = engine.run_backtest("TEST", sample_ohlcv, signals)
        
        # Should complete with zero trades
        assert result["metrics"]["total_trades"] == 0
    
    def test_all_signals_above_threshold(self, engine, sample_ohlcv):
        """Should handle case with all qualifying signals."""
        signals = pd.DataFrame(index=sample_ohlcv.index)
        signals['score'] = 90  # All above threshold
        signals['direction'] = 'bullish'
        signals['confidence'] = 90
        
        result = engine.run_backtest("TEST", sample_ohlcv, signals)
        
        # Should complete (may have limited trades due to overlap)
        assert result is not None
    
    def test_empty_dataframe(self, engine):
        """Should handle empty data gracefully."""
        empty_df = pd.DataFrame()
        empty_signals = pd.DataFrame()
        
        try:
            result = engine.run_backtest("TEST", empty_df, empty_signals)
            # If it returns, should have zero trades
            assert result["metrics"]["total_trades"] == 0
        except (ValueError, KeyError):
            # OK to raise error for empty input
            pass
    
    def test_single_day_data(self, engine):
        """Should handle single day of data."""
        single_day = pd.DataFrame({
            'Open': [100],
            'High': [101],
            'Low': [99],
            'Close': [100.5],
            'Volume': [1000000],
        }, index=[datetime(2023, 1, 1)])
        
        signals = pd.DataFrame({
            'score': [80],
            'direction': ['bullish'],
            'confidence': [70],
        }, index=[datetime(2023, 1, 1)])
        
        try:
            result = engine.run_backtest("TEST", single_day, signals)
            # Should not crash, may have 0 trades
            assert result is not None
        except (ValueError, IndexError):
            # OK to raise error for insufficient data
            pass


class TestMultiTickerBacktest:
    """Tests for multi-ticker backtesting."""
    
    def test_multi_ticker_runs(self, engine, sample_ohlcv, sample_signals):
        """Multi-ticker backtest should run without errors."""
        data_dict = {
            "AAPL": sample_ohlcv,
            "GOOGL": sample_ohlcv.copy(),
        }
        
        signal_dict = {
            "AAPL": sample_signals,
            "GOOGL": sample_signals.copy(),
        }
        
        result = engine.run_multi_ticker_backtest(data_dict, signal_dict)
        
        assert result is not None
        assert "combined_metrics" in result
        assert "per_ticker" in result
    
    def test_multi_ticker_aggregation(self, engine, sample_ohlcv, sample_signals):
        """Multi-ticker results should aggregate correctly."""
        data_dict = {
            "AAPL": sample_ohlcv,
            "GOOGL": sample_ohlcv.copy(),
        }
        
        signal_dict = {
            "AAPL": sample_signals,
            "GOOGL": sample_signals.copy(),
        }
        
        result = engine.run_multi_ticker_backtest(data_dict, signal_dict)
        
        # Combined trades should be sum of per-ticker trades
        combined_trades = result["combined_metrics"]["total_trades"]
        per_ticker_trades = sum(
            r["metrics"]["total_trades"] 
            for r in result["per_ticker"].values()
        )
        
        assert combined_trades == per_ticker_trades


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
