"""
Backtesting Engine for Trend Terminal.
Implements realistic backtesting with no look-ahead bias.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import pandas as pd

from storage.models import (
    BacktestRun,
    BacktestTrade,
    BacktestMetrics,
    SignalDirection,
    OHLCVRecord,
)
from core.indicators import IndicatorCalculator
from core.signals import SignalGenerator, get_config_for_style
from core.news_classifier import NewsClassifier

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    holding_period: int = 5  # Days
    exit_strategy: str = "fixed"  # "fixed" or "signal_flip"
    commission_per_trade: float = 0.0
    slippage_pct: float = 0.001  # 0.1%
    entry_timing: str = "next_open"  # "next_open" or "next_close"
    include_news: bool = True
    min_signal_score: float = 40.0
    trading_style: str = "swing"  # scalp, swing, position, long_term


# Pre-configured backtest configs for different trading styles
SCALP_BACKTEST = BacktestConfig(
    holding_period=1,         # 1 day max
    exit_strategy="signal_flip",
    slippage_pct=0.002,       # Higher slippage for quick trades
    min_signal_score=55,
    trading_style="scalp",
)

SWING_BACKTEST = BacktestConfig(
    holding_period=5,         # 5 days
    exit_strategy="fixed",
    slippage_pct=0.001,
    min_signal_score=50,
    trading_style="swing",
)

POSITION_BACKTEST = BacktestConfig(
    holding_period=20,        # 20 days
    exit_strategy="signal_flip",
    slippage_pct=0.0005,
    min_signal_score=45,
    trading_style="position",
)

LONG_TERM_BACKTEST = BacktestConfig(
    holding_period=60,        # 60 days
    exit_strategy="fixed",
    slippage_pct=0.0003,
    min_signal_score=40,
    trading_style="long_term",
)


def get_backtest_config_for_style(style: str) -> BacktestConfig:
    """Get backtest config for a trading style."""
    style = style.lower()
    if style in ("scalp", "scalping", "day"):
        return SCALP_BACKTEST
    elif style in ("swing", "swing_trade"):
        return SWING_BACKTEST
    elif style in ("position", "position_trade"):
        return POSITION_BACKTEST
    elif style in ("long_term", "longterm", "investment"):
        return LONG_TERM_BACKTEST
    else:
        return SWING_BACKTEST


class BacktestEngine:
    """
    Backtesting engine with strict no look-ahead enforcement.
    """
    
    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        indicator_calculator: Optional[IndicatorCalculator] = None,
        signal_generator: Optional[SignalGenerator] = None,
        news_classifier: Optional[NewsClassifier] = None,
        trading_style: Optional[str] = None,
    ):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
            indicator_calculator: Indicator calculator instance
            signal_generator: Signal generator instance
            news_classifier: News classifier instance
            trading_style: Trading style for preset configs
        """
        # Use trading_style preset if provided and no explicit config
        if trading_style and not config:
            self.config = get_backtest_config_for_style(trading_style)
        else:
            self.config = config or BacktestConfig()
        
        self.indicators = indicator_calculator or IndicatorCalculator()
        self.signals = signal_generator or SignalGenerator()
        self.news_classifier = news_classifier or NewsClassifier()
    
    def _get_data_up_to_date(
        self,
        df: pd.DataFrame,
        as_of_date: date,
    ) -> pd.DataFrame:
        """
        Get data only up to a specific date (no look-ahead).
        
        Args:
            df: Full DataFrame
            as_of_date: Date to filter up to
            
        Returns:
            Filtered DataFrame
        """
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index)
        return df_copy[df_copy.index.date <= as_of_date]
    
    def _generate_signal_for_date(
        self,
        ticker: str,
        df: pd.DataFrame,
        signal_date: date,
        include_news: bool = True,
    ) -> Optional[Tuple[SignalDirection, float]]:
        """
        Generate signal for a specific date using only data available up to that date.
        
        Args:
            ticker: Stock ticker
            df: Full DataFrame with OHLCV
            signal_date: Date to generate signal for
            include_news: Whether to include news in signal
            
        Returns:
            Tuple of (direction, score) or None
        """
        # Get data only up to signal date (no look-ahead)
        df_up_to_date = self._get_data_up_to_date(df, signal_date)
        
        if len(df_up_to_date) < 50:  # Need enough data for indicators
            return None
        
        # Calculate indicators
        df_with_indicators = self.indicators.calculate_all_indicators(df_up_to_date)
        
        if df_with_indicators.empty:
            return None
        
        latest = df_with_indicators.iloc[-1]
        
        # Get indicator values
        from storage.models import IndicatorValues
        ind_values = IndicatorValues(
            ticker=ticker,
            date=signal_date,
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
        
        # Generate signal (without news for backtesting simplicity)
        signal = self.signals.generate_signal(
            ticker=ticker,
            indicators=ind_values,
            news_list=None,  # Skip news for backtest (would need historical news)
            fundamentals=None,
            as_of_date=signal_date,
        )
        
        return signal.direction, signal.final_score
    
    def _calculate_trade_return(
        self,
        entry_price: float,
        exit_price: float,
        direction: SignalDirection,
    ) -> float:
        """
        Calculate trade return based on direction.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            direction: Trade direction
            
        Returns:
            Return percentage
        """
        if direction == SignalDirection.BULLISH:
            return (exit_price - entry_price) / entry_price
        elif direction == SignalDirection.BEARISH:
            return (entry_price - exit_price) / entry_price
        else:
            return 0.0
    
    def _apply_costs(self, gross_return: float) -> float:
        """
        Apply transaction costs to return.
        
        Args:
            gross_return: Gross return before costs
            
        Returns:
            Net return after costs
        """
        # Apply slippage on entry and exit
        slippage_cost = self.config.slippage_pct * 2
        
        # Add commission (as percentage of trade)
        total_cost = slippage_cost + (self.config.commission_per_trade * 2 / 10000)
        
        return gross_return - total_cost
    
    def run_backtest(
        self,
        ticker: str,
        ohlcv_data: List[OHLCVRecord],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[BacktestTrade]:
        """
        Run backtest for a single ticker.
        
        Args:
            ticker: Stock ticker
            ohlcv_data: List of OHLCV records
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            List of BacktestTrade objects
        """
        if not ohlcv_data:
            return []
        
        # Convert to DataFrame
        df = self.indicators.records_to_dataframe(ohlcv_data)
        
        if df.empty:
            return []
        
        # Apply date filters
        if start_date:
            df = df[df.index.date >= start_date]
        if end_date:
            df = df[df.index.date <= end_date]
        
        if len(df) < 60:  # Need enough data
            logger.warning(f"Insufficient data for {ticker} backtest")
            return []
        
        # Calculate indicators for full period
        df = self.indicators.calculate_all_indicators(df)
        
        trades = []
        in_position = False
        current_trade = None
        
        # Get list of trading days
        trading_days = df.index.tolist()
        
        # Iterate through trading days
        i = 50  # Start after enough data for indicators
        while i < len(trading_days) - self.config.holding_period:
            current_date = trading_days[i].date()
            
            if in_position:
                # Check exit conditions
                entry_idx = trading_days.index(pd.Timestamp(current_trade["entry_date"]))
                days_held = i - entry_idx
                
                exit_trade = False
                exit_reason = ""
                
                if self.config.exit_strategy == "fixed":
                    if days_held >= self.config.holding_period:
                        exit_trade = True
                        exit_reason = "holding_period"
                
                elif self.config.exit_strategy == "signal_flip":
                    # Check for signal flip
                    result = self._generate_signal_for_date(ticker, df, current_date)
                    if result:
                        new_direction, _ = result
                        if new_direction != current_trade["direction"]:
                            exit_trade = True
                            exit_reason = "signal_flip"
                    
                    # Also exit after max holding period
                    if days_held >= self.config.holding_period * 2:
                        exit_trade = True
                        exit_reason = "max_holding_period"
                
                if exit_trade:
                    # Exit at today's open or close
                    if self.config.entry_timing == "next_open":
                        exit_price = df.iloc[i]['open']
                    else:
                        exit_price = df.iloc[i]['close']
                    
                    # Calculate return
                    gross_return = self._calculate_trade_return(
                        current_trade["entry_price"],
                        exit_price,
                        current_trade["direction"],
                    )
                    net_return = self._apply_costs(gross_return)
                    
                    trade = BacktestTrade(
                        ticker=ticker,
                        entry_date=current_trade["entry_date"],
                        entry_price=current_trade["entry_price"],
                        exit_date=current_date,
                        exit_price=exit_price,
                        direction=current_trade["direction"],
                        signal_score=current_trade["score"],
                        return_pct=gross_return,
                        return_after_costs=net_return,
                        holding_days=days_held,
                        exit_reason=exit_reason,
                    )
                    trades.append(trade)
                    
                    in_position = False
                    current_trade = None
            
            else:
                # Look for entry signal
                result = self._generate_signal_for_date(ticker, df, current_date)
                
                if result:
                    direction, score = result
                    
                    # Only enter if score meets threshold and direction is not neutral
                    if (
                        score >= self.config.min_signal_score and
                        direction != SignalDirection.NEUTRAL
                    ):
                        # Entry at next day's open
                        if i + 1 < len(trading_days):
                            entry_date = trading_days[i + 1].date()
                            
                            if self.config.entry_timing == "next_open":
                                entry_price = df.iloc[i + 1]['open']
                            else:
                                entry_price = df.iloc[i + 1]['close']
                            
                            in_position = True
                            current_trade = {
                                "entry_date": entry_date,
                                "entry_price": entry_price,
                                "direction": direction,
                                "score": score,
                            }
            
            i += 1
        
        # Close any open position at end
        if in_position and current_trade:
            exit_idx = len(trading_days) - 1
            exit_date = trading_days[exit_idx].date()
            exit_price = df.iloc[exit_idx]['close']
            
            entry_idx = trading_days.index(pd.Timestamp(current_trade["entry_date"]))
            days_held = exit_idx - entry_idx
            
            gross_return = self._calculate_trade_return(
                current_trade["entry_price"],
                exit_price,
                current_trade["direction"],
            )
            net_return = self._apply_costs(gross_return)
            
            trade = BacktestTrade(
                ticker=ticker,
                entry_date=current_trade["entry_date"],
                entry_price=current_trade["entry_price"],
                exit_date=exit_date,
                exit_price=exit_price,
                direction=current_trade["direction"],
                signal_score=current_trade["score"],
                return_pct=gross_return,
                return_after_costs=net_return,
                holding_days=days_held,
                exit_reason="end_of_data",
            )
            trades.append(trade)
        
        return trades
    
    def run_multi_ticker_backtest(
        self,
        ticker_data: Dict[str, List[OHLCVRecord]],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        progress_callback=None,
    ) -> BacktestRun:
        """
        Run backtest for multiple tickers.
        
        Args:
            ticker_data: Dict mapping ticker to OHLCV data
            start_date: Backtest start date
            end_date: Backtest end date
            progress_callback: Optional progress callback
            
        Returns:
            BacktestRun object with results
        """
        all_trades = []
        tickers = list(ticker_data.keys())
        total = len(tickers)
        
        for i, ticker in enumerate(tickers):
            ohlcv = ticker_data[ticker]
            trades = self.run_backtest(ticker, ohlcv, start_date, end_date)
            all_trades.extend(trades)
            
            if progress_callback:
                progress_callback((i + 1) / total)
        
        # Calculate metrics
        from backtest.metrics import calculate_metrics, calculate_equity_curve
        
        metrics = calculate_metrics(all_trades)
        equity_curve = calculate_equity_curve(all_trades)
        
        return BacktestRun(
            name=f"Backtest {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            holding_period=self.config.holding_period,
            exit_strategy=self.config.exit_strategy,
            commission=self.config.commission_per_trade,
            slippage_pct=self.config.slippage_pct,
            include_news=self.config.include_news,
            trades=all_trades,
            metrics=metrics,
            equity_curve=equity_curve,
            created_at=datetime.now(),
        )
