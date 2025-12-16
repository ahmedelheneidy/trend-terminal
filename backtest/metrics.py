"""
Backtest Metrics Calculator for Trend Terminal.
Calculates performance metrics from backtest trades.
"""

import logging
from typing import List, Dict, Any
from datetime import date
import numpy as np

from storage.models import BacktestTrade, BacktestMetrics

logger = logging.getLogger(__name__)


def calculate_metrics(trades: List[BacktestTrade]) -> BacktestMetrics:
    """
    Calculate performance metrics from a list of trades.
    
    Args:
        trades: List of BacktestTrade objects
        
    Returns:
        BacktestMetrics object
    """
    if not trades:
        return BacktestMetrics()
    
    returns = [t.return_after_costs for t in trades]
    winning_trades = [t for t in trades if t.return_after_costs > 0]
    losing_trades = [t for t in trades if t.return_after_costs <= 0]
    
    total_trades = len(trades)
    win_count = len(winning_trades)
    lose_count = len(losing_trades)
    
    # Win rate
    win_rate = win_count / total_trades if total_trades > 0 else 0.0
    
    # Average returns
    avg_return = np.mean(returns) if returns else 0.0
    avg_winner = np.mean([t.return_after_costs for t in winning_trades]) if winning_trades else 0.0
    avg_loser = np.mean([t.return_after_costs for t in losing_trades]) if losing_trades else 0.0
    
    # Cumulative return (compounded)
    cumulative = 1.0
    for r in returns:
        cumulative *= (1 + r)
    cumulative_return = cumulative - 1
    
    # Max drawdown
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    
    peak = equity[0]
    max_drawdown = 0.0
    for e in equity:
        if e > peak:
            peak = e
        drawdown = (peak - e) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Sharpe ratio (simplified - assuming 0 risk-free rate)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 / 5)  # Annualized
    else:
        sharpe_ratio = None
    
    # Profit factor
    gross_profits = sum(t.return_after_costs for t in winning_trades) if winning_trades else 0
    gross_losses = abs(sum(t.return_after_costs for t in losing_trades)) if losing_trades else 0
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else None
    
    # Average holding days
    avg_holding_days = np.mean([t.holding_days for t in trades]) if trades else 0.0
    
    # Turnover (trades per year assuming 252 trading days)
    if trades:
        first_trade = min(t.entry_date for t in trades)
        last_trade = max(t.exit_date for t in trades)
        days_span = (last_trade - first_trade).days
        if days_span > 0:
            turnover = (total_trades / days_span) * 252
        else:
            turnover = 0.0
    else:
        turnover = 0.0
    
    return BacktestMetrics(
        total_trades=total_trades,
        winning_trades=win_count,
        losing_trades=lose_count,
        win_rate=win_rate,
        avg_return=avg_return,
        avg_winner=avg_winner,
        avg_loser=avg_loser,
        cumulative_return=cumulative_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        profit_factor=profit_factor,
        avg_holding_days=avg_holding_days,
        turnover=turnover,
        total_commission=0.0,  # Would need to calculate from config
        total_slippage=0.0,
    )


def calculate_equity_curve(
    trades: List[BacktestTrade],
    initial_capital: float = 10000.0,
) -> List[Dict[str, Any]]:
    """
    Calculate equity curve from trades.
    
    Args:
        trades: List of BacktestTrade objects
        initial_capital: Starting capital
        
    Returns:
        List of dicts with date and equity value
    """
    if not trades:
        return []
    
    # Sort trades by exit date
    sorted_trades = sorted(trades, key=lambda t: t.exit_date)
    
    equity_curve = [{
        "date": sorted_trades[0].entry_date.isoformat(),
        "equity": initial_capital,
        "return": 0.0,
    }]
    
    current_equity = initial_capital
    
    for trade in sorted_trades:
        trade_return = trade.return_after_costs
        current_equity *= (1 + trade_return)
        
        equity_curve.append({
            "date": trade.exit_date.isoformat(),
            "equity": current_equity,
            "return": trade_return,
            "ticker": trade.ticker,
        })
    
    return equity_curve


def calculate_monthly_returns(
    trades: List[BacktestTrade],
) -> Dict[str, float]:
    """
    Calculate monthly returns.
    
    Args:
        trades: List of BacktestTrade objects
        
    Returns:
        Dict mapping month (YYYY-MM) to return
    """
    if not trades:
        return {}
    
    monthly = {}
    
    for trade in trades:
        month_key = trade.exit_date.strftime("%Y-%m")
        if month_key not in monthly:
            monthly[month_key] = []
        monthly[month_key].append(trade.return_after_costs)
    
    # Calculate compounded monthly return
    monthly_returns = {}
    for month, returns in monthly.items():
        compounded = 1.0
        for r in returns:
            compounded *= (1 + r)
        monthly_returns[month] = compounded - 1
    
    return monthly_returns


def calculate_trade_distribution(
    trades: List[BacktestTrade],
    bins: int = 20,
) -> Dict[str, List]:
    """
    Calculate trade return distribution for histogram.
    
    Args:
        trades: List of BacktestTrade objects
        bins: Number of bins
        
    Returns:
        Dict with bins and counts
    """
    if not trades:
        return {"bins": [], "counts": []}
    
    returns = [t.return_after_costs * 100 for t in trades]  # Convert to percentage
    
    counts, bin_edges = np.histogram(returns, bins=bins)
    
    return {
        "bins": [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(counts))],
        "counts": counts.tolist(),
    }


def compare_strategies(
    trades_a: List[BacktestTrade],
    trades_b: List[BacktestTrade],
    name_a: str = "Strategy A",
    name_b: str = "Strategy B",
) -> Dict[str, Any]:
    """
    Compare two sets of backtest results.
    
    Args:
        trades_a: Trades from first strategy
        trades_b: Trades from second strategy
        name_a: Name for first strategy
        name_b: Name for second strategy
        
    Returns:
        Comparison dict
    """
    metrics_a = calculate_metrics(trades_a)
    metrics_b = calculate_metrics(trades_b)
    
    return {
        name_a: metrics_a.to_dict(),
        name_b: metrics_b.to_dict(),
        "comparison": {
            "win_rate_diff": metrics_a.win_rate - metrics_b.win_rate,
            "return_diff": metrics_a.cumulative_return - metrics_b.cumulative_return,
            "drawdown_diff": metrics_a.max_drawdown - metrics_b.max_drawdown,
            "sharpe_diff": (metrics_a.sharpe_ratio or 0) - (metrics_b.sharpe_ratio or 0),
        }
    }
