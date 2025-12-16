"""
Backtest Page - Run and analyze backtests.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta, date

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.components.dashboard import get_services
from backtest.engine import BacktestEngine, BacktestConfig
from backtest.metrics import calculate_monthly_returns, calculate_trade_distribution
from storage.models import SignalDirection

st.set_page_config(
    page_title="Backtest - Trend Terminal",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Strategy Backtester")
st.markdown("*Test signal strategies on historical data*")


def create_equity_curve_chart(equity_curve: list) -> go.Figure:
    """Create equity curve chart."""
    if not equity_curve:
        return go.Figure()
    
    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='#2196F3', width=2),
        fill='tozeroy',
        fillcolor='rgba(33, 150, 243, 0.1)',
    ))
    
    # Add reference line at starting equity
    fig.add_hline(
        y=df['equity'].iloc[0],
        line_dash="dash",
        line_color="gray",
        annotation_text="Starting Capital",
    )
    
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        height=400,
        showlegend=False,
    )
    
    return fig


def create_drawdown_chart(equity_curve: list) -> go.Figure:
    """Create drawdown chart."""
    if not equity_curve:
        return go.Figure()
    
    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate drawdown
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak'] * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['drawdown'],
        mode='lines',
        name='Drawdown',
        line=dict(color='#F44336', width=2),
        fill='tozeroy',
        fillcolor='rgba(244, 67, 54, 0.3)',
    ))
    
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        height=250,
        showlegend=False,
    )
    
    return fig


def create_returns_histogram(trades: list) -> go.Figure:
    """Create trade returns histogram."""
    if not trades:
        return go.Figure()
    
    returns = [t.return_after_costs * 100 for t in trades]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=30,
        marker_color='#2196F3',
        opacity=0.7,
    ))
    
    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="white")
    
    fig.update_layout(
        title="Trade Return Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Count",
        template="plotly_dark",
        height=300,
    )
    
    return fig


def create_monthly_returns_heatmap(trades: list) -> go.Figure:
    """Create monthly returns heatmap."""
    monthly = calculate_monthly_returns(trades)
    
    if not monthly:
        return go.Figure()
    
    # Convert to matrix format
    months_data = {}
    for month_str, ret in monthly.items():
        year, month = month_str.split('-')
        if year not in months_data:
            months_data[year] = [None] * 12
        months_data[year][int(month) - 1] = ret * 100
    
    years = sorted(months_data.keys())
    z = [months_data[y] for y in years]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=years,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{v:.1f}%" if v is not None else "" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title="Monthly Returns (%)",
        template="plotly_dark",
        height=250,
    )
    
    return fig


def main():
    services = get_services()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Backtest Settings")
        
        # Ticker selection
        st.subheader("Tickers")
        
        # Check for tickers from screener
        default_tickers = st.session_state.get("backtest_tickers", ["AAPL", "MSFT", "GOOGL"])
        
        ticker_input = st.text_area(
            "Enter tickers (one per line)",
            value="\n".join(default_tickers),
            height=100,
        )
        tickers = [t.strip().upper() for t in ticker_input.split("\n") if t.strip()]
        
        st.caption(f"📊 {len(tickers)} tickers selected")
        
        st.divider()
        
        # Date range
        st.subheader("📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                value=datetime.now() - timedelta(days=365),
            )
        with col2:
            end_date = st.date_input(
                "End",
                value=datetime.now(),
            )
        
        st.divider()
        
        # Trading Style
        st.subheader("🎯 Trading Style")
        trading_style = st.radio(
            "Select Style",
            options=["scalp", "swing", "position", "long_term"],
            format_func=lambda x: {
                "scalp": "⚡ Scalping (1-2 days)",
                "swing": "🔄 Swing (3-10 days)",
                "position": "📊 Position (10-30 days)",
                "long_term": "📈 Long-term (30+ days)",
            }.get(x, x),
            horizontal=False,
        )
        
        st.divider()
        
        # Strategy settings
        st.subheader("📈 Strategy")
        
        # Pre-fill based on trading style
        style_defaults = {
            "scalp": (1, "signal_flip", 55),
            "swing": (5, "fixed", 50),
            "position": (20, "signal_flip", 45),
            "long_term": (60, "fixed", 40),
        }
        default_period, default_exit, default_score = style_defaults.get(trading_style, (5, "fixed", 50))
        
        holding_period = st.selectbox(
            "Holding Period (days)",
            options=[1, 2, 3, 5, 10, 20, 30, 60, 90],
            index=[1, 2, 3, 5, 10, 20, 30, 60, 90].index(default_period) if default_period in [1, 2, 3, 5, 10, 20, 30, 60, 90] else 3,
        )
        
        exit_strategy = st.selectbox(
            "Exit Strategy",
            options=["fixed", "signal_flip"],
            index=0 if default_exit == "fixed" else 1,
            format_func=lambda x: "Fixed Period" if x == "fixed" else "Signal Flip",
        )
        
        min_score = st.slider("Min Signal Score", 0, 100, default_score)
        
        st.divider()
        
        # Costs (adjust based on style)
        st.subheader("💰 Transaction Costs")
        
        style_slippage = {"scalp": 0.2, "swing": 0.1, "position": 0.05, "long_term": 0.03}
        default_slippage = style_slippage.get(trading_style, 0.1)
        
        slippage = st.slider(
            "Slippage (%)",
            0.0, 1.0, default_slippage, 0.05,
            format="%.2f%%",
        )
        
        commission = st.number_input(
            "Commission ($/trade)",
            0.0, 10.0, 0.0, 0.5,
        )
    
    # Main content
    col1, col2 = st.columns([3, 1])
    with col2:
        run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    # Store results in session state
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None
    
    if run_backtest:
        if not tickers:
            st.warning("Please enter at least one ticker.")
        else:
            # Create config
            config = BacktestConfig(
                holding_period=holding_period,
                exit_strategy=exit_strategy,
                commission_per_trade=commission,
                slippage_pct=slippage / 100,
                min_signal_score=min_score,
            )
            
            engine = BacktestEngine(config)
            
            # Fetch data for all tickers
            progress_bar = st.progress(0)
            status = st.empty()
            
            status.text("Fetching historical data...")
            ticker_data = {}
            for i, ticker in enumerate(tickers):
                try:
                    ohlcv = services["fetcher"].fetch_ohlcv(ticker, period="2y")
                    if ohlcv:
                        ticker_data[ticker] = ohlcv
                except Exception as e:
                    st.warning(f"Failed to fetch data for {ticker}: {e}")
                progress_bar.progress((i + 1) / len(tickers) * 0.5)
            
            if not ticker_data:
                st.error("No valid data found for any tickers.")
            else:
                status.text("Running backtest...")
                
                def update_progress(p):
                    progress_bar.progress(0.5 + p * 0.5)
                
                # Run backtest
                result = engine.run_multi_ticker_backtest(
                    ticker_data,
                    start_date=start_date,
                    end_date=end_date,
                    progress_callback=update_progress,
                )
                
                progress_bar.empty()
                status.empty()
                
                st.session_state.backtest_results = result
                st.success(f"✅ Backtest complete! {len(result.trades)} trades executed.")
    
    # Display results
    if st.session_state.backtest_results:
        result = st.session_state.backtest_results
        metrics = result.metrics
        
        st.divider()
        
        # Key metrics
        st.subheader("📊 Performance Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            color = "normal" if metrics.cumulative_return >= 0 else "inverse"
            st.metric(
                "Total Return",
                f"{metrics.cumulative_return * 100:+.2f}%",
                delta_color=color,
            )
        with col2:
            st.metric(
                "Win Rate",
                f"{metrics.win_rate * 100:.1f}%",
            )
        with col3:
            st.metric(
                "Total Trades",
                f"{metrics.total_trades}",
            )
        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics.max_drawdown * 100:.1f}%",
            )
        with col5:
            sharpe = f"{metrics.sharpe_ratio:.2f}" if metrics.sharpe_ratio else "N/A"
            st.metric(
                "Sharpe Ratio",
                sharpe,
            )
        
        # Second row of metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Avg Return/Trade",
                f"{metrics.avg_return * 100:+.2f}%",
            )
        with col2:
            st.metric(
                "Avg Winner",
                f"{metrics.avg_winner * 100:+.2f}%",
            )
        with col3:
            st.metric(
                "Avg Loser",
                f"{metrics.avg_loser * 100:.2f}%",
            )
        with col4:
            pf = f"{metrics.profit_factor:.2f}" if metrics.profit_factor else "N/A"
            st.metric(
                "Profit Factor",
                pf,
            )
        with col5:
            st.metric(
                "Avg Hold Days",
                f"{metrics.avg_holding_days:.1f}",
            )
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Equity curve
            equity_fig = create_equity_curve_chart(result.equity_curve)
            st.plotly_chart(equity_fig, use_container_width=True)
            
            # Drawdown
            drawdown_fig = create_drawdown_chart(result.equity_curve)
            st.plotly_chart(drawdown_fig, use_container_width=True)
        
        with col2:
            # Returns histogram
            hist_fig = create_returns_histogram(result.trades)
            st.plotly_chart(hist_fig, use_container_width=True)
            
            # Monthly returns
            monthly_fig = create_monthly_returns_heatmap(result.trades)
            st.plotly_chart(monthly_fig, use_container_width=True)
        
        st.divider()
        
        # Trade list
        st.subheader("📋 Trade History")
        
        if result.trades:
            trades_df = pd.DataFrame([
                {
                    "Ticker": t.ticker,
                    "Direction": "🟢 Long" if t.direction == SignalDirection.BULLISH else "🔴 Short",
                    "Entry Date": t.entry_date,
                    "Entry Price": f"${t.entry_price:.2f}",
                    "Exit Date": t.exit_date,
                    "Exit Price": f"${t.exit_price:.2f}",
                    "Return": f"{t.return_after_costs * 100:+.2f}%",
                    "Days Held": t.holding_days,
                    "Exit Reason": t.exit_reason.replace("_", " ").title(),
                    "Score": f"{t.signal_score:.0f}",
                }
                for t in sorted(result.trades, key=lambda x: x.entry_date, reverse=True)
            ])
            
            st.dataframe(
                trades_df,
                use_container_width=True,
                hide_index=True,
                height=400,
            )
        
        # AI Analysis
        if services["llm"].is_available:
            st.divider()
            st.subheader("🤖 AI Analysis")
            
            with st.spinner("Generating AI insights..."):
                analysis = services["llm"].analyze_backtest(
                    metrics,
                    len(result.trades),
                    f"Signal Strategy (Score >= {min_score})",
                )
            
            st.markdown(analysis)
    else:
        st.info("👆 Configure settings and click 'Run Backtest' to test your strategy.")
    
    # Footer
    st.divider()
    st.caption("""
    ⚠️ **Disclaimer:** Backtest results are hypothetical and do not represent actual trading.
    Past performance does not guarantee future results. Always consider transaction costs,
    market impact, and other real-world factors not fully captured in backtesting.
    """)


if __name__ == "__main__":
    main()
