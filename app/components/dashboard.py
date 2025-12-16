"""
Dashboard Components for Trend Terminal.
Renders the main dashboard with market overview and top signals.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from storage.database import DatabaseManager
from storage.cache import CacheManager
from core.data_fetcher import DataFetcher
from core.indicators import IndicatorCalculator
from core.signals import SignalGenerator, SignalConfig
from core.news_classifier import NewsClassifier
from core.universes import UniverseLoader, MARKET_INDICES
from core.llm_analyzer import LLMAnalyzer
from storage.models import SignalDirection, MarketOverview, ScanResult

logger = logging.getLogger(__name__)


@st.cache_resource
def get_services():
    """Get cached service instances."""
    db = DatabaseManager()
    cache = CacheManager()  # CacheManager creates its own DB connection
    fetcher = DataFetcher(cache)
    indicators = IndicatorCalculator()
    signals = SignalGenerator()
    news_classifier = NewsClassifier()
    universe_loader = UniverseLoader()
    llm = LLMAnalyzer()
    
    return {
        "db": db,
        "cache": cache,
        "fetcher": fetcher,
        "indicators": indicators,
        "signals": signals,
        "news_classifier": news_classifier,
        "universe": universe_loader,
        "llm": llm,
    }


def fetch_market_overview(fetcher: DataFetcher, indicators: IndicatorCalculator, index_ticker: str) -> Optional[MarketOverview]:
    """Fetch market overview for an index."""
    try:
        ohlcv = fetcher.fetch_ohlcv(index_ticker, period="3mo")
        if not ohlcv:
            return None
        
        # Calculate indicators
        ind_values = indicators.get_latest_indicators(ohlcv)
        if not ind_values:
            return None
        
        # Calculate 5D change
        if len(ohlcv) >= 5:
            change_5d_pct = (ohlcv[-1].close - ohlcv[-5].close) / ohlcv[-5].close * 100
        else:
            change_5d_pct = 0.0
        
        # Determine trend
        if ind_values.ema20 and ind_values.ema50:
            if ind_values.close > ind_values.ema20 and ind_values.ema20 > ind_values.ema50:
                trend: SignalDirection = SignalDirection.BULLISH
            elif ind_values.close < ind_values.ema20 and ind_values.ema20 < ind_values.ema50:
                trend = SignalDirection.BEARISH
            else:
                trend = SignalDirection.NEUTRAL
        else:
            trend = SignalDirection.NEUTRAL
        
        # Calculate volume average
        if len(ohlcv) >= 30:
            avg_volume = sum(r.volume for r in ohlcv[-30:]) / 30
        else:
            avg_volume = sum(r.volume for r in ohlcv) / len(ohlcv) if ohlcv else 0
        
        return MarketOverview(
            index_ticker=index_ticker,
            index_name=MARKET_INDICES.get(index_ticker, {}).get("name", index_ticker),
            last_price=ohlcv[-1].close,
            change_1d=(ind_values.return_1d or 0) * ohlcv[-1].close,
            change_1d_pct=(ind_values.return_1d or 0) * 100,
            change_5d_pct=change_5d_pct,
            volume=ohlcv[-1].volume,
            avg_volume=avg_volume,
            trend=trend,
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error fetching market overview: {e}")
        return None


def scan_universe(
    services: Dict,
    tickers: List[str],
    progress_callback=None,
) -> List[ScanResult]:
    """Scan a universe and generate signals."""
    fetcher = services["fetcher"]
    indicators = services["indicators"]
    signals = services["signals"]
    news_classifier = services["news_classifier"]
    
    results = []
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        try:
            # Fetch data
            ohlcv = fetcher.fetch_ohlcv(ticker, period="6mo")
            if not ohlcv or len(ohlcv) < 50:
                continue
            
            # Get fundamentals
            fundamentals = fetcher.fetch_fundamentals(ticker)
            
            # Calculate indicators
            ind_values = indicators.get_latest_indicators(ohlcv)
            if not ind_values:
                continue
            
            # Get news
            news = fetcher.fetch_news(ticker)
            if news:
                news = news_classifier.classify_batch(news)
            
            # Generate signal
            signal = signals.generate_signal(
                ticker=ticker,
                indicators=ind_values,
                news_list=news,
                fundamentals=fundamentals,
            )
            
            # Convert to scan result
            result = signals.generate_scan_result(signal, fundamentals)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error scanning {ticker}: {e}")
        
        if progress_callback:
            progress_callback((i + 1) / total)
    
    return results


def render_market_metrics(overview: MarketOverview):
    """Render market metrics cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"{overview.index_name}",
            value=f"${overview.last_price:.2f}",
            delta=f"{overview.change_1d_pct:+.2f}%",
        )
    
    with col2:
        st.metric(
            label="5-Day Change",
            value=f"{overview.change_5d_pct:+.2f}%",
        )
    
    with col3:
        vol_ratio = overview.volume / overview.avg_volume if overview.avg_volume > 0 else 1
        st.metric(
            label="Volume",
            value=f"{overview.volume/1e6:.1f}M",
            delta=f"{(vol_ratio - 1) * 100:+.1f}% vs avg",
        )
    
    with col4:
        trend_display = {
            SignalDirection.BULLISH: "🟢 Bullish",
            SignalDirection.BEARISH: "🔴 Bearish",
            SignalDirection.NEUTRAL: "⚪ Neutral",
        }
        st.metric(
            label="Market Trend",
            value=trend_display.get(overview.trend, "⚪ Neutral"),
        )


def render_signals_table(results: List[ScanResult], direction: str, max_rows: int = 10):
    """Render a signals table."""
    if direction == "bullish":
        filtered = [r for r in results if r.direction == SignalDirection.BULLISH]
        filtered = sorted(filtered, key=lambda x: x.final_score, reverse=True)[:max_rows]
        title = "🟢 Top Bullish Signals"
    else:
        filtered = [r for r in results if r.direction == SignalDirection.BEARISH]
        filtered = sorted(filtered, key=lambda x: x.final_score, reverse=True)[:max_rows]
        title = "🔴 Top Bearish Signals"
    
    st.subheader(title)
    
    if not filtered:
        st.info(f"No {direction} signals detected.")
        return
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            "Ticker": r.ticker,
            "Price": f"${r.last_price:.2f}",
            "1D %": f"{r.change_1d_pct:+.1f}%",
            "5D %": f"{r.change_5d_pct:+.1f}%",
            "Vol Spike": f"{r.volume_spike:.1f}x",
            "Score": f"{r.final_score:.0f}",
            "Confidence": f"{r.confidence:.0%}",
        }
        for r in filtered
    ])
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def render_dashboard():
    """Main dashboard rendering function."""
    services = get_services()
    
    # Index selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_index = st.selectbox(
            "Market Index",
            options=list(MARKET_INDICES.keys()),
            format_func=lambda x: f"{x} - {MARKET_INDICES[x]['name']}",
        )
    with col2:
        scan_button = st.button("🔄 Refresh Data", use_container_width=True)
    
    st.divider()
    
    # Market Overview
    with st.spinner("Loading market data..."):
        overview = fetch_market_overview(
            services["fetcher"],
            services["indicators"],
            selected_index,
        )
    
    if overview:
        render_market_metrics(overview)
    else:
        st.error("Failed to load market data. Please try again.")
    
    st.divider()
    
    # Scan controls
    st.subheader("📡 Signal Scanner")
    
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col1:
        universe_options = {
            "watchlist": "📋 Watchlist (Custom)",
            "sp500": "🏛️ S&P 500 (Top 100)",
            "nasdaq100": "💻 Nasdaq 100",
            "scalp": "⚡ Scalp/Day Trade",
            "swing": "🔄 Swing Trade",
            "long_term": "📈 Long-Term Investment",
            "penny": "💰 Penny Stocks",
            "momentum": "🚀 High Momentum",
        }
        selected_universe = st.selectbox(
            "Universe",
            options=list(universe_options.keys()),
            format_func=lambda x: universe_options[x],
        )
    with col2:
        trading_style = st.selectbox(
            "Trading Style",
            options=["auto", "scalp", "swing", "position", "long_term"],
            format_func=lambda x: {
                "auto": "🎯 Auto (Match Universe)",
                "scalp": "⚡ Scalping (Minutes-Hours)",
                "swing": "🔄 Swing (Days-Weeks)",
                "position": "📊 Position (Weeks-Months)",
                "long_term": "📈 Investment (Months-Years)",
            }.get(x, x),
        )
    with col3:
        max_signals = st.slider("Top N", 5, 20, 10)
    with col4:
        run_scan = st.button("🚀 Scan", use_container_width=True, type="primary")
    
    # Store scan results in session state
    if "scan_results" not in st.session_state:
        st.session_state.scan_results = []
    
    if run_scan or scan_button:
        # Get tickers based on universe and trading style
        if trading_style != "auto":
            tickers = services["universe"].get_by_trading_style(trading_style)
        else:
            tickers = services["universe"].get_universe(selected_universe)
        
        if not tickers:
            st.warning("No tickers found in selected universe. Check your watchlist.")
        else:
            # Limit for performance
            tickers = tickers[:50]  # Limit to 50 for dashboard
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Scanning... {int(progress * 100)}%")
            
            with st.spinner("Scanning universe..."):
                results = scan_universe(services, tickers, update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            st.session_state.scan_results = results
            st.success(f"✅ Scanned {len(tickers)} stocks, found {len(results)} signals.")
    
    # Display results
    if st.session_state.scan_results:
        results = st.session_state.scan_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            render_signals_table(results, "bullish", max_signals)
        
        with col2:
            render_signals_table(results, "bearish", max_signals)
        
        # AI Market Summary
        if services["llm"].is_available and overview:
            st.divider()
            st.subheader("🤖 AI Market Analysis")
            
            bullish = [r for r in results if r.direction == SignalDirection.BULLISH][:5]
            bearish = [r for r in results if r.direction == SignalDirection.BEARISH][:5]
            
            with st.spinner("Generating AI insights..."):
                summary = services["llm"].generate_market_summary(overview, bullish, bearish)
            
            st.markdown(summary)
    
    # Quick stats
    st.divider()
    st.subheader("📈 Quick Statistics")
    
    if st.session_state.scan_results:
        results = st.session_state.scan_results
        bullish_count = sum(1 for r in results if r.direction == SignalDirection.BULLISH)
        bearish_count = sum(1 for r in results if r.direction == SignalDirection.BEARISH)
        neutral_count = len(results) - bullish_count - bearish_count
        avg_score = sum(r.final_score for r in results) / len(results) if results else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Signals", len(results))
        col2.metric("Bullish", bullish_count, delta_color="normal")
        col3.metric("Bearish", bearish_count, delta_color="inverse")
        col4.metric("Avg Score", f"{avg_score:.0f}/100")
    else:
        st.info("Run a scan to see statistics.")
    
    # Footer
    st.divider()
    st.caption("""
    ⚠️ **Disclaimer:** This tool provides algorithmic analysis for educational purposes only.
    It does not constitute financial advice. Past performance does not guarantee future results.
    Always conduct your own research and consult with a financial advisor before making investment decisions.
    """)
