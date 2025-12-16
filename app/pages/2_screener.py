"""
Screener Page - Filter and scan stocks with custom criteria.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.components.dashboard import get_services, scan_universe
from storage.models import SignalDirection, ScanResult, NewsCategory

st.set_page_config(
    page_title="Screener - Trend Terminal",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Stock Screener")
st.markdown("*Filter stocks by technical, fundamental, and news criteria*")


def apply_filters(
    results: List[ScanResult],
    price_min: float,
    price_max: float,
    volume_min: int,
    volume_spike_min: float,
    market_cap_min: Optional[float],
    market_cap_max: Optional[float],
    direction_filter: str,
    min_score: float,
    has_news: Optional[bool],
) -> List[ScanResult]:
    """Apply filters to scan results."""
    filtered = []
    
    for r in results:
        # Price filter
        if r.last_price < price_min or r.last_price > price_max:
            continue
        
        # Volume filter
        if r.volume < volume_min:
            continue
        
        # Volume spike filter
        if r.volume_spike < volume_spike_min:
            continue
        
        # Market cap filter
        if market_cap_min is not None and r.market_cap is not None:
            if r.market_cap < market_cap_min:
                continue
        if market_cap_max is not None and r.market_cap is not None:
            if r.market_cap > market_cap_max:
                continue
        
        # Direction filter
        if direction_filter != "All":
            if direction_filter == "Bullish" and r.direction != SignalDirection.BULLISH:
                continue
            if direction_filter == "Bearish" and r.direction != SignalDirection.BEARISH:
                continue
            if direction_filter == "Neutral" and r.direction != SignalDirection.NEUTRAL:
                continue
        
        # Score filter
        if r.final_score < min_score:
            continue
        
        # News filter
        if has_news is True and not r.has_recent_news:
            continue
        if has_news is False and r.has_recent_news:
            continue
        
        filtered.append(r)
    
    return filtered


def main():
    services = get_services()
    
    # Sidebar filters
    with st.sidebar:
        st.header("📊 Filters")
        
        # Universe selection
        st.subheader("Universe")
        universe_options = {
            "watchlist": "📋 Watchlist",
            "sp500": "🏛️ S&P 500",
            "nasdaq100": "💻 Nasdaq 100",
            "scalp": "⚡ Scalp/Day Trade",
            "swing": "🔄 Swing Trade",
            "long_term": "📈 Long-Term",
            "penny": "💰 Penny Stocks",
            "momentum": "🚀 High Momentum",
        }
        selected_universe = st.selectbox(
            "Stock Universe",
            options=list(universe_options.keys()),
            format_func=lambda x: universe_options[x],
            key="screener_universe",
        )
        
        # Trading style
        st.subheader("🎯 Trading Style")
        trading_style = st.radio(
            "Select Style",
            options=["scalp", "swing", "position", "long_term"],
            format_func=lambda x: {
                "scalp": "⚡ Scalping",
                "swing": "🔄 Swing",
                "position": "📊 Position",
                "long_term": "📈 Investment",
            }.get(x, x),
            horizontal=True,
            key="trading_style",
        )
        
        st.divider()
        
        # Price filters
        st.subheader("💰 Price")
        
        # Preset price ranges based on trading style
        if trading_style == "scalp":
            default_min, default_max = 5.0, 500.0
        elif trading_style == "swing":
            default_min, default_max = 10.0, 1000.0
        elif selected_universe == "penny":
            default_min, default_max = 0.1, 10.0
        else:
            default_min, default_max = 1.0, 5000.0
        
        col1, col2 = st.columns(2)
        with col1:
            price_min = st.number_input("Min $", 0.0, 10000.0, default_min, step=1.0)
        with col2:
            price_max = st.number_input("Max $", 0.0, 100000.0, default_max, step=100.0)
        
        st.divider()
        
        # Volume filters
        st.subheader("📊 Volume")
        volume_min = st.number_input(
            "Min Volume",
            0, 1000000000, 100000,
            step=100000,
            format="%d",
        )
        volume_spike_min = st.slider(
            "Min Volume Spike (x avg)",
            0.0, 5.0, 1.0, 0.1,
        )
        
        st.divider()
        
        # Market cap filters
        st.subheader("🏢 Market Cap")
        market_cap_options = {
            "Any": (None, None),
            "Mega (>200B)": (200e9, None),
            "Large (10B-200B)": (10e9, 200e9),
            "Mid (2B-10B)": (2e9, 10e9),
            "Small (300M-2B)": (300e6, 2e9),
            "Micro (<300M)": (None, 300e6),
        }
        market_cap_filter = st.selectbox(
            "Market Cap",
            options=list(market_cap_options.keys()),
        )
        market_cap_min, market_cap_max = market_cap_options[market_cap_filter]
        
        st.divider()
        
        # Signal filters
        st.subheader("📈 Signal")
        direction_filter = st.selectbox(
            "Direction",
            ["All", "Bullish", "Bearish", "Neutral"],
        )
        min_score = st.slider("Min Score", 0, 100, 40)
        
        st.divider()
        
        # News filters
        st.subheader("📰 News")
        news_filter = st.selectbox(
            "Recent News",
            ["Any", "Has News", "No News"],
        )
        has_news = None if news_filter == "Any" else news_filter == "Has News"
    
    # Main content
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"**Universe:** {universe_options[selected_universe]} | **Direction:** {direction_filter} | **Min Score:** {min_score}")
    with col2:
        run_scan = st.button("🚀 Run Screener", type="primary", use_container_width=True)
    
    # Store results in session state
    if "screener_results" not in st.session_state:
        st.session_state.screener_results = []
    
    if run_scan:
        tickers = services["universe"].get_universe(selected_universe)
        
        if not tickers:
            st.warning("No tickers found in selected universe.")
        else:
            # Limit for performance
            tickers = tickers[:100]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Scanning... {int(progress * 100)}%")
            
            with st.spinner("Running screener..."):
                results = scan_universe(services, tickers, update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            st.session_state.screener_results = results
            st.success(f"✅ Scanned {len(tickers)} stocks.")
    
    # Apply filters and display
    if st.session_state.screener_results:
        filtered = apply_filters(
            st.session_state.screener_results,
            price_min,
            price_max,
            volume_min,
            volume_spike_min,
            market_cap_min,
            market_cap_max,
            direction_filter,
            min_score,
            has_news,
        )
        
        # Sort options
        col1, col2 = st.columns([3, 1])
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                ["Score (High to Low)", "Score (Low to High)", "1D Change", "Volume Spike", "Price"],
            )
        
        # Apply sorting
        if sort_by == "Score (High to Low)":
            filtered = sorted(filtered, key=lambda x: x.final_score, reverse=True)
        elif sort_by == "Score (Low to High)":
            filtered = sorted(filtered, key=lambda x: x.final_score)
        elif sort_by == "1D Change":
            filtered = sorted(filtered, key=lambda x: abs(x.change_1d_pct), reverse=True)
        elif sort_by == "Volume Spike":
            filtered = sorted(filtered, key=lambda x: x.volume_spike, reverse=True)
        elif sort_by == "Price":
            filtered = sorted(filtered, key=lambda x: x.last_price, reverse=True)
        
        st.markdown(f"**Found {len(filtered)} stocks matching criteria**")
        
        if filtered:
            # Create DataFrame
            df = pd.DataFrame([
                {
                    "Ticker": r.ticker,
                    "Company": r.company_name or "-",
                    "Sector": r.sector or "-",
                    "Price": r.last_price,
                    "1D %": r.change_1d_pct,
                    "5D %": r.change_5d_pct,
                    "Volume": r.volume,
                    "Vol Spike": r.volume_spike,
                    "Market Cap": r.market_cap,
                    "Direction": r.direction.value.title() if r.direction else "Neutral",
                    "Score": r.final_score,
                    "Confidence": r.confidence,
                    "News": "✅" if r.has_recent_news else "❌",
                }
                for r in filtered
            ])
            
            # Format columns
            styled_df = df.style.format({
                "Price": "${:.2f}",
                "1D %": "{:+.1f}%",
                "5D %": "{:+.1f}%",
                "Volume": "{:,.0f}",
                "Vol Spike": "{:.1f}x",
                "Market Cap": lambda x: f"${x/1e9:.1f}B" if x and x >= 1e9 else f"${x/1e6:.0f}M" if x else "-",
                "Score": "{:.0f}",
                "Confidence": "{:.0%}",
            })
            
            # Color direction
            def color_direction(val):
                if val == "Bullish":
                    return "background-color: rgba(76, 175, 80, 0.3)"
                elif val == "Bearish":
                    return "background-color: rgba(244, 67, 54, 0.3)"
                return ""
            
            styled_df = styled_df.applymap(color_direction, subset=["Direction"])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                height=500,
            )
            
            # Selection for backtest
            st.divider()
            st.subheader("📉 Backtest Selected")
            
            selected_tickers = st.multiselect(
                "Select tickers for backtest (max 20)",
                options=[r.ticker for r in filtered],
                max_selections=20,
            )
            
            if selected_tickers:
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("Run Backtest →", type="secondary"):
                        st.session_state.backtest_tickers = selected_tickers
                        st.switch_page("pages/4_backtest.py")
        else:
            st.warning("No stocks match the current filters. Try adjusting your criteria.")
    else:
        st.info("👆 Configure filters and click 'Run Screener' to find stocks.")
    
    # Footer
    st.divider()
    st.caption("""
    ⚠️ **Disclaimer:** This screener provides algorithmic analysis for educational purposes.
    It does not constitute financial advice. Always conduct your own research.
    """)


if __name__ == "__main__":
    main()
