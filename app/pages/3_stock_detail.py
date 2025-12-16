"""
Stock Detail Page - Detailed analysis for a single stock.
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.components.dashboard import get_services
from storage.models import SignalDirection
from core.news_classifier import get_category_display_name, get_sentiment_display

st.set_page_config(
    page_title="Stock Detail - Trend Terminal",
    page_icon="📈",
    layout="wide",
)


def create_tradingview_chart(ticker: str, height: int = 500, theme: str = "dark") -> None:
    """Embed TradingView advanced chart widget."""
    # TradingView widget HTML
    tradingview_html = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="height:{height}px;width:100%">
      <div id="tradingview_chart" style="height:calc(100% - 32px);width:100%"></div>
      <div class="tradingview-widget-copyright">
        <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
          <span class="blue-text">Track all markets on TradingView</span>
        </a>
      </div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "autosize": true,
        "symbol": "{ticker}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "{theme}",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "hide_side_toolbar": false,
        "studies": [
          "STD;EMA",
          "STD;RSI",
          "STD;MACD"
        ],
        "container_id": "tradingview_chart"
      }}
      );
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    components.html(tradingview_html, height=height)


def create_tradingview_mini_chart(ticker: str, height: int = 350) -> None:
    """Embed TradingView mini chart widget."""
    mini_chart_html = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
      {{
        "symbol": "{ticker}",
        "width": "100%",
        "height": "{height}",
        "locale": "en",
        "dateRange": "12M",
        "colorTheme": "dark",
        "isTransparent": true,
        "autosize": true,
        "largeChartUrl": ""
      }}
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    components.html(mini_chart_html, height=height)


def create_tradingview_technical_analysis(ticker: str, height: int = 450) -> None:
    """Embed TradingView technical analysis widget."""
    ta_html = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
      {{
        "interval": "1D",
        "width": "100%",
        "isTransparent": true,
        "height": "{height}",
        "symbol": "{ticker}",
        "showIntervalTabs": true,
        "displayMode": "single",
        "locale": "en",
        "colorTheme": "dark"
      }}
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    components.html(ta_html, height=height)


def create_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create an interactive price chart with indicators."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} Price", "Volume"),
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
        ),
        row=1, col=1,
    )
    
    # EMA20
    if 'ema20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema20'],
                name="EMA20",
                line=dict(color='orange', width=1.5),
            ),
            row=1, col=1,
        )
    
    # EMA50
    if 'ema50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema50'],
                name="EMA50",
                line=dict(color='blue', width=1.5),
            ),
            row=1, col=1,
        )
    
    # Volume bars
    colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7,
        ),
        row=2, col=1,
    )
    
    # Average volume line
    if 'avg_volume_30' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['avg_volume_30'],
                name="Avg Volume (30D)",
                line=dict(color='gray', dash='dash', width=1),
            ),
            row=2, col=1,
        )
    
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    
    return fig


def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """Create RSI indicator chart."""
    fig = go.Figure()
    
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                name="RSI",
                line=dict(color='purple', width=2),
            )
        )
        
        # Overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig.add_hline(y=50, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        height=250,
        template="plotly_dark",
        title="RSI (14)",
        yaxis=dict(range=[0, 100]),
        showlegend=False,
    )
    
    return fig


def main():
    services = get_services()
    
    st.title("📈 Stock Detail")
    
    # Stock selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        # Get available tickers
        all_tickers = services["universe"].get_sp500()[:50] + services["universe"].get_watchlist()
        all_tickers = sorted(set(all_tickers))
        
        # Check if ticker was passed from screener
        default_ticker = st.session_state.get("detail_ticker", "AAPL")
        if default_ticker not in all_tickers:
            all_tickers.insert(0, default_ticker)
        
        ticker = st.selectbox(
            "Select Stock",
            options=all_tickers,
            index=all_tickers.index(default_ticker) if default_ticker in all_tickers else 0,
        )
    with col2:
        period = st.selectbox(
            "Period",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            index=2,
        )
    with col3:
        refresh = st.button("🔄 Refresh", use_container_width=True)
    
    st.divider()
    
    # Fetch data
    with st.spinner(f"Loading data for {ticker}..."):
        ohlcv = services["fetcher"].fetch_ohlcv(ticker, period=period, force_refresh=refresh)
        fundamentals = services["fetcher"].fetch_fundamentals(ticker, force_refresh=refresh)
        news = services["fetcher"].fetch_news(ticker, 
            company_name=fundamentals.company_name if fundamentals else None,
            force_refresh=refresh)
    
    if not ohlcv:
        st.error(f"Failed to load data for {ticker}. Please try again.")
        return
    
    # Classify news
    if news:
        news = services["news_classifier"].classify_batch(news)
    
    # Calculate indicators
    df = services["indicators"].records_to_dataframe(ohlcv)
    df = services["indicators"].calculate_all_indicators(df)
    
    # Get latest indicators
    ind_values = services["indicators"].get_latest_indicators(ohlcv)
    
    # Generate signal
    signal = services["signals"].generate_signal(
        ticker=ticker,
        indicators=ind_values,
        news_list=news,
        fundamentals=fundamentals,
    )
    
    # Company info header
    if fundamentals:
        st.markdown(f"### {fundamentals.company_name or ticker}")
        st.caption(f"{fundamentals.sector or ''} | {fundamentals.industry or ''} | {fundamentals.exchange or ''}")
    else:
        st.markdown(f"### {ticker}")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Price",
            f"${ind_values.close:.2f}",
            f"{(ind_values.return_1d or 0) * 100:+.2f}%",
        )
    with col2:
        st.metric(
            "5D Change",
            f"{(ind_values.return_5d or 0) * 100:+.2f}%",
        )
    with col3:
        st.metric(
            "Volume",
            f"{ind_values.volume/1e6:.1f}M",
            f"{((ind_values.volume_spike or 1) - 1) * 100:+.1f}%",
        )
    with col4:
        if fundamentals and fundamentals.market_cap:
            cap = fundamentals.market_cap
            cap_str = f"${cap/1e12:.1f}T" if cap >= 1e12 else f"${cap/1e9:.1f}B"
            st.metric("Market Cap", cap_str)
        else:
            st.metric("Market Cap", "-")
    with col5:
        st.metric("RSI", f"{ind_values.rsi:.1f}" if ind_values.rsi else "-")
    
    st.divider()
    
    # Main content - two columns
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Price chart - TradingView or Plotly
        st.subheader("📊 Price Chart")
        
        chart_type = st.radio(
            "Chart Type",
            options=["tradingview", "plotly"],
            format_func=lambda x: "📈 TradingView (Live)" if x == "tradingview" else "📊 Plotly (Local)",
            horizontal=True,
            label_visibility="collapsed",
        )
        
        if chart_type == "tradingview":
            create_tradingview_chart(ticker, height=550, theme="dark")
        else:
            chart = create_price_chart(df.tail(200), ticker)
            st.plotly_chart(chart, use_container_width=True)
            
            # RSI chart (only for Plotly mode)
            rsi_chart = create_rsi_chart(df.tail(200))
            st.plotly_chart(rsi_chart, use_container_width=True)
    
    with col_right:
        # TradingView Technical Analysis Widget
        if chart_type == "tradingview":
            st.subheader("📊 Technical Analysis")
            create_tradingview_technical_analysis(ticker, height=400)
        
        # Signal box
        st.subheader("🎯 Signal Analysis")
        
        direction_styles = {
            SignalDirection.BULLISH: ("bullish-box", "🟢 BULLISH"),
            SignalDirection.BEARISH: ("bearish-box", "🔴 BEARISH"),
            SignalDirection.NEUTRAL: ("neutral-box", "⚪ NEUTRAL"),
        }
        
        style_class, direction_text = direction_styles.get(
            signal.direction, ("neutral-box", "⚪ NEUTRAL")
        )
        
        st.markdown(f"""
        <div class="{style_class}">
            <h3 style="margin-top: 0;">{direction_text}</h3>
            <p><strong>Score:</strong> {signal.final_score:.0f}/100</p>
            <p><strong>Confidence:</strong> {signal.confidence:.0%}</p>
            <p><strong>Trending:</strong> {"Yes ✅" if signal.is_trending else "No"}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Score breakdown
        st.markdown("#### Score Breakdown")
        scores_df = pd.DataFrame({
            "Component": ["Technical", "Volume", "News"],
            "Score": [signal.trend_score, signal.volume_score, signal.news_score],
            "Weight": ["60%", "20%", "20%"],
        })
        st.dataframe(scores_df, hide_index=True, use_container_width=True)
        
        # Reasons
        st.markdown("#### 📝 Key Factors")
        for reason in signal.reasons:
            st.markdown(f"- {reason}")
        
        # AI Explanation
        if services["llm"].is_available:
            st.divider()
            st.markdown("#### 🤖 AI Analysis")
            with st.spinner("Generating insights..."):
                explanation = services["llm"].explain_signal(signal, news)
            st.markdown(explanation)
    
    # News section
    st.divider()
    st.subheader("📰 Recent News")
    
    if news:
        # News synthesis
        if services["llm"].is_available:
            with st.spinner("Synthesizing news..."):
                synthesis = services["llm"].synthesize_news(
                    ticker, news, 
                    fundamentals.company_name if fundamentals else None
                )
            st.markdown(synthesis)
            st.divider()
        
        # Individual news items
        for item in sorted(news, key=lambda x: x.published_at, reverse=True)[:10]:
            sentiment_icon = "🟢" if item.sentiment.value > 0 else "🔴" if item.sentiment.value < 0 else "⚪"
            category_name = get_category_display_name(item.category)
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{sentiment_icon} {item.headline}**")
                st.caption(f"{item.source or 'Unknown'} | {item.published_at.strftime('%m/%d %H:%M')} | {category_name}")
            with col2:
                if item.url:
                    st.link_button("Read →", item.url, use_container_width=True)
    else:
        st.info("No recent news available for this stock.")
    
    # Technical indicators table
    st.divider()
    st.subheader("📐 Technical Indicators")
    
    indicators_data = {
        "Indicator": ["Close", "EMA20", "EMA50", "RSI", "Volume", "Avg Vol (30D)", "Vol Spike", "ATR"],
        "Value": [
            f"${ind_values.close:.2f}",
            f"${ind_values.ema20:.2f}" if ind_values.ema20 else "-",
            f"${ind_values.ema50:.2f}" if ind_values.ema50 else "-",
            f"{ind_values.rsi:.1f}" if ind_values.rsi else "-",
            f"{ind_values.volume:,}",
            f"{ind_values.avg_volume_30:,.0f}" if ind_values.avg_volume_30 else "-",
            f"{ind_values.volume_spike:.2f}x" if ind_values.volume_spike else "-",
            f"${df['atr'].iloc[-1]:.2f}" if 'atr' in df.columns else "-",
        ],
        "Signal": [
            "📈" if ind_values.return_1d and ind_values.return_1d > 0 else "📉" if ind_values.return_1d else "-",
            "✅" if ind_values.ema20 and ind_values.close > ind_values.ema20 else "❌" if ind_values.ema20 else "-",
            "✅" if ind_values.ema50 and ind_values.close > ind_values.ema50 else "❌" if ind_values.ema50 else "-",
            "Overbought" if ind_values.rsi and ind_values.rsi > 70 else "Oversold" if ind_values.rsi and ind_values.rsi < 30 else "Neutral" if ind_values.rsi else "-",
            "High" if ind_values.volume_spike and ind_values.volume_spike > 1.5 else "Normal",
            "-",
            "🔥" if ind_values.volume_spike and ind_values.volume_spike > 2 else "✅" if ind_values.volume_spike and ind_values.volume_spike > 1.5 else "Normal",
            "-",
        ],
    }
    
    st.dataframe(pd.DataFrame(indicators_data), hide_index=True, use_container_width=True)
    
    # Footer
    st.divider()
    st.caption("""
    ⚠️ **Disclaimer:** This analysis is for educational purposes only and does not constitute financial advice.
    Past performance does not guarantee future results. Always do your own research.
    """)


if __name__ == "__main__":
    main()
