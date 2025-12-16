"""
Trend Terminal - Professional Stock Scanning & Signal Detection Application.

A Streamlit-based GUI for stock screening, signal generation, backtesting,
and intelligent market analysis.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Trend Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': """
        # Trend Terminal
        Professional Stock Scanning & Signal Detection
        
        ⚠️ **Disclaimer:** This tool provides algorithmic analysis for educational purposes.
        It does not constitute financial advice. Always conduct your own research.
        """
    }
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1E88E5;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.85rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Signal boxes */
    .bullish-box {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    .bearish-box {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #F44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    .neutral-box {
        background-color: rgba(158, 158, 158, 0.1);
        border-left: 4px solid #9E9E9E;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def main():
    """Main entry point for the Dashboard page."""
    
    # Sidebar branding
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
        st.title("Trend Terminal")
        st.caption("Professional Stock Analysis")
        st.divider()
        
        # Navigation info
        st.markdown("""
        ### Navigation
        Use the sidebar to access:
        - 📊 **Dashboard** - Market overview
        - 🔍 **Screener** - Filter stocks
        - 📈 **Stock Detail** - Deep analysis
        - 📉 **Backtest** - Test strategies
        - ⚙️ **Settings** - Configuration
        """)
        
        st.divider()
        
        # Quick status
        st.markdown("### System Status")
        
        # Check OpenAI API
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_openai_api_key_here":
            st.success("🤖 AI Insights: Active")
        else:
            st.warning("🤖 AI Insights: Not configured")
        
        # Check Telegram
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if telegram_token and telegram_token != "your_telegram_bot_token_here":
            st.success("📱 Telegram: Connected")
        else:
            st.info("📱 Telegram: Not configured")
    
    # Main content
    st.title("📊 Market Dashboard")
    st.markdown("*Real-time market analysis and signal detection*")
    
    # Import components (lazy load to avoid circular imports)
    from app.components.dashboard import render_dashboard
    
    render_dashboard()


if __name__ == "__main__":
    main()
