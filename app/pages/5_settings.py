"""
Settings Page - Application Configuration

Manage app settings including:
- Telegram notifications
- Alert thresholds
- Scan schedules
- API keys
- Data preferences
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from storage.database import DatabaseManager
from notifications.telegram import TelegramNotifier


def load_settings(db: DatabaseManager) -> dict:
    """Load settings from database with defaults."""
    defaults = {
        # Telegram settings
        "telegram_enabled": False,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "telegram_on_signals": True,
        "telegram_on_alerts": True,
        "telegram_on_errors": False,
        "telegram_min_score": 70,
        
        # Alert thresholds
        "alert_rsi_oversold": 30,
        "alert_rsi_overbought": 70,
        "alert_volume_spike": 2.0,
        "alert_price_change": 5.0,
        
        # Scan settings
        "scan_auto_enabled": False,
        "scan_interval_minutes": 15,
        "scan_universe": "sp500",
        "scan_include_watchlist": True,
        
        # Data settings
        "data_cache_hours": 1,
        "data_lookback_days": 180,
        "data_include_premarket": False,
        
        # API settings
        "openai_api_key": "",
        "openai_model": "gpt-4o-mini",
        
        # Display settings
        "display_theme": "dark",
        "display_chart_height": 500,
        "display_max_results": 50,
    }
    
    settings = {}
    for key, default in defaults.items():
        value = db.get_setting(key)
        if value is None:
            settings[key] = default
        else:
            # Convert to appropriate type
            if isinstance(default, bool):
                settings[key] = str(value).lower() in ("true", "1", "yes")
            elif isinstance(default, int):
                settings[key] = int(float(value))
            elif isinstance(default, float):
                settings[key] = float(value)
            else:
                settings[key] = value
    
    return settings


def save_settings(db: DatabaseManager, settings: dict) -> None:
    """Save settings to database."""
    for key, value in settings.items():
        db.save_setting(key, str(value))


def render_telegram_section(db: DatabaseManager, settings: dict) -> dict:
    """Render Telegram notification settings."""
    st.subheader("📱 Telegram Notifications")
    
    # Instructions expander
    with st.expander("📖 How to Set Up Telegram Bot", expanded=not settings["telegram_bot_token"]):
        st.markdown("""
        ### Setting up your Telegram Bot
        
        **Step 1: Create a Bot**
        1. Open Telegram and search for `@BotFather`
        2. Send `/newbot` command
        3. Follow the prompts to name your bot
        4. Copy the **API token** provided (looks like `123456789:ABCdefGHI...`)
        
        **Step 2: Get Your Chat ID**
        1. Search for `@userinfobot` on Telegram
        2. Start a chat and it will show your **Chat ID** (a number like `123456789`)
        
        Or use `@RawDataBot` - forward any message to it to see the chat ID.
        
        **Step 3: Start Your Bot**
        1. Find your new bot by its username
        2. Press **Start** to activate it
        3. Now you can receive notifications!
        
        ---
        
        **For Group Notifications:**
        - Add your bot to the group
        - Send a message in the group
        - Use `@RawDataBot` or API to get the group chat ID (negative number)
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        settings["telegram_enabled"] = st.toggle(
            "Enable Telegram Notifications",
            value=settings["telegram_enabled"],
            help="Turn on/off all Telegram notifications"
        )
    
    with col2:
        # Test connection button
        if settings["telegram_bot_token"] and settings["telegram_chat_id"]:
            if st.button("🧪 Test Connection", use_container_width=True):
                notifier = TelegramNotifier(
                    bot_token=settings["telegram_bot_token"],
                    chat_id=settings["telegram_chat_id"]
                )
                result = notifier.test_connection()
                
                if result["success"]:
                    st.success(f"✅ Connected to @{result.get('bot_name', 'your bot')}")
                else:
                    st.error(f"❌ Connection failed: {result.get('error', 'Unknown error')}")
    
    # Credentials
    col1, col2 = st.columns(2)
    
    with col1:
        settings["telegram_bot_token"] = st.text_input(
            "Bot Token",
            value=settings["telegram_bot_token"],
            type="password",
            help="Token from BotFather",
            placeholder="123456789:ABCdefGHI..."
        )
    
    with col2:
        settings["telegram_chat_id"] = st.text_input(
            "Chat ID",
            value=settings["telegram_chat_id"],
            help="Your user or group chat ID",
            placeholder="123456789"
        )
    
    # Notification preferences
    if settings["telegram_enabled"]:
        st.markdown("**Notification Types:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            settings["telegram_on_signals"] = st.checkbox(
                "Signal Alerts",
                value=settings["telegram_on_signals"],
                help="Notify on new buy/sell signals"
            )
        
        with col2:
            settings["telegram_on_alerts"] = st.checkbox(
                "Price/Volume Alerts",
                value=settings["telegram_on_alerts"],
                help="Notify on threshold breaches"
            )
        
        with col3:
            settings["telegram_on_errors"] = st.checkbox(
                "Error Alerts",
                value=settings["telegram_on_errors"],
                help="Notify on system errors"
            )
        
        settings["telegram_min_score"] = st.slider(
            "Minimum Signal Score for Notification",
            min_value=50,
            max_value=95,
            value=settings["telegram_min_score"],
            step=5,
            help="Only send notifications for signals above this score"
        )
    
    return settings


def render_alerts_section(settings: dict) -> dict:
    """Render alert threshold settings."""
    st.subheader("🎯 Alert Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**RSI Levels**")
        settings["alert_rsi_oversold"] = st.slider(
            "RSI Oversold",
            min_value=10,
            max_value=40,
            value=settings["alert_rsi_oversold"],
            help="Alert when RSI falls below this level"
        )
        
        settings["alert_rsi_overbought"] = st.slider(
            "RSI Overbought",
            min_value=60,
            max_value=90,
            value=settings["alert_rsi_overbought"],
            help="Alert when RSI rises above this level"
        )
    
    with col2:
        st.markdown("**Volume & Price**")
        settings["alert_volume_spike"] = st.slider(
            "Volume Spike Multiplier",
            min_value=1.5,
            max_value=5.0,
            value=settings["alert_volume_spike"],
            step=0.5,
            format="%.1fx",
            help="Alert when volume exceeds average by this factor"
        )
        
        settings["alert_price_change"] = st.slider(
            "Price Change (%)",
            min_value=2.0,
            max_value=15.0,
            value=settings["alert_price_change"],
            step=0.5,
            format="%.1f%%",
            help="Alert on daily price changes exceeding this percentage"
        )
    
    return settings


def render_scan_section(settings: dict) -> dict:
    """Render scan settings."""
    st.subheader("🔍 Scan Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        settings["scan_auto_enabled"] = st.toggle(
            "Auto-Scan Enabled",
            value=settings["scan_auto_enabled"],
            help="Automatically run scans at scheduled intervals"
        )
        
        if settings["scan_auto_enabled"]:
            settings["scan_interval_minutes"] = st.selectbox(
                "Scan Interval",
                options=[5, 10, 15, 30, 60],
                index=[5, 10, 15, 30, 60].index(settings["scan_interval_minutes"]) if settings["scan_interval_minutes"] in [5, 10, 15, 30, 60] else 2,
                format_func=lambda x: f"{x} minutes",
                help="How often to run automatic scans"
            )
    
    with col2:
        settings["scan_universe"] = st.selectbox(
            "Default Universe",
            options=["sp500", "nasdaq100", "watchlist", "all"],
            index=["sp500", "nasdaq100", "watchlist", "all"].index(settings["scan_universe"]) if settings["scan_universe"] in ["sp500", "nasdaq100", "watchlist", "all"] else 0,
            format_func=lambda x: {
                "sp500": "S&P 500",
                "nasdaq100": "NASDAQ 100",
                "watchlist": "Watchlist Only",
                "all": "S&P 500 + NASDAQ 100"
            }.get(x, x)
        )
        
        settings["scan_include_watchlist"] = st.checkbox(
            "Include Watchlist in Scans",
            value=settings["scan_include_watchlist"],
            help="Always include watchlist tickers in universe scans"
        )
    
    return settings


def render_data_section(settings: dict) -> dict:
    """Render data settings."""
    st.subheader("📊 Data Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        settings["data_cache_hours"] = st.selectbox(
            "Cache Duration",
            options=[1, 2, 4, 8, 24],
            index=[1, 2, 4, 8, 24].index(settings["data_cache_hours"]) if settings["data_cache_hours"] in [1, 2, 4, 8, 24] else 0,
            format_func=lambda x: f"{x} hour{'s' if x > 1 else ''}",
            help="How long to cache market data"
        )
        
        settings["data_lookback_days"] = st.selectbox(
            "Default Lookback Period",
            options=[30, 60, 90, 180, 365],
            index=[30, 60, 90, 180, 365].index(settings["data_lookback_days"]) if settings["data_lookback_days"] in [30, 60, 90, 180, 365] else 3,
            format_func=lambda x: f"{x} days",
            help="Default historical data period"
        )
    
    with col2:
        settings["data_include_premarket"] = st.checkbox(
            "Include Pre/Post Market Data",
            value=settings["data_include_premarket"],
            help="Include extended hours data when available"
        )
    
    return settings


def render_api_section(settings: dict) -> dict:
    """Render API settings."""
    st.subheader("🔑 API Settings")
    
    st.info("💡 API keys are stored locally in the database. For extra security, you can also set them as environment variables: `OPENAI_API_KEY`")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        settings["openai_api_key"] = st.text_input(
            "OpenAI API Key",
            value=settings["openai_api_key"],
            type="password",
            help="Required for AI-powered summaries and analysis",
            placeholder="sk-..."
        )
    
    with col2:
        settings["openai_model"] = st.selectbox(
            "OpenAI Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"].index(settings["openai_model"]) if settings["openai_model"] in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"] else 0,
            help="Model to use for AI analysis"
        )
    
    # Show status
    if settings["openai_api_key"] or os.getenv("OPENAI_API_KEY"):
        st.success("✅ OpenAI API key configured")
    else:
        st.warning("⚠️ No OpenAI API key - AI features will be limited")
    
    return settings


def render_display_section(settings: dict) -> dict:
    """Render display settings."""
    st.subheader("🎨 Display Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        settings["display_theme"] = st.selectbox(
            "Chart Theme",
            options=["dark", "light"],
            index=0 if settings["display_theme"] == "dark" else 1,
            format_func=lambda x: x.title()
        )
    
    with col2:
        settings["display_chart_height"] = st.selectbox(
            "Chart Height",
            options=[400, 500, 600, 700],
            index=[400, 500, 600, 700].index(settings["display_chart_height"]) if settings["display_chart_height"] in [400, 500, 600, 700] else 1,
            format_func=lambda x: f"{x}px"
        )
    
    with col3:
        settings["display_max_results"] = st.selectbox(
            "Max Results",
            options=[25, 50, 100, 200],
            index=[25, 50, 100, 200].index(settings["display_max_results"]) if settings["display_max_results"] in [25, 50, 100, 200] else 1,
            help="Maximum items to show in tables"
        )
    
    return settings


def render_danger_zone(db: DatabaseManager) -> None:
    """Render danger zone settings."""
    st.subheader("⚠️ Danger Zone")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Cache", use_container_width=True, type="secondary"):
            if st.session_state.get("confirm_clear_cache"):
                with db.get_connection() as conn:
                    conn.execute("DELETE FROM ohlcv_cache")
                    conn.execute("DELETE FROM fundamentals_cache")
                    conn.execute("DELETE FROM news_cache")
                st.success("Cache cleared!")
                st.session_state.confirm_clear_cache = False
            else:
                st.session_state.confirm_clear_cache = True
                st.warning("Click again to confirm")
    
    with col2:
        if st.button("🗑️ Clear Alerts", use_container_width=True, type="secondary"):
            if st.session_state.get("confirm_clear_alerts"):
                with db.get_connection() as conn:
                    conn.execute("DELETE FROM alert_history")
                st.success("Alert history cleared!")
                st.session_state.confirm_clear_alerts = False
            else:
                st.session_state.confirm_clear_alerts = True
                st.warning("Click again to confirm")
    
    with col3:
        if st.button("🗑️ Clear Backtests", use_container_width=True, type="secondary"):
            if st.session_state.get("confirm_clear_backtests"):
                with db.get_connection() as conn:
                    conn.execute("DELETE FROM backtest_runs")
                st.success("Backtest history cleared!")
                st.session_state.confirm_clear_backtests = False
            else:
                st.session_state.confirm_clear_backtests = True
                st.warning("Click again to confirm")


def render_settings_page():
    """Render the settings page."""
    st.title("⚙️ Settings")
    st.markdown("Configure application settings and preferences.")
    
    # Initialize database
    db = DatabaseManager()
    
    # Load current settings
    settings = load_settings(db)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📱 Notifications",
        "🎯 Alerts",
        "🔍 Scanning",
        "🔑 API Keys",
        "🎨 Display"
    ])
    
    with tab1:
        settings = render_telegram_section(db, settings)
    
    with tab2:
        settings = render_alerts_section(settings)
    
    with tab3:
        settings = render_scan_section(settings)
        st.divider()
        settings = render_data_section(settings)
    
    with tab4:
        settings = render_api_section(settings)
    
    with tab5:
        settings = render_display_section(settings)
        st.divider()
        render_danger_zone(db)
    
    # Save button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("💾 Save Settings", use_container_width=True, type="primary"):
            save_settings(db, settings)
            
            # Update environment variable for OpenAI if provided
            if settings["openai_api_key"]:
                os.environ["OPENAI_API_KEY"] = settings["openai_api_key"]
            
            st.success("✅ Settings saved successfully!")
            st.rerun()
    
    # Footer info
    st.divider()
    with st.expander("ℹ️ About Settings Storage"):
        st.markdown("""
        **Settings are stored in:**
        - Local SQLite database (`trend_terminal.db`)
        - Settings persist across sessions
        
        **Security Notes:**
        - API keys are stored locally only
        - For production use, consider using environment variables
        - The `.env` file is ignored by git
        
        **Environment Variables:**
        ```
        OPENAI_API_KEY=sk-...
        TELEGRAM_BOT_TOKEN=123456789:ABC...
        TELEGRAM_CHAT_ID=123456789
        ```
        """)


# Run the page
if __name__ == "__main__":
    render_settings_page()
else:
    render_settings_page()
