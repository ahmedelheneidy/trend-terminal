"""
Telegram Notification Module

Handles sending alerts and updates via Telegram bot.
Includes message formatting, rate limiting, and deduplication.
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import os

logger = logging.getLogger(__name__)


@dataclass
class NotificationMessage:
    """Represents a notification message."""
    
    message_type: str  # 'signal', 'alert', 'summary', 'error'
    title: str
    body: str
    ticker: Optional[str] = None
    priority: str = "normal"  # 'low', 'normal', 'high', 'urgent'
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def message_hash(self) -> str:
        """Generate hash for deduplication."""
        content = f"{self.message_type}:{self.ticker}:{self.title}"
        return hashlib.md5(content.encode()).hexdigest()


class TelegramNotifier:
    """
    Telegram notification handler with rate limiting and deduplication.
    
    Features:
    - Async message sending
    - Rate limiting (max messages per minute)
    - Deduplication window
    - Message formatting with Markdown
    - Error handling and retry logic
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        rate_limit: int = 20,  # messages per minute
        dedup_window_minutes: int = 60,
    ):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token (from BotFather)
            chat_id: Chat ID to send messages to
            rate_limit: Maximum messages per minute
            dedup_window_minutes: Window for deduplication in minutes
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.rate_limit = rate_limit
        self.dedup_window = timedelta(minutes=dedup_window_minutes)
        
        # Rate limiting
        self._message_times: deque = deque(maxlen=rate_limit)
        
        # Deduplication cache
        self._sent_hashes: Dict[str, datetime] = {}
        
        # Status tracking
        self._enabled = bool(self.bot_token and self.chat_id)
        self._last_error: Optional[str] = None
        self._messages_sent = 0
        self._messages_blocked = 0
    
    @property
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self.bot_token and self.chat_id)
    
    @property
    def is_enabled(self) -> bool:
        """Check if notifications are enabled."""
        return self._enabled and self.is_configured
    
    def enable(self) -> None:
        """Enable notifications."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable notifications."""
        self._enabled = False
    
    def configure(self, bot_token: str, chat_id: str) -> bool:
        """
        Configure Telegram credentials.
        
        Args:
            bot_token: Bot token from BotFather
            chat_id: Chat ID to send to
            
        Returns:
            True if configuration is valid
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._enabled = self.is_configured
        return self.is_configured
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        
        # Remove old timestamps
        while self._message_times and (now - self._message_times[0]).seconds > 60:
            self._message_times.popleft()
        
        return len(self._message_times) < self.rate_limit
    
    def _check_duplicate(self, message: NotificationMessage) -> bool:
        """Check if message is a duplicate."""
        now = datetime.now()
        
        # Clean old hashes
        expired_keys = [
            h for h, t in self._sent_hashes.items()
            if now - t > self.dedup_window
        ]
        for key in expired_keys:
            del self._sent_hashes[key]
        
        # Check for duplicate
        return message.message_hash in self._sent_hashes
    
    def _format_signal_message(self, message: NotificationMessage) -> str:
        """Format a signal notification message."""
        meta = message.metadata
        
        # Direction emoji
        direction = meta.get("direction", "neutral")
        direction_emoji = {
            "bullish": "🟢",
            "bearish": "🔴",
            "neutral": "⚪"
        }.get(direction, "⚪")
        
        # Priority emoji
        priority_emoji = {
            "urgent": "🚨",
            "high": "⚠️",
            "normal": "📊",
            "low": "📝"
        }.get(message.priority, "📊")
        
        # Build message
        lines = [
            f"{priority_emoji} *{message.title}*",
            "",
            f"{direction_emoji} *{message.ticker}* - {direction.upper()}",
            f"📈 Score: {meta.get('score', 'N/A')}/100",
            f"💪 Confidence: {meta.get('confidence', 'N/A')}%",
        ]
        
        # Price info
        if "price" in meta:
            lines.append(f"💵 Price: ${meta['price']:.2f}")
        
        # Change info
        if "change_pct" in meta:
            change = meta["change_pct"]
            change_emoji = "📈" if change >= 0 else "📉"
            lines.append(f"{change_emoji} Change: {change:+.2f}%")
        
        # Reasons
        reasons = meta.get("reasons", [])
        if reasons:
            lines.append("")
            lines.append("*Key Factors:*")
            for reason in reasons[:3]:
                lines.append(f"• {reason}")
        
        # Timestamp
        lines.extend([
            "",
            f"⏰ {message.timestamp.strftime('%Y-%m-%d %H:%M')}"
        ])
        
        return "\n".join(lines)
    
    def _format_alert_message(self, message: NotificationMessage) -> str:
        """Format an alert notification message."""
        meta = message.metadata
        
        alert_type = meta.get("alert_type", "price")
        
        # Alert type emoji
        type_emoji = {
            "price": "💰",
            "volume": "📊",
            "rsi": "📉",
            "news": "📰",
            "signal": "🎯"
        }.get(alert_type, "🔔")
        
        lines = [
            f"{type_emoji} *ALERT: {message.title}*",
            "",
            message.body,
        ]
        
        if message.ticker:
            lines.insert(1, f"Ticker: *{message.ticker}*")
        
        # Condition details
        if "condition" in meta:
            lines.append(f"Condition: {meta['condition']}")
        
        if "current_value" in meta:
            lines.append(f"Current: {meta['current_value']}")
        
        if "threshold" in meta:
            lines.append(f"Threshold: {meta['threshold']}")
        
        lines.extend([
            "",
            f"⏰ {message.timestamp.strftime('%Y-%m-%d %H:%M')}"
        ])
        
        return "\n".join(lines)
    
    def _format_summary_message(self, message: NotificationMessage) -> str:
        """Format a summary notification message."""
        meta = message.metadata
        
        lines = [
            f"📋 *{message.title}*",
            "",
            message.body,
        ]
        
        # Stats if available
        if "bullish_count" in meta:
            lines.extend([
                "",
                "*Signal Summary:*",
                f"🟢 Bullish: {meta.get('bullish_count', 0)}",
                f"🔴 Bearish: {meta.get('bearish_count', 0)}",
                f"⚪ Neutral: {meta.get('neutral_count', 0)}",
            ])
        
        if "top_signals" in meta:
            lines.append("")
            lines.append("*Top Signals:*")
            for sig in meta["top_signals"][:5]:
                direction_emoji = "🟢" if sig.get("direction") == "bullish" else "🔴"
                lines.append(f"{direction_emoji} {sig['ticker']}: {sig['score']}/100")
        
        lines.extend([
            "",
            f"⏰ {message.timestamp.strftime('%Y-%m-%d %H:%M')}"
        ])
        
        return "\n".join(lines)
    
    def _format_error_message(self, message: NotificationMessage) -> str:
        """Format an error notification message."""
        lines = [
            f"❌ *ERROR: {message.title}*",
            "",
            f"```{message.body}```",
            "",
            f"⏰ {message.timestamp.strftime('%Y-%m-%d %H:%M')}"
        ]
        
        return "\n".join(lines)
    
    def format_message(self, message: NotificationMessage) -> str:
        """Format a message based on its type."""
        formatters = {
            "signal": self._format_signal_message,
            "alert": self._format_alert_message,
            "summary": self._format_summary_message,
            "error": self._format_error_message,
        }
        
        formatter = formatters.get(message.message_type, self._format_alert_message)
        return formatter(message)
    
    async def _send_async(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send message asynchronously using aiohttp."""
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not installed, using sync method")
            return self._send_sync(text, parse_mode)
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        self._last_error = f"HTTP {response.status}: {error_text}"
                        logger.error(f"Telegram send failed: {self._last_error}")
                        return False
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Telegram send error: {e}")
            return False
    
    def _send_sync(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send message synchronously using requests."""
        try:
            import requests
        except ImportError:
            logger.error("requests not installed")
            return False
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            else:
                self._last_error = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Telegram send failed: {self._last_error}")
                return False
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Telegram send error: {e}")
            return False
    
    def send(
        self,
        message: NotificationMessage,
        force: bool = False,
    ) -> bool:
        """
        Send a notification message.
        
        Args:
            message: The notification message to send
            force: Skip rate limiting and deduplication checks
            
        Returns:
            True if message was sent successfully
        """
        if not self.is_enabled:
            logger.debug("Telegram notifications disabled")
            return False
        
        # Check rate limit
        if not force and not self._check_rate_limit():
            logger.warning("Rate limit exceeded, message blocked")
            self._messages_blocked += 1
            return False
        
        # Check duplicate
        if not force and self._check_duplicate(message):
            logger.debug(f"Duplicate message blocked: {message.title}")
            self._messages_blocked += 1
            return False
        
        # Format and send
        text = self.format_message(message)
        
        # Try sync method (simpler for Streamlit)
        success = self._send_sync(text)
        
        if success:
            self._message_times.append(datetime.now())
            self._sent_hashes[message.message_hash] = datetime.now()
            self._messages_sent += 1
            logger.info(f"Telegram message sent: {message.title}")
        
        return success
    
    def send_signal(
        self,
        ticker: str,
        direction: str,
        score: float,
        confidence: float,
        reasons: List[str],
        price: Optional[float] = None,
        change_pct: Optional[float] = None,
        priority: str = "normal",
    ) -> bool:
        """
        Send a signal notification.
        
        Args:
            ticker: Stock ticker
            direction: Signal direction (bullish/bearish/neutral)
            score: Signal score (0-100)
            confidence: Confidence percentage
            reasons: List of signal reasons
            price: Current price
            change_pct: Percent change
            priority: Message priority
            
        Returns:
            True if sent successfully
        """
        message = NotificationMessage(
            message_type="signal",
            title="New Signal Alert",
            body=f"{direction.upper()} signal for {ticker}",
            ticker=ticker,
            priority=priority,
            metadata={
                "direction": direction,
                "score": score,
                "confidence": confidence,
                "reasons": reasons,
                "price": price,
                "change_pct": change_pct,
            }
        )
        
        return self.send(message)
    
    def send_alert(
        self,
        title: str,
        body: str,
        ticker: Optional[str] = None,
        alert_type: str = "price",
        condition: Optional[str] = None,
        current_value: Optional[Any] = None,
        threshold: Optional[Any] = None,
        priority: str = "normal",
    ) -> bool:
        """
        Send a generic alert notification.
        
        Args:
            title: Alert title
            body: Alert body text
            ticker: Associated ticker (if any)
            alert_type: Type of alert (price/volume/rsi/news/signal)
            condition: Condition that triggered the alert
            current_value: Current value that triggered
            threshold: Threshold value
            priority: Message priority
            
        Returns:
            True if sent successfully
        """
        message = NotificationMessage(
            message_type="alert",
            title=title,
            body=body,
            ticker=ticker,
            priority=priority,
            metadata={
                "alert_type": alert_type,
                "condition": condition,
                "current_value": current_value,
                "threshold": threshold,
            }
        )
        
        return self.send(message)
    
    def send_summary(
        self,
        title: str,
        body: str,
        bullish_count: int = 0,
        bearish_count: int = 0,
        neutral_count: int = 0,
        top_signals: Optional[List[Dict]] = None,
    ) -> bool:
        """
        Send a summary notification.
        
        Args:
            title: Summary title
            body: Summary body text
            bullish_count: Number of bullish signals
            bearish_count: Number of bearish signals
            neutral_count: Number of neutral signals
            top_signals: List of top signal dicts
            
        Returns:
            True if sent successfully
        """
        message = NotificationMessage(
            message_type="summary",
            title=title,
            body=body,
            priority="normal",
            metadata={
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "neutral_count": neutral_count,
                "top_signals": top_signals or [],
            }
        )
        
        return self.send(message)
    
    def send_error(self, title: str, error_message: str) -> bool:
        """
        Send an error notification.
        
        Args:
            title: Error title
            error_message: Error details
            
        Returns:
            True if sent successfully
        """
        message = NotificationMessage(
            message_type="error",
            title=title,
            body=error_message,
            priority="high",
        )
        
        return self.send(message, force=True)
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the Telegram connection.
        
        Returns:
            Dict with test results
        """
        if not self.is_configured:
            return {
                "success": False,
                "error": "Telegram not configured. Please set bot token and chat ID.",
            }
        
        try:
            import requests
            
            # Test getMe endpoint
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Invalid bot token: {response.text}",
                }
            
            bot_info = response.json().get("result", {})
            
            # Try sending a test message
            test_message = NotificationMessage(
                message_type="alert",
                title="Connection Test",
                body="✅ Trend Terminal is connected!",
                priority="normal",
            )
            
            sent = self.send(test_message, force=True)
            
            return {
                "success": sent,
                "bot_name": bot_info.get("username", "Unknown"),
                "bot_id": bot_info.get("id"),
                "error": self._last_error if not sent else None,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        return {
            "enabled": self.is_enabled,
            "configured": self.is_configured,
            "messages_sent": self._messages_sent,
            "messages_blocked": self._messages_blocked,
            "rate_limit": self.rate_limit,
            "dedup_window_minutes": self.dedup_window.total_seconds() / 60,
            "last_error": self._last_error,
        }


def get_notifier() -> TelegramNotifier:
    """Get a configured TelegramNotifier instance."""
    return TelegramNotifier()
