"""
LLM Analyzer for Trend Terminal.
Uses OpenAI GPT for intelligent market summaries and signal explanations.
"""

import logging
import os
from typing import List, Dict, Optional, Any
from datetime import datetime

from openai import OpenAI

from storage.models import (
    SignalRecord,
    SignalDirection,
    NewsRecord,
    ScanResult,
    BacktestMetrics,
    MarketOverview,
)

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """
    Uses LLM to generate intelligent market analysis and summaries.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ):
        """
        Initialize LLM analyzer.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
            model: Model to use
            max_tokens: Maximum response tokens
            temperature: Response creativity (0-1)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        
        if self.api_key:
            self._client = OpenAI(api_key=self.api_key)
            logger.info(f"LLM Analyzer initialized with model: {model}")
        else:
            logger.warning("No OpenAI API key found. LLM features will be disabled.")
    
    @property
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self._client is not None
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Make a call to the LLM.
        
        Args:
            system_prompt: System context prompt
            user_prompt: User query
            
        Returns:
            LLM response or None on error
        """
        if not self.is_available:
            return None
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
    
    def generate_market_summary(
        self,
        market_overview: MarketOverview,
        top_bullish: List[ScanResult],
        top_bearish: List[ScanResult],
    ) -> str:
        """
        Generate an executive market summary.
        
        Args:
            market_overview: Current market overview
            top_bullish: Top bullish signals
            top_bearish: Top bearish signals
            
        Returns:
            Professional market summary
        """
        if not self.is_available:
            return self._generate_fallback_market_summary(market_overview, top_bullish, top_bearish)
        
        system_prompt = """You are a professional financial analyst providing market insights.
Your tone is informative, objective, and professional. Never give direct buy/sell recommendations.
Focus on explaining market conditions, key movers, and notable patterns.
Keep responses concise and actionable. Use professional financial language."""

        # Prepare data
        bullish_summary = "\n".join([
            f"- {s.ticker}: +{s.change_1d_pct:.1f}% today, score {s.final_score:.0f}/100"
            for s in top_bullish[:5]
        ])
        
        bearish_summary = "\n".join([
            f"- {s.ticker}: {s.change_1d_pct:.1f}% today, score {s.final_score:.0f}/100"
            for s in top_bearish[:5]
        ])
        
        user_prompt = f"""Generate a brief market summary (3-4 paragraphs) based on:

Market Index ({market_overview.index_ticker}):
- Last Price: ${market_overview.last_price:.2f}
- 1-Day Change: {market_overview.change_1d_pct:.2f}%
- 5-Day Change: {market_overview.change_5d_pct:.2f}%
- Trend: {market_overview.trend.value}

Top Bullish Signals:
{bullish_summary if bullish_summary else "None detected"}

Top Bearish Signals:
{bearish_summary if bearish_summary else "None detected"}

Provide:
1. Overall market sentiment and context
2. Key bullish themes/sectors if any
3. Key bearish themes/concerns if any
4. Important considerations for traders

Remember: No direct trading recommendations. Focus on analysis and education."""

        response = self._call_llm(system_prompt, user_prompt)
        return response or self._generate_fallback_market_summary(market_overview, top_bullish, top_bearish)
    
    def _generate_fallback_market_summary(
        self,
        market_overview: MarketOverview,
        top_bullish: List[ScanResult],
        top_bearish: List[ScanResult],
    ) -> str:
        """Generate a basic summary without LLM."""
        trend_desc = {
            SignalDirection.BULLISH: "showing bullish momentum",
            SignalDirection.BEARISH: "exhibiting bearish pressure",
            SignalDirection.NEUTRAL: "trading in a neutral range",
        }
        
        summary = f"""## Market Overview

The {market_overview.index_name} ({market_overview.index_ticker}) is currently {trend_desc.get(market_overview.trend, 'neutral')}.

**Key Metrics:**
- Last Price: ${market_overview.last_price:.2f}
- 1-Day Change: {market_overview.change_1d_pct:+.2f}%
- 5-Day Change: {market_overview.change_5d_pct:+.2f}%

**Signal Summary:**
- Bullish signals detected: {len(top_bullish)}
- Bearish signals detected: {len(top_bearish)}

*Note: This is an automated summary. Enable AI insights for detailed analysis.*
"""
        return summary
    
    def explain_signal(
        self,
        signal: SignalRecord,
        news_list: Optional[List[NewsRecord]] = None,
    ) -> str:
        """
        Generate a detailed explanation for a signal.
        
        Args:
            signal: Signal to explain
            news_list: Related news items
            
        Returns:
            Detailed signal explanation
        """
        if not self.is_available:
            return self._generate_fallback_signal_explanation(signal)
        
        system_prompt = """You are a technical analyst explaining trading signals.
Be educational and objective. Explain the WHY behind the signal.
Never guarantee outcomes or give direct trading advice.
Use proper risk disclosure language."""

        indicators = signal.indicators
        news_text = ""
        if news_list:
            news_text = "\n".join([
                f"- {n.headline} (Sentiment: {n.sentiment.value})"
                for n in news_list[:5]
            ])
        
        user_prompt = f"""Explain this {signal.direction.value} signal for {signal.ticker}:

Technical Indicators:
- Price: ${indicators.close:.2f}
- EMA20: ${indicators.ema20:.2f if indicators.ema20 else 'N/A'}
- EMA50: ${indicators.ema50:.2f if indicators.ema50 else 'N/A'}
- RSI: {indicators.rsi:.1f if indicators.rsi else 'N/A'}
- Volume Spike: {indicators.volume_spike:.1f}x average
- 1-Day Return: {(indicators.return_1d or 0) * 100:.1f}%

Signal Metrics:
- Trend Score: {signal.trend_score:.0f}/100
- Volume Score: {signal.volume_score:.0f}/100
- News Score: {signal.news_score:.0f}/100
- Final Score: {signal.final_score:.0f}/100
- Confidence: {signal.confidence:.0%}

Recent News:
{news_text if news_text else "No recent news"}

System Reasons:
{chr(10).join(f"- {r}" for r in signal.reasons)}

Provide:
1. What this signal means technically (2-3 sentences)
2. Key factors driving the signal
3. Important risks/considerations
4. What to watch going forward

Keep it educational. No specific price targets or guarantees."""

        response = self._call_llm(system_prompt, user_prompt)
        return response or self._generate_fallback_signal_explanation(signal)
    
    def _generate_fallback_signal_explanation(self, signal: SignalRecord) -> str:
        """Generate a basic signal explanation without LLM."""
        direction_text = {
            SignalDirection.BULLISH: "bullish (potentially upward)",
            SignalDirection.BEARISH: "bearish (potentially downward)",
            SignalDirection.NEUTRAL: "neutral (no clear direction)",
        }
        
        explanation = f"""## Signal Analysis for {signal.ticker}

**Direction:** {direction_text.get(signal.direction, 'neutral')}

**Score Breakdown:**
- Technical/Trend: {signal.trend_score:.0f}/100
- Volume: {signal.volume_score:.0f}/100
- News Sentiment: {signal.news_score:.0f}/100
- **Overall: {signal.final_score:.0f}/100**

**Key Factors:**
"""
        for reason in signal.reasons:
            explanation += f"- {reason}\n"
        
        explanation += """
**⚠️ Risk Notice:** This is an algorithmic signal based on technical indicators and news sentiment. 
Past performance does not guarantee future results. Always conduct your own research before making investment decisions.

*Enable AI insights for more detailed analysis.*
"""
        return explanation
    
    def synthesize_news(
        self,
        ticker: str,
        news_list: List[NewsRecord],
        company_name: Optional[str] = None,
    ) -> str:
        """
        Synthesize multiple news items into a coherent summary.
        
        Args:
            ticker: Stock ticker
            news_list: List of news records
            company_name: Company name
            
        Returns:
            News synthesis
        """
        if not news_list:
            return f"No recent news available for {ticker}."
        
        if not self.is_available:
            return self._generate_fallback_news_synthesis(ticker, news_list)
        
        system_prompt = """You are a financial news analyst synthesizing multiple news items.
Extract the key themes and implications. Be objective and balanced.
Identify sentiment drivers and potential market impact."""

        news_text = "\n".join([
            f"- [{n.published_at.strftime('%m/%d %H:%M')}] {n.headline} (Source: {n.source or 'Unknown'})"
            for n in sorted(news_list, key=lambda x: x.published_at, reverse=True)[:10]
        ])
        
        user_prompt = f"""Synthesize these news items for {company_name or ticker} ({ticker}):

{news_text}

Provide:
1. Main narrative/theme (1-2 sentences)
2. Key developments
3. Sentiment assessment
4. Potential implications

Keep it concise and professional."""

        response = self._call_llm(system_prompt, user_prompt)
        return response or self._generate_fallback_news_synthesis(ticker, news_list)
    
    def _generate_fallback_news_synthesis(
        self,
        ticker: str,
        news_list: List[NewsRecord],
    ) -> str:
        """Generate a basic news synthesis without LLM."""
        bullish = sum(1 for n in news_list if n.sentiment.value > 0)
        bearish = sum(1 for n in news_list if n.sentiment.value < 0)
        neutral = len(news_list) - bullish - bearish
        
        synthesis = f"""## News Summary for {ticker}

**Recent Headlines ({len(news_list)} articles):**
"""
        for news in sorted(news_list, key=lambda x: x.published_at, reverse=True)[:5]:
            sentiment_icon = "🟢" if news.sentiment.value > 0 else "🔴" if news.sentiment.value < 0 else "⚪"
            synthesis += f"- {sentiment_icon} {news.headline}\n"
        
        synthesis += f"""
**Sentiment Distribution:**
- Bullish: {bullish}
- Bearish: {bearish}
- Neutral: {neutral}

*Enable AI insights for detailed news synthesis.*
"""
        return synthesis
    
    def analyze_backtest(
        self,
        metrics: BacktestMetrics,
        trades_count: int,
        strategy_name: str = "Signal Strategy",
    ) -> str:
        """
        Generate backtest analysis and insights.
        
        Args:
            metrics: Backtest performance metrics
            trades_count: Number of trades
            strategy_name: Name of strategy
            
        Returns:
            Backtest analysis
        """
        if not self.is_available:
            return self._generate_fallback_backtest_analysis(metrics, trades_count, strategy_name)
        
        system_prompt = """You are a quantitative analyst reviewing backtest results.
Provide objective assessment of strategy performance.
Highlight strengths, weaknesses, and areas for improvement.
Include appropriate caveats about backtesting limitations."""

        user_prompt = f"""Analyze this backtest for "{strategy_name}":

Performance Metrics:
- Total Trades: {trades_count}
- Win Rate: {metrics.win_rate:.1%}
- Average Return: {metrics.avg_return:.2%}
- Cumulative Return: {metrics.cumulative_return:.2%}
- Max Drawdown: {metrics.max_drawdown:.2%}
- Average Winner: {metrics.avg_winner:.2%}
- Average Loser: {metrics.avg_loser:.2%}
- Profit Factor: {metrics.profit_factor:.2f if metrics.profit_factor else 'N/A'}
- Sharpe Ratio: {metrics.sharpe_ratio:.2f if metrics.sharpe_ratio else 'N/A'}

Provide:
1. Overall assessment (is this a viable strategy?)
2. Key strengths
3. Key weaknesses/risks
4. Suggestions for improvement
5. Important caveats

Be balanced and realistic. Acknowledge backtesting limitations."""

        response = self._call_llm(system_prompt, user_prompt)
        return response or self._generate_fallback_backtest_analysis(metrics, trades_count, strategy_name)
    
    def _generate_fallback_backtest_analysis(
        self,
        metrics: BacktestMetrics,
        trades_count: int,
        strategy_name: str,
    ) -> str:
        """Generate a basic backtest analysis without LLM."""
        assessment = "promising" if metrics.cumulative_return > 0 and metrics.win_rate > 0.5 else "needs review"
        
        analysis = f"""## Backtest Analysis: {strategy_name}

**Overall Assessment:** {assessment.title()}

**Performance Summary:**
- Total Trades: {trades_count}
- Win Rate: {metrics.win_rate:.1%}
- Cumulative Return: {metrics.cumulative_return:+.2%}
- Max Drawdown: {metrics.max_drawdown:.2%}

**Risk Metrics:**
- Average Winner: {metrics.avg_winner:+.2%}
- Average Loser: {metrics.avg_loser:.2%}
- Risk/Reward: {abs(metrics.avg_winner/metrics.avg_loser) if metrics.avg_loser != 0 else 'N/A':.2f}

**⚠️ Important Caveats:**
- Backtesting does not guarantee future performance
- Results may be affected by survivorship bias
- Transaction costs and slippage are estimates
- Market conditions change over time

*Enable AI insights for detailed analysis.*
"""
        return analysis
    
    def generate_alert_message(
        self,
        signal: SignalRecord,
        include_ai_insight: bool = True,
    ) -> str:
        """
        Generate a notification alert message.
        
        Args:
            signal: Signal to alert about
            include_ai_insight: Whether to include AI-generated insight
            
        Returns:
            Formatted alert message
        """
        direction_emoji = {
            SignalDirection.BULLISH: "🟢",
            SignalDirection.BEARISH: "🔴",
            SignalDirection.NEUTRAL: "⚪",
        }
        
        emoji = direction_emoji.get(signal.direction, "⚪")
        indicators = signal.indicators
        
        message = f"""
{emoji} **{signal.direction.value.upper()} Signal: {signal.ticker}**

📊 **Score:** {signal.final_score:.0f}/100 (Confidence: {signal.confidence:.0%})

💰 **Price:** ${indicators.close:.2f}
📈 **1D Change:** {(indicators.return_1d or 0) * 100:+.1f}%
📊 **Volume:** {indicators.volume_spike:.1f}x average

**Key Factors:**
"""
        for reason in signal.reasons[:5]:
            message += f"• {reason}\n"
        
        # Add AI insight if available and requested
        if include_ai_insight and self.is_available:
            system_prompt = "You are a concise financial analyst. Provide a 1-2 sentence insight."
            user_prompt = f"Brief insight for {signal.direction.value} signal on {signal.ticker} with score {signal.final_score:.0f}/100?"
            insight = self._call_llm(system_prompt, user_prompt)
            if insight:
                message += f"\n💡 **AI Insight:** {insight}\n"
        
        message += """
⚠️ *This is an automated signal, not financial advice. Do your own research.*
"""
        return message
