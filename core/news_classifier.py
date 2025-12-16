"""
News Classifier for Trend Terminal.
Classifies news by category and sentiment using keyword matching.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import math

from storage.models import NewsRecord, NewsCategory, NewsSentiment

logger = logging.getLogger(__name__)


# Category keywords mapping
CATEGORY_KEYWORDS = {
    NewsCategory.EARNINGS: [
        "earnings", "quarterly", "revenue", "eps", "profit", "loss",
        "guidance", "forecast", "beat", "miss", "results", "fiscal",
        "quarter", "annual", "outlook", "q1", "q2", "q3", "q4",
    ],
    NewsCategory.UPGRADE_DOWNGRADE: [
        "upgrade", "downgrade", "price target", "rating", "analyst",
        "buy", "sell", "hold", "outperform", "underperform", "overweight",
        "underweight", "neutral", "raises", "lowers", "maintains",
        "initiates", "coverage", "recommendation",
    ],
    NewsCategory.MA: [
        "acquisition", "merger", "takeover", "buyout", "deal", "bid",
        "acquire", "merge", "purchase", "divest", "spin-off", "spinoff",
        "combination", "consolidation",
    ],
    NewsCategory.REGULATORY: [
        "fda", "sec", "regulation", "approval", "clearance", "compliance",
        "antitrust", "ftc", "doj", "regulatory", "approve", "reject",
        "investigation", "probe", "subpoena",
    ],
    NewsCategory.LEGAL: [
        "lawsuit", "sued", "settlement", "fraud", "fine", "penalty",
        "litigation", "court", "judge", "verdict", "legal", "class action",
        "securities fraud", "whistleblower",
    ],
    NewsCategory.PRODUCT: [
        "launch", "partnership", "contract", "innovation", "patent",
        "technology", "release", "announce", "unveil", "introduce",
        "collaborate", "agreement", "alliance", "venture",
    ],
}

# Sentiment keywords
BULLISH_KEYWORDS = [
    "surge", "soar", "rally", "jump", "beat", "exceed", "strong", "growth",
    "upgrade", "positive", "record", "breakthrough", "gain", "rise", "climb",
    "outperform", "bullish", "optimistic", "boost", "accelerate", "expand",
    "profit", "success", "win", "exceed expectations", "above estimates",
    "raises guidance", "upside", "momentum", "recovery",
]

BEARISH_KEYWORDS = [
    "drop", "fall", "plunge", "miss", "decline", "weak", "downgrade",
    "negative", "concern", "warning", "cut", "layoff", "bearish", "pessimistic",
    "slump", "crash", "tumble", "sink", "lose", "loss", "below estimates",
    "lowers guidance", "downside", "risk", "recession", "slowdown",
    "disappointing", "struggle", "fail", "trouble", "crisis",
]


class NewsClassifier:
    """
    Classifies news articles by category and sentiment.
    """
    
    def __init__(
        self,
        decay_rate: float = 0.1,
        lookback_hours: int = 48,
    ):
        """
        Initialize news classifier.
        
        Args:
            decay_rate: Exponential decay rate for freshness weighting
            lookback_hours: How far back to consider news
        """
        self.decay_rate = decay_rate
        self.lookback_hours = lookback_hours
        
        # Compile regex patterns for efficiency
        self._category_patterns = {
            cat: re.compile(r'\b(' + '|'.join(kw) + r')\b', re.IGNORECASE)
            for cat, kw in CATEGORY_KEYWORDS.items()
        }
        self._bullish_pattern = re.compile(
            r'\b(' + '|'.join(BULLISH_KEYWORDS) + r')\b', re.IGNORECASE
        )
        self._bearish_pattern = re.compile(
            r'\b(' + '|'.join(BEARISH_KEYWORDS) + r')\b', re.IGNORECASE
        )
    
    def classify_category(self, headline: str, summary: Optional[str] = None) -> NewsCategory:
        """
        Classify news into a category.
        
        Args:
            headline: News headline
            summary: Optional news summary
            
        Returns:
            NewsCategory enum value
        """
        text = headline
        if summary:
            text += " " + summary
        
        # Count matches for each category
        scores = {}
        for category, pattern in self._category_patterns.items():
            matches = pattern.findall(text)
            scores[category] = len(matches)
        
        # Return category with highest score, or OTHER if no matches
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return NewsCategory.OTHER
    
    def classify_sentiment(
        self,
        headline: str,
        summary: Optional[str] = None,
    ) -> Tuple[NewsSentiment, float]:
        """
        Classify news sentiment.
        
        Args:
            headline: News headline
            summary: Optional news summary
            
        Returns:
            Tuple of (NewsSentiment, score from -1 to 1)
        """
        text = headline
        if summary:
            text += " " + summary
        
        # Count bullish and bearish keywords
        bullish_matches = self._bullish_pattern.findall(text)
        bearish_matches = self._bearish_pattern.findall(text)
        
        bullish_count = len(bullish_matches)
        bearish_count = len(bearish_matches)
        
        # Calculate sentiment score
        total = bullish_count + bearish_count
        if total == 0:
            return NewsSentiment.NEUTRAL, 0.0
        
        score = (bullish_count - bearish_count) / total
        
        # Determine sentiment
        if score > 0.2:
            return NewsSentiment.BULLISH, score
        elif score < -0.2:
            return NewsSentiment.BEARISH, score
        else:
            return NewsSentiment.NEUTRAL, score
    
    def classify_news(self, news: NewsRecord) -> NewsRecord:
        """
        Classify a single news record.
        
        Args:
            news: NewsRecord to classify
            
        Returns:
            NewsRecord with category and sentiment populated
        """
        news.category = self.classify_category(news.headline, news.summary)
        news.sentiment, news.sentiment_score = self.classify_sentiment(
            news.headline, news.summary
        )
        return news
    
    def classify_batch(self, news_list: List[NewsRecord]) -> List[NewsRecord]:
        """
        Classify multiple news records.
        
        Args:
            news_list: List of NewsRecord objects
            
        Returns:
            List of classified NewsRecord objects
        """
        return [self.classify_news(n) for n in news_list]
    
    def calculate_freshness_weight(
        self,
        published_at: datetime,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """
        Calculate freshness weight for a news item.
        Uses exponential decay.
        
        Args:
            published_at: News publish time
            reference_time: Reference time (default: now)
            
        Returns:
            Weight between 0 and 1
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        hours_old = (reference_time - published_at).total_seconds() / 3600
        
        if hours_old < 0:  # Future date
            return 1.0
        
        if hours_old > self.lookback_hours:
            return 0.0
        
        # Exponential decay
        weight = math.exp(-self.decay_rate * hours_old)
        return weight
    
    def calculate_news_score(
        self,
        news_list: List[NewsRecord],
        reference_time: Optional[datetime] = None,
    ) -> Tuple[float, List[NewsRecord]]:
        """
        Calculate aggregate news score for a list of news items.
        
        Args:
            news_list: List of NewsRecord objects (should be classified)
            reference_time: Reference time for freshness
            
        Returns:
            Tuple of (normalized score 0-100, top news items)
        """
        if not news_list:
            return 50.0, []  # Neutral score if no news
        
        if reference_time is None:
            reference_time = datetime.now()
        
        # Filter to recent news
        recent_news = [
            n for n in news_list
            if (reference_time - n.published_at).total_seconds() / 3600 <= self.lookback_hours
        ]
        
        if not recent_news:
            return 50.0, []
        
        # Calculate weighted sentiment
        total_weight = 0.0
        weighted_sentiment = 0.0
        
        for news in recent_news:
            weight = self.calculate_freshness_weight(news.published_at, reference_time)
            sentiment_value = news.sentiment.value if isinstance(news.sentiment, NewsSentiment) else news.sentiment
            
            weighted_sentiment += sentiment_value * news.sentiment_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 50.0, recent_news
        
        # Normalize to 0-100 scale
        avg_sentiment = weighted_sentiment / total_weight  # Range: -1 to 1
        normalized_score = (avg_sentiment + 1) * 50  # Range: 0 to 100
        
        # Sort by freshness and sentiment strength
        sorted_news = sorted(
            recent_news,
            key=lambda n: (
                self.calculate_freshness_weight(n.published_at, reference_time) *
                abs(n.sentiment_score)
            ),
            reverse=True
        )
        
        return normalized_score, sorted_news[:5]  # Return top 5
    
    def get_news_summary(
        self,
        news_list: List[NewsRecord],
        max_items: int = 5,
    ) -> Dict:
        """
        Get a summary of news items.
        
        Args:
            news_list: List of NewsRecord objects
            max_items: Maximum items to include
            
        Returns:
            Summary dict with counts and top items
        """
        if not news_list:
            return {
                "total_count": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "categories": {},
                "top_items": [],
            }
        
        # Count by sentiment
        bullish = sum(1 for n in news_list if n.sentiment == NewsSentiment.BULLISH)
        bearish = sum(1 for n in news_list if n.sentiment == NewsSentiment.BEARISH)
        neutral = len(news_list) - bullish - bearish
        
        # Count by category
        categories = {}
        for news in news_list:
            cat = news.category.value if isinstance(news.category, NewsCategory) else news.category
            categories[cat] = categories.get(cat, 0) + 1
        
        # Top items by recency
        sorted_news = sorted(news_list, key=lambda n: n.published_at, reverse=True)
        top_items = [
            {
                "headline": n.headline,
                "source": n.source,
                "url": n.url,
                "published_at": n.published_at.isoformat() if n.published_at else None,
                "sentiment": n.sentiment.value if isinstance(n.sentiment, NewsSentiment) else n.sentiment,
                "category": n.category.value if isinstance(n.category, NewsCategory) else n.category,
            }
            for n in sorted_news[:max_items]
        ]
        
        return {
            "total_count": len(news_list),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "categories": categories,
            "top_items": top_items,
        }
    
    def filter_by_category(
        self,
        news_list: List[NewsRecord],
        categories: List[NewsCategory],
    ) -> List[NewsRecord]:
        """
        Filter news by categories.
        
        Args:
            news_list: List of NewsRecord objects
            categories: Categories to include
            
        Returns:
            Filtered list
        """
        return [n for n in news_list if n.category in categories]
    
    def filter_by_sentiment(
        self,
        news_list: List[NewsRecord],
        sentiments: List[NewsSentiment],
    ) -> List[NewsRecord]:
        """
        Filter news by sentiments.
        
        Args:
            news_list: List of NewsRecord objects
            sentiments: Sentiments to include
            
        Returns:
            Filtered list
        """
        return [n for n in news_list if n.sentiment in sentiments]


def get_category_display_name(category: NewsCategory) -> str:
    """Get display name for category."""
    display_names = {
        NewsCategory.EARNINGS: "📊 Earnings",
        NewsCategory.UPGRADE_DOWNGRADE: "📈 Analyst Rating",
        NewsCategory.MA: "🤝 M&A",
        NewsCategory.REGULATORY: "📋 Regulatory",
        NewsCategory.LEGAL: "⚖️ Legal",
        NewsCategory.PRODUCT: "🚀 Product/Partnership",
        NewsCategory.OTHER: "📰 General",
    }
    return display_names.get(category, "📰 General")


def get_sentiment_display(sentiment: NewsSentiment, score: float) -> str:
    """Get display string for sentiment."""
    if sentiment == NewsSentiment.BULLISH:
        if score > 0.5:
            return "🟢 Very Bullish"
        return "🟢 Bullish"
    elif sentiment == NewsSentiment.BEARISH:
        if score < -0.5:
            return "🔴 Very Bearish"
        return "🔴 Bearish"
    else:
        return "⚪ Neutral"
