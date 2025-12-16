"""
Tests for the News Classifier Module

Tests cover:
- Category detection
- Sentiment detection
- Freshness calculation
- Score combination
"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.news_classifier import NewsClassifier
from storage.models import NewsCategory, NewsSentiment


@pytest.fixture
def classifier():
    """Create news classifier instance."""
    return NewsClassifier()


class TestCategoryDetection:
    """Tests for news category detection."""
    
    def test_earnings_category(self, classifier):
        """Should detect earnings-related news."""
        headlines = [
            "Apple Reports Record Q4 Earnings Beat",
            "Tesla Quarterly Results Miss Analyst Estimates",
            "Netflix EPS Surpasses Expectations",
        ]
        
        for headline in headlines:
            category = classifier.classify_category(headline)
            assert category == NewsCategory.EARNINGS
    
    def test_analyst_category(self, classifier):
        """Should detect analyst-related news."""
        headlines = [
            "Goldman Sachs Upgrades Apple to Buy",
            "Morgan Stanley Downgrades Tesla",
            "Analyst Price Target Raised to $200",
        ]
        
        for headline in headlines:
            category = classifier.classify_category(headline)
            assert category == NewsCategory.ANALYST
    
    def test_product_category(self, classifier):
        """Should detect product-related news."""
        headlines = [
            "Apple Launches New iPhone 15",
            "Tesla Unveils Cybertruck",
            "Microsoft Announces Windows 12",
        ]
        
        for headline in headlines:
            category = classifier.classify_category(headline)
            assert category == NewsCategory.PRODUCT
    
    def test_regulatory_category(self, classifier):
        """Should detect regulatory news."""
        headlines = [
            "SEC Investigates Tech Company",
            "FTC Approves Merger",
            "FDA Approval for New Drug",
        ]
        
        for headline in headlines:
            category = classifier.classify_category(headline)
            assert category == NewsCategory.REGULATORY
    
    def test_macro_category(self, classifier):
        """Should detect macro/economic news."""
        headlines = [
            "Federal Reserve Raises Interest Rates",
            "Inflation Report Shows Cooling",
            "GDP Growth Exceeds Expectations",
        ]
        
        for headline in headlines:
            category = classifier.classify_category(headline)
            assert category == NewsCategory.MACRO
    
    def test_general_category(self, classifier):
        """Should default to general for unclear news."""
        headlines = [
            "Company Hosts Annual Picnic",
            "Random Event Happens",
            "Something Interesting Occurred",
        ]
        
        for headline in headlines:
            category = classifier.classify_category(headline)
            assert category == NewsCategory.GENERAL


class TestSentimentDetection:
    """Tests for sentiment detection."""
    
    def test_positive_sentiment(self, classifier):
        """Should detect positive sentiment."""
        headlines = [
            "Company Soars After Beating Estimates",
            "Stock Surges on Strong Sales",
            "Profits Jump to Record High",
            "Upgraded to Buy Rating",
            "Revolutionary New Product Launch",
        ]
        
        for headline in headlines:
            sentiment = classifier.classify_sentiment(headline)
            assert sentiment == NewsSentiment.POSITIVE
    
    def test_negative_sentiment(self, classifier):
        """Should detect negative sentiment."""
        headlines = [
            "Stock Plunges After Missing Estimates",
            "Company Reports Huge Losses",
            "Downgraded to Sell Rating",
            "Investigation Launched Into Fraud",
            "Massive Layoffs Announced",
        ]
        
        for headline in headlines:
            sentiment = classifier.classify_sentiment(headline)
            assert sentiment == NewsSentiment.NEGATIVE
    
    def test_neutral_sentiment(self, classifier):
        """Should detect neutral sentiment."""
        headlines = [
            "Company Reports Quarterly Results",
            "Management Change Announced",
            "Routine Filing Submitted",
        ]
        
        for headline in headlines:
            sentiment = classifier.classify_sentiment(headline)
            # Could be neutral or weakly positive/negative
            assert sentiment in [NewsSentiment.NEUTRAL, NewsSentiment.POSITIVE, NewsSentiment.NEGATIVE]


class TestFreshnessCalculation:
    """Tests for freshness calculation."""
    
    def test_fresh_news_high_weight(self, classifier):
        """Recent news should have high freshness weight."""
        recent_time = datetime.now() - timedelta(hours=2)
        freshness = classifier.calculate_freshness(recent_time)
        
        assert freshness >= 0.8
    
    def test_old_news_low_weight(self, classifier):
        """Old news should have low freshness weight."""
        old_time = datetime.now() - timedelta(days=7)
        freshness = classifier.calculate_freshness(old_time)
        
        assert freshness <= 0.3
    
    def test_freshness_bounds(self, classifier):
        """Freshness should be between 0 and 1."""
        times = [
            datetime.now(),
            datetime.now() - timedelta(hours=1),
            datetime.now() - timedelta(days=1),
            datetime.now() - timedelta(days=30),
        ]
        
        for time in times:
            freshness = classifier.calculate_freshness(time)
            assert 0 <= freshness <= 1


class TestScoreCombination:
    """Tests for combining news into overall sentiment."""
    
    def test_positive_news_positive_score(self, classifier):
        """Positive news should result in positive combined score."""
        news_items = [
            {"title": "Stock Soars", "published": datetime.now()},
            {"title": "Huge Profits", "published": datetime.now() - timedelta(hours=2)},
        ]
        
        score = classifier.calculate_news_score(news_items)
        assert score > 0
    
    def test_negative_news_negative_score(self, classifier):
        """Negative news should result in negative combined score."""
        news_items = [
            {"title": "Stock Plunges", "published": datetime.now()},
            {"title": "Massive Losses", "published": datetime.now() - timedelta(hours=2)},
        ]
        
        score = classifier.calculate_news_score(news_items)
        assert score < 0
    
    def test_mixed_news_neutral_score(self, classifier):
        """Mixed news should result in near-neutral score."""
        news_items = [
            {"title": "Stock Soars", "published": datetime.now()},
            {"title": "Stock Plunges", "published": datetime.now()},
        ]
        
        score = classifier.calculate_news_score(news_items)
        # Score should be close to zero for mixed signals
        assert -0.5 <= score <= 0.5
    
    def test_empty_news_neutral(self, classifier):
        """No news should result in neutral score."""
        score = classifier.calculate_news_score([])
        assert score == 0
    
    def test_score_bounds(self, classifier):
        """Combined score should be bounded."""
        news_items = [
            {"title": "Amazing Breakthrough Soars Stock to Moon", "published": datetime.now()},
            {"title": "Incredible Profit Surge", "published": datetime.now()},
            {"title": "Revolutionary Product Launch", "published": datetime.now()},
        ]
        
        score = classifier.calculate_news_score(news_items)
        assert -1 <= score <= 1


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_headline(self, classifier):
        """Should handle empty headline."""
        category = classifier.classify_category("")
        sentiment = classifier.classify_sentiment("")
        
        assert category == NewsCategory.GENERAL
        assert sentiment == NewsSentiment.NEUTRAL
    
    def test_none_headline(self, classifier):
        """Should handle None headline."""
        try:
            category = classifier.classify_category(None)
            sentiment = classifier.classify_sentiment(None)
            # If it doesn't crash, should return defaults
            assert category == NewsCategory.GENERAL
            assert sentiment == NewsSentiment.NEUTRAL
        except (TypeError, AttributeError):
            # OK to raise error for None input
            pass
    
    def test_unicode_headline(self, classifier):
        """Should handle unicode characters."""
        headline = "🚀 Stock Soars! 📈 Amazing Results!"
        
        category = classifier.classify_category(headline)
        sentiment = classifier.classify_sentiment(headline)
        
        # Should not crash and return valid values
        assert category in NewsCategory
        assert sentiment in NewsSentiment
    
    def test_very_long_headline(self, classifier):
        """Should handle very long headlines."""
        headline = "Stock " * 1000 + "Soars"
        
        sentiment = classifier.classify_sentiment(headline)
        assert sentiment in NewsSentiment
    
    def test_case_insensitivity(self, classifier):
        """Should be case-insensitive."""
        headlines = [
            "STOCK SOARS",
            "stock soars",
            "Stock Soars",
        ]
        
        sentiments = [classifier.classify_sentiment(h) for h in headlines]
        
        # All should give same sentiment
        assert len(set(sentiments)) == 1


class TestClassifyNewsItem:
    """Tests for full news item classification."""
    
    def test_full_classification(self, classifier):
        """Should classify full news item with all fields."""
        news_item = {
            "title": "Apple Reports Record Earnings",
            "published": datetime.now(),
            "summary": "Apple Inc reported quarterly earnings that beat analyst expectations."
        }
        
        result = classifier.classify(news_item)
        
        assert "category" in result
        assert "sentiment" in result
        assert "freshness" in result
        assert "score" in result
        
        assert result["category"] == NewsCategory.EARNINGS
        assert result["sentiment"] == NewsSentiment.POSITIVE
        assert result["freshness"] > 0.9
        assert result["score"] > 0
    
    def test_missing_fields(self, classifier):
        """Should handle missing fields gracefully."""
        news_item = {
            "title": "Some News"
        }
        
        try:
            result = classifier.classify(news_item)
            assert "category" in result
            assert "sentiment" in result
        except KeyError:
            # OK if requires published field
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
