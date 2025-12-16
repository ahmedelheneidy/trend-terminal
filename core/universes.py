"""
Stock Universe Loader

Provides stock universes for scanning:
- S&P 500 (top ~100 by market cap)
- NASDAQ 100
- Custom watchlist
- Sector mappings
"""

from typing import List, Dict, Optional, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Market indices for overview
MARKET_INDICES = {
    "^GSPC": {"name": "S&P 500", "symbol": "^GSPC"},
    "^IXIC": {"name": "NASDAQ Composite", "symbol": "^IXIC"},
    "^DJI": {"name": "Dow Jones", "symbol": "^DJI"},
    "^RUT": {"name": "Russell 2000", "symbol": "^RUT"},
    "^VIX": {"name": "VIX Volatility", "symbol": "^VIX"},
}


# Trading style definitions
class TradingStyle:
    """Trading style configurations."""
    SCALP = "scalp"           # Minutes to hours, high frequency
    SWING = "swing"           # Days to weeks
    POSITION = "position"     # Weeks to months
    LONG_TERM = "long_term"   # Months to years (investment)


# Penny stocks (under $5, high volatility potential)
PENNY_STOCKS = [
    "SIRI", "PLUG", "SOFI", "PLTR", "NIO", "LCID", "RIVN", "SNAP", "HOOD",
    "AMC", "BB", "NOK", "CLOV", "WISH", "TLRY", "SNDL", "ACB", "CGC", "HEXO",
    "SPCE", "DKNG", "OPEN", "RKT", "UWMC", "BARK", "CHPT", "GOEV", "RIDE",
    "FSR", "NKLA", "WKHS", "HYLN", "ARVL", "FFIE", "MULN", "ASTS", "DNA",
    "IONQ", "RGTI", "QUBT", "BTBT", "MARA", "RIOT", "COIN", "BITF", "HUT",
    "CLSK", "HIVE", "GREE", "SOS", "EBON", "ANY", "CIFR", "BITF", "DMRC",
]

# High-beta momentum stocks (good for scalping)
SCALP_TICKERS = [
    "TSLA", "NVDA", "AMD", "AAPL", "META", "AMZN", "GOOGL", "NFLX", "COIN",
    "MARA", "RIOT", "SQ", "SHOP", "SNOW", "PLTR", "RBLX", "DKNG", "ROKU",
    "MRNA", "BNTX", "ZM", "DOCU", "NET", "CRWD", "DDOG", "ZS", "OKTA",
    "ABNB", "DASH", "UBER", "LYFT", "PINS", "SNAP", "TWLO", "TTD", "U",
    "AFRM", "UPST", "HOOD", "SOFI", "LCID", "RIVN", "NIO", "XPEV", "LI",
    "SPCE", "ASTR", "RKLB", "ASTS", "IONQ", "RGTI", "DNA", "CRSP", "EDIT",
]

# Swing trading candidates (good momentum, moderate volatility)
SWING_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "CRM",
    "NFLX", "ADBE", "PYPL", "SQ", "SHOP", "SNOW", "DDOG", "CRWD", "ZS",
    "NET", "MDB", "PANW", "FTNT", "WDAY", "NOW", "TEAM", "OKTA", "TWLO",
    "TTD", "ROKU", "PINS", "SNAP", "RBLX", "U", "DKNG", "PENN", "ABNB",
    "UBER", "LYFT", "DASH", "COIN", "MARA", "RIOT", "HOOD", "SOFI", "AFRM",
    "UPST", "LMND", "ROOT", "OPEN", "RDFN", "Z", "ZG", "CVNA", "CARG",
    "W", "ETSY", "CHWY", "PTON", "LULU", "NKE", "SBUX", "CMG", "DPZ",
]

# Long-term investment candidates (stable growth, dividends)
LONG_TERM_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "BRK-B", "JNJ", "UNH", "V", "MA", "PG",
    "HD", "JPM", "BAC", "WFC", "XOM", "CVX", "KO", "PEP", "WMT", "COST",
    "MCD", "DIS", "CMCSA", "VZ", "T", "INTC", "CSCO", "IBM", "ORCL", "ACN",
    "TXN", "QCOM", "AVGO", "LLY", "MRK", "PFE", "ABBV", "TMO", "ABT", "DHR",
    "MDT", "BMY", "AMGN", "GILD", "ISRG", "SYK", "BDX", "ZTS", "CI", "HUM",
    "CVS", "UNP", "UPS", "FDX", "CAT", "DE", "HON", "GE", "MMM", "RTX",
    "BA", "LMT", "NOC", "GD", "NEE", "DUK", "SO", "D", "AEP", "XEL",
    "SPG", "PLD", "AMT", "CCI", "O", "EQIX", "DLR", "PSA", "AVB", "EQR",
]

# Top S&P 500 stocks by market cap (representative sample)
SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH",
    "JNJ", "XOM", "V", "JPM", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "AVGO", "COST", "TMO", "WMT", "MCD", "CSCO", "ACN",
    "ABT", "DHR", "NEE", "VZ", "ADBE", "NKE", "CMCSA", "PM", "TXN", "CRM",
    "BMY", "RTX", "HON", "QCOM", "UPS", "T", "COP", "ORCL", "LOW", "UNP",
    "BA", "INTC", "IBM", "SPGI", "CAT", "SBUX", "GE", "INTU", "PLD", "AMD",
    "AMGN", "DE", "MS", "BLK", "GS", "MDLZ", "GILD", "ADI", "AXP", "BKNG",
    "ISRG", "LMT", "SYK", "TMUS", "TJX", "MMC", "CB", "ADP", "VRTX", "MO",
    "CI", "REGN", "NOW", "ZTS", "SO", "ETN", "DUK", "CME", "PGR", "EOG",
    "BDX", "NOC", "SLB", "ITW", "APD", "CSX", "AON", "CL", "HUM", "FDX",
]

# NASDAQ 100 stocks
NASDAQ100_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "PEP",
    "COST", "CSCO", "ADBE", "CMCSA", "TXN", "NFLX", "AMD", "INTC", "QCOM", "INTU",
    "AMGN", "ISRG", "HON", "AMAT", "BKNG", "SBUX", "VRTX", "ADI", "ADP", "GILD",
    "MDLZ", "REGN", "LRCX", "PYPL", "MU", "CSX", "SNPS", "PANW", "KLAC", "ORLY",
    "CDNS", "MELI", "CTAS", "MAR", "MNST", "ASML", "FTNT", "NXPI", "AZN", "MRVL",
    "KDP", "WDAY", "KHC", "ABNB", "DXCM", "AEP", "CHTR", "ADSK", "MRNA", "PAYX",
    "CPRT", "EXC", "ODFL", "XEL", "LULU", "PCAR", "BIIB", "ROST", "SGEN", "IDXX",
    "EA", "CTSH", "FAST", "VRSK", "CSGP", "CRWD", "GEHC", "WBD", "DDOG", "ILMN",
    "BKR", "ZS", "EBAY", "ANSS", "CEG", "ALGN", "FANG", "TEAM", "DLTR", "ENPH",
    "ZM", "JD", "PDD", "LCID", "RIVN", "SIRI", "WBA", "MTCH", "OKTA", "SPLK",
]

# Sector mapping for common stocks
SECTOR_MAP: Dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "META": "Technology", "NVDA": "Technology", "AVGO": "Technology", "CSCO": "Technology",
    "ADBE": "Technology", "CRM": "Technology", "ORCL": "Technology", "INTC": "Technology",
    "AMD": "Technology", "TXN": "Technology", "QCOM": "Technology", "IBM": "Technology",
    "NOW": "Technology", "INTU": "Technology", "AMAT": "Technology", "MU": "Technology",
    
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "MCD": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary", "BKNG": "Consumer Discretionary",
    
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "MRK": "Healthcare", "ABBV": "Healthcare",
    "LLY": "Healthcare", "TMO": "Healthcare", "PFE": "Healthcare", "ABT": "Healthcare",
    "DHR": "Healthcare", "BMY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare",
    "ISRG": "Healthcare", "VRTX": "Healthcare", "REGN": "Healthcare", "SYK": "Healthcare",
    
    # Financials
    "JPM": "Financials", "V": "Financials", "MA": "Financials", "BAC": "Financials",
    "WFC": "Financials", "GS": "Financials", "MS": "Financials", "BLK": "Financials",
    "SPGI": "Financials", "AXP": "Financials", "C": "Financials", "CB": "Financials",
    
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "OXY": "Energy", "PSX": "Energy", "VLO": "Energy",
    
    # Consumer Staples
    "PG": "Consumer Staples", "PEP": "Consumer Staples", "KO": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "MDLZ": "Consumer Staples", "CL": "Consumer Staples",
    
    # Industrials
    "HON": "Industrials", "UPS": "Industrials", "UNP": "Industrials", "BA": "Industrials",
    "CAT": "Industrials", "GE": "Industrials", "DE": "Industrials", "LMT": "Industrials",
    "RTX": "Industrials", "MMM": "Industrials", "FDX": "Industrials", "NOC": "Industrials",
    
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "AEP": "Utilities",
    "XEL": "Utilities", "EXC": "Utilities", "SRE": "Utilities", "D": "Utilities",
    
    # Communication Services
    "NFLX": "Communication Services", "DIS": "Communication Services", "CMCSA": "Communication Services",
    "VZ": "Communication Services", "T": "Communication Services", "TMUS": "Communication Services",
    
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "CCI": "Real Estate", "EQIX": "Real Estate",
    "SPG": "Real Estate", "PSA": "Real Estate", "O": "Real Estate", "DLR": "Real Estate",
    
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials", "ECL": "Materials",
    "NEM": "Materials", "FCX": "Materials", "NUE": "Materials", "DOW": "Materials",
}


class UniverseLoader:
    """
    Load and manage stock universes for scanning.
    
    Features:
    - Pre-defined universes (S&P 500, NASDAQ 100)
    - Trading style universes (Scalp, Swing, Long-term)
    - Penny stocks support
    - Custom watchlist loading
    - Sector filtering
    - Deduplication
    """
    
    def __init__(self, watchlist_path: Optional[str] = None):
        """
        Initialize the universe loader.
        
        Args:
            watchlist_path: Path to custom watchlist file (one ticker per line)
        """
        self.watchlist_path = watchlist_path or self._find_watchlist()
        self._watchlist_cache: Optional[List[str]] = None
    
    # Alias for backwards compatibility
    def load_watchlist(self) -> List[str]:
        """Alias for get_watchlist() for backwards compatibility."""
        return self.get_watchlist()
    
    def _find_watchlist(self) -> Optional[str]:
        """Find watchlist file in common locations."""
        possible_paths = [
            Path("config/watchlist.txt"),
            Path("watchlist.txt"),
            Path(__file__).parent.parent / "config" / "watchlist.txt",
            Path(__file__).parent.parent / "watchlist.txt",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def get_sp500(self) -> List[str]:
        """Get S&P 500 tickers (top ~100)."""
        return SP500_TICKERS.copy()
    
    def get_nasdaq100(self) -> List[str]:
        """Get NASDAQ 100 tickers."""
        return NASDAQ100_TICKERS.copy()
    
    def get_penny_stocks(self) -> List[str]:
        """Get penny stocks (typically under $5, high volatility)."""
        return PENNY_STOCKS.copy()
    
    def get_scalp_tickers(self) -> List[str]:
        """Get high-beta momentum stocks good for scalping."""
        return SCALP_TICKERS.copy()
    
    def get_swing_tickers(self) -> List[str]:
        """Get swing trading candidates (moderate volatility, good momentum)."""
        return SWING_TICKERS.copy()
    
    def get_long_term_tickers(self) -> List[str]:
        """Get long-term investment candidates (stable growth, dividends)."""
        return LONG_TERM_TICKERS.copy()
    
    def get_by_trading_style(self, style: str) -> List[str]:
        """
        Get tickers appropriate for a trading style.
        
        Args:
            style: Trading style ('scalp', 'swing', 'position', 'long_term', 'penny')
            
        Returns:
            List of tickers for that trading style
        """
        style = style.lower().strip()
        
        if style in ('scalp', 'scalping', 'day', 'daytrading', 'day_trade'):
            return self.get_scalp_tickers()
        elif style in ('swing', 'swing_trade', 'swingtrading'):
            return self.get_swing_tickers()
        elif style in ('position', 'position_trade', 'medium_term'):
            # Position trading uses a mix of swing and long-term
            combined = set(self.get_swing_tickers() + self.get_long_term_tickers()[:30])
            return sorted(list(combined))
        elif style in ('long_term', 'longterm', 'long', 'investment', 'invest'):
            return self.get_long_term_tickers()
        elif style in ('penny', 'pennystocks', 'penny_stocks', 'small_cap'):
            return self.get_penny_stocks()
        else:
            logger.warning(f"Unknown trading style: {style}, defaulting to swing")
            return self.get_swing_tickers()
    
    def get_watchlist(self) -> List[str]:
        """
        Load custom watchlist from file.
        
        Returns:
            List of tickers from watchlist file
        """
        if self._watchlist_cache is not None:
            return self._watchlist_cache.copy()
        
        if not self.watchlist_path:
            logger.debug("No watchlist file found")
            return []
        
        try:
            path = Path(self.watchlist_path)
            if not path.exists():
                logger.warning(f"Watchlist file not found: {self.watchlist_path}")
                return []
            
            with open(path, 'r') as f:
                tickers = []
                for line in f:
                    # Remove comments and whitespace
                    line = line.split('#')[0].strip()
                    if line:
                        # Handle comma-separated tickers on same line
                        for ticker in line.replace(',', ' ').split():
                            ticker = ticker.strip().upper()
                            if ticker and ticker.isalpha() or '-' in ticker:
                                tickers.append(ticker)
            
            self._watchlist_cache = tickers
            logger.info(f"Loaded {len(tickers)} tickers from watchlist")
            return tickers.copy()
            
        except Exception as e:
            logger.error(f"Error loading watchlist: {e}")
            return []
    
    def get_universe(
        self,
        name: str,
        include_watchlist: bool = True,
    ) -> List[str]:
        """
        Get a stock universe by name.
        
        Args:
            name: Universe name ('sp500', 'nasdaq100', 'watchlist', 'all',
                  'scalp', 'swing', 'long_term', 'penny')
            include_watchlist: Whether to include watchlist tickers
            
        Returns:
            List of unique tickers
        """
        tickers: Set[str] = set()
        
        name = name.lower()
        
        if name in ('sp500', 's&p500', 's&p 500'):
            tickers.update(SP500_TICKERS)
        elif name in ('nasdaq100', 'nasdaq', 'nasdaq 100'):
            tickers.update(NASDAQ100_TICKERS)
        elif name == 'watchlist':
            tickers.update(self.get_watchlist())
        elif name == 'all':
            tickers.update(SP500_TICKERS)
            tickers.update(NASDAQ100_TICKERS)
        elif name in ('scalp', 'scalping', 'day', 'daytrading'):
            tickers.update(SCALP_TICKERS)
        elif name in ('swing', 'swing_trade', 'swingtrading'):
            tickers.update(SWING_TICKERS)
        elif name in ('long_term', 'longterm', 'investment', 'invest'):
            tickers.update(LONG_TERM_TICKERS)
        elif name in ('penny', 'pennystocks', 'penny_stocks', 'small_cap'):
            tickers.update(PENNY_STOCKS)
        elif name in ('momentum', 'momo', 'high_beta'):
            # High beta momentum stocks
            tickers.update(SCALP_TICKERS[:30])
            tickers.update(PENNY_STOCKS[:20])
        else:
            logger.warning(f"Unknown universe: {name}, defaulting to S&P 500")
            tickers.update(SP500_TICKERS)
        
        # Add watchlist if requested, but avoid contaminating style-specific
        # universes (penny, scalp, swing, long_term, momentum) which should
        # remain "pure" and not automatically pull in large caps from the
        # user's personal watchlist.
        if include_watchlist and name not in (
            'watchlist',
            'penny',
            'scalp', 'scalping', 'day', 'daytrading',
            'swing', 'swing_trade', 'swingtrading',
            'long_term', 'longterm', 'investment', 'invest',
            'momentum', 'momo', 'high_beta',
        ):
            tickers.update(self.get_watchlist())
        
        return sorted(list(tickers))
    
    def get_sector(self, ticker: str) -> str:
        """
        Get the sector for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Sector name or 'Unknown'
        """
        return SECTOR_MAP.get(ticker.upper(), "Unknown")
    
    def get_tickers_by_sector(self, sector: str) -> List[str]:
        """
        Get all tickers in a specific sector.
        
        Args:
            sector: Sector name
            
        Returns:
            List of tickers in that sector
        """
        return [
            ticker for ticker, s in SECTOR_MAP.items()
            if s.lower() == sector.lower()
        ]
    
    def get_all_sectors(self) -> List[str]:
        """Get list of all available sectors."""
        return sorted(list(set(SECTOR_MAP.values())))
    
    def filter_by_sector(
        self,
        tickers: List[str],
        sectors: List[str],
    ) -> List[str]:
        """
        Filter tickers by sector.
        
        Args:
            tickers: List of tickers to filter
            sectors: List of sectors to include
            
        Returns:
            Filtered list of tickers
        """
        sectors_lower = [s.lower() for s in sectors]
        return [
            t for t in tickers
            if self.get_sector(t).lower() in sectors_lower
        ]
    
    def reload_watchlist(self) -> List[str]:
        """Force reload of watchlist from file."""
        self._watchlist_cache = None
        return self.get_watchlist()
    
    def add_to_watchlist(self, ticker: str) -> bool:
        """
        Add a ticker to the watchlist file.
        
        Args:
            ticker: Ticker to add
            
        Returns:
            True if successful
        """
        if not self.watchlist_path:
            logger.error("No watchlist file configured")
            return False
        
        try:
            ticker = ticker.upper().strip()
            
            # Check if already in watchlist
            current = self.get_watchlist()
            if ticker in current:
                logger.debug(f"{ticker} already in watchlist")
                return True
            
            # Append to file
            with open(self.watchlist_path, 'a') as f:
                f.write(f"\n{ticker}")
            
            # Update cache
            if self._watchlist_cache is not None:
                self._watchlist_cache.append(ticker)
            
            logger.info(f"Added {ticker} to watchlist")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to watchlist: {e}")
            return False
    
    def remove_from_watchlist(self, ticker: str) -> bool:
        """
        Remove a ticker from the watchlist file.
        
        Args:
            ticker: Ticker to remove
            
        Returns:
            True if successful
        """
        if not self.watchlist_path:
            logger.error("No watchlist file configured")
            return False
        
        try:
            ticker = ticker.upper().strip()
            
            # Read current watchlist
            current = self.get_watchlist()
            if ticker not in current:
                logger.debug(f"{ticker} not in watchlist")
                return True
            
            # Remove and rewrite
            current.remove(ticker)
            
            with open(self.watchlist_path, 'w') as f:
                f.write('\n'.join(current))
            
            # Update cache
            self._watchlist_cache = current
            
            logger.info(f"Removed {ticker} from watchlist")
            return True
            
        except Exception as e:
            logger.error(f"Error removing from watchlist: {e}")
            return False


def get_universe_loader() -> UniverseLoader:
    """Get a configured UniverseLoader instance."""
    return UniverseLoader()
