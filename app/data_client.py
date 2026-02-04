import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical import NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from .config import settings

class AlpacaClient:
    def __init__(self):
        self.data_client = StockHistoricalDataClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY
        )
        self.news_client = NewsClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY
        )

    def get_historical_ohlcv(self, symbol: str, days: int = 1260):
        """Fetches historical OHLCV data for a given symbol."""
        start_date = datetime.now() - timedelta(days=days)
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start_date
        )
        bars = self.data_client.get_stock_bars(request_params)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol)
        return df

    def get_latest_news(self, symbol: str, limit: int = 5):
        """
        Fetches the latest news articles for a given symbol using the proper NewsClient.
        """
        try:
            # NewsRequest symbols parameter expects a string or comma-separated string, not a list in recent versions
            request_params = NewsRequest(
                symbols=symbol,
                limit=limit
            )
            response = self.news_client.get_news(request_params)
            
            # Handle different Alpaca-py response formats
            # Format 1: Object with .news attribute (list)
            if hasattr(response, 'news') and isinstance(response.news, list):
                return response.news
            
            # Format 2: NewsSet (v2) which is a mapping {symbol: [News, ...]}
            # Check for items() and that it's callable (dict-like behavior)
            if hasattr(response, 'items') and callable(getattr(response, 'items', None)):
                all_news = []
                for _, news_list in response.items():
                    if isinstance(news_list, list):
                        all_news.extend(news_list)
                    else:
                        all_news.append(news_list)
                return all_news
            
            # Convert to list for further inspection
            news_list = list(response) if not isinstance(response, list) else response
            
            # Format 3: List of tuples [(symbol, Article), ...] - extract articles
            if news_list and len(news_list) > 0:
                first_item = news_list[0]
                if isinstance(first_item, tuple) and len(first_item) == 2:
                    # It's a list of (symbol, article) tuples, extract the articles
                    return [item[1] for item in news_list]
                    
            # Format 4: Already a flat list of articles
            return news_list
        except Exception as e:
            print(f"⚠️ Error fetching news from Alpaca: {e}")
            return []

    def get_fundamentals(self, symbol: str):
        """
        Placeholder for fundamentals data.
        """
        return {
            "symbol": symbol,
            "note": "Fundamentals data fetching implementation pending."
        }
