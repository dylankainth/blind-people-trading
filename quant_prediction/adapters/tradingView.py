# tradingview_adapter.py
import requests
from typing import List, Dict, Optional


class TradingViewAdapter:
    """
    Adapter for fetching news data from TradingView for specific symbols.
    """

    def __init__(self, symbol: str, lang: str = "en", client: str = "landing", streaming: bool = True):
        self.base_url = "https://news-mediator.tradingview.com/public/view/v1/symbol"
        self.symbol = symbol
        self.lang = lang
        self.client = client
        self.streaming = streaming
        self.data = None

    def fetch_data(self) -> None:
        """
        Fetch news data from TradingView API.
        """
        params = [
            ("filter", f"lang:{self.lang}"),
            ("filter", f"symbol:{self.symbol}"),
            ("client", self.client),
            ("streaming", str(self.streaming).lower())
        ]
        try:
            response = requests.get(self.base_url, params=params, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"})
            response.raise_for_status()
            json_response = response.json()
            self.data = json_response.get('items', [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from TradingView: {e}")
            self.data = []

    def parse_data(self) -> List[Dict]:
        """
        Parse the fetched data into structured news items.
        """
        if not self.data:
            print("No data available to parse.")
            return []

        parsed_news = []
        for item in self.data:
            news_item = {
                "id": item.get("id"),
                "title": item.get("title"),
                "published_at": item.get("published"),
                "source": item.get("provider", {}).get("name"),
                "url": item.get("link"),
                "summary": item.get("summary", ""),
                "urgency": item.get("urgency"),
                "relatedSymbols": [symbol.get("symbol") for symbol in item.get("relatedSymbols", [])]
            }
            parsed_news.append(news_item)

        return parsed_news

    def get_news(self) -> List[Dict]:
        """
        Fetch and return parsed news articles for the symbol.
        """
        self.fetch_data()
        return self.parse_data()

    def get_analysis(self,
                     ticker):  # This function is not used in the current example, but kept for potential future use.
        """
        Fetch and return analysis data for the given ticker.
        """
        self.fetch_data()
        analysis = []
        for item in self.data:
            if item.get("symbol") == ticker:
                analysis.append({
                    "title": item.get("title"),
                    "analysis": item.get("analysis"),
                    # Note: TradingView news items might not always have a distinct 'analysis' field.
                    "timestamp": item.get("timestamp")
                })
        return analysis


if __name__ == "__main__":
    adapter = TradingViewAdapter(symbol="COINBASE:SOLUSD")
    news = adapter.get_news()
    print(news)
