import requests
from typing import Any, List, Dict

class GoogleNewsAdapter:
    """
    Fetch the latest news articles using Google's Custom Search API.
    """

    def __init__(
            self,
            query: str,
            api_key: str,
            search_engine_id: str,
            max_results: int = 8,
            session: Any = None,
            proxy: Any = None,
            timeout: int = 30,
            raise_errors: bool = True
    ) -> None:
        self.query = query
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.params = {
            "q": query,
            "key": api_key,
            "cx": search_engine_id,
            "num": max_results
        }
        self.session = session or requests.Session()
        self.proxy = proxy
        self.timeout = timeout
        self.raise_errors = raise_errors
        self.data = None

    def fetch_data(self) -> None:
        """
        Fetch the latest news data from Google's Custom Search API.
        """
        url = "https://www.googleapis.com/customsearch/v1"
        try:
            response = self.session.get(url, params=self.params, proxies=self.proxy, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            self.data = result.get('items', [])
        except requests.exceptions.RequestException as e:
            if self.raise_errors:
                raise e
            print(f"Error fetching data from Google Custom Search API: {e}")
            self.data = None
        except Exception as e:
            if self.raise_errors:
                raise e
            print(f"An unexpected error occurred: {e}")
            self.data = None

    def parse_data(self) -> List[Dict]:
        """
        Parse the fetched news data into a structured format.
        """
        print(f"Parsing data from Google Custom Search API: {self.query}")
        if not self.data:
            print("No data to parse.")
            return []

        parsed_data = []
        for article in self.data:
            parsed_article = {
                "title": article.get("title"),
                "link": article.get("link"),
                "snippet": article.get("snippet"),
                "display_link": article.get("displayLink"),
                "pagemap": article.get("pagemap")
            }
            parsed_data.append(parsed_article)

        return parsed_data

    def get_news(self) -> List[Dict]:
        """
        Get the latest news articles for the specified query.
        """
        self.fetch_data()
        return self.parse_data()


if __name__ == "__main__":
    # Replace 'YOUR_API_KEY' and 'YOUR_SEARCH_ENGINE_ID' with your actual credentials
    adapter = GoogleNewsAdapter("SOL-USD", api_key='AIzaSyBejU9dqhGbgx7n8WXzhIIzmoiZ0iSAo5E', search_engine_id='838203a83d5f34895')
    news = adapter.get_news()
    print(news)
