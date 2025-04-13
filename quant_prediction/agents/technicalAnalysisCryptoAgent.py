# technical_analysis_crypto_agent.py
import datetime
import os
import json
from typing import Dict, Any, List, Optional
import autogen
from adapters.tradingView import TradingViewAdapter
from adapters.price import run as get_yahoo_finance_price_data  # Import the new function


class TechnicalAnalysisCryptoAgent:
    def __init__(self, name: str = "TechnicalAnalyst", config_list: Optional[List[Dict]] = None):
        """
        Initializes the Technical Analysis Crypto Agent.
        """
        self.name = name
        self._validate_config(config_list)
        self.llm_config = self._create_llm_config(config_list)

        system_message = """You are a cryptocurrency technical analysis expert AI agent. Your analysis is based on technical indicators, chart patterns, trading volume, and price data from Yahoo Finance to provide investment recommendations.

                    When analyzing a cryptocurrency, consider the following technical factors:
                    1. Price Action: Analyze recent price trends, support and resistance levels, and chart patterns (e.g., triangles, head and shoulders).
                    2. Technical Indicators: Utilize common indicators such as Moving Averages (MA), Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), and Volume.
                    3. Trading Volume: Assess the volume of trades to confirm price movements and identify potential breakouts or breakdowns.
                    4. Price Data from Yahoo Finance: Analyze historical price data, including open, close, high, low prices and volume, fetched using the 'get_crypto_yahoo_price_data' tool. Focus on recent price action (last 3 days of data is available).
                    5. Market Sentiment (Technical Perspective): Gauge the overall market sentiment based on technical indicators and price action, not fundamental news unless it directly impacts price.
                    6. Risk Management: Consider volatility and potential risk levels based on technical analysis.

                    When providing analysis and recommendation:
                    1. Clearly state your recommendation: BULLISH, BEARISH, or NEUTRAL.
                    2. Provide a Confidence Level (0-100%).
                    3. Deliver Thorough Reasoning:
                        a. Explicitly mention the technical indicators, chart patterns, and price data you analyzed.
                        b. Explain how these factors lead to your conclusion.
                        c. Quantify your analysis where possible (e.g., "RSI is currently at 70, indicating overbought conditions", "Current price is $X, a 10% increase in the last 24 hours based on Yahoo Finance data"). If data isn't available, state that.
                        d. Focus on the technical aspects and avoid fundamental news unless it directly impacts technical indicators or price action.
                        e. When referring to price data, explicitly state that it's from Yahoo Finance and mention the date range if relevant.
                        f. Use a technical analyst's voice, focusing on data-driven observations and interpretations.

                    You have access to the tools 'get_crypto_technical_news' and 'get_crypto_yahoo_price_data'. Use them to inform your analysis."""

        self.agent = autogen.AssistantAgent(
            name=self.name,
            llm_config=self.llm_config,
            system_message=system_message,
        )

        self.agent.register_function(
            function_map={
                "get_crypto_technical_news": self._get_crypto_technical_news_wrapper,
                "get_crypto_yahoo_price_data": self._get_crypto_yahoo_price_data_wrapper
                # New function for Yahoo Finance price data
            }
        )

    def _validate_config(self, config_list):
        """Basic validation for essential configurations."""
        if not config_list:
            raise ValueError("LLM configuration 'config_list' is required.")

    def _create_llm_config(self, config_list):
        """Creates the LLM config dictionary with function calling."""
        return {
            "config_list": config_list,
            "cache_seed": 42,
            "timeout": 120,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_crypto_technical_news",
                        "description": "Fetches technical market news for a cryptocurrency from TradingView.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "symbol": {
                                    "type": "string",
                                    "description": "The cryptocurrency symbol and exchange (e.g., 'COINBASE:SOLUSD', 'BINANCE:BTCUSDT')."
                                }
                            },
                            "required": ["symbol"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_crypto_yahoo_price_data",  # New function for Yahoo Finance price data
                        "description": "Fetches historical price data for a cryptocurrency from Yahoo Finance for the last 3 days.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "symbol": {
                                    "type": "string",
                                    "description": "The cryptocurrency symbol (e.g., 'SOL-USD', 'BTC-USD'). Must be in Yahoo Finance format."
                                }
                            },
                            "required": ["symbol"]
                        }
                    }
                }
            ]
        }

    def _get_crypto_technical_news_wrapper(self, symbol: str) -> str:
        """
        Wrapper function to fetch technical news using TradingViewAdapter.
        """
        print(f"{self.name}: Fetching technical news for symbol: '{symbol}'")
        try:
            adapter = TradingViewAdapter(symbol=symbol)
            news_articles = adapter.get_news()

            if news_articles:
                formatted_news = []
                for article in news_articles:
                    formatted_news.append({
                        "title": article.get("title", "N/A"),
                        "summary": article.get("summary", "N/A"),
                        "source": article.get("source", "N/A"),
                        "url": article.get("url", "N/A"),
                        "published_at": article.get("published_at", "N/A")
                    })
                print("get_crypto_technical_news triggered!, returning news articles.")
                MAX_ARTICLES_FOR_LLM = 5
                return json.dumps(formatted_news[:MAX_ARTICLES_FOR_LLM])
            elif news_articles == []:
                print(f"{self.name}: No technical news found for '{symbol}'.")
                return json.dumps({"status": "success", "message": f"No news found for '{symbol}'."})
            else:
                print(f"{self.name}: Failed to fetch technical news for '{symbol}'.")
                return json.dumps({"status": "error",
                                   "message": f"Could not retrieve news for '{symbol}' due to an upstream error."})

        except Exception as e:
            print(f"{self.name}: Error in _get_crypto_technical_news_wrapper for symbol '{symbol}': {e}")
            return json.dumps(
                {"status": "error", "message": f"An internal error occurred while trying to fetch news: {str(e)}"})

    def _get_crypto_yahoo_price_data_wrapper(self, symbol: str) -> str:
        """
        Wrapper function to fetch price data from Yahoo Finance.
        """
        print(f"{self.name}: Fetching price data from Yahoo Finance for symbol: '{symbol}'")
        try:
            price_data = get_yahoo_finance_price_data(symbol)
            if price_data:
                print("get_crypto_yahoo_price_data triggered!, returning price data.")
                return json.dumps(price_data)  # Return the parsed JSON data
            else:
                print(f"{self.name}: Failed to fetch or parse price data from Yahoo Finance for '{symbol}'.")
                return json.dumps(
                    {"status": "error", "message": f"Could not retrieve price data for '{symbol}' from Yahoo Finance."})

        except Exception as e:
            print(f"{self.name}: Error in _get_crypto_yahoo_price_data_wrapper for symbol '{symbol}': {e}")
            return json.dumps(
                {"status": "error",
                 "message": f"An internal error occurred while trying to fetch price data from Yahoo Finance: {str(e)}"})


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "UoVIqe36pIbC7t6NSAXXWEVN81aknXPpkF056GUVy5G5DePZ"
    os.environ["OPENAI_BASE_URL"] = "https://ai.tzpro.xyz/v1"
    agent = TechnicalAnalysisCryptoAgent(config_list= [
        {
            'model': 'gemini-2.5-pro-exp-03-25',  # Or your preferred model
            'api_key': os.environ["OPENAI_API_KEY"],
            'base_url': os.environ["OPENAI_BASE_URL"],
        }
    ]
    )
