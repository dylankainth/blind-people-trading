import asyncio
import datetime
import os
import json
from typing import Dict, Any, List, Optional
import autogen
from adapters.tradingView import TradingViewAdapter # 假设你已经创建了 tradingview_adapter.py 文件

class TechnicalAnalysisCryptoAgent:
    def __init__(self, name: str = "TechnicalAnalyst", config_list: Optional[List[Dict]] = None):
        """
        Initializes the Technical Analysis Crypto Agent.
        """
        self.name = name
        self._validate_config(config_list)
        self.llm_config = self._create_llm_config(config_list)

        system_message = """You are a cryptocurrency technical analysis expert AI agent. Your analysis is based on technical indicators, chart patterns, and trading volume to provide investment recommendations.

                    When analyzing a cryptocurrency, consider the following technical factors:
                    1. Price Action: Analyze recent price trends, support and resistance levels, and chart patterns (e.g., triangles, head and shoulders).
                    2. Technical Indicators: Utilize common indicators such as Moving Averages (MA), Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), and Volume.
                    3. Trading Volume: Assess the volume of trades to confirm price movements and identify potential breakouts or breakdowns.
                    4. Market Sentiment (Technical Perspective): Gauge the overall market sentiment based on technical indicators and price action, not fundamental news.
                    5. Risk Management: Consider volatility and potential risk levels based on technical analysis.

                    When providing analysis and recommendation:
                    1. Clearly state your recommendation: BULLISH, BEARISH, or NEUTRAL.
                    2. Provide a Confidence Level (0-100%).
                    3. Deliver Thorough Reasoning:
                        a. Explicitly mention the technical indicators and chart patterns you analyzed.
                        b. Explain how these indicators and patterns lead to your conclusion.
                        c. Quantify your analysis where possible (e.g., "RSI is currently at 70, indicating overbought conditions"). If data isn't available, state that.
                        d. Focus on the technical aspects and avoid fundamental news unless it directly impacts technical indicators (e.g., news of a major exchange listing causing a price pump).
                        e. Use a technical analyst's voice, focusing on data-driven observations and interpretations.

                    You have access to the tool 'get_crypto_technical_data' to fetch technical market data from TradingView. Use it to inform your analysis."""

        self.agent = autogen.AssistantAgent(
            name=self.name,
            llm_config=self.llm_config,
            system_message=system_message,
        )

        self.agent.register_function(
            function_map={
                "get_crypto_technical_data": self._get_crypto_technical_data_wrapper
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
                        "name": "get_crypto_technical_data",
                        "description": "Fetches technical market data for a cryptocurrency from TradingView, including news headlines related to the symbol.",
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
                }
            ]
        }

    def _get_crypto_technical_data_wrapper(self, symbol: str) -> str:
        """
        Wrapper function to fetch technical data using TradingViewAdapter.
        """
        print(f"{self.name}: Fetching technical data for symbol: '{symbol}'")
        try:
            adapter = TradingViewAdapter(symbol=symbol)
            news_articles = adapter.get_news() # TradingView adapter can fetch news too.

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
                print("get_crypto_technical_data triggered!, returning news articles.")
                MAX_ARTICLES_FOR_LLM = 5
                return json.dumps(formatted_news[:MAX_ARTICLES_FOR_LLM])
            elif news_articles == []:
                print(f"{self.name}: No technical data or news found for '{symbol}'.")
                return json.dumps({"status": "success", "message": f"No data found for '{symbol}'."})
            else:
                print(f"{self.name}: Failed to fetch technical data for '{symbol}'.")
                return json.dumps({"status": "error",
                                   "message": f"Could not retrieve data for '{symbol}' due to an upstream error."})

        except Exception as e:
            print(f"{self.name}: Error in _get_crypto_technical_data_wrapper for symbol '{symbol}': {e}")
            return json.dumps(
                {"status": "error", "message": f"An internal error occurred while trying to fetch data: {str(e)}"})


# --- Example Usage ---
async def main():
    # --- Configuration ---
    from dotenv import load_dotenv
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")

    if not openai_api_key:
        print("ERROR: Ensure OPENAI_API_KEY is set (e.g., in a .env file).")
        return

    config_list_openai = [
        {
            'model': 'gemini-2.5-pro-exp-03-25',  # Or your preferred model
            'api_key': openai_api_key,
            'base_url': openai_base_url,
        }
    ]

    # --- Agent Setup ---
    try:
        tech_agent_instance = TechnicalAnalysisCryptoAgent(
            config_list=config_list_openai,
        )
    except ValueError as e:
        print(f"Error initializing agent: {e}")
        return

    # UserProxyAgent
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,
    )

    # --- Run Analysis ---
    crypto_symbol = "COINBASE:SOLUSD"  # Solana on Coinbase
    # crypto_symbol = "BINANCE:BTCUSDT" # Bitcoin on Binance

    print(f"\n--- Requesting Technical Analysis for: {crypto_symbol} ---")

    user_proxy.initiate_chat(
        recipient=tech_agent_instance.agent,
        message=f"Provide a technical analysis-based investment recommendation for {crypto_symbol}. "
                f"Use the 'get_crypto_technical_data' tool to get market data. "
                f"Conclude your response with TERMINATE.",
        clear_history=True
    )

    print(f"\n--- Technical Analysis Complete for {crypto_symbol} ---")


if __name__ == "__main__":
    import os
    os.environ["OPENAI_API_KEY"] = "UoVIqe36pIbC7t6NSAXXWEVN81aknXPpkF056GUVy5G5DePZ"
    os.environ["OPENAI_BASE_URL"] = "https://ai.tzpro.xyz/v1"
    asyncio.run(main())