import datetime
import os
import json
from typing import Dict, Any, List, Optional
import autogen
from adapters.google import GoogleNewsAdapter # 假设你已经创建了 google_adapter.py 文件


class CryptoSentimentNewsAgent:
    def __init__(self, name: str = "CryptoSentimentAnalyst", config_list: Optional[List[Dict]] = None,
                 news_api_key: Optional[str] = None, news_search_engine_id: Optional[str] = None):
        """
        Initializes the Crypto Sentiment News Agent.
        """
        self.name = name
        self._validate_config(config_list, news_api_key, news_search_engine_id)

        self.news_api_key = news_api_key
        self.news_search_engine_id = news_search_engine_id

        self.news_adapter = GoogleNewsAdapter(
            api_key=self.news_api_key,
            search_engine_id=self.news_search_engine_id,
            raise_errors=False,
            query=f"crypto after:{(datetime.datetime.now() - datetime.timedelta(days=3)).strftime('%Y-%m-%d')}" # Default query for general crypto news
        )

        self.llm_config = self._create_llm_config(config_list)

        system_message = """You are a cryptocurrency sentiment analysis expert AI agent. Your investment recommendations are based on analyzing news sentiment surrounding cryptocurrencies.

                    When analyzing a cryptocurrency or the crypto market in general, consider the following:
                    1. News Sentiment: Analyze recent news headlines and snippets to gauge the overall sentiment (positive, negative, neutral) regarding the cryptocurrency or the market.
                    2. Market Trends (News Driven): Identify trends in the news that could impact price movements (e.g., regulatory news, adoption news, technology updates).
                    3. Fear and Greed Index (Optional): While primarily news-driven, also consider the general market sentiment reflected in fear and greed indices if available.
                    4. Risk Assessment (Sentiment Based): Assess the potential risks and opportunities based on the prevailing news sentiment.

                    When providing analysis and recommendation:
                    1. Clearly state your recommendation: BULLISH, BEARISH, or NEUTRAL.
                    2. Provide a Confidence Level (0-100%).
                    3. Deliver Thorough Reasoning:
                        a. Explain the overall news sentiment you identified (e.g., "Predominantly positive news sentiment due to recent adoption announcements").
                        b. Summarize key news headlines and snippets that support your sentiment analysis.
                        c. Explain how this sentiment leads to your investment recommendation.
                        d. If news is mixed or inconclusive, state that and adjust your recommendation and confidence accordingly.
                        e. Use a sentiment analyst's voice, focusing on interpreting news and its potential market impact.

                    You have access to the tool 'get_crypto_news_sentiment' to fetch recent crypto news headlines. Use it to inform your sentiment analysis."""

        self.agent = autogen.AssistantAgent(
            name=self.name,
            llm_config=self.llm_config,
            system_message=system_message,
        )

        self.agent.register_function(
            function_map={
                "get_crypto_news_sentiment": self._get_crypto_news_sentiment_wrapper
            }
        )

    def _validate_config(self, config_list, key, cx):
        """Basic validation for essential configurations."""
        if not config_list:
            raise ValueError("LLM configuration 'config_list' is required.")
        if not key:
            raise ValueError("Google News API key 'news_api_key' is required.")
        if not cx:
            raise ValueError("Google News Search Engine ID 'news_search_engine_id' is required.")

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
                        "name": "get_crypto_news_sentiment",
                        "description": "Fetches the latest news headlines and snippets about cryptocurrencies or a specific crypto symbol from Google News.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query for crypto news (e.g., 'Bitcoin', 'Ethereum', 'crypto regulation', 'Solana news'). Be specific."
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
        }

    def _get_crypto_news_sentiment_wrapper(self, query: str) -> str:
        """
        Wrapper function to fetch crypto news using GoogleNewsAdapter.
        """
        print(f"{self.name}: Fetching news for sentiment analysis with query: '{query}'")
        try:
            # Re-initialize adapter with the specific query
            self.news_adapter.query = query  # Update the query for the news adapter
            news_articles = self.news_adapter.get_news()

            if news_articles:
                formatted_news = []
                for article in news_articles:
                    formatted_news.append({
                        "title": article.get("title", "N/A"),
                        "snippet": article.get("snippet", "N/A"),
                    })
                print("get_crypto_news_sentiment triggered!, returning news articles.")
                MAX_ARTICLES_FOR_LLM = 5
                return json.dumps(formatted_news[:MAX_ARTICLES_FOR_LLM])
            elif news_articles == []:
                print(f"{self.name}: No news articles found for '{query}'.")
                return json.dumps({"status": "success", "message": f"No news articles found for '{query}'."})
            else:
                print(f"{self.name}: Failed to fetch news articles for '{query}'.")
                return json.dumps({"status": "error",
                                   "message": f"Could not retrieve news articles for '{query}' due to an upstream error."})

        except Exception as e:
            print(f"{self.name}: Error in _get_crypto_news_sentiment_wrapper for query '{query}': {e}")
            return json.dumps(
                {"status": "error", "message": f"An internal error occurred while trying to fetch news: {str(e)}"})


# --- Example Usage ---
async def main():
    # --- Configuration ---
    from dotenv import load_dotenv
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_search_engine_id = os.getenv("GOOGLE_CSE_ID")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")

    if not all([google_api_key, google_search_engine_id, openai_api_key]):
        print("ERROR: Ensure GOOGLE_API_KEY, GOOGLE_CSE_ID, and OPENAI_API_KEY are set (e.g., in a .env file).")
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
        sentiment_agent_instance = CryptoSentimentNewsAgent(
            config_list=config_list_openai,
            news_api_key=google_api_key,
            news_search_engine_id=google_search_engine_id
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
    crypto_query = "Solana crypto news" # Specific crypto news query
    # crypto_query = "cryptocurrency market regulation" # Broader crypto market topic

    print(f"\n--- Requesting Sentiment Analysis for: {crypto_query} ---")

    user_proxy.initiate_chat(
        recipient=sentiment_agent_instance.agent,
        message=f"Provide a sentiment-based investment recommendation based on news for '{crypto_query}'. "
                f"Use the 'get_crypto_news_sentiment' tool to fetch news. "
                f"Conclude your response with TERMINATE.",
        clear_history=True
    )

    print(f"\n--- Sentiment Analysis Complete for {crypto_query} ---")


if __name__ == "__main__":
    import os
    os.environ["GOOGLE_CSE_ID"] = "838203a83d5f34895"
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBejU9dqhGbgx7n8WXzhIIzmoiZ0iSAo5E"
    os.environ["OPENAI_API_KEY"] = "UoVIqe36pIbC7t6NSAXXWEVN81aknXPpkF056GUVy5G5DePZ"
    os.environ["OPENAI_BASE_URL"] = "https://ai.tzpro.xyz/v1"
    asyncio.run(main())