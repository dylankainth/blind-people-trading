import datetime
import httpx
from fastapi import FastAPI, Query, HTTPException
from dotenv import load_dotenv


# --- Configuration ---
# Valid ranges as specified
VALID_RANGES = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
# Yahoo Finance base URL
YAHOO_FINANCE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
# Headers (consider managing cookies more dynamically if needed)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
    'Cache-Control': 'no-cache',
    'Origin': 'https://uk.finance.yahoo.com',
    'Pragma': 'no-cache',
    'Referer': 'https://uk.finance.yahoo.com/',  # Simplified Referer
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
}


# --- Data Parsing Function ---
def parse_to_time_based_json(data):
    """
    Parse the Yahoo Finance data to a time-based JSON format.
    :param data: The data returned from Yahoo Finance API.
    :return: A dictionary containing the parsed data, or None if parsing fails.
    """
    try:
        if not data or "chart" not in data or not data["chart"]["result"]:
            print("Warning: Empty data or invalid format received from Yahoo Finance.")
            return None  # Return None for empty/invalid structure before accessing keys

        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp")
        quotes_data = result.get("indicators", {}).get("quote", [])

        # Handle cases where essential data might be missing
        if not timestamps or not quotes_data:
            print("Warning: Timestamps or quote data missing in Yahoo Finance response.")
            return None

        # Ensure quotes_data is a list and take the first element
        if isinstance(quotes_data, list) and len(quotes_data) > 0:
            quotes = quotes_data[0]
        else:
            print("Warning: Quote data is not in the expected format.")
            return None

        keys = ['open', 'close', 'high', 'low', 'volume']
        parsed_data = []

        for i, timestamp in enumerate(timestamps):
            # Check for None timestamp before processing
            if timestamp is None:
                print(f"Warning: Skipping null timestamp at index {i}")
                continue

            try:
                # Use UTC for consistency if timezone info isn't critical for display
                readable_time = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).strftime(
                    '%Y-%m-%d %H:%M:%S %Z')
                # Or keep original format if preferred:
                # readable_time = datetime.datetime.fromtimestamp(timestamp).strftime('%m/%d %I:%M %p')
            except (OSError, TypeError) as time_err:
                print(f"Warning: Skipping invalid timestamp {timestamp} at index {i}. Error: {time_err}")
                continue

            point_data = {"TimestampUTC": readable_time}  # Using a more descriptive key
            valid_point = True
            for key in keys:
                if key in quotes and quotes[key] is not None and i < len(quotes[key]) and quotes[key][i] is not None:
                    point_data[key.capitalize()] = quotes[key][i]
                else:
                    # Decide how to handle missing data points within a timestamp
                    # Option 1: Include as None (as originally)
                    point_data[key.capitalize()] = None
                    # Option 2: Skip the entire timestamp if critical data is missing (e.g., close)
                    # if key == 'close':
                    #     valid_point = False
                    #     break
                    # Option 3: Log a warning but still include None
                    # print(f"Warning: Missing value for '{key}' at timestamp {readable_time}")

            if valid_point:
                parsed_data.append(point_data)

        # Return None if no valid data points were parsed
        if not parsed_data:
            print("Warning: No valid data points could be parsed from the response.")
            return None

        final_json = {
            "chart": {
                "result": [{
                    "meta": result.get("meta", {}),  # Include metadata
                    "parsed_quotes": parsed_data  # Use a more descriptive key
                }],
                "error": data["chart"].get("error")  # Include error info from Yahoo if present
            }
        }
        return final_json

    except (KeyError, IndexError, TypeError) as e:
        print(f"Error parsing Yahoo Finance data: {e}")
        # Optionally log the raw data here for debugging
        # print("Raw data causing parsing error:", data)
        return None  # Indicate parsing failure

# --- Import Adapters (Ensure these files exist in an 'adapters' directory) ---
try:
    from adapters.google import GoogleNewsAdapter
    from adapters.tradingView import TradingViewAdapter
    from adapters.price import run as get_yahoo_finance_price_data  # Import the specific function
except ImportError as e:
    print(f"Error importing adapters: {e}. Make sure 'adapters' directory and files exist.")
    # Depending on your setup, you might want to exit or handle this differently
    exit(1)

# --- Import Agent Classes (Ensure these files exist in an 'agents' directory) ---
from agents.technicalAnalysisCryptoAgent import TechnicalAnalysisCryptoAgent
from agents.cryptoSentimentNewsAgent import CryptoSentimentNewsAgent

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

# Autogen LLM Configuration
# main.py
import autogen
import asyncio
import os
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse  # Added JSONResponse
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Union

# --- Import Adapters (Ensure these files exist in an 'adapters' directory) ---
try:
    from adapters.google import GoogleNewsAdapter
    from adapters.tradingView import TradingViewAdapter
    # Import the specific function for Yahoo Finance
    from adapters.price import run as get_yahoo_finance_price_data
except ImportError as e:
    print(f"Error importing adapters: {e}. Make sure 'adapters' directory and files exist.")
    # Depending on your setup, you might want to exit or handle this differently
    exit(1)

# --- Import Agent Classes (Ensure these files exist in an 'agents' directory) ---
try:
    from agents.technicalAnalysisCryptoAgent import TechnicalAnalysisCryptoAgent
    from agents.cryptoSentimentNewsAgent import CryptoSentimentNewsAgent
except ImportError as e:
    print(f"Error importing agents: {e}. Make sure 'agents' directory and files exist.")
    # Depending on your setup, you might want to exit or handle this differently
    exit(1)

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

# Autogen LLM Configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")  # Default if not set
google_api_key = os.getenv("GOOGLE_NEWS_API_KEY")
google_search_engine_id = os.getenv("GOOGLE_NEWS_SEARCH_ENGINE_ID")

# Validate essential keys
if not openai_api_key:
    raise ValueError("Missing environment variable: OPENAI_API_KEY")
if not google_api_key:
    raise ValueError("Missing environment variable: GOOGLE_API_KEY for Sentiment Agent")
if not google_search_engine_id:
    raise ValueError("Missing environment variable: GOOGLE_CSE_ID for Sentiment Agent")

config_list_openai = [
    {
        'model': os.getenv("OPENAI_MODEL", "gpt-4o"),  # Use env var or default
        'api_key': openai_api_key,
        'base_url': openai_base_url,
    }
]

# --- FastAPI Setup ---
app = FastAPI()
# Queue to send messages from Autogen to WebSocket
message_queue = asyncio.Queue()


# --- Autogen Message Interception Callback (WITH DEBUG PRINT) ---
def get_websocket_message_callback(queue: asyncio.Queue):
    """Creates an ASYNC callback function that puts messages onto the provided asyncio Queue."""

    async def websocket_message_callback(
            recipient: autogen.Agent,
            messages: Optional[List[Dict]] = None,
            sender: Optional[autogen.Agent] = None,
            config: Optional[Any] = None,
    ) -> tuple[bool, Optional[Dict]]:
        """
        ASYNC Callback implementation. Puts sender, recipient, and the latest message onto the queue.
        Must return (bool, Optional[Dict]) for use with a_initiate_chat.
        """
        # --- DEBUG PRINT ---
        sender_name_debug = sender.name if sender else 'None'
        recipient_name_debug = recipient.name if recipient else 'None'
        print(
            f"DEBUG: websocket_message_callback called by sender: {sender_name_debug}, recipient: {recipient_name_debug}")
        # --- END DEBUG PRINT ---

        if messages is None or sender is None:
            print("DEBUG: Callback exited early (no messages or sender)")
            return False, None

        last_message = messages[-1]
        sender_name = sender.name if sender else "System"
        recipient_name = recipient.name if recipient else "System"

        message_data = {
            "sender": sender_name,
            "recipient": recipient_name,
            "content": last_message.get("content", ""),
            "role": last_message.get("role", ""),
        }
        # Add more detail for debugging message content if needed
        # print(f"DEBUG: Callback preparing to queue message: {json.dumps(message_data, indent=2)}")

        try:
            queue.put_nowait(message_data)
            # print(f"DEBUG: Message from {sender_name} queued successfully.") # Can be noisy
        except asyncio.QueueFull:
            print("Error: Message queue is full. Dropping message.")
        except Exception as e:
            print(f"Error in websocket_message_callback putting to queue: {e}")

        # Return False, None because this callback is just for listening, not replying
        return False, None

    return websocket_message_callback


# --- FastAPI Endpoints ---

# Basic HTML page for WebSocket testing (Unchanged from previous working version)
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Autogen Crypto Advisor</title>
    </head>
    <body>
        <h1>Autogen Crypto Advisor Chat</h1>
        <button onclick="startChat()">Start Analysis for Solana (SOL)</button>
        <h2>Messages:</h2>
        <ul id='messages'>
        </ul>
        <script>
            var ws = null;
            function connectWebSocket() {
                var wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                var wsUrl = wsProtocol + '//' + window.location.host + '/ws';
                console.log("Connecting WebSocket to:", wsUrl);
                ws = new WebSocket(wsUrl);
                ws.onopen = function(event) {
                    console.log("WebSocket connection established");
                    document.getElementById('messages').innerHTML += '<li>WebSocket Connected</li>';
                };
                ws.onmessage = function(event) {
                    var messages = document.getElementById('messages')
                    var messageData = JSON.parse(event.data);
                    var messageItem = document.createElement('li');
                    // Format message nicely, handle potential function calls/tool outputs if needed
                    var content = messageData.content ? messageData.content.replace(/\\n/g, '<br>') : '';
                    // Optionally identify tool calls/results for different styling
                    if (messageData.role === 'function' || (messageData.content && messageData.content.includes('tool_calls'))) {
                         messageItem.innerHTML = `<i><b>Tool Call/Result</b> from ${messageData.sender}</i><br><pre style='background-color:#eee; padding: 5px; border-radius: 3px;'>${content}</pre>`;
                    } else {
                         messageItem.innerHTML = `<b>${messageData.sender}</b> (to ${messageData.recipient}):<br>${content}`;
                    }

                    messages.appendChild(messageItem);
                    messages.scrollTop = messages.scrollHeight; // Auto-scroll
                };
                ws.onerror = function(event) {
                    console.error("WebSocket error observed:", event);
                    document.getElementById('messages').innerHTML += '<li>WebSocket Error</li>';
                };
                ws.onclose = function(event) {
                    console.log("WebSocket connection closed:", event.reason, "Code:", event.code);
                    var closeReason = event.reason ? `Reason: ${event.reason}` : 'Unknown Reason';
                    document.getElementById('messages').innerHTML += `<li style='color:red;'>WebSocket Disconnected (Code: ${event.code}, ${closeReason})</li>`;
                    ws = null; // Reset ws variable
                };
            }

            function startChat() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    alert("WebSocket not connected. Attempting to connect...");
                    connectWebSocket();
                    return;
                }
                 // Clear previous messages
                document.getElementById('messages').innerHTML = '<li>WebSocket Connected</li><li>Starting analysis...</li>';
                fetch('/start_chat', { method: 'POST' })
                    .then(response => {
                        if (!response.ok) {
                             // Try to get error message from response body
                            return response.json().then(errData => {
                                throw new Error(`HTTP error! status: ${response.status}, message: ${errData.message || 'No details'}`);
                            }).catch(() => {
                                // Fallback if parsing fails
                                throw new Error(`HTTP error! status: ${response.status}`);
                            });
                        }
                        return response.json();
                     })
                    .then(data => {
                        console.log("Start chat response:", data);
                        document.getElementById('messages').innerHTML += `<li>${data.message}</li>`;
                    })
                    .catch(error => {
                        console.error('Error starting chat:', error);
                        alert('Error starting chat: ' + error);
                        document.getElementById('messages').innerHTML += `<li style='color:red;'>Error starting analysis: ${error}</li>`;
                     });
            }

            // Automatically connect on load
            window.onload = connectWebSocket;
        </script>
        <style>
            body { font-family: sans-serif; }
            #messages {
                list-style-type: none;
                padding: 10px;
                margin: 0;
                height: 60vh; /* Adjust height as needed */
                overflow-y: scroll;
                border: 1px solid #ccc;
                margin-top: 10px;
                background-color: #f9f9f9;
            }
            #messages li {
                padding: 8px;
                border-bottom: 1px solid #eee;
                margin-bottom: 5px;
                line-height: 1.4;
            }
             #messages li:last-child {
                border-bottom: none;
            }
             #messages li b {
                color: #0056b3; /* Dark blue for sender */
             }
             pre { /* Style for tool call outputs */
                white-space: pre-wrap; /* Wrap long lines */
                word-wrap: break-word;
                background-color: #e9ecef;
                padding: 10px;
                border-radius: 4px;
                border: 1px solid #ced4da;
                font-family: monospace;
                font-size: 0.9em;
             }
        </style>
    </body>
</html>
"""


@app.get("/")
async def get():
    """Serves the simple HTML test client."""
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections and sends messages from the queue."""
    await websocket.accept()
    print("WebSocket client connected")
    client_ip = websocket.client.host if websocket.client else "Unknown"
    print(f"Client IP: {client_ip}")
    active_connection = True
    try:
        while active_connection:
            try:
                message = await asyncio.wait_for(message_queue.get(), timeout=1.0)  # Check queue with timeout
                await websocket.send_json(message)
                message_queue.task_done()
            except asyncio.TimeoutError:
                # Check connection liveness - simple text send/receive is prone to issues
                # A proper ping/pong implementation is more robust
                try:
                    await websocket.send_text("")  # Send a minimal text frame
                    # Note: Waiting for a specific pong is better than receiving any text
                except WebSocketDisconnect:
                    print("WebSocket client disconnected (detected during keep-alive).")
                    active_connection = False
                except Exception as e:
                    print(f"Error during WebSocket keep-alive check: {e}")
                    # Consider disconnecting if keep-alive fails repeatedly
                    # active_connection = False
            except WebSocketDisconnect as e:
                print(f"WebSocket client disconnected (Code: {e.code}, Reason: {e.reason})")
                active_connection = False
                break  # Exit loop immediately on disconnect
            except Exception as e:  # Catch broader exceptions during send/receive
                print(f"Error in WebSocket loop: {e}")
                active_connection = False  # Assume connection is broken

    except asyncio.CancelledError:
        print("WebSocket task cancelled.")
    finally:
        print("WebSocket connection closed")
        # Ensure queue is cleared if needed, or handle pending messages


# --- Modified /start_chat Endpoint ---
@app.post("/start_chat")
async def start_chat_endpoint():
    """Triggers the Autogen group chat sequence using async methods."""
    print("Received request to start async group chat...")

    # --- Instantiate Agents ---
    try:
        tech_agent = TechnicalAnalysisCryptoAgent(
            name="TechnicalAnalyst",
            config_list=config_list_openai,
        )

        sentiment_agent = CryptoSentimentNewsAgent(
            name="SentimentAnalyst",
            config_list=config_list_openai,
            news_api_key=google_api_key,
            news_search_engine_id=google_search_engine_id
        )

        # --- NEW: Investment Advisor Agent ---
        investment_advisor = autogen.AssistantAgent(
            name="InvestmentAdvisor",
            llm_config={"config_list": config_list_openai},
            system_message="""You are a Senior Investment Advisor AI.
            Your role is to synthesize the technical analysis (indicators, price action, patterns, volume) and the sentiment analysis (news sentiment, market mood) provided by the TechnicalAnalyst and SentimentAnalyst.

            You MUST perform the following steps:
            1. Wait for inputs summarizing the findings from BOTH the TechnicalAnalyst and the SentimentAnalyst. Do not provide an opinion until you have both.
            2. Carefully weigh the conclusions from both analyses. Consider if they align or contradict.
            3. Provide a SINGLE, clear final investment recommendation:
                - STRONG BUY
                - BUY
                - HOLD (or NEUTRAL)
                - SELL
                - STRONG SELL
            4. State a Confidence Level (0-100%) for your recommendation.
            5. Provide concise reasoning, explicitly referencing key points from BOTH the technical and sentiment analyses that led to your final recommendation. Explain how you weighted conflicting information if applicable. Please provide a reference if applicable!
            6. Your final response containing the recommendation MUST conclude with the word TERMINATE."""
        )
        # --- End NEW Agent ---

    except ValueError as e:
        print(f"Error initializing agents: {e}")
        return JSONResponse(status_code=500, content={"message": f"Error initializing agents: {e}"})
    except Exception as e:
        print(f"Unexpected error initializing agents: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for unexpected errors
        return JSONResponse(status_code=500, content={"message": f"Unexpected error initializing agents: {e}"})

    # --- User Proxy Agent ---
    # Use a lambda that checks for the specific termination phrase from the advisor
    def is_advisor_termination(x):
        content = x.get("content", "").strip()
        # Check if the message is from the InvestmentAdvisor and ends with TERMINATE
        return x.get("name") == investment_advisor.name and content.endswith("TERMINATE")

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,  # Only needs to receive the final message
        # is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"), # Old termination
        is_termination_msg=is_advisor_termination,  # Terminate specifically on Advisor's signal
        code_execution_config=False,
        system_message="""You are the User Proxy. Your role is to initiate the analysis request and receive the final synthesized recommendation from the InvestmentAdvisor. You do not perform analysis or make decisions.""",
        llm_config={"config_list": config_list_openai}  # Give it LLM config just in case manager asks it something
    )

    # --- Group Chat Setup ---
    # Added InvestmentAdvisor to the list
    agents_list = [user_proxy, tech_agent.agent, sentiment_agent.agent, investment_advisor]
    groupchat = autogen.GroupChat(
        agents=agents_list,
        messages=[],
        max_round=20,  # Increased rounds slightly for the extra step
        # speaker_selection_method="auto" # Default manager selection is usually best
    )
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        name="ChatManager",
        llm_config={"config_list": config_list_openai},
        # Optional: System message for the manager to guide its orchestration
        system_message="""You are the Chat Manager. Orchestrate the discussion between the UserProxy, TechnicalAnalyst, SentimentAnalyst, and InvestmentAdvisor.
        1. Ensure TechnicalAnalyst provides technical analysis using its tools.
        2. Ensure SentimentAnalyst provides sentiment analysis using its tools.
        3. Ensure BOTH analyses are summarized and available before passing control to the InvestmentAdvisor. You might need to explicitly ask the analysts to summarize if they provide raw data.
        4. Instruct the InvestmentAdvisor to synthesize the findings and provide the final recommendation.
        5. Facilitate the conversation until the InvestmentAdvisor provides the final recommendation ending in TERMINATE."""
    )

    # --- Register the Callback (Corrected Version) ---
    callback = get_websocket_message_callback(message_queue)
    print("Registering Autogen async callbacks for group chat...")

    # Register for all agents in the group AND the manager
    for agent in agents_list:
        # Use the positional argument for the trigger condition
        agent.register_reply([autogen.Agent, None], reply_func=callback, config={})
        print(f"DEBUG: Callback registered for agent: {agent.name}")

    # Register for the manager
    manager.register_reply([autogen.Agent, None], reply_func=callback, config={})
    print(f"DEBUG: Callback registered for manager: {manager.name}")

    # Register for the user_proxy sending the initial message TO the manager
    user_proxy.register_reply(manager, reply_func=callback, config={})
    print(f"DEBUG: Callback registered for user_proxy -> manager.")

    print("Async callbacks registered.")

    # --- Initiate Chat Asynchronously ---
    crypto_symbol_tv = "COINBASE:SOLUSD"
    crypto_symbol_yahoo = "SOL-USD"
    crypto_google_query = "Solana crypto news"

    # Updated chat task message to reflect the new workflow involving the advisor
    chat_task_msg = (
        f"Please coordinate an investment analysis for Solana (SOL).\n"
        f"1. Ask the TechnicalAnalyst to perform technical analysis (TradingView: '{crypto_symbol_tv}', Yahoo Finance: '{crypto_symbol_yahoo}') and provide a summary (BULLISH/BEARISH/NEUTRAL, confidence, key indicators).\n"
        f"2. Ask the SentimentAnalyst to perform sentiment analysis (Google News query: '{crypto_google_query}') and provide a summary (BULLISH/BEARISH/NEUTRAL, confidence, key news points).\n"
        f"3. Once both summaries are ready, provide them to the InvestmentAdvisor.\n"
        f"4. Instruct the InvestmentAdvisor to synthesize these analyses and give a final recommendation (STRONG BUY/BUY/HOLD/SELL/STRONG SELL), confidence level, and reasoning.\n"
        f"5. The InvestmentAdvisor's final message should end with TERMINATE."
    )
    print(f"Initiating async group chat with task:\n--- TASK START ---\n{chat_task_msg}\n--- TASK END ---")

    # Define an async function to run the async chat
    async def run_chat_async():
        print("DEBUG: run_chat_async started.")
        start_time = asyncio.get_event_loop().time()
        final_message_content = "Chat session ended unexpectedly."  # Default message
        final_message_role = "system"
        try:
            # UserProxy initiates the chat with the GroupChatManager
            await user_proxy.a_initiate_chat(
                manager,
                message=chat_task_msg,
                clear_history=True
            )
            print("Async group chat finished normally.")
            final_message_content = "Chat session finished."
            final_message_role = "system"

        except Exception as e:
            print(f"Error during Autogen async group chat: {e}")
            import traceback
            traceback.print_exc()
            final_message_content = f"Error during chat: {str(e)}"
            final_message_role = "error"

        finally:
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            print(
                f"DEBUG: run_chat_async finished. Duration: {duration:.2f} seconds. Sending final status: '{final_message_content}'")
            final_message = {"sender": "System", "recipient": "User", "content": final_message_content,
                             "role": final_message_role}
            try:
                message_queue.put_nowait(final_message)
            except asyncio.QueueFull:
                print(f"Error: Message queue full when trying to send final status: '{final_message['content']}'")

    # Run the async function as a background task
    asyncio.create_task(run_chat_async())
    print("DEBUG: asyncio.create_task(run_chat_async) called.")

    return {"message": "Autogen group chat initiated in background. Check WebSocket and server logs for messages."}


# --- Run FastAPI Server ---
if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI server...")
    # Use 0.0.0.0 to be accessible externally (e.g., Docker)
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Validate essential keys
if not openai_api_key:
    raise ValueError("Missing environment variable: OPENAI_API_KEY")
if not google_api_key:
    raise ValueError("Missing environment variable: GOOGLE_API_KEY for Sentiment Agent")
if not google_search_engine_id:
    raise ValueError("Missing environment variable: GOOGLE_CSE_ID for Sentiment Agent")

config_list_openai = [
    {
        'model': os.getenv("OPENAI_MODEL", "gpt-4o"),  # Use env var or default
        'api_key': openai_api_key,
        'base_url': openai_base_url,
    }
]

# --- FastAPI Setup ---
app = FastAPI()
# Queue to send messages from Autogen to WebSocket
message_queue = asyncio.Queue()


# --- Autogen Message Interception Callback (Unchanged) ---
def get_websocket_message_callback(queue: asyncio.Queue):
    """Creates an ASYNC callback function that puts messages onto the provided asyncio Queue."""

    async def websocket_message_callback(
            recipient: autogen.Agent,
            messages: Optional[List[Dict]] = None,
            sender: Optional[autogen.Agent] = None,
            config: Optional[Any] = None,
    ) -> tuple[bool, Optional[Dict]]:
        """
        ASYNC Callback implementation. Puts sender, recipient, and the latest message onto the queue.
        Must return (bool, Optional[Dict]) for use with a_initiate_chat.
        """
        if messages is None or sender is None:
            return False, None

        last_message = messages[-1]
        sender_name = sender.name if sender else "System"
        recipient_name = recipient.name if recipient else "System"

        message_data = {
            "sender": sender_name,
            "recipient": recipient_name,
            "content": last_message.get("content", ""),
            "role": last_message.get("role", ""),
        }

        try:
            queue.put_nowait(message_data)
        except asyncio.QueueFull:
            print("Error: Message queue is full. Dropping message.")
        except Exception as e:
            print(f"Error in websocket_message_callback putting to queue: {e}")

        return False, None

    return websocket_message_callback


# --- FastAPI Endpoints (Unchanged) ---

# Basic HTML page for WebSocket testing
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Autogen Real-time Output</title>
    </head>
    <body>
        <h1>Autogen Agent Conversation</h1>
        <button onclick="startChat()">Start Analysis for Solana (SOL)</button>
        <h2>Messages:</h2>
        <ul id='messages'>
        </ul>
        <script>
            var ws = null;
            function connectWebSocket() {
                // Use window.location.host to dynamically get host and port
                var wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                var wsUrl = wsProtocol + '//' + window.location.host + '/ws';
                console.log("Connecting WebSocket to:", wsUrl);
                ws = new WebSocket(wsUrl);
                ws.onopen = function(event) {
                    console.log("WebSocket connection established");
                    document.getElementById('messages').innerHTML += '<li>WebSocket Connected</li>';
                };
                ws.onmessage = function(event) {
                    var messages = document.getElementById('messages')
                    var messageData = JSON.parse(event.data);
                    var messageItem = document.createElement('li');
                    // Simple formatting, you can make this nicer
                    var content = messageData.content ? messageData.content.replace(/\\n/g, '<br>') : ''; // Handle newlines
                    messageItem.innerHTML = `<b>${messageData.sender}</b> (to ${messageData.recipient}):<br>${content}`;
                    messages.appendChild(messageItem);
                    // Scroll to bottom
                    messages.scrollTop = messages.scrollHeight;
                };
                ws.onerror = function(event) {
                    console.error("WebSocket error observed:", event);
                    document.getElementById('messages').innerHTML += '<li>WebSocket Error</li>';
                };
                ws.onclose = function(event) {
                    console.log("WebSocket connection closed:", event.reason, "Code:", event.code);
                    document.getElementById('messages').innerHTML += `<li>WebSocket Disconnected (Reason: ${event.reason || 'Unknown'}, Code: ${event.code})</li>`;
                    ws = null; // Reset ws variable
                };
            }

            function startChat() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    alert("WebSocket not connected. Attempting to connect...");
                    connectWebSocket(); // Attempt to connect if not already
                    // Optionally disable button and re-enable onopen, or provide feedback
                    return;
                }
                 // Clear previous messages
                document.getElementById('messages').innerHTML = '<li>WebSocket Connected</li><li>Starting analysis...</li>';
                fetch('/start_chat', { method: 'POST' }) // Use POST for actions
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        return response.json();
                     })
                    .then(data => {
                        console.log("Start chat response:", data);
                        document.getElementById('messages').innerHTML += `<li>${data.message}</li>`;
                    })
                    .catch(error => {
                        console.error('Error starting chat:', error);
                        alert('Error starting chat: ' + error);
                        document.getElementById('messages').innerHTML += `<li>Error starting analysis: ${error}</li>`;
                     });
            }

            // Automatically connect on load
            window.onload = connectWebSocket;
        </script>
        <style>
            #messages {
                list-style-type: none;
                padding: 0;
                margin: 0;
                height: 400px; /* Or desired height */
                overflow-y: scroll;
                border: 1px solid #ccc;
                margin-top: 10px;
            }
            #messages li {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
             #messages li:last-child {
                border-bottom: none;
            }
             #messages li b {
                color: navy;
             }
        </style>
    </body>
</html>
"""


@app.get("/")
async def get():
    """Serves the simple HTML test client."""
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections and sends messages from the queue."""
    await websocket.accept()
    print("WebSocket client connected")
    client_ip = websocket.client.host if websocket.client else "Unknown"
    print(f"Client IP: {client_ip}")
    active_connection = True
    try:
        while active_connection:
            # Wait for a message from the Autogen callback via the queue
            try:
                message = await asyncio.wait_for(message_queue.get(), timeout=1.0)  # Timeout to check connection
                # print(f"WebSocket: Sending message to client: {message}")
                await websocket.send_json(message)
                message_queue.task_done()  # Notify queue that task is complete
            except asyncio.TimeoutError:
                # Check if client is still connected
                try:
                    # Ping the client to see if it's responsive
                    await websocket.send_text("")  # Sending empty text as a lightweight ping
                    pong_waiter = await websocket.receive_text()  # Should ideally implement proper ping/pong
                except WebSocketDisconnect:
                    print("WebSocket client disconnected during timeout check.")
                    active_connection = False
                except Exception as e:
                    # Handle other potential errors during receive
                    print(f"Error during WebSocket keep-alive check: {e}")
                    active_connection = False


    except WebSocketDisconnect as e:
        print(f"WebSocket client disconnected (Code: {e.code}, Reason: {e.reason})")
        active_connection = False
    except asyncio.CancelledError:
        print("WebSocket task cancelled.")
        active_connection = False
    except Exception as e:
        print(f"Error in WebSocket endpoint: {e}")
        active_connection = False  # Ensure loop termination on unexpected errors
    finally:
        # Clean up if necessary
        print("WebSocket connection closed")
        # Optional: You might want to signal the Autogen task to stop if the WS disconnects
        # or handle this based on your application's logic.


# --- Modified /start_chat Endpoint ---
@app.post("/start_chat")
async def start_chat_endpoint():
    """Triggers the Autogen group chat sequence using async methods."""
    print("Received request to start async group chat...")

    # --- Instantiate Agents ---
    try:
        tech_agent = TechnicalAnalysisCryptoAgent(
            name="TechnicalAnalyst",
            config_list=config_list_openai,
        )

        sentiment_agent = CryptoSentimentNewsAgent(
            name="SentimentAnalyst",
            config_list=config_list_openai,
            news_api_key=google_api_key,
            news_search_engine_id=google_search_engine_id
        )
    except ValueError as e:
        print(f"Error initializing agents: {e}")
        return {"message": f"Error initializing agents: {e}"}
    except Exception as e:
        print(f"Unexpected error initializing agents: {e}")
        return {"message": f"Unexpected error initializing agents: {e}"}

    # --- User Proxy Agent ---
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,
        # Add system message to guide the user proxy if needed
        system_message="""You are the User Proxy.
        Your role is to initiate the analysis request for a specific cryptocurrency (e.g., Solana) to the group chat manager.
        Ensure the request specifies the need for both technical analysis (using TradingView and Yahoo Finance data) and sentiment analysis (using Google News).
        Mention the specific symbols (e.g., TradingView: COINBASE:SOLUSD, Yahoo Finance: SOL-USD, Google Query: 'Solana crypto news').
        Receive the final combined recommendation from the group chat.
        You do not need to perform analysis yourself.
        Terminate the conversation once the final recommendation is provided by the manager or another designated agent.
        """,
        llm_config={"config_list": config_list_openai}  # Manager might need LLM too
    )

    # --- Group Chat Setup ---
    agents_list = [user_proxy, tech_agent.agent, sentiment_agent.agent]  # Use the .agent attribute
    groupchat = autogen.GroupChat(
        agents=agents_list,
        messages=[],
        max_round=15,  # Increased max rounds for potentially more interaction
        # speaker_selection_method="auto" # Default, manager decides next speaker
        # speaker_selection_method='round_robin' # Alternative if needed
    )

    # Group Chat Manager (using a basic AssistantAgent as manager)
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        name="ChatManager",
        llm_config={"config_list": config_list_openai}  # Manager might need LLM to decide speaker
    )

    # --- Register the Callback ---
    callback = get_websocket_message_callback(message_queue)
    print("Registering Autogen async callbacks for group chat...")

    # Register for all agents in the group AND the manager
    for agent in agents_list:
        # FIX: Removed redundant trigger keyword argument
        agent.register_reply([autogen.Agent, None], reply_func=callback, config={})

    # Also register for the manager if it might send messages directly
    # FIX: Removed redundant trigger keyword argument
    manager.register_reply([autogen.Agent, None], reply_func=callback, config={})

    # Crucially, register for the user_proxy sending the initial message TO the manager
    # FIX: Removed redundant trigger keyword argument
    user_proxy.register_reply(manager, reply_func=callback, config={})

    print("Async callbacks registered.")

    # --- Initiate Chat Asynchronously ---
    # Define the initial task for the user proxy to send to the manager
    crypto_symbol_tv = "COINBASE:SOLUSD"
    crypto_symbol_yahoo = "SOL-USD"
    crypto_google_query = "Solana crypto news"

    chat_task_msg = (
        f"Please provide a combined investment recommendation for Solana (SOL). "
        f"Involve the TechnicalAnalyst (using TradingView symbol '{crypto_symbol_tv}' and Yahoo Finance symbol '{crypto_symbol_yahoo}') "
        f"and the SentimentAnalyst (using Google News query '{crypto_google_query}'). "
        f"The final output should be a consolidated recommendation (BULLISH/BEARISH/NEUTRAL) with a confidence score and reasoning based on *both* technicals and sentiment. "
        f"Ensure the analysts use their respective tools. Conclude the final response with TERMINATE."
    )
    print(f"Initiating async group chat with task: '{chat_task_msg}'")

    # Define an async function to run the async chat
    async def run_chat_async():
        try:
            # UserProxy initiates the chat with the GroupChatManager
            await user_proxy.a_initiate_chat(
                manager,
                message=chat_task_msg,
                clear_history=True
            )
            print("Async group chat finished.")
            final_message = {"sender": "System", "recipient": "User", "content": "Chat session finished.",
                             "role": "system"}

        except Exception as e:
            print(f"Error during Autogen async group chat: {e}")
            final_message = {"sender": "System", "recipient": "User", "content": f"Error during chat: {e}",
                             "role": "error"}

        finally:
            # Send final status message to WebSocket
            try:
                # Using put_nowait as it's called from the running async task
                message_queue.put_nowait(final_message)
            except asyncio.QueueFull:
                print(f"Error: Message queue full when trying to send final status: '{final_message['content']}'")

    # Run the async function as a background task
    asyncio.create_task(run_chat_async())

    return {"message": "Autogen group chat initiated in background. Check WebSocket for messages."}

@app.get(
    "/price",
    summary="Get Historical Stock Prices",
    description=f"""Fetches historical price data for a given stock symbol.
    You can specify a date range using `ts_start` and `ts_end` (as UNIX timestamps)
    OR use a predefined `range`. If both are provided, `ts_start`/`ts_end` take precedence.
    Valid ranges: {', '.join(VALID_RANGES)}.""",
    response_description="Parsed stock price data in JSON format or error details.",
)
async def get_price(
        symbol: str = Query(..., description="Stock symbol (e.g., SOL-USD, AAPL, MSFT)"),
        interval: str = Query("1d", description="Data interval (e.g., 1m, 5m, 1h, 1d, 1wk, 1mo)"),
        range: Optional[str] = Query(None,
                                     description=f"Predefined date range. Valid options: {', '.join(VALID_RANGES)}"),
        ts_start: Optional[int] = Query(None, description="Start date as UNIX timestamp (seconds)"),
        ts_end: Optional[int] = Query(None, description="End date as UNIX timestamp (seconds)")
):
    """
    FastAPI route to fetch and parse stock price data.
    """
    params = {
        "interval": interval,
        "includePrePost": "true",
        "events": "div|split|earn",
        "lang": "en-GB",
        "region": "GB",
        "corsDomain": "uk.finance.yahoo.com"  # Found this param often used
    }

    # --- Parameter Validation and Selection ---
    if ts_start is not None and ts_end is not None:
        if ts_start >= ts_end:
            raise HTTPException(status_code=400, detail="ts_start must be before ts_end")
        params["period1"] = str(ts_start)
        params["period2"] = str(ts_end)
        print(f"Using timestamp range: {ts_start} to {ts_end}")
    elif range is not None:
        if range not in VALID_RANGES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid range '{range}'. Valid ranges are: {', '.join(VALID_RANGES)}"
            )
        params["range"] = range
        print(f"Using predefined range: {range}")
    else:
        # Default behavior if neither is specified (e.g., default to '1d' or raise error)
        # raise HTTPException(status_code=400, detail="Either 'range' or both 'ts_start' and 'ts_end' must be provided.")
        print("Defaulting to range '1d' as no range or timestamps were provided.")
        params["range"] = "1d"  # Default to 1 day if nothing else specified

    # --- Construct URL ---
    request_url = YAHOO_FINANCE_URL.format(symbol=symbol)

    # --- Make HTTP Request ---
    async with httpx.AsyncClient() as client:
        try:
            print(f"Requesting URL: {request_url} with params: {params}")
            response = await client.get(request_url, headers=HEADERS, params=params, timeout=15.0)  # Added timeout
            response.raise_for_status()  # Raise HTTPStatusError for 4xx/5xx responses
            raw_data = response.json()

        except httpx.TimeoutException:
            print(f"Request timed out for symbol {symbol}")
            raise HTTPException(status_code=504, detail="Request to Yahoo Finance timed out.")
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}: {e}")
            raise HTTPException(status_code=503,
                                detail=f"Service Unavailable: Could not connect to Yahoo Finance. Error: {e}")
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            # Forward Yahoo's error if possible
            detail_msg = f"Yahoo Finance API error ({e.response.status_code})"
            try:
                # Try to parse Yahoo's error message
                yahoo_error = e.response.json()
                detail_msg += f": {yahoo_error.get('chart', {}).get('error', {}).get('description', e.response.text)}"
            except json.JSONDecodeError:
                detail_msg += f": {e.response.text}"  # Fallback to raw text
            raise HTTPException(status_code=e.response.status_code, detail=detail_msg)
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON response from Yahoo Finance: {e}")
            print(f"Response text: {response.text[:500]}...")  # Log part of the invalid response
            raise HTTPException(status_code=500, detail="Received invalid JSON response from Yahoo Finance.")

    # --- Parse Data ---
    parsed_data = parse_to_time_based_json(raw_data)

    if parsed_data is None:
        # Check if Yahoo returned an error message within the JSON
        yahoo_error_info = raw_data.get("chart", {}).get("error")
        if yahoo_error_info:
            error_code = yahoo_error_info.get("code", "N/A")
            error_desc = yahoo_error_info.get("description", "Unknown error from Yahoo Finance.")
            print(f"Yahoo Finance returned error: {error_code} - {error_desc}")
            # Decide on appropriate status code, 404 if symbol not found, 500 otherwise?
            status = 404 if "No data found" in error_desc else 500
            raise HTTPException(status_code=status, detail=f"Yahoo Finance error: {error_desc} (Code: {error_code})")
        else:
            # Parsing failed for other reasons or data was empty/malformed
            print(f"Failed to parse data for symbol {symbol}. Check logs for details.")
            raise HTTPException(status_code=500,
                                detail="Failed to parse data received from Yahoo Finance or data was empty/invalid.")

    # --- Return Success ---
    return parsed_data


# --- Run FastAPI Server ---
if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI server...")
    # Use 0.0.0.0 to be accessible externally (e.g., Docker)
    uvicorn.run(app, host="0.0.0.0", port=8000)
