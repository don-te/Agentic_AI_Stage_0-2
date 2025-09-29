import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from search_tool import search_web
from news_tool import fetch_news

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model_name = "deepseek/deepseek-chat-v3.1:free"

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches the web for the given query using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to use."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_news",
            "description": "Fetches the latest news articles for a given topic and returns their content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The news topic to fetch articles for."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# This is the new conversational loop
while True:
    user_prompt = input("You: ")
    if user_prompt.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    messages = [{"role": "user", "content": user_prompt}]
    
    # First call: Get the model's decision on what to do.
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
    )
    
    # Check if the response contains a tool call.
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0].function
        
        print(f"Model wants to call tool: {tool_call.name}")
        print(f"Arguments: {tool_call.arguments}")

        arguments = json.loads(tool_call.arguments)
        
        if tool_call.name == "search_web":
            tool_results = search_web(**arguments)
        elif tool_call.name == "fetch_news":
            tool_results = fetch_news(**arguments)
        
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments
                    }
                }
            ]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": "call_123",
            "content": tool_results
        })

        # Second call: Send the tool results back to the model.
        final_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

        print("\nAgent:", final_response.choices[0].message.content)
    
    else:
        # If no tool call, print the text response directly.
        print("\nAgent:", response.choices[0].message.content)