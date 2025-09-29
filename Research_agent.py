import os
from dotenv import load_dotenv
# LangChain components
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.tools import Tool
from search_tool import search_web
from langchain_openai import OpenAI # Ensure you have langchain-openai installed
from langchain.chat_models import ChatOpenAI



# --- Wikipedia Tool ---
# 1. Configure the API wrapper for Wikipedia
wikipedia_wrapper = WikipediaAPIWrapper(
    top_k_results=1,               # Only fetch the top result
    doc_content_chars_max=1500     # Limit the content size
)
# 2. Create the LangChain tool from the wrapper
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

# 3. Create the DDGS Tool object (using the @tool decorator would be the modern way, 
# but using the Tool class here to ensure compatibility)
ddgs_tool = Tool(
    name="DuckDuckGo_Search",
    func=search_web,
    description="Useful for questions requiring real-time or general web search. Input should be a concise query string."
)

# 4. List all tools available to the agent
tools = [wikipedia_tool, ddgs_tool]

# --- Setup LLM and Prompt ---
load_dotenv()

# Pull the ReAct prompt template from LangChain's hub
# This template is pre-designed to guide the LLM's Thought/Action/Observation process.
prompt = hub.pull("hwchase17/react")

# Configure the LLM to use OpenRouter with the ChatOpenAI client
# LangChain's ChatOpenAI is compatible with OpenRouter's API endpoint.
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY") # Uses your existing .env variable
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=openrouter_api_key,
    model_name="deepseek/deepseek-chat-v3.1:free", # Use the powerful Gemini model on OpenRouter
    temperature=0
)

# --- Create the ReAct Agent ---
# 1. Create the agent: This wraps the LLM, the tools, and the ReAct prompt.
agent = create_react_agent(llm, tools, prompt)

# 2. Create the Agent Executor: This is the core loop manager. 
# It continuously calls the model and runs the tools until a final answer is generated.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Run the Research Assistant ---
print("--- Research Assistant Agent Initialized ---")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
        
    try:
        # The agent executor runs the multi-step ReAct loop
        result = agent_executor.invoke({"input": user_input})
        print(f"\nAgent Final Answer: {result['output']}\n")
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        print("Please try a different query or check your API keys.")