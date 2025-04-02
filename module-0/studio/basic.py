from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(override=True)

# Verify the API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Create a search tool
tavily_search = TavilySearchResults(max_results=3)

# Tools
tools = [tavily_search]

# LLMs
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.0)

# LLM with bound tools
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with searching the web for information using the tavily search tool. Provide a summary of the search results."
)


# Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
builder.add_edge("tools", END)

# Compile graph
graph = builder.compile()

# Run graph
response = graph.invoke({"messages": [HumanMessage(content="What is LangGraph?")]})
print(response)
