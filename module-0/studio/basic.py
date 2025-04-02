from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools import TavilySearchResults

import os

# Load environment variables from .env file
load_dotenv()

gpt4o_chat = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0
)

gpt35_chat = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0.0
)

# Create a message
msg = HumanMessage(content="Hello World!", name="Ramon")

# Message list
messages = [msg]

# Send the message to the chat model
response = gpt4o_chat.invoke(messages)

# Print the response
print(response)

# Send the message to the chat model
response = gpt4o_chat.invoke("Hello World!")

# Print the response
print(response)

# Send the message to the chat model
response = gpt35_chat.invoke("Hello World!")

# Print the response
print(response)

# Create a search tool
search = TavilySearchResults(max_results=3)

# Search a query
response = search.invoke("What is LangGraph?")

# Print the response
print(response)