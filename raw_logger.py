import re
from llm import OllamaClient
import logging
logging.basicConfig(level=logging.DEBUG)

def check():
    client = OllamaClient("llama3.1:8b")
    messages = [{'role': 'user', 'content': 'Say hi'}]
    print("RES:", repr(client.chat(messages)))
check()
