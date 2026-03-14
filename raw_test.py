import requests
import json
payload = {
    "model": "llama3.1:8b",
    "messages": [
        {"role": "system", "content": "You are a senior HR round interviewer conducting a 20-minute interview for an SDET. Ask a question."},
        {"role": "user", "content": "Hi"}
    ],
    "stream": False,
    "options": {"temperature": 0.7, "num_ctx": 8192}
}
r = requests.post("http://localhost:11434/api/chat", json=payload)
data = r.json()
print("RAW:")
print(repr(data["message"]["content"]))

from llm import _strip_think
print("\nSTRIPPED:")
print(repr(_strip_think(data["message"]["content"])))
