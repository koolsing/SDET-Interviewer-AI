import requests
from config import OLLAMA_BASE_URL
from interview_engine import _build_system_prompt

def run():
    prompt = _build_system_prompt("HR", "hr-round", 20, "")
    payload = {
        "model": "llama3.1:8b",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Please begin the interview by greeting me and asking the first question."}
        ],
        "stream": False,
        "options": {"temperature": 0.7, "num_ctx": 8192}
    }
    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
    out = r.json().get("message", {}).get("content", "ERROR_NO_CONTENT")
    print("OLLAMA RAW RESPONSE WITH USER KICKOFF:")
    print(repr(out))

if __name__ == "__main__":
    run()
