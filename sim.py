import sys
from llm import get_client
from interview_engine import InterviewSession
from main import _clean_response

def run():
    client = get_client("llama3.1:8b")
    session = InterviewSession("HR", "hr-round", 20, client)
    
    print("STARTING...")
    q1 = session.start()
    print("Q1 RAW:", repr(q1))
    print("Q1 CLEAN:", repr(_clean_response(q1)))

if __name__ == "__main__":
    run()
