import sys
from llm import get_client

def test():
    c = get_client("llama3.1:8b")
    out = c.chat([{"role": "user", "content": "Hello"}], stream=False)
    print("OUTPUT IS:", repr(out))

if __name__ == "__main__":
    test()
