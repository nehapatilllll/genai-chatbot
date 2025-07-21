import tiktoken, streamlit as st
from openai import OpenAI
client = OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"],
    base_url=st.secrets["DEEPSEEK_BASE_URL"],
)

MODEL = "deepseek-chat"
enc = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def trim_history(history, max_tokens=6000):
    while history and sum(count_tokens(m["content"]) for m in history) > max_tokens:
        history.pop(0)
    return history

def chat_completion(messages):
    messages = trim_history(messages)
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.3)
    return resp.choices[0].message.content