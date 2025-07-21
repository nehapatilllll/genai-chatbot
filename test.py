import requests, os, streamlit as st
url = "https://api.deepseek.com/v1/models"
headers = {"Authorization": f"Bearer {st.secrets['DEEPSEEK_API_KEY']}"}
print(requests.get(url, headers=headers, timeout=10).json())