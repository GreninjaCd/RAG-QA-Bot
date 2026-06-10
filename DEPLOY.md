Streamlit Cloud deployment

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io and log in with GitHub.
3. Click "New app" and select your repository and branch.
4. Set the main file to `chatbot_streamlit.py`.
5. In "Advanced settings" add any required environment variables or secrets (e.g., GROQ_API_KEY) in the "Secrets" panel.
6. Click "Deploy".

Notes:
- Use `requirements.txt` to specify dependencies.
- If the app needs large local models, prefer hosted LLM APIs for cloud.
