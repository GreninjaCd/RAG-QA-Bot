# RAG-QA Bot

## Overview
RAG-QA Bot is a question-answering system that uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses based on provided documents.

This app's knowledge base is a local document corpus stored in the `chroma_db` vector store. It answers questiosn related to Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. It retrieves relevant document chunks using embeddings, then passes those retrieved passages to the LLM so answers are grounded in the indexed source material.

## Features
- Document retrieval and indexing
- Question answering with source attribution
- RAG-based response generation
- Support for multiple document formats

## Installation
```bash
git clone <repository-url>
cd RAG-QA-Bot
pip install -r requirements.txt
```

## Usage (local)
Run the Streamlit app:
```powershell
& "C:/Users/Arghojit/AppData/Local/Programs/Python/Python313/python.exe" -m pip install -r requirements.txt
& "C:/Users/Arghojit/AppData/Local/Programs/Python/Python313/python.exe" -m streamlit run chatbot_streamlit.py --server.port 8501
```

## Project Structure
```
RAG-QA Bot/
├── src/
├── data/
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Deploying to Streamlit Cloud
1. Push this repository to GitHub.
2. Add a `requirements.txt` (already included).
3. Add any API keys to Streamlit secrets or a `.env` file and set them in the Streamlit Cloud UI.
4. On Streamlit Cloud, select the GitHub repo and the `chatbot_streamlit.py` file as the app entrypoint.

Notes:
- If `ChatGroq` requires provider credentials, add them to `.env` or Streamlit secrets before deploying.
- If you need a smaller deployment, consider replacing large runtime dependencies (ONNX/onnxruntime) with hosted LLM APIs.

## Contributing
Contributions are welcome. Please submit pull requests with clear descriptions.

## License
MIT License