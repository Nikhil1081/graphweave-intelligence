# GraphWeave Intelligence

Enterprise Knowledge Graph RAG built with Streamlit.

## Features

- Ingest multiple enterprise documents (policies, meeting notes, specs).
- Build a knowledge graph of entities and their relationships.
- Ask questions and get answers grounded in graph-based context.
- See graph stats and central entities.
- Modern, themed Streamlit UI.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

Set your LLM credentials via `.env` or Streamlit secrets.

### Streamlit Community Cloud secrets (Gemini)

In Streamlit Cloud → App settings → Secrets:

```toml
LLM_PROVIDER = "gemini"
LLM_API_KEY = "PASTE_YOUR_GEMINI_KEY"
MODEL_NAME = "gemini-1.5-flash"
```
