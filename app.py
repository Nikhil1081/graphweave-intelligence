import os
import textwrap
import io

import streamlit as st
from dotenv import load_dotenv
import requests
import pandas as pd
from pypdf import PdfReader
from docx import Document

from graph_builder import ingest_text, get_context_for_query, graph_stats, load_graph, GRAPH_PATH

load_dotenv()


def get_setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


st.set_page_config(
    page_title="GraphWeave Intelligence",
    page_icon="🕸️",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* App background */
    [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(900px circle at 8% 8%, rgba(124, 58, 237, 0.22), transparent 55%),
          radial-gradient(800px circle at 92% 18%, rgba(34, 211, 238, 0.18), transparent 50%),
          radial-gradient(700px circle at 30% 88%, rgba(59, 130, 246, 0.16), transparent 55%),
          linear-gradient(180deg, #050816 0%, #050816 55%, #070A16 100%);
    }

    /* Sidebar polish */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Typography */
    .big-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #A78BFA 0%, #22D3EE 55%, #60A5FA 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .subtitle {
        color: rgba(229,231,235,0.78);
        font-size: 0.98rem;
        margin-bottom: 1.35rem;
        line-height: 1.45;
    }

    /* Pills */
    .pill {
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(124,58,237,0.22), rgba(34,211,238,0.18));
        border: 1px solid rgba(255,255,255,0.10);
        color: rgba(229,231,235,0.92);
        font-size: 0.82rem;
        font-weight: 650;
        margin-right: 0.5rem;
        backdrop-filter: blur(10px);
    }

    /* “Card” feel for common containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(11,18,37,0.50);
        backdrop-filter: blur(10px);
    }

    /* Inputs */
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        background: rgba(5,8,22,0.55) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.14);
        background: linear-gradient(90deg, rgba(124,58,237,0.85), rgba(34,211,238,0.75));
        color: #0B1025;
        font-weight: 750;
    }
    .stButton > button:hover {
        filter: brightness(1.06);
        border-color: rgba(255,255,255,0.20);
    }

    /* Metrics */
    [data-testid="stMetric"] {
        padding: 12px 14px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(5,8,22,0.35);
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.10);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def call_llm(context: str, question: str) -> str:
    llm_api_key = get_setting("LLM_API_KEY")
    llm_api_url = get_setting("LLM_API_URL")
    model_name = get_setting("MODEL_NAME", "gpt-4o-mini")
    provider = (get_setting("LLM_PROVIDER", "openai") or "openai").lower()

    if not llm_api_key:
        return (
            "LLM credentials are not configured. "
            "Please set LLM_API_KEY and LLM_API_URL as environment variables or Streamlit secrets."
        )

    if provider == "gemini":
        # Gemini (Google AI Studio) API
        # Docs: https://ai.google.dev/gemini-api/docs
        if not model_name:
            model_name = "gemini-1.5-flash"

        url = (
            llm_api_url
            or f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        )

        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "You are an assistant that answers using only the provided enterprise context. "
                            "If the context is insufficient, clearly say so and do NOT hallucinate."
                        )
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"Context:\n{context}\n\nQuestion: {question}"}],
                }
            ],
            "generationConfig": {"temperature": 0.2},
        }

        try:
            resp = requests.post(url, params={"key": llm_api_key}, json=payload, timeout=40)
            resp.raise_for_status()
            data = resp.json()
            return (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
        except Exception as e:
            return f"Error calling Gemini: {e}"

    # Default: OpenAI-compatible Chat Completions API
    if not llm_api_url:
        return (
            "LLM_API_URL is missing. "
            "Set it to your provider endpoint (e.g. OpenAI chat completions URL) or switch to Gemini with LLM_PROVIDER=gemini."
        )

    headers = {
        "Authorization": f"Bearer {llm_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant that answers using only the provided enterprise context. "
                    "If the context is insufficient, clearly say so and do NOT hallucinate."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        "temperature": 0.2,
    }

    try:
        resp = requests.post(llm_api_url, headers=headers, json=payload, timeout=40)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error calling LLM: {e}"


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.sidebar:
    st.markdown("### 🧠 GraphWeave Intelligence")
    st.caption(
        "Enterprise Knowledge Graph RAG — ingest company docs, "
        "build a graph, and ask high-quality questions."
    )

    st.markdown("#### Ingest documents")
    mode = st.radio(
        "Source",
        options=["Paste text", "Upload files"],
        index=0,
        help="Start with simple text; you can add more documents over time.",
    )

    doc_id = st.text_input(
        "Document ID",
        placeholder="e.g. policy-001 or sprint-notes-2026-05-09",
    )

    def _text_from_upload(uploaded_file) -> str:
        name = (uploaded_file.name or "").lower()
        raw = uploaded_file.getvalue()

        if name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(raw))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n\n".join(pages).strip()

        if name.endswith(".docx"):
            doc = Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs).strip()

        # txt / md fallback
        return raw.decode("utf-8", errors="ignore").strip()

    uploaded_text = ""
    uploaded_files = []
    if mode == "Upload files":
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True,
            help="Supported: .txt, .md, .pdf, .docx",
        )
        st.caption("Tip: If Document ID is empty, the filename will be used.")
    else:
        uploaded_text = st.text_area(
            "Paste document text",
            height=200,
            placeholder="Paste meeting notes, policy text, design docs, etc.",
        )

    if st.button("Ingest into Knowledge Graph", use_container_width=True):
        if mode == "Upload files":
            if not uploaded_files:
                st.error("Please upload at least one file.")
            else:
                with st.spinner("Extracting text and building knowledge graph..."):
                    ok = 0
                    for f in uploaded_files:
                        inferred_id = (doc_id.strip() or f.name).strip()
                        try:
                            text = _text_from_upload(f)
                            if not text:
                                continue
                            ingest_text(inferred_id, text)
                            ok += 1
                        except Exception as e:
                            st.warning(f"Failed to ingest `{f.name}`: {e}")
                if ok:
                    st.success(f"Ingested {ok} document(s) successfully.")
                else:
                    st.error("No text could be extracted from the uploaded file(s).")
        else:
            if not doc_id.strip():
                st.error("Please provide a document ID.")
            elif not uploaded_text.strip():
                st.error("Please provide some text to ingest.")
            else:
                with st.spinner("Building knowledge graph from this document..."):
                    ingest_text(doc_id.strip(), uploaded_text)
                st.success(f"Document `{doc_id}` ingested successfully into the graph.")

    st.markdown("---")
    st.markdown("#### LLM status")
    provider = (get_setting("LLM_PROVIDER", "openai") or "openai").lower()
    model_name = get_setting("MODEL_NAME")
    has_key = bool(get_setting("LLM_API_KEY"))
    st.caption(f"Provider: **{provider}**")
    if model_name:
        st.caption(f"Model: **{model_name}**")
    st.caption(f"API key configured: **{'yes' if has_key else 'no'}**")

    st.markdown("---")
    st.markdown("#### Graph status")

    num_nodes, num_edges, avg_degree, top_nodes = graph_stats()
    st.metric("Entities", num_nodes)
    st.metric("Connections", num_edges)
    st.metric("Average degree", round(avg_degree, 2))

    if top_nodes:
        st.caption("Top central entities")
        for name, deg in top_nodes[:5]:
            st.write(f"• **{name}** — degree {deg}")

    if st.button("Reset graph (delete all data)", use_container_width=True, type="secondary"):
        if GRAPH_PATH.exists():
            GRAPH_PATH.unlink()
        st.session_state.chat_history = []
        st.success("Graph reset. Ingest a document to start again.")
        st.rerun()


col_left, col_right = st.columns([2.5, 1.5])

with col_left:
    st.markdown('<div class="big-title">GraphWeave Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Ask questions over your enterprise knowledge graph, '
        'backed by structured entity relationships instead of plain keyword search.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="pill">Knowledge Graph RAG</span>'
        '<span class="pill">Multi-document context</span>'
        '<span class="pill">Explainable answers</span>',
        unsafe_allow_html=True,
    )

    st.markdown("### 💬 Ask a question")

    question = st.text_input(
        "Query",
        placeholder="e.g. What dependencies exist between the Payments team and the Billing service?",
    )

    col_q1, col_q2 = st.columns([1, 4])
    with col_q1:
        ask_btn = st.button("Ask", use_container_width=True)
    with col_q2:
        clear_btn = st.button("Clear chat", use_container_width=True)

    if clear_btn:
        st.session_state.chat_history = []

    if ask_btn and question.strip():
        with st.spinner("Retrieving graph context and querying the model..."):
            context = get_context_for_query(question)
            if not context:
                answer = (
                    "I don't have any knowledge yet. Please ingest at least one document "
                    "from the sidebar before asking questions."
                )
            else:
                answer = call_llm(context, question)

        st.session_state.chat_history.append((question, answer))
        context_used = context
    else:
        context_used = None

    if st.session_state.chat_history:
        st.markdown("### 🧾 Conversation")
        for q, a in reversed(st.session_state.chat_history[-6:]):
            with st.container():
                st.markdown(f"**You:** {q}")
                st.markdown(f"**GraphWeave:** {a}")
                st.markdown("---")

with col_right:
    st.markdown("### 🔍 Context used")
    if context_used:
        st.caption("Latest answer was based on this extracted context:")
        st.code(textwrap.shorten(context_used, width=1600, placeholder="\n…"), language="markdown")
    else:
        st.caption(
            "When you ask a question, the app will show the exact sentences "
            "from your documents that were used to answer."
        )

    st.markdown("### 📚 How it works")
    st.markdown(
        """
        1. **Ingest** documents from the sidebar (policies, notes, specs).  
        2. The app builds a **knowledge graph** of entities and relationships.  
        3. Your question is mapped onto the graph to fetch a focused context.  
        4. An LLM answers using only that context for **grounded responses**.
        """
    )

    st.markdown("### 🕸️ Graph explorer")
    G = load_graph()
    if len(G) == 0:
        st.caption("No entities yet. Ingest a document to explore the graph.")
    else:
        node_degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
        all_nodes = [n for n, _ in node_degrees]

        search = st.text_input("Find entity", placeholder="Type to filter…", key="entity_search")
        if search.strip():
            filtered = [n for n in all_nodes if search.lower() in str(n).lower()]
        else:
            filtered = all_nodes

        if not filtered:
            st.info("No matches. Try a different search.")
        else:
            selected = st.selectbox(
                "Entity",
                options=filtered[:500],
                index=0,
                help="Explore an entity’s neighbors, supporting sentences, and source docs.",
            )

            node_data = G.nodes[selected]
            st.caption(
                f"Degree: **{G.degree(selected)}** · "
                f"Docs: **{len(node_data.get('docs', []))}** · "
                f"Sentences: **{len(node_data.get('sentences', []))}**"
            )

            neighbors = []
            for neigh in G.neighbors(selected):
                edge = G.get_edge_data(selected, neigh) or {}
                neighbors.append(
                    {
                        "neighbor": neigh,
                        "weight": int(edge.get("weight", 1)),
                        "neighbor_degree": int(G.degree(neigh)),
                    }
                )

            if neighbors:
                df = pd.DataFrame(neighbors).sort_values(
                    by=["weight", "neighbor_degree"], ascending=False
                )
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("No neighbors yet for this entity.")

            with st.expander("Supporting sentences", expanded=False):
                sentences = node_data.get("sentences", [])
                if sentences:
                    for s in sentences[:30]:
                        st.write(f"- {s}")
                else:
                    st.caption("No sentences stored.")

            with st.expander("Source documents", expanded=False):
                docs = node_data.get("docs", [])
                if docs:
                    for d in docs:
                        st.write(f"- {d}")
                else:
                    st.caption("No doc ids stored.")
