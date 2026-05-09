import os
import textwrap

import streamlit as st
from dotenv import load_dotenv
import requests

from graph_builder import ingest_text, get_context_for_query, graph_stats

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
    .big-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: rgba(1,105,111,0.08);
        color: #01696f;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def call_llm(context: str, question: str) -> str:
    llm_api_key = get_setting("LLM_API_KEY")
    llm_api_url = get_setting("LLM_API_URL")
    model_name = get_setting("MODEL_NAME", "gpt-4o-mini")

    if not llm_api_key or not llm_api_url:
        return (
            "LLM credentials are not configured. "
            "Please set LLM_API_KEY and LLM_API_URL as environment variables or Streamlit secrets."
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
        resp = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=40)
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
        options=["Paste text", "Upload .txt"],
        index=0,
        help="Start with simple text; you can add more documents over time.",
    )

    doc_id = st.text_input(
        "Document ID",
        placeholder="e.g. policy-001 or sprint-notes-2026-05-09",
    )

    uploaded_text = ""
    if mode == "Upload .txt":
        file = st.file_uploader("Upload a text file", type=["txt"])
        if file is not None:
            uploaded_text = file.read().decode("utf-8")
    else:
        uploaded_text = st.text_area(
            "Paste document text",
            height=200,
            placeholder="Paste meeting notes, policy text, design docs, etc.",
        )

    if st.button("Ingest into Knowledge Graph", use_container_width=True):
        if not doc_id.strip():
            st.error("Please provide a document ID.")
        elif not uploaded_text.strip():
            st.error("Please provide some text to ingest.")
        else:
            with st.spinner("Building knowledge graph from this document..."):
                ingest_text(doc_id.strip(), uploaded_text)
            st.success(f"Document `{doc_id}` ingested successfully into the graph.")

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
