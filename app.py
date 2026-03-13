# app.py
import streamlit as st
import google.generativeai as genai

from config.config import GEMINI_API_KEY, PDF_DATA_PATH, GEMINI_MODEL_NAME
from utils.rag_pipeline import RAGPipeline
from utils.web_search import web_search
from models.llm import GeminiLLM

# -------------------------
# Validate API Key
# -------------------------
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please add it in your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# -------------------------
# Initialize RAG + LLM
# -------------------------
@st.cache_resource
def init_rag():
    pipeline = RAGPipeline()
    try:
        pipeline.load_pdf(PDF_DATA_PATH)
    except Exception as e:
        st.warning(f"PDF not loaded: {e}")
    return pipeline

rag = init_rag()
llm = GeminiLLM()

# -------------------------
# Context retrieval
# -------------------------
def get_context(question):
    try:
        doc, score = rag.retrieve(question)

        if doc is None or score > 1.2:
            source = "Web Search"
            context = web_search(question)
        else:
            source = "PDF Document"
            context = doc

        return context, source

    except Exception as e:
        return web_search(question), "Web Search (fallback)"

# -------------------------
# Prompt builder
# -------------------------
def build_prompt(question, context, mode):
    if mode == "Concise":
        instruction = "Answer in 2-3 sentences only. Be brief and to the point."
    else:
        instruction = "Answer in detail with explanation, examples if needed."

    return f"""You are an AI research assistant.

{instruction}

Use the context below to answer the question.
If the answer is not in the context, say "I don't know based on available sources."

Context:
{context}

Question:
{question}

Answer:"""

# -------------------------
# Streamlit UI
# -------------------------
st.title("NeoStats AI Assistant")
st.caption("RAG-powered chatbot with web search fallback")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    response_mode = st.radio("Response Mode", ["Concise", "Detailed"], index=0)
    st.markdown("---")
    st.markdown("**Concise** — Short, summarized answers")
    st.markdown("**Detailed** — In-depth explanations")
    st.markdown("---")
    st.markdown(f"**Model:** `{GEMINI_MODEL_NAME}`")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                context, source = get_context(question)
                prompt = build_prompt(question, context, response_mode)
                answer = llm.invoke(prompt)

                st.write(answer)
                st.caption(f"📄 Source: **{source}** | 🎛️ Mode: **{response_mode}**")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Error generating answer: {e}")