import streamlit as st
import os
import re
import tempfile
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from rouge_score import rouge_scorer

# ─────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────
st.set_page_config(page_title="RAG Assistant", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #e6edf3; }
[data-testid="stSidebar"] { background-color: #161b22; }
div.stTextInput > label, div.stFileUploader > label { color: #e6edf3; }
div.stButton > button {
    background-color: #21262d; color: #e6edf3;
    border: 1px solid #30363d; border-radius: 6px;
    transition: background 0.2s;
}
div.stButton > button:hover { background-color: #30363d; }
.chat-user {
    background: #1f2937; border-left: 3px solid #3b82f6;
    padding: 10px 14px; border-radius: 0 8px 8px 0; margin: 8px 0;
    color: #e2e8f0;
}
.chat-ai {
    background: #1a2332; border-left: 3px solid #10b981;
    padding: 10px 14px; border-radius: 0 8px 8px 0; margin: 8px 0;
    color: #e2e8f0;
}
.guard-pass {
    background: #0d2818; border: 1px solid #166534;
    padding: 8px 12px; border-radius: 6px; color: #86efac;
    font-size: 13px; margin: 6px 0;
}
.guard-fail {
    background: #2d1515; border: 1px solid #7f1d1d;
    padding: 8px 12px; border-radius: 6px; color: #fca5a5;
    font-size: 13px; margin: 6px 0;
}
.guard-warn {
    background: #2d2000; border: 1px solid #854d0e;
    padding: 8px 12px; border-radius: 6px; color: #fde68a;
    font-size: 13px; margin: 6px 0;
}
.source-chunk {
    background: #161b22; border: 1px solid #30363d;
    padding: 8px 12px; border-radius: 6px;
    font-size: 12px; color: #8b949e; margin: 4px 0;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Guardrails
# ─────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?(instructions?|prompts?|rules?|system)",
    r"\byou are now\b",
    r"forget (everything|all|your instructions)",
    r"(act|pretend|roleplay|behave) as (if )?(?!a helpful)",
    r"\bjailbreak\b",
    r"bypass (safety|filter|restriction|guardrail)",
    r"\bdo anything now\b",
    r"\bdan mode\b",
    r"disregard (your |the )?(previous |above )?(instructions?|rules?|guidelines?)",
]

def detect_injection(text: str):
    for p in INJECTION_PATTERNS:
        if re.search(p, text.lower()):
            return True, p
    return False, ""

def rouge_l_recall(answer: str, docs: list, threshold: float = 0.08):
    if not docs:
        return False, 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    # Score answer against each chunk; best-match prevents penalising short answers
    recalls = [
        scorer.score(doc.page_content, answer)["rougeL"].recall
        for doc in docs[:5]
    ]
    best = max(recalls)
    return best >= threshold, round(best, 4)

def full_rouge_l(reference: str, hypothesis: str):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    s = scorer.score(reference, hypothesis)["rougeL"]
    return {
        "precision": round(s.precision, 4),
        "recall": round(s.recall, 4),
        "f1": round(s.fmeasure, 4),
    }

# ─────────────────────────────────────────
#  Core RAG logic  (THE FIX IS HERE)
# ─────────────────────────────────────────

def build_standalone_chain(llm):
    """Rewrite the user question to be history-aware."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the user's latest question, "
         "rewrite it as a fully self-contained standalone question. "
         "Do NOT answer — only rewrite. If no rewrite is needed, "
         "return the question exactly as given."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


def build_qa_chain(llm):
    """Answer given retrieved context."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that answers questions strictly "
         "based on the provided document context.\n\n"
         "Rules:\n"
         "- If the answer is in the context, answer clearly and concisely.\n"
         "- If the context does not contain the answer, say: "
         "'The document does not appear to contain information about this topic.'\n"
         "- Never fabricate information not present in the context.\n"
         "- Cite specific parts of the context when relevant.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


def run_rag(user_input: str, chat_history: list, retriever, standalone_chain, qa_chain):
    """
    Full RAG pipeline:
    1. Rewrite question using history
    2. Retrieve docs using the REWRITTEN question
    3. Answer using retrieved docs
    Returns answer + source_docs
    """
    # Step 1: contextualize
    standalone_q = user_input
    if chat_history:
        standalone_q = standalone_chain.invoke({
            "input": user_input,
            "chat_history": chat_history,
        })

    # Step 2: retrieve — using the standalone question
    source_docs = retriever.invoke(standalone_q)

    # Step 3: format context and answer
    context_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(source_docs)
    )

    answer = qa_chain.invoke({
        "input": user_input,          # original question for the human turn
        "chat_history": chat_history,
        "context": context_text,
    })

    return answer, source_docs, standalone_q

# ─────────────────────────────────────────
#  Session state helpers
# ─────────────────────────────────────────

def get_history(session_id: str) -> ChatMessageHistory:
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# ─────────────────────────────────────────
#  App UI
# ─────────────────────────────────────────
load_dotenv()

st.title("🧠 Conversational RAG — PDF Q&A")
st.caption("Upload PDFs · ask questions · get grounded answers. Guardrails active.")

# Sidebar config
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password",
                             placeholder="gsk_...")
    session_id = st.text_input("Session ID", value="session_1",
                                help="Each session keeps its own chat history")
    model_name = st.selectbox("Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "gemma2-9b-it",
    ])
    top_k = st.slider("Retrieval top-k", 2, 8, 4)
    rouge_threshold = st.slider("ROUGE-L recall threshold", 0.02, 0.20, 0.08, 0.01,
                                  help="Below this = hallucination warning")
    chunk_size = st.slider("Chunk size", 256, 1024, 512, 64)
    chunk_overlap = st.slider("Chunk overlap", 32, 256, 64, 32)

    st.divider()
    if st.button("🗑️ Clear chat history"):
        if "store" in st.session_state:
            st.session_state.store = {}
        st.success("History cleared")

if not api_key:
    st.warning("⚠️ Enter your Groq API key in the sidebar to start.")
    st.stop()

# Build LLM + chains
llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=0.1)
standalone_chain = build_standalone_chain(llm)
qa_chain = build_qa_chain(llm)

# File upload
uploaded_files = st.file_uploader(
    "📄 Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more PDFs to begin.")
    st.stop()

# Index docs (cache by file names + chunk settings to avoid re-embedding)
cache_key = f"vstore_{'-'.join(f.name for f in uploaded_files)}_{chunk_size}_{chunk_overlap}"

if cache_key not in st.session_state:
    with st.spinner("🔍 Indexing documents..."):
        all_docs = []
        for uf in uploaded_files:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uf.getvalue())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            all_docs.extend(loader.load())
            os.unlink(tmp_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        splits = splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma.from_documents(splits, embedding=embeddings)
        st.session_state[cache_key] = vectorstore
        st.session_state[f"{cache_key}_splits"] = len(splits)

vectorstore = st.session_state[cache_key]
retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
n_chunks = st.session_state.get(f"{cache_key}_splits", "?")

with st.sidebar:
    st.success(f"✅ {n_chunks} chunks indexed from {len(uploaded_files)} file(s)")

# ─── Chat input ───
user_input = st.chat_input("Ask a question about your documents…")

if user_input:

    # INPUT GUARDRAIL
    injected, reason = detect_injection(user_input)
    if injected:
        st.markdown(
            f'<div class="guard-fail">🛡️ <b>Input guardrail triggered</b> — '
            f'Prompt injection detected and blocked.<br><small>Pattern: <code>{reason}</code></small></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    history = get_history(session_id)
    chat_history = history.messages

    with st.spinner("Thinking…"):
        answer, source_docs, standalone_q = run_rag(
            user_input, chat_history, retriever, standalone_chain, qa_chain
        )

    # Save to history
    history.add_user_message(user_input)
    history.add_ai_message(answer)

    # OUTPUT GUARDRAIL
    is_relevant, recall_score = rouge_l_recall(answer, source_docs, rouge_threshold)
    if not is_relevant:
        st.markdown(
            f'<div class="guard-warn">⚠️ <b>Hallucination warning</b> — '
            f'Context recall score ({recall_score}) is below threshold ({rouge_threshold}). '
            f'Answer may not be grounded in the document.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="guard-pass">✅ Output grounded — context recall: {recall_score}</div>',
            unsafe_allow_html=True,
        )

    # Show Q rewrite if it changed
    if standalone_q.strip().lower() != user_input.strip().lower():
        st.caption(f"🔄 Standalone question: *{standalone_q}*")

    # Answer
    st.markdown(f'<div class="chat-user">🧑 {user_input}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-ai">🤖 {answer}</div>', unsafe_allow_html=True)

    # ROUGE-L metrics
    with st.expander("📊 ROUGE-L Evaluation Metrics"):
        combined_ctx = " ".join(d.page_content for d in source_docs[:4])
        rouge = full_rouge_l(combined_ctx, answer)
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", rouge["precision"])
        c2.metric("Recall",    rouge["recall"])
        c3.metric("F1",        rouge["f1"])
        st.caption("Higher recall = answer is more grounded in retrieved context.")

    # Source chunks
    with st.expander(f"📄 Retrieved source chunks ({len(source_docs)})"):
        for i, doc in enumerate(source_docs):
            src = doc.metadata.get("source", "unknown")
            pg  = doc.metadata.get("page", "?")
            st.markdown(
                f'<div class="source-chunk">'
                f'<b>Chunk {i+1}</b> · {os.path.basename(src)} · page {pg}<br>'
                f'{doc.page_content[:400]}{"…" if len(doc.page_content) > 400 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

# Full chat history
with st.expander("🕑 Full Chat History"):
    for msg in get_history(session_id).messages:
        role = "🧑 You" if isinstance(msg, HumanMessage) else "🤖 Assistant"
        css = "chat-user" if isinstance(msg, HumanMessage) else "chat-ai"
        st.markdown(
            f'<div class="{css}">{role}: {msg.content}</div>',
            unsafe_allow_html=True,
        )
