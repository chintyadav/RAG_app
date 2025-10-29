# app.py (corrected)
import logging
logging.basicConfig(level=logging.INFO)
import streamlit as st
import os
from dotenv import load_dotenv

# --- Robust compatibility + diagnostics (replace your current import block) ---
import logging
logging.basicConfig(level=logging.INFO)
import streamlit as st

def _try_import_create_retrieval_chain():
    # Try new path
    try:
        from langchain.chains.retrieval import create_retrieval_chain
        logging.info("Imported create_retrieval_chain from langchain.chains.retrieval")
        return create_retrieval_chain
    except Exception as e_new:
        logging.warning("new-path import failed: %s", e_new)
    # Try legacy path
    try:
        from langchain.chains import create_retrieval_chain
        logging.info("Imported create_retrieval_chain from langchain.chains (legacy)")
        return create_retrieval_chain
    except Exception as e_legacy:
        logging.warning("legacy-path import failed: %s", e_legacy)

    # If we reach here both attempts failed
    logging.exception("Both create_retrieval_chain import attempts failed.")
    return None

def _try_import_history_aware_retriever():
    try:
        from langchain.chains.history_aware_retriever import create_history_aware_retriever
        logging.info("Imported create_history_aware_retriever (new path)")
        return create_history_aware_retriever
    except Exception as e_new:
        logging.warning("new-path import failed: %s", e_new)
    try:
        from langchain.chains import create_history_aware_retriever
        logging.info("Imported create_history_aware_retriever (legacy)")
        return create_history_aware_retriever
    except Exception as e_legacy:
        logging.warning("legacy-path import failed: %s", e_legacy)

    logging.exception("Both create_history_aware_retriever import attempts failed.")
    return None

create_retrieval_chain = _try_import_create_retrieval_chain()
create_history_aware_retriever = _try_import_history_aware_retriever()

# Diagnostic: print versions (these will appear in Streamlit logs)
try:
    import langchain
    logging.info("langchain.__version__ = %s", getattr(langchain, "__version__", "unknown"))
except Exception:
    logging.warning("langchain import unavailable for version check.")

try:
    import langchain_core
    logging.info("langchain_core.__version__ = %s", getattr(langchain_core, "__version__", "unknown"))
except Exception:
    logging.info("langchain_core not present or import failed.")

# If imports failed, show user-friendly instructions in the app and stop
if create_retrieval_chain is None or create_history_aware_retriever is None:
    st.error(
        """LangChain imports failed in deployed environment.
        Please ensure your `requirements.txt` contains the correct packages and versions (see instructions below),
        then Clear build cache and redeploy. Check deploy logs for pip install output."""
    )
    # give a short actionable hint in the app
    st.info("Recommended requirements.txt (add to repo root) and redeploy.")
    st.code("""
streamlit>=1.25.0
langchain>=0.2.11
langchain-core>=0.2.10
langchain-community>=0.2.10
langchain-groq
langchain-huggingface
chromadb
sentence-transformers
python-dotenv
tiktoken
    """)
    st.stop()


# --- Now import the rest of your LangChain / app dependencies ---
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- (rest of your app unchanged) ---
# --- Apply Modern Aesthetic Dark Theme ---
st.markdown(
    """
    <style>
    /* ... keep your theme CSS ... */
    </style>
    """,
    unsafe_allow_html=True
)

# Page config and UI (unchanged)
st.set_page_config(page_title="Web Q&A with Groq", layout="wide")
st.title("🌐 Web Q&A using Groq LLM + LangChain")
st.write("Provide a webpage URL, load its content, and ask questions based on it.")

# Sidebar for API Key
st.sidebar.header("🔑 API Configuration")
groq_api_key = st.sidebar.text_input("Enter your GROQ_API_KEY", type="password")
if not groq_api_key:
    st.sidebar.warning("Please enter your GROQ API key to continue.")
    st.stop()

# Main Inputs
url = st.text_input("Enter a URL to load", "https://www.investopedia.com/articles/basics/06/invest1000.asp")
if st.button("🔍 Load & Process URL"):
    with st.spinner("Loading and processing webpage..."):
        try:
            loader = WebBaseLoader(web_path=(url,))
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            st.session_state.retriever = retriever
            st.success("✅ URL processed successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"Error loading URL: {e}")

# Initialize LLM and rest of your logic (unchanged)
if "retriever" in st.session_state:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

    system_prompt = (
        "You are a helpful assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Keep answers concise and within three sentences.\n\n{context}"
    )

    # Prompts
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question, "
         "reformulate a standalone question which can be understood without chat history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt)
    question_ans_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_ans_chain)

    # Chat history store & RunnableWithMessageHistory setup
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Chat UI (your existing UI code)
    st.subheader("💬 Ask Questions About This Page")
    session_id = "user_session"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Your question:")
    if st.button("Ask"):
        if user_question.strip():
            with st.spinner("Thinking..."):
                response = conversational_rag_chain.invoke(
                    {"input": user_question},
                    config={"configurable": {"session_id": session_id}}
                )
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("AI", response["answer"]))

    if st.session_state.chat_history:
        for sender, msg in st.session_state.chat_history:
            if sender == "You":
                st.markdown(f"🧑‍💻 **{sender}:** {msg}")
            else:
                st.markdown(f"🤖 **{sender}:** {msg}")

else:
    st.info("👆 Enter a URL and click 'Load & Process URL' to start.")

