import streamlit as st
import os
from dotenv import load_dotenv

# LangChain / integrations (use integration packages in requirements.txt)
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma

# Correct chain imports (fixed paths)
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Apply Modern Aesthetic Dark Theme ---
st.markdown(
    """
    <style>
        /* Root theme settings */
        :root {
            color-scheme: dark;
        }

        /* Global Background and Text */
        body, .stApp {
            background-color: #0b0f19;
            color: #e5e5e5;
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar */
        .stSidebar {
            background-color: #111522;
            color: #ffffff;
        }

        .stSidebar [data-testid="stSidebarNav"] {
            background-color: #111522;
        }

        /* Text Inputs */
        .stTextInput > div > div > input {
            background-color: #1b1f2a;
            color: #ffffff;
            border: 1px solid #343a46;
            border-radius: 8px;
            padding: 8px;
        }

        .stTextInput > div > div > input:focus {
            border-color: #4f9eed;
            box-shadow: 0 0 8px #4f9eed80;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #2563eb, #1d4ed8);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.3);
        }

        .stButton > button:hover {
            background: linear-gradient(90deg, #1e40af, #2563eb);
            transform: scale(1.03);
            box-shadow: 0px 4px 12px rgba(37, 99, 235, 0.4);
        }

        /* Titles and Headers */
        h1, h2, h3, h4, h5 {
            color: #60a5fa;
            font-weight: 600;
        }

        /* Markdown / Paragraph Text */
        p, span, label, li {
            color: #d1d5db;
        }

        /* Divider Lines */
        hr {
            border: 1px solid #2d3748;
        }

        /* Dataframe / Tables */
        .stDataFrame, .stTable {
            background-color: #111827;
            color: #f3f4f6;
            border-radius: 10px;
            border: 1px solid #374151;
        }

        /* Select boxes, sliders, etc. */
        .stSelectbox, .stSlider, .stMultiSelect {
            background-color: #1b1f2a !important;
            color: #ffffff !important;
        }

        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: #3b82f6;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #2563eb;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Streamlit Page Setup ---
st.set_page_config(page_title="Web Q&A with Groq", layout="wide")
st.title("🌐 Web Q&A using Groq LLM + LangChain")
st.write("Provide a webpage URL, load its content, and ask questions based on it.")

# --- Sidebar for API Key ---
st.sidebar.header("🔑 API Configuration")
groq_api_key = st.sidebar.text_input("Enter your GROQ_API_KEY", type="password")
if not groq_api_key:
    st.sidebar.warning("Please enter your GROQ API key to continue.")
    st.stop()

# --- Main Inputs ---
url = st.text_input("Enter a URL to load", "https://www.investopedia.com/articles/basics/06/invest1000.asp")
if st.button("🔍 Load & Process URL"):
    with st.spinner("Loading and processing webpage..."):
        try:
            # Load web page
            loader = WebBaseLoader(web_path=(url,))
            docs = loader.load()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # Store vectors
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            # Save retriever in session state
            st.session_state.retriever = retriever
            st.success("✅ URL processed successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"Error loading URL: {e}")

# --- Initialize LLM ---
if "retriever" in st.session_state:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

    # --- System prompt ---
    system_prompt = (
        "You are a helpful assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Keep answers concise and within three sentences.\n\n{context}"
    )

    # --- Prompts ---
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Given a chat history and the latest user question, "
         "which might reference context in the chat history, "
         "formulate a standalone question which can be understood without the chat history. "
         "Do not answer the question, just reformulate it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # --- Create retriever and chain ---
    history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt)
    question_ans_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_ans_chain)

    # --- Chat History ---
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

    # --- Chat Section ---
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

    # --- Display Chat History ---
    if st.session_state.chat_history:
        for sender, msg in st.session_state.chat_history:
            if sender == "You":
                st.markdown(f"🧑‍💻 **{sender}:** {msg}")
            else:
                st.markdown(f"🤖 **{sender}:** {msg}")

else:
    st.info("👆 Enter a URL and click 'Load & Process URL' to start.")




