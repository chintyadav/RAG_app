---
title: "RAG App"
emoji: "🏃"
colorFrom: "red"
colorTo: "yellow"
sdk: "streamlit"
sdk_version: "1.25.0"
app_file: "app.py"
pinned: false
license: "apache-2.0"
short_description: "RAG app with llm"
---
# 🧠 RAG App – Retrieval Augmented Generation LLM Demo

This project demonstrates a **Retrieval-Augmented Generation (RAG)** application built using **LangChain**, **Streamlit**, and **Hugging Face Embeddings**.  
The app allows users to upload or fetch documents from the web, store them as embeddings, and ask context-aware questions that are answered by a connected **Large Language Model (LLM)**.

---

## 🚀 Features

- 📚 **Document Ingestion:** Load data from local files or web pages  
- 🧩 **Text Splitting:** Efficient document chunking with `RecursiveCharacterTextSplitter`  
- 🔍 **Vector Store:** Uses `Chroma` for semantic retrieval  
- 💬 **Conversational Retrieval:** Maintains chat history using LangChain’s memory system  
- 🤖 **LLM Support:** Integrated with **Groq**, Hugging Face, or OpenAI compatible LLMs  
- 🌐 **Streamlit UI:** Clean and interactive web app interface

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| UI | Streamlit |
| Backend | LangChain |
| Embeddings | HuggingFace Embeddings |
| Vector Store | Chroma |
| LLM | ChatGroq / OpenAI-compatible |

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/Harsh262002/Rag_llm.git
cd Rag_llm

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # (On Windows use: .\.venv\Scripts\activate)

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
