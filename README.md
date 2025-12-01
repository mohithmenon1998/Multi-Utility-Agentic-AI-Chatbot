# 🧠 Multi Utility Agentic AI Chatbot

A multi‑utility **AI assistant** built with **LangChain**, **LangGraph**, **Chroma**, and **Streamlit**.  
This project demonstrates how to combine **LLMs, RAG (Retrieval‑Augmented Generation), tool orchestration, and user management** into a production‑style chatbot.

---

## ✨ Features

- 🔐 **User Authentication**  
  Register & login with secure password hashing (bcrypt + SQLite).

- 💬 **Multi‑Threaded Conversations**  
  Each user can maintain multiple chat threads with persistent history.

- 📄 **Document Q&A (RAG)**  
  Upload PDFs → automatically chunked, embedded, and stored in **Chroma** for semantic search.  
  Ask natural questions about your documents and get context‑aware answers.

- 🌐 **Tool Integration**  
  - `rag_tool` → query uploaded PDFs  
  - `weather_tool` → fetch real‑time weather data  
  - `ddg_tool` → a duck_duck_go search tool to answer questions about current events.  
  - Extendable: add more tools easily

- 🗂️ **Persistent Storage**  
  - **SQLite** → users, threads, metadata  
  - **Chroma** → embeddings & vector search

- 🎨 **Frontend (Streamlit)**  
  - Modern chat UI with streaming responses  
  - Sidebar for threads, documents, and PDF upload  
  - Tool usage status indicators

---

## 📄 Usage

- **Login/Register** in the UI.
- Start a **new chat thread** or continue past ones.
- **Upload a PDF** in the sidebar → automatically indexed in Chroma.
- Ask questions like:
  - *“Summarize chapter 2 of my document.”*
  - *“What’s the weather in Bangalore?”*
- Watch the assistant stream responses and call tools when needed.

---

## 🛠️ Tech Stack

- **LangChain** + **LangGraph** → agent orchestration
- **Chroma** → vector database for RAG
- **SQLite** → metadata & auth
- **Streamlit** → frontend UI
- **bcrypt** → password hashing
- **Ollama** → LLM + embeddings

---

## 🤝 Contributing

Pull requests welcome! For major changes, open an issue first to discuss.

---
