# 🩺 Medical RAG Chatbot

> An AI-powered medical Q&A chatbot that answers questions using **verified encyclopedic knowledge** — not the open internet. Built with Retrieval-Augmented Generation (RAG) so every answer is grounded in real medical literature.

---

## What It Does

Ask any medical question in plain English and the chatbot retrieves the most relevant passages from a curated library of medical encyclopedias, then uses a large language model to synthesize a structured, detailed answer — citing only what the documents actually say.

---

## Knowledge Base

The chatbot is powered by **5 volumes of the Gale Encyclopedia of Medicine (2nd Edition)** — a comprehensive, peer-reviewed medical reference covering thousands of conditions, treatments, symptoms, and diagnostics.

| Volume | Coverage |
|--------|----------|
| Vol. 1 (A–B) | Abdominal disorders → Blood disorders |
| Vol. 2 (C–F) | Cancer → Fever & infectious disease |
| Vol. 3 (G–M) | Genetic disorders → Mental health |
| Vol. 4 (N–S) | Nervous system → Surgical procedures |
| Vol. 5 (T–Z) | Thyroid → Zoonotic diseases |

Together these volumes contain **thousands of medical entries** spanning diseases, drugs, diagnostic tests, and clinical procedures — making the chatbot capable of answering a wide range of medical questions with authoritative context.

---

## ✨ Features

-  **PDF ingestion** — Bulk-loads and parses all encyclopedia volumes using PyPDF
-  **Semantic search** — Finds the most relevant passages using vector similarity (not just keywords)
-  **Multi-turn conversation** — Remembers chat history and resolves follow-up questions correctly
-  **Grounded answers** — LLM is strictly constrained to answer only from retrieved context
-  **Clean chat UI** 

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│   History-Aware Reformulation   │  ← Rewrites follow-up questions into
│   (LLaMA 3.1 via Groq)         │    standalone queries before retrieval
└────────────────┬────────────────┘
                 │ standalone query
                 ▼
┌─────────────────────────────────┐
│        FAISS Vector Store       │  ← Finds top-5 most relevant chunks
│  (all-MiniLM-L6-v2 embeddings) │    from the encyclopedia corpus
└────────────────┬────────────────┘
                 │ retrieved context (5 chunks)
                 ▼
┌─────────────────────────────────┐
│     Answer Generation Chain     │  ← Stuffs context + question into
│   (LLaMA 3.1 8B via Groq)      │    the LLM with a strict medical prompt
└────────────────┬────────────────┘
                 │
                 ▼
           Structured Answer
```

**Two-stage retrieval** is the key design decision: a condensation LLM call first rewrites ambiguous follow-up questions ("What are its side effects?") into self-contained queries ("What are the side effects of metformin?") before hitting the vector store — making multi-turn conversations work correctly.

---

## 🧱 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | LLaMA 3.1 8B via Groq | 
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) | 
| Vector Store | FAISS (local) | 
| RAG Framework | LangChain | History-aware retrieval chain |
| PDF Parsing | PyPDF + LangChain DirectoryLoader | 
| Backend | Flask |
| Frontend | HTML / CSS |

---

## ⚙️ Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/tuhina2005/medRAG.git
cd medRAG
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -e .
```

### 4. Set up your API key
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Build the vector store (first time only)
```bash
python src/pipeline/vector_store_pipeline.py
```
This ingests all PDFs from `data/` and builds the FAISS index at `vector_store/db_faiss/`. Only needs to be run once.

### 6. Run the app
```bash
python src/app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📁 Project Structure

```
medRAG/
├── data/                        # Gale Encyclopedia PDFs (knowledge base)
├── vector_store/db_faiss/       # Pre-built FAISS index (gitignored)
├── src/
│   ├── app.py                   # Flask entry point (routes + session)
│   ├── config/
│   │   ├── config.py            # All tunable constants (chunk size, model names)
│   │   └── prompts.py           # LLM system prompt template
│   ├── components/
│   │   ├── data_loader.py       # PDF ingestion + text chunking
│   │   ├── embeddings.py        # HuggingFace embedding model loader
│   │   ├── vector_store.py      # FAISS create / load / save
│   │   ├── retriever.py         # Two-stage RAG chain assembly
│   │   └── llm_setup.py        # Groq LLM instantiation
│   ├── pipeline/
│   │   └── vector_store_pipeline.py  # One-shot offline ingestion script
│   ├── common/
│   │   ├── logger.py            # Timestamped file logger
│   │   └── custom_exception.py  # Rich exceptions with file + line info
│   └── templates/
│       └── index.html           # Chat UI
├── requirements.txt
├── setup.py
└── .env                         # API keys (gitignored)
```

---

