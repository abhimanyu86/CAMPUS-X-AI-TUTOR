# 🎓 Campus X AI Tutor

A Retrieval-Augmented Generation (RAG) chatbot that answers AI / ML / DL questions using transcripts scraped from [Campus X](https://www.youtube.com/@campusx-official) YouTube videos.

The project has two phases:

1. **Phase 1 — Data Pipeline:** an agentic [LangGraph](https://github.com/langchain-ai/langgraph) workflow that discovers Campus X playlists, pulls video transcripts, chunks + embeds them, and stores them in a local [ChromaDB](https://www.trychroma.com/) vector store.
2. **Phase 2 — RAG Chatbot:** a [Streamlit](https://streamlit.io/) chat app that retrieves the most relevant transcript chunks for a question and answers it with an LLM (Groq Llama 3 70B), citing the source videos.

---

## ✨ Features

- **Agentic ingestion pipeline** built on LangGraph (`search → get videos → fetch transcripts → store`).
- **Resilient transcript fetching** — exponential backoff on IP blocks, circuit breaker on consecutive failures, randomized pacing, per-run caps, and resume-on-rerun (skips already-stored videos).
- **Semantic search** with `all-MiniLM-L6-v2` sentence embeddings + cosine similarity in ChromaDB.
- **Source-cited answers** — every response links back to the Campus X videos it drew from.
- **Conversational memory** — chat history is passed back into the LLM for follow-up questions.

---

## 🏗️ Architecture

```
                    Phase 1: Pipeline (LangGraph)
 ┌─────────────────┐   ┌──────────────┐   ┌────────────────────┐   ┌──────────────┐
 │ search_playlists│ → │  get_videos  │ → │ fetch_transcripts  │ → │ store_vectors│
 └─────────────────┘   └──────────────┘   └────────────────────┘   └──────┬───────┘
        YouTube Data API v3        youtube-transcript-api                  │
                                                                           ▼
                                                                  ┌─────────────────┐
                                                                  │    ChromaDB     │
                                                                  │ (embedded text) │
                                                                  └────────┬────────┘
                    Phase 2: Chatbot (Streamlit + RAG)                     │
 ┌──────────┐   ┌──────────────────┐   ┌────────────────┐   ┌──────────────▼──────┐
 │  User Q  │ → │ embed + retrieve  │ → │  Groq Llama 3  │ → │  answer + sources   │
 └──────────┘   │   top-k chunks    │   │   (RAG chain)  │   └─────────────────────┘
                └──────────────────┘   └────────────────┘
```

---

## 📂 Project Structure

```
youtube_notes_ai_agent/
├── main.py                      # Entry point — runs the Phase 1 pipeline
├── requirements.txt
├── .env                         # API keys (NOT committed)
├── data/
│   └── chroma_db/               # Local vector store (generated, NOT committed)
├── phase1_pipeline/
│   ├── agent.py                 # LangGraph pipeline definition
│   ├── youtube_tools.py         # Playlist/video discovery + transcript fetching
│   └── vector_store.py          # Chunking, embedding, ChromaDB writes
└── phase2_chatbot/
    ├── app.py                   # Streamlit chat UI
    ├── rag_chain.py             # RAG prompt + Groq LLM call
    └── retriever.py             # Query embedding + ChromaDB search
```

---

## ⚙️ Setup

### 1. Clone

```bash
git clone https://github.com/abhimanyu86/CAMPUS-X-AI-TUTOR.git
cd CAMPUS-X-AI-TUTOR
```

### 2. Install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
YOUTUBE_API_KEY=your_youtube_data_api_key

# Optional — caps transcripts fetched per pipeline run (default 30)
MAX_TRANSCRIPTS_PER_RUN=30
```

- **GROQ_API_KEY** — get one free at [console.groq.com](https://console.groq.com/keys).
- **YOUTUBE_API_KEY** — create a project + enable *YouTube Data API v3* in the [Google Cloud Console](https://console.cloud.google.com/).

---

## 🚀 Usage

### Step 1 — Build the vector store (Phase 1)

Run the ingestion pipeline. It discovers AI/ML Campus X playlists, fetches transcripts, and stores embeddings in `data/chroma_db/`.

```bash
python main.py
```

> ⚠️ Transcript fetching is intentionally slow (randomized 4–8s delays + backoff) to avoid YouTube IP bans. Reruns resume from where they left off — already-stored videos are skipped. Use `MAX_TRANSCRIPTS_PER_RUN` to process in batches across multiple runs.

### Step 2 — Launch the chatbot (Phase 2)

```bash
streamlit run phase2_chatbot/app.py
```

Open the URL Streamlit prints (default http://localhost:8501) and ask away — e.g. *"Explain how backpropagation works"* or *"What is a transformer?"* Answers cite the Campus X videos they came from.

---

## 🧰 Tech Stack

| Layer            | Tool                                            |
|------------------|-------------------------------------------------|
| Orchestration    | LangGraph                                       |
| LLM              | Groq — `llama3-70b-8192` via `langchain-groq`   |
| Embeddings       | `sentence-transformers` (`all-MiniLM-L6-v2`)    |
| Vector store     | ChromaDB (persistent, local)                    |
| Data source      | YouTube Data API v3 + `youtube-transcript-api`  |
| UI               | Streamlit                                       |

---

## 📝 Notes

- The embedding model runs **offline** after first download (`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` are set in code to avoid hangs).
- `data/chroma_db/` and `.env` are git-ignored — the vector store is generated locally and keys must stay private.
- The pipeline filters playlists by AI/ML keywords (machine learning, deep learning, NLP, transformer, LLM, etc.). Adjust the keyword list in `phase1_pipeline/youtube_tools.py` to broaden coverage.

---

## 📄 License

This project is for educational purposes. Campus X content belongs to its original creators.
