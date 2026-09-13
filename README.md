# UniBot - College Query Assistant

A LangGraph-based AI assistant with RAG (Retrieval Augmented Generation) capabilities for answering questions about BMS Institute of Technology and Management (BMSIT). UniBot scrapes and indexes college website content and PDFs into a vector database, then uses a multi-agent RAG pipeline to generate accurate, source-cited responses.

**Live demo:** https://unibot-gkrl.onrender.com
**Backend API:** https://unibot-odr4.onrender.com

> Both are hosted on Render's free tier and spin down after inactivity - the first request after idle time can take 30-60 seconds to wake up.

## Features

### Core RAG Capabilities
- **Website & PDF Ingestion**: Scrapes college website content and PDFs (with OCR fallback for scanned documents) into a structured knowledge base
- **Vector Database**: 20,000+ chunks from 700+ sources (web pages, PDFs, newsletters) indexed in ChromaDB
- **Local Embeddings**: Uses FastEmbed (BAAI/bge-small-en-v1.5) for embeddings - runs entirely on CPU, no embedding API calls or cost
- **Hybrid Search**: Combines semantic similarity and keyword matching, with category-aware and recency-based ranking for department/faculty/leadership queries
- **Intent Detection**: Validates queries are college-domain only, with a keyword fast-path that skips the LLM call entirely for clearly in-scope questions
- **Multi-Agent Pipeline**: A Worker agent retrieves context and drafts an answer; an Evaluator agent independently reviews it before it's returned, with automatic refinement on rejection

### Deliberately NOT Included
This is a focused college-FAQ assistant, not a general-purpose agent. Generic web browsing, file read/write, code execution, and push notifications were intentionally left out of the chat agent's tool list - the only tool exposed to the Worker is knowledge-base retrieval. Live scraping exists in the codebase but is disabled from the conversational agent (it was previously causing accidental mid-conversation scrape attempts and timeouts); re-indexing is a separate, manually-run offline step.

## Architecture

```
UniBot/
|-- agents/                  # Agent modules
|   |-- unibot.py            # LangGraph Worker + Evaluator orchestration
|   |-- tools.py             # RAG tool (StructuredTool w/ Pydantic schemas)
|   `-- intent_detector.py   # College-domain validation + intent classification
|
|-- backend/
|   `-- api_server.py        # FastAPI backend server
|
|-- ingestion/                # Data ingestion (run offline, not at chat time)
|   |-- scraper.py           # Website scraper (Playwright)
|   |-- pdf_processor.py     # PDF extraction with OCR fallback + failure-reason tracking
|   |-- scrape_tracker.py    # Tracks succeeded vs. retryable-failed URLs
|   |-- scrape_checkpoint.py # Checkpoint system for resuming interrupted scrapes
|   `-- categorize.py        # Shared content-category detection
|
|-- retrieval/
|   `-- rag_system.py        # ChromaDB + FastEmbed, hybrid search
|
|-- frontend/                 # React + Vite frontend
|   |-- src/
|   |   |-- App.jsx
|   |   |-- components/
|   |   `-- utils/
|   `-- verify_setup.py
|
|-- utils/
|   |-- inspect_knowledge_base.py
|   `-- migrate_scrape_tracker.py  # One-time tracker migration helper
|
|-- index_bmsit.py            # Production ingestion entrypoint (run manually to re-index)
|-- chroma_db_fastembed/      # Vector database (tracked via Git LFS - see below)
|-- Dockerfile
`-- requirements.txt
```

## Tech Stack

- **Backend**: FastAPI, LangChain, LangGraph
- **LLM**: Groq (`openai/gpt-oss-120b` for Worker/Evaluator, `openai/gpt-oss-20b` for intent classification)
- **Vector store / embeddings**: ChromaDB + FastEmbed (local, no API calls)
- **Frontend**: React + Vite
- **Deployment**: Docker, Render (backend as a Web Service, frontend as a Static Site)
- **Large file storage**: Git LFS (the vector database exceeds GitHub's 100MB file limit)

## Quick Start (Local Development)

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install Playwright browsers (only needed if you plan to re-scrape/re-index)
playwright install chromium

# For OCR support (optional, for scanned PDFs):
# Windows: Download Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract

# Install Node.js dependencies for the frontend
cd frontend
npm install
cd ..
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# Required
GROQ_API_KEY=your-groq-api-key-here

# Optional
COLLEGE_WEBSITE_URL=https://bmsit.ac.in/
ALLOWED_ORIGINS=http://localhost:8081,http://127.0.0.1:8081
```

Get a free Groq API key at console.groq.com - no credit card required.

Create `frontend/.env`:

```env
VITE_API_URL=http://127.0.0.1:8001
```

### 3. Fetch the Vector Database (Git LFS)

The `chroma_db_fastembed/` directory is tracked with Git LFS. Make sure LFS is installed before cloning, or run:

```bash
git lfs install
git lfs pull
```

### 4. Start the Application

Run the backend and frontend in separate terminals.

**Backend:**
```bash
python -m uvicorn backend.api_server:app --host 127.0.0.1 --port 8001
```

**Frontend:**
```bash
cd frontend
npm run dev
```

- Backend: http://127.0.0.1:8001
- Frontend: http://localhost:8081

## Docker

Build and run the backend as a container:

```bash
docker build -t unibot .
docker run -p 8001:8001 --env-file .env unibot
```

The Dockerfile pre-downloads the FastEmbed model at build time and bakes in the vector database, so the container needs no network access to HuggingFace at runtime.

## Updating the Knowledge Base

The knowledge base is not updated live - it's rebuilt manually and redeployed:

```bash
python index_bmsit.py
```

This resumes from `scrape_checkpoint.json`/`scrape_tracker.json` automatically - previously succeeded pages/PDFs are skipped, and only new or previously-failed items are (re)attempted. See `utils/migrate_scrape_tracker.py` if migrating from an older tracker format.

## Current Stats

- **20,785** indexed document chunks
- **617** successfully processed PDFs (out of 868 discovered)
- **~106** HTML pages indexed

## Troubleshooting

### Backend Not Starting
- Check `GROQ_API_KEY` is set in `.env`
- Verify dependencies are installed: `pip install -r requirements.txt`
- Check port 8001 isn't already in use

### Frontend Shows "Backend Offline" or "Backend Timeout"
- Locally: ensure the backend is running on `http://127.0.0.1:8001` and `VITE_API_URL` matches
- On Render free tier: the backend spins down after inactivity - the first request wakes it up and can take 30-60+ seconds; reload after waiting

### Knowledge Base Empty or Stale
- Run `python index_bmsit.py` to (re)index
- Use `python utils/inspect_knowledge_base.py` to inspect current contents

## License

This project is for educational purposes.

## Author

Created for a 5th-semester external project presentation, BMS Institute of Technology and Management.