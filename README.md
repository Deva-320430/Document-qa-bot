# Document Q&A Bot (RAG)

A Retrieval-Augmented Generation (RAG) system that answers questions about a
collection of documents (PDF and TXT), grounded strictly in their content,
with source citations and hallucination prevention.

## Tech Stack

| Component          | Choice                                   | Why                                  |
| ------------------- | ----------------------------------------- | ------------------------------------- |
| Language            | Python 3.11+                              | Standard for ML/RAG tooling           |
| Orchestration       | LangChain                                 | Mature ecosystem, easy to explain     |
| Vector Database     | ChromaDB (persistent)                     | Free, local, simple persistence       |
| Embeddings          | `sentence-transformers/all-MiniLM-L6-v2`  | Free, fast, runs locally, no API key  |
| LLM (answer gen)    | Google Gemini 1.5 Flash                   | Free tier available                   |
| Document loading    | `PyPDFLoader` / `TextLoader`              | Built-in LangChain loaders            |
| Interface           | CLI                                       | Simple, no extra dependencies         |

## Project Structure

```text
rag-document-qa-bot/
│
├── data/                      # Source documents (PDF + TXT)
│   ├── ai_report.pdf
│   ├── climate_change.pdf
│   ├── business_strategy.pdf
│   ├── cybersecurity.txt
│   └── cloud_computing.txt
│
├── db/
│   └── chroma_db/             # Persistent vector store (created by main.py)
│
├── src/
│   ├── ingest.py              # Document loading
│   ├── utils.py                # Chunking
│   ├── rag.py                  # Embeddings + vector store
│   └── query.py                 # Retrieval + Gemini answer generation
│
├── main.py                     # Indexing entry point (run once)
├── query_bot.py                 # Interactive CLI for asking questions
├── requirements.txt
├── .env.example
└── README.md
```

## Architecture

```text
Documents (PDF/TXT)
        ↓
  Document Loader (PyPDFLoader / TextLoader)
        ↓
  Chunking (RecursiveCharacterTextSplitter, 1000 chars, 200 overlap)
        ↓
  Embeddings (all-MiniLM-L6-v2, local, free)
        ↓
  ChromaDB (persistent vector store)
        ↓
  Similarity Search (top-k retrieval)
        ↓
  Top-K Chunks  ──→  Prompt (context restricted)
        ↓
  Gemini 1.5 Flash
        ↓
  Answer + Source Citations (file, page, excerpt)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your Gemini API key

Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).

Copy the example env file and fill in your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
GOOGLE_API_KEY=your_actual_key_here
```

### 3. Add your documents (optional)

Sample documents on AI, climate change, cybersecurity, cloud computing, and
business strategy are already included in `data/`. To use your own, just
drop additional `.pdf` or `.txt` files into `data/`.

## Usage

### Step 1: Build the index (run once, or whenever documents change)

```bash
python main.py
```

This loads all documents, splits them into overlapping chunks, embeds them,
and saves the vector store to `db/chroma_db/`.

### Step 2: Ask questions

```bash
python query_bot.py
```

Example session:

```text
Ask: What are the main causes of climate change?

ANSWER:
The main causes include burning fossil fuels, deforestation, industrial
processes, agriculture, and waste management...

SOURCES:
  [1] data/climate_change.pdf | Page 1
      Excerpt: Burning Fossil Fuels: The combustion of coal, oil...
  [2] data/climate_change.pdf | Page 1
      Excerpt: Deforestation: Forests act as carbon sinks...
```

Type `exit` or `quit` to leave.

## Design Decisions

**Chunking strategy (1000 chars, 200 overlap).** `RecursiveCharacterTextSplitter`
tries to split on paragraph and sentence boundaries before falling back to
smaller separators, which preserves context and avoids cutting sentences
mid-way. 200-character overlap ensures information near chunk boundaries
isn't lost from either chunk's context. This is a widely-used industry default
for general-purpose RAG.

**Local embeddings.** `all-MiniLM-L6-v2` runs entirely on your machine, has no
API cost or rate limit, and produces good-quality 384-dimensional embeddings
for general-purpose semantic search — appropriate for a project like this one.

**Separate indexing and querying.** `main.py` (indexing) and `query_bot.py`
(querying) are deliberately separate scripts. Indexing is a slower, one-time
(or occasional) operation; querying should be fast and repeatable without
re-embedding everything each time.

**Hallucination control.** The prompt explicitly restricts Gemini to the
retrieved context and instructs it to say `"I could not find the answer in
the documents."` when the answer isn't present, rather than guessing.

**Source citations.** Every answer is returned alongside the document
filename, page number (for PDFs), and a text excerpt of each retrieved
chunk, so answers are verifiable against the original source.

## Example Questions to Try

```text
1. What are the main causes of climate change?
2. Explain the shared responsibility model in cloud computing.
3. What cybersecurity threats are discussed in the documents?
4. What is Porter's Five Forces framework?
5. Summarize the AI report's key findings.
6. Who won the FIFA World Cup 2022?   ← not in the documents
```

Question 6 should return:

```text
I could not find the answer in the documents.
```

This demonstrates that the bot is grounded in the provided documents rather
than relying on the LLM's own background knowledge.

## Requirements Checklist

| Requirement                  | Status |
| ------------------------------ | ------ |
| PDF support                    | ✅     |
| TXT support                    | ✅     |
| Chunking with overlap           | ✅     |
| Metadata (source, page)         | ✅     |
| Embeddings                      | ✅     |
| Vector database (Chroma)        | ✅     |
| Persistent storage               | ✅     |
| Separate indexing / querying     | ✅     |
| Similarity-based retrieval        | ✅     |
| Configurable top-k                | ✅     |
| LLM-based answer generation        | ✅ (Gemini) |
| Source citations                   | ✅     |
| CLI interface                       | ✅     |
| Hallucination control                | ✅     |

## Troubleshooting

**`GOOGLE_API_KEY not found`**
Make sure you've created a `.env` file (not just `.env.example`) in the
project root with your actual key.

**First run is slow / seems stuck**
The first time you run `main.py`, the embedding model (~90MB) is downloaded
and cached locally. Subsequent runs will be fast.

**`No supported documents found`**
Make sure `data/` contains at least one `.pdf` or `.txt` file.

**Want to re-index from scratch?**
Delete the `db/chroma_db/` folder and re-run `python main.py`.
